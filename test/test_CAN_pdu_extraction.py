#!/usr/bin/env python
"""Tests for AUTOSAR dynamic PDU-container extraction from CAN bus logging.

Fully offline: a container is built in-memory with canmatrix and
``canmatrix.Frame.unpack`` is used as the decoding oracle.
"""

import random
import unittest

import canmatrix
import numpy as np

from asammdf import MDF, Signal
from asammdf.blocks import v4_constants as v4c
from asammdf.blocks.bus_logging_utils import extract_pdus
from asammdf.blocks.source_utils import Source

FRAME_BYTES = 32
CONTAINER_ID = 0x555


def build_container(header_big_endian: bool = True) -> tuple[canmatrix.Frame, dict[int, canmatrix.Pdu]]:
    cont = canmatrix.Frame(name="Cont", size=FRAME_BYTES)
    cont.arbitration_id = canmatrix.ArbitrationId(id=CONTAINER_ID, extended=False)
    cont.add_signal(canmatrix.Signal(name="Header_ID", start_bit=0, size=24, is_little_endian=not header_big_endian))
    cont.add_signal(canmatrix.Signal(name="Header_DLC", start_bit=24, size=8, is_little_endian=not header_big_endian))

    pA = canmatrix.Pdu(name="PduA", id=0x100, size=3)
    pA.add_signal(canmatrix.Signal(name="A_u8", start_bit=0, size=8, is_little_endian=True, is_signed=False))
    pA.add_signal(canmatrix.Signal(name="A_i16", start_bit=8, size=16, is_little_endian=True, is_signed=True))
    pB = canmatrix.Pdu(name="PduB", id=0x2AB, size=2)
    pB.add_signal(canmatrix.Signal(name="B_u16", start_bit=0, size=16, is_little_endian=False, is_signed=False))
    pC = canmatrix.Pdu(name="PduC", id=0x77, size=1)
    pC.add_signal(canmatrix.Signal(name="C_u8", start_bit=0, size=8, is_little_endian=True, is_signed=False))
    # PduD exercises bit-packed, non-byte-aligned *signed* signals (the geometry
    # that real OEM containers use heavily and that the signedness handling of
    # extract_signal must get right regardless of alignment).
    pD = canmatrix.Pdu(name="PduD", id=0x3C0, size=3)
    pD.add_signal(canmatrix.Signal(name="D_nib", start_bit=0, size=4, is_little_endian=True, is_signed=False))
    pD.add_signal(canmatrix.Signal(name="D_i8_be", start_bit=4, size=8, is_little_endian=False, is_signed=True))
    pD.add_signal(canmatrix.Signal(name="D_i8_le", start_bit=12, size=8, is_little_endian=True, is_signed=True))
    pD.add_signal(canmatrix.Signal(name="D_end", start_bit=20, size=4, is_little_endian=True, is_signed=False))
    for p in (pA, pB, pC, pD):
        cont.add_pdu(p)
    return cont, {p.id: p for p in (pA, pB, pC, pD)}


def _encode_header(pdu_id: int, dlc: int, big_endian: bool) -> bytes:
    if big_endian:
        return bytes([(pdu_id >> 16) & 0xFF, (pdu_id >> 8) & 0xFF, pdu_id & 0xFF, dlc & 0xFF])
    return bytes([pdu_id & 0xFF, (pdu_id >> 8) & 0xFF, (pdu_id >> 16) & 0xFF, dlc & 0xFF])


def build_frames(pdus: dict[int, canmatrix.Pdu], n_frames: int, big_endian: bool, pad: int, seed: int) -> list[bytes]:
    random.seed(seed)
    ids = list(pdus)
    frames = []
    for _ in range(n_frames):
        buf, used = bytearray(), 0
        for _ in range(random.randint(0, 4)):
            pid = random.choice(ids)
            need = 4 + pdus[pid].size
            if used + need > FRAME_BYTES:
                break
            buf += _encode_header(pid, pdus[pid].size, big_endian)
            buf += bytes(random.randrange(256) for _ in range(pdus[pid].size))
            used += need
        frames.append(bytes(buf).ljust(FRAME_BYTES, bytes([pad])))
    return frames


def oracle(cont: canmatrix.Frame, frames: list[bytes], t: np.ndarray):
    """Ground truth via canmatrix.Frame.unpack, accumulated in frame order."""
    values: dict[tuple[str, str], list] = {}
    times: dict[str, list] = {}
    for i, frame in enumerate(frames):
        for pdu_entry in cont.unpack(frame)["pdus"]:
            if pdu_entry is None:
                continue
            for pdu_name, sigdict in pdu_entry.items():
                times.setdefault(pdu_name, []).append(t[i])
                for sig_name, decoded in sigdict.items():
                    values.setdefault((pdu_name, sig_name), []).append(decoded.raw_value)
    return values, times


class TestPduContainerExtraction(unittest.TestCase):
    def test_extract_pdus_matches_canmatrix_unpack(self) -> None:
        for big_endian in (True, False):
            for pad in (0x00, 0xFF):
                with self.subTest(header_big_endian=big_endian, pad=pad):
                    cont, pdus = build_container(big_endian)
                    frames = build_frames(pdus, n_frames=200, big_endian=big_endian, pad=pad, seed=1)
                    payload = np.array([list(f) for f in frames], dtype="u1")
                    t = np.arange(len(frames), dtype="f8")

                    extracted = extract_pdus(payload, cont, message_id=CONTAINER_ID, bus=1, t=t, raw=True)

                    exp_values, exp_times = oracle(cont, frames, t)
                    got = {entry[4].split(":")[-1]: sigs for entry, sigs in extracted.items()}

                    self.assertEqual(set(got), set(exp_times))
                    for (pdu_name, sig_name), exp in exp_values.items():
                        samples = got[pdu_name][sig_name]["samples"]
                        self.assertTrue(
                            np.array_equal(samples, np.array(exp)),
                            f"{pdu_name}.{sig_name} mismatch",
                        )
                        self.assertTrue(
                            np.array_equal(got[pdu_name][sig_name]["t"], np.array(exp_times[pdu_name], dtype="f8"))
                        )

    def test_extract_bus_logging_container_e2e(self) -> None:
        cont, pdus = build_container(header_big_endian=True)
        db = canmatrix.CanMatrix()
        db.add_frame(cont)

        frames = build_frames(pdus, n_frames=150, big_endian=True, pad=0x00, seed=7)
        payload = np.array([list(f) for f in frames], dtype="u1")
        t = np.arange(len(frames), dtype="f8") * 0.01

        dtype = np.dtype(
            [
                ("CAN_DataFrame.BusChannel", "u1"),
                ("CAN_DataFrame.ID", "u4"),
                ("CAN_DataFrame.IDE", "u1"),
                ("CAN_DataFrame.DataBytes", "u1", (FRAME_BYTES,)),
            ]
        )
        rec = np.zeros(len(frames), dtype=dtype)
        rec["CAN_DataFrame.ID"] = CONTAINER_ID
        rec["CAN_DataFrame.DataBytes"] = payload

        acq_source = Source(name="CAN", path="CAN", comment="", source_type=v4c.SOURCE_BUS, bus_type=v4c.BUS_TYPE_CAN)
        with MDF(version="4.10") as mdf:
            cg_nr = mdf.append(Signal(samples=rec, timestamps=t, name="CAN_DataFrame"), acq_source=acq_source)
            mdf.groups[cg_nr].channel_group.flags = v4c.FLAG_CG_BUS_EVENT

            out = mdf.extract_bus_logging({"CAN": [(db, 0)]}, ignore_value2text_conversion=True)

            # one channel group per contained PDU that actually appeared
            exp_values, _exp_times = oracle(cont, frames, t)
            self.assertEqual(len(out.groups), len({p for (p, _s) in exp_values}))

            for (pdu_name, sig_name), exp in exp_values.items():
                sig = out.get(sig_name, raw=True)
                self.assertTrue(np.array_equal(sig.samples, np.array(exp)), f"{pdu_name}.{sig_name} mismatch")

    def test_static_container_without_header_is_skipped(self) -> None:
        """A container lacking Header_ID/Header_DLC (static) yields no signals."""
        cont = canmatrix.Frame(name="StaticCont", size=8)
        pdu = canmatrix.Pdu(name="Pdu", id=0x1, size=2)
        pdu.add_signal(canmatrix.Signal(name="S", start_bit=0, size=16, is_little_endian=True))
        cont.add_pdu(pdu)
        self.assertTrue(cont.is_pdu_container)

        payload = np.zeros((5, 8), dtype="u1")
        t = np.arange(5, dtype="f8")
        self.assertEqual(extract_pdus(payload, cont, message_id=1, bus=0, t=t, raw=True), {})


if __name__ == "__main__":
    unittest.main()

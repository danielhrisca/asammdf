#!/usr/bin/env python
"""Tests for AUTOSAR PDU-container extraction from CAN bus logging.

Fully offline: containers are built in-memory with canmatrix and
``canmatrix.Frame.unpack`` is used as the decoding oracle for dynamic
containers. Static (header-less) containers are decoded against a flat frame
carrying the same frame-relative signals, because ``Frame.unpack`` itself
refuses static containers.
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
FD_FRAME_BYTES = 64
CONTAINER_ID = 0x555
STATIC_CONTAINER_ID = 0x556


def build_container(
    header_big_endian: bool = True, frame_bytes: int = FRAME_BYTES
) -> tuple[canmatrix.Frame, dict[int, canmatrix.Pdu]]:
    cont = canmatrix.Frame(name="Cont", size=frame_bytes)
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


def build_static_container(frame_bytes: int = 8) -> tuple[canmatrix.Frame, canmatrix.Frame]:
    """A header-less (static) container plus the equivalent flat frame oracle.

    Contained PDUs sit at fixed byte offsets and their signal ``start_bit``
    values are frame-relative, exactly as canmatrix emits after applying each
    PDU's ``OFFSET``; there is no header id, so ``pdu.id`` is ``None``. The flat
    frame carries the same signals so ``canmatrix.Frame.unpack`` (which refuses
    static containers) can serve as the decoding oracle.
    """
    cont = canmatrix.Frame(name="StaticCont", size=frame_bytes)
    cont.arbitration_id = canmatrix.ArbitrationId(id=STATIC_CONTAINER_ID, extended=False)

    # PDU at frame byte 0 (byte-aligned little-endian, incl. signed multi-byte).
    p1 = canmatrix.Pdu(name="StatA", id=None, size=3)
    p1.add_signal(canmatrix.Signal(name="SA_u8", start_bit=0, size=8, is_little_endian=True, is_signed=False))
    p1.add_signal(canmatrix.Signal(name="SA_i16", start_bit=8, size=16, is_little_endian=True, is_signed=True))
    # PDU at frame byte 3 (byte-aligned big-endian).
    p2 = canmatrix.Pdu(name="StatB", id=None, size=2)
    p2.add_signal(canmatrix.Signal(name="SB_u16_be", start_bit=24, size=16, is_little_endian=False, is_signed=False))
    # PDU at frame byte 5 (bit-packed, non-byte-aligned signed -> the path the
    # extract_signal signedness fix must handle at a frame offset).
    p3 = canmatrix.Pdu(name="StatC", id=None, size=3)
    p3.add_signal(canmatrix.Signal(name="SC_nib", start_bit=40, size=4, is_little_endian=True, is_signed=False))
    p3.add_signal(canmatrix.Signal(name="SC_i8", start_bit=44, size=8, is_little_endian=True, is_signed=True))
    p3.add_signal(canmatrix.Signal(name="SC_end", start_bit=52, size=12, is_little_endian=True, is_signed=False))

    flat = canmatrix.Frame(name="StaticFlat", size=frame_bytes)
    for p in (p1, p2, p3):
        cont.add_pdu(p)
        for s in p.signals:
            flat.add_signal(s)
    return cont, flat


def _encode_header(pdu_id: int, dlc: int, big_endian: bool) -> bytes:
    if big_endian:
        return bytes([(pdu_id >> 16) & 0xFF, (pdu_id >> 8) & 0xFF, pdu_id & 0xFF, dlc & 0xFF])
    return bytes([pdu_id & 0xFF, (pdu_id >> 8) & 0xFF, (pdu_id >> 16) & 0xFF, dlc & 0xFF])


def build_frames(
    pdus: dict[int, canmatrix.Pdu],
    n_frames: int,
    big_endian: bool,
    pad: int,
    seed: int,
    frame_bytes: int = FRAME_BYTES,
) -> list[bytes]:
    random.seed(seed)
    ids = list(pdus)
    frames = []
    for _ in range(n_frames):
        buf, used = bytearray(), 0
        for _ in range(random.randint(0, 4)):
            pid = random.choice(ids)
            need = 4 + pdus[pid].size
            if used + need > frame_bytes:
                break
            buf += _encode_header(pid, pdus[pid].size, big_endian)
            buf += bytes(random.randrange(256) for _ in range(pdus[pid].size))
            used += need
        frames.append(bytes(buf).ljust(frame_bytes, bytes([pad])))
    return frames


def build_wide_signal_container(frame_bytes: int = FD_FRAME_BYTES) -> canmatrix.Frame:
    """Container whose contained PDU carries an opaque blob wider than 64 bits and
    declared *signed*, the shape real AUTOSAR databases use for key/ID payloads.
    """
    cont = canmatrix.Frame(name="WideCont", size=frame_bytes)
    cont.arbitration_id = canmatrix.ArbitrationId(id=CONTAINER_ID, extended=False)
    cont.add_signal(canmatrix.Signal(name="Header_ID", start_bit=0, size=24, is_little_endian=False))
    cont.add_signal(canmatrix.Signal(name="Header_DLC", start_bit=24, size=8, is_little_endian=False))

    pdu = canmatrix.Pdu(name="WidePdu", id=0x123, size=28)
    pdu.add_signal(canmatrix.Signal(name="W_blob", start_bit=0, size=216, is_little_endian=True, is_signed=True))
    pdu.add_signal(canmatrix.Signal(name="W_u8", start_bit=216, size=8, is_little_endian=True, is_signed=False))
    cont.add_pdu(pdu)
    return cont


def build_short_dlc_container(frame_bytes: int = FRAME_BYTES) -> canmatrix.Frame:
    """Container whose PDU is transmitted shorter (header DLC) than declared."""
    cont = canmatrix.Frame(name="ShortCont", size=frame_bytes)
    cont.arbitration_id = canmatrix.ArbitrationId(id=CONTAINER_ID, extended=False)
    cont.add_signal(canmatrix.Signal(name="Header_ID", start_bit=0, size=24, is_little_endian=False))
    cont.add_signal(canmatrix.Signal(name="Header_DLC", start_bit=24, size=8, is_little_endian=False))

    pdu = canmatrix.Pdu(name="ShortPdu", id=0x321, size=6)
    pdu.add_signal(canmatrix.Signal(name="S_early", start_bit=0, size=16, is_little_endian=True, is_signed=False))
    pdu.add_signal(canmatrix.Signal(name="S_mid", start_bit=16, size=16, is_little_endian=True, is_signed=False))
    # lives in bytes 4..5, i.e. past a 4-byte transmitted length
    pdu.add_signal(canmatrix.Signal(name="S_late", start_bit=32, size=16, is_little_endian=True, is_signed=False))
    cont.add_pdu(pdu)
    return cont


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

    def test_extract_pdus_wide_signed_signal(self) -> None:
        """A contained PDU signal wider than 64 bits and declared signed must not
        blow up: it is kept as raw bytes, exactly like an unsigned one would be.
        Real OEM containers carry 216/288/400-bit signed blobs, and
        two's complement cannot be applied to a byte-matrix sample."""
        cont = build_wide_signal_container()
        pdu = cont.pdus[0]

        rng = np.random.default_rng(11)
        blobs = rng.integers(0, 256, size=(50, 27), dtype="u1")
        frames = []
        for row in blobs:
            frames.append(
                (_encode_header(pdu.id, pdu.size, True) + bytes(row) + bytes([0xA5])).ljust(FD_FRAME_BYTES, b"\x00")
            )
        payload = np.array([list(f) for f in frames], dtype="u1")
        t = np.arange(len(frames), dtype="f8")

        extracted = extract_pdus(payload, cont, message_id=CONTAINER_ID, bus=1, t=t, raw=True)

        got = {entry[4].split(":")[-1]: sigs for entry, sigs in extracted.items()}
        self.assertEqual(set(got), {pdu.name})
        blob = got[pdu.name]["W_blob"]["samples"]
        self.assertEqual(len(blob), len(frames))
        # the 216-bit blob comes back as its raw bytes, low byte first
        for i, row in enumerate(blobs):
            self.assertEqual(bytes(np.asarray(blob[i]).tobytes()[:27]), bytes(row))
        self.assertTrue(np.array_equal(got[pdu.name]["W_u8"]["samples"], np.full(len(frames), 0xA5)))

    def test_extract_pdus_short_header_dlc_invalidates(self) -> None:
        """When the header DLC is shorter than the declared PDU size the trailing
        bytes belong to the next PDU / container padding, so signals reaching into
        them are flagged invalid instead of reporting padding as measured data."""
        cont = build_short_dlc_container()
        pdu = cont.pdus[0]
        sent = 4  # only 4 of the declared 6 bytes are transmitted

        frames = []
        for i in range(30):
            body = bytes([i, 0x00, i + 1, 0x00])
            frames.append((_encode_header(pdu.id, sent, True) + body).ljust(FRAME_BYTES, b"\xee"))
        payload = np.array([list(f) for f in frames], dtype="u1")
        t = np.arange(len(frames), dtype="f8")

        extracted = extract_pdus(payload, cont, message_id=CONTAINER_ID, bus=1, t=t, raw=True)
        sigs = next(iter(extracted.values()))

        # transmitted signals: real values, no invalidation
        self.assertIsNone(sigs["S_early"]["invalidation_bits"])
        self.assertIsNone(sigs["S_mid"]["invalidation_bits"])
        self.assertTrue(np.array_equal(sigs["S_early"]["samples"], np.arange(30)))
        self.assertTrue(np.array_equal(sigs["S_mid"]["samples"], np.arange(1, 31)))

        # the signal past the transmitted length is fully invalidated
        invalid = sigs["S_late"]["invalidation_bits"]
        self.assertIsNotNone(invalid)
        self.assertTrue(invalid.all())

        # a full-length transmission keeps everything valid
        full = []
        for i in range(30):
            full.append(
                (_encode_header(pdu.id, pdu.size, True) + bytes([i, 0, i + 1, 0, i + 2, 0])).ljust(FRAME_BYTES, b"\xee")
            )
        payload = np.array([list(f) for f in full], dtype="u1")
        extracted = extract_pdus(payload, cont, message_id=CONTAINER_ID, bus=1, t=t, raw=True)
        sigs = next(iter(extracted.values()))
        self.assertIsNone(sigs["S_late"]["invalidation_bits"])
        self.assertTrue(np.array_equal(sigs["S_late"]["samples"], np.arange(2, 32)))

    def test_extract_pdus_static_container(self) -> None:
        """A static (header-less) container: every contained PDU is present in
        every frame and decodes at its fixed frame offset, one channel group
        each. Oracle is the equivalent flat frame."""
        cont, flat = build_static_container(frame_bytes=8)
        rng = np.random.default_rng(3)
        payload = rng.integers(0, 256, size=(120, 8), dtype="u1")
        frames = [bytes(row) for row in payload]
        t = np.arange(len(frames), dtype="f8")

        extracted = extract_pdus(payload, cont, message_id=STATIC_CONTAINER_ID, bus=2, t=t, raw=True)

        got = {entry[4].split(":")[-1]: sigs for entry, sigs in extracted.items()}
        self.assertEqual(set(got), {p.name for p in cont.pdus})

        decoded = [flat.unpack(f) for f in frames]
        for pdu in cont.pdus:
            for sig in pdu.signals:
                exp = np.array([d[sig.name].raw_value for d in decoded])
                samples = got[pdu.name][sig.name]["samples"]
                self.assertTrue(np.array_equal(samples, exp), f"{pdu.name}.{sig.name} mismatch")
                self.assertTrue(np.array_equal(got[pdu.name][sig.name]["t"], t))

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

    def test_extract_bus_logging_canfd_container_e2e(self) -> None:
        """Full pipeline on genuine CAN-FD container frames: 64-byte payload
        with the EDL (extended data length) flag set and a DataLength member,
        the real-world transport for AUTOSAR containers."""
        cont, pdus = build_container(header_big_endian=True, frame_bytes=FD_FRAME_BYTES)
        db = canmatrix.CanMatrix()
        db.add_frame(cont)

        frames = build_frames(pdus, n_frames=200, big_endian=True, pad=0x00, seed=11, frame_bytes=FD_FRAME_BYTES)
        payload = np.array([list(f) for f in frames], dtype="u1")
        t = np.arange(len(frames), dtype="f8") * 0.01

        dtype = np.dtype(
            [
                ("CAN_DataFrame.BusChannel", "u1"),
                ("CAN_DataFrame.ID", "u4"),
                ("CAN_DataFrame.IDE", "u1"),
                ("CAN_DataFrame.EDL", "u1"),
                ("CAN_DataFrame.DataLength", "u1"),
                ("CAN_DataFrame.DataBytes", "u1", (FD_FRAME_BYTES,)),
            ]
        )
        rec = np.zeros(len(frames), dtype=dtype)
        rec["CAN_DataFrame.ID"] = CONTAINER_ID
        rec["CAN_DataFrame.EDL"] = 1  # CAN-FD frame
        rec["CAN_DataFrame.DataLength"] = FD_FRAME_BYTES
        rec["CAN_DataFrame.DataBytes"] = payload

        acq_source = Source(name="CAN", path="CAN", comment="", source_type=v4c.SOURCE_BUS, bus_type=v4c.BUS_TYPE_CAN)
        with MDF(version="4.10") as mdf:
            cg_nr = mdf.append(Signal(samples=rec, timestamps=t, name="CAN_DataFrame"), acq_source=acq_source)
            mdf.groups[cg_nr].channel_group.flags = v4c.FLAG_CG_BUS_EVENT

            out = mdf.extract_bus_logging({"CAN": [(db, 0)]}, ignore_value2text_conversion=True)

            exp_values, _exp_times = oracle(cont, frames, t)
            self.assertEqual(len(out.groups), len({p for (p, _s) in exp_values}))

            for (pdu_name, sig_name), exp in exp_values.items():
                sig = out.get(sig_name, raw=True)
                self.assertTrue(np.array_equal(sig.samples, np.array(exp)), f"{pdu_name}.{sig_name} mismatch")


if __name__ == "__main__":
    unittest.main()

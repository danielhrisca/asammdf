#!/usr/bin/env python
from hashlib import md5
import os
from pathlib import Path
import tempfile
import unittest
import urllib
import urllib.request
from zipfile import ZipFile

import numpy as np

from asammdf import MDF, Signal
from asammdf.blocks import v4_constants as v4c
from asammdf.blocks.source_utils import Source
from asammdf.blocks.utils import load_can_database


@unittest.skipIf(os.getenv("NO_NET_ACCESS"), "Test requires Internet access")
class TestCANBusLogging(unittest.TestCase):
    tempdir_obd: tempfile.TemporaryDirectory[str]
    tempdir_j1939: tempfile.TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir_obd = tempfile.TemporaryDirectory()
        cls.tempdir_j1939 = tempfile.TemporaryDirectory()

        url = "https://github.com/danielhrisca/asammdf/files/4328945/OBD2-DBC-MDF4.zip"
        urllib.request.urlretrieve(url, "test.zip")
        ZipFile(r"test.zip").extractall(cls.tempdir_obd.name)
        Path("test.zip").unlink()

        url = "https://github.com/danielhrisca/asammdf/files/4076869/J1939-DBC-MDF4.zip"
        urllib.request.urlretrieve(url, "test.zip")
        ZipFile(r"test.zip").extractall(cls.tempdir_j1939.name)
        Path("test.zip").unlink()

    def test_obd_extract(self) -> None:
        print("OBD extract")

        temp_dir = Path(TestCANBusLogging.tempdir_obd.name)

        for file in temp_dir.iterdir():
            print(file)

        path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]
        dbc = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]
        signals = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]

        with MDF(path) as mdf:
            out = mdf.extract_bus_logging({"CAN": [(dbc, 0)]}, ignore_value2text_conversion=True)

        for signal in signals:
            name = signal.stem

            target = np.load(signal)
            sig = out.get(name)
            if sig.samples.dtype.kind == "S":
                sig = out.get(name, raw=True)

            self.assertTrue(np.array_equal(sig.samples, target), f"{name} {sig} {target}")

    def test_j1939_extract(self) -> None:
        print("J1939 extract")

        temp_dir = Path(TestCANBusLogging.tempdir_j1939.name)
        dbc = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]
        signals = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]
        path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]

        with MDF(path) as mdf:
            out = mdf.extract_bus_logging({"CAN": [(dbc, 0)]}, ignore_value2text_conversion=True)

        for signal in signals:
            name = signal.stem

            target = np.load(signal)
            values = out.get(name).samples
            if values.dtype.kind == "S":
                values = out.get(name, raw=True).samples

            self.assertTrue(np.array_equal(values, target))

    def test_almost_j1939_extract(self) -> None:
        print("non-standard J1939 extract")

        temp_dir = Path(TestCANBusLogging.tempdir_j1939.name)

        # dbc = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]
        # This dbc throws exception without the suggested changes in branch "relaxed_j1939"...
        # else it is identical to the CSS Electronics demo file in test package
        d = os.path.dirname(__file__)
        dbc = os.path.join(d, "almost-J1939.dbc")  # Pls replace with file from expanded zip file
        signals = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]
        path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]

        with MDF(path) as mdf:
            out = mdf.extract_bus_logging({"CAN": [(dbc, 0)]})

        for signal in signals:
            name = signal.stem

            target = np.load(signal)
            values = out.get(name).samples

            self.assertTrue(np.array_equal(values, target))

    def test_j1939_get_can_signal(self) -> None:
        print("J1939 get CAN signal")

        temp_dir = Path(TestCANBusLogging.tempdir_j1939.name)

        path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]
        dbc = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]
        signals = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]

        with MDF(path) as mdf:
            for signal in signals:
                name = signal.stem

                target = np.load(signal)

                values = mdf.get_can_signal(name=name, database=str(dbc)).samples
                self.assertTrue(np.array_equal(values, target))

                values = mdf.get_bus_signal("CAN", name=name, database=str(dbc)).samples
                self.assertTrue(np.array_equal(values, target))


J1939_DBC = """VERSION ""

NS_ :

BS_:

BU_:

BO_ 2348810494 TSC1: 8 Vector__XXX
 SG_ EngRqedSpeed : 8|16@1+ (0.125,0) [0|8031.875] "rpm" Vector__XXX

BA_DEF_ BO_  "VFrameFormat" ENUM  "StandardCAN","ExtendedCAN","reserved","J1939PG";
BA_DEF_  "ProtocolType" STRING ;
BA_DEF_DEF_  "VFrameFormat" "J1939PG";
BA_DEF_DEF_  "ProtocolType" "";
BA_ "ProtocolType" "J1939";
BA_ "VFrameFormat" BO_ 2348810494 3;
"""


def can_logging(
    ids: list[int], ide: list[int], speed: list[float], attachment: tuple[bytes, Path, str] | None = None
) -> MDF:
    """CAN log on bus 1 with the TSC1 engine speed in bytes 1 and 2 of every frame."""
    dtype = np.dtype(
        [
            ("CAN_DataFrame.BusChannel", "u1"),
            ("CAN_DataFrame.ID", "u4"),
            ("CAN_DataFrame.IDE", "u1"),
            ("CAN_DataFrame.DataBytes", "u1", (8,)),
        ]
    )
    rec = np.zeros(len(ids), dtype=dtype)
    rec["CAN_DataFrame.BusChannel"] = 1
    rec["CAN_DataFrame.ID"] = ids
    rec["CAN_DataFrame.IDE"] = ide
    rec["CAN_DataFrame.DataBytes"][:, 1:3] = (np.array(speed) / 0.125).astype("<u2").view("u1").reshape(-1, 2)

    acq_source = Source(name="CAN", path="CAN", comment="", source_type=v4c.SOURCE_BUS, bus_type=v4c.BUS_TYPE_CAN)
    mdf = MDF(version="4.10")
    t = np.arange(len(ids), dtype="f8") * 0.01
    cg_nr = mdf.append(
        Signal(samples=rec, timestamps=t, name="CAN_DataFrame", attachment=attachment), acq_source=acq_source
    )
    mdf.groups[cg_nr].channel_group.flags = v4c.FLAG_CG_BUS_EVENT
    return mdf


class TestBusFrameSelection(unittest.TestCase):
    def test_j1939_ignores_standard_frames(self) -> None:
        dbc = load_can_database("j1939.dbc", contents=J1939_DBC)
        # TSC1 from source address 0x0B next to 11-bit frames with IDs 0x00B and 0x123
        with can_logging([0x0C00000B, 0x00B, 0x123] * 4, [1, 0, 0] * 4, [1000.0, 2000.0, 3000.0] * 4) as mdf:
            out = mdf.extract_bus_logging({"CAN": [(dbc, 0)]})

            self.assertEqual(len(out.groups), 1)
            speed = out.get("EngRqedSpeed")
            self.assertTrue(np.array_equal(speed.samples, [1000.0] * 4))
            self.assertTrue(np.array_equal(speed.timestamps, np.arange(0, 12, 3) * 0.01))

    def test_j1939_message_per_destination(self) -> None:
        dbc = load_can_database(
            "j1939.dbc",
            contents=J1939_DBC.replace(
                'BO_ 2348810494 TSC1: 8 Vector__XXX\n SG_ EngRqedSpeed : 8|16@1+ (0.125,0) [0|8031.875] "rpm" Vector__XXX',
                'BO_ 2348810251 TSC1_A: 8 Vector__XXX\n SG_ SpeedA : 8|16@1+ (0.125,0) [0|8031.875] "rpm" Vector__XXX\n'
                'BO_ 2348814347 TSC1_B: 8 Vector__XXX\n SG_ SpeedB : 8|16@1+ (0.125,0) [0|8031.875] "rpm" Vector__XXX',
            ).replace(
                'BA_ "VFrameFormat" BO_ 2348810494 3;',
                'BA_ "VFrameFormat" BO_ 2348810251 3;\nBA_ "VFrameFormat" BO_ 2348814347 3;',
            ),
        )
        # the database has one TSC1 message for each of the destinations 0x00 and 0x10
        with can_logging([0x0C00000B, 0x0C00100B] * 3, [1] * 6, [100.0, 200.0] * 3) as mdf:
            out = mdf.extract_bus_logging({"CAN": [(dbc, 0)]})

            self.assertEqual(sorted(group.channels[1].name for group in out.groups), ["SpeedA", "SpeedB"])

    def test_j1939_one_group_per_pgn_and_source_address(self) -> None:
        dbc = load_can_database("j1939.dbc", contents=J1939_DBC)
        # TSC1 from source address 0x0B to destinations 0x00 and 0x10, only the second one in the last frames
        ids = [0x0C00000B, 0x0C00100B] * 2 + [0x0C00100B] * 6
        speed = [100.0 * i for i in range(len(ids))]
        with can_logging(ids, [1] * len(ids), speed) as mdf:
            for fragment_size in (None, 64):
                with self.subTest(fragment_size=fragment_size):
                    if fragment_size:
                        mdf.configure(read_fragment_size=fragment_size)
                    out = mdf.extract_bus_logging({"CAN": [(dbc, 0)]})

                    self.assertEqual(len(out.groups), 1)
                    self.assertTrue(np.array_equal(out.get("EngRqedSpeed").samples, speed))

    def test_lin_frames_of_each_bus(self) -> None:
        dbc = load_can_database(
            "lin.dbc",
            contents='VERSION ""\n\nBO_ 32 LinFrame: 1 Vector__XXX\n SG_ LinValue : 0|8@1+ (1,0) [0|255] "" Vector__XXX\n',
        )
        # frame 0x20 with value 11 on bus 1 and 22 on bus 2
        dtype = np.dtype([("LIN_Frame.BusChannel", "u1"), ("LIN_Frame.ID", "u1"), ("LIN_Frame.DataBytes", "u1", (1,))])
        rec = np.zeros(6, dtype=dtype)
        rec["LIN_Frame.BusChannel"] = [1, 2] * 3
        rec["LIN_Frame.ID"] = 0x20
        rec["LIN_Frame.DataBytes"][:, 0] = [11, 22] * 3

        acq_source = Source(name="LIN", path="LIN", comment="", source_type=v4c.SOURCE_BUS, bus_type=v4c.BUS_TYPE_LIN)
        with MDF(version="4.10") as mdf:
            cg_nr = mdf.append(
                Signal(samples=rec, timestamps=np.arange(6, dtype="f8"), name="LIN_Frame"), acq_source=acq_source
            )
            mdf.groups[cg_nr].channel_group.flags = v4c.FLAG_CG_BUS_EVENT

            for bus, values in ((1, [11] * 3), (2, [22] * 3)):
                with self.subTest(bus=bus):
                    out = mdf.extract_bus_logging({"LIN": [(dbc, bus)]})
                    self.assertTrue(np.array_equal(out.get("LinValue").samples, values))

            out = mdf.extract_bus_logging({"LIN": [(dbc, 0)]})
            self.assertEqual([group.channel_group.cycles_nr for group in out.groups], [3, 3])

    def test_get_can_signal_j1939_ignores_standard_frames(self) -> None:
        dbc = load_can_database("j1939.dbc", contents=J1939_DBC)
        with tempfile.TemporaryDirectory() as tmp:
            with can_logging([0x0C00000B, 0x00B, 0x123] * 4, [1, 0, 0] * 4, [1000.0, 2000.0, 3000.0] * 4) as mdf:
                path = mdf.save(Path(tmp) / "j1939.mf4")

            with MDF(path) as mdf:
                speed = mdf.get_can_signal("EngRqedSpeed", database=dbc)
                self.assertTrue(np.array_equal(speed.samples, [1000.0] * 4))

    def test_embedded_j1939_database_ignores_standard_frames(self) -> None:
        dbc = J1939_DBC.encode()
        with tempfile.TemporaryDirectory() as tmp:
            with can_logging(
                [0x0C00000B, 0x00B, 0x123] * 4,
                [1, 0, 0] * 4,
                [1000.0, 2000.0, 3000.0] * 4,
                attachment=(dbc, Path("j1939.dbc"), md5(dbc).hexdigest()),
            ) as mdf:
                # with the database attached to the CAN_DataFrame channel the frames are decoded on opening
                mdf.groups[0].channel_group.flags |= v4c.FLAG_CG_PLAIN_BUS_EVENT
                path = mdf.save(Path(tmp) / "j1939.mf4")

            with MDF(path) as mdf:
                self.assertTrue(np.array_equal(mdf.get("EngRqedSpeed").samples, [1000.0] * 4))


if __name__ == "__main__":
    unittest.main()

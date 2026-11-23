#!/usr/bin/env python
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


def can_logging(ids: list[int], ide: list[int], speed: list[float]) -> MDF:
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
    cg_nr = mdf.append(Signal(samples=rec, timestamps=t, name="CAN_DataFrame"), acq_source=acq_source)
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


if __name__ == "__main__":
    unittest.main()

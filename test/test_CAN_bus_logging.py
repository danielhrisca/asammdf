#!/usr/bin/env python
import os
from pathlib import Path
import tempfile
import unittest
import urllib
import urllib.request
from zipfile import ZipFile

import numpy as np

from asammdf import MDF


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

        mdf = MDF(path)

        dbc = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]

        signals = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]

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

        path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]

        mdf = MDF(path)

        dbc = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]

        signals = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]

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

        path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]

        mdf = MDF(path)

        # dbc = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]
        # This dbc throws exception without the suggested changes in branch "relaxed_j1939"...
        # else it is identical to the CSS Electronics demo file in test package
        d = os.path.dirname(__file__)
        dbc = os.path.join(d, "almost-J1939.dbc")  # Pls replace with file from expanded zip file

        signals = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]

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

        mdf = MDF(path)

        dbc = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]

        signals = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]

        for signal in signals:
            name = signal.stem

            target = np.load(signal)

            values = mdf.get_can_signal(name=name, database=str(dbc)).samples

            self.assertTrue(np.array_equal(values, target))

            values = mdf.get_bus_signal("CAN", name=name, database=str(dbc)).samples

            self.assertTrue(np.array_equal(values, target))


if __name__ == "__main__":
    unittest.main()

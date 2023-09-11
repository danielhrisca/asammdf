#!/usr/bin/env python
from pathlib import Path
import tempfile
import unittest
import urllib
from zipfile import ZipFile

import numpy as np

from asammdf import MDF


class TestCANBusLogging(unittest.TestCase):
    tempdir_obd = None
    tempdir_j1939 = None

    @classmethod
    def setUpClass(cls):
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

    def test_obd_extract(self):
        print("OBD extract")

        temp_dir = Path(TestCANBusLogging.tempdir_obd.name)

        for file in temp_dir.iterdir():
            print(file)

        mdf = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]

        mdf = MDF(mdf)

        dbc = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]

        signals = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]

        out = mdf.extract_bus_logging({"CAN": [(dbc, 0)]})

        for signal in signals:
            name = signal.stem

            target = np.load(signal)
            sig = out.get(name)

            self.assertTrue(np.array_equal(sig.samples, target), f"{name} {sig} {target}")

    def test_j1939_extract(self):
        print("J1939 extract")

        temp_dir = Path(TestCANBusLogging.tempdir_j1939.name)

        mdf = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]

        mdf = MDF(mdf)

        dbc = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]

        signals = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]

        out = mdf.extract_bus_logging({"CAN": [(dbc, 0)]})

        for signal in signals:
            name = signal.stem

            target = np.load(signal)
            values = out.get(name).samples

            self.assertTrue(np.array_equal(values, target))

    def test_j1939_get_can_signal(self):
        print("J1939 get CAN signal")

        temp_dir = Path(TestCANBusLogging.tempdir_j1939.name)

        mdf = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]

        mdf = MDF(mdf)

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

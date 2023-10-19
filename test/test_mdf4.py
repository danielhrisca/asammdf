#!/usr/bin/env python
from pathlib import Path
import tempfile
import unittest

import numpy as np

from asammdf import MDF, Signal
from asammdf.blocks.mdf_v4 import MDF4

CHANNEL_LEN = 100000


class TestMDF4(unittest.TestCase):
    tempdir = None

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()

    def test_measurement(self):
        self.assertTrue(MDF4)

    def test_read_mdf4_00(self):
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print("Read 4.00 using seed =", seed)

        sig_int = Signal(
            np.random.randint(-(2**31), 2**31, CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name="Integer Channel",
            unit="unit1",
        )

        sig_float = Signal(
            np.random.random(CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name="Float Channel",
            unit="unit2",
        )

        with MDF(version="4.00") as mdf:
            mdf.append([sig_int, sig_float], common_timebase=True)
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "tmp", overwrite=True)

        with MDF(outfile) as mdf:
            ret_sig_int = mdf.get(sig_int.name)
            ret_sig_float = mdf.get(sig_float.name)

        self.assertTrue(np.array_equal(ret_sig_int.samples, sig_int.samples))
        self.assertTrue(np.array_equal(ret_sig_float.samples, sig_float.samples))

    def test_read_mdf4_10(self):
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print("Read 4.10 using seed =", seed)

        sig_int = Signal(
            np.random.randint(-(2**31), 2**31, CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name="Integer Channel",
            unit="unit1",
        )

        sig_float = Signal(
            np.random.random(CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name="Float Channel",
            unit="unit2",
        )

        with MDF(version="4.10") as mdf:
            mdf.append([sig_int, sig_float], common_timebase=True)
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "tmp", overwrite=True)

        with MDF(outfile) as mdf:
            ret_sig_int = mdf.get(sig_int.name)
            ret_sig_float = mdf.get(sig_float.name)

        self.assertTrue(np.array_equal(ret_sig_int.samples, sig_int.samples))
        self.assertTrue(np.array_equal(ret_sig_float.samples, sig_float.samples))

    def test_attachment_blocks_wo_filename(self):
        original_data = b"Testing attachemnt block\nTest line 1"
        mdf = MDF()
        mdf.attach(
            original_data,
            file_name=None,
            comment=None,
            compression=True,
            mime=r"text/plain",
            embedded=True,
        )
        outfile = mdf.save(Path(TestMDF4.tempdir.name) / "attachment.mf4", overwrite=True)

        with MDF(outfile) as attachment_mdf:
            data, filename, md5_sum = attachment_mdf.extract_attachment(index=0)
            self.assertEqual(data, original_data)
            self.assertEqual(filename, Path("bin.bin"))

        mdf.close()

    def test_attachment_blocks_w_filename(self):
        original_data = b"Testing attachemnt block\nTest line 1"
        original_file_name = "file.txt"

        mdf = MDF()
        mdf.attach(
            original_data,
            file_name=original_file_name,
            comment=None,
            compression=True,
            mime=r"text/plain",
            embedded=True,
        )
        outfile = mdf.save(Path(TestMDF4.tempdir.name) / "attachment.mf4", overwrite=True)

        with MDF(outfile) as attachment_mdf:
            data, filename, md5_sum = attachment_mdf.extract_attachment(index=0)
            self.assertEqual(data, original_data)
            self.assertEqual(filename, Path(original_file_name))

        mdf.close()

    def test_channel_with_boolean_array(self):
        timestamps = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        samples = [np.ones((5, 2), dtype=np.uint8)]
        types = [("boolean_array_channel", "(2, )<u1")]
        record = np.core.records.fromarrays(samples, dtype=np.dtype(types))
        boolean_array_channel = Signal(
            record,
            timestamps=timestamps,
            name="boolean_array_channel",
        )

        mdf4 = MDF(version="4.10")
        mdf4.append(signals=[boolean_array_channel])
        # set bit count to 1 to indicate that each uint8 value is a boolean flag in boolean_array_channel
        mdf4.groups[0].channels[1].bit_count = 1
        signal = mdf4.select([("boolean_array_channel", 0, 1)])[0]

        self.assertTrue((record == signal.samples).all())


if __name__ == "__main__":
    unittest.main()

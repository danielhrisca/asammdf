#!/usr/bin/env python
from pathlib import Path
import tempfile
import unittest

import numpy as np

from asammdf import MDF, Signal
from asammdf.blocks.mdf_v4 import MDF4

CHANNEL_LEN = 100000


class TestMDF4(unittest.TestCase):
    tempdir: tempfile.TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()

    def test_measurement(self) -> None:
        self.assertTrue(MDF4)

    def test_read_mdf4_00(self) -> None:
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

    def test_read_mdf4_10(self) -> None:
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

    def test_read_mdf4_20_column_storage(self) -> None:
        # regression test: 4.20 column storage wraps data in LDBLOCKs, whose
        # parsing raised AttributeError ('ListData' object has no attribute 'self')
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print("Read 4.20 using seed =", seed)

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

        with MDF(version="4.20") as mdf:
            mdf.append([sig_int], common_timebase=True)
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "tmp", overwrite=True)

        # column storage (and therefore LDBLOCK output) only engages for
        # column-oriented groups, appended when the file was opened with
        # column_storage=True
        with MDF(outfile, column_storage=True) as mdf:
            mdf.append([sig_float], common_timebase=True)
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "tmp_ld", overwrite=True)

        with open(outfile, "rb") as ld_stream:
            self.assertIn(b"##LD", ld_stream.read())

        with MDF(outfile) as mdf:
            ret_sig_int = mdf.get(sig_int.name)
            ret_sig_float = mdf.get(sig_float.name)

        self.assertTrue(np.array_equal(ret_sig_int.samples, sig_int.samples))
        self.assertTrue(np.array_equal(ret_sig_float.samples, sig_float.samples))

    def test_attachment_blocks_wo_filename(self) -> None:
        original_data = b"Testing attachemnt block\nTest line 1"
        mdf = MDF()
        mdf.attach(
            original_data,
            file_name=None,
            comment="",
            compression=True,
            mime=r"text/plain",
            embedded=True,
        )
        outfile = mdf.save(Path(TestMDF4.tempdir.name) / "attachment.mf4", overwrite=True)

        with MDF(outfile) as attachment_mdf:
            data, filename, _md5_sum = attachment_mdf.extract_attachment(index=0)
            self.assertEqual(data, original_data)
            self.assertEqual(filename, Path("bin.bin"))

        mdf.close()

    def test_attachment_blocks_w_filename(self) -> None:
        original_data = b"Testing attachemnt block\nTest line 1"
        original_file_name = "file.txt"

        mdf = MDF()
        mdf.attach(
            original_data,
            file_name=original_file_name,
            comment="",
            compression=True,
            mime=r"text/plain",
            embedded=True,
        )
        outfile = mdf.save(Path(TestMDF4.tempdir.name) / "attachment.mf4", overwrite=True)

        with MDF(outfile) as attachment_mdf:
            data, filename, _md5_sum = attachment_mdf.extract_attachment(index=0)
            self.assertEqual(data, original_data)
            self.assertEqual(filename, Path(original_file_name))

        mdf.close()

    @unittest.skip("temporary skip")
    def test_channel_with_boolean_array(self) -> None:
        timestamps = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        samples = [np.ones((5, 2), dtype=np.uint8)]
        types = [("boolean_array_channel", "(2, )<u1")]
        record = np.rec.fromarrays(samples, dtype=np.dtype(types))
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

    def test_vlsd_channel_after_structure_composition(self) -> None:
        """A VLSD channel that sits after a structure (composed) channel in the
        same group must still find its signal data.

        ``Group.signal_data`` is indexed by channel index, so an extra entry
        pushed for a composed channel shifts every following channel's VLSD
        block info and makes the payload unreadable.
        """
        count = 20
        timestamps = np.arange(count, dtype="<f8")

        record = np.rec.fromarrays(
            [np.arange(count, dtype="<u2"), np.arange(count, dtype="<u1") % 5],
            dtype=np.dtype([("a", "<u2"), ("b", "<u1")]),
        )
        structure = Signal(record, timestamps=timestamps, name="Structure")
        texts = np.array([("x" * (i % 5 + 1)).encode("latin-1") for i in range(count)], dtype="S8")
        text = Signal(texts, timestamps=timestamps, name="Text", encoding="latin-1")

        path = Path(TestMDF4.tempdir.name) / "vlsd_after_structure.mf4"
        with MDF(version="4.10") as mdf:
            mdf.append([structure, text])
            mdf.save(path, overwrite=True)

        with MDF(path) as mdf:
            group = mdf.groups[0]
            self.assertEqual(len(group.signal_data), len(group.channels))
            self.assertTrue(np.array_equal(mdf.get("Text", group=0).samples, texts))


if __name__ == "__main__":
    unittest.main()

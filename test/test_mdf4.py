#!/usr/bin/env python
from pathlib import Path
import tempfile
import unittest

import numpy as np

from asammdf import MDF, Signal
from asammdf.blocks import v4_constants as v4c
from asammdf.blocks.mdf_v4 import MDF4
from asammdf.blocks.utils import get_fmt_v4

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

    def test_row_oriented_array_components(self) -> None:
        # regression test: element offsets of non-square row oriented arrays
        # used the wrong stride and could point outside of the record
        rows, cols = 7, 6
        expected = (10 * np.arange(rows)[:, None] + np.arange(cols)).astype("<f4")
        samples = np.zeros(2, dtype=[("matrix", "<f4", (rows, cols))])
        samples["matrix"] = expected

        with MDF(version="4.10") as mdf:
            mdf.append([Signal(samples, timestamps=[0.0, 1.0], name="matrix")])
            outfile = mdf.save(Path(TestMDF4.tempdir.name) / "array.mf4", overwrite=True)

        with MDF(outfile) as mdf:
            record_size = mdf.groups[0].channel_group.samples_byte_nr
            for channel in mdf.groups[0].channels:
                self.assertLessEqual(channel.byte_offset + channel.bit_count // 8, record_size)

            names = [f"matrix[{r}][{c}]" for r in range(rows) for c in range(cols)]
            signals = mdf.select(names)

        for signal, value in zip(signals, expected.ravel(), strict=True):
            self.assertTrue(np.array_equal(signal.samples, [value, value]), signal.name)

    def test_real_channel_wider_than_128_bits(self) -> None:
        # regression test: some measurement systems store a whole array (here
        # 100 float32 values) as one REAL channel of 3200 bits without a CA
        # block. get_fmt_v4 returned "<f400", so the file could not be opened
        # at all. REAL channels wider than any numpy float type are now read as
        # raw bytes, like integer channels wider than 64 bits.
        self.assertEqual(get_fmt_v4(v4c.DATA_TYPE_REAL_INTEL, 128), "<f16")
        self.assertEqual(get_fmt_v4(v4c.DATA_TYPE_REAL_MOTOROLA, 3200), "(400,)u1")

        payload = (np.arange(3)[:, None] + np.linspace(0, 1, 100)).astype("<f4").view("u1")
        reference = np.array([1.5, 2.5, 3.5])

        for data_type in (v4c.DATA_TYPE_REAL_INTEL, v4c.DATA_TYPE_REAL_MOTOROLA):
            with self.subTest(data_type=data_type):
                with MDF(version="4.10") as mdf:
                    mdf.append(
                        [
                            Signal(payload, timestamps=[0.0, 1.0, 2.0], name="payload"),
                            Signal(reference, timestamps=[0.0, 1.0, 2.0], name="reference"),
                        ]
                    )
                    outfile = mdf.save(Path(TestMDF4.tempdir.name) / "wide_real.mf4", overwrite=True)

                with MDF(outfile) as mdf:
                    address = mdf.groups[0].channels[1].address
                    self.assertEqual(mdf.groups[0].channels[1].name, "payload")

                # change cn_data_type from byte array to REAL
                with open(outfile, "r+b") as stream:
                    stream.seek(address + 16)
                    links_nr = int.from_bytes(stream.read(8), "little")
                    stream.seek(address + 24 + 8 * links_nr + 2)
                    self.assertEqual(stream.read(1), bytes([v4c.DATA_TYPE_BYTEARRAY]))
                    stream.seek(-1, 1)
                    stream.write(bytes([data_type]))

                with MDF(outfile) as mdf:
                    self.assertEqual(mdf.groups[0].channels[1].data_type, data_type)
                    self.assertEqual(mdf.groups[0].channels[1].bit_count, 3200)
                    ret_payload = mdf.get("payload")
                    ret_reference = mdf.get("reference")

                self.assertTrue(np.array_equal(ret_reference.samples, reference))
                self.assertEqual(ret_payload.samples.dtype, np.uint8)
                self.assertTrue(np.array_equal(ret_payload.samples, payload))

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


if __name__ == "__main__":
    unittest.main()

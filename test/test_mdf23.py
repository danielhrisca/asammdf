#!/usr/bin/env python
from pathlib import Path
import tempfile
import unittest

import numpy as np

from asammdf import MDF, Signal
from asammdf.blocks.mdf_v2 import MDF2
from asammdf.blocks.mdf_v3 import MDF3

CHANNEL_LEN = 10000


class TestMDF23(unittest.TestCase):
    tempdir: tempfile.TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()

    def test_measurement(self) -> None:
        self.assertTrue(MDF2)
        self.assertTrue(MDF3)

    def test_read_mdf2_00(self) -> None:
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print("Read 2.00 using seed =", seed)

        sig_int = Signal(
            np.random.randint(-(2**15), -1, CHANNEL_LEN),
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

        with MDF(version="2.00") as mdf:
            mdf.append([sig_int, sig_float], common_timebase=True)
            outfile = mdf.save(Path(TestMDF23.tempdir.name) / "tmp", overwrite=True)

        with MDF(outfile) as mdf:
            ret_sig_int = mdf.get(sig_int.name)
            ret_sig_float = mdf.get(sig_float.name)

        self.assertTrue(np.array_equal(ret_sig_int.samples, sig_int.samples))
        self.assertTrue(np.array_equal(ret_sig_float.samples, sig_float.samples))

    def test_read_mdf2_14(self) -> None:
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print("Read 2.14 using seed =", seed)

        sig_int = Signal(
            np.random.randint(-(2**29), 2**29, CHANNEL_LEN),
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

        with MDF(version="2.14") as mdf:
            mdf.append([sig_int, sig_float], common_timebase=True)
            outfile = mdf.save(Path(TestMDF23.tempdir.name) / "tmp", overwrite=True)

        with MDF(outfile) as mdf:
            ret_sig_int = mdf.get(sig_int.name)
            ret_sig_float = mdf.get(sig_float.name)

        self.assertTrue(np.array_equal(ret_sig_int.samples, sig_int.samples))
        self.assertTrue(np.array_equal(ret_sig_float.samples, sig_float.samples))

    def test_read_mdf3_00(self) -> None:
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print("Read 3.00 using seed =", seed)

        sig_int = Signal(
            np.random.randint(-(2**16), 2**16, CHANNEL_LEN, np.int32),
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

        with MDF(version="3.00") as mdf:
            mdf.append([sig_int, sig_float], common_timebase=True)
            outfile = mdf.save(Path(TestMDF23.tempdir.name) / "tmp", overwrite=True)

        with MDF(outfile) as mdf:
            ret_sig_int = mdf.get(sig_int.name)
            ret_sig_float = mdf.get(sig_float.name)

        self.assertTrue(np.array_equal(ret_sig_int.samples, sig_int.samples))
        self.assertTrue(np.array_equal(ret_sig_float.samples, sig_float.samples))

    def test_read_mdf3_10(self) -> None:
        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print("Read 3.10 using seed =", seed)

        sig_int = Signal(
            np.random.randint(-(2**9), 2**7, CHANNEL_LEN),
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

        with MDF(version="3.10") as mdf:
            mdf.append([sig_int, sig_float], common_timebase=True)
            outfile = mdf.save(Path(TestMDF23.tempdir.name) / "tmp", overwrite=True)

        with MDF(outfile) as mdf:
            ret_sig_int = mdf.get(sig_int.name)
            ret_sig_float = mdf.get(sig_float.name)

        self.assertTrue(np.array_equal(ret_sig_int.samples, sig_int.samples))
        self.assertTrue(np.array_equal(ret_sig_float.samples, sig_float.samples))

    def test_convert_value_range_conversion(self) -> None:
        conversion: dict[str, object] = {
            "lower_0": 0.0,
            "upper_0": 10.0,
            "phys_0": 1.0,
            "lower_1": 10.0,
            "upper_1": 20.0,
            "phys_1": 2.0,
            "default": -1.0,
        }
        integer_conversion: dict[str, object] = {
            "lower_0": 0,
            "upper_0": 9,
            "phys_0": 1.0,
            "lower_1": 10,
            "upper_1": 19,
            "phys_1": 2.0,
            "lower_2": 20,
            "upper_2": 20,
            "phys_2": 3.0,
            "default": -1.0,
        }
        raw = np.array([0.0, 2.5, 7.5, 10.0, 12.5, 17.5])
        integer_raw = np.array([0, 5, 9, 10, 15, 19], dtype="<i2")
        timestamps = np.arange(len(raw), dtype="f8")
        sigs = [
            Signal(raw, timestamps, name="Range Channel", conversion=conversion),
            Signal(integer_raw, timestamps, name="Integer Range Channel", conversion=integer_conversion),
        ]

        for version in ("3.20", "3.30"):
            with self.subTest(version=version):
                with MDF(version="4.10") as mdf:
                    mdf.append(sigs)
                    with mdf.convert(version) as converted:
                        outfile = converted.save(Path(TestMDF23.tempdir.name) / "tmp", overwrite=True)

                with MDF(outfile) as mdf:
                    for sig in sigs:
                        samples = mdf.get(sig.name).samples
                        np.testing.assert_array_equal(samples, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], strict=True)

                        # MDF 3 requires strictly increasing raw values, also where two ranges share a bound
                        saved = mdf.get(sig.name, raw=True).conversion
                        assert saved is not None
                        points = [saved[f"raw_{i}"] for i in range(saved.ref_param_nr)]
                        self.assertTrue(np.all(np.diff(points) > 0))

    def test_convert_unsorted_value_to_text_conversions(self) -> None:
        value_to_text = {
            "val_0": 0,
            "text_0": b"zero",
            "val_1": 1,
            "text_1": b"one",
            "val_2": -5,
            "text_2": b"minus five",
            "default_addr": b"other",
        }
        range_to_text = {
            "lower_0": 10,
            "upper_0": 19,
            "text_0": b"ten",
            "lower_1": 0,
            "upper_1": 9,
            "text_1": b"zero",
            "default_addr": b"other",
        }
        raw = np.array([0, 1, -5, 15, 25], dtype="<i4")
        timestamps = np.arange(len(raw), dtype="f8")
        sigs = [
            Signal(raw, timestamps, name="Value Text Channel", conversion=value_to_text),
            Signal(raw, timestamps, name="Range Text Channel", conversion=range_to_text),
        ]

        for version in ("3.20", "3.30"):
            with self.subTest(version=version):
                with MDF(version="4.10") as mdf:
                    mdf.append(sigs)
                    with mdf.convert(version) as converted:
                        outfile = converted.save(Path(TestMDF23.tempdir.name) / "tmp", overwrite=True)

                with MDF(outfile) as mdf:
                    value_text = mdf.get(sigs[0].name).samples
                    range_text = mdf.get(sigs[1].name).samples

                self.assertEqual(value_text.tolist(), [b"zero", b"one", b"minus five", b"other", b"other"])
                self.assertEqual(range_text.tolist(), [b"zero", b"zero", b"other", b"ten", b"other"])

    def test_save_keeps_header_comment_text(self) -> None:
        for version in ("3.20", "3.30"):
            with self.subTest(version=version):
                with MDF(version=version) as mdf:
                    mdf.append(Signal(np.arange(5), np.arange(5, dtype="f8"), name="Channel"))
                    mdf.header.comment = "measurement comment"
                    outfile = mdf.save(Path(TestMDF23.tempdir.name) / "tmp_0", overwrite=True)

                with MDF(outfile) as mdf:
                    outfile = mdf.save(Path(TestMDF23.tempdir.name) / "tmp_1", overwrite=True)

                with MDF(outfile) as mdf:
                    lines = mdf.header.description.splitlines()

                self.assertEqual(lines[0], "measurement comment")
                self.assertEqual(len(lines), 3)
                self.assertTrue(all(": updated by asammdf " in line for line in lines[1:]))

                with MDF(version=version) as mdf:
                    mdf.append(Signal(np.arange(5), np.arange(5, dtype="f8"), name="Channel"))
                    outfile = mdf.save(Path(TestMDF23.tempdir.name) / "tmp_2", overwrite=True)

                with MDF(outfile) as mdf:
                    self.assertTrue(mdf.header.description.startswith("created by asammdf "))

    def test_array_to_mdf3(self) -> None:
        timestamps = np.arange(5) * 0.5 + 10
        samples = np.zeros(5, dtype=[("vec", "<f8", (3,))])
        samples["vec"] = np.arange(15).reshape(5, 3)
        signals = [
            Signal(samples, timestamps, name="vec"),
            Signal(np.arange(5) * 2.0, timestamps, name="scalar"),
        ]

        mdf4 = MDF(version="4.10")
        mdf4.append(signals)
        mdf3 = MDF(version="3.30")
        mdf3.append(signals)

        for mdf in (mdf4.convert("3.30"), mdf3):
            outfile = mdf.save(Path(TestMDF23.tempdir.name) / "array", overwrite=True)
            mdf.close()

            with MDF(outfile) as mdf:
                for i in range(3):
                    sig = mdf.get(f"vec[{i}]")
                    self.assertTrue(np.array_equal(sig.samples, samples["vec"][:, i]))
                    self.assertTrue(np.array_equal(sig.timestamps, timestamps))
                self.assertTrue(np.array_equal(mdf.get("vec").samples["vec"], samples["vec"]))
                self.assertTrue(np.array_equal(mdf.get("scalar").samples, signals[1].samples))

        mdf4.close()


if __name__ == "__main__":
    unittest.main()

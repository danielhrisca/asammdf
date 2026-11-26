#!/usr/bin/env python
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

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

    def test_header_start_time_is_local_standard_time(self) -> None:
        # since 3.20 the HD time stamp is local time without DST, local time = UTC + UTC time offset
        start = datetime(2020, 1, 1, 12, tzinfo=timezone(timedelta(hours=1)))
        local_ns = int(datetime(2020, 1, 1, 12, tzinfo=UTC).timestamp()) * 10**9

        with MDF(version="3.30") as mdf:
            mdf.header.abs_time = local_ns
            mdf.header.tz_offset = 1
            self.assertEqual(mdf.header.start_time, start)

            mdf.header.start_time = start
            self.assertEqual((mdf.header.abs_time, mdf.header.tz_offset), (local_ns, 1))

            # the offset excludes DST, the date and time fields give it back
            summer = datetime(2020, 7, 1, 12, tzinfo=ZoneInfo("Europe/Berlin"))
            mdf.header.start_time = summer
            self.assertEqual(mdf.header.tz_offset, 1)
            self.assertEqual(mdf.header.start_time.isoformat(), "2020-07-01T12:00:00+02:00")

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


if __name__ == "__main__":
    unittest.main()

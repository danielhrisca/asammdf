#!/usr/bin/env python
from pathlib import Path
import tempfile
import unittest

import numpy as np

from asammdf import MDF, Signal
import asammdf.blocks.v2_v3_constants as v23c
import asammdf.blocks.v4_constants as v4c


class TestEndianess(unittest.TestCase):
    tempdir = None

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()

    def test_mixed(self):
        t = np.arange(15, dtype="<f8")

        s1 = Signal(np.frombuffer(b"\x00\x00\x00\x02" * 15, dtype=">u4"), t, name="Motorola")

        s2 = Signal(np.frombuffer(b"\x04\x00\x00\x00" * 15, dtype="<u4"), t, name="Intel")

        for version in ("3.30", "4.10"):
            mdf = MDF(version=version)
            mdf.append([s1, s2], common_timebase=True)
            outfile = mdf.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)
            mdf.close()

            with MDF(outfile) as mdf:
                assert np.array_equal(mdf.get("Motorola").samples, [2] * 15)
                assert np.array_equal(mdf.get("Intel").samples, [4] * 15)

        for version in ("3.30", "4.10"):
            mdf = MDF(version=version)
            mdf.append([s2, s1], common_timebase=True)
            outfile = mdf.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)
            mdf.close()

            with MDF(outfile) as mdf:
                assert np.array_equal(mdf.get("Motorola").samples, [2] * 15)
                assert np.array_equal(mdf.get("Intel").samples, [4] * 15)

    def test_not_aligned_mdf_v3(self):
        t = np.arange(15, dtype="<f8")

        s1 = Signal(np.frombuffer(b"\x00\x00\x3F\x02" * 15, dtype=">u4"), t, name="Motorola")

        s2 = Signal(np.frombuffer(b"\x04\xF8\x00\x00" * 15, dtype="<u4"), t, name="Intel")

        s3 = Signal(
            np.frombuffer(b"\xBB\x55" * 2 * 15, dtype="<u4"),
            t,
            name="NotAlignedMotorola",
        )

        s4 = Signal(np.frombuffer(b"\xBB\x55" * 2 * 15, dtype="<u4"), t, name="NotAlignedIntel")

        with MDF(version="3.30") as mdf_source:
            mdf_source.append([s1, s2, s3, s4], common_timebase=True)

            ch3 = mdf_source.groups[0].channels[3]
            ch3.start_offset = 24 + t.itemsize * 8
            ch3.data_type = v23c.DATA_TYPE_UNSIGNED_MOTOROLA
            ch3.bit_count = 16

            ch4 = mdf_source.groups[0].channels[4]
            ch4.start_offset = 24 + t.itemsize * 8
            ch4.data_type = v23c.DATA_TYPE_UNSIGNED_INTEL
            ch4.bit_count = 16

            outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

            with MDF(outfile) as mdf:
                assert np.array_equal(mdf.get("NotAlignedMotorola").samples, [0x0204] * 15)
                assert np.array_equal(mdf.get("NotAlignedIntel").samples, [0x0402] * 15)

            ch3 = mdf_source.groups[0].channels[3]
            ch3.start_offset = 16 + t.itemsize * 8
            ch3.data_type = v23c.DATA_TYPE_UNSIGNED_MOTOROLA
            ch3.bit_count = 24

            ch4 = mdf_source.groups[0].channels[4]
            ch4.start_offset = 16 + t.itemsize * 8
            ch4.data_type = v23c.DATA_TYPE_UNSIGNED_INTEL
            ch4.bit_count = 24

            outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

            with MDF(outfile) as mdf:
                assert np.array_equal(mdf.get("NotAlignedMotorola").samples, [0x3F0204] * 15)

                assert np.array_equal(mdf.get("NotAlignedIntel").samples, [0x04023F] * 15)

            ch3 = mdf_source.groups[0].channels[3]
            ch3.start_offset = 16 + t.itemsize * 8 + 5
            ch3.data_type = v23c.DATA_TYPE_UNSIGNED_MOTOROLA
            ch3.bit_count = 21

            ch4 = mdf_source.groups[0].channels[4]
            ch4.start_offset = 16 + t.itemsize * 8 + 6
            ch4.data_type = v23c.DATA_TYPE_UNSIGNED_INTEL
            ch4.bit_count = 21

            outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

            with MDF(outfile) as mdf:
                assert np.array_equal(
                    mdf.get("NotAlignedMotorola").samples,
                    [(0x3F0204F8 >> 5) & (2**21 - 1)] * 15,
                )

                assert np.array_equal(
                    mdf.get("NotAlignedIntel").samples,
                    [(0xF8040200 >> 6) & (2**21 - 1)] * 15,
                )

    def test_not_aligned_mdf_v4(self):
        t = np.arange(15, dtype="<f8")

        s1 = Signal(np.frombuffer(b"\x00\x00\x3F\x02" * 15, dtype=">u4"), t, name="Motorola")

        s2 = Signal(np.frombuffer(b"\x04\xF8\x00\x00" * 15, dtype="<u4"), t, name="Intel")

        s3 = Signal(
            np.frombuffer(b"\xBB\x55" * 2 * 15, dtype="<u4"),
            t,
            name="NotAlignedMotorola",
        )

        s4 = Signal(np.frombuffer(b"\xBB\x55" * 2 * 15, dtype="<u4"), t, name="NotAlignedIntel")

        with MDF(version="4.11") as mdf_source:
            mdf_source.append([s1, s2, s3, s4], common_timebase=True)

            ch3 = mdf_source.groups[0].channels[3]
            ch3.byte_offset = 3 + t.itemsize
            ch3.data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
            ch3.bit_count = 16

            ch4 = mdf_source.groups[0].channels[4]
            ch4.byte_offset = 3 + t.itemsize
            ch4.data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            ch4.bit_count = 16

            outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

            with MDF(outfile) as mdf:
                assert np.array_equal(mdf.get("NotAlignedMotorola").samples, [0x0204] * 15)

                assert np.array_equal(mdf.get("NotAlignedIntel").samples, [0x0402] * 15)

            ch3 = mdf_source.groups[0].channels[3]
            ch3.byte_offset = 2 + t.itemsize
            ch3.data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
            ch3.bit_count = 24

            ch4 = mdf_source.groups[0].channels[4]
            ch4.byte_offset = 2 + t.itemsize
            ch4.data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            ch4.bit_count = 24

            outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

            with MDF(outfile) as mdf:
                assert np.array_equal(mdf.get("NotAlignedMotorola").samples, [0x3F0204] * 15)

                assert np.array_equal(mdf.get("NotAlignedIntel").samples, [0x04023F] * 15)

            ch3 = mdf_source.groups[0].channels[3]
            ch3.byte_offset = 2 + t.itemsize
            ch3.data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
            ch3.bit_count = 21
            ch3.bit_offset = 5

            ch4 = mdf_source.groups[0].channels[4]
            ch4.byte_offset = 2 + t.itemsize
            ch4.data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            ch4.bit_count = 21
            ch4.bit_offset = 6

            outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

            with MDF(outfile) as mdf:
                assert np.array_equal(
                    mdf.get("NotAlignedMotorola").samples,
                    [(0x3F0204F8 >> 5) & (2**21 - 1)] * 15,
                )

                assert np.array_equal(
                    mdf.get("NotAlignedIntel").samples,
                    [(0xF8040200 >> 6) & (2**21 - 1)] * 15,
                )

    def test_overlapping_channels_mdf_v3(self):
        t = np.arange(15, dtype="<f8")

        s1 = Signal(
            np.frombuffer(b"\x00\x00\x3F\x02\x04\xF8\x00\x00" * 15, dtype="<u8"),
            t,
            name="Parent",
        )

        s2 = Signal(
            np.frombuffer(b"\x04\xF8\x00\x00" * 15, dtype="<u4"),
            t,
            name="OverlappingMotorola",
        )

        s3 = Signal(
            np.frombuffer(b"\x04\x00\x00\x00" * 15, dtype="<u4"),
            t,
            name="OverlappingIntel",
        )

        with MDF(version="3.30") as mdf_source:
            mdf_source.append([s1, s2, s3], common_timebase=True)

            for parent_data_type in (
                v23c.DATA_TYPE_DOUBLE_MOTOROLA,
                v23c.DATA_TYPE_SIGNED_MOTOROLA,
                v23c.DATA_TYPE_UNSIGNED_MOTOROLA,
                v23c.DATA_TYPE_DOUBLE_INTEL,
                v23c.DATA_TYPE_SIGNED_INTEL,
                v23c.DATA_TYPE_UNSIGNED_INTEL,
            ):
                parent = mdf_source.groups[0].channels[1]
                parent.data_type = parent_data_type

                ch3 = mdf_source.groups[0].channels[2]
                ch3.start_offset = 24 + t.itemsize * 8
                ch3.data_type = v23c.DATA_TYPE_UNSIGNED_MOTOROLA
                ch3.bit_count = 16

                ch4 = mdf_source.groups[0].channels[3]
                ch4.start_offset = 24 + t.itemsize * 8
                ch4.data_type = v23c.DATA_TYPE_UNSIGNED_INTEL
                ch4.bit_count = 16

                outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

                with MDF(outfile) as mdf:
                    assert np.array_equal(mdf.get("OverlappingMotorola").samples, [0x0204] * 15)

                    assert np.array_equal(mdf.get("OverlappingIntel").samples, [0x0402] * 15)

                ch3 = mdf_source.groups[0].channels[2]
                ch3.start_offset = 16 + t.itemsize * 8
                ch3.data_type = v23c.DATA_TYPE_UNSIGNED_MOTOROLA
                ch3.bit_count = 24

                ch4 = mdf_source.groups[0].channels[3]
                ch4.start_offset = 16 + t.itemsize * 8
                ch4.data_type = v23c.DATA_TYPE_UNSIGNED_INTEL
                ch4.bit_count = 24

                outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

                with MDF(outfile) as mdf:
                    assert np.array_equal(mdf.get("OverlappingMotorola").samples, [0x3F0204] * 15)

                    assert np.array_equal(mdf.get("OverlappingIntel").samples, [0x04023F] * 15)

                ch3 = mdf_source.groups[0].channels[2]
                ch3.start_offset = 16 + t.itemsize * 8 + 5
                ch3.data_type = v23c.DATA_TYPE_UNSIGNED_MOTOROLA
                ch3.bit_count = 21

                ch4 = mdf_source.groups[0].channels[3]
                ch4.start_offset = 16 + t.itemsize * 8 + 6
                ch4.data_type = v23c.DATA_TYPE_UNSIGNED_INTEL
                ch4.bit_count = 21

                outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

                with MDF(outfile) as mdf:
                    assert np.array_equal(
                        mdf.get("OverlappingMotorola").samples,
                        [(0x3F0204F8 >> 5) & (2**21 - 1)] * 15,
                    )

                    assert np.array_equal(
                        mdf.get("OverlappingIntel").samples,
                        [(0xF8040200 >> 6) & (2**21 - 1)] * 15,
                    )

    def test_overlapping_channels_mdf_v4(self):
        t = np.arange(15, dtype="<f8")

        s1 = Signal(
            np.frombuffer(b"\x00\x00\x3F\x02\x04\xF8\x00\x00" * 15, dtype="<u8"),
            t,
            name="Parent",
        )

        s2 = Signal(
            np.frombuffer(b"\x04\xF8\x00\x00" * 15, dtype="<u4"),
            t,
            name="OverlappingMotorola",
        )

        s3 = Signal(
            np.frombuffer(b"\x04\x00\x00\x00" * 15, dtype="<u4"),
            t,
            name="OverlappingIntel",
        )

        with MDF(version="4.11") as mdf_source:
            mdf_source.append([s1, s2, s3], common_timebase=True)

            for parent_data_type in (
                v4c.DATA_TYPE_REAL_MOTOROLA,
                v4c.DATA_TYPE_SIGNED_MOTOROLA,
                v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                v4c.DATA_TYPE_REAL_INTEL,
                v4c.DATA_TYPE_SIGNED_INTEL,
                v4c.DATA_TYPE_UNSIGNED_INTEL,
            ):
                parent = mdf_source.groups[0].channels[1]
                parent.data_type = parent_data_type

                ch3 = mdf_source.groups[0].channels[2]
                ch3.byte_offset = 3 + t.itemsize
                ch3.data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
                ch3.bit_count = 16
                ch3.bit_offset = 0

                ch4 = mdf_source.groups[0].channels[3]
                ch4.byte_offset = 3 + t.itemsize
                ch4.data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
                ch4.bit_count = 16
                ch4.bit_offset = 0

                outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

                with MDF(outfile) as mdf:
                    assert np.array_equal(mdf.get("OverlappingMotorola").samples, [0x0204] * 15)

                    assert np.array_equal(mdf.get("OverlappingIntel").samples, [0x0402] * 15)

                ch3 = mdf_source.groups[0].channels[2]
                ch3.byte_offset = 2 + t.itemsize
                ch3.data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
                ch3.bit_count = 24
                ch3.bit_offset = 0

                ch4 = mdf_source.groups[0].channels[3]
                ch4.byte_offset = 2 + t.itemsize
                ch4.data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
                ch4.bit_count = 24
                ch4.bit_offset = 0

                outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

                with MDF(outfile) as mdf:
                    assert np.array_equal(mdf.get("OverlappingMotorola").samples, [0x3F0204] * 15)

                    assert np.array_equal(mdf.get("OverlappingIntel").samples, [0x04023F] * 15)

                ch3 = mdf_source.groups[0].channels[2]
                ch3.byte_offset = 2 + t.itemsize
                ch3.data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
                ch3.bit_count = 21
                ch3.bit_offset = 5

                ch4 = mdf_source.groups[0].channels[3]
                ch4.byte_offset = 2 + t.itemsize
                ch4.data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
                ch4.bit_count = 21
                ch4.bit_offset = 6

                outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

                with MDF(outfile) as mdf:
                    assert np.array_equal(
                        mdf.get("OverlappingMotorola").samples,
                        [(0x3F0204F8 >> 5) & (2**21 - 1)] * 15,
                    )

                    assert np.array_equal(
                        mdf.get("OverlappingIntel").samples,
                        [(0xF8040200 >> 6) & (2**21 - 1)] * 15,
                    )

                ch3 = mdf_source.groups[0].channels[2]
                ch3.byte_offset = 3 + t.itemsize
                ch3.data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
                ch3.bit_count = 2
                ch3.bit_offset = 1

                ch4 = mdf_source.groups[0].channels[3]
                ch4.byte_offset = 3 + t.itemsize
                ch4.data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
                ch4.bit_count = 2
                ch4.bit_offset = 1

                outfile = mdf_source.save(Path(TestEndianess.tempdir.name) / "out", overwrite=True)

                with MDF(outfile) as mdf:
                    assert np.array_equal(mdf.get("OverlappingMotorola").samples, [0x1] * 15)

                    assert np.array_equal(mdf.get("OverlappingIntel").samples, [0x1] * 15)


if __name__ == "__main__":
    unittest.main()

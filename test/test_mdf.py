#!/usr/bin/env python
import random
import unittest
import urllib
import numexpr
from zipfile import ZipFile
import tempfile
from pathlib import Path
from io import BytesIO

import numpy as np

from utils import (
    CHANNELS_DEMO,
    CHANNELS_ARRAY,
    cycles,
    channels_count,
    generate_test_file,
    generate_arrays_test_file,
)
from asammdf import MDF, Signal, SUPPORTED_VERSIONS
from asammdf.blocks.utils import MdfException
from pandas import DataFrame

SUPPORTED_VERSIONS = [version for version in SUPPORTED_VERSIONS if version >= "4.00"]

CHANNEL_LEN = 100000


class TestMDF(unittest.TestCase):

    tempdir_demo = None
    tempdir_array = None
    tempdir_general = None
    tempdir = None

    def etest_measurement(self):
        self.assertTrue(MDF)

    @classmethod
    def setUpClass(cls):

        url = "https://github.com/danielhrisca/asammdf/files/1834464/test.demo.zip"
        urllib.request.urlretrieve(url, "test.zip")

        cls.tempdir_demo = tempfile.TemporaryDirectory()
        cls.tempdir_general = tempfile.TemporaryDirectory()
        cls.tempdir= tempfile.TemporaryDirectory()
        cls.tempdir_array = tempfile.TemporaryDirectory()

        ZipFile(r"test.zip").extractall(cls.tempdir_demo.name)
        Path("test.zip").unlink()
        for version in ("3.30", "4.10"):
            generate_test_file(cls.tempdir_general.name, version)

        generate_arrays_test_file(cls.tempdir_array.name)

    def etest_mdf_header(self):
        mdf = BytesIO(b'M' * 100)
        with self.assertRaises(MdfException):
            MDF(mdf)

    def etest_wrong_header_version(self):
        mdf = BytesIO(b'MDF     AAAA    amdf500d\x00\x00\x00\x00\x9f\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
        with self.assertRaises(MdfException):
            MDF(mdf)

    def etest_read(self):
        print("MDF read big files")
        for input_file in Path(TestMDF.tempdir_general.name).iterdir():
            print(input_file)

            equal = True

            for inp in (input_file, BytesIO(input_file.read_bytes())):

                with MDF(inp) as mdf:

                    for i, group in enumerate(mdf.groups):
                        if i == 0:
                            v = np.ones(cycles, dtype=np.uint64)
                            for j in range(1, 20):
                                vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                if not np.array_equal(vals, v * (j - 1)):
                                    equal = False
                        elif i == 1:
                            v = np.ones(cycles, dtype=np.int64)
                            for j in range(1, 20):
                                vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                if not np.array_equal(vals, v * (j - 1) - 0.5):
                                    equal = False
                        elif i == 2:
                            v = np.arange(cycles, dtype=np.int64) / 100.0
                            form = "{} * sin(v)"
                            for j in range(1, 20):
                                vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                f = form.format(j - 1)
                                if not np.array_equal(vals, numexpr.evaluate(f)):
                                    equal = False
                        elif i == 3:
                            v = np.ones(cycles, dtype=np.int64)
                            form = "({} * v -0.5) / 1"
                            for j in range(1, 20):
                                f = form.format(j - 1)
                                vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                if not np.array_equal(vals, numexpr.evaluate(f)):
                                    equal = False
                        elif i == 4:

                            for j in range(1, 20):
                                target = np.array(
                                    [
                                        "Channel {} sample {}".format(j, k).encode(
                                            "ascii"
                                        )
                                        for k in range(cycles)
                                    ]
                                )
                                vals = mdf.get(group=i, index=j + 1, samples_only=True)[
                                    0
                                ]
                                cond = np.array_equal(vals, target)
                                if not cond:
                                    print(i, j, vals, target, len(vals), len(target))
                                self.assertTrue(cond)

                        elif i == 5:
                            v = np.ones(cycles, dtype=np.dtype("(8,)u1"))
                            for j in range(1, 20):
                                target = v * j
                                vals = mdf.get(group=i, index=j + 1, samples_only=True)[
                                    0
                                ]
                                cond = np.array_equal(vals, target)
                                if not cond:
                                    print(i, j, vals, target, len(vals), len(target))
                                self.assertTrue(cond)

                        elif i == 6:
                            for j in range(1, 20):
                                target = np.array([b'Value %d' % j for _ in range(cycles)])
                                vals = mdf.get(group=i, index=j + 1, samples_only=True)[
                                    0
                                ]
                                cond = np.array_equal(vals, target)
                                if not cond:
                                    print(i, j, vals, target, len(vals), len(target))
                                self.assertTrue(cond)
            self.assertTrue(equal)


    def etest_read_arrays(self):
        print("MDF read big array files")
        for input_file in Path(TestMDF.tempdir_array.name).iterdir():
            print(input_file)

            equal = True

            for inp in (input_file, BytesIO(input_file.read_bytes())):

                with MDF(inp) as mdf:
                    mdf.configure(read_fragment_size=8000)

                    for i, group in enumerate(mdf.groups):

                        if i == 0:

                            samples = [
                                np.ones((cycles, 2, 3), dtype=np.uint64),
                                np.ones((cycles, 2), dtype=np.uint64),
                                np.ones((cycles, 3), dtype=np.uint64),
                            ]

                            for j in range(1, 20):

                                types = [
                                    ("Channel_{}".format(j), "(2, 3)<u8"),
                                    ("channel_{}_axis_1".format(j), "(2, )<u8"),
                                    ("channel_{}_axis_2".format(j), "(3, )<u8"),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get(
                                    "Channel_{}".format(j), group=i, samples_only=True
                                )[0]
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(target, dtype=types)
                                if not np.array_equal(vals, target):
                                    equal = False
                                    1 / 0

                        elif i == 1:

                            samples = np.ones((cycles, 2, 3), dtype=np.uint64)
                            axis_0 = np.ones((cycles, 2), dtype=np.uint64)
                            axis_1 = np.ones((cycles, 3), dtype=np.uint64)

                            for j in range(1, 20):

                                types = [("Channel_{}".format(j), "(2, 3)<u8")]
                                types = np.dtype(types)

                                vals = mdf.get(
                                    "Channel_{}".format(j), group=i, samples_only=True
                                )[0]
                                target = [samples * j]
                                target = np.core.records.fromarrays(target, dtype=types)
                                if not np.array_equal(vals, target):
                                    equal = False
                                    1 / 0

                        elif i == 2:

                            samples = [
                                np.ones(cycles, dtype=np.uint8),
                                np.ones(cycles, dtype=np.uint16),
                                np.ones(cycles, dtype=np.uint32),
                                np.ones(cycles, dtype=np.uint64),
                                np.ones(cycles, dtype=np.int8),
                                np.ones(cycles, dtype=np.int16),
                                np.ones(cycles, dtype=np.int32),
                                np.ones(cycles, dtype=np.int64),
                            ]

                            for j in range(1, 20):

                                types = [
                                    ("struct_{}_channel_0".format(j), np.uint8),
                                    ("struct_{}_channel_1".format(j), np.uint16),
                                    ("struct_{}_channel_2".format(j), np.uint32),
                                    ("struct_{}_channel_3".format(j), np.uint64),
                                    ("struct_{}_channel_4".format(j), np.int8),
                                    ("struct_{}_channel_5".format(j), np.int16),
                                    ("struct_{}_channel_6".format(j), np.int32),
                                    ("struct_{}_channel_7".format(j), np.int64),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get(
                                    "Channel_{}".format(j), group=i, samples_only=True
                                )[0]
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(target, dtype=types)
                                if not np.array_equal(vals, target):
                                    equal = False
                                    print(target)
                                    print(vals)
                                    1 / 0

            self.assertTrue(equal)

    def test_read_demo(self):

        print("MDF read tests")

        ret = True

        for enable in (True, False):
            for mdf in Path(TestMDF.tempdir_demo.name).iterdir():

                for inp in (mdf, BytesIO(mdf.read_bytes())):

                    with MDF(inp) as input_file:
                        if input_file.version == "2.00":
                            continue
                        print(mdf)
                        for name in set(input_file.channels_db) - {"time", "t"}:

                            if (
                                name.endswith("[0]")
                                or name.startswith("DI")
                                or "\\" in name
                            ):
                                continue
                            signal = input_file.get(name)
                            if name == 'ASAM.M.SCALAR.UBYTE.VTAB_RANGE_DEFAULT_VALUE':
                                print(repr(signal.conversion.convert(signal.samples)))
                                1/0
                            try:
                                original_samples = CHANNELS_DEMO[name.split("\\")[0]]
                            except:
                                print(name)
                                raise
                            if signal.samples.dtype.kind == "f":
                                signal = signal.astype(np.float32)
                            res = np.array_equal(signal.samples, original_samples)
                            if not res:
                                ret = False
                                print(name, repr(signal.samples), original_samples)
                                1/0

        self.assertTrue(ret)

    def etest_convert(self):
        print("MDF convert big files tests")

        t = np.arange(cycles, dtype=np.float64)

        for out in SUPPORTED_VERSIONS:
            for input_file in Path(TestMDF.tempdir_general.name).iterdir():
                for compression in range(3):
                    print(input_file, out)

                    with MDF(input_file) as mdf:
                        mdf.configure(read_fragment_size=8000)
                        outfile = mdf.convert(out).save(
                            Path(TestMDF.tempdir.name) / f"tmp_convert_{out}",
                            overwrite=True,
                            compression=compression,
                        )

                    equal = True

                    with MDF(outfile) as mdf:
                        mdf.configure(read_fragment_size=8000)

                        for i, group in enumerate(mdf.groups):
                            if i == 0:
                                v = np.ones(cycles, dtype=np.uint64)
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[
                                        0
                                    ]
                                    if not np.array_equal(vals, v * (j - 1)):
                                        equal = False
                                        print(vals, len(vals))
                                        print(v * (j - 1), len(v))

                                        input(outfile)
                                        1 / 0
                            elif i == 1:
                                v = np.ones(cycles, dtype=np.int64)
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[
                                        0
                                    ]
                                    if not np.array_equal(vals, v * (j - 1) - 0.5):
                                        equal = False
                                        1 / 0
                            elif i == 2:
                                v = np.arange(cycles, dtype=np.int64) / 100.0
                                form = "{} * sin(v)"
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[
                                        0
                                    ]
                                    f = form.format(j - 1)
                                    if not np.array_equal(vals, numexpr.evaluate(f)):
                                        equal = False
                                        1 / 0
                            elif i == 3:
                                v = np.ones(cycles, dtype=np.int64)
                                form = "({} * v -0.5) / 1"
                                for j in range(1, 20):
                                    f = form.format(j - 1)
                                    vals = mdf.get(group=i, index=j, samples_only=True)[
                                        0
                                    ]
                                    if not np.array_equal(vals, numexpr.evaluate(f)):
                                        equal = False
                                        target = numexpr.evaluate(f)
                                        print(
                                            i, j, vals, target, len(vals), len(target)
                                        )
                                        1 / 0
                            elif i == 4:

                                for j in range(1, 20):
                                    target = np.array(
                                        [
                                            "Channel {} sample {}".format(j, k).encode(
                                                "ascii"
                                            )
                                            for k in range(cycles)
                                        ]
                                    )
                                    vals = mdf.get(
                                        group=i, index=j + 1, samples_only=True
                                    )[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(
                                            i, j, vals, target, len(vals), len(target)
                                        )
                                        1 / 0
                                    self.assertTrue(cond)

                            elif i == 5:
                                v = np.ones(cycles, dtype=np.dtype("(8,)u1"))
                                for j in range(1, 20):
                                    target = v * j
                                    vals = mdf.get(
                                        group=i, index=j + 1, samples_only=True
                                    )[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(
                                            i, j, vals, target, len(vals), len(target)
                                        )
                                        1 / 0
                                    self.assertTrue(cond)

                            elif i == 6:
                                for j in range(1, 20):
                                    target = np.array([b'Value %d' % j for _ in range(cycles)])
                                    vals = mdf.get(
                                        group=i, index=j + 1, samples_only=True
                                    )[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(
                                            i, j, vals, target, len(vals), len(target)
                                        )
                                        1 / 0
                                    self.assertTrue(cond)

                    self.assertTrue(equal)

    def etest_convert_demo(self):
        print("MDF convert tests")

        for out in SUPPORTED_VERSIONS:
            for input_file in Path(TestMDF.tempdir_demo.name).iterdir():
                if MDF(input_file).version == "2.00":
                    continue
                print(input_file, out)
                with MDF(input_file) as mdf:
                    outfile = mdf.convert(out).save(
                        Path(TestMDF.tempdir_demo.name) / "tmp",
                        overwrite=True,
                    )

                equal = True

                with MDF(input_file) as mdf, MDF(outfile) as mdf2:

                    for name in set(mdf2.channels_db) - {"t", "time"}:
                        original = mdf.get(name)
                        converted = mdf2.get(name)
                        if not np.array_equal(original.samples, converted.samples):
                            equal = False
                            print(original, converted, outfile)
                            1 / 0
                        if not np.array_equal(
                            original.timestamps, converted.timestamps
                        ):
                            equal = False
                            1 / 0

                self.assertTrue(equal)

    def etest_cut(self):
        print("MDF cut big files tests")

        t = np.arange(cycles, dtype=np.float64)

        for input_file in Path(TestMDF.tempdir_general.name).iterdir():
            for whence in (0, 1):
                for compression in range(3):
                    print(input_file)

                    outfile0 = MDF(input_file)
                    outfile0.configure(read_fragment_size=8000)
                    outfile0 = outfile0.cut(stop=-1, whence=whence, include_ends=False).save(
                        Path(TestMDF.tempdir.name) / "tmp0", overwrite=True,
                        compression=compression,
                    )

                    outfile1 = MDF(input_file)
                    outfile1.configure(read_fragment_size=8000)
                    outfile1 = outfile1.cut(stop=105, whence=whence, include_ends=False).save(
                        Path(TestMDF.tempdir.name) / "tmp1", overwrite=True,
                        compression=compression,
                    )

                    outfile2 = MDF(input_file)
                    outfile2.configure(read_fragment_size=8000)
                    outfile2 = outfile2.cut(start=105.1, stop=201, whence=whence, include_ends=False).save(
                        Path(TestMDF.tempdir.name) / "tmp2", overwrite=True,
                        compression=compression,
                    )

                    outfile3 = MDF(input_file)
                    outfile3.configure(read_fragment_size=8000)
                    outfile3 = outfile3.cut(start=201.1, whence=whence, include_ends=False).save(
                        Path(TestMDF.tempdir.name) / "tmp3", overwrite=True
                    )

                    outfile4 = MDF(input_file)
                    outfile4.configure(read_fragment_size=8000)
                    outfile4 = outfile4.cut(start=7000, whence=whence, include_ends=False).save(
                        Path(TestMDF.tempdir.name) / "tmp4", overwrite=True,
                        compression=compression,
                    )

                    outfile = MDF.concatenate(
                        [outfile0, outfile1, outfile2, outfile3, outfile4],
                        version=MDF(input_file).version,
                        sync=whence,
                    ).save(Path(TestMDF.tempdir.name) / "tmp_cut", overwrite=True,
                        compression=compression,)

                    with MDF(outfile) as mdf:

                        for i, group in enumerate(mdf.groups):
                            if i == 0:
                                v = np.ones(cycles, dtype=np.uint64)
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[
                                        0
                                    ]
                                    cond = np.array_equal(vals, v * (j - 1))
                                    if not cond:
                                        print(
                                            i, j, vals, v * (j - 1), len(vals), len(v)
                                        )
                                    self.assertTrue(cond)
                            elif i == 1:
                                v = np.ones(cycles, dtype=np.int64)
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[
                                        0
                                    ]
                                    cond = np.array_equal(vals, v * (j - 1) - 0.5)
                                    if not cond:
                                        print(
                                            vals, v * (j - 1) - 0.5, len(vals), len(v)
                                        )
                                    self.assertTrue(cond)
                            elif i == 2:
                                v = np.arange(cycles, dtype=np.int64) / 100.0
                                form = "{} * sin(v)"
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[
                                        0
                                    ]
                                    f = form.format(j - 1)
                                    cond = np.array_equal(vals, numexpr.evaluate(f))
                                    if not cond:
                                        print(
                                            i, j, vals, target, len(vals), len(target)
                                        )
                                    self.assertTrue(cond)
                            elif i == 3:
                                v = np.ones(cycles, dtype=np.int64)
                                form = "({} * v -0.5) / 1"
                                for j in range(1, 20):
                                    f = form.format(j - 1)
                                    vals = mdf.get(group=i, index=j, samples_only=True)[
                                        0
                                    ]
                                    cond = np.array_equal(vals, numexpr.evaluate(f))
                                    target = numexpr.evaluate(f)
                                    if not cond:
                                        print(
                                            i, j, vals, target, len(vals), len(target)
                                        )
                                    self.assertTrue(cond)
                            elif i == 4:

                                for j in range(1, 20):
                                    target = np.array(
                                        [
                                            "Channel {} sample {}".format(j, k).encode(
                                                "ascii"
                                            )
                                            for k in range(cycles)
                                        ]
                                    )
                                    vals = mdf.get(
                                        group=i, index=j + 1, samples_only=True
                                    )[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(
                                            i, j, vals, target, len(vals), len(target)
                                        )
                                    self.assertTrue(cond)

                            elif i == 5:
                                v = np.ones(cycles, dtype=np.dtype("(8,)u1"))
                                for j in range(1, 20):
                                    target = v * j
                                    vals = mdf.get(
                                        group=i, index=j + 1, samples_only=True
                                    )[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(
                                            i, j, vals, target, len(vals), len(target)
                                        )
                                    self.assertTrue(cond)

                            elif i == 6:
                                for j in range(1, 20):
                                    target = np.array([b'Value %d' % j for _ in range(cycles)])
                                    vals = mdf.get(
                                        group=i, index=j + 1, samples_only=True
                                    )[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(
                                            i, j, vals, target, len(vals), len(target)
                                        )
                                    self.assertTrue(cond)

    def etest_cut_arrays(self):
        print("MDF cut big array files")
        for input_file in Path(TestMDF.tempdir_array.name).iterdir():
            for whence in (0, 1):
                print(input_file, whence)

                outfile1 = MDF(input_file)
                outfile1.configure(read_fragment_size=8000)
                outfile1 = outfile1.cut(stop=105.5, whence=whence, include_ends=False).save(
                    Path(TestMDF.tempdir.name) / "tmp1", overwrite=True
                )
                outfile2 = MDF(input_file)
                outfile2.configure(read_fragment_size=8000)
                outfile2 = outfile2.cut(
                    start=105.5, stop=201.5, whence=whence, include_ends=False
                ).save(Path(TestMDF.tempdir.name) / "tmp2", overwrite=True)
                outfile3 = MDF(input_file)
                outfile3.configure(read_fragment_size=8000)
                outfile3 = outfile3.cut(start=201.5, whence=whence, include_ends=False).save(
                    Path(TestMDF.tempdir.name) / "tmp3", overwrite=True
                )

                outfile = MDF.concatenate(
                    [outfile1, outfile2, outfile3],
                    MDF(input_file).version
                ).save(Path(TestMDF.tempdir.name) / "tmp_cut", overwrite=True)

                equal = True

                with MDF(outfile) as mdf:
                    mdf.configure(read_fragment_size=8000)

                    for i, group in enumerate(mdf.groups):

                        if i == 0:

                            samples = [
                                np.ones((cycles, 2, 3), dtype=np.uint64),
                                np.ones((cycles, 2), dtype=np.uint64),
                                np.ones((cycles, 3), dtype=np.uint64),
                            ]

                            for j in range(1, 20):

                                types = [
                                    ("Channel_{}".format(j), "(2, 3)<u8"),
                                    ("channel_{}_axis_1".format(j), "(2, )<u8"),
                                    ("channel_{}_axis_2".format(j), "(3, )<u8"),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get(
                                    "Channel_{}".format(j),
                                    group=i,
                                    samples_only=True,
                                )[0]
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(
                                    target, dtype=types
                                )
                                if not np.array_equal(vals, target):
                                    equal = False
                                    print(
                                        i,
                                        j,
                                        len(target),
                                        len(vals),
                                        vals,
                                        target,
                                        sep="\n\n",
                                    )
                                    1 / 0

                        elif i == 1:

                            samples = np.ones((cycles, 2, 3), dtype=np.uint64)
                            axis_0 = np.ones((cycles, 2), dtype=np.uint64)
                            axis_1 = np.ones((cycles, 3), dtype=np.uint64)

                            for j in range(1, 20):

                                types = [("Channel_{}".format(j), "(2, 3)<u8")]
                                types = np.dtype(types)

                                vals = mdf.get(
                                    "Channel_{}".format(j),
                                    group=i,
                                    samples_only=True,
                                )[0]
                                target = [samples * j]
                                target = np.core.records.fromarrays(
                                    target, dtype=types
                                )
                                if not np.array_equal(vals, target):
                                    equal = False
                                    1 / 0

                        elif i == 2:

                            samples = [
                                np.ones(cycles, dtype=np.uint8),
                                np.ones(cycles, dtype=np.uint16),
                                np.ones(cycles, dtype=np.uint32),
                                np.ones(cycles, dtype=np.uint64),
                                np.ones(cycles, dtype=np.int8),
                                np.ones(cycles, dtype=np.int16),
                                np.ones(cycles, dtype=np.int32),
                                np.ones(cycles, dtype=np.int64),
                            ]

                            for j in range(1, 20):

                                types = [
                                    ("struct_{}_channel_0".format(j), np.uint8),
                                    ("struct_{}_channel_1".format(j), np.uint16),
                                    ("struct_{}_channel_2".format(j), np.uint32),
                                    ("struct_{}_channel_3".format(j), np.uint64),
                                    ("struct_{}_channel_4".format(j), np.int8),
                                    ("struct_{}_channel_5".format(j), np.int16),
                                    ("struct_{}_channel_6".format(j), np.int32),
                                    ("struct_{}_channel_7".format(j), np.int64),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get(
                                    "Channel_{}".format(j),
                                    group=i,
                                    samples_only=True,
                                )[0]
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(
                                    target, dtype=types
                                )
                                if not np.array_equal(vals, target):
                                    equal = False
                                    1 / 0

            self.assertTrue(equal)

    def etest_cut_demo(self):
        print("MDF cut absolute tests")

        for input_file in Path(TestMDF.tempdir_demo.name).iterdir():

            if "2.00" in input_file.name:
                continue
            for whence in (0, 1):
                print(input_file, whence)

                outfile1 = (
                    MDF(input_file)
                    .cut(stop=2, whence=whence, include_ends=False)
                    .save(Path(TestMDF.tempdir.name) / "tmp1", overwrite=True)
                )
                outfile2 = (
                    MDF(input_file)
                    .cut(start=2, stop=6, whence=whence, include_ends=False)
                    .save(Path(TestMDF.tempdir.name) / "tmp2", overwrite=True)
                )
                outfile3 = (
                    MDF(input_file)
                    .cut(start=6, whence=whence, include_ends=False)
                    .save(Path(TestMDF.tempdir.name) / "tmp3", overwrite=True)
                )

                outfile = MDF.concatenate(
                    [outfile1, outfile2, outfile3],
                    vedrsion=MDF(input_file).version
                ).save(Path(TestMDF.tempdir.name) / "tmp", overwrite=True)

                print("OUT", outfile)

                equal = True

                with MDF(input_file) as mdf, MDF(
                    outfile) as mdf2:

                    for i, group in enumerate(mdf.groups):
                        for j, _ in enumerate(group["channels"][1:], 1):
                            original = mdf.get(group=i, index=j)
                            converted = mdf2.get(group=i, index=j)
                            if not np.array_equal(
                                original.samples, converted.samples
                            ):
                                equal = False
                            if not np.array_equal(
                                original.timestamps, converted.timestamps
                            ):
                                equal = False

                self.assertTrue(equal)

    def etest_filter(self):
        print("MDF filter tests")

        for input_file in Path(TestMDF.tempdir_demo.name).iterdir():

            if MDF(input_file).version <= "2.00":
                # if MDF(input_file, memory=memory).version < '4.00':
                continue

            channels_nr = np.random.randint(1, len(CHANNELS_DEMO) + 1)

            channel_list = random.sample(list(CHANNELS_DEMO), channels_nr)

            filtered_mdf = MDF(input_file).filter(
                channel_list)

            target = set(
                k
                for k in filtered_mdf.channels_db
                if not k.endswith("[0]")
                and not k.startswith("DI")
                and "\\" not in k
            )

            self.assertTrue((target - {"t", "time"}) == set(channel_list))

            equal = True

            with MDF(input_file) as mdf:
                print(input_file)
                names = list(channel_list)
                for name in channel_list:
                    self.assertTrue(name in mdf)

                names = [name + '_' for name in names]
                for name in names:
                    self.assertFalse(name in mdf)

                names = [name[:-3] for name in names]
                for name in names:
                    self.assertFalse(name in mdf)

                for name in channel_list:
                    original = mdf.get(name)
                    filtered = filtered_mdf.get(name)
                    if not np.array_equal(original.samples, filtered.samples):
                        equal = False
                    if not np.array_equal(original.timestamps, filtered.timestamps):
                        equal = False

            self.assertTrue(equal)

    def etest_select(self):
        print("MDF select tests")

        for input_file in Path(TestMDF.tempdir_demo.name).iterdir():

            if MDF(input_file).version == "2.00":
                continue

            print(input_file)

            channels_nr = np.random.randint(1, len(CHANNELS_DEMO) + 1)

            channel_list = random.sample(list(CHANNELS_DEMO), channels_nr)

            selected_signals = MDF(input_file).select(channel_list)

            self.assertTrue(len(selected_signals) == len(channel_list))

            self.assertTrue(
                all(
                    ch.name == name
                    for ch, name in zip(selected_signals, channel_list)
                )
            )

            equal = True

            with MDF(input_file) as mdf:

                for selected in selected_signals:
                    original = mdf.get(selected.name)
                    if not np.array_equal(original.samples, selected.samples):
                        equal = False
                    if not np.array_equal(original.timestamps, selected.timestamps):
                        equal = False

            self.assertTrue(equal)

    def etest_scramble(self):
        print("MDF scramble tests")

        for input_file in Path(TestMDF.tempdir_demo.name).iterdir():
            scrambled = MDF.scramble(input_file)
            self.assertTrue(scrambled)
            Path(scrambled).unlink()

    def etest_iter_groups(self):
        dfs = [
            DataFrame({f'df_{i}_column_0': np.ones(5) * i, f'df_{i}_column_1': np.arange(5) * i})
            for i in range(5)
        ]

        mdf = MDF()
        for df in dfs:
            mdf.append(df)

        for i, mdf_df in enumerate(mdf.iter_groups()):
            self.assertTrue(mdf_df.equals(dfs[i]))

    def etest_resample_raster_0(self):
        sigs = [
            Signal(
                samples=np.ones(1000) * i,
                timestamps=np.arange(1000),
                name=f'Signal_{i}',
            )
            for i in range(20)
        ]

        mdf = MDF()
        mdf.append(sigs)
        mdf.configure(read_fragment_size=1)
        mdf = mdf.resample(raster=0)

        for i, sig in enumerate(mdf.iter_channels(skip_master=True)):
            self.assertTrue(np.array_equal(sig.samples, sigs[i].samples))
            self.assertTrue(np.array_equal(sig.timestamps, sigs[i].timestamps))

    def etest_resample(self):
        raster = 1.33
        sigs = [
            Signal(
                samples=np.arange(1000, dtype='f8'),
                timestamps=np.concatenate([np.arange(500), np.arange(1000, 1500)]),
                name=f'Signal_{i}',
            )
            for i in range(20)
        ]

        mdf = MDF()
        mdf.append(sigs)
        mdf = mdf.resample(raster=raster)

        target_timestamps = np.arange(0, 1500, 1.33)
        target_samples = np.concatenate([
             np.arange(0, 500, 1.33),
             np.linspace(499.00215568862274, 499.9976646706587, 376),
             np.arange(500.1600000000001, 1000, 1.33)
             ]
        )

        for i, sig in enumerate(mdf.iter_channels(skip_master=True)):
            self.assertTrue(np.array_equal(sig.timestamps, target_timestamps))
            self.assertTrue(np.allclose(sig.samples, target_samples))

    def etest_to_dataframe(self):
        dfs = [
            DataFrame({f'df_{i}_column_0': np.ones(5) * i, f'df_{i}_column_1': np.arange(5) * i})
            for i in range(5)
        ]

        mdf = MDF()
        for df in dfs:
            mdf.append(df)

        target = {}
        for i in range(5):
            target[f'df_{i}_column_0'] = np.ones(5) * i
            target[f'df_{i}_column_1'] = np.arange(5) * i

        target = DataFrame(target)

        self.assertTrue(target.equals(mdf.to_dataframe()))


if __name__ == "__main__":
    unittest.main()

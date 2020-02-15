#!/usr/bin/env python
import random
import unittest
import urllib
import urllib.request
import numexpr
from zipfile import ZipFile
import tempfile
from pathlib import Path
from io import BytesIO

import numpy as np

from .utils import (
    cycles,
    generate_test_file,
    generate_arrays_test_file,
)
from asammdf import MDF, Signal, SUPPORTED_VERSIONS
from asammdf.blocks.utils import MdfException
from pandas import DataFrame

SUPPORTED_VERSIONS = [version for version in SUPPORTED_VERSIONS if "4.20" > version >= "3.20" ]

CHANNEL_LEN = 100000


class TestMDF(unittest.TestCase):

    tempdir_demo = None
    tempdir_array = None
    tempdir_general = None
    tempdir = None

    def test_measurement(self):
        self.assertTrue(MDF)

    @classmethod
    def setUpClass(cls):

        url = "https://github.com/danielhrisca/asammdf/files/4078993/test.demo.zip"
        urllib.request.urlretrieve(url, "test.zip")

        cls.tempdir_demo = tempfile.TemporaryDirectory()
        cls.tempdir_general = tempfile.TemporaryDirectory()
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.tempdir_array = tempfile.TemporaryDirectory()

        ZipFile(r"test.zip").extractall(cls.tempdir_demo.name)
        Path("test.zip").unlink()
        for version in ("3.30", "4.10"):
            generate_test_file(cls.tempdir_general.name, version)

        generate_arrays_test_file(cls.tempdir_array.name)

    def test_mdf_header(self):
        mdf = BytesIO(b"M" * 100)
        with self.assertRaises(MdfException):
            MDF(mdf)

    def test_wrong_header_version(self):
        mdf = BytesIO(
            b"MDF     AAAA    amdf500d\x00\x00\x00\x00\x9f\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        )
        with self.assertRaises(MdfException):
            MDF(mdf)

    def test_read(self):
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
                                target = np.array(
                                    [b"Value %d" % j for _ in range(cycles)]
                                )
                                vals = mdf.get(group=i, index=j + 1, samples_only=True)[
                                    0
                                ]
                                cond = np.array_equal(vals, target)
                                if not cond:
                                    print(i, j, vals, target, len(vals), len(target))
                                self.assertTrue(cond)
            self.assertTrue(equal)

    def test_read_arrays(self):
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

        mdf_files = [
            file
            for file in Path(TestMDF.tempdir_demo.name).iterdir()
            if file.suffix in ('.mdf', '.mf4')
        ]

        signals = [
            file
            for file in Path(TestMDF.tempdir_demo.name).iterdir()
            if file.suffix == '.npy'
        ]

        for file in mdf_files:
            print(file)

            for inp in (file, BytesIO(file.read_bytes())):

                with MDF(inp, use_display_names=True) as input_file:

                    for signal in signals:
                        name = signal.stem
                        target = np.load(signal)
                        values = input_file.get(name).samples

                        self.assertTrue(np.array_equal(target, values))

    def test_convert(self):
        print("MDF convert big files tests")

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
                                    target = np.array(
                                        [b"Value %d" % j for _ in range(cycles)]
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

                    self.assertTrue(equal)

    def test_convert_demo(self):
        print("MDF convert demo tests")

        mdf_files = [
            file
            for file in Path(TestMDF.tempdir_demo.name).iterdir()
            if file.suffix in ('.mdf', '.mf4')
        ]

        signals = [
            file
            for file in Path(TestMDF.tempdir_demo.name).iterdir()
            if file.suffix == '.npy'
        ]

        for file in mdf_files:

            for inp in (file, BytesIO(file.read_bytes())):

                with MDF(inp, use_display_names=True) as input_file:

                    for out in SUPPORTED_VERSIONS:
                        print(file, out, type(inp))

                    outfile = input_file.convert(out).save(
                        Path(TestMDF.tempdir_demo.name) / "tmp", overwrite=True,
                    )

                    with MDF(outfile, use_display_names=True) as mdf:

                        for signal in signals:
                            name = signal.stem
                            target = np.load(signal)
                            values = mdf.get(name).samples

                            self.assertTrue(np.array_equal(target, values))

    def test_cut(self):
        print("MDF cut big files tests")

        t = np.arange(cycles, dtype=np.float64)

        for input_file in Path(TestMDF.tempdir_general.name).iterdir():
            for whence in (0, 1):
                for compression in range(3):
                    print(input_file)

                    outfile0 = MDF(input_file)
                    outfile0.configure(read_fragment_size=8000)
                    outfile0 = outfile0.cut(
                        stop=-1, whence=whence, include_ends=False
                    ).save(
                        Path(TestMDF.tempdir.name) / "tmp0",
                        overwrite=True,
                        compression=compression,
                    )

                    outfile1 = MDF(input_file)
                    outfile1.configure(read_fragment_size=8000)
                    outfile1 = outfile1.cut(
                        stop=105, whence=whence, include_ends=False
                    ).save(
                        Path(TestMDF.tempdir.name) / "tmp1",
                        overwrite=True,
                        compression=compression,
                    )

                    outfile2 = MDF(input_file)
                    outfile2.configure(read_fragment_size=8000)
                    outfile2 = outfile2.cut(
                        start=105.1, stop=201, whence=whence, include_ends=False
                    ).save(
                        Path(TestMDF.tempdir.name) / "tmp2",
                        overwrite=True,
                        compression=compression,
                    )

                    outfile3 = MDF(input_file)
                    outfile3.configure(read_fragment_size=8000)
                    outfile3 = outfile3.cut(
                        start=201.1, whence=whence, include_ends=False
                    ).save(Path(TestMDF.tempdir.name) / "tmp3", overwrite=True)

                    outfile4 = MDF(input_file)
                    outfile4.configure(read_fragment_size=8000)
                    outfile4 = outfile4.cut(
                        start=7000, whence=whence, include_ends=False
                    ).save(
                        Path(TestMDF.tempdir.name) / "tmp4",
                        overwrite=True,
                        compression=compression,
                    )

                    outfile = MDF.concatenate(
                        [outfile0, outfile1, outfile2, outfile3, outfile4],
                        version=MDF(input_file).version,
                        sync=whence,
                    ).save(
                        Path(TestMDF.tempdir.name) / "tmp_cut",
                        overwrite=True,
                        compression=compression,
                    )

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
                                    target = np.array(
                                        [b"Value %d" % j for _ in range(cycles)]
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

    def test_cut_arrays(self):
        print("MDF cut big array files")
        for input_file in Path(TestMDF.tempdir_array.name).iterdir():
            for whence in (0, 1):
                print(input_file, whence)

                outfile1 = MDF(input_file)
                outfile1.configure(read_fragment_size=8000)
                outfile1 = outfile1.cut(
                    stop=105.5, whence=whence, include_ends=False
                ).save(Path(TestMDF.tempdir.name) / "tmp1", overwrite=True)
                outfile2 = MDF(input_file)
                outfile2.configure(read_fragment_size=8000)
                outfile2 = outfile2.cut(
                    start=105.5, stop=201.5, whence=whence, include_ends=False
                ).save(Path(TestMDF.tempdir.name) / "tmp2", overwrite=True)
                outfile3 = MDF(input_file)
                outfile3.configure(read_fragment_size=8000)
                outfile3 = outfile3.cut(
                    start=201.5, whence=whence, include_ends=False
                ).save(Path(TestMDF.tempdir.name) / "tmp3", overwrite=True)

                outfile = MDF.concatenate(
                    [outfile1, outfile2, outfile3], MDF(input_file).version
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
                                    "Channel_{}".format(j), group=i, samples_only=True,
                                )[0]
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(target, dtype=types)
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
                                    "Channel_{}".format(j), group=i, samples_only=True,
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
                                    "Channel_{}".format(j), group=i, samples_only=True,
                                )[0]
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(target, dtype=types)
                                if not np.array_equal(vals, target):
                                    equal = False
                                    1 / 0

            self.assertTrue(equal)

    def test_cut_demo(self):
        print("MDF cut demo tests")

        mdf_files = [
            file
            for file in Path(TestMDF.tempdir_demo.name).iterdir()
            if file.suffix in ('.mdf', '.mf4')
        ]

        signals = [
            file
            for file in Path(TestMDF.tempdir_demo.name).iterdir()
            if file.suffix == '.npy'
        ]

        for file in mdf_files:
            print(file)

            for inp in (file, BytesIO(file.read_bytes())):

                with MDF(inp, use_display_names=True) as input_file:

                    for whence in (0, 1):
                        print(file, whence)

                        outfile1 = (
                            input_file
                            .cut(stop=2, whence=whence, include_ends=False)
                            .save(Path(TestMDF.tempdir.name) / "tmp1", overwrite=True)
                        )
                        outfile2 = (
                            input_file
                            .cut(start=2, stop=6, whence=whence, include_ends=False)
                            .save(Path(TestMDF.tempdir.name) / "tmp2", overwrite=True)
                        )
                        outfile3 = (
                            input_file
                            .cut(start=6, whence=whence, include_ends=False)
                            .save(Path(TestMDF.tempdir.name) / "tmp3", overwrite=True)
                        )

                        outfile = MDF.concatenate(
                            [outfile1, outfile2, outfile3], version=input_file.version,
                            use_display_names=True,
                        ).save(Path(TestMDF.tempdir.name) / "tmp", overwrite=True)

                        print("OUT", outfile)

                        with MDF(outfile, use_display_names=True) as mdf2:

                            for signal in signals:
                                target = np.load(signal)
                                sig = mdf2.get(signal.stem)
                                timestamps = input_file.get(signal.stem).timestamps

                                self.assertTrue(np.array_equal(sig.samples, target))
                                self.assertTrue(np.array_equal(timestamps, sig.timestamps))

    def test_filter(self):

        print("MDF read tests")

        mdf_files = [
            file
            for file in Path(TestMDF.tempdir_demo.name).iterdir()
            if file.suffix in ('.mdf', '.mf4')
        ]

        signals = {
            file.stem: file
            for file in Path(TestMDF.tempdir_demo.name).iterdir()
            if file.suffix == '.npy'
        }

        for file in mdf_files:
            print(file)

            for inp in (file, BytesIO(file.read_bytes())):

                with MDF(inp, use_display_names=True) as input_file:

                    names = [
                        ch.name
                        for gp in input_file.groups
                        for ch in gp.channels[1:]
                    ]

                    channels_nr = np.random.randint(1, len(names) + 1)

                    channel_list = random.sample(names, channels_nr)

                    filtered_mdf = input_file.filter(channel_list)

                    target_names = set(filtered_mdf.channels_db)


                    self.assertFalse(set(channel_list) - (target_names - {'time'}))

                    for name in channel_list:
                        target = np.load(signals[name])
                        filtered = filtered_mdf.get(name)
                        self.assertTrue(np.array_equal(target, filtered.samples))

    def test_select(self):
        print("MDF select tests")

        mdf_files = [
            file
            for file in Path(TestMDF.tempdir_demo.name).iterdir()
            if file.suffix in ('.mdf', '.mf4')
        ]

        signals = {
            file.stem: file
            for file in Path(TestMDF.tempdir_demo.name).iterdir()
            if file.suffix == '.npy'
        }

        for file in mdf_files:
            print(file)

            for inp in (file, BytesIO(file.read_bytes())):

                with MDF(inp, use_display_names=True) as input_file:

                    names = [
                        ch.name
                        for gp in input_file.groups
                        for ch in gp.channels[1:]
                    ]

                    input_file.configure(read_fragment_size=200)

                    channels_nr = np.random.randint(1, len(names) + 1)
                    channel_list = random.sample(names, channels_nr)

                    selected_signals = input_file.select(channel_list)

                    target_names = set(s.name for s in selected_signals)

                    self.assertFalse(set(target_names) - set(channel_list))

                    for name, sig in zip(channel_list, selected_signals):
                        target = np.load(signals[name])
                        self.assertTrue(np.array_equal(target, sig.samples))

    def test_scramble(self):
        print("MDF scramble tests")

        for input_file in Path(TestMDF.tempdir_demo.name).iterdir():
            if input_file.suffix.lower() in ('.mdf', '.mf4'):
                scrambled = MDF.scramble(input_file)
                self.assertTrue(scrambled)
                Path(scrambled).unlink()

    def test_iter_groups(self):
        dfs = [
            DataFrame(
                {
                    f"df_{i}_column_0": np.ones(5) * i,
                    f"df_{i}_column_1": np.arange(5) * i,
                }
            )
            for i in range(5)
        ]

        mdf = MDF()
        for df in dfs:
            mdf.append(df)

        for i, mdf_df in enumerate(mdf.iter_groups()):
            self.assertTrue(mdf_df.equals(dfs[i]))

    def test_resample_raster_0(self):
        sigs = [
            Signal(
                samples=np.ones(1000) * i,
                timestamps=np.arange(1000),
                name=f"Signal_{i}",
            )
            for i in range(20)
        ]

        mdf = MDF()
        mdf.append(sigs)
        mdf.configure(read_fragment_size=1)
        with self.assertRaises(AssertionError):
            mdf = mdf.resample(raster=0)

    def test_resample(self):
        raster = 1.33
        sigs = [
            Signal(
                samples=np.arange(1000, dtype="f8"),
                timestamps=np.concatenate([np.arange(500), np.arange(1000, 1500)]),
                name=f"Signal_{i}",
            )
            for i in range(20)
        ]

        mdf = MDF()
        mdf.append(sigs)
        mdf = mdf.resample(raster=raster)

        target_timestamps = np.arange(0, 1500, 1.33)
        target_samples = np.concatenate(
            [
                np.arange(0, 500, 1.33),
                np.linspace(499.00215568862274, 499.9976646706587, 376),
                np.arange(500.1600000000001, 1000, 1.33),
            ]
        )

        for i, sig in enumerate(mdf.iter_channels(skip_master=True)):
            self.assertTrue(np.array_equal(sig.timestamps, target_timestamps))
            self.assertTrue(np.allclose(sig.samples, target_samples))

    def test_to_dataframe(self):
        dfs = [
            DataFrame(
                {
                    f"df_{i}_column_0": np.ones(5) * i,
                    f"df_{i}_column_1": np.arange(5) * i,
                }
            )
            for i in range(5)
        ]

        mdf = MDF()
        for df in dfs:
            mdf.append(df)

        target = {}
        for i in range(5):
            target[f"df_{i}_column_0"] = np.ones(5) * i
            target[f"df_{i}_column_1"] = np.arange(5) * i

        target = DataFrame(target)

        self.assertTrue(target.equals(mdf.to_dataframe()))


if __name__ == "__main__":
    unittest.main()

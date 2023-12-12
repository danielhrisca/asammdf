#!/usr/bin/env python
from io import BytesIO
from pathlib import Path
import random
import tempfile
import unittest
import urllib
import urllib.request
from zipfile import ZipFile

import numexpr
import numpy as np
from pandas import DataFrame

from asammdf import MDF, Signal, SUPPORTED_VERSIONS
from asammdf.blocks.utils import MdfException
from asammdf.mdf import SearchMode

from .utils import cycles, generate_arrays_test_file, generate_test_file

SUPPORTED_VERSIONS = [version for version in SUPPORTED_VERSIONS if "4.20" > version >= "3.20"]

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

        mdf.close()

    def test_wrong_header_version(self):
        mdf = BytesIO(
            b"MDF     AAAA    amdf500d\x00\x00\x00\x00\x9f\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        )
        with self.assertRaises(MdfException):
            MDF(mdf)

        mdf.close()

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
                                target = np.array([f"Channel {j} sample {k}".encode("ascii") for k in range(cycles)])
                                vals = mdf.get(group=i, index=j + 1, samples_only=True)[0]
                                cond = np.array_equal(vals, target)
                                if not cond:
                                    print(i, j, vals, target, len(vals), len(target))
                                self.assertTrue(cond)

                        elif i == 5:
                            v = np.ones(cycles, dtype=np.dtype("(8,)u1"))
                            for j in range(1, 20):
                                target = v * j
                                vals = mdf.get(group=i, index=j + 1, samples_only=True)[0]
                                cond = np.array_equal(vals, target)
                                if not cond:
                                    print(i, j, vals, target, len(vals), len(target))
                                self.assertTrue(cond)

                        elif i == 6:
                            for j in range(1, 20):
                                target = np.array([b"Value %d" % j for _ in range(cycles)])
                                vals = mdf.get(group=i, index=j + 1, samples_only=True)[0]
                                cond = np.array_equal(vals, target)
                                if not cond:
                                    print(i, j, vals, target, len(vals), len(target))
                                self.assertTrue(cond)

                if isinstance(inp, BytesIO):
                    inp.close()
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
                                    (f"Channel_{j}", "(2, 3)<u8"),
                                    (f"channel_{j}_axis_1", "(2, )<u8"),
                                    (f"channel_{j}_axis_2", "(3, )<u8"),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get(f"Channel_{j}", group=i, samples_only=True)[0]
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(target, dtype=types)
                                if not np.array_equal(vals, target):
                                    equal = False
                                    raise Exception

                        elif i == 1:
                            samples = np.ones((cycles, 2, 3), dtype=np.uint64)
                            axis_0 = np.ones((cycles, 2), dtype=np.uint64)
                            axis_1 = np.ones((cycles, 3), dtype=np.uint64)

                            for j in range(1, 20):
                                types = [(f"Channel_{j}", "(2, 3)<u8")]
                                types = np.dtype(types)

                                vals = mdf.get(f"Channel_{j}", group=i, samples_only=True)[0]
                                target = [samples * j]
                                target = np.core.records.fromarrays(target, dtype=types)
                                if not np.array_equal(vals, target):
                                    equal = False
                                    raise Exception

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
                                    (f"struct_{j}_channel_0", np.uint8),
                                    (f"struct_{j}_channel_1", np.uint16),
                                    (f"struct_{j}_channel_2", np.uint32),
                                    (f"struct_{j}_channel_3", np.uint64),
                                    (f"struct_{j}_channel_4", np.int8),
                                    (f"struct_{j}_channel_5", np.int16),
                                    (f"struct_{j}_channel_6", np.int32),
                                    (f"struct_{j}_channel_7", np.int64),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get(f"Channel_{j}", group=i, samples_only=True)[0]
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(target, dtype=types)
                                if not np.array_equal(vals, target):
                                    equal = False
                                    print(target)
                                    print(vals)
                                    raise Exception

                if isinstance(inp, BytesIO):
                    inp.close()

            self.assertTrue(equal)

    def test_read_demo(self):
        print("MDF read tests")

        mdf_files = [file for file in Path(TestMDF.tempdir_demo.name).iterdir() if file.suffix in (".mdf", ".mf4")]

        signals = [file for file in Path(TestMDF.tempdir_demo.name).iterdir() if file.suffix == ".npy"]

        for file in mdf_files:
            print(file)

            for inp in (file, BytesIO(file.read_bytes())):
                with MDF(inp, use_display_names=True) as input_file:
                    for signal in signals:
                        name = signal.stem
                        target = np.load(signal)
                        values = input_file.get(name, *input_file.whereis(name)[0]).samples

                        self.assertTrue(np.array_equal(target, values))

                if isinstance(inp, BytesIO):
                    inp.close()

    def test_convert(self):
        print("MDF convert big files tests")

        for out in SUPPORTED_VERSIONS:
            for input_file in Path(TestMDF.tempdir_general.name).iterdir():
                for compression in range(3):
                    print(input_file, out)

                    with MDF(input_file) as mdf:
                        mdf.configure(read_fragment_size=8000)
                        converted = mdf.convert(out)
                        outfile = converted.save(
                            Path(TestMDF.tempdir.name) / f"tmp_convert_{out}",
                            overwrite=True,
                            compression=compression,
                        )
                        converted.close()

                    equal = True

                    with MDF(outfile) as mdf:
                        mdf.configure(read_fragment_size=8000)

                        for i, group in enumerate(mdf.groups):
                            if i == 0:
                                v = np.ones(cycles, dtype=np.uint64)
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                    if not np.array_equal(vals, v * (j - 1)):
                                        equal = False
                                        print(vals, len(vals))
                                        print(v * (j - 1), len(v))

                                        input(outfile)
                                        raise Exception
                            elif i == 1:
                                v = np.ones(cycles, dtype=np.int64)
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                    if not np.array_equal(vals, v * (j - 1) - 0.5):
                                        equal = False
                                        raise Exception
                            elif i == 2:
                                v = np.arange(cycles, dtype=np.int64) / 100.0
                                form = "{} * sin(v)"
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                    f = form.format(j - 1)
                                    if not np.array_equal(vals, numexpr.evaluate(f)):
                                        equal = False
                                        raise Exception
                            elif i == 3:
                                v = np.ones(cycles, dtype=np.int64)
                                form = "({} * v -0.5) / 1"
                                for j in range(1, 20):
                                    f = form.format(j - 1)
                                    vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                    if not np.array_equal(vals, numexpr.evaluate(f)):
                                        equal = False
                                        target = numexpr.evaluate(f)
                                        print(i, j, vals, target, len(vals), len(target))
                                        raise Exception
                            elif i == 4:
                                for j in range(1, 20):
                                    target = np.array(
                                        [f"Channel {j} sample {k}".encode("ascii") for k in range(cycles)]
                                    )
                                    vals = mdf.get(group=i, index=j + 1, samples_only=True)[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(i, j, vals, target, len(vals), len(target))
                                        raise Exception
                                    self.assertTrue(cond)

                            elif i == 5:
                                v = np.ones(cycles, dtype=np.dtype("(8,)u1"))
                                for j in range(1, 20):
                                    target = v * j
                                    vals = mdf.get(group=i, index=j + 1, samples_only=True)[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(i, j, vals, target, len(vals), len(target))
                                        raise Exception
                                    self.assertTrue(cond)

                            elif i == 6:
                                for j in range(1, 20):
                                    target = np.array([b"Value %d" % j for _ in range(cycles)])
                                    vals = mdf.get(group=i, index=j + 1, samples_only=True)[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(i, j, vals, target, len(vals), len(target))
                                        raise Exception
                                    self.assertTrue(cond)

                    self.assertTrue(equal)

    def test_convert_demo(self):
        print("MDF convert demo tests")

        mdf_files = [file for file in Path(TestMDF.tempdir_demo.name).iterdir() if file.suffix in (".mdf", ".mf4")]

        signals = [file for file in Path(TestMDF.tempdir_demo.name).iterdir() if file.suffix == ".npy"]

        for file in mdf_files:
            for inp in (file, BytesIO(file.read_bytes())):
                with MDF(inp, use_display_names=True) as input_file:
                    for out in SUPPORTED_VERSIONS:
                        print(file, out, type(inp))

                    converted = input_file.convert(out)
                    outfile = converted.save(Path(TestMDF.tempdir_demo.name) / "tmp", overwrite=True)
                    converted.close()

                    with MDF(outfile, use_display_names=True) as mdf:
                        for signal in signals:
                            name = signal.stem
                            target = np.load(signal)
                            values = mdf.get(name, *mdf.whereis(name)[0]).samples

                            self.assertTrue(np.array_equal(target, values))

                if isinstance(inp, BytesIO):
                    inp.close()

    def test_cut(self):
        print("MDF cut big files tests")

        for input_file in Path(TestMDF.tempdir_general.name).iterdir():
            for whence in (0, 1):
                for compression in range(3):
                    mdf = MDF(input_file)

                    mdf.configure(read_fragment_size=8000)
                    cut = mdf.cut(stop=-1, whence=whence, include_ends=False)
                    outfile0 = cut.save(Path(TestMDF.tempdir.name) / "tmp0", overwrite=True)
                    cut.close()

                    mdf.configure(read_fragment_size=8000)
                    cut = mdf.cut(stop=105, whence=whence, include_ends=False)
                    outfile1 = cut.save(Path(TestMDF.tempdir.name) / "tmp1", overwrite=True)
                    cut.close()

                    mdf.configure(read_fragment_size=8000)
                    cut = mdf.cut(start=105.1, stop=201, whence=whence, include_ends=False)
                    outfile2 = cut.save(Path(TestMDF.tempdir.name) / "tmp2", overwrite=True)
                    cut.close()

                    mdf.configure(read_fragment_size=8000)
                    cut = mdf.cut(start=201.1, whence=whence, include_ends=False)
                    outfile3 = cut.save(Path(TestMDF.tempdir.name) / "tmp3", overwrite=True)
                    cut.close()

                    mdf.configure(read_fragment_size=8000)
                    cut = mdf.cut(start=7000, whence=whence, include_ends=False)
                    outfile4 = cut.save(Path(TestMDF.tempdir.name) / "tmp4", overwrite=True)
                    cut.close()

                    concatenated = MDF.concatenate(
                        [outfile0, outfile1, outfile2, outfile3, outfile4],
                        version=mdf.version,
                        sync=whence,
                    )

                    outfile = concatenated.save(
                        Path(TestMDF.tempdir.name) / "tmp_cut",
                        overwrite=True,
                        compression=compression,
                    )

                    mdf.close()
                    concatenated.close()

                    with MDF(outfile) as mdf:
                        for i, group in enumerate(mdf.groups):
                            if i == 0:
                                v = np.ones(cycles, dtype=np.uint64)
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                    cond = np.array_equal(vals, v * (j - 1))
                                    if not cond:
                                        print(i, j, vals, v * (j - 1), len(vals), len(v))
                                    self.assertTrue(cond)
                            elif i == 1:
                                v = np.ones(cycles, dtype=np.int64)
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                    cond = np.array_equal(vals, v * (j - 1) - 0.5)
                                    if not cond:
                                        print(vals, v * (j - 1) - 0.5, len(vals), len(v))
                                    self.assertTrue(cond)
                            elif i == 2:
                                v = np.arange(cycles, dtype=np.int64) / 100.0
                                form = "{} * sin(v)"
                                for j in range(1, 20):
                                    vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                    f = form.format(j - 1)
                                    target = numexpr.evaluate(f)
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(i, j, vals, target, len(vals), len(target))
                                    self.assertTrue(cond)
                            elif i == 3:
                                v = np.ones(cycles, dtype=np.int64)
                                form = "({} * v -0.5) / 1"
                                for j in range(1, 20):
                                    f = form.format(j - 1)
                                    vals = mdf.get(group=i, index=j, samples_only=True)[0]
                                    target = numexpr.evaluate(f)
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(i, j, vals, target, len(vals), len(target))
                                    self.assertTrue(cond)
                            elif i == 4:
                                for j in range(1, 20):
                                    target = np.array(
                                        [f"Channel {j} sample {k}".encode("ascii") for k in range(cycles)]
                                    )
                                    vals = mdf.get(group=i, index=j + 1, samples_only=True)[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(i, j, vals, target, len(vals), len(target))
                                    self.assertTrue(cond)

                            elif i == 5:
                                v = np.ones(cycles, dtype=np.dtype("(8,)u1"))
                                for j in range(1, 20):
                                    target = v * j
                                    vals = mdf.get(group=i, index=j + 1, samples_only=True)[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(i, j, vals, target, len(vals), len(target))
                                    self.assertTrue(cond)

                            elif i == 6:
                                for j in range(1, 20):
                                    target = np.array([b"Value %d" % j for _ in range(cycles)])
                                    vals = mdf.get(group=i, index=j + 1, samples_only=True)[0]
                                    cond = np.array_equal(vals, target)
                                    if not cond:
                                        print(i, j, vals, target, len(vals), len(target))
                                    self.assertTrue(cond)

    def test_cut_arrays(self):
        print("MDF cut big array files")
        for input_file in Path(TestMDF.tempdir_array.name).iterdir():
            for whence in (0, 1):
                print(input_file, whence)

                mdf = MDF(input_file)

                mdf.configure(read_fragment_size=8000)
                cut = mdf.cut(stop=105.5, whence=whence, include_ends=False)
                outfile1 = cut.save(Path(TestMDF.tempdir.name) / "tmp1", overwrite=True)
                cut.close()

                cut = mdf.cut(start=105.5, stop=201.5, whence=whence, include_ends=False)
                outfile2 = cut.save(Path(TestMDF.tempdir.name) / "tmp2", overwrite=True)
                cut.close()

                cut = mdf.cut(start=201.5, whence=whence, include_ends=False)
                outfile3 = cut.save(Path(TestMDF.tempdir.name) / "tmp3", overwrite=True)
                cut.close()

                concatenated = MDF.concatenate([outfile1, outfile2, outfile3], mdf.version)
                outfile = concatenated.save(Path(TestMDF.tempdir.name) / "tmp_cut", overwrite=True)

                mdf.close()
                concatenated.close()

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
                                    (f"Channel_{j}", "(2, 3)<u8"),
                                    (f"channel_{j}_axis_1", "(2, )<u8"),
                                    (f"channel_{j}_axis_2", "(3, )<u8"),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get(f"Channel_{j}", group=i, samples_only=True)[0]
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
                                    raise Exception

                        elif i == 1:
                            samples = np.ones((cycles, 2, 3), dtype=np.uint64)
                            axis_0 = np.ones((cycles, 2), dtype=np.uint64)
                            axis_1 = np.ones((cycles, 3), dtype=np.uint64)

                            for j in range(1, 20):
                                types = [(f"Channel_{j}", "(2, 3)<u8")]
                                types = np.dtype(types)

                                vals = mdf.get(f"Channel_{j}", group=i, samples_only=True)[0]
                                target = [samples * j]
                                target = np.core.records.fromarrays(target, dtype=types)
                                if not np.array_equal(vals, target):
                                    equal = False
                                    raise Exception

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
                                    (f"struct_{j}_channel_0", np.uint8),
                                    (f"struct_{j}_channel_1", np.uint16),
                                    (f"struct_{j}_channel_2", np.uint32),
                                    (f"struct_{j}_channel_3", np.uint64),
                                    (f"struct_{j}_channel_4", np.int8),
                                    (f"struct_{j}_channel_5", np.int16),
                                    (f"struct_{j}_channel_6", np.int32),
                                    (f"struct_{j}_channel_7", np.int64),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get(f"Channel_{j}", group=i, samples_only=True)[0]
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(target, dtype=types)
                                if not np.array_equal(vals, target):
                                    equal = False
                                    raise Exception

            self.assertTrue(equal)

    def test_cut_demo(self):
        print("MDF cut demo tests")

        mdf_files = [file for file in Path(TestMDF.tempdir_demo.name).iterdir() if file.suffix in (".mdf", ".mf4")]

        signals = [file for file in Path(TestMDF.tempdir_demo.name).iterdir() if file.suffix == ".npy"]

        for file in mdf_files:
            print(file)

            for inp in (file, BytesIO(file.read_bytes())):
                with MDF(inp, use_display_names=True) as input_file:
                    for whence in (0, 1):
                        print(file, whence)

                        cut = input_file.cut(stop=2, whence=whence, include_ends=False)
                        outfile1 = cut.save(Path(TestMDF.tempdir.name) / "tmp1", overwrite=True)
                        cut.close()

                        cut = input_file.cut(start=2, stop=6, whence=whence, include_ends=False)
                        outfile2 = cut.save(Path(TestMDF.tempdir.name) / "tmp2", overwrite=True)
                        cut.close()

                        cut = input_file.cut(start=6, whence=whence, include_ends=False)
                        outfile3 = cut.save(Path(TestMDF.tempdir.name) / "tmp3", overwrite=True)
                        cut.close()

                        concatenated = MDF.concatenate(
                            [outfile1, outfile2, outfile3],
                            version=input_file.version,
                            use_display_names=True,
                        )

                        outfile = concatenated.save(Path(TestMDF.tempdir.name) / "tmp", overwrite=True)

                        print("OUT", outfile)
                        concatenated.close()

                        with MDF(outfile, use_display_names=True) as mdf2:
                            for signal in signals:
                                target = np.load(signal)
                                sig = mdf2.get(signal.stem, *mdf2.whereis(signal.stem)[0])
                                timestamps = input_file.get(signal.stem, *input_file.whereis(signal.stem)[0]).timestamps

                                self.assertTrue(np.array_equal(sig.samples, target))
                                self.assertTrue(np.array_equal(timestamps, sig.timestamps))

                if isinstance(inp, BytesIO):
                    inp.close()

    def test_filter(self):
        print("MDF read tests")

        mdf_files = [file for file in Path(TestMDF.tempdir_demo.name).iterdir() if file.suffix in (".mdf", ".mf4")]

        signals = {file.stem: file for file in Path(TestMDF.tempdir_demo.name).iterdir() if file.suffix == ".npy"}

        for file in mdf_files:
            print(file)

            for inp in (file, BytesIO(file.read_bytes())):
                with MDF(inp, use_display_names=True) as input_file:
                    names = [ch.name for gp in input_file.groups for ch in gp.channels[1:]]

                    channels_nr = np.random.randint(1, len(names) + 1)

                    channel_list = random.sample(names, channels_nr)

                    filtered_mdf = input_file.filter(channel_list)

                    target_names = set(filtered_mdf.channels_db)

                    self.assertFalse(set(channel_list) - (target_names - {"time"}))

                    for name in channel_list:
                        target = np.load(signals[name])
                        filtered = filtered_mdf.get(name)
                        self.assertTrue(np.array_equal(target, filtered.samples))

                    filtered_mdf.close()

                if isinstance(inp, BytesIO):
                    inp.close()

    def test_select(self):
        print("MDF select tests")

        mdf_files = [file for file in Path(TestMDF.tempdir_demo.name).iterdir() if file.suffix in (".mdf", ".mf4")]

        signals = {file.stem: file for file in Path(TestMDF.tempdir_demo.name).iterdir() if file.suffix == ".npy"}

        for file in mdf_files:
            print(file)

            for inp in (file, BytesIO(file.read_bytes())):
                with MDF(inp, use_display_names=True) as input_file:
                    names = [ch.name for gp in input_file.groups for ch in gp.channels[1:]]

                    input_file.configure(read_fragment_size=200)

                    channels_nr = np.random.randint(1, len(names) + 1)
                    channel_list = random.sample(names, channels_nr)

                    selected_signals = input_file.select(channel_list)

                    target_names = {s.name for s in selected_signals}

                    self.assertFalse(set(target_names) - set(channel_list))

                    for name, sig in zip(channel_list, selected_signals):
                        target = np.load(signals[name])
                        self.assertTrue(np.array_equal(target, sig.samples))

                if isinstance(inp, BytesIO):
                    inp.close()

    def test_scramble(self):
        print("MDF scramble tests")

        for input_file in Path(TestMDF.tempdir_demo.name).iterdir():
            if input_file.suffix.lower() in (".mdf", ".mf4"):
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

        mdf.close()

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

        mdf.close()

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
        resampled = mdf.resample(raster=raster)
        mdf.close()

        target_timestamps = np.arange(0, 1500, 1.33)
        target_samples = np.concatenate(
            [
                np.arange(0, 500, 1.33),
                np.linspace(499.00215568862274, 499.9976646706587, 376),
                np.arange(500.1600000000001, 1000, 1.33),
            ]
        )

        for i, sig in enumerate(resampled.iter_channels(skip_master=True)):
            self.assertTrue(np.array_equal(sig.timestamps, target_timestamps))
            self.assertTrue(np.allclose(sig.samples, target_samples))

        resampled.close()

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
        mdf.close()

    def test_search(self):
        sigs = [
            Signal(
                samples=np.ones(1),
                timestamps=np.arange(1),
                name=ch_name,
            )
            for ch_name in ["foo", "Bar", "baz"]
        ]
        mdf = MDF()
        mdf.append(sigs)

        # plain text matching
        self.assertEqual(["foo"], mdf.search(pattern="foo"), msg="name match full")
        self.assertEqual(["foo"], mdf.search(pattern="oo", mode="plain"), msg="name match part")
        self.assertEqual(
            [],
            mdf.search(pattern="FOO", mode=SearchMode.plain, case_insensitive=False),
            msg="name match case-sensitive (no match)",
        )
        self.assertEqual(
            ["foo"],
            mdf.search(pattern="FOO", mode=SearchMode.plain, case_insensitive=True),
            msg="name match case-insensitive (match)",
        )

        # regex matching
        self.assertEqual(["Bar"], mdf.search(pattern="^Bar$", mode="regex"), msg="regex match full")
        self.assertEqual(
            ["baz"],
            mdf.search(pattern="z$", mode=SearchMode.regex),
            msg="regex match part",
        )
        self.assertEqual(
            ["baz"],
            mdf.search(pattern="b.*", mode=SearchMode.regex, case_insensitive=False),
            msg="regex match case-sensitive",
        )
        self.assertEqual(
            ["Bar", "baz"],
            mdf.search(pattern="b.*", mode=SearchMode.regex, case_insensitive=True),
            msg="regex match case-insensitive",
        )

        # wildcard matching
        self.assertEqual(
            ["Bar"],
            mdf.search(pattern="Bar", mode="wildcard"),
            msg="wildcard match full",
        )
        self.assertEqual(
            ["foo"],
            mdf.search(pattern="*oo", mode=SearchMode.wildcard),
            msg="wildcard match part",
        )
        self.assertEqual(
            ["baz"],
            mdf.search(pattern="b*", mode=SearchMode.wildcard, case_insensitive=False),
            msg="wildcard match case-sensitive",
        )
        self.assertEqual(
            ["Bar", "baz"],
            mdf.search(pattern="b*", mode=SearchMode.wildcard, case_insensitive=True),
            msg="wildcard match case-insensitive",
        )


if __name__ == "__main__":
    unittest.main()

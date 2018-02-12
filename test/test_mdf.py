#!/usr/bin/env python
from __future__ import print_function
import os
import random
import sys
import unittest
import shutil
import urllib
import numexpr
from itertools import product
from zipfile import ZipFile


import numpy as np

from utils import (
    CHANNELS_DEMO,
    CHANNELS_ARRAY,
    COMMENTS,
    MEMORY,
    UNITS,
    cycles,
    channels_count,
    generate_test_file,
    generate_arrays_test_file,
    cleanup_files,
)
from asammdf import MDF, SUPPORTED_VERSIONS

SUPPORTED_VERSIONS = SUPPORTED_VERSIONS[1:]

CHANNEL_LEN = 100000


class TestMDF(unittest.TestCase):

    def etest_measurement(self):
        self.assertTrue(MDF)

    @classmethod
    def setUpClass(cls):
        PYVERSION = sys.version_info[0]

        url = 'https://github.com/danielhrisca/asammdf/files/1594267/test.demo.zip'
        if PYVERSION == 3:
            urllib.request.urlretrieve(url, 'test.zip')
        else:
            urllib.urlretrieve(url, 'test.zip')
        ZipFile(r'test.zip').extractall('tmpdir_demo')

        url = 'https://github.com/danielhrisca/asammdf/files/1592123/test.arrays.zip'
        if PYVERSION == 3:
            urllib.request.urlretrieve(url, 'test.zip')
        else:
            urllib.urlretrieve(url, 'test.zip')
        ZipFile(r'test.zip').extractall('tmpdir_array')

        if not os.path.exists('tmpdir_big'):
            os.mkdir('tmpdir_big')
        if not os.path.exists('tmpdir_array_big'):
            os.mkdir('tmpdir_array_big')
        for version in ('3.30', '4.10'):
            generate_test_file(version)

        generate_arrays_test_file()

    @classmethod
    def etearDownClass(cls):
        shutil.rmtree('tmpdir_demo', True)
        shutil.rmtree('tmpdir_array', True)
        shutil.rmtree('tmpdir_big', True)
        shutil.rmtree('tmpdir_array_big', True)
        os.remove('test.zip')
        cleanup_files()

    def etest_read(self):

        print("MDF read tests")

        ret = True

        for enable in (True, False):
            for mdf in os.listdir('tmpdir_demo'):
                for memory in MEMORY:
                    with MDF(os.path.join('tmpdir_demo', mdf), memory=memory) as input_file:
                        if input_file.version == '2.00':
                            continue
                        for name in set(input_file.channels_db) - {'time', 't'}:
                            signal = input_file.get(name)
                            original_samples = CHANNELS_DEMO[name]
                            if signal.samples.dtype.kind == 'f':
                                signal = signal.astype(np.float32)
                            res = np.array_equal(signal.samples, original_samples)
                            if not res:
                                ret = False

            self.assertTrue(ret)
        cleanup_files()

    def etest_get_channel_comment_v4(self):
        print("MDF get channel comment tests")

        ret = True

        for mdf in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                with MDF(os.path.join('tmpdir_demo', mdf), memory=memory) as input_file:
                    if input_file.version < '4.00':
                        continue
                    print(mdf, memory)
                    for channel_name, original_comment in COMMENTS.items():
                        comment = input_file.get_channel_comment(channel_name)
                        if comment != original_comment:
                            ret = False

        self.assertTrue(ret)
        cleanup_files()

    def etest_get_channel_units(self):
        print("MDF get channel units tests")

        ret = True

        for mdf in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                with MDF(os.path.join('tmpdir_demo', mdf), memory=memory) as input_file:
                    if input_file.version == '2.00':
                        continue
                    print(mdf, memory)
                    for channel_name, original_unit in UNITS.items():
                        comment = input_file.get_channel_unit(channel_name)
                        if comment != original_unit:
                            ret = False

        self.assertTrue(ret)
        cleanup_files()


    def etest_read_array(self):

        print("MDF read array tests")

        ret = True

        for mdf in os.listdir('tmpdir_array'):
            for memory in MEMORY:
                with MDF(os.path.join('tmpdir_array', mdf), memory=memory) as input_file:
                    if input_file.version == '2.00':
                        continue
                    for name in set(input_file.channels_db) - {'time', 't'}:
                        signal = input_file.get(name)
                        original_samples = CHANNELS_ARRAY[name]
                        res = np.array_equal(signal.samples, original_samples)
                        if not res:
                            ret = False

        self.assertTrue(ret)
        cleanup_files()

    def etest_convert(self):
        print("MDF convert tests")

        for out in SUPPORTED_VERSIONS:
            for mdfname in os.listdir('tmpdir_demo'):
                for memory in MEMORY:
                    input_file = os.path.join('tmpdir_demo', mdfname)
                    if MDF(input_file).version == '2.00':
                        continue
                    print(input_file, memory, out)
                    with MDF(input_file, memory=memory) as mdf:
                        outfile = mdf.convert(out, memory=memory).save('tmp',
                                                             overwrite=True)

                    equal = True

                    with MDF(input_file, memory=memory) as mdf, \
                            MDF(outfile, memory=memory) as mdf2:

                        for name in set(mdf.channels_db) - {'t', 'time'}:
                            original = mdf.get(name)
                            converted = mdf2.get(name)
                            if not np.array_equal(
                                    original.samples,
                                    converted.samples):
                                equal = False
                            if not np.array_equal(
                                    original.timestamps,
                                    converted.timestamps):
                                equal = False

                    self.assertTrue(equal)
        cleanup_files()

    def etest_read_big(self):
        print("MDF read big files")
        for mdfname in os.listdir('tmpdir_big'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_big', mdfname)
                print(input_file, memory)

                equal = True

                with MDF(input_file, memory=memory) as mdf:

                    for i, group in enumerate(mdf.groups):
                        if i == 0:
                            v = np.ones(cycles, dtype=np.uint64)
                            for j in range(10, 200, 50):
                                vals = mdf.get(group=i, index=j, samples_only=True)
                                if not np.array_equal(
                                        vals,
                                        v * (j - 1)):
                                    equal = False
                        elif i == 1:
                            v = np.ones(cycles, dtype=np.int64)
                            for j in range(10, 200, 50):
                                vals = mdf.get(group=i, index=j, samples_only=True)
                                if not np.array_equal(
                                        vals,
                                        v * (j - 1) - 0.5):
                                    equal = False
                        elif i == 2:
                            v = np.arange(cycles, dtype=np.int64) / 100.0
                            form = '{} * sin(v)'
                            for j in range(10, 200, 50):
                                vals = mdf.get(group=i, index=j, samples_only=True)
                                f = form.format(j-1)
                                if not np.array_equal(
                                        vals,
                                        numexpr.evaluate(f)):
                                    equal = False
                        elif i == 3:
                            v = np.ones(cycles, dtype=np.int64)
                            form = '({} * v -0.5) / 1'
                            for j in range(10, 200, 50):
                                f = form.format(j - 1)
                                vals = mdf.get(group=i, index=j, samples_only=True)
                                if not np.array_equal(
                                        vals,
                                        numexpr.evaluate(f)):
                                    equal = False
                        else:

                            for j in range(10, 200, 50):
                                target = np.array([
                                    'Channel {} sample {}'.format(j, k).encode('ascii')
                                    for k in range(cycles)
                                ])
                                vals = mdf.get(group=i, index=j + 1, samples_only=True)
                                if not np.array_equal(
                                        vals,
                                        target):
                                    equal = False
                self.assertTrue(equal)
        cleanup_files()

    def etest_read_big_arrays(self):
        print("MDF read big array files")
        for mdfname in os.listdir('tmpdir_array_big'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_array_big', mdfname)
                print(input_file, memory)

                equal = True

                with MDF(input_file, memory=memory) as mdf:
                    mdf.configure(read_fragment_size=500000)

                    for i, group in enumerate(mdf.groups):

                        if i == 0:

                            samples = [
                                np.ones((cycles, 2, 3), dtype=np.uint64),
                                np.ones((cycles, 2), dtype=np.uint64),
                                np.ones((cycles, 3), dtype=np.uint64),
                            ]

                            for j in range(10, 200, 50):

                                types = [
                                    ('Channel_{}'.format(j), '(2, 3)<u8'),
                                    ('channel_{}_axis_1'.format(j), '(2, )<u8'),
                                    ('channel_{}_axis_2'.format(j), '(3, )<u8'),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get('Channel_{}'.format(j), group=i, samples_only=True)
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(
                                    target,
                                    dtype=types,
                                )
                                if not np.array_equal(
                                        vals,
                                        target):
                                    equal = False

                        elif i == 1:

                            samples = np.ones((cycles, 2, 3), dtype=np.uint64)
                            axis_0 = np.ones((cycles, 2), dtype=np.uint64)
                            axis_1 = np.ones((cycles, 3), dtype=np.uint64)

                            for j in range(10, 200, 50):

                                types = [
                                    ('Channel_{}'.format(j), '(2, 3)<u8'),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get('Channel_{}'.format(j), group=i, samples_only=True)
                                target = [
                                    samples * j,
                                ]
                                target = np.core.records.fromarrays(
                                    target,
                                    dtype=types,
                                )
                                if not np.array_equal(
                                        vals,
                                        target):
                                    equal = False

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

                            for j in range(10, 200, 50):

                                types = [
                                    ('struct_{}_channel_0'.format(j), np.uint8),
                                    ('struct_{}_channel_1'.format(j), np.uint16),
                                    ('struct_{}_channel_2'.format(j), np.uint32),
                                    ('struct_{}_channel_3'.format(j), np.uint64),
                                    ('struct_{}_channel_4'.format(j), np.int8),
                                    ('struct_{}_channel_5'.format(j), np.int16),
                                    ('struct_{}_channel_6'.format(j), np.int32),
                                    ('struct_{}_channel_7'.format(j), np.int64),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get('Channel_{}'.format(j), group=i, samples_only=True)
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(
                                    target,
                                    dtype=types,
                                )
                                if not np.array_equal(
                                        vals,
                                        target):
                                    equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_cut_big_arrays(self):
        print("MDF cut big array files")
        for mdfname in os.listdir('tmpdir_array_big'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_array_big', mdfname)
                print(input_file, memory)

                outfile1 = MDF(input_file, memory=memory)
                outfile1.configure(read_fragment_size=500000)
                outfile1 = outfile1.cut(stop=1005).save('tmp1', overwrite=True)
                outfile2 = MDF(input_file, memory=memory)
                outfile2.configure(read_fragment_size=500000)
                outfile2 = outfile2.cut(start=1005, stop=2010).save('tmp2', overwrite=True)
                outfile3 = MDF(input_file, memory=memory)
                outfile3.configure(read_fragment_size=500000)
                outfile3 = outfile3.cut(start=2010).save('tmp3', overwrite=True)

                outfile = MDF.merge(
                    [outfile1, outfile2, outfile3],
                    MDF(input_file, memory='minimum').version,
                ).save('tmp_cut_big', overwrite=True)

                equal = True

                with MDF(input_file, memory=memory) as mdf:
                    mdf.configure(read_fragment_size=500000)

                    for i, group in enumerate(mdf.groups):

                        if i == 0:

                            samples = [
                                np.ones((cycles, 2, 3), dtype=np.uint64),
                                np.ones((cycles, 2), dtype=np.uint64),
                                np.ones((cycles, 3), dtype=np.uint64),
                            ]

                            for j in range(10, 200, 50):

                                types = [
                                    ('Channel_{}'.format(j), '(2, 3)<u8'),
                                    ('channel_{}_axis_1'.format(j), '(2, )<u8'),
                                    ('channel_{}_axis_2'.format(j), '(3, )<u8'),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get('Channel_{}'.format(j), group=i, samples_only=True)
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(
                                    target,
                                    dtype=types,
                                )
                                if not np.array_equal(
                                        vals,
                                        target):
                                    equal = False

                        elif i == 1:

                            samples = np.ones((cycles, 2, 3), dtype=np.uint64)
                            axis_0 = np.ones((cycles, 2), dtype=np.uint64)
                            axis_1 = np.ones((cycles, 3), dtype=np.uint64)

                            for j in range(10, 200, 50):

                                types = [
                                    ('Channel_{}'.format(j), '(2, 3)<u8'),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get('Channel_{}'.format(j), group=i, samples_only=True)
                                target = [
                                    samples * j,
                                ]
                                target = np.core.records.fromarrays(
                                    target,
                                    dtype=types,
                                )
                                if not np.array_equal(
                                        vals,
                                        target):
                                    equal = False

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

                            for j in range(10, 200, 50):

                                types = [
                                    ('struct_{}_channel_0'.format(j), np.uint8),
                                    ('struct_{}_channel_1'.format(j), np.uint16),
                                    ('struct_{}_channel_2'.format(j), np.uint32),
                                    ('struct_{}_channel_3'.format(j), np.uint64),
                                    ('struct_{}_channel_4'.format(j), np.int8),
                                    ('struct_{}_channel_5'.format(j), np.int16),
                                    ('struct_{}_channel_6'.format(j), np.int32),
                                    ('struct_{}_channel_7'.format(j), np.int64),
                                ]
                                types = np.dtype(types)

                                vals = mdf.get('Channel_{}'.format(j), group=i, samples_only=True)
                                target = [arr * j for arr in samples]
                                target = np.core.records.fromarrays(
                                    target,
                                    dtype=types,
                                )
                                if not np.array_equal(
                                        vals,
                                        target):
                                    equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_convert_big(self):
        print("MDF convert big files tests")

        t = np.arange(cycles, dtype=np.float64)

        for out in ('3.30', '4.10'):
            for mdfname in os.listdir('tmpdir_big')[::-1]:
                for memory in MEMORY:
                    input_file = os.path.join('tmpdir_big', mdfname)
                    print(input_file, memory, out)
                    with MDF(input_file, memory=memory) as mdf:
                        mdf.configure(read_fragment_size=500000)
                        outfile = mdf.convert(out, memory=memory).save(
                            'tmp_convert_{}_{}'.format(out, memory),
                            overwrite=True,
                        )

                    equal = True

                    with MDF(outfile) as mdf:
                        mdf.configure(read_fragment_size=500000)

                        for i, group in enumerate(mdf.groups):
                            if i == 0:
                                v = np.ones(cycles, dtype=np.uint64)
                                for j in range(10, 200, 50):
                                    vals = mdf.get(group=i, index=j, samples_only=True)
                                    if not np.array_equal(
                                            vals,
                                            v * (j - 1)):
                                        equal = False
                            elif i == 1:
                                v = np.ones(cycles, dtype=np.int64)
                                for j in range(10, 200, 50):
                                    vals = mdf.get(group=i, index=j, samples_only=True)
                                    if not np.array_equal(
                                            vals,
                                            v * (j - 1) - 0.5):
                                        equal = False
                            elif i == 2:
                                v = np.arange(cycles, dtype=np.int64) / 100.0
                                form = '{} * sin(v)'
                                for j in range(10, 200, 50):
                                    vals = mdf.get(group=i, index=j, samples_only=True)
                                    f = form.format(j - 1)
                                    if not np.array_equal(
                                            vals,
                                            numexpr.evaluate(f)):
                                        equal = False
                            elif i == 3:
                                v = np.ones(cycles, dtype=np.int64)
                                form = '({} * v -0.5) / 1'
                                for j in range(10, 200, 50):
                                    f = form.format(j - 1)
                                    vals = mdf.get(group=i, index=j, samples_only=True)
                                    if not np.array_equal(
                                            vals,
                                            numexpr.evaluate(f)):
                                        equal = False
                            else:

                                for j in range(10, 200, 50):
                                    target = np.array([
                                        'Channel {} sample {}'.format(j, k).encode('ascii')
                                        for k in range(cycles)
                                    ])
                                    vals = mdf.get(group=i, index=j+1, samples_only=True)
                                    if not np.array_equal(
                                            vals,
                                            target):
                                        equal = False

                    self.assertTrue(equal)
        cleanup_files()

    def test_cut_big(self):
        print("MDF cut big files tests")

        t = np.arange(cycles, dtype=np.float64)

        for mdfname in os.listdir('tmpdir_big'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_big', mdfname)
                print(input_file, memory)

                outfile1 = MDF(input_file, memory=memory)
                outfile1.configure(read_fragment_size=500000)
                outfile1 = outfile1.cut(stop=1005).save('tmp1', overwrite=True)
                outfile2 = MDF(input_file, memory=memory)
                outfile2.configure(read_fragment_size=500000)
                outfile2 = outfile2.cut(start=1005.1, stop=2010).save('tmp2', overwrite=True)
                outfile3 = MDF(input_file, memory=memory)
                outfile3.configure(read_fragment_size=500000)
                outfile3 = outfile3.cut(start=2010.1).save('tmp3', overwrite=True)
                outfile4 = MDF(input_file, memory=memory)
                outfile4.configure(read_fragment_size=500000)
                outfile4 = outfile4.cut(start=7000).save('tmp4', overwrite=True)

                outfile = MDF.merge(
                    [outfile1, outfile2, outfile3, outfile4],
                    MDF(input_file, memory='minimum').version,
                ).save('tmp_cut_big', overwrite=True)

                equal = True

                with MDF(outfile) as mdf:

                    for i, group in enumerate(mdf.groups):
                        if i == 0:
                            v = np.ones(cycles, dtype=np.uint64)
                            for j in range(10, 200, 50):
                                vals = mdf.get(group=i, index=j, samples_only=True)
                                if not np.array_equal(
                                        vals,
                                        v * (j - 1)):
                                    equal = False
                        elif i == 1:
                            v = np.ones(cycles, dtype=np.int64)
                            for j in range(10, 200, 50):
                                vals = mdf.get(group=i, index=j, samples_only=True)
                                if not np.array_equal(
                                        vals,
                                        v * (j - 1) - 0.5):
                                    equal = False
                        elif i == 2:
                            v = np.arange(cycles, dtype=np.int64) / 100.0
                            form = '{} * sin(v)'
                            for j in range(10, 200, 50):
                                vals = mdf.get(group=i, index=j, samples_only=True)
                                f = form.format(j-1)
                                if not np.array_equal(
                                        vals,
                                        numexpr.evaluate(f)):
                                    equal = False
                        elif i == 3:
                            v = np.ones(cycles, dtype=np.int64)
                            form = '({} * v -0.5) / 1'
                            for j in range(10, 200, 50):
                                f = form.format(j - 1)
                                vals = mdf.get(group=i, index=j, samples_only=True)
                                if not np.array_equal(
                                        vals,
                                        numexpr.evaluate(f)):
                                    equal = False
                        else:

                            for j in range(10, 200, 50):
                                target = np.array([
                                    'Channel {} sample {}'.format(j, k).encode('ascii')
                                    for k in range(cycles)
                                ])
                                vals = mdf.get(group=i, index=j + 1, samples_only=True)
                                if not np.array_equal(
                                        vals,
                                        target):
                                    equal = False
                self.assertTrue(equal)
        cleanup_files()

    def etest_merge(self):
        print("MDF merge tests")

        for out in SUPPORTED_VERSIONS:
            for mdfname in os.listdir('tmpdir_demo'):
                for memory in MEMORY:

                    input_file = os.path.join('tmpdir_demo', mdfname)
                    if '2.00' in input_file:
                        continue
                    files = [input_file, ] * 4

                    outfile = MDF.merge(files, out, memory).save('tmp', overwrite=True)

                    equal = True

                    with MDF(input_file, memory=memory) as mdf, \
                            MDF(outfile, memory=memory) as mdf2:

                        for i, group in enumerate(mdf.groups):
                            for j, _ in enumerate(group['channels'][1:], 1):
                                original = mdf.get(group=i, index=j)
                                converted = mdf2.get(group=i, index=j)
                                if not np.array_equal(
                                        np.tile(original.samples, 4),
                                        converted.samples):
                                    equal = False

                    self.assertTrue(equal)
        cleanup_files()

    def etest_merge_array(self):
        print("MDF merge array tests")

        for out in (version for version in SUPPORTED_VERSIONS if version >= '4.00'):
            for mdfname in os.listdir('tmpdir_array'):
                for memory in MEMORY:
                    input_file = os.path.join('tmpdir_array', mdfname)
                    files = [input_file, ] * 4

                    outfile = MDF.merge(files, out, memory).save('tmp', overwrite=True)

                    equal = True

                    with MDF(input_file, memory=memory) as mdf, \
                            MDF(outfile, memory=memory) as mdf2:

                        for i, group in enumerate(mdf.groups):
                            for j, _ in enumerate(group['channels'][1:], 1):
                                original = mdf.get(group=i, index=j)
                                converted = mdf2.get(group=i, index=j)
                                if not np.array_equal(
                                        np.tile(original.samples, 4),
                                        converted.samples):
                                    equal = False

                    self.assertTrue(equal)
        cleanup_files()

    def etest_cut_absolute(self):
        print("MDF cut absolute tests")

        for mdfname in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_demo', mdfname)

                if '2.00' in input_file:
                    continue
                print(input_file, memory)

                outfile1 = MDF(input_file, memory=memory).cut(stop=2).save('tmp1', overwrite=True)
                outfile2 = MDF(input_file, memory=memory).cut(start=2, stop=6).save('tmp2', overwrite=True)
                outfile3 = MDF(input_file, memory=memory).cut(start=6).save('tmp3', overwrite=True)

                outfile = MDF.merge(
                    [outfile1, outfile2, outfile3],
                    MDF(input_file, memory='minimum').version,
                ).save('tmp', overwrite=True)

                print('OUT', outfile)

                equal = True

                with MDF(input_file, memory=memory) as mdf, \
                        MDF(outfile, memory=memory) as mdf2:

                    for i, group in enumerate(mdf.groups):
                        for j, _ in enumerate(group['channels'][1:], 1):
                            original = mdf.get(group=i, index=j)
                            converted = mdf2.get(group=i, index=j)
                            if not np.array_equal(
                                    original.samples,
                                    converted.samples):
                                equal = False
                            if not np.array_equal(
                                    original.timestamps,
                                    converted.timestamps):
                                equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_cut_absolute_array(self):
        print("MDF cut absolute array tests")

        for mdfname in os.listdir('tmpdir_array'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_array', mdfname)

                outfile1 = MDF(input_file, memory=memory).cut(stop=2.1).save('tmp1', overwrite=True)
                outfile2 = MDF(input_file, memory=memory).cut(start=2.1, stop=6.1).save('tmp2', overwrite=True)
                outfile3 = MDF(input_file, memory=memory).cut(start=6.1).save('tmp3', overwrite=True)

                outfile = MDF.merge(
                    [outfile1, outfile2, outfile3],
                    MDF(input_file, memory='minimum').version,
                ).save('tmp', overwrite=True)

                equal = True

                with MDF(input_file, memory=memory) as mdf, \
                        MDF(outfile, memory=memory) as mdf2:

                    for i, group in enumerate(mdf.groups):
                        for j, _ in enumerate(group['channels'][1:], 1):
                            original = mdf.get(group=i, index=j)
                            converted = mdf2.get(group=i, index=j)
                            if not np.array_equal(
                                    original.samples,
                                    converted.samples):
                                equal = False
                            if not np.array_equal(
                                    original.timestamps,
                                    converted.timestamps):
                                equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_cut_relative(self):
        print("MDF cut relative tests")

        for mdfname in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_demo', mdfname)
                if '2.00' in input_file:
                    continue

                outfile1 = MDF(input_file, memory=memory).cut(stop=3, whence=1).save('tmp1', overwrite=True)
                outfile2 = MDF(input_file, memory=memory).cut(start=3, stop=5, whence=1).save('tmp2', overwrite=True)
                outfile3 = MDF(input_file, memory=memory).cut(start=5, whence=1).save('tmp3', overwrite=True)

                outfile = MDF.merge(
                    [outfile1, outfile2, outfile3],
                    MDF(input_file, memory='minimum').version,
                ).save('tmp', overwrite=True)

                equal = True

                with MDF(input_file, memory=memory) as mdf, \
                        MDF(outfile, memory=memory) as mdf2:

                    for i, group in enumerate(mdf.groups):
                        for j, _ in enumerate(group['channels'][1:], 1):
                            original = mdf.get(group=i, index=j)
                            converted = mdf2.get(group=i, index=j)
                            if not np.array_equal(
                                    original.samples,
                                    converted.samples):
                                equal = False
                            if not np.array_equal(
                                    original.timestamps,
                                    converted.timestamps):
                                equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_cut_relative_array(self):
        print("MDF cut relative array tests")

        for mdfname in os.listdir('tmpdir_array'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_array', mdfname)

                outfile1 = MDF(input_file, memory=memory).cut(stop=3.1, whence=1).save('tmp1', overwrite=True)
                outfile2 = MDF(input_file, memory=memory).cut(start=3.1, stop=5.1, whence=1).save('tmp2', overwrite=True)
                outfile3 = MDF(input_file, memory=memory).cut(start=5.1, whence=1).save('tmp3', overwrite=True)

                outfile = MDF.merge(
                    [outfile1, outfile2, outfile3],
                    MDF(input_file, memory='minimum').version,
                ).save('tmp', overwrite=True)

                equal = True

                with MDF(input_file, memory=memory) as mdf, \
                        MDF(outfile, memory=memory) as mdf2:

                    for i, group in enumerate(mdf.groups):
                        for j, _ in enumerate(group['channels'][1:], 1):
                            original = mdf.get(group=i, index=j)
                            converted = mdf2.get(group=i, index=j)
                            if not np.array_equal(
                                    original.samples,
                                    converted.samples):
                                equal = False
                            if not np.array_equal(
                                    original.timestamps,
                                    converted.timestamps):
                                equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_filter(self):
        print("MDF filter tests")

        for mdfname in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_demo', mdfname)

                if MDF(input_file, memory=memory).version <= '2.00':
                # if MDF(input_file, memory=memory).version < '4.00':
                    continue

                channels_nr = np.random.randint(1, len(CHANNELS_DEMO) + 1)

                channel_list = random.sample(list(CHANNELS_DEMO), channels_nr)

                filtered_mdf = MDF(input_file, memory=memory).filter(channel_list, memory=memory)

                self.assertTrue((set(filtered_mdf.channels_db) - {'t', 'time'}) == set(channel_list))

                equal = True

                with MDF(input_file, memory=memory) as mdf:
                    print(input_file, memory)

                    for name in channel_list:
                        original = mdf.get(name)
                        filtered = filtered_mdf.get(name)
                        if not np.array_equal(
                                original.samples,
                                filtered.samples):
                            equal = False
                        if not np.array_equal(
                                original.timestamps,
                                filtered.timestamps):
                            equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_filter_array(self):
        print("MDF filter array tests")

        for mdfname in os.listdir('tmpdir_array'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_array', mdfname)

                channels_nr = np.random.randint(1, len(CHANNELS_ARRAY) + 1)

                channel_list = random.sample(list(CHANNELS_ARRAY), channels_nr)

                filtered_mdf = MDF(input_file, memory=memory).filter(channel_list, memory=memory)

                target = set(channel_list)
                if 'Int16Array' in target:
                    target = target - {'XAxis', 'YAxis'}
                if 'Maths' in target:
                    target = target - {'Saw', 'Ones', 'Cos', 'Sin', 'Zeros'}
                if 'Composed' in target:
                    target = target - {'Int32', 'Float64', 'Uint8', 'Uint64'}

                actual = set(filtered_mdf.channels_db) - {'t', 'time'}

                if 'Int16Array' in actual:
                    actual = actual - {'XAxis', 'YAxis'}
                if 'Maths' in actual:
                    actual = actual - {'Saw', 'Ones', 'Cos', 'Sin', 'Zeros'}
                if 'Composed' in actual:
                    actual = actual - {'Int32', 'Float64', 'Uint8', 'Uint64'}

                self.assertTrue(actual == target)

                equal = True

                with MDF(input_file, memory=memory) as mdf:

                    for name in channel_list:
                        original = mdf.get(name)
                        filtered = filtered_mdf.get(name)
                        if not np.array_equal(
                                original.samples,
                                filtered.samples):
                            equal = False
                        if not np.array_equal(
                                original.timestamps,
                                filtered.timestamps):
                            equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_save(self):
        print("MDF save tests")

        compressions = [0, 1, 2]
        overwrite_enables = [True, False]
        for compression, memory, overwrite in product(compressions, MEMORY, overwrite_enables):

            for mdfname in os.listdir('tmpdir_demo'):
                input_file = os.path.join('tmpdir_demo', mdfname)
                if MDF(input_file).version == '2.00':
                # if MDF(input_file).version < '4.10':
                    continue
                print(input_file, compression, memory,overwrite)
                with MDF(input_file, memory=memory) as mdf:
                    out_file = mdf.save('tmp', compression=compression, overwrite=overwrite)
                    print(out_file)

                equal = True

                with MDF(input_file, memory=memory) as mdf, \
                        MDF(out_file, memory=memory) as mdf2:

                    for name in set(mdf.channels_db) - {'t', 'time'}:
                        original = mdf.get(name)
                        converted = mdf2.get(name)
                        if not np.array_equal(
                                original.samples,
                                converted.samples):
                            equal = False

                            os.rename(out_file, out_file+'err')
                            raise ValueError(1)
                        if not np.array_equal(
                                original.timestamps,
                                converted.timestamps):
                            equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_save_array(self):
        print("MDF save array tests")

        compressions = [0, 1, 2]
        overwrite_enables = [True, False]
        for compression, memory, overwrite in product(compressions, MEMORY, overwrite_enables):

            for mdfname in os.listdir('tmpdir_array'):
                input_file = os.path.join('tmpdir_array', mdfname)
                print(input_file, compression, memory, overwrite)
                with MDF(input_file, memory=memory) as mdf:
                    out_file = mdf.save('tmp', compression=compression)

                    mdf.save('tmp_array_{}_{}.mf4'.format(compression, memory), compression=compression, overwrite=overwrite)

                equal = True

                with MDF(input_file, memory=memory) as mdf, \
                        MDF(out_file, memory=memory) as mdf2:

                    for name in set(mdf.channels_db) - {'t', 'time'}:
                        original = mdf.get(name)
                        converted = mdf2.get(name)
                        if not np.array_equal(
                                original.samples,
                                converted.samples):
                            equal = False
                        if not np.array_equal(
                                original.timestamps,
                                converted.timestamps):
                            equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_select(self):
        print("MDF select tests")

        for mdfname in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_demo', mdfname)

                if MDF(input_file).version == '2.00':
                    continue

                print(input_file, memory)

                channels_nr = np.random.randint(1, len(CHANNELS_DEMO) + 1)

                channel_list = random.sample(list(CHANNELS_DEMO), channels_nr)

                selected_signals = MDF(input_file, memory=memory).select(channel_list)

                self.assertTrue(len(selected_signals) == len(channel_list))

                self.assertTrue(all(ch.name == name for ch, name in zip(selected_signals, channel_list)))

                equal = True

                with MDF(input_file, memory=memory) as mdf:

                    for selected in selected_signals:
                        original = mdf.get(selected.name)
                        if not np.array_equal(
                                original.samples,
                                selected.samples):
                            equal = False
                        if not np.array_equal(
                                original.timestamps,
                                selected.timestamps):
                            equal = False

                self.assertTrue(equal)
        cleanup_files()

    def etest_select_array(self):
        print("MDF select array tests")

        for mdfname in os.listdir('tmpdir_array'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_array', mdfname)

                channels_nr = np.random.randint(1, len(CHANNELS_ARRAY) + 1)

                channel_list = random.sample(list(CHANNELS_ARRAY), channels_nr)

                selected_signals = MDF(input_file, memory=memory).select(channel_list)

                self.assertTrue(len(selected_signals) == len(channel_list))

                self.assertTrue(all(ch.name == name for ch, name in zip(selected_signals, channel_list)))

                equal = True

                with MDF(input_file, memory=memory) as mdf:

                    for selected in selected_signals:
                        original = mdf.get(selected.name)
                        if not np.array_equal(
                                original.samples,
                                selected.samples):
                            equal = False
                        if not np.array_equal(
                                original.timestamps,
                                selected.timestamps):
                            equal = False

                self.assertTrue(equal)
        cleanup_files()


if __name__ == '__main__':
    unittest.main()

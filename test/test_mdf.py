#!/usr/bin/env python
from __future__ import print_function
import os
import random
import sys
import unittest
import shutil
import urllib
from pprint import pprint
from zipfile import ZipFile


import numpy as np

from numpy import (
    array,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

from utils import (
    CHANNELS_DEMO,
    CHANNELS_ARRAY,
    MEMORY,
)
from asammdf import MDF, SUPPORTED_VERSIONS, configure

CHANNEL_LEN = 100000


class TestMDF(unittest.TestCase):

    def test_measurement(self):
        self.assertTrue(MDF)

    @classmethod
    def setUpClass(cls):
        PYVERSION = sys.version_info[0]

        url = 'https://github.com/danielhrisca/asammdf/files/1593570/test.demo.zip'
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

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmpdir_demo', True)
        shutil.rmtree('tmpdir_array', True)
        os.remove('test.zip')
        for filename in ('tmp', 'tmp1', 'tmp2'):
            if os.path.isfile(filename):
                os.remove(filename)

    def test_read(self):

        print("MDF read tests")

        ret = True

        for mdf in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                with MDF(os.path.join('tmpdir_demo', mdf), memory=memory) as input_file:
                    if input_file.version == '2.00':
                        continue
                    for name in set(input_file.channels_db) - {'time', 't'}:
                        signal = input_file.get(name)
                        original_samples = CHANNELS_DEMO[name]
                        if signal.samples.dtype.kind == 'f':
                            signal = signal.astype(float32)
                        res = np.array_equal(signal.samples, original_samples)
                        if not res:
                            ret = False

        self.assertTrue(ret)

    def test_read_array(self):

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

    def test_convert(self):
        print("MDF convert tests")

        for out in SUPPORTED_VERSIONS[1:]:
            for mdfname in os.listdir('tmpdir_demo'):
                for memory in MEMORY:
                    input_file = os.path.join('tmpdir_demo', mdfname)
                    if MDF(input_file).version == '2.00':
                        continue
                    with MDF(input_file, memory=memory) as mdf:
                        mdf.convert(out, memory=memory).save('tmp',
                                                             overwrite=True)

                    equal = True

                    with MDF(input_file, memory=memory) as mdf, \
                            MDF('tmp', memory=memory) as mdf2:

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

    def test_merge(self):
        print("MDF merge tests")

        for out in SUPPORTED_VERSIONS:
            for mdfname in os.listdir('tmpdir_demo'):
                for memory in MEMORY:
                    input_file = os.path.join('tmpdir_demo', mdfname)
                    files = [input_file, ] * 4

                    MDF.merge(files, out, memory).save('tmp', overwrite=True)

                    equal = True

                    with MDF(input_file, memory=memory) as mdf, \
                            MDF('tmp', memory=memory) as mdf2:

                        for i, group in enumerate(mdf.groups):
                            for j, channel in enumerate(group['channels'][1:], 1):
                                original = mdf.get(group=i, index=j)
                                converted = mdf2.get(group=i, index=j)
                                if not np.array_equal(
                                        np.tile(original.samples, 4),
                                        converted.samples):
                                    equal = False

                    self.assertTrue(equal)

    def test_merge_array(self):
        print("MDF merge array tests")

        for out in (version for version in SUPPORTED_VERSIONS if version >= '4.00'):
            for mdfname in os.listdir('tmpdir_array'):
                for memory in MEMORY:
                    input_file = os.path.join('tmpdir_array', mdfname)
                    files = [input_file, ] * 4

                    MDF.merge(files, out, memory).save('tmp', overwrite=True)

                    equal = True

                    with MDF(input_file, memory=memory) as mdf, \
                            MDF('tmp', memory=memory) as mdf2:

                        for i, group in enumerate(mdf.groups):
                            for j, channel in enumerate(group['channels'][1:], 1):
                                original = mdf.get(group=i, index=j)
                                converted = mdf2.get(group=i, index=j)
                                if not np.array_equal(
                                        np.tile(original.samples, 4),
                                        converted.samples):
                                    equal = False

                    self.assertTrue(equal)

    def test_cut_absolute(self):
        print("MDF cut absolute tests")

        for mdfname in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_demo', mdfname)

                MDF(input_file, memory=memory).cut(stop=2).save('tmp1', overwrite=True)
                MDF(input_file, memory=memory).cut(start=2, stop=6).save('tmp2', overwrite=True)
                MDF(input_file, memory=memory).cut(start=6).save('tmp3', overwrite=True)

                MDF.merge(
                    ['tmp1', 'tmp2', 'tmp3'],
                    MDF(input_file, memory='minimum').version,
                ).save('tmp', overwrite=True)

                equal = True

                with MDF(input_file, memory=memory) as mdf, \
                        MDF('tmp', memory=memory) as mdf2:

                    for i, group in enumerate(mdf.groups):
                        for j, channel in enumerate(group['channels'][1:], 1):
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


    def test_cut_absolute_array(self):
        print("MDF cut absolute array tests")

        for mdfname in os.listdir('tmpdir_array'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_array', mdfname)

                MDF(input_file, memory=memory).cut(stop=2.1).save('tmp1', overwrite=True)
                MDF(input_file, memory=memory).cut(start=2.1, stop=6.1).save('tmp2', overwrite=True)
                MDF(input_file, memory=memory).cut(start=6.1).save('tmp3', overwrite=True)

                MDF.merge(
                    ['tmp1', 'tmp2', 'tmp3'],
                    MDF(input_file, memory='minimum').version,
                ).save('tmp', overwrite=True)

                equal = True

                with MDF(input_file, memory=memory) as mdf, \
                        MDF('tmp', memory=memory) as mdf2:

                    for i, group in enumerate(mdf.groups):
                        for j, channel in enumerate(group['channels'][1:], 1):
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

    def test_cut_relative(self):
        print("MDF cut relative tests")

        for mdfname in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_demo', mdfname)

                MDF(input_file, memory=memory).cut(stop=3, whence=1).save('tmp1', overwrite=True)
                MDF(input_file, memory=memory).cut(start=3, stop=5, whence=1).save('tmp2', overwrite=True)
                MDF(input_file, memory=memory).cut(start=5, whence=1).save('tmp3', overwrite=True)

                MDF.merge(
                    ['tmp1', 'tmp2', 'tmp3'],
                    MDF(input_file, memory='minimum').version,
                ).save('tmp', overwrite=True)

                equal = True

                with MDF(input_file, memory=memory) as mdf, \
                        MDF('tmp', memory=memory) as mdf2:

                    for i, group in enumerate(mdf.groups):
                        for j, channel in enumerate(group['channels'][1:], 1):
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

    def test_cut_relative_array(self):
        print("MDF cut relative array tests")

        for mdfname in os.listdir('tmpdir_array'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_array', mdfname)

                MDF(input_file, memory=memory).cut(stop=3.1, whence=1).save('tmp1', overwrite=True)
                MDF(input_file, memory=memory).cut(start=3.1, stop=5.1, whence=1).save('tmp2', overwrite=True)
                MDF(input_file, memory=memory).cut(start=5.1, whence=1).save('tmp3', overwrite=True)

                MDF.merge(
                    ['tmp1', 'tmp2', 'tmp3'],
                    MDF(input_file, memory='minimum').version,
                ).save('tmp', overwrite=True)

                equal = True

                with MDF(input_file, memory=memory) as mdf, \
                        MDF('tmp', memory=memory) as mdf2:

                    for i, group in enumerate(mdf.groups):
                        for j, channel in enumerate(group['channels'][1:], 1):
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

    def test_filter(self):
        print("MDF filter tests")

        for mdfname in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_demo', mdfname)

                if MDF(input_file, memory=memory).version == '2.00':
                    continue

                channels_nr = np.random.randint(1, len(CHANNELS_DEMO) + 1)

                channel_list = random.sample(list(CHANNELS_DEMO), channels_nr)

                filtered_mdf = MDF(input_file, memory=memory).filter(channel_list, memory=memory)

                self.assertTrue((set(filtered_mdf.channels_db) - {'t', 'time'}) == set(channel_list))

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

    def test_filter_array(self):
        print("MDF filter array tests")

        for mdfname in os.listdir('tmpdir_array'):
            for memory in MEMORY[:1]:
                input_file = os.path.join('tmpdir_array', mdfname)

                channels_nr = np.random.randint(1, len(CHANNELS_ARRAY) + 1)

                channel_list = random.sample(list(CHANNELS_ARRAY), channels_nr)

                filtered_mdf = MDF(input_file, memory=memory).filter(channel_list, memory=memory)

                filtered_mdf.save('fiteed.mf4', overwrite=True)

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

    def test_select(self):
        print("MDF select tests")

        for mdfname in os.listdir('tmpdir_demo'):
            for memory in MEMORY:
                input_file = os.path.join('tmpdir_demo', mdfname)

                if MDF(input_file).version == '2.00':
                    continue

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

    def test_select_array(self):
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


if __name__ == '__main__':
    unittest.main()

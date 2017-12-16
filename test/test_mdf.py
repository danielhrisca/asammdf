#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import unittest
import shutil
import urllib
from zipfile import ZipFile

import numpy as np

from utils import MEMORY
from asammdf import MDF, SUPPORTED_VERSIONS

CHANNEL_LEN = 100000


class TestMDF(unittest.TestCase):

    def test_measurement(self):
        self.assertTrue(MDF)

    @classmethod
    def setUpClass(cls):

        url = 'https://github.com/danielhrisca/asammdf/files/1565090/test.files.zip'

        PYVERSION = sys.version_info[0]
        if PYVERSION == 3:
            urllib.request.urlretrieve(url, 'test.zip')
        else:
            urllib.urlretrieve(url, 'test.zip')
        ZipFile(r'test.zip').extractall('tmpdir')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmpdir', True)
        os.remove('test.zip')
        if os.path.isfile('tmp'):
            os.remove('tmp')

    def test_read(self):
        print("MDF read tests")

        for mdf in os.listdir('tmpdir'):
            for memory in MEMORY:
                MDF(os.path.join('tmpdir', mdf), memory=memory).close()

    def test_convert(self):
        print("MDF convert tests")

        for out in SUPPORTED_VERSIONS[1:]:
            for mdfname in os.listdir('tmpdir'):
                for memory in MEMORY:
                    input_file = os.path.join('tmpdir', mdfname)
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
            for mdfname in os.listdir('tmpdir'):
                for memory in MEMORY:
                    input_file = os.path.join('tmpdir', mdfname)
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
                                    print(input_file, i, j, np.tile(original.samples, 4), converted.samples)

                    self.assertTrue(equal)


if __name__ == '__main__':
    unittest.main()

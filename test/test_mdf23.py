#!/usr/bin/env python
from __future__ import print_function
import unittest

import numpy as np

from utils import MEMORY
from asammdf import MDF, MDF2, MDF3, Signal

CHANNEL_LEN = 10000


class TestMDF23(unittest.TestCase):

    def test_measurement(self):
        self.assertTrue(MDF2)
        self.assertTrue(MDF3)

    def test_read_mdf2_00(self):

        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print('Read 2.00 using seed =', seed)

        sig_int = Signal(
            np.random.randint(-2**15, -1, CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name='Integer Channel',
            unit='unit1',
        )

        sig_float = Signal(
            np.random.random(CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name='Float Channel',
            unit='unit2',
        )

        for memory in MEMORY:
            print(memory)

            with MDF(version='2.00', memory=memory) as mdf:
                mdf.append([sig_int, sig_float], common_timebase=True)
                outfile = mdf.save('tmp', overwrite=True)

            with MDF(outfile, memory=memory) as mdf:
                ret_sig_int = mdf.get(sig_int.name)
                ret_sig_float = mdf.get(sig_float.name)

            self.assertTrue(np.array_equal(ret_sig_int.samples,
                                           sig_int.samples))
            self.assertTrue(np.array_equal(ret_sig_float.samples,
                                           sig_float.samples))

    def test_read_mdf2_14(self):

        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print('Read 2.14 using seed =', seed)

        sig_int = Signal(
            np.random.randint(-2**29, 2**29, CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name='Integer Channel',
            unit='unit1',
        )

        sig_float = Signal(
            np.random.random(CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name='Float Channel',
            unit='unit2',
        )

        for memory in MEMORY:
            print(memory)
            with MDF(version='2.14', memory=memory) as mdf:
                mdf.append([sig_int, sig_float], common_timebase=True)
                outfile = mdf.save('tmp', overwrite=True)

            with MDF(outfile, memory=memory) as mdf:
                ret_sig_int = mdf.get(sig_int.name)
                ret_sig_float = mdf.get(sig_float.name)

            self.assertTrue(np.array_equal(ret_sig_int.samples,
                                           sig_int.samples))
            self.assertTrue(np.array_equal(ret_sig_float.samples,
                                           sig_float.samples))

    def test_read_mdf3_00(self):

        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print('Read 3.00 using seed =', seed)

        sig_int = Signal(
            np.random.randint(-2**16, 2**16, CHANNEL_LEN, np.int32),
            np.arange(CHANNEL_LEN),
            name='Integer Channel',
            unit='unit1',
        )

        sig_float = Signal(
            np.random.random(CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name='Float Channel',
            unit='unit2',
        )

        for memory in MEMORY:
            print(memory)

            with MDF(version='3.00', memory=memory) as mdf:
                mdf.append([sig_int, sig_float], common_timebase=True)
                outfile = mdf.save('tmp', overwrite=True)

            with MDF(outfile, memory=memory) as mdf:
                ret_sig_int = mdf.get(sig_int.name)
                ret_sig_float = mdf.get(sig_float.name)

            self.assertTrue(np.array_equal(ret_sig_int.samples,
                                           sig_int.samples))
            self.assertTrue(np.array_equal(ret_sig_float.samples,
                                           sig_float.samples))

    def test_read_mdf3_10(self):

        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print('Read 3.10 using seed =', seed)

        sig_int = Signal(
            np.random.randint(-2**9, 2**7, CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name='Integer Channel',
            unit='unit1',
        )

        sig_float = Signal(
            np.random.random(CHANNEL_LEN),
            np.arange(CHANNEL_LEN),
            name='Float Channel',
            unit='unit2',
        )

        for memory in MEMORY:
            with MDF(version='3.10', memory=memory) as mdf:
                mdf.append([sig_int, sig_float], common_timebase=True)
                outfile = mdf.save('tmp', overwrite=True)

            print(memory)

            with MDF(outfile, memory=memory) as mdf:
                ret_sig_int = mdf.get(sig_int.name)
                ret_sig_float = mdf.get(sig_float.name)

            self.assertTrue(np.array_equal(ret_sig_int.samples,
                                           sig_int.samples))
            self.assertTrue(np.array_equal(ret_sig_float.samples,
                                           sig_float.samples))


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
from __future__ import print_function
import unittest

import numpy as np

from utils import MEMORY
from asammdf import MDF, MDF4, Signal

CHANNEL_LEN = 100000


class TestMDF4(unittest.TestCase):

    def test_measurement(self):
        self.assertTrue(MDF4)

    def test_read_mdf4_00(self):

        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print('Read 4.00 using seed =', seed)

        sig_int = Signal(
            np.random.randint(-2**31, 2**31, CHANNEL_LEN),
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

            with MDF(version='4.00', memory=memory) as mdf:
                mdf.append([sig_int, sig_float], common_timebase=True)
                outfile = mdf.save('tmp', overwrite=True)

            with MDF(outfile, memory=memory) as mdf:
                ret_sig_int = mdf.get(sig_int.name)
                ret_sig_float = mdf.get(sig_float.name)

            self.assertTrue(np.array_equal(ret_sig_int.samples,
                                           sig_int.samples))
            self.assertTrue(np.array_equal(ret_sig_float.samples,
                                           sig_float.samples))

    def test_read_mdf4_10(self):

        seed = np.random.randint(0, 2**31)

        np.random.seed(seed)
        print('Read 4.10 using seed =', seed)

        sig_int = Signal(
            np.random.randint(-2**31, 2**31, CHANNEL_LEN),
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
            with MDF(version='4.10', memory=memory) as mdf:
                mdf.append([sig_int, sig_float], common_timebase=True)
                outfile = mdf.save('tmp', overwrite=True)

            with MDF(outfile, memory=memory) as mdf:
                ret_sig_int = mdf.get(sig_int.name)
                ret_sig_float = mdf.get(sig_float.name)

            self.assertTrue(np.array_equal(ret_sig_int.samples,
                                           sig_int.samples))
            self.assertTrue(np.array_equal(ret_sig_float.samples,
                                           sig_float.samples))


if __name__ == '__main__':
    unittest.main()

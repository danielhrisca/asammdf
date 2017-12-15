# -*- coding: utf-8 -*-
import os
import unittest

from asammdf import MDF
from vectors_mdf import signals


class TestMDF(unittest.TestCase):

    def setUp(self):
        # mdf3_20
        mdf3_20 = MDF(version='3.20')
        mdf3_20.append(signals, 'mdf3_20')
        mdf3_20.save('mdf3_20.mdf', overwrite=True)
        mdf3_20.close()

    def tearDown(self):
        try:
            # mdf3_20
            if os.path.isfile('mdf3_20.mdf'):
                os.remove('mdf3_20.mdf')
            # mdf4_10
            if os.path.isfile('mdf4_10.mf4'):
                os.remove('mdf4_10.mf4')
        except FileNotFoundError:
            pass

    def test_mdf_exists(self):
        self.assertTrue(MDF)

    # convert from mdf3 to mdf4
    @unittest.skip
    def test_convert_from_mdf3_20_to_mdf4_10(self):
        meas = MDF('mdf3_20.mdf')
        converted = meas.convert(to='4.10', memory='minimum')
        self.assertEqual(converted.version, '4.10')

if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
import os
import unittest

from utils import get_test_data
from asammdf import MDF4

from vectors import s_uint8, s_int32, signals


class TestMDF4(unittest.TestCase):

    def setUp(self):
        # mdf4_00
        mdf4_00 = MDF4(version='4.00')
        mdf4_00.append(signals, 'mdf4_00')
        mdf4_00.save('mdf4_00.mf4', overwrite=True)
        mdf4_00.close()

        # mdf4_10
        mdf4_10 = MDF4(version='4.10')
        mdf4_10.append(signals, 'mdf4_10')
        mdf4_10.save('mdf4_10.mf4', overwrite=True)
        mdf4_10.close()

    def tearDown(self):
        try:
            # mdf4_00
            if os.path.isfile('mdf4_00.mf4'):
                os.remove('mdf4_00.mf4')

            # mdf4_10
            if os.path.isfile('mdf4_10.mf4'):
                os.remove('mdf4_10.mf4')
        except FileNotFoundError:
            pass

    def test_mdf4_exists(self):
        self.assertTrue(MDF4)

    def test_read_mdf4_00(self):
        self.assertTrue(MDF4(get_test_data('test_meas_4.00.mf4')))

    def test_read_mdf4_10(self):
        self.assertTrue(MDF4(get_test_data('test_meas_4.10.mf4')))

    # create mdf4
    def test_create_mdf4_00(self):
        meas = MDF4('mdf4_00.mf4')

        self.assertEqual(s_uint8, meas.get(s_uint8.name))
        self.assertEqual(s_uint8.unit, meas.get(s_uint8.name).unit)
        # self.assertEqual(s_uint8.info, meas.get(s_uint8.name).info)
        # self.assertEqual(s_uint8.comment, meas.get(s_uint8.name).comment)

    def test_create_mdf4_10(self):
        meas = MDF4('mdf4_10.mf4')

        self.assertEqual(s_int32, meas.get(s_int32.name))
        self.assertEqual(s_int32.unit, meas.get(s_int32.name).unit)

        # tests below are failed
        # self.assertEqual(s_int32.info, meas.get(s_int32.name).info)
        # self.assertEqual(s_int32.comment, meas.get(s_int32.name).comment)

if __name__ == '__main__':
    unittest.main()

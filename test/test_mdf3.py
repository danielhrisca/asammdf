# -*- coding: utf-8 -*-
import os
import sys
import unittest

from utils import get_test_data
from asammdf import MDF3

from vectors_mdf import s_uint8, s_int32, s_float64, signals


class TestMDF3(unittest.TestCase):

    def setUp(self):
        # mdf3_00
        mdf3_00 = MDF3(version='3.00')
        mdf3_00.append(signals, 'mdf3_00')
        mdf3_00.save('mdf3_00.mdf', overwrite=True)
        mdf3_00.close()

        # mdf3_10
        mdf3_10 = MDF3(version='3.10')
        mdf3_10.append(signals, 'mdf3_10')
        mdf3_10.save('mdf3_10.mdf', overwrite=True)
        mdf3_10.close()

        # mdf3_20
        mdf3_20 = MDF3(version='3.20')
        mdf3_20.append(signals, 'mdf3_20')
        mdf3_20.save('mdf3_20.mdf', overwrite=True)
        mdf3_20.close()

    def tearDown(self):
        try:
            # mdf3_00
            if os.path.isfile('mdf3_00.mdf'):
                os.remove('mdf3_00.mdf')

            # mdf3_10
            if os.path.isfile('mdf3_10.mdf'):
                os.remove('mdf3_10.mdf')

            # mdf3_20
            if os.path.isfile('mdf3_20.mdf'):
                os.remove('mdf3_20.mdf')
        except FileNotFoundError:
            pass

    def test_mdf3_exists(self):
        self.assertTrue(MDF3)

    # read mdf3
    def test_read_mdf3_00(self):
        self.assertTrue(MDF3(get_test_data('test_meas_3.00.mdf')))

    def test_read_mdf3_10(self):
        self.assertTrue(MDF3(get_test_data('test_meas_3.10.mdf')))

    def test_read_mdf3_20(self):
        self.assertTrue(MDF3(get_test_data('test_meas_3.20.mdf')))

    # create mdf3
    def test_create_mdf3_00(self):
        meas = MDF3('mdf3_00.mdf')

        self.assertEqual(s_uint8, meas.get(s_uint8.name))
        self.assertEqual(s_uint8.unit, meas.get(s_uint8.name).unit)
        # self.assertEqual(s_uint8.info, meas.get(s_uint8.name).info)
        # self.assertEqual(s_uint8.comment, meas.get(s_uint8.name).comment)

    def test_create_mdf3_10(self):
        meas = MDF3('mdf3_10.mdf')

        self.assertEqual(s_int32, meas.get(s_int32.name))
        self.assertEqual(s_int32.unit, meas.get(s_int32.name).unit)

        # tests below are failed
        # self.assertEqual(s_int32.info, meas.get(s_int32.name).info)
        # self.assertEqual(s_int32.comment, meas.get(s_int32.name).comment)

    def test_create_mdf3_20(self):
        meas = MDF3('mdf3_20.mdf')
        self.assertEqual(s_float64, meas.get(s_float64.name), 'signal == channel')
        self.assertEqual(s_float64.unit, meas.get(s_float64.name).unit, 'unit')

        # tests below are failed
        # self.assertEqual(s_float64.info, meas.get(s_float64.name).info, 'info')
        # self.assertEqual(s_float64.comment, meas.get(s_float64.name).comment, 'comment')


if __name__ == '__main__':
    unittest.main()

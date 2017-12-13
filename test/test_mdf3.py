#!/usr/bin/env python
import unittest

from utils import get_test_data
from asammdf import MDF3



class TestMDF3(unittest.TestCase):

    def test_measurement(self):
        self.assertTrue(MDF3)

    def test_read_mdf3_00(self):
        self.assertTrue(MDF3(get_test_data('test_meas_3.00.mdf')))

    def test_read_mdf3_10(self):
        self.assertTrue(MDF3(get_test_data('test_meas_3.10.mdf')))

    def test_read_mdf3_20(self):
        self.assertTrue(MDF3(get_test_data('test_meas_3.20.mdf')))


if __name__ == '__main__':
    unittest.main()

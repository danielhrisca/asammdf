#!/usr/bin/env python
import os
import sys
import unittest

from utils import get_test_data
from asammdf import MDF4


class TestMDF4(unittest.TestCase):

    def test_measurement(self):
        self.assertTrue(MDF4)

    def test_read_mdf4_00(self):
        self.assertTrue(MDF4(get_test_data('test_meas_4.00.mf4')))

    def test_read_mdf4_10(self):
        self.assertTrue(MDF4(get_test_data('test_meas_4.10.mf4')))


if __name__ == '__main__':
    unittest.main()

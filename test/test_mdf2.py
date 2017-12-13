#!/usr/bin/env python
import os
import sys
import unittest

from utils import get_test_data
from asammdf import MDF2



class TestMDF2(unittest.TestCase):

    def test_mdf2_exists(self):
        self.assertTrue(MDF2)

    def test_read_mdf2_00(self):
        self.assertTrue(MDF2(get_test_data('test_meas_2.00.mdf')))


if __name__ == '__main__':
    unittest.main()

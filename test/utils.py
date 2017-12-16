#!/usr/bin/env python
import os
import sys

here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


MEMORY = ('minimum', 'low', 'full')


def get_test_data(filename=""):
    """
    Utility functions needed by all test scripts.
    """
    return os.path.dirname(__file__) + "/data/" + filename

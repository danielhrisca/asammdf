#!/usr/bin/env python
import os


def get_test_data(filename=""):
    """
    Utility functions needed by all test scripts.
    """
    return os.path.dirname(__file__) + "/data/" + filename

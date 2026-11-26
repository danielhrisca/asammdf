#!/usr/bin/env python
import unittest

import numpy as np

from asammdf.blocks.conversion_utils import from_dict


class TestValueRangeToValue(unittest.TestCase):
    def test_integer_raw_values_include_both_bounds(self) -> None:
        conversion = from_dict({"lower_0": 10.0, "upper_0": 20.0, "phys_0": 42.0, "default": -1.0})
        raw = [9, 10, 15, 20, 21]

        values = conversion.convert(np.array(raw, dtype=np.int16))
        self.assertEqual(values.tolist(), [-1.0, 42.0, 42.0, 42.0, -1.0])

        # float raw values exclude the upper bound
        values = conversion.convert(np.array(raw, dtype=np.float64))
        self.assertEqual(values.tolist(), [-1.0, 42.0, 42.0, -1.0, -1.0])

    def test_nan_default_with_integer_table_value(self) -> None:
        conversion = from_dict({"lower_0": 10, "upper_0": 20, "phys_0": 42, "default": float("nan")})
        values = conversion.convert(np.array([9, 15], dtype=np.int16))
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(values[1], 42)


if __name__ == "__main__":
    unittest.main()

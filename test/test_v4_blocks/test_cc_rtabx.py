#!/usr/bin/env python
import unittest

import numpy as np

from asammdf.blocks.conversion_utils import from_dict


class TestValueRangeToText(unittest.TestCase):
    def test_float_bounds_from_100_samples_on(self) -> None:
        # float raw values: 0.5 <= x < 2.5 is "in"
        raw = np.resize([0.0, 0.5, 2.0, 2.25, 2.5], 100)
        expected = [b"out", b"in", b"in", b"in", b"out"]

        for count in (99, 100):  # 100 and more samples take the vectorized path
            conversion = from_dict({"lower_0": 0.5, "upper_0": 2.5, "text_0": b"in", "default_addr": b"out"})
            values = conversion.convert(raw[:count])
            self.assertEqual(values[:5].tolist(), expected, count)


if __name__ == "__main__":
    unittest.main()

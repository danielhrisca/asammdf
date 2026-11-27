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

    def test_uint64_bounds_from_100_samples_on(self) -> None:
        # extract_bus_logging stores a DBC value as lower = upper, here 0 and the SNA value of a 64-bit signal
        raw = np.resize(np.array([0, 1, 2**64 - 2, 2**64 - 1], dtype=np.uint64), 100)
        expected = [b"zero", b"other", b"other", b"SNA"]

        for count in (99, 100):
            conversion = from_dict(
                {
                    "lower_0": 0,
                    "upper_0": 0,
                    "text_0": b"zero",
                    "lower_1": 2**64 - 1,
                    "upper_1": 2**64 - 1,
                    "text_1": b"SNA",
                    "default_addr": b"other",
                }
            )
            values = conversion.convert(raw[:count])
            self.assertEqual(values[:4].tolist(), expected, count)


if __name__ == "__main__":
    unittest.main()

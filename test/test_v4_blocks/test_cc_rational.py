#!/usr/bin/env python
import unittest

import numpy as np

from asammdf.blocks.conversion_utils import conversion_transfer, from_dict


class TestRationalConversion(unittest.TestCase):
    def test_quadratic_term_of_integer_raw_values(self) -> None:
        # (X**2 + 3) / 2
        conversion = from_dict({"P1": 1.0, "P2": 0.0, "P3": 3.0, "P4": 0.0, "P5": 0.0, "P6": 2.0})
        raw = np.array([46340, 46341, 65535], dtype=np.uint16)
        expected = ((raw.astype(np.float64) ** 2 + 3) / 2).tolist()

        self.assertEqual(conversion.convert(raw).tolist(), expected)
        # MDF 3 uses the same formula
        self.assertEqual(conversion_transfer(conversion, version=3).convert(raw).tolist(), expected)


if __name__ == "__main__":
    unittest.main()

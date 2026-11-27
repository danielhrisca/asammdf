#!/usr/bin/env python
import unittest

import numpy as np

from asammdf.blocks.conversion_utils import from_dict


class TestValueToText(unittest.TestCase):
    def test_real_keys_from_150_samples_on(self) -> None:
        # the keys are REAL values: 0.5 of a float channel and 2**64 - 1, the SNA value of a 64-bit signal
        conversion = from_dict(
            {"val_0": 0.5, "val_1": float(2**64 - 1), "text_0": b"half", "text_1": b"SNA", "default_addr": b"other"}
        )
        float_raw = np.resize(np.array([0.0, 0.5]), 150)
        integer_raw = np.resize(np.array([0, 2**64 - 1], dtype=np.uint64), 150)

        np.testing.assert_array_equal(conversion.convert(float_raw)[:2], np.array([b"other", b"half"]), strict=True)
        np.testing.assert_array_equal(conversion.convert(integer_raw)[:2], np.array([b"other", b"SNA"]), strict=True)


if __name__ == "__main__":
    unittest.main()

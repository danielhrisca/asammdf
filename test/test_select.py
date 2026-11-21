from io import BytesIO
import unittest

import numpy as np

from asammdf import MDF, Signal


class TestSelect(unittest.TestCase):
    def test_fallback_invalidation_bits_across_fragments(self) -> None:
        timestamps = np.arange(64, dtype=np.float64)
        masks = {"early": timestamps == 3, "late": timestamps == 60, "valid": np.zeros(len(timestamps), dtype=bool)}
        with BytesIO() as stream:
            with MDF(version="4.10") as mdf:
                mdf.append(
                    [Signal(timestamps, timestamps, name=name, invalidation_bits=mask) for name, mask in masks.items()]
                )
                mdf.save(stream)

            with MDF(stream) as mdf:
                mdf.configure(read_fragment_size=256)
                signals = mdf.select(list(masks), raw=True)

        for signal in signals:
            with self.subTest(channel=signal.name):
                np.testing.assert_array_equal(signal.samples, timestamps, strict=True)
                mask = masks[signal.name]
                if not mask.any():
                    self.assertIsNone(signal.invalidation_bits)
                else:
                    np.testing.assert_array_equal(signal.invalidation_bits, mask, strict=True)

import unittest

import numpy as np

from asammdf.blocks import bus_logging_utils as blu


class TestCANTP(unittest.TestCase):
    tempdir = None

    payload = np.vstack(
        [
            np.frombuffer(b"\x10\x0b\x52\x49\x47\x20\x20\x39", dtype="uint8"),  # Initial part
            np.frombuffer(b"\x30\xff\x00\x4c\x40\x00\xd5\x54", dtype="uint8"),  # Flow control
            np.frombuffer(b"\x21\x30\x30\x30\x38\x33\x00\x00", dtype="uint8"),  # Final (second) part
            np.frombuffer(b"\x10\x0b\x52\x49\x47\x20\x20\x39", dtype="uint8"),  # Initial part of next frame...
        ]
    )
    ts = np.array([0.112, 0.113, 0.116, 0.201])

    def test_merge_cantp(self) -> None:
        merged, t = blu.merge_cantp(TestCANTP.payload, TestCANTP.ts)

        assert merged.shape == (1, 11)
        assert merged[0, -1] == 0x33
        assert t.shape == (1,)
        assert t[0] == 0.116

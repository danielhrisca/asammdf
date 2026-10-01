#!/usr/bin/env python
import unittest

from asammdf.blocks.cutils import get_channel_raw_bytes


class TestGetChannelRawBytes(unittest.TestCase):
    data = bytes(range(1, 9))  # two records of 4 bytes

    def test_channel_inside_record(self) -> None:
        self.assertEqual(get_channel_raw_bytes(self.data, 4, 1, 2), b"\x02\x03\x06\x07")

    def test_channel_partially_outside_record(self) -> None:
        self.assertEqual(
            get_channel_raw_bytes(self.data, 4, 2, 4),
            b"\x03\x04\x00\x00\x07\x08\x00\x00",
        )

    def test_channel_outside_record(self) -> None:
        # used to write past the output buffer, run with PYTHONMALLOC=debug to detect it
        self.assertEqual(get_channel_raw_bytes(self.data, 4, 6, 2), b"\x00" * 4)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
from pathlib import Path
import tempfile
import unittest

from asammdf.blocks import v4_constants as v4c
from asammdf.blocks.cutils import get_channel_raw_bytes, get_channel_raw_bytes_complete
from asammdf.blocks.utils import DataBlockInfo


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


class TestGetChannelRawBytesComplete(unittest.TestCase):
    data = bytes(range(1, 9))  # two records of 4 bytes

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.file_name = str(Path(cls.tempdir.name) / "records.bin")
        Path(cls.file_name).write_bytes(cls.data)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def raw_bytes(self, byte_offset: int, byte_count: int) -> bytearray:
        block = DataBlockInfo(
            address=0,
            block_type=v4c.DT_BLOCK,
            original_size=len(self.data),
            compressed_size=len(self.data),
            param=0,
        )
        ((raw, _),) = get_channel_raw_bytes_complete(
            [block], [(byte_offset, byte_count, -1)], self.file_name, 2, 4, 0, 0
        )
        return raw

    def test_channel_inside_record(self) -> None:
        self.assertEqual(self.raw_bytes(1, 2), b"\x02\x03\x06\x07")

    def test_channel_partially_outside_record(self) -> None:
        self.assertEqual(self.raw_bytes(2, 4), b"\x03\x04\x00\x00\x07\x08\x00\x00")

    def test_channel_outside_record(self) -> None:
        # used to read the next record, and past the end of the block buffer for the last one
        self.assertEqual(self.raw_bytes(6, 2), b"\x00" * 4)


if __name__ == "__main__":
    unittest.main()

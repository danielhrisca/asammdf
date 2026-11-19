#!/usr/bin/env python
from pathlib import Path
import sys
import tempfile
import tracemalloc
from typing import Any
import unittest

import numpy as np

from asammdf.blocks import v4_constants as v4c
from asammdf.blocks.cutils import (
    data_block_from_arrays,
    get_channel_raw_bytes,
    get_channel_raw_bytes_complete,
    get_invalidation_bits_array,
)
from asammdf.blocks.utils import DataBlockInfo


class TestDataBlockFromArrays(unittest.TestCase):
    def test_short_array_view(self) -> None:
        # Keep the backing allocation alive so the old unchecked read is safe.
        backing = np.array([1, 2, 3, 4], dtype=np.uint8)
        for width in (1, 2):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "too small"):
                data_block_from_arrays([(backing[:width], width)], 2)

    def test_interleaved_records(self) -> None:
        arrays: list[Any] = [b"\x01\x02", bytearray(b"\x01\x02"), np.array([1, 2], dtype=np.uint8)]
        arrays.append(np.array([[1, 9], [2, 9]], dtype=np.uint8)[:, 0])
        for array in arrays:
            for threads in (1, 2, 11):
                with self.subTest(array=type(array), threads=threads):
                    self.assertEqual(
                        data_block_from_arrays([(array, 1), (b"\x03\x04", 1)], 2, threads),
                        b"\x01\x03\x02\x04",
                    )

    def test_empty_output(self) -> None:
        self.assertEqual(data_block_from_arrays([], 3), b"")
        self.assertEqual(data_block_from_arrays([(b"", 1)], 0), b"")

    def test_short_buffers(self) -> None:
        buffers: tuple[Any, ...] = (b"\x01", bytearray(b"\x01"), np.arange(4, dtype=np.uint8)[::2])
        for data in buffers:
            with self.subTest(buffer=type(data)), self.assertRaisesRegex(ValueError, "too small"):
                data_block_from_arrays([(data, 2)], 2)

    def test_invalid_counts(self) -> None:
        cases: list[tuple[Any, int, int]] = [
            ([(b"", 0)], -1, 1),
            ([(b"", -1)], 0, 1),
            ([(b"", 0)], 0, 0),
            ([(b"", 0)], 0, -1),
        ]
        for blocks, cycles, threads in cases:
            with self.subTest(cycles=cycles, threads=threads, blocks=blocks), self.assertRaises(ValueError):
                data_block_from_arrays(blocks, cycles, threads)

    def test_large_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "too small"):
            data_block_from_arrays([(b"", 1)], sys.maxsize)
        with self.assertRaises(OverflowError):
            data_block_from_arrays([(b"", 0)], sys.maxsize + 1)
        with self.assertRaises(OverflowError):
            data_block_from_arrays([(b"", sys.maxsize + 1)], 0)
        with self.assertRaisesRegex(OverflowError, "record size"):
            data_block_from_arrays([(b"", sys.maxsize), (b"", 1)], 0)
        self.assertEqual(data_block_from_arrays([(b"\x01", 1)], 1, sys.maxsize), b"\x01")

    def test_invalid_blocks(self) -> None:
        cases: list[tuple[Any, type[Exception]]] = [
            (None, TypeError),
            ([None], TypeError),
            ([(object(), 1)], TypeError),
            ([(b"", "one")], TypeError),
            ([(b"",)], ValueError),
            ([(b"", 0, 1)], ValueError),
            ([[b""]], ValueError),
        ]
        for blocks, error in cases:
            with self.subTest(blocks=blocks), self.assertRaises(error):
                data_block_from_arrays(blocks, 1)

    def test_tuple_and_list_blocks(self) -> None:
        blocks: Any = ([b"\x01\x02", 1],)
        self.assertEqual(data_block_from_arrays(blocks, 2), b"\x01\x02")


class TestGetInvalidationBitsArray(unittest.TestCase):
    def test_all_valid_array_is_released(self) -> None:
        cycles = 100_000
        data = bytes(cycles)
        tracemalloc.start()
        try:
            for _ in range(10):
                self.assertIsNone(get_invalidation_bits_array(data, 1, 0, cycles, True))
            grown = tracemalloc.get_traced_memory()[0]
        finally:
            tracemalloc.stop()
        self.assertLess(grown, 5 * cycles)


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

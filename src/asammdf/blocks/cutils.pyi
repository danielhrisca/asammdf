import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any

from ..signal import InvalidationArray
from .blocks_common import UnpackFrom
from .utils import DataBlockInfo

def sort_data_block(
    signal_data: bytes,
    partial_records: dict[int, list[bytes]],
    record_size: dict[int, int],
    id_size: int,
    optional: UnpackFrom[tuple[int]] | None,
) -> bytes: ...
def extract(signal_data: bytes, is_byte_array: bool, offsets: NDArray[np.uintp]) -> NDArray[np.uint8]: ...
def get_vlsd_max_sample_size(data: bytes, offsets: NDArray[np.uint64], count: int) -> int: ...
def get_channel_raw_bytes(
    data_block: bytes | bytearray, record_size: int, byte_offset: int, byte_count: int
) -> bytearray: ...
def get_invalidation_bits_array(
    data_block: bytes | bytearray, invalidation_size: int, invalidation_pos: int
) -> NDArray[np.uint8]: ...
def get_channel_raw_bytes_parallel(
    data_block: bytes | bytearray, record_size: int, signals: list[list[int]], thread_count: int = 11
) -> list[bytearray]: ...
def data_block_from_arrays(
    data_blocks: list[tuple[bytes | NDArray[Any], int]], cycles_obj: int, thread_count: int = 11
) -> bytearray: ...
def bytes_dtype_size(ret: NDArray[Any]) -> int: ...
def get_channel_raw_bytes_complete(
    data_blocks_info: list[DataBlockInfo],
    signals: list[tuple[int, int, int]],
    file_name: str,
    cycles: int,
    record_size: int,
    invalidation_bytes: int,
    group_index: int,
    thread_count: int = 11,
) -> tuple[tuple[bytearray, InvalidationArray | None], ...]: ...

"""ASAM MDF version 3 file format module"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from itertools import product
import logging
from math import ceil
import mmap
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time
from traceback import format_exc
import typing
from typing import BinaryIO, IO, Literal, TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np
from numpy import (
    arange,
    array,
    array_equal,
    ascontiguousarray,
    column_stack,
    concatenate,
    float32,
    float64,
    frombuffer,
    linspace,
    searchsorted,
    uint16,
    unique,
    zeros,
)
from numpy.typing import ArrayLike, DTypeLike, NDArray
from pandas import DataFrame
from typing_extensions import Any, Buffer, overload, SupportsBytes, TypedDict, Unpack

from .. import tool
from ..signal import Signal
from . import mdf_common, utils
from . import v2_v3_constants as v23c
from .conversion_utils import conversion_transfer
from .cutils import data_block_from_arrays, get_channel_raw_bytes
from .mdf_common import MDF_Common, MdfCommonKwargs
from .options import GLOBAL_OPTIONS
from .source_utils import Source
from .types import ChannelsType, CompressionType, RasterType, StrPath
from .utils import (
    as_non_byte_sized_signed_int,
    CHANNEL_COUNT,
    CONVERT,
    count_channel_groups,
    DataBlockInfo,
    FileLike,
    fmt_to_datatype_v3,
    get_fmt_v3,
    get_text_v3,
    is_file_like,
    MdfException,
    Terminated,
    UniqueDB,
    validate_version_argument,
    VirtualChannelGroup,
)
from .v2_v3_blocks import (
    Channel,
    ChannelConversion,
    ChannelConversionKwargs,
    ChannelDependency,
    ChannelDependencyKwargs,
    ChannelExtension,
    ChannelExtensionKwargs,
    ChannelGroup,
    ChannelGroupKwargs,
    ChannelKwargs,
    DataGroup,
    DataGroupKwargs,
    FileIdentificationBlock,
    HeaderBlock,
    TextBlock,
    TriggerBlock,
)
from .v2_v3_constants import Version, Version2

if TYPE_CHECKING:
    from ..mdf import MDF


try:
    decode = np.strings.decode
    encode = np.strings.encode
except:
    decode = np.char.decode
    encode = np.char.encode

logger = logging.getLogger("asammdf")

__all__ = ["MDF3"]


Group = mdf_common.GroupV3


class Kwargs(MdfCommonKwargs, total=False):
    skip_sorting: bool


class TriggerInfoDict(TypedDict):
    comment: str
    index: int
    group: int
    time: float
    pre_time: float
    post_time: float


class MDF3(MDF_Common[Group]):
    """The `header` attribute is a `HeaderBlock`.

    The `groups` attribute is a list of `Group` objects, each one with the
    following attributes:

    * ``data_group`` - `DataGroup` object
    * ``channel_group`` - `ChannelGroup` object
    * ``channels`` - list of `Channel` objects with the same order as found in
      the MDF file
    * ``channel_dependencies`` - list of `ChannelArrayBlock` objects in case of
      channel arrays; list of `Channel` objects in case of structure channel
      composition
    * ``data_blocks`` - list of `DataBlockInfo` objects, each one containing
      address, type, size and other information about the block
    * ``data_location``- integer code for data location (original file,
      temporary file or memory)
    * ``data_block_addr`` - list of raw samples starting addresses
    * ``data_block_type`` - list of codes for data block type
    * ``data_block_size`` - list of raw samples block size
    * ``sorted`` - sorted indicator flag
    * ``record_size`` - dict that maps record IDs to record sizes in bytes
    * ``size`` - total size of data block for the current group
    * ``trigger`` - `Trigger` object for current group

    Parameters
    ----------
    name : str | path-like | file-like, optional
        MDF file name (if provided it must be a real file name) or file-like
        object.
    version : str, default '3.30'
        MDF file version ('2.00', '2.10', '2.14', '3.00', '3.10', '3.20' or
        '3.30').
    callback : function, optional
        Function to call to update the progress; the function must accept two
        arguments (the current progress and maximum progress value).

    Attributes
    ----------
    channels_db : dict
        Used for fast channel access by name; for each name key the value is a
        list of (group index, channel index) tuples.
    groups : list
        List of data group dicts.
    header : HeaderBlock
        MDF file header.
    identification : FileIdentificationBlock
        MDF file start block.
    last_call_info : dict | None
        A dict to hold information about the last called method.

        .. versionadded:: 5.12.0

    masters_db : dict
        Used for fast master channel access; for each group index key the value
        is the master channel index.
    name : pathlib.Path
        MDF file name.
    version : str
        MDF version.
    """

    def __init__(
        self,
        name: StrPath | FileLike | None = None,
        version: Version2 | Version = "3.30",
        channels: list[str] | None = None,
        **kwargs: Unpack[Kwargs],
    ) -> None:
        if not kwargs.get("__internal__", False):
            raise MdfException("Always use the MDF class; do not use the class MDF3 directly")

        # bind cache to instance to avoid memory leaks
        self.determine_max_vlsd_sample_size = lru_cache(maxsize=1024 * 1024)(self._determine_max_vlsd_sample_size)

        self._kwargs = kwargs
        self._password = kwargs.get("password", None)
        self.original_name = kwargs["original_name"]
        if channels is None:
            self.load_filter = set()
            self.use_load_filter = False
        else:
            self.load_filter = set(channels)
            self.use_load_filter = True

        self.temporary_folder = kwargs.get("temporary_folder", GLOBAL_OPTIONS["temporary_folder"])

        self.masters_db: dict[int, int] = {}
        self.version: str = version

        self._master_channel_metadata: dict[int, tuple[str, Literal[1]]] = {}
        self._closed = False

        self._tempfile = NamedTemporaryFile(dir=self.temporary_folder)
        self._tempfile.write(b"\0")
        self._mapped_file: BinaryIO | None = None
        self._file: FileLike | mmap.mmap | None = self._mapped_file

        self._remove_source_from_channel_names = kwargs.get("remove_source_from_channel_names", False)

        self._read_fragment_size = GLOBAL_OPTIONS["read_fragment_size"]
        self._write_fragment_size = GLOBAL_OPTIONS["write_fragment_size"]
        self._single_bit_uint_as_bool = GLOBAL_OPTIONS["single_bit_uint_as_bool"]
        self._integer_interpolation = GLOBAL_OPTIONS["integer_interpolation"]
        self._float_interpolation = GLOBAL_OPTIONS["float_interpolation"]
        self._use_display_names = kwargs.get("use_display_names", GLOBAL_OPTIONS["use_display_names"])
        self._fill_0_for_missing_computation_channels = kwargs.get(
            "fill_0_for_missing_computation_channels", GLOBAL_OPTIONS["fill_0_for_missing_computation_channels"]
        )

        self._si_map: dict[bytes | int, ChannelExtension] = {}
        self._cc_map: dict[bytes | int, ChannelConversion] = {}

        self._master: NDArray[Any] | None = None

        self.virtual_groups_map: dict[int, int] = {}
        self.virtual_groups: dict[int, VirtualChannelGroup] = {}

        self.vlsd_max_length: dict[tuple[int, str], int] = {}

        self._delete_on_close = False

        progress = kwargs.get("progress", None)

        super().__init__(kwargs.get("raise_on_multiple_occurrences", GLOBAL_OPTIONS["raise_on_multiple_occurrences"]))

        if name:
            if is_file_like(name):
                self._file = name
                self.name = self.original_name = Path("From_FileLike.mdf")
                self._from_filelike = True
                self._read(self._file, mapped=False, progress=progress)
            else:
                try:
                    if sys.maxsize < 2**32:
                        self.name = Path(name)
                        self._file = open(self.name, "rb")
                        self._from_filelike = False
                        self._read(self._file, mapped=False, progress=progress)
                    else:
                        self.name = Path(name)
                        self._mapped_file = open(self.name, "rb")
                        self._file = mmap.mmap(self._mapped_file.fileno(), 0, access=mmap.ACCESS_READ)
                        self._from_filelike = False
                        self._read(self._file, mapped=True, progress=progress)
                except:
                    if self._file:
                        self._file.close()
                    if self._mapped_file:
                        self._mapped_file.close()
                    self._file = self._mapped_file = None
                    raise
        else:
            self._from_filelike = False
            version = validate_version_argument(version, hint=3)
            self.identification = FileIdentificationBlock(version=version)
            self.version = version
            self.header = HeaderBlock(version=self.version)
            self.name = Path("__new__.mdf")

        if not kwargs.get("skip_sorting", False):
            self._sort(progress=progress)

        for index, grp in enumerate(self.groups):
            self.virtual_groups_map[index] = index
            if index not in self.virtual_groups:
                self.virtual_groups[index] = VirtualChannelGroup()

            virtual_channel_group = self.virtual_groups[index]
            virtual_channel_group.groups.append(index)
            virtual_channel_group.record_size = grp.channel_group.samples_byte_nr
            virtual_channel_group.cycles_nr = grp.channel_group.cycles_nr

        self._parent: MDF | None = None

    def __del__(self) -> None:
        self.close()

    def _load_data(
        self,
        group: Group,
        record_offset: int = 0,
        record_count: int | None = None,
        optimize_read: bool = True,
    ) -> Iterator[tuple[bytes, int, int | None]]:
        """Get group's data block bytes."""
        has_yielded = False
        offset = 0
        _count = record_count
        channel_group = group.channel_group

        stream: FileLike | mmap.mmap | IO[bytes]
        if group.data_location == v23c.LOCATION_ORIGINAL_FILE:
            # go to the first data block of the current data group
            if self._file is None:
                raise RuntimeError(f"file was not opened '{self.name}'")
            stream = self._file
        else:
            stream = self._tempfile

        samples_size = channel_group.samples_byte_nr

        record_offset *= samples_size
        if record_count is not None:
            record_count *= samples_size

        finished = False

        # go to the first data block of the current data group
        if group.sorted:
            if not samples_size:
                yield b"", 0, _count
                has_yielded = True
            else:
                if self._read_fragment_size:
                    split_size = self._read_fragment_size // samples_size
                    split_size *= samples_size
                else:
                    channels_nr = len(group.channels)

                    y_axis = CONVERT

                    idx = int(searchsorted(CHANNEL_COUNT, channels_nr, side="right") - 1)
                    idx = max(idx, 0)
                    split_size = y_axis[idx]

                    split_size = split_size // samples_size
                    split_size *= samples_size

                if split_size == 0:
                    split_size = samples_size

                split_size = int(split_size)

                blocks = iter(group.data_blocks)

                cur_size = 0
                data_list: list[bytes] = []

                while True:
                    try:
                        info = next(blocks)
                        address, size = info.address, typing.cast(int, info.original_size)
                        current_address = address
                    except StopIteration:
                        break

                    if offset + size < record_offset + 1:
                        offset += size
                        continue

                    stream.seek(address)

                    if offset < record_offset:
                        delta = record_offset - offset
                        stream.seek(delta, 1)
                        current_address += delta
                        size -= delta
                        offset = record_offset

                    if record_count:
                        while size >= split_size - cur_size:
                            stream.seek(current_address)
                            if data_list:
                                data_list.append(stream.read(min(record_count, split_size - cur_size)))

                                bts = b"".join(data_list)[:record_count]
                                record_count -= len(bts)
                                __count = len(bts) // samples_size
                                yield bts, offset // samples_size, __count
                                has_yielded = True
                                current_address += split_size - cur_size

                                if record_count <= 0:
                                    finished = True
                                    break
                            else:
                                bts = stream.read(min(split_size, record_count))[:record_count]
                                record_count -= len(bts)
                                __count = len(bts) // samples_size
                                yield bts, offset // samples_size, __count
                                has_yielded = True
                                current_address += split_size - cur_size

                                if record_count <= 0:
                                    finished = True
                                    break

                            offset += split_size

                            size -= split_size - cur_size
                            data_list = []
                            cur_size = 0
                    else:
                        while size >= split_size - cur_size:
                            stream.seek(current_address)
                            if data_list:
                                data_list.append(stream.read(split_size - cur_size))

                                yield b"".join(data_list), offset, _count
                                has_yielded = True
                                current_address += split_size - cur_size
                            else:
                                yield stream.read(split_size), offset, _count
                                has_yielded = True
                                current_address += split_size

                            offset += split_size

                            size -= split_size - cur_size
                            data_list = []
                            cur_size = 0

                    if finished:
                        data_list = []
                        offset = -1
                        break

                    if size:
                        stream.seek(current_address)
                        if record_count:
                            data_list.append(stream.read(min(record_count, size)))
                        else:
                            data_list.append(stream.read(size))

                        cur_size += size
                        offset += size

                if data_list:
                    data = b"".join(data_list)
                    if record_count is not None:
                        data = data[:record_count]
                        yield data, offset, len(data) // samples_size
                        has_yielded = True
                    else:
                        yield data, offset, _count
                        has_yielded = True

                elif not offset:
                    yield b"", 0, _count
                    has_yielded = True
                if not has_yielded:
                    yield b"", 0, _count

        else:
            record_id = group.channel_group.record_id
            cg_size = group.record_size
            if group.data_group.record_id_len <= 2:
                record_id_nr = group.data_group.record_id_len
            else:
                record_id_nr = 0
            data_list = []

            blocks = iter(group.data_blocks)

            for info in blocks:
                address, size = info.address, typing.cast(int, info.original_size)
                stream.seek(address)
                data = stream.read(size)

                i = 0
                while i < size:
                    rec_id = data[i]
                    # skip record id
                    i += 1
                    rec_size = cg_size[rec_id]
                    if rec_id == record_id:
                        rec_data = data[i : i + rec_size]
                        data_list.append(rec_data)
                    # consider the second record ID if it exists
                    if record_id_nr == 2:
                        i += rec_size + 1
                    else:
                        i += rec_size
                data = b"".join(data_list)
                size = len(data)

                if size:
                    if offset + size < record_offset + 1:
                        offset += size
                        continue

                    if offset < record_offset:
                        delta = record_offset - offset
                        size -= delta
                        offset = record_offset

                    yield data, offset, _count
                    has_yielded = True
                    offset += size
        if not has_yielded:
            yield b"", 0, _count

    def _prepare_record(self, group: Group) -> list[tuple[np.dtype[Any], int, int, int] | None]:
        """Compute record list.

        Parameters
        ----------
        group : dict
            MDF group dict.

        Returns
        -------
        record : list
            Mapping of channels to records fields, records fields dtype.
        """
        if group.record is None:
            byte_order = self.identification.byte_order
            channels = group.channels

            record: list[tuple[np.dtype[Any], int, int, int] | None] = []

            for new_ch in channels:
                start_offset = new_ch.start_offset
                try:
                    additional_byte_offset = new_ch.additional_byte_offset
                    start_offset += 8 * additional_byte_offset
                except AttributeError:
                    pass

                byte_offset, bit_offset = divmod(start_offset, 8)
                data_type = new_ch.data_type
                bit_count = new_ch.bit_count

                if not new_ch.component_addr:
                    # adjust size to 1, 2, 4 or 8 bytes
                    size = bit_offset + bit_count

                    byte_size, rem = divmod(size, 8)
                    if rem:
                        byte_size += 1
                    bit_size = byte_size * 8

                    if data_type in (
                        v23c.DATA_TYPE_SIGNED_MOTOROLA,
                        v23c.DATA_TYPE_UNSIGNED_MOTOROLA,
                    ):
                        if size > 32:
                            bit_offset += 64 - bit_size
                        elif size > 16:
                            bit_offset += 32 - bit_size
                        elif size > 8:
                            bit_offset += 16 - bit_size

                    if not new_ch.dtype_fmt:
                        new_ch.dtype_fmt = np.dtype(get_fmt_v3(data_type, size, byte_order))

                    record.append(
                        (
                            new_ch.dtype_fmt,
                            new_ch.dtype_fmt.itemsize,
                            byte_offset,
                            bit_offset,
                        )
                    )
                else:
                    record.append(None)

            group.record = record

        return group.record

    def _get_not_byte_aligned_data(self, data: bytes, group: Group, ch_nr: int) -> NDArray[Any]:
        big_endian_types = (
            v23c.DATA_TYPE_UNSIGNED_MOTOROLA,
            v23c.DATA_TYPE_FLOAT_MOTOROLA,
            v23c.DATA_TYPE_DOUBLE_MOTOROLA,
            v23c.DATA_TYPE_SIGNED_MOTOROLA,
        )

        record_size = group.channel_group.samples_byte_nr

        channel = group.channels[ch_nr]

        byte_offset, bit_offset = divmod(channel.start_offset, 8)
        bit_count = channel.bit_count

        byte_size = bit_offset + bit_count
        if byte_size % 8:
            byte_size = (byte_size // 8) + 1
        else:
            byte_size //= 8

        types = [
            ("", f"S{byte_offset}"),
            ("vals", f"({byte_size},)u1"),
            ("", f"S{record_size - byte_size - byte_offset}"),
        ]

        vals: NDArray[Any] = np.rec.fromstring(data, dtype=np.dtype(types))

        vals = vals["vals"]

        if byte_size in {1, 2, 4, 8}:
            extra_bytes = 0
        else:
            extra_bytes = 4 - (byte_size % 4)

        std_size = byte_size + extra_bytes

        big_endian = channel.data_type in big_endian_types

        # prepend or append extra bytes columns
        # to get a standard size number of bytes

        if extra_bytes:
            if big_endian:
                vals = column_stack([vals, zeros(len(vals), dtype=f"<({extra_bytes},)u1")])
                try:
                    vals = vals.view(f">u{std_size}").ravel()
                except:
                    vals = frombuffer(vals.tobytes(), dtype=f">u{std_size}")

                vals = vals >> (extra_bytes * 8 + bit_offset)
                vals &= (1 << bit_count) - 1

            else:
                vals = column_stack([vals, zeros(len(vals), dtype=f"<({extra_bytes},)u1")])
                try:
                    vals = vals.view(f"<u{std_size}").ravel()
                except:
                    vals = frombuffer(vals.tobytes(), dtype=f"<u{std_size}")

                vals = vals >> bit_offset
                vals &= (1 << bit_count) - 1

        else:
            if big_endian:
                try:
                    vals = vals.view(f">u{std_size}").ravel()
                except:
                    vals = frombuffer(vals.tobytes(), dtype=f">u{std_size}")

                vals = vals >> bit_offset
                vals &= (1 << bit_count) - 1

            else:
                try:
                    vals = vals.view(f"<u{std_size}").ravel()
                except:
                    vals = frombuffer(vals.tobytes(), dtype=f"<u{std_size}")

                vals = vals >> bit_offset
                vals &= (1 << bit_count) - 1

        data_type = channel.data_type

        if data_type in v23c.SIGNED_INT:
            return as_non_byte_sized_signed_int(vals, bit_count)
        elif data_type in v23c.FLOATS:
            return vals.view(get_fmt_v3(data_type, bit_count, self.identification.byte_order))
        else:
            return vals

    def _read(
        self,
        stream: FileLike | mmap.mmap,
        mapped: bool = False,
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> None:
        filter_channels = self.use_load_filter

        cg_count, _ = count_channel_groups(stream)
        if progress is not None:
            if callable(progress):
                progress(0, cg_count)
        current_cg_index = 0

        stream.seek(0, 2)
        self.file_limit = stream.tell()
        stream.seek(0)

        dg_cntr = 0

        self.identification = FileIdentificationBlock(stream=stream)
        self.header = HeaderBlock(stream=stream)

        self.version = self.identification.version_str.decode("latin-1").strip(" \n\t\0")

        # this will hold mapping from channel address to Channel object
        # needed for linking dependency blocks to referenced channels after
        # the file is loaded
        ch_map = {}

        # go to first data group
        dg_addr = self.header.first_dg_addr
        # read each data group sequentially
        while dg_addr:
            if dg_addr > self.file_limit:
                logger.warning(f"Data group address {dg_addr:X} is outside the file size {self.file_limit}")
                break
            data_group = DataGroup(address=dg_addr, stream=stream, mapped=mapped)
            record_id_nr = data_group.record_id_len
            cg_nr = data_group.cg_nr
            cg_addr = data_group.first_cg_addr
            data_addr = data_group.data_block_addr

            # read trigger information if available
            trigger_addr = data_group.trigger_addr
            if trigger_addr:
                if trigger_addr > self.file_limit:
                    logger.warning(f"Trigger address {trigger_addr:X} is outside the file size {self.file_limit}")
                    trigger = None
                else:
                    trigger = TriggerBlock(address=trigger_addr, stream=stream)
            else:
                trigger = None

            new_groups: list[Group] = []
            for i in range(cg_nr):
                kargs: DataGroupKwargs = {"first_cg_addr": cg_addr, "data_block_addr": data_addr}
                if self.version >= "3.20":
                    kargs["block_len"] = v23c.DG_POST_320_BLOCK_SIZE
                else:
                    kargs["block_len"] = v23c.DG_PRE_320_BLOCK_SIZE
                kargs["record_id_len"] = record_id_nr
                kargs["address"] = data_group.address

                new_groups.append(Group(DataGroup(**kargs)))
                grp = new_groups[-1]
                grp.channels = []
                grp.trigger = trigger
                grp.channel_dependencies = []

                if record_id_nr:
                    grp.sorted = False
                else:
                    grp.sorted = True

                # read each channel group sequentially
                if cg_addr > self.file_limit:
                    logger.warning(f"Channel group address {cg_addr:X} is outside the file size {self.file_limit}")
                    break
                grp.channel_group = ChannelGroup(address=cg_addr, stream=stream)

                # go to first channel of the current channel group
                ch_addr = grp.channel_group.first_ch_addr
                ch_cntr = 0
                grp_chs = grp.channels

                while ch_addr:
                    if ch_addr > self.file_limit:
                        logger.warning(f"Channel address {ch_addr:X} is outside the file size {self.file_limit}")
                        break

                    if filter_channels:
                        display_names = {}
                        if utils.stream_is_mmap(stream, mapped):
                            (
                                id_,
                                block_len,
                                next_ch_addr,
                                channel_type,
                                name_bytes,
                            ) = v23c.CHANNEL_FILTER_uf(stream, ch_addr)
                            name = name_bytes.decode("latin-1").strip(" \t\n\r\0")
                            if block_len >= v23c.CN_LONGNAME_BLOCK_SIZE:
                                tx_address = v23c.UINT32_uf(stream, ch_addr + v23c.CN_SHORT_BLOCK_SIZE)[0]
                                if tx_address:
                                    name = get_text_v3(tx_address, stream, mapped=mapped)
                                if block_len == v23c.CN_DISPLAYNAME_BLOCK_SIZE:
                                    tx_address = v23c.UINT32_uf(stream, ch_addr + v23c.CN_LONGNAME_BLOCK_SIZE)[0]
                                    if tx_address:
                                        display_names = {
                                            get_text_v3(tx_address, stream, mapped=mapped): "display_name",
                                        }

                        else:
                            stream.seek(ch_addr)
                            (
                                id_,
                                block_len,
                                next_ch_addr,
                                channel_type,
                                name_bytes,
                            ) = v23c.CHANNEL_FILTER_u(stream.read(v23c.CHANNEL_FILTER_SIZE))
                            name = name_bytes.decode("latin-1").strip(" \t\n\r\0")

                            if block_len >= v23c.CN_LONGNAME_BLOCK_SIZE:
                                stream.seek(ch_addr + v23c.CN_SHORT_BLOCK_SIZE)
                                tx_address = v23c.UINT32_u(stream.read(4))[0]
                                if tx_address:
                                    name = get_text_v3(tx_address, stream, mapped=mapped)
                                if block_len == v23c.CN_DISPLAYNAME_BLOCK_SIZE:
                                    stream.seek(ch_addr + v23c.CN_LONGNAME_BLOCK_SIZE)
                                    tx_address = v23c.UINT32_u(stream.read(4))[0]
                                    if tx_address:
                                        display_names = {
                                            get_text_v3(tx_address, stream, mapped=mapped): "display_name",
                                        }

                        if id_ != b"CN":
                            message = f'Expected "CN" block @{hex(ch_addr)} but found "{id_!r}"'
                            raise MdfException(message)

                        if self._remove_source_from_channel_names:
                            name = name.split("\\", 1)[0]
                            display_names = {_name.split("\\", 1)[0]: val for _name, val in display_names.items()}

                        if (
                            channel_type == v23c.CHANNEL_TYPE_MASTER
                            or name in self.load_filter
                            or (any(_name in self.load_filter for _name in display_names))
                        ):
                            new_ch = Channel(
                                address=ch_addr,
                                stream=stream,
                                mapped=mapped,
                                si_map=self._si_map,
                                cc_map=self._cc_map,
                                parsed_strings=(name, display_names),
                            )
                        else:
                            ch_addr = next_ch_addr
                            continue
                    else:
                        # read channel block and create channel object
                        new_ch = Channel(
                            address=ch_addr,
                            stream=stream,
                            mapped=mapped,
                            si_map=self._si_map,
                            cc_map=self._cc_map,
                            parsed_strings=None,
                        )

                    if new_ch.data_type not in v23c.VALID_DATA_TYPES:
                        ch_addr = new_ch.next_ch_addr
                        continue

                    if self._remove_source_from_channel_names:
                        new_ch.name = new_ch.name.split("\\", 1)[0]
                        new_ch.display_names = {
                            _name.split("\\", 1)[0]: val for _name, val in new_ch.display_names.items()
                        }

                    # check if it has channel dependencies
                    if new_ch.component_addr:
                        dep = ChannelDependency(address=new_ch.component_addr, stream=stream)
                    else:
                        dep = None
                    grp.channel_dependencies.append(dep)

                    # update channel map
                    entry = dg_cntr, ch_cntr
                    ch_map[ch_addr] = entry

                    for name in (new_ch.name, *tuple(new_ch.display_names)):
                        if name:
                            self.channels_db.add(name, entry)

                    if new_ch.channel_type == v23c.CHANNEL_TYPE_MASTER:
                        self.masters_db[dg_cntr] = ch_cntr
                    # go to next channel of the current channel group

                    ch_cntr += 1
                    grp_chs.append(new_ch)
                    ch_addr = new_ch.next_ch_addr

                cg_addr = grp.channel_group.next_cg_addr
                dg_cntr += 1

                current_cg_index += 1
                if progress is not None:
                    if callable(progress):
                        progress(current_cg_index, cg_count)
                    else:
                        if progress.stop:
                            self.close()
                            raise Terminated

            # store channel groups record sizes dict and data block size in
            # each new group data belong to the initial unsorted group, and
            # add the key 'sorted' with the value False to use a flag;
            # this is used later if memory=False

            cg_size: dict[int, int] = {}
            total_size = 0

            for grp in new_groups:
                record_id = grp.channel_group.record_id
                cycles_nr = grp.channel_group.cycles_nr
                record_size = grp.channel_group.samples_byte_nr
                self._prepare_record(grp)

                cg_size[record_id] = record_size

                record_size += record_id_nr
                total_size += record_size * cycles_nr

                grp.record_size = cg_size

            for grp in new_groups:
                grp.data_location = v23c.LOCATION_ORIGINAL_FILE
                if total_size:
                    grp.data_blocks.append(
                        DataBlockInfo(
                            address=data_group.data_block_addr,
                            block_type=0,
                            original_size=total_size,
                            compressed_size=total_size,
                            param=0,
                        )
                    )

            self.groups.extend(new_groups)

            # go to next data group
            dg_addr = data_group.next_dg_addr

        # finally update the channel dependency references
        for grp in self.groups:
            for dep in grp.channel_dependencies:
                if dep:
                    for i in range(dep.sd_nr):
                        ref_channel_addr = typing.cast(int, dep[f"ch_{i}"])
                        channel = ch_map[ref_channel_addr]
                        dep.referenced_channels.append(channel)

    def _filter_occurrences(
        self,
        occurrences: Iterator[tuple[int, int]],
        source_name: str | None = None,
        source_path: str | None = None,
        acq_name: str | None = None,
    ) -> Iterator[tuple[int, int]]:
        if source_name is not None:
            occurrences = (
                (gp_idx, cn_idx)
                for gp_idx, cn_idx in occurrences
                if (source := self.groups[gp_idx].channels[cn_idx].source) is not None and source.name == source_name
            )

        if source_path is not None:
            occurrences = (
                (gp_idx, cn_idx)
                for gp_idx, cn_idx in occurrences
                if (source := self.groups[gp_idx].channels[cn_idx].source) is not None and source.path == source_path
            )

        return occurrences

    def add_trigger(
        self,
        group: int,
        timestamp: float,
        pre_time: float = 0,
        post_time: float = 0,
        comment: str = "",
    ) -> None:
        """Add trigger to data group.

        Parameters
        ----------
        group : int
            Group index.
        timestamp : float
            Trigger time.
        pre_time : float, default 0
            Trigger pre time.
        post_time : float, default 0
            Trigger post time.
        comment : str, optional
            Trigger comment.
        """
        comment_template = """<EVcomment>
    <TX>{}</TX>
</EVcomment>"""
        try:
            gp = self.groups[group]
        except IndexError:
            return

        trigger = gp.trigger

        if comment:
            try:
                comment_elem = ET.fromstring(comment)
                tx_elem = comment_elem.find(".//TX")
                if tx_elem is not None:
                    comment = tx_elem.text or ""
                else:
                    comment = ""
            except ET.ParseError:
                pass

        if trigger:
            count = trigger.trigger_events_nr
            trigger.trigger_events_nr += 1
            trigger.block_len += 24
            trigger[f"trigger_{count}_time"] = timestamp
            trigger[f"trigger_{count}_pretime"] = pre_time
            trigger[f"trigger_{count}_posttime"] = post_time
            if comment:
                if trigger.comment is None:
                    comment = f"{count + 1}. {comment}"
                    comment = comment_template.format(comment)
                    trigger.comment = comment
                else:
                    current_comment = trigger.comment
                    try:
                        comment_elem = ET.fromstring(current_comment)
                        tx_elem = comment_elem.find(".//TX")
                        if tx_elem is not None:
                            current_comment = tx_elem.text or ""
                        else:
                            current_comment = ""
                    except ET.ParseError:
                        pass

                    comment = f"{current_comment}\n{count + 1}. {comment}"
                    comment = comment_template.format(comment)
                    trigger.comment = comment
        else:
            trigger = TriggerBlock(  # type: ignore[call-arg]
                trigger_0_time=timestamp,
                trigger_0_pretime=pre_time,
                trigger_0_posttime=post_time,
            )
            if comment:
                comment = f"1. {comment}"
                comment = comment_template.format(comment)
                trigger.comment = comment

            gp.trigger = trigger

    @overload
    def append(
        self,
        signals: list[Signal] | Signal,
        acq_name: str | None = ...,
        acq_source: Source | None = ...,
        comment: str = ...,
        common_timebase: bool = ...,
        units: dict[str, str] | None = ...,
    ) -> int: ...

    @overload
    def append(
        self,
        signals: DataFrame,
        acq_name: str | None = ...,
        acq_source: Source | None = ...,
        comment: str = ...,
        common_timebase: bool = ...,
        units: dict[str, str] | None = ...,
    ) -> None: ...

    @overload
    def append(
        self,
        signals: list[Signal] | Signal | DataFrame,
        acq_name: str | None = ...,
        acq_source: Source | None = ...,
        comment: str = ...,
        common_timebase: bool = ...,
        units: dict[str, str] | None = ...,
    ) -> int | None: ...

    def append(
        self,
        signals: list[Signal] | Signal | DataFrame,
        acq_name: str | None = None,
        acq_source: Source | None = None,
        comment: str = "Python",
        common_timebase: bool = False,
        units: dict[str, str] | None = None,
    ) -> int | None:
        """Append a new data group.

        For channel dependencies type Signals, the `samples` attribute must be
        a np.recarray.

        Parameters
        ----------
        signals : list | Signal | pandas.DataFrame
            List of `Signal` objects, or a single `Signal` object, or a pandas
            DataFrame object. All bytes columns in the DataFrame must be
            *latin-1* encoded.
        acq_name : str, optional
            Channel group acquisition name.
        acq_source : Source, optional
            Channel group acquisition source.
        comment : str, default 'Python'
            Channel group comment.
        common_timebase : bool, default False
            Flag to hint that the signals have the same timebase. Only set this
            if you know for sure that all appended channels share the same time
            base.
        units : dict, optional
            Will contain the signal units mapped to the signal names when
            appending a pandas DataFrame.

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> import pandas as pd

        Case 1: Conversion type None.

        >>> s1 = np.array([1, 2, 3, 4, 5])
        >>> s2 = np.array([-1, -2, -3, -4, -5])
        >>> s3 = np.array([0.1, 0.04, 0.09, 0.16, 0.25])
        >>> t = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        >>> s1 = Signal(samples=s1, timestamps=t, unit='+', name='Positive')
        >>> s2 = Signal(samples=s2, timestamps=t, unit='-', name='Negative')
        >>> s3 = Signal(samples=s3, timestamps=t, unit='flts', name='Floats')
        >>> mdf = MDF(version='3.30')
        >>> mdf.append([s1, s2, s3], comment='created by asammdf')

        Case 2: VTAB conversions from channels inside another file.

        >>> mdf1 = MDF('in.mdf')
        >>> ch1 = mdf1.get("Channel1_VTAB")
        >>> ch2 = mdf1.get("Channel2_VTABR")
        >>> mdf2 = MDF('out.mdf')
        >>> mdf2.append([ch1, ch2], comment='created by asammdf')
        >>> df = pd.DataFrame.from_dict({'s1': np.array([1, 2, 3, 4, 5]), 's2': np.array([-1, -2, -3, -4, -5])})
        >>> units = {'s1': 'V', 's2': 'A'}
        >>> mdf2.append(df, units=units)
        """
        if isinstance(signals, Signal):
            signals = [signals]
        elif isinstance(signals, DataFrame):
            self._append_dataframe(signals, comment=comment, units=units)
            return None

        integer_interp_mode = self._integer_interpolation
        float_interp_mode = self._float_interpolation

        # check if the signals have a common timebase
        # if not interpolate the signals using the union of all timebases
        if signals:
            timestamps = signals[0].timestamps
            if not common_timebase:
                for signal in signals[1:]:
                    if not array_equal(signal.timestamps, timestamps):
                        different = True
                        break
                else:
                    different = False

                if different:
                    times = [s.timestamps for s in signals]
                    timestamps = unique(concatenate(times)).astype(float64)
                    signals = [
                        s.interp(
                            timestamps,
                            integer_interpolation_mode=integer_interp_mode,
                            float_interpolation_mode=float_interp_mode,
                        )
                        for s in signals
                    ]
                    del times
        else:
            timestamps = array([])

        if self.version >= "3.00":
            channel_size = v23c.CN_DISPLAYNAME_BLOCK_SIZE
        elif self.version >= "2.10":
            channel_size = v23c.CN_LONGNAME_BLOCK_SIZE
        else:
            channel_size = v23c.CN_SHORT_BLOCK_SIZE

        file = self._tempfile
        tell = file.tell

        ce_kargs: ChannelExtensionKwargs = {
            "module_nr": 0,
            "module_address": 0,
            "type": v23c.SOURCE_ECU,
            "description": b"Channel inserted by Python Script",
        }
        ce_block = ChannelExtension(**ce_kargs)

        canopen_time_fields = ("ms", "days")
        canopen_date_fields = (
            "ms",
            "min",
            "hour",
            "day",
            "month",
            "year",
            "summer_time",
            "day_of_week",
        )

        dg_cntr = len(self.groups)

        gp = Group(DataGroup())
        gp_channels = gp.channels = []
        gp_dep = gp.channel_dependencies = []
        gp_sig_types = gp.signal_types = []
        gp.string_dtypes = []
        record = gp.record = []

        self.groups.append(gp)

        cycles_nr = len(timestamps)
        fields: list[NDArray[Any]] = []
        types: list[DTypeLike | tuple[str, np.dtype[Any], tuple[int, ...]]] = []
        ch_cntr = 0
        offset = 0
        field_names = UniqueDB()

        if signals:
            master_metadata = signals[0].master_metadata
        else:
            master_metadata = None
        if master_metadata:
            time_name = master_metadata[0]
        else:
            time_name = "time"

        if signals:
            # conversion for time channel
            cc_kargs: ChannelConversionKwargs = {
                "conversion_type": v23c.CONVERSION_TYPE_NONE,
                "unit": b"s",
                "min_phy_value": timestamps[0] if cycles_nr else 0,
                "max_phy_value": timestamps[-1] if cycles_nr else 0,
            }
            cc_block = ChannelConversion(**cc_kargs)
            cc_block.unit = "s"
            new_source = ce_block

            # time channel
            t_type, t_size = fmt_to_datatype_v3(timestamps.dtype, timestamps.shape)
            cn_kargs: ChannelKwargs = {
                "short_name": time_name.encode("latin-1"),
                "channel_type": v23c.CHANNEL_TYPE_MASTER,
                "data_type": t_type,
                "start_offset": 0,
                "min_raw_value": timestamps[0] if cycles_nr else 0,
                "max_raw_value": timestamps[-1] if cycles_nr else 0,
                "bit_count": t_size,
                "block_len": channel_size,
            }
            channel = Channel(**cn_kargs)
            channel.name = name = time_name
            channel.conversion = cc_block
            channel.source = new_source

            gp_channels.append(channel)

            self.channels_db.add(name, (dg_cntr, ch_cntr))
            self.masters_db[dg_cntr] = 0

            # time channel doesn't have channel dependencies
            gp_dep.append(None)

            fields.append(timestamps)

            types.append((field_names.get_unique_name(name), timestamps.dtype))

            offset += t_size
            ch_cntr += 1

            gp_sig_types.append(0)
            record.append(
                (
                    timestamps.dtype,
                    timestamps.dtype.itemsize,
                    0,
                    0,
                )
            )

        for signal in signals:
            sig = signal
            names = sig.samples.dtype.names
            name = signal.name
            if names is None:
                sig_type = v23c.SIGNAL_TYPE_SCALAR
            else:
                if names in (canopen_time_fields, canopen_date_fields):
                    sig_type = v23c.SIGNAL_TYPE_CANOPEN
                elif names[0] != sig.name:
                    sig_type = v23c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v23c.SIGNAL_TYPE_ARRAY

            gp_sig_types.append(sig_type)

            # conversions for channel

            cc_block = conversion_transfer(signal.conversion)
            cc_block.unit = unit = signal.unit

            israw = signal.raw

            if not israw and not unit:
                conversion = None
            else:
                conversion = cc_block

            if sig_type == v23c.SIGNAL_TYPE_SCALAR:
                # source for channel
                if signal.source:
                    source = signal.source
                    if source.source_type != 2:
                        ce_kargs = {
                            "type": v23c.SOURCE_ECU,
                            "description": source.name.encode("latin-1"),
                            "ECU_identification": source.path.encode("latin-1"),
                        }
                    else:
                        ce_kargs = {
                            "type": v23c.SOURCE_VECTOR,
                            "message_name": source.name.encode("latin-1"),
                            "sender_name": source.path.encode("latin-1"),
                        }

                    new_source = ChannelExtension(**ce_kargs)

                else:
                    new_source = ce_block

                # compute additional byte offset for large records size
                if offset > v23c.MAX_UINT16:
                    additional_byte_offset = ceil((offset - v23c.MAX_UINT16) / 8)
                    start_bit_offset = offset - additional_byte_offset * 8

                else:
                    start_bit_offset = offset
                    additional_byte_offset = 0

                s_type, s_size = fmt_to_datatype_v3(signal.samples.dtype, signal.samples.shape)

                name = signal.name
                display_names = signal.display_names

                if signal.samples.dtype.kind == "u" and signal.bit_count <= 4:
                    s_size_ = signal.bit_count
                else:
                    s_size_ = s_size

                cn_kargs = {
                    "channel_type": v23c.CHANNEL_TYPE_VALUE,
                    "data_type": s_type,
                    "start_offset": start_bit_offset,
                    "bit_count": s_size_,
                    "additional_byte_offset": additional_byte_offset,
                    "block_len": channel_size,
                }

                s_size = max(s_size, 8)

                channel = Channel(**cn_kargs)
                channel.name = signal.name
                channel.comment = signal.comment
                channel.source = new_source
                channel.conversion = conversion
                channel.display_names = display_names
                gp_channels.append(channel)

                if len(signal.samples.shape) > 1:
                    dtype_fmt = np.dtype((signal.samples.dtype, signal.samples.shape[1:]))
                else:
                    dtype_fmt = signal.samples.dtype

                channel.dtype_fmt = dtype_fmt

                record.append(
                    (
                        dtype_fmt,
                        dtype_fmt.itemsize,
                        offset // 8,
                        0,
                    )
                )

                offset += s_size

                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in display_names:
                    self.channels_db.add(_name, entry)

                field_name = field_names.get_unique_name(name)

                if signal.samples.dtype.kind == "S":
                    gp.string_dtypes.append(signal.samples.dtype)

                fields.append(signal.samples)
                if s_type != v23c.DATA_TYPE_BYTEARRAY:
                    types.append((field_name, signal.samples.dtype))
                else:
                    types.append((field_name, signal.samples.dtype, signal.samples.shape[1:]))

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            # second, add the composed signals
            elif sig_type in (
                v23c.SIGNAL_TYPE_CANOPEN,
                v23c.SIGNAL_TYPE_STRUCTURE_COMPOSITION,
            ):
                new_dg_cntr = len(self.groups)
                new_gp = Group(DataGroup())
                new_gp_channels = new_gp.channels = []
                new_gp_dep = new_gp.channel_dependencies = []
                new_gp_sig_types = new_gp.signal_types = []
                new_record = new_gp.record = []
                self.groups.append(new_gp)

                new_fields: list[NDArray[Any]] = []
                new_types: list[DTypeLike | tuple[str, np.dtype[Any], tuple[int, ...]]] = []
                new_ch_cntr = 0
                new_offset = 0
                new_field_names = UniqueDB()

                # conversion for time channel
                cc_kargs = {
                    "conversion_type": v23c.CONVERSION_TYPE_NONE,
                    "unit": b"s",
                    "min_phy_value": timestamps[0] if cycles_nr else 0,
                    "max_phy_value": timestamps[-1] if cycles_nr else 0,
                }
                cc_block = ChannelConversion(**cc_kargs)
                cc_block.unit = "s"

                new_source = ce_block

                # time channel
                t_type, t_size = fmt_to_datatype_v3(timestamps.dtype, timestamps.shape)
                cn_kargs = {
                    "short_name": time_name.encode("latin-1"),
                    "channel_type": v23c.CHANNEL_TYPE_MASTER,
                    "data_type": t_type,
                    "start_offset": 0,
                    "min_raw_value": timestamps[0] if cycles_nr else 0,
                    "max_raw_value": timestamps[-1] if cycles_nr else 0,
                    "bit_count": t_size,
                    "block_len": channel_size,
                }
                channel = Channel(**cn_kargs)
                channel.name = name = time_name
                channel.source = new_source
                channel.conversion = cc_block
                new_gp_channels.append(channel)

                new_record.append(
                    (
                        timestamps.dtype,
                        timestamps.dtype.itemsize,
                        0,
                        0,
                    )
                )

                self.channels_db.add(name, (new_dg_cntr, new_ch_cntr))

                self.masters_db[new_dg_cntr] = 0

                # time channel doesn't have channel dependencies
                new_gp_dep.append(None)

                new_fields.append(timestamps)
                new_types.append((name, timestamps.dtype))
                new_field_names.get_unique_name(name)
                new_gp_sig_types.append(0)

                new_offset += t_size
                new_ch_cntr += 1

                names = signal.samples.dtype.names
                if names == ("ms", "days"):
                    channel_group_comment = "From mdf v4 CANopen Time channel"
                elif names == (
                    "ms",
                    "min",
                    "hour",
                    "day",
                    "month",
                    "year",
                    "summer_time",
                    "day_of_week",
                ):
                    channel_group_comment = "From mdf v4 CANopen Date channel"
                else:
                    channel_group_comment = "From mdf v4 structure channel composition"

                for name in names or ():
                    samples = signal.samples[name]

                    new_record.append(
                        (
                            samples.dtype,
                            samples.dtype.itemsize,
                            new_offset // 8,
                            0,
                        )
                    )

                    # conversions for channel

                    cc_kargs = {
                        "conversion_type": v23c.CONVERSION_TYPE_NONE,
                        "unit": signal.unit.encode("latin-1"),
                        "min_phy_value": 0,
                        "max_phy_value": 0,
                    }
                    cc_block = ChannelConversion(**cc_kargs)

                    # source for channel
                    if signal.source:
                        source = signal.source
                        if source.source_type != 2:
                            ce_kargs = {
                                "type": v23c.SOURCE_ECU,
                                "description": source.name.encode("latin-1"),
                                "ECU_identification": source.path.encode("latin-1"),
                            }
                        else:
                            ce_kargs = {
                                "type": v23c.SOURCE_VECTOR,
                                "message_name": source.name.encode("latin-1"),
                                "sender_name": source.path.encode("latin-1"),
                            }

                        new_source = ChannelExtension(**ce_kargs)

                    else:
                        new_source = ce_block

                    # compute additional byte offset for large records size
                    if new_offset > v23c.MAX_UINT16:
                        additional_byte_offset = ceil((new_offset - v23c.MAX_UINT16) / 8)
                        start_bit_offset = new_offset - additional_byte_offset * 8
                    else:
                        start_bit_offset = new_offset
                        additional_byte_offset = 0
                    s_type, s_size = fmt_to_datatype_v3(samples.dtype, samples.shape)

                    cn_kargs = {
                        "channel_type": v23c.CHANNEL_TYPE_VALUE,
                        "data_type": s_type,
                        "start_offset": start_bit_offset,
                        "bit_count": s_size,
                        "additional_byte_offset": additional_byte_offset,
                        "block_len": channel_size,
                    }

                    s_size = max(s_size, 8)

                    channel = Channel(**cn_kargs)
                    channel.name = name
                    channel.source = new_source
                    channel.conversion = cc_block

                    new_gp_channels.append(channel)
                    new_offset += s_size

                    self.channels_db.add(name, (new_dg_cntr, new_ch_cntr))

                    field_name = new_field_names.get_unique_name(name)

                    new_fields.append(samples)
                    new_types.append((field_name, samples.dtype))

                    new_ch_cntr += 1

                    # simple channels don't have channel dependencies
                    new_gp_dep.append(None)

                # channel group
                cg_kargs: ChannelGroupKwargs = {
                    "cycles_nr": cycles_nr,
                    "samples_byte_nr": new_offset // 8,
                    "ch_nr": new_ch_cntr,
                }
                new_gp.channel_group = ChannelGroup(**cg_kargs)
                new_gp.channel_group.comment = channel_group_comment

                # data group
                if self.version >= "3.20":
                    block_len = v23c.DG_POST_320_BLOCK_SIZE
                else:
                    block_len = v23c.DG_PRE_320_BLOCK_SIZE
                new_gp.data_group = DataGroup(block_len=block_len)

                # data block
                new_gp.sorted = True

                block: Buffer

                try:
                    samples = np.rec.fromarrays(new_fields, dtype=np.dtype(new_types))
                    block = samples.tobytes()
                except:
                    struct_fields: list[tuple[bytes | NDArray[Any], int]] = []
                    for samples in new_fields:
                        size = samples.dtype.itemsize

                        if len(samples.shape) > 1:
                            shape = samples.shape[1:]

                            for dim in shape:
                                size *= dim

                        if not samples.flags["C_CONTIGUOUS"]:
                            samples = ascontiguousarray(samples)

                        struct_fields.append((samples, size))

                    block = data_block_from_arrays(struct_fields, cycles_nr)

                new_gp.data_location = v23c.LOCATION_TEMPORARY_FILE
                if cycles_nr:
                    data_address = tell()
                    new_gp.data_group.data_block_addr = data_address
                    self._tempfile.write(block)
                    size = len(block)
                    new_gp.data_blocks.append(
                        DataBlockInfo(
                            address=data_address,
                            original_size=size,
                            compressed_size=size,
                            block_type=0,
                            param=0,
                        )
                    )
                else:
                    new_gp.data_group.data_block_addr = 0

                # data group trigger
                new_gp.trigger = None

            else:
                new_dg_cntr = len(self.groups)
                new_gp = Group(DataGroup())
                new_gp_channels = new_gp.channels = []
                new_gp_dep = new_gp.channel_dependencies = []
                new_gp_sig_types = new_gp.signal_types = []
                new_record = new_gp.record = []
                self.groups.append(new_gp)

                new_fields = []
                new_types = []
                new_ch_cntr = 0
                new_offset = 0
                new_field_names = UniqueDB()

                names = signal.samples.dtype.names
                name = signal.name

                component_names: list[str] = []
                component_samples: list[NDArray[Any]] = []
                if names:
                    samples = signal.samples[names[0]]
                else:
                    samples = signal.samples

                shape = samples.shape[1:]
                dims = [list(range(size)) for size in shape]

                for indexes in product(*dims):
                    subarray = samples
                    for idx in indexes:
                        subarray = subarray[:, idx]
                    component_samples.append(subarray)

                    indexes_str = "".join(f"[{idx}]" for idx in indexes)
                    component_name = f"{name}{indexes_str}"
                    component_names.append(component_name)

                # add channel dependency block for composed parent channel
                sd_nr = len(component_samples)
                cd_kargs: ChannelDependencyKwargs = {"sd_nr": sd_nr}
                for i, dim in enumerate(shape[::-1]):
                    cd_kargs[f"dim_{i}"] = dim  # type: ignore[literal-required]
                parent_dep = ChannelDependency(**cd_kargs)
                new_gp_dep.append(parent_dep)

                # source for channel
                if signal.source:
                    source = signal.source
                    if source.source_type != 2:
                        ce_kargs = {
                            "type": v23c.SOURCE_ECU,
                            "description": source.name.encode("latin-1"),
                            "ECU_identification": source.path.encode("latin-1"),
                        }
                    else:
                        ce_kargs = {
                            "type": v23c.SOURCE_VECTOR,
                            "message_name": source.name.encode("latin-1"),
                            "sender_name": source.path.encode("latin-1"),
                        }

                    new_source = ChannelExtension(**ce_kargs)

                else:
                    new_source = ce_block

                s_type, s_size = fmt_to_datatype_v3(samples.dtype, (), True)
                # compute additional byte offset for large records size
                if new_offset > v23c.MAX_UINT16:
                    additional_byte_offset = ceil((new_offset - v23c.MAX_UINT16) / 8)
                    start_bit_offset = new_offset - additional_byte_offset * 8
                else:
                    start_bit_offset = offset
                    additional_byte_offset = 0

                cn_kargs = {
                    "channel_type": v23c.CHANNEL_TYPE_VALUE,
                    "data_type": s_type,
                    "start_offset": start_bit_offset,
                    "bit_count": s_size,
                    "additional_byte_offset": additional_byte_offset,
                    "block_len": channel_size,
                }

                s_size = max(s_size, 8)

                new_record.append(None)

                channel = Channel(**cn_kargs)
                channel.comment = signal.comment
                channel.display_names = signal.display_names

                new_gp_channels.append(channel)

                self.channels_db.add(name, (new_dg_cntr, new_ch_cntr))

                new_ch_cntr += 1

                for i, (name, samples) in enumerate(zip(component_names, component_samples, strict=False)):
                    if i < sd_nr:
                        dep_pair = new_dg_cntr, new_ch_cntr
                        parent_dep.referenced_channels.append(dep_pair)
                        description = b"\0"
                    else:
                        description_str = f"{signal.name} - axis {name}"
                        description = description_str.encode("latin-1")

                    s_type, s_size = fmt_to_datatype_v3(samples.dtype, ())
                    shape = samples.shape[1:]

                    # source for channel
                    if signal.source:
                        source = signal.source
                        if source.source_type != 2:
                            ce_kargs = {
                                "type": v23c.SOURCE_ECU,
                                "description": source.name.encode("latin-1"),
                                "ECU_identification": source.path.encode("latin-1"),
                            }
                        else:
                            ce_kargs = {
                                "type": v23c.SOURCE_VECTOR,
                                "message_name": source.name.encode("latin-1"),
                                "sender_name": source.path.encode("latin-1"),
                            }

                        new_source = ChannelExtension(**ce_kargs)
                    else:
                        new_source = ce_block

                    # compute additional byte offset for large records size
                    if new_offset > v23c.MAX_UINT16:
                        additional_byte_offset = ceil((new_offset - v23c.MAX_UINT16) / 8)
                        start_bit_offset = new_offset - additional_byte_offset * 8
                    else:
                        start_bit_offset = new_offset
                        additional_byte_offset = 0

                    new_record.append(
                        (
                            samples.dtype,
                            samples.dtype.itemsize,
                            new_offset // 8,
                            0,
                        )
                    )

                    cn_kargs = {
                        "channel_type": v23c.CHANNEL_TYPE_VALUE,
                        "data_type": s_type,
                        "start_offset": start_bit_offset,
                        "bit_count": s_size,
                        "additional_byte_offset": additional_byte_offset,
                        "block_len": channel_size,
                        "description": description,
                    }

                    s_size = max(s_size, 8)

                    channel = Channel(**cn_kargs)
                    channel.name = name
                    channel.source = new_source
                    new_gp_channels.append(channel)

                    size = s_size
                    for dim in shape:
                        size *= dim
                    new_offset += size

                    self.channels_db.add(name, (new_dg_cntr, new_ch_cntr))

                    field_name = field_names.get_unique_name(name)

                    new_fields.append(samples)
                    new_types.append((field_name, samples.dtype, shape))

                    new_gp_dep.append(None)

                    ch_cntr += 1

                for name in names[1:] if names else ():
                    samples = signal.samples[name]

                    component_names = []
                    component_samples = []

                    shape = samples.shape[1:]
                    dims = [list(range(size)) for size in shape]

                    for indexes in product(*dims):
                        subarray = samples
                        for idx in indexes:
                            subarray = subarray[:, idx]
                        component_samples.append(subarray)

                        indexes_str = "".join(f"[{idx}]" for idx in indexes)
                        component_name = f"{name}{indexes_str}"
                        component_names.append(component_name)

                    # add channel dependency block for composed parent channel
                    sd_nr = len(component_samples)
                    cd_kargs = {"sd_nr": sd_nr}
                    for i, dim in enumerate(shape[::-1]):
                        cd_kargs[f"dim_{i}"] = dim  # type: ignore[literal-required]
                    parent_dep = ChannelDependency(**cd_kargs)
                    new_gp_dep.append(parent_dep)

                    # source for channel
                    if signal.source:
                        source = signal.source
                        if source.source_type != 2:
                            ce_kargs = {
                                "type": v23c.SOURCE_ECU,
                                "description": source.name.encode("latin-1"),
                                "ECU_identification": source.path.encode("latin-1"),
                            }
                        else:
                            ce_kargs = {
                                "type": v23c.SOURCE_VECTOR,
                                "message_name": source.name.encode("latin-1"),
                                "sender_name": source.path.encode("latin-1"),
                            }

                        new_source = ChannelExtension(**ce_kargs)

                    else:
                        new_source = ce_block

                    s_type, s_size = fmt_to_datatype_v3(samples.dtype, ())
                    # compute additional byte offset for large records size
                    if new_offset > v23c.MAX_UINT16:
                        additional_byte_offset = ceil((new_offset - v23c.MAX_UINT16) / 8)
                        start_bit_offset = new_offset - additional_byte_offset * 8
                    else:
                        start_bit_offset = new_offset
                        additional_byte_offset = 0

                    cn_kargs = {
                        "channel_type": v23c.CHANNEL_TYPE_VALUE,
                        "data_type": s_type,
                        "start_offset": start_bit_offset,
                        "bit_count": s_size,
                        "additional_byte_offset": additional_byte_offset,
                        "block_len": channel_size,
                    }

                    s_size = max(s_size, 8)

                    new_record.append(None)

                    channel = Channel(**cn_kargs)
                    channel.name = name
                    channel.comment = signal.comment
                    channel.source = new_source
                    new_gp_channels.append(channel)

                    self.channels_db.add(name, (new_dg_cntr, new_ch_cntr))

                    new_ch_cntr += 1

                    for i, (name, samples) in enumerate(zip(component_names, component_samples, strict=False)):
                        if i < sd_nr:
                            dep_pair = new_dg_cntr, new_ch_cntr
                            parent_dep.referenced_channels.append(dep_pair)
                            description = b"\0"
                        else:
                            description_str = f"{signal.name} - axis {name}"
                            description = description_str.encode("latin-1")

                        s_type, s_size = fmt_to_datatype_v3(samples.dtype, ())
                        shape = samples.shape[1:]

                        # source for channel
                        if signal.source:
                            source = signal.source
                            if source.source_type != 2:
                                ce_kargs = {
                                    "type": v23c.SOURCE_ECU,
                                    "description": source.name.encode("latin-1"),
                                    "ECU_identification": source.path.encode("latin-1"),
                                }
                            else:
                                ce_kargs = {
                                    "type": v23c.SOURCE_VECTOR,
                                    "message_name": source.name.encode("latin-1"),
                                    "sender_name": source.path.encode("latin-1"),
                                }

                            new_source = ChannelExtension(**ce_kargs)

                        else:
                            new_source = ce_block

                        # compute additional byte offset for large records size
                        if new_offset > v23c.MAX_UINT16:
                            additional_byte_offset = ceil((new_offset - v23c.MAX_UINT16) / 8)
                            start_bit_offset = new_offset - additional_byte_offset * 8
                        else:
                            start_bit_offset = new_offset
                            additional_byte_offset = 0

                        cn_kargs = {
                            "channel_type": v23c.CHANNEL_TYPE_VALUE,
                            "data_type": s_type,
                            "start_offset": start_bit_offset,
                            "bit_count": s_size,
                            "additional_byte_offset": additional_byte_offset,
                            "block_len": channel_size,
                            "description": description,
                        }

                        s_size = max(s_size, 8)

                        new_record.append(
                            (
                                samples.dtype,
                                samples.dtype.itemsize,
                                new_offset // 8,
                                0,
                            )
                        )

                        channel = Channel(**cn_kargs)
                        channel.name = name
                        channel.source = new_source
                        new_gp_channels.append(channel)

                        size = s_size
                        for dim in shape:
                            size *= dim
                        new_offset += size

                        self.channels_db.add(name, (new_dg_cntr, new_ch_cntr))

                        field_name = field_names.get_unique_name(name)

                        new_fields.append(samples)
                        new_types.append((field_name, samples.dtype, shape))

                        new_gp_dep.append(None)

                        ch_cntr += 1

                # channel group
                cg_kargs = {
                    "cycles_nr": cycles_nr,
                    "samples_byte_nr": new_offset // 8,
                    "ch_nr": new_ch_cntr,
                }
                new_gp.channel_group = ChannelGroup(**cg_kargs)
                new_gp.channel_group.comment = "From mdf v4 channel array"

                # data group
                if self.version >= "3.20":
                    block_len = v23c.DG_POST_320_BLOCK_SIZE
                else:
                    block_len = v23c.DG_PRE_320_BLOCK_SIZE
                new_gp.data_group = DataGroup(block_len=block_len)

                # data block
                new_gp.sorted = True

                samples = np.rec.fromarrays(new_fields, dtype=np.dtype(new_types))

                block = samples.tobytes()

                new_gp.data_location = v23c.LOCATION_TEMPORARY_FILE
                if cycles_nr:
                    data_address = tell()
                    new_gp.data_group.data_block_addr = data_address
                    self._tempfile.write(block)
                    size = len(block)
                    new_gp.data_blocks.append(
                        DataBlockInfo(
                            address=data_address,
                            original_size=size,
                            compressed_size=size,
                            block_type=0,
                            param=0,
                        )
                    )
                else:
                    new_gp.data_group.data_block_addr = 0

                # data group trigger
                new_gp.trigger = None

        # channel group
        cg_kargs = {
            "cycles_nr": cycles_nr,
            "samples_byte_nr": offset // 8,
            "ch_nr": ch_cntr,
        }
        if self.version >= "3.30":
            cg_kargs["block_len"] = v23c.CG_POST_330_BLOCK_SIZE
        else:
            cg_kargs["block_len"] = v23c.CG_PRE_330_BLOCK_SIZE
        gp.channel_group = ChannelGroup(**cg_kargs)
        gp.channel_group.comment = comment

        # data group
        if self.version >= "3.20":
            block_len = v23c.DG_POST_320_BLOCK_SIZE
        else:
            block_len = v23c.DG_PRE_320_BLOCK_SIZE
        gp.data_group = DataGroup(block_len=block_len)

        # data block
        gp.sorted = True

        if signals:
            samples = np.rec.fromarrays(fields, dtype=np.dtype(types))
        else:
            samples = array([])

        block = samples.tobytes()

        self._tempfile.seek(0, 2)

        gp.data_location = v23c.LOCATION_TEMPORARY_FILE
        if cycles_nr:
            data_address = tell()
            gp.data_group.data_block_addr = data_address
            size = len(block)
            self._tempfile.write(block)

            gp.data_blocks.append(
                DataBlockInfo(
                    address=data_address,
                    block_type=0,
                    original_size=size,
                    compressed_size=size,
                    param=0,
                )
            )

        self.virtual_groups_map[dg_cntr] = dg_cntr
        if dg_cntr not in self.virtual_groups:
            self.virtual_groups[dg_cntr] = VirtualChannelGroup()

        virtual_channel_group = self.virtual_groups[dg_cntr]
        virtual_channel_group.groups.append(dg_cntr)
        virtual_channel_group.record_size = gp.channel_group.samples_byte_nr
        virtual_channel_group.cycles_nr = gp.channel_group.cycles_nr

        # data group trigger
        gp.trigger = None

        return dg_cntr

    def _append_dataframe(
        self,
        df: DataFrame,
        comment: str = "",
        units: dict[str, str] | None = None,
    ) -> None:
        """Append a new data group from a pandas DataFrame."""
        units = units or {}

        t = df.index.values
        time_name = df.index.name if isinstance(df.index.name, str) and df.index.name else "time"

        version = self.version

        timestamps = t

        if self.version >= "3.00":
            channel_size = v23c.CN_DISPLAYNAME_BLOCK_SIZE
        elif self.version >= "2.10":
            channel_size = v23c.CN_LONGNAME_BLOCK_SIZE
        else:
            channel_size = v23c.CN_SHORT_BLOCK_SIZE

        file = self._tempfile
        tell = file.tell

        ce_kargs: ChannelExtensionKwargs = {
            "module_nr": 0,
            "module_address": 0,
            "type": v23c.SOURCE_ECU,
            "description": b"Channel inserted by Python Script",
        }
        ce_block = ChannelExtension(**ce_kargs)

        dg_cntr = len(self.groups)

        gp = Group(DataGroup())
        gp_channels = gp.channels = []
        gp_dep = gp.channel_dependencies = []
        gp_sig_types = gp.signal_types = []
        gp.string_dtypes = []
        record = gp.record = []

        self.groups.append(gp)

        cycles_nr = len(timestamps)
        fields: list[ArrayLike] = []
        types: list[DTypeLike] = []
        ch_cntr = 0
        offset = 0
        field_names = UniqueDB()

        if df.shape[0]:
            # conversion for time channel
            cc_kargs: ChannelConversionKwargs = {
                "conversion_type": v23c.CONVERSION_TYPE_NONE,
                "unit": b"s",
                "min_phy_value": typing.cast(float, timestamps[0]) if cycles_nr else 0,
                "max_phy_value": typing.cast(float, timestamps[-1]) if cycles_nr else 0,
            }
            conversion = ChannelConversion(**cc_kargs)
            conversion.unit = "s"
            source = ce_block

            # time channel
            t_type, t_size = fmt_to_datatype_v3(timestamps.dtype, timestamps.shape)
            cn_kargs: ChannelKwargs = {
                "short_name": time_name.encode("latin-1"),
                "channel_type": v23c.CHANNEL_TYPE_MASTER,
                "data_type": t_type,
                "start_offset": 0,
                "min_raw_value": typing.cast(float, timestamps[0]) if cycles_nr else 0,
                "max_raw_value": typing.cast(float, timestamps[-1]) if cycles_nr else 0,
                "bit_count": t_size,
                "block_len": channel_size,
            }
            channel = Channel(**cn_kargs)
            channel.name = name = time_name
            channel.conversion = conversion
            channel.source = source

            gp_channels.append(channel)

            self.channels_db.add(name, (dg_cntr, ch_cntr))
            self.masters_db[dg_cntr] = 0

            record.append(
                (
                    timestamps.dtype,
                    timestamps.dtype.itemsize,
                    0,
                    0,
                )
            )

            # time channel doesn't have channel dependencies
            gp_dep.append(None)

            fields.append(timestamps)
            types.append((field_names.get_unique_name(name), timestamps.dtype))

            offset += t_size
            ch_cntr += 1

            gp_sig_types.append(0)

        for signal in df.columns:
            sig = df[signal]
            name = signal

            sig_type = v23c.SIGNAL_TYPE_SCALAR

            gp_sig_types.append(sig_type)

            new_source = ce_block

            # compute additional byte offset for large records size
            if offset > v23c.MAX_UINT16:
                additional_byte_offset = ceil((offset - v23c.MAX_UINT16) / 8)
                start_bit_offset = offset - additional_byte_offset * 8
            else:
                start_bit_offset = offset
                additional_byte_offset = 0

            s_type, s_size = fmt_to_datatype_v3(sig.dtype, sig.shape)

            cn_kwargs: ChannelKwargs = {
                "channel_type": v23c.CHANNEL_TYPE_VALUE,
                "data_type": s_type,
                "min_raw_value": 0,
                "max_raw_value": 0,
                "start_offset": start_bit_offset,
                "bit_count": s_size,
                "additional_byte_offset": additional_byte_offset,
                "block_len": channel_size,
            }

            s_size = max(s_size, 8)

            channel = Channel(**cn_kwargs)
            channel.name = name
            channel.source = new_source
            channel.dtype_fmt = np.dtype((sig.dtype, sig.shape[1:]))

            record.append(
                (
                    channel.dtype_fmt,
                    channel.dtype_fmt.itemsize,
                    offset // 8,
                    0,
                )
            )

            unit = units.get(name, "")
            if unit:
                # conversion for time channel
                cc_kwargs: ChannelConversionKwargs = {
                    "conversion_type": v23c.CONVERSION_TYPE_NONE,
                    "unit": unit.encode(encoding="latin-1"),
                    "min_phy_value": 0,
                    "max_phy_value": 0,
                }
                conversion = ChannelConversion(**cc_kwargs)
                conversion.unit = unit

            gp_channels.append(channel)

            offset += s_size

            self.channels_db.add(name, (dg_cntr, ch_cntr))

            field_name = field_names.get_unique_name(name)

            if sig.dtype.kind == "S":
                dtype = typing.cast(np.dtype[np.bytes_], sig.dtype)
                gp.string_dtypes.append(dtype)

            fields.append(sig)
            types.append((field_name, sig.dtype))

            ch_cntr += 1

            # simple channels don't have channel dependencies
            gp_dep.append(None)

        # channel group
        cg_kwargs: ChannelGroupKwargs = {
            "cycles_nr": cycles_nr,
            "samples_byte_nr": offset // 8,
            "ch_nr": ch_cntr,
        }
        if self.version >= "3.30":
            cg_kwargs["block_len"] = v23c.CG_POST_330_BLOCK_SIZE
        else:
            cg_kwargs["block_len"] = v23c.CG_PRE_330_BLOCK_SIZE
        gp.channel_group = ChannelGroup(**cg_kwargs)
        gp.channel_group.comment = comment

        # data group
        if self.version >= "3.20":
            block_len = v23c.DG_POST_320_BLOCK_SIZE
        else:
            block_len = v23c.DG_PRE_320_BLOCK_SIZE
        gp.data_group = DataGroup(block_len=block_len)

        # data block
        gp.sorted = True

        samples: NDArray[Any]
        if df.shape[0]:
            samples = np.rec.fromarrays(fields, dtype=np.dtype(types))
        else:
            samples = array([])

        block = samples.tobytes()

        gp.data_location = v23c.LOCATION_TEMPORARY_FILE
        if cycles_nr:
            data_address = tell()
            gp.data_group.data_block_addr = data_address
            size = len(block)
            self._tempfile.write(block)

            gp.data_blocks.append(
                DataBlockInfo(
                    address=data_address,
                    block_type=0,
                    original_size=size,
                    compressed_size=size,
                    param=0,
                )
            )
        else:
            gp.data_location = v23c.LOCATION_TEMPORARY_FILE

        self.virtual_groups_map[dg_cntr] = dg_cntr
        if dg_cntr not in self.virtual_groups:
            self.virtual_groups[dg_cntr] = VirtualChannelGroup()

        virtual_channel_group = self.virtual_groups[dg_cntr]
        virtual_channel_group.groups.append(dg_cntr)
        virtual_channel_group.record_size = gp.channel_group.samples_byte_nr
        virtual_channel_group.cycles_nr = gp.channel_group.cycles_nr

        # data group trigger
        gp.trigger = None

    def close(self) -> None:
        """Call this just before the object is not used anymore to clean up the
        temporary file and close the file object.
        """
        try:
            if self._closed:
                return
            else:
                self._closed = True

            self._parent = None
            if self._tempfile is not None:
                self._tempfile.close()
            if self._file is not None and not self._from_filelike:
                self._file.close()

            if self._mapped_file is not None:
                self._mapped_file.close()

            if self._delete_on_close:
                try:
                    Path(self.name).unlink()
                except:
                    pass

            if self.original_name is not None:
                if Path(self.original_name).suffix.lower() in (
                    ".bz2",
                    ".gzip",
                    ".mf4z",
                    ".zip",
                ):
                    try:
                        os.remove(self.name)
                    except:
                        pass

            self._call_back = None
            self.groups.clear()
            self.channels_db.clear()
            self.masters_db.clear()
            self._master_channel_metadata.clear()
            self._si_map.clear()
            self._cc_map.clear()
        except:
            print(format_exc())

    def extend(self, index: int, signals: Sequence[tuple[NDArray[Any], NDArray[np.bool] | None]]) -> None:
        """Extend a group with new samples.

        `signals` contains (values, invalidation_bits) pairs for each extended
        signal. Since MDF3 does not support invalidation bits, the second item
        of each pair must be None. The first pair is the master channel's pair,
        and the next pairs must respect the same order in which the signals
        were appended. The samples must have raw or physical values according
        to the signals used for the initial append.

        Parameters
        ----------
        index : int
            Group index.
        signals : sequence
            Sequence of (np.ndarray, None) tuples.

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> s1 = np.array([1, 2, 3, 4, 5])
        >>> s2 = np.array([-1, -2, -3, -4, -5])
        >>> s3 = np.array([0.1, 0.04, 0.09, 0.16, 0.25])
        >>> t = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        >>> s1 = Signal(samples=s1, timestamps=t, unit='+', name='Positive')
        >>> s2 = Signal(samples=s2, timestamps=t, unit='-', name='Negative')
        >>> s3 = Signal(samples=s3, timestamps=t, unit='flts', name='Floats')
        >>> mdf = MDF(version='3.30')
        >>> mdf.append([s1, s2, s3], comment='created by asammdf')
        >>> t = np.array([0.006, 0.007, 0.008, 0.009, 0.010])
        >>> mdf.extend(0, [(t, None), (s1.samples, None), (s2.samples, None), (s3.samples, None)])
        """
        new_group_offset = 0
        gp = self.groups[index]
        if not signals:
            message = '"append" requires a non-empty list of Signal objects'
            raise MdfException(message)

        stream: FileLike | mmap.mmap | IO[bytes]
        if gp.data_location == v23c.LOCATION_ORIGINAL_FILE:
            if self._file is None:
                raise ValueError("self._file cannot be None")
            stream = self._file
        else:
            stream = self._tempfile

        canopen_time_fields = ("ms", "days")
        canopen_date_fields = (
            "ms",
            "min",
            "hour",
            "day",
            "month",
            "year",
            "summer_time",
            "day_of_week",
        )

        fields: list[NDArray[Any]] = []
        types: list[DTypeLike | tuple[str, np.dtype[Any], tuple[int, ...]]] = []
        samples: NDArray[Any]

        cycles_nr = len(signals[0][0])
        string_counter = 0

        for k_i, ((signal, invalidation_bits), sig_type) in enumerate(zip(signals, gp.signal_types, strict=False)):
            sig = signal
            names = sig.dtype.names

            if sig_type == v23c.SIGNAL_TYPE_SCALAR:
                if signal.dtype.kind == "S":
                    str_dtype = gp.string_dtypes[string_counter]
                    signal = signal.astype(str_dtype)
                    string_counter += 1

                fields.append(signal)

                if signal.shape[1:]:
                    types.append(("", signal.dtype, signal.shape[1:]))
                else:
                    types.append(("", signal.dtype))

            # second, add the composed signals
            elif sig_type in (
                v23c.SIGNAL_TYPE_CANOPEN,
                v23c.SIGNAL_TYPE_STRUCTURE_COMPOSITION,
            ):
                new_group_offset += 1
                new_gp = self.groups[index + new_group_offset]

                new_fields: list[NDArray[Any]] = []
                new_types: list[DTypeLike] = []

                names = signal.dtype.names
                for name in names or ():
                    new_fields.append(signal[name])
                    new_types.append(("", signal.dtype))

                # data block
                samples = np.rec.fromarrays(new_fields, dtype=np.dtype(new_types))
                data = samples.tobytes()

                record_size = new_gp.channel_group.samples_byte_nr
                extended_size = cycles_nr * record_size

                if data:
                    stream.seek(0, 2)
                    data_address = stream.tell()
                    stream.write(data)

                    new_gp.data_blocks.append(
                        DataBlockInfo(
                            address=data_address,
                            original_size=extended_size,
                            compressed_size=extended_size,
                            block_type=0,
                            param=0,
                        )
                    )

            else:
                names = signal.dtype.names

                component_samples: list[NDArray[Any]] = []
                if names:
                    samples = signal[names[0]]
                else:
                    samples = signal

                shape = samples.shape[1:]
                dims = [list(range(size)) for size in shape]

                for indexes in product(*dims):
                    subarray = samples
                    for idx in indexes:
                        subarray = subarray[:, idx]
                    component_samples.append(subarray)

                if names:
                    new_samples = [signal[fld] for fld in names[1:]]
                    component_samples.extend(new_samples)

                for samples in component_samples:
                    shape = samples.shape[1:]

                    fields.append(samples)
                    types.append(("", samples.dtype, shape))

        record_size = gp.channel_group.samples_byte_nr
        extended_size = cycles_nr * record_size

        # data block
        samples = np.rec.fromarrays(fields, dtype=np.dtype(types))
        data = samples.tobytes()

        if cycles_nr:
            stream.seek(0, 2)
            data_address = stream.tell()
            stream.write(data)
            gp.channel_group.cycles_nr += cycles_nr

            gp.data_blocks.append(
                DataBlockInfo(
                    address=data_address,
                    block_type=0,
                    original_size=extended_size,
                    compressed_size=extended_size,
                    param=0,
                )
            )

        virtual_channel_group = self.virtual_groups[index]
        virtual_channel_group.cycles_nr += cycles_nr

    def get_channel_name(self, group: int, index: int) -> str:
        """Get channel name.

        Parameters
        ----------
        group : int
            0-based group index.
        index : int
            0-based channel index.

        Returns
        -------
        name : str
            Found channel name.
        """
        gp_nr, ch_nr = self._validate_channel_selection(None, group, index)

        grp = self.groups[gp_nr]
        channel = grp.channels[ch_nr]

        return channel.name

    def get_channel_metadata(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
    ) -> Channel:
        gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]

        channel = grp.channels[ch_nr]
        channel = deepcopy(channel)

        return channel

    def get_channel_unit(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
    ) -> str:
        """Get channel unit.

        The channel can be specified in two ways:

        * Using the first positional argument `name`.

          * If there are multiple occurrences for this channel, then the `group`
            and `index` arguments can be used to select a specific group.
          * If there are multiple occurrences for this channel and either the
            `group` or `index` arguments is None, then a warning is issued.

        * Using the group number (keyword argument `group`) and the channel
          number (keyword argument `index`). Use `info` method for group and
          channel numbers.

        Parameters
        ----------
        name : str, optional
            Name of channel.
        group : int, optional
            0-based group index.
        index : int, optional
            0-based channel index.

        Returns
        -------
        unit : str
            Found channel unit.
        """
        gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]
        channel = grp.channels[ch_nr]

        if channel.conversion:
            unit = channel.conversion.unit
        else:
            unit = ""

        return unit

    def get_channel_comment(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
    ) -> str:
        """Get channel comment.

        The channel can be specified in two ways:

        * Using the first positional argument `name`.

          * If there are multiple occurrences for this channel, then the `group`
            and `index` arguments can be used to select a specific group.
          * If there are multiple occurrences for this channel and either the
            `group` or `index` arguments is None, then a warning is issued.

        * Using the group number (keyword argument `group`) and the channel
          number (keyword argument `index`). Use `info` method for group and
          channel numbers.

        Parameters
        ----------
        name : str, optional
            Name of channel.
        group : int, optional
            0-based group index.
        index : int, optional
            0-based channel index.

        Returns
        -------
        comment : str
            Found channel comment.
        """
        gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]
        channel = grp.channels[ch_nr]

        return channel.comment

    @overload
    def get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: RasterType | None = ...,
        samples_only: Literal[False] = ...,
        data: tuple[bytes, int, int | None] | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = ...,
        record_count: int | None = ...,
        skip_channel_validation: bool = ...,
    ) -> Signal: ...

    @overload
    def get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: RasterType | None = ...,
        *,
        samples_only: Literal[True],
        data: tuple[bytes, int, int | None] | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = ...,
        record_count: int | None = ...,
        skip_channel_validation: bool = ...,
    ) -> tuple[NDArray[Any], None]: ...

    @overload
    def get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: RasterType | None = ...,
        samples_only: bool = ...,
        data: tuple[bytes, int, int | None] | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = ...,
        record_count: int | None = ...,
        skip_channel_validation: bool = ...,
    ) -> Signal | tuple[NDArray[Any], None]: ...

    def get(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
        raster: RasterType | None = None,
        samples_only: bool = False,
        data: tuple[bytes, int, int | None] | None = None,
        raw: bool = False,
        ignore_invalidation_bits: bool = False,
        record_offset: int = 0,
        record_count: int | None = None,
        skip_channel_validation: bool = False,
    ) -> Signal | tuple[NDArray[Any], None]:
        """Get channel samples.

        The channel can be specified in two ways:

        * Using the first positional argument `name`.

          * If there are multiple occurrences for this channel, then the `group`
            and `index` arguments can be used to select a specific group.
          * If there are multiple occurrences for this channel and either the
            `group` or `index` arguments is None, then a warning is issued.

        * Using the group number (keyword argument `group`) and the channel
          number (keyword argument `index`). Use `info` method for group and
          channel numbers.

        If the `raster` keyword argument is not None, the output is interpolated
        accordingly.

        Parameters
        ----------
        name : str, optional
            Name of channel.
        group : int, optional
            0-based group index.
        index : int, optional
            0-based channel index.
        raster : float, optional
            Time raster in seconds.
        samples_only : bool, default False
            If True, return only the channel samples as np.ndarray; if False,
            return a `Signal` object.
        data : bytes, optional
            Prevent redundant data read by providing the raw data group samples.
        raw : bool, default False
            Return channel samples without applying the conversion rule.
        ignore_invalidation_bits : bool, default False
            Only defined to have the same API with the MDF v4.
        record_offset : int, optional
            If `data=None`, use this to select the record offset from which the
            group data should be loaded.
        record_count : int, optional
            Number of records to read; default is None and in this case all
            available records are used.
        skip_channel_validation : bool, default False
            Skip validation of channel name, group index and channel index. If
            True, the caller has to make sure that the `group` and `index`
            arguments are provided and are correct.

            .. versionadded:: 7.0.0

        Returns
        -------
        res : (np.ndarray, None) | Signal
            Returns `Signal` if `samples_only=False` (default option),
            otherwise returns a (np.ndarray, None) tuple (for compatibility
            with MDF v4 class).

            The `Signal` samples are:

            * np.recarray for channels that have CDBLOCK or BYTEARRAY
              type channels
            * np.ndarray for all the rest

        Raises
        ------
        MdfException
            * if the channel name is not found
            * if the group index is out of range
            * if the channel index is out of range
            * if there are multiple channel occurrences in the file and the
              arguments `name`, `group`, `index` are ambiguous. This behaviour
              can be turned off by setting `raise_on_multiple_occurrences` to
              False.

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> t = np.arange(5)
        >>> s = np.ones(5)
        >>> mdf = MDF(version='3.30')
        >>> for i in range(4):
        ...     sigs = [Signal(s * (i * 10 + j), t, name='Sig') for j in range(1, 4)]
        ...     mdf.append(sigs)

        Specifying only the channel name is not enough when there are multiple
        channels with that name.

        >>> mdf.get('Sig')
        MdfException: Multiple occurrences for channel "Sig": ((0, 1), (0, 2),
        (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2),
        (3, 3)). Provide both "group" and "index" arguments to select another
        data group

        In this case, adding the group number is also not enough since there
        are multiple channels with that name in that group.

        >>> mdf.get('Sig', 1)
        MdfException: Multiple occurrences for channel "Sig": ((1, 1), (1, 2),
        (1, 3)). Provide both "group" and "index" arguments to select another
        data group

        Get the channel named "Sig" from group 1, channel index 2.

        >>> mdf.get('Sig', 1, 2)
        <Signal Sig:
                samples=[ 12.  12.  12.  12.  12.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">

        Get the channel from group 2 and channel index 1.

        >>> mdf.get(None, 2, 1)
        <Signal Sig:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">
        >>> mdf.get(group=2, index=1)
        <Signal Sig:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">
        """

        if skip_channel_validation:
            if group is None or index is None:
                raise ValueError("'group' or 'index' cannot be None if 'skip_channel_validation' is True")
            gp_nr, ch_nr = group, index
        else:
            gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        original_data = data

        grp = self.groups[gp_nr]

        channel = grp.channels[ch_nr]

        conversion = channel.conversion
        name = channel.name
        display_names = channel.display_names

        bit_count = channel.bit_count or 64

        dep = grp.channel_dependencies[ch_nr]
        cycles_nr = grp.channel_group.cycles_nr

        encoding = "latin-1"

        # get data group record
        data_: Iterable[tuple[bytes, int, int | None]]
        if data is None:
            data_ = self._load_data(grp, record_offset=record_offset, record_count=record_count)
        else:
            data_ = (data,)

        # check if this is a channel array
        vals: NDArray[Any]
        if dep:
            if dep.dependency_type == v23c.DEPENDENCY_TYPE_VECTOR:
                arrays: list[NDArray[Any]] = []
                types: list[DTypeLike] = []

                for dg_nr, ch_nr in dep.referenced_channels:
                    sig = self.get(
                        group=dg_nr,
                        index=ch_nr,
                        raw=raw,
                        data=original_data,
                        record_offset=record_offset,
                        record_count=record_count,
                    )

                    arrays.append(sig.samples)
                    types.append((sig.name, sig.samples.dtype))

                vals = np.rec.fromarrays(arrays, dtype=types)

            elif dep.dependency_type >= v23c.DEPENDENCY_TYPE_NDIM:
                shape: list[int] = []
                i = 0
                while True:
                    try:
                        dim = typing.cast(int, dep[f"dim_{i}"])
                        shape.append(dim)
                        i += 1
                    except (KeyError, AttributeError):
                        break
                shape = shape[::-1]

                record_shape = tuple(shape)

                arrays = [
                    self.get(
                        group=dg_nr,
                        index=ch_nr,
                        samples_only=True,
                        raw=raw,
                        data=original_data,
                        record_offset=record_offset,
                        record_count=record_count,
                    )[0]
                    for dg_nr, ch_nr in dep.referenced_channels
                ]
                shape.insert(0, cycles_nr)

                vals = column_stack(arrays).flatten().reshape(tuple(shape))

                arrays = [vals]

                vals = np.rec.fromarrays(arrays, dtype=np.dtype([(channel.name, vals.dtype, record_shape)]))

            else:
                raise ValueError(f"unexpected dependency_type '{dep.dependency_type}'")

            if not samples_only or raster:
                timestamps = self.get_master(
                    gp_nr,
                    original_data,
                    record_offset=record_offset,
                    record_count=record_count,
                )
                if raster and len(timestamps) > 1:
                    num = float(float32((timestamps[-1] - timestamps[0]) / raster))
                    if num.is_integer():
                        t = linspace(timestamps[0], timestamps[-1], int(num))
                    else:
                        t = arange(timestamps[0], timestamps[-1], raster)

                    vals = (
                        Signal(vals, timestamps, name="_")
                        .interp(
                            t,
                            integer_interpolation_mode=self._integer_interpolation,
                            float_interpolation_mode=self._float_interpolation,
                        )
                        .samples
                    )

                    timestamps = t

        else:
            # get channel values
            channel_values: list[NDArray[Any]] = []
            times: list[NDArray[Any]] = []
            count = 0
            records = self._prepare_record(grp)
            for fragment in data_:
                data_bytes, _offset, _count = fragment
                info = records[ch_nr]

                bits = channel.bit_count

                if info is not None:
                    dtype_, byte_size, byte_offset, bit_offset = info

                    buffer = get_channel_raw_bytes(
                        data_bytes,
                        grp.channel_group.samples_byte_nr,
                        byte_offset,
                        byte_size,
                    )

                    vals = frombuffer(buffer, dtype=dtype_)
                    data_type = channel.data_type
                    size = byte_size

                    vals_dtype = vals.dtype.kind
                    if vals_dtype not in "ui" and (bit_offset or bits != size * 8):
                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)
                    else:
                        dtype_ = vals.dtype
                        kind_ = dtype_.kind

                        if data_type in v23c.INT_TYPES:
                            dtype_fmt = get_fmt_v3(data_type, bits, self.identification.byte_order)
                            channel_dtype: np.dtype[Any] = np.dtype(dtype_fmt.split(")")[-1])

                            if channel_dtype.byteorder == "=" and data_type in (
                                v23c.DATA_TYPE_SIGNED_MOTOROLA,
                                v23c.DATA_TYPE_UNSIGNED_MOTOROLA,
                            ):
                                view = f">u{vals.itemsize}"
                            else:
                                view = f"{channel_dtype.byteorder}u{vals.itemsize}"

                            vals = vals.view(view)

                            if bit_offset:
                                vals = vals >> bit_offset

                            if bits != size * 8:
                                if data_type in v23c.SIGNED_INT:
                                    vals = as_non_byte_sized_signed_int(vals, bits)
                                else:
                                    mask = (1 << bits) - 1
                                    vals = vals & mask
                            elif data_type in v23c.SIGNED_INT:
                                view = f"{channel_dtype.byteorder}i{vals.itemsize}"
                                vals = vals.view(view)
                        else:
                            if bits != size * 8:
                                vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                else:
                    vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                if not samples_only or raster:
                    times.append(self.get_master(gp_nr, fragment))

                if bits == 1 and self._single_bit_uint_as_bool:
                    vals = array(vals, dtype=bool)

                channel_values.append(vals)
                count += 1

            if count > 1:
                vals = concatenate(channel_values)
            elif count == 1:
                vals = channel_values[0]
            else:
                vals = np.array([])

            if not samples_only or raster:
                if count > 1:
                    timestamps = concatenate(times)
                else:
                    timestamps = times[0]

                if raster and len(timestamps) > 1:
                    num = float(float32((timestamps[-1] - timestamps[0]) / raster))
                    if num.is_integer():
                        t = linspace(timestamps[0], timestamps[-1], int(num))
                    else:
                        t = arange(timestamps[0], timestamps[-1], raster)

                    vals = (
                        Signal(vals, timestamps, name="_")
                        .interp(
                            t,
                            integer_interpolation_mode=self._integer_interpolation,
                            float_interpolation_mode=self._float_interpolation,
                        )
                        .samples
                    )

                    timestamps = t

            if not raw:
                if conversion:
                    vals = conversion.convert(vals)
                    conversion = None

        if vals.dtype.kind == "S":
            encoding = "latin-1"
            vals = array([e.rsplit(b"\0")[0] for e in typing.cast(list[bytes], vals.tolist())], dtype=vals.dtype)

        res: tuple[NDArray[Any], None] | Signal
        if samples_only:
            res = vals, None
        else:
            if channel.conversion:
                unit = channel.conversion.unit
            else:
                unit = ""

            comment = channel.comment

            description = channel.description.decode("latin-1").strip(" \t\n\0")
            if comment:
                comment = f"{comment}\n{description}"
            else:
                comment = description

            if channel.source:
                source = Source.from_source(channel.source)
            else:
                source = None

            master_metadata = self._master_channel_metadata.get(gp_nr, None)

            res = Signal(
                samples=vals,
                timestamps=timestamps,
                unit=unit,
                name=channel.name,
                comment=comment,
                conversion=conversion,
                raw=raw,
                master_metadata=master_metadata,
                display_names=display_names,
                source=source,
                bit_count=bit_count,
                encoding=encoding,
                group_index=gp_nr,
                channel_index=ch_nr,
            )

        return res

    def get_master(
        self,
        index: int,
        data: tuple[bytes, int, int | None] | None = None,
        record_offset: int = 0,
        record_count: int | None = None,
        one_piece: bool = False,
    ) -> NDArray[Any]:
        """Get master channel samples for the given group.

        Parameters
        ----------
        index : int
            Group index.
        data : (bytes, int), optional
            (data block raw bytes, fragment offset).
        record_offset : int, optional
            If `data=None`, use this to select the record offset from which the
            group data should be loaded.
        record_count : int, optional
            Number of records to read; default is None and in this case all
            available records are used.

        Returns
        -------
        t : np.ndarray
            Master channel samples.
        """
        if self._master is not None:
            return self._master

        fragment = data
        if fragment:
            data_bytes, offset, _count = fragment
        else:
            offset = 0

        group = self.groups[index]

        time_ch_nr = self.masters_db.get(index, None)
        cycles_nr = group.channel_group.cycles_nr

        t: NDArray[Any]
        metadata: tuple[str, Literal[1]]

        if time_ch_nr is None:
            if fragment:
                count = len(data_bytes) // group.channel_group.samples_byte_nr
            else:
                count = cycles_nr
            t = arange(count, dtype=float64)
            metadata = ("time", 1)
        else:
            time_ch = group.channels[time_ch_nr]

            metadata = (time_ch.name, 1)

            if time_ch.bit_count == 0:
                if time_ch.sampling_rate:
                    sampling_rate = time_ch.sampling_rate
                else:
                    sampling_rate = 1
                t = arange(cycles_nr, dtype=float64) * sampling_rate
            else:
                # get data group record
                data_: Iterable[tuple[bytes, int, int | None]]
                if data is None:
                    data_ = self._load_data(group, record_offset=record_offset, record_count=record_count)
                    _count = record_count
                else:
                    data_ = (data,)

                records = self._prepare_record(group)
                record = records[time_ch_nr]

                if record is None:
                    raise ValueError("record is None")

                time_values: list[NDArray[Any]] = []
                count = 0
                for fragment in data_:
                    data_bytes, offset, _count = fragment
                    dtype_, byte_size, byte_offset, bit_offset = record

                    buffer = get_channel_raw_bytes(
                        data_bytes,
                        group.channel_group.samples_byte_nr,
                        byte_offset,
                        byte_size,
                    )

                    t = frombuffer(buffer, dtype=dtype_)

                    time_values.append(t)
                    count += 1

                if count > 1:
                    t = concatenate(time_values)
                elif count == 1:
                    t = time_values[0]
                else:
                    t = array([], dtype=float64)

                if time_ch.data_type in v23c.INT_TYPES:
                    dtype_fmt = get_fmt_v3(
                        time_ch.data_type,
                        time_ch.bit_count,
                        self.identification.byte_order,
                    )
                    channel_dtype: np.dtype[Any] = np.dtype(dtype_fmt.split(")")[-1])

                    if channel_dtype.byteorder == "=" and time_ch.data_type in (
                        v23c.DATA_TYPE_SIGNED_MOTOROLA,
                        v23c.DATA_TYPE_UNSIGNED_MOTOROLA,
                    ):
                        view = f">u{t.itemsize}"
                    else:
                        view = f"{channel_dtype.byteorder}u{t.itemsize}"

                    if bit_offset:
                        t >>= bit_offset

                    if time_ch.bit_count != t.itemsize * 8:
                        if time_ch.data_type in v23c.SIGNED_INT:
                            t = as_non_byte_sized_signed_int(t, time_ch.bit_count)
                        else:
                            mask = (1 << time_ch.bit_count) - 1
                            t &= mask
                    elif time_ch.data_type in v23c.SIGNED_INT:
                        view = f"{channel_dtype.byteorder}i{t.itemsize}"
                        t = t.view(view)

                # get timestamps
                conversion = time_ch.conversion
                if conversion is None:
                    time_conv_type = v23c.CONVERSION_TYPE_NONE
                else:
                    time_conv_type = conversion.conversion_type
                    if time_conv_type == v23c.CONVERSION_TYPE_LINEAR:
                        time_a = conversion.a
                        time_b = conversion.b
                        t = t * time_a
                        if time_b:
                            t += time_b

        if t.dtype != float64:
            timestamps = t.astype(float64)
        else:
            timestamps = t

        self._master_channel_metadata[index] = metadata

        return timestamps

    def iter_get_triggers(self) -> Iterator[TriggerInfoDict]:
        """Generator that yields triggers.

        Yields
        ------
        trigger_info : dict
            Trigger information with the following keys:

            * comment : trigger comment
            * time : trigger time
            * pre_time : trigger pre time
            * post_time : trigger post time
            * index : trigger index
            * group : data group index of trigger
        """
        for i, gp in enumerate(self.groups):
            trigger = gp.trigger
            if trigger:
                for j in range(trigger.trigger_events_nr):
                    trigger_info: TriggerInfoDict = {
                        "comment": trigger.comment,
                        "index": j,
                        "group": i,
                        "time": typing.cast(float, trigger[f"trigger_{j}_time"]),
                        "pre_time": typing.cast(float, trigger[f"trigger_{j}_pretime"]),
                        "post_time": typing.cast(float, trigger[f"trigger_{j}_posttime"]),
                    }
                    yield trigger_info

    def info(self) -> dict[str, object]:
        """Get MDF information as a dict.

        Examples
        --------
        >>> mdf = MDF('test.mdf')
        >>> mdf.info()
        """
        info: dict[str, object] = {
            "author": self.header.author,
            "department": self.header.department,
            "project": self.header.project,
            "subject": self.header.subject,
        }
        info["version"] = self.version
        info["groups"] = len(self.groups)
        for i, gp in enumerate(self.groups):
            inf: dict[str, object] = {}
            info[f"group {i}"] = inf
            inf["cycles"] = gp.channel_group.cycles_nr
            inf["comment"] = gp.channel_group.comment
            inf["channels count"] = len(gp.channels)
            for j, channel in enumerate(gp.channels):
                name = channel.name

                if channel.channel_type == v23c.CHANNEL_TYPE_MASTER:
                    ch_type = "master"
                else:
                    ch_type = "value"
                inf[f"channel {j}"] = f'name="{name}" type={ch_type}'

        return info

    @property
    def start_time(self) -> datetime:
        """Getter and setter of the measurement start timestamp.

        Returns
        -------
        timestamp : datetime.datetime
            Start timestamp.
        """

        return self.header.start_time

    @start_time.setter
    def start_time(self, timestamp: datetime) -> None:
        self.header.start_time = timestamp

    def save(
        self,
        dst: StrPath,
        overwrite: bool = False,
        compression: CompressionType = 0,
        progress: Any | None = None,
        add_history_block: bool = True,
    ) -> Path:
        """Save `MDF` to `dst`. If `overwrite` is True, then the destination
        file is overwritten, otherwise the file name is appended with '.<cntr>',
        where '<cntr>' is the first counter that produces a new file name that
        does not already exist in the filesystem.

        Parameters
        ----------
        dst : str | path-like
            Destination file name.
        overwrite : bool, default False
            Overwrite flag.
        compression : int, optional
            Does nothing for MDF version 3; introduced here to share the same
            API as MDF version 4 files.

        Returns
        -------
        output_file : pathlib.Path
            Path to saved file.
        """

        dst = Path(dst).with_suffix(".mdf")

        destination_dir = dst.parent
        destination_dir.mkdir(parents=True, exist_ok=True)

        if overwrite is False:
            if dst.is_file():
                cntr = 0
                while True:
                    name = dst.with_suffix(f".{cntr}.mdf")
                    if not name.exists():
                        break
                    else:
                        cntr += 1
                message = (
                    f'Destination file "{dst}" already exists and "overwrite" is False. Saving MDF file as "{name}"'
                )
                logger.warning(message)
                dst = name

        if not self.header.comment:
            self.header.comment = f"""<FHcomment>
<TX>created</TX>
<tool_id>{tool.__tool__}</tool_id>
<tool_vendor>{tool.__vendor__}</tool_vendor>
<tool_version>{tool.__version__}</tool_version>
</FHcomment>"""
        else:
            old_history = self.header.comment
            timestamp = time.asctime()

            text = f"{old_history}\n{timestamp}: updated by {tool.__tool__} {tool.__version__}"
            self.header.comment = text

        defined_texts: dict[bytes | str, int] = {}
        cc_map: dict[bytes, int] = {}
        si_map: dict[bytes, int] = {}

        if dst == self.name:
            destination = dst.with_suffix(".savetemp")
        else:
            destination = dst

        with open(destination, "wb+") as dst_:
            groups_nr = len(self.groups)

            write = dst_.write
            seek = dst_.seek
            # list of all blocks
            blocks: list[bytes | SupportsBytes] = []

            address = 0

            write(bytes(self.identification))
            address += v23c.ID_BLOCK_SIZE

            write(bytes(self.header))
            address += self.header.block_len

            if self.header.program:
                write(bytes(self.header.program))
                self.header.program_addr = address
                address += self.header.program.block_len
            else:
                self.header.program_addr = 0

            comment = TextBlock(text=self.header.comment)
            write(bytes(comment))
            self.header.comment_addr = address
            address += comment.block_len

            # DataGroup
            # put them first in the block list so they will be written first to
            # disk this way, in case of memory=False, we can safely
            # restore he original data block address
            gp_rec_ids = []

            original_data_block_addrs = [group.data_group.data_block_addr for group in self.groups]

            for idx, gp in enumerate(self.groups):
                dg = gp.data_group
                gp_rec_ids.append(dg.record_id_len)
                dg.record_id_len = 0

                # DataBlock
                dim = 0
                for data_bytes, _, __ in self._load_data(gp):
                    dim += len(data_bytes)
                    write(data_bytes)

                if gp.data_blocks:
                    gp.data_group.data_block_addr = address
                else:
                    gp.data_group.data_block_addr = 0
                address += dim

                if progress is not None:
                    if callable(progress):
                        progress(int(33 * (idx + 1) / groups_nr), 100)

            for gp in self.groups:
                dg = gp.data_group
                blocks.append(dg)
                dg.address = address
                address += dg.block_len

            if self.groups:
                for i, gp in enumerate(self.groups[:-1]):
                    addr = self.groups[i + 1].data_group.address
                    gp.data_group.next_dg_addr = addr
                self.groups[-1].data_group.next_dg_addr = 0

            for idx, gp in enumerate(self.groups):
                # Channel Dependency
                cd = gp.channel_dependencies
                for dep in cd:
                    if dep:
                        dep.address = address
                        blocks.append(dep)
                        address += dep.block_len

                for channel, dep in zip(gp.channels, gp.channel_dependencies, strict=False):
                    if dep:
                        channel.component_addr = dep.address = address
                        blocks.append(dep)
                        address += dep.block_len
                    else:
                        channel.component_addr = 0
                    address = channel.to_blocks(address, blocks, defined_texts, cc_map, si_map)

                count = len(gp.channels)
                if count:
                    for i in range(count - 1):
                        gp.channels[i].next_ch_addr = gp.channels[i + 1].address
                    gp.channels[-1].next_ch_addr = 0

                # ChannelGroup
                cg = gp.channel_group
                if gp.channels:
                    cg.first_ch_addr = gp.channels[0].address
                else:
                    cg.first_ch_addr = 0
                cg.next_cg_addr = 0
                address = cg.to_blocks(address, blocks, defined_texts, si_map)

                # TriggerBLock
                trigger = gp.trigger
                if trigger:
                    address = trigger.to_blocks(address, blocks)

                if progress is not None:
                    progress.signals.setValue.emit(int(33 * (idx + 1) / groups_nr) + 33)
                    progress.signals.setMaximum.emit(100)

                    if progress.stop:
                        dst_.close()
                        self.close()

                        raise Terminated

            # update referenced channels addresses in the channel dependencies
            for gp in self.groups:
                for dep in gp.channel_dependencies:
                    if not dep:
                        continue

                    for i, pair_ in enumerate(dep.referenced_channels):
                        dg_nr, ch_nr = pair_
                        grp = self.groups[dg_nr]
                        ch = grp.channels[ch_nr]
                        dep[f"ch_{i}"] = ch.address
                        dep[f"cg_{i}"] = grp.channel_group.address
                        dep[f"dg_{i}"] = grp.data_group.address

            # DataGroup
            for gp in self.groups:
                gp.data_group.first_cg_addr = gp.channel_group.address
                if gp.trigger:
                    gp.data_group.trigger_addr = gp.trigger.address
                else:
                    gp.data_group.trigger_addr = 0

            if self.groups:
                address = self.groups[0].data_group.address
                self.header.first_dg_addr = address
                self.header.dg_nr = len(self.groups)

            if progress is not None and progress.stop:
                dst_.close()
                self.close()
                raise Terminated

            if progress is not None:
                blocks_nr = len(blocks)
                threshold = blocks_nr / 33
                count = 1
                for i, block in enumerate(blocks):
                    write(bytes(block))
                    if i >= threshold:
                        progress.signals.setValue.emit(66 + count)
                        count += 1
                        threshold += blocks_nr / 33
            else:
                for block in blocks:
                    write(bytes(block))

            for gp, rec_id, original_address in zip(self.groups, gp_rec_ids, original_data_block_addrs, strict=False):
                gp.data_group.record_id_len = rec_id
                gp.data_group.data_block_addr = original_address

            seek(0)
            write(bytes(self.identification))
            write(bytes(self.header))

        if dst == self.name:
            self.close()
            Path.unlink(self.name)
            Path.rename(destination, self.name)

            self.groups.clear()
            self.channels_db.clear()
            self.masters_db.clear()

            self._tempfile = NamedTemporaryFile(dir=self.temporary_folder)
            self._file = open(self.name, "rb")
            self._read(self._file)

        return dst

    def _sort(self, progress: Callable[[int, int], None] | Any | None = None) -> None:
        if self._file is None:
            return
        common: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for i, group in enumerate(self.groups):
            if group.sorted:
                continue

            if group.data_blocks:
                address = group.data_blocks[0].address

                common[address].append((i, group.channel_group.record_id))

        read = self._file.read
        seek = self._file.seek

        self._tempfile.seek(0, 2)

        tell = self._tempfile.tell
        write = self._tempfile.write

        for address, groups in common.items():
            partial_records: dict[int, list[bytes]] = {id_: [] for (_, id_) in groups}

            group = self.groups[groups[0][0]]

            record_id_nr = group.data_group.record_id_len
            cg_size = group.record_size

            for info in group.data_blocks:
                address, size, block_size, block_type, param = (
                    info.address,
                    info.original_size,
                    info.compressed_size,
                    info.block_type,
                    info.param,
                )
                seek(address)
                data = read(block_size)

                size = len(data)
                i = 0
                while i < size:
                    rec_id = data[i]
                    # skip record id
                    i += 1
                    rec_size = cg_size[rec_id]
                    partial_records[rec_id].append(data[i : i + rec_size])
                    # consider the second record ID if it exists
                    if record_id_nr == 2:
                        i += rec_size + 1
                    else:
                        i += rec_size

            data_blocks: dict[int, list[DataBlockInfo]] = {}

            for rec_id, new_data in partial_records.items():
                if new_data:
                    data = b"".join(new_data)
                    size = len(data)

                    address = tell()
                    write(bytes(data))
                    block_info = DataBlockInfo(
                        address=address,
                        block_type=0,
                        original_size=size,
                        compressed_size=size,
                        param=0,
                    )
                    data_blocks[rec_id] = [block_info]

            for idx, rec_id in groups:
                group = self.groups[idx]

                group.data_location = v23c.LOCATION_TEMPORARY_FILE
                group.set_blocks_info(data_blocks[rec_id])
                group.sorted = True

    def included_channels(
        self,
        index: int | None = None,
        channels: ChannelsType | None = None,
        skip_master: bool = True,
        minimal: bool = True,
    ) -> dict[int, dict[int, list[int]]]:
        if channels is None:
            if index is None:
                raise ValueError("index argument must be set if channels is unset")
            group = self.groups[index]
            gps: dict[int, list[int]] = {}
            included_channels = set(range(len(group.channels)))
            master_index = self.masters_db.get(index, None)
            if master_index is not None and len(included_channels) > 1:
                included_channels.remove(master_index)

            for dep in group.channel_dependencies:
                if dep is None:
                    continue
                for gp_nr, ch_nr in dep.referenced_channels:
                    if gp_nr == index:
                        included_channels.remove(ch_nr)

            if included_channels:
                gps[index] = sorted(included_channels)

            result = {index: gps}
        else:
            group_sets: dict[int, set[int]] = {}
            for item in channels:
                if isinstance(item, (list, tuple)):
                    if len(item) not in (2, 3):
                        raise MdfException(
                            "The items used for filtering must be strings, "
                            "or they must match the first 3 arguments of the get "
                            "method"
                        )
                    else:
                        gp_idx, idx = self._validate_channel_selection(*item)
                        if gp_idx not in group_sets:
                            group_sets[gp_idx] = {idx}
                        else:
                            group_sets[gp_idx].add(idx)
                else:
                    name = item
                    gp_idx, idx = self._validate_channel_selection(name)
                    if gp_idx not in group_sets:
                        group_sets[gp_idx] = {idx}
                    else:
                        group_sets[gp_idx].add(idx)

            result = {}

            for group_index, _channels in group_sets.items():
                group = self.groups[group_index]

                channel_dependencies = [group.channel_dependencies[ch_nr] for ch_nr in _channels]

                if minimal:
                    for dep in channel_dependencies:
                        if dep is None:
                            continue
                        for gp_nr, ch_nr in dep.referenced_channels:
                            if gp_nr == group_index:
                                try:
                                    _channels.remove(ch_nr)
                                except KeyError:
                                    pass

                gp_master = self.masters_db.get(group_index, None)
                if skip_master and gp_master is not None and gp_master in _channels and len(_channels) > 1:
                    _channels.remove(gp_master)

                result[group_index] = {group_index: sorted(_channels)}

        return result

    def _yield_selected_signals(
        self,
        index: int,
        groups: dict[int, list[int]] | None = None,
        record_offset: int = 0,
        record_count: int | None = None,
        skip_master: bool = True,
        version: str = "4.20",
    ) -> Iterator[list[Signal] | list[tuple[NDArray[Any], None]]]:
        if groups is None:
            groups = self.included_channels(index)[index]

        channels = groups[index]

        group = self.groups[index]

        encodings: list[str | None] = [
            None,
        ]

        self._set_temporary_master(None)

        for idx, fragment in enumerate(self._load_data(group, record_offset=record_offset, record_count=record_count)):
            master = self.get_master(index, data=fragment)
            self._set_temporary_master(master)

            self._prepare_record(group)

            signals: list[Signal] | list[tuple[NDArray[Any], None]]
            # the first fragment triggers and append that will add the
            # metadata for all channels
            if idx == 0:
                signals = [
                    self.get(
                        group=index,
                        index=channel_index,
                        data=fragment,
                        raw=True,
                        ignore_invalidation_bits=True,
                        samples_only=False,
                    )
                    for channel_index in channels
                ]
            else:
                signals = [(master, None)]

                for channel_index in channels:
                    signals.append(
                        self.get(
                            group=index,
                            index=channel_index,
                            data=fragment,
                            raw=True,
                            ignore_invalidation_bits=True,
                            samples_only=True,
                        )
                    )

            if version < "4.00":
                if idx == 0:
                    signals = typing.cast(list[Signal], signals)
                    for sig, channel_index in zip(signals, channels, strict=False):
                        if sig.samples.dtype.kind == "S":
                            encodings.append(sig.encoding)
                            strsig = self.get(
                                group=index,
                                index=channel_index,
                                samples_only=True,
                                ignore_invalidation_bits=True,
                            )[0]
                            sig.samples = sig.samples.astype(strsig.dtype)
                            del strsig
                            if sig.encoding != "latin-1":
                                if sig.encoding == "utf-16-le":
                                    sig.samples = sig.samples.view(uint16).byteswap().view(sig.samples.dtype)
                                    sig.samples = encode(decode(sig.samples, "utf-16-be"), "latin-1")
                                else:
                                    sig.samples = encode(decode(sig.samples, sig.encoding), "latin-1")
                        else:
                            encodings.append(None)
                else:
                    signals = typing.cast(list[tuple[NDArray[Any], None]], signals)
                    for i, (signal_samples, encoding) in enumerate(zip(signals, encodings, strict=False)):
                        if encoding:
                            samples = signal_samples[0]
                            if encoding != "latin-1":
                                if encoding == "utf-16-le":
                                    samples = samples.view(uint16).byteswap().view(samples.dtype)
                                    samples = encode(decode(samples, "utf-16-be"), "latin-1")
                                else:
                                    samples = encode(decode(samples, encoding), "latin-1")
                                signals[i] = (samples, signal_samples[1])

            self._set_temporary_master(None)
            yield signals

    def reload_header(self) -> None:
        if self._file is None:
            raise RuntimeError("self._file is None")
        self.header = HeaderBlock(address=0x40, stream=self._file)

    def _determine_max_vlsd_sample_size(self, group: int, index: int) -> int:
        return 0


if __name__ == "__main__":
    pass

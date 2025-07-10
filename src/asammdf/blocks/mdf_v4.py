"""ASAM MDF version 4 file format module"""

import bisect
from collections import defaultdict, deque
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from hashlib import md5
from io import StringIO
import logging
from math import ceil, floor
from mimetypes import guess_type
import mmap
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from tempfile import gettempdir, NamedTemporaryFile
from traceback import format_exc
import typing
from typing import BinaryIO, Final, Literal, TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile

import canmatrix
from canmatrix.canmatrix import CanMatrix
from lz4.frame import compress as lz_compress
from lz4.frame import decompress as lz_decompress
import numpy as np
from numpy import (
    arange,
    argwhere,
    array,
    array_equal,
    column_stack,
    concatenate,
    empty,
    float32,
    float64,
    frombuffer,
    full,
    linspace,
    nonzero,
    searchsorted,
    transpose,
    uint8,
    uint16,
    uint32,
    uint64,
    unique,
    where,
    zeros,
)
from numpy.typing import DTypeLike, NDArray
from pandas import DataFrame, Series
from typing_extensions import (
    Any,
    Buffer,
    overload,
    SupportsBytes,
    TypedDict,
    TypeIs,
    Unpack,
)

from .. import tool
from ..signal import InvalidationArray, Signal
from . import bus_logging_utils, mdf_common, utils
from . import v4_constants as v4c
from .conversion_utils import conversion_transfer
from .cutils import (
    data_block_from_arrays,
    extract,
    get_channel_raw_bytes,
    get_channel_raw_bytes_parallel,
    get_invalidation_bits_array,
    get_vlsd_max_sample_size,
    sort_data_block,
)
from .mdf_common import LastCallInfo, MDF_Common, MdfCommonKwargs
from .options import GLOBAL_OPTIONS
from .source_utils import Source
from .types import (
    BusType,
    ChannelsType,
    CompressionType,
    DbcFileType,
    RasterType,
    StrPath,
)
from .utils import (
    all_blocks_addresses,
    as_non_byte_sized_signed_int,
    CHANNEL_COUNT,
    CONVERT,
    count_channel_groups,
    DataBlockInfo,
    extract_display_names,
    extract_encryption_information,
    extract_xml_comment,
    FileLike,
    fmt_to_datatype_v4,
    Fragment,
    get_fmt_v4,
    get_text_v4,
    InvalidationBlockInfo,
    is_file_like,
    load_can_database,
    MdfException,
    SignalDataBlockInfo,
    Terminated,
    THREAD_COUNT,
    TxMap,
    UINT8_uf,
    UINT16_uf,
    UINT32_p,
    UINT32_uf,
    UINT64_uf,
    UniqueDB,
    validate_version_argument,
    VirtualChannelGroup,
)
from .v4_blocks import (
    AttachmentBlock,
    Channel,
    ChannelArrayBlock,
    ChannelArrayBlockKwargs,
    ChannelConversion,
    ChannelGroup,
    ChannelGroupKwargs,
    ChannelKwargs,
    DataBlock,
    DataGroup,
    DataList,
    DataListKwargs,
    DataZippedBlock,
    DataZippedBlockKwargs,
    EventBlock,
    FileHistory,
    FileIdentificationBlock,
    HeaderBlock,
    HeaderList,
    HeaderListKwargs,
    ListData,
    ListDataKwargs,
    SourceInformation,
    TextBlock,
)
from .v4_constants import Version

try:
    from cryptography.hazmat.primitives.ciphers import algorithms, Cipher, modes

    CRYPTOGRAPHY_AVAILABLE = True
except:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    from isal.isal_zlib import decompress
except ImportError:
    from zlib import decompress

if TYPE_CHECKING:
    from ..mdf import MDF

try:
    decode = np.strings.decode
    encode = np.strings.encode
except:
    decode = np.char.decode
    encode = np.char.encode

MASTER_CHANNELS: Final = (v4c.CHANNEL_TYPE_MASTER, v4c.CHANNEL_TYPE_VIRTUAL_MASTER)
COMMON_SIZE: Final = v4c.COMMON_SIZE
COMMON_u = v4c.COMMON_u
COMMON_uf = v4c.COMMON_uf

COMMON_SHORT_SIZE: Final = v4c.COMMON_SHORT_SIZE
COMMON_SHORT_uf = v4c.COMMON_SHORT_uf
COMMON_SHORT_u = v4c.COMMON_SHORT_u
VALID_DATA_TYPES: Final = v4c.VALID_DATA_TYPES

EMPTY_TUPLE: Final = ()

# 100 extra steps for the sorting, 1 step after sorting and 1 step at finish
SORT_STEPS: Final = 102


logger = logging.getLogger("asammdf")

__all__ = ["MDF4"]

Group = mdf_common.GroupV4


class BusLoggingMap(TypedDict):
    CAN: dict[int, dict[int, int]]
    ETHERNET: dict[int, int]
    FLEXRAY: dict[int, int]
    LIN: dict[int, int]


class Kwargs(MdfCommonKwargs, total=False):
    column_storage: bool
    process_bus_logging: bool


class MDF4(MDF_Common[Group]):
    r"""The `header` attribute is a `HeaderBlock`.

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
    * ``data_location`` - integer code for data location (original file,
      temporary file or memory)
    * ``data_block_addr`` - list of raw samples starting addresses
    * ``data_block_type`` - list of codes for data block type
    * ``data_block_size`` - list of raw samples block size
    * ``sorted`` - sorted indicator flag
    * ``record_size`` - dict that maps record IDs to record sizes in bytes
      (including invalidation bytes)
    * ``param`` - row size used for transposition, in case of transposed zipped
      blocks

    Parameters
    ----------
    name : str | path-like | file-like, optional
        MDF file name (if provided it must be a real file name) or file-like
        object.
    version : str, default '4.10'
        MDF file version ('4.00', '4.10', '4.11', '4.20').
    use_display_names : bool, default True
        Parse the XML channel comment to search for the display name; XML
        parsing is quite expensive so setting this to False can decrease the
        loading times very much.
    remove_source_from_channel_names : bool, default False
        Remove source from channel names ("Speed\XCP3" -> "Speed").
    compact_vlsd (False) : bool, optional
        Use slower method to save the exact sample size for VLSD channels.
    column_storage : bool, default True
        Use column storage for MDF version >= 4.20.
    password : bytes | str, optional
        Use this password to decode encrypted attachments.

    Attributes
    ----------
    attachments : list
        List of file attachments.
    channels_db : dict
        Used for fast channel access by name; for each name key the value is a
        list of (group index, channel index) tuples.
    events : list
        List event blocks.
    file_history : list
        List of (FileHistory, TextBlock) pairs.
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
        version: Version = "4.10",
        channels: list[str] | None = None,
        **kwargs: Unpack[Kwargs],
    ) -> None:
        if not kwargs.get("__internal__", False):
            raise MdfException("Always use the MDF class; do not use the class MDF4 directly")

        # bind cache to instance to avoid memory leaks
        self.determine_max_vlsd_sample_size = lru_cache(maxsize=1024 * 1024)(self._determine_max_vlsd_sample_size)
        self.extract_attachment = lru_cache(maxsize=128)(self._extract_attachment)

        self._kwargs = kwargs
        self.original_name = kwargs["original_name"]
        self.file_history: list[FileHistory] = []
        self.masters_db: dict[int, int] = {}
        self.attachments: list[AttachmentBlock] = []
        self._attachments_cache: dict[bytes | str, int] = {}
        self.events: list[EventBlock] = []
        self.bus_logging_map: BusLoggingMap = {"CAN": {}, "ETHERNET": {}, "FLEXRAY": {}, "LIN": {}}

        self._attachments_map: dict[int, int] = {}
        self._ch_map: dict[int, tuple[int, int]] = {}
        self._master_channel_metadata: dict[int, tuple[str, int]] = {}
        self._invalidation_cache: dict[tuple[int, int, int, int], InvalidationArray] = {}
        self._external_dbc_cache: dict[bytes, CanMatrix] = {}
        self._si_map: dict[bytes | int | Source, SourceInformation] = {}
        self._file_si_map: dict[object, object] = {}
        self._cc_map: dict[bytes | int, ChannelConversion] = {}
        self._file_cc_map: dict[object, object] = {}
        self._cg_map: dict[int, int] = {}
        self._cn_data_map: dict[int, tuple[int, int]] = {}
        self._dbc_cache: dict[int, CanMatrix] = {}
        self._interned_strings: TxMap = {}

        self._closed = False

        self.temporary_folder = kwargs.get("temporary_folder", GLOBAL_OPTIONS["temporary_folder"])

        if channels is None:
            self.load_filter: set[str] = set()
            self.use_load_filter = False
        else:
            self.load_filter = set(channels)
            self.use_load_filter = True

        self._tempfile = NamedTemporaryFile(dir=self.temporary_folder)
        self._mapped_file: BinaryIO | None = None
        self._file: FileLike | mmap.mmap | None = None

        self._read_fragment_size = GLOBAL_OPTIONS["read_fragment_size"]
        self._write_fragment_size = GLOBAL_OPTIONS["write_fragment_size"]
        self._single_bit_uint_as_bool = GLOBAL_OPTIONS["single_bit_uint_as_bool"]
        self._integer_interpolation = GLOBAL_OPTIONS["integer_interpolation"]
        self._float_interpolation = GLOBAL_OPTIONS["float_interpolation"]
        self._use_display_names = kwargs.get("use_display_names", GLOBAL_OPTIONS["use_display_names"])
        self._fill_0_for_missing_computation_channels = kwargs.get(
            "fill_0_for_missing_computation_channels",
            GLOBAL_OPTIONS["fill_0_for_missing_computation_channels"],
        )

        self._remove_source_from_channel_names = kwargs.get("remove_source_from_channel_names", False)
        self._password = kwargs.get("password", None)
        self._force_attachment_encryption = kwargs.get("force_attachment_encryption", False)
        self.compact_vlsd = kwargs.get("compact_vlsd", False)

        self.virtual_groups: dict[int, VirtualChannelGroup] = {}  # master group 2 referencing groups
        self.virtual_groups_map: dict[int, int] = {}  # group index 2 master group

        self.vlsd_max_length: dict[
            tuple[int, str], int
        ] = {}  # hint about the maximum vlsd length for group_index, name pairs

        self._master = None

        self.last_call_info: LastCallInfo = {}

        # make sure no appended block has the address 0
        self._tempfile.write(b"\0")

        self._delete_on_close = False

        progress = kwargs.get("progress", None)

        self._column_storage = False

        super().__init__(kwargs.get("raise_on_multiple_occurrences", GLOBAL_OPTIONS["raise_on_multiple_occurrences"]))

        if name:
            if is_file_like(name):
                self._file = name
                self.name = self.original_name = Path("From_FileLike.mf4")
                self._from_filelike = True
                self._read(self._file, mapped=False, progress=progress)
            else:
                try:
                    with open(name, "rb") as stream:
                        identification = FileIdentificationBlock(stream=stream)
                        version_str = identification.version_str.decode("utf-8").strip(" \n\t\0")
                        flags = identification.unfinalized_standard_flags

                    if version_str >= "4.10" and flags:
                        tmpdir = Path(gettempdir())
                        if self.temporary_folder:
                            tmpdir = Path(self.temporary_folder)
                        self.name = tmpdir / f"{os.urandom(6).hex()}_{Path(name).name}"
                        shutil.copy(name, self.name)
                        self._file = open(self.name, "rb+")
                        self._from_filelike = False
                        self._delete_on_close = True
                        self._read(self._file, mapped=False, progress=progress)
                    else:
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
                    del self._file
                    self._mapped_file = None
                    raise

        else:
            self._from_filelike = False
            version = validate_version_argument(version)
            self.header = HeaderBlock()
            self.identification = FileIdentificationBlock(version=version)
            self.version: str = version

            if self.version >= "4.20":
                self._column_storage = kwargs.get("column_storage", True)
            else:
                self._column_storage = False

            self.name = Path("__new__.mf4")

        self._parent: MDF | None = None

    def __del__(self) -> None:
        self.close()

    def _check_finalised(self) -> int:
        flags = self.identification.unfinalized_standard_flags

        if flags & 1:
            message = f"Unfinalised file {self.name}: Update of cycle counters for CG/CA blocks required"

            logger.info(message)
        if flags & 1 << 1:
            message = f"Unfinalised file {self.name}: Update of cycle counters for SR blocks required"

            logger.info(message)
        if flags & 1 << 2:
            message = f"Unfinalised file {self.name}: Update of length for last DT block required"

            logger.info(message)
        if flags & 1 * 8:
            message = f"Unfinalised file {self.name}: Update of length for last RD block required"

            logger.info(message)
        if flags & 1 << 4:
            message = (
                f"Unfinalised file {self.name}: Update of last DL block in each chained list of DL blocks required"
            )

            logger.info(message)
        if flags & 1 << 5:
            message = (
                f"Unfinalised file {self.name}: Update of cg_data_bytes and cg_inval_bytes in VLSD CG block required"
            )

            logger.info(message)
        if flags & 1 << 6:
            message = (
                f"Unfinalised file {self.name}:"
                " Update of offset values for VLSD channel required"
                " in case a VLSD CG block is used"
            )

            logger.info(message)

        return flags

    def _read(
        self,
        stream: FileLike | mmap.mmap,
        mapped: bool = False,
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> None:
        self._mapped = mapped
        dg_cntr = 0

        stream.seek(0, 2)
        self.file_limit = stream.tell()
        stream.seek(0)

        cg_count, _ = count_channel_groups(stream)
        progress_steps = cg_count + SORT_STEPS

        if progress is not None:
            if callable(progress):
                progress(0, progress_steps)
        current_cg_index = 0

        self.identification = FileIdentificationBlock(stream=stream, mapped=mapped)
        version = self.identification.version_str
        self.version = version.decode("utf-8").strip(" \n\t\r\0")

        if self.version >= "4.20":
            self._column_storage = self._kwargs.get("column_storage", True)
        else:
            self._column_storage = False

        if self.version >= "4.10":
            # Check for finalization past version 4.10
            finalisation_flags = self._check_finalised()

            if finalisation_flags:
                message = f"Attempting finalization of {self.name}"
                logger.info(message)
                self._finalize(stream)
                self._mapped = mapped = False

        self.header = HeaderBlock(address=0x40, stream=stream, mapped=mapped)

        # read file history
        fh_addr = self.header.file_history_addr
        while fh_addr:
            if (fh_addr + v4c.FH_BLOCK_SIZE) > self.file_limit:
                logger.warning(f"File history address {fh_addr:X} is outside the file size {self.file_limit}")
                break
            history_block = FileHistory(address=fh_addr, stream=stream, mapped=mapped)
            self.file_history.append(history_block)
            fh_addr = history_block.next_fh_addr

        # read attachments
        at_addr = self.header.first_attachment_addr
        index = 0
        while at_addr:
            if (at_addr + v4c.AT_COMMON_SIZE) > self.file_limit:
                logger.warning(f"Attachment address {at_addr:X} is outside the file size {self.file_limit}")
                break
            at_block = AttachmentBlock(address=at_addr, stream=stream, mapped=mapped)
            self._attachments_map[at_addr] = index
            self.attachments.append(at_block)
            at_addr = at_block.next_at_addr
            index += 1

        # go to first date group and read each data group sequentially
        dg_addr = self.header.first_dg_addr

        while dg_addr:
            if (dg_addr + v4c.DG_BLOCK_SIZE) > self.file_limit:
                logger.warning(f"Data group address {dg_addr:X} is outside the file size {self.file_limit}")
                break
            new_groups = []
            group = DataGroup(address=dg_addr, stream=stream, mapped=mapped)
            record_id_nr = group.record_id_len

            # go to first channel group of the current data group
            cg_addr = first_cg_addr = group.first_cg_addr

            cg_nr = 0

            cg_size: dict[int, int] = {}

            while cg_addr:
                if (cg_addr + v4c.CG_BLOCK_SIZE) > self.file_limit:
                    logger.warning(f"Channel group address {cg_addr:X} is outside the file size {self.file_limit}")
                    break
                cg_nr += 1

                if cg_addr == first_cg_addr:
                    grp = Group(group)
                else:
                    grp = Group(group.copy())

                # read each channel group sequentially
                block = ChannelGroup(
                    address=cg_addr,
                    stream=stream,
                    mapped=mapped,
                    si_map=self._si_map,
                    tx_map=self._interned_strings,
                )
                self._cg_map[cg_addr] = dg_cntr
                channel_group = grp.channel_group = block

                grp.record_size = cg_size

                if channel_group.flags & v4c.FLAG_CG_VLSD:
                    # VLDS flag
                    record_id = channel_group.record_id
                    cg_size[record_id] = 0
                elif channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                    samples_size = channel_group.samples_byte_nr
                    inval_size = channel_group.invalidation_bytes_nr
                    record_id = channel_group.record_id
                    cg_size[record_id] = samples_size + inval_size
                else:
                    # in case no `cg_flags` are set
                    samples_size = channel_group.samples_byte_nr
                    inval_size = channel_group.invalidation_bytes_nr
                    record_id = channel_group.record_id
                    cg_size[record_id] = samples_size + inval_size

                if record_id_nr:
                    grp.sorted = False
                else:
                    grp.sorted = True

                # go to first channel of the current channel group
                ch_addr = channel_group.first_ch_addr
                ch_cntr = 0

                # Read channels by walking recursively in the channel group
                # starting from the first channel
                self._read_channels(ch_addr, grp, stream, dg_cntr, ch_cntr, mapped=mapped)

                cg_addr = channel_group.next_cg_addr

                dg_cntr += 1

                current_cg_index += 1
                if progress is not None:
                    if callable(progress):
                        progress(current_cg_index, progress_steps)

                new_groups.append(grp)

            # store channel groups record sizes dict in each
            # new group data belong to the initial unsorted group, and add
            # the key 'sorted' with the value False to use a flag;

            address = group.data_block_addr

            total_size = 0
            inval_total_size = 0
            block_type = b"##DT"
            record_size = 0

            for new_group in new_groups:
                channel_group = new_group.channel_group
                if channel_group.flags & v4c.FLAG_CG_REMOTE_MASTER:
                    block_type = b"##DV"
                    total_size += channel_group.samples_byte_nr * channel_group.cycles_nr
                    inval_total_size += channel_group.invalidation_bytes_nr * channel_group.cycles_nr
                    record_size = channel_group.samples_byte_nr
                else:
                    block_type = b"##DT"
                    total_size += (
                        channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr
                    ) * channel_group.cycles_nr

                    record_size = channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr

            if self.identification.unfinalized_standard_flags & v4c.FLAG_UNFIN_UPDATE_CG_COUNTER:
                total_size = 10**12
                inval_total_size = 10**12

            data_blocks_info = self._get_data_blocks_info(
                address=address,
                stream=stream,
                block_type=block_type,
                mapped=mapped,
                total_size=total_size,
                inval_total_size=inval_total_size,
                record_size=record_size,
            )
            data_blocks = list(data_blocks_info)  # load the info blocks directly here
            uses_ld = self._uses_ld(
                address=address,
                stream=stream,
                block_type=block_type,
                mapped=mapped,
            )

            for grp in new_groups:
                grp.data_location = v4c.LOCATION_ORIGINAL_FILE
                grp.data_blocks_info_generator = data_blocks_info
                grp.data_blocks = data_blocks
                grp.uses_ld = uses_ld
                self._prepare_record(grp)

            self.groups.extend(new_groups)

            dg_addr = group.next_dg_addr

        # all channels have been loaded so now we can link the
        # channel dependencies and load the signal data for VLSD channels
        for gp_index, grp in enumerate(self.groups):
            if self.version >= "4.20" and grp.channel_group.flags & v4c.FLAG_CG_REMOTE_MASTER:
                grp.channel_group.cg_master_index = self._cg_map[grp.channel_group.cg_master_addr]
                index = grp.channel_group.cg_master_index

            else:
                index = gp_index

            self.virtual_groups_map[gp_index] = index
            if index not in self.virtual_groups:
                self.virtual_groups[index] = VirtualChannelGroup()

            virtual_channel_group = self.virtual_groups[index]
            virtual_channel_group.groups.append(gp_index)
            virtual_channel_group.record_size += (
                grp.channel_group.samples_byte_nr + grp.channel_group.invalidation_bytes_nr
            )
            virtual_channel_group.cycles_nr = grp.channel_group.cycles_nr

            for ch_index, dep_list in enumerate(grp.channel_dependencies):
                if not dep_list:
                    continue

                for dep in dep_list:
                    if isinstance(dep, ChannelArrayBlock):
                        if dep.flags & v4c.FLAG_CA_DYNAMIC_AXIS:
                            for i in range(dep.dims):
                                ch_addr = typing.cast(int, dep[f"dynamic_size_{i}_ch_addr"])
                                if ch_addr:
                                    ref_channel = self._ch_map[ch_addr]
                                    dep.dynamic_size_channels.append(ref_channel)
                                else:
                                    dep.dynamic_size_channels.append(None)

                        if dep.flags & v4c.FLAG_CA_INPUT_QUANTITY:
                            for i in range(dep.dims):
                                ch_addr = typing.cast(int, dep[f"input_quantity_{i}_ch_addr"])
                                if ch_addr:
                                    ref_channel = self._ch_map[ch_addr]
                                    dep.input_quantity_channels.append(ref_channel)
                                else:
                                    dep.input_quantity_channels.append(None)

                        if dep.flags & v4c.FLAG_CA_OUTPUT_QUANTITY:
                            ch_addr = dep.output_quantity_ch_addr
                            if ch_addr:
                                ref_channel = self._ch_map[ch_addr]
                                dep.output_quantity_channel = ref_channel
                            else:
                                dep.output_quantity_channel = None

                        if dep.flags & v4c.FLAG_CA_COMPARISON_QUANTITY:
                            ch_addr = dep.comparison_quantity_ch_addr
                            if ch_addr:
                                ref_channel = self._ch_map[ch_addr]
                                dep.comparison_quantity_channel = ref_channel
                            else:
                                dep.comparison_quantity_channel = None

                        if dep.flags & v4c.FLAG_CA_AXIS:
                            for i in range(dep.dims):
                                cc_addr = typing.cast(int, dep[f"axis_conversion_{i}"])
                                if cc_addr:
                                    conv = ChannelConversion(
                                        stream=stream,
                                        address=cc_addr,
                                        mapped=mapped,
                                        tx_map={},
                                    )
                                    dep.axis_conversions.append(conv)
                                else:
                                    dep.axis_conversions.append(None)

                        if (dep.flags & v4c.FLAG_CA_AXIS) and not (dep.flags & v4c.FLAG_CA_FIXED_AXIS):
                            for i in range(dep.dims):
                                ch_addr = typing.cast(int, dep[f"scale_axis_{i}_ch_addr"])
                                if ch_addr:
                                    ref_channel = self._ch_map[ch_addr]
                                    dep.axis_channels.append(ref_channel)
                                else:
                                    dep.axis_channels.append(None)
                    else:
                        break

        self._sort(
            stream,
            current_progress_index=current_cg_index,
            max_progress_count=progress_steps,
            progress=progress,
        )
        if progress is not None:
            if callable(progress):
                progress(progress_steps - 1, progress_steps)  # second to last step now

        for grp in self.groups:
            channels = grp.channels
            if len(channels) == 1 and channels[0].dtype_fmt.itemsize == grp.channel_group.samples_byte_nr:
                grp.single_channel_dtype = channels[0].dtype_fmt

        if self._kwargs.get("process_bus_logging", True):
            self._process_bus_logging()

        # read events
        addr = self.header.first_event_addr
        ev_map: dict[int, int] = {}
        event_index = 0
        while addr:
            if (addr + v4c.COMMON_SIZE) > self.file_limit:
                logger.warning(f"Event address {addr:X} is outside the file size {self.file_limit}")
                break
            event = EventBlock(address=addr, stream=stream, mapped=mapped)
            event.update_references(self._ch_map, self._cg_map)
            self.events.append(event)
            ev_map[addr] = event_index
            event_index += 1

            addr = event.next_ev_addr

        for event in self.events:
            addr = event.parent_ev_addr
            if addr:
                parent = ev_map.get(addr, None)
                if parent is not None:
                    event.parent = parent
                else:
                    event.parent = None

            addr = event.range_start_ev_addr
            if addr:
                range_start_ev_addr = ev_map.get(addr, None)
                if range_start_ev_addr is not None:
                    event.parent = range_start_ev_addr
                else:
                    event.parent = None

        self._si_map.clear()
        self._ch_map.clear()
        self._cc_map.clear()

        self._interned_strings.clear()
        self._attachments_map.clear()

        if progress is not None:
            if callable(progress):
                progress(progress_steps, progress_steps)  # last step, we've completely loaded the file for sure

        self.progress = cg_count, cg_count

    @overload
    def _read_channels(
        self,
        ch_addr: int,
        grp: Group,
        stream: FileLike | mmap.mmap,
        dg_cntr: int,
        ch_cntr: int,
        parent_channel: Channel,
        mapped: bool = ...,
    ) -> tuple[int, list[ChannelArrayBlock] | list[tuple[int, int]], np.dtype[Any]]: ...

    @overload
    def _read_channels(
        self,
        ch_addr: int,
        grp: Group,
        stream: FileLike | mmap.mmap,
        dg_cntr: int,
        ch_cntr: int,
        parent_channel: None = ...,
        mapped: bool = ...,
    ) -> tuple[int, None, None]: ...

    def _read_channels(
        self,
        ch_addr: int,
        grp: Group,
        stream: FileLike | mmap.mmap,
        dg_cntr: int,
        ch_cntr: int,
        parent_channel: Channel | None = None,
        mapped: bool = False,
    ) -> tuple[int, list[ChannelArrayBlock] | list[tuple[int, int]] | None, np.dtype[Any] | None]:
        filter_channels = self.use_load_filter
        use_display_names = self._use_display_names

        channels = grp.channels

        dependencies = grp.channel_dependencies

        unique_names = UniqueDB()

        if parent_channel:
            composition: list[tuple[int, int]] | None = []
            composition_channels: list[Channel] | None = []
        else:
            composition = composition_channels = None

        if grp.channel_group.path_separator:
            path_separator = chr(grp.channel_group.path_separator)
        else:
            path_separator = "."

        while ch_addr:
            # read channel block and create channel object

            if (ch_addr + v4c.COMMON_SIZE) > self.file_limit:
                logger.warning(f"Channel address {ch_addr:X} is outside the file size {self.file_limit}")
                break

            if filter_channels:
                if utils.stream_is_mmap(stream, mapped):
                    (
                        id_,
                        links_nr,
                        next_ch_addr,
                        component_addr,
                        name_addr,
                        comment_addr,
                    ) = v4c.CHANNEL_FILTER_uf(stream, ch_addr)
                    channel_type = stream[ch_addr + v4c.COMMON_SIZE + links_nr * 8]
                    name = get_text_v4(name_addr, stream, mapped=mapped, tx_map=self._interned_strings)
                    if use_display_names:
                        comment = get_text_v4(comment_addr, stream, mapped=mapped, tx_map=self._interned_strings)
                        display_names = extract_display_names(comment)
                    else:
                        display_names = {}
                        comment = None

                else:
                    stream.seek(ch_addr)
                    (
                        id_,
                        links_nr,
                        next_ch_addr,
                        component_addr,
                        name_addr,
                        comment_addr,
                    ) = v4c.CHANNEL_FILTER_u(stream.read(v4c.CHANNEL_FILTER_SIZE))
                    stream.seek(ch_addr + v4c.COMMON_SIZE + links_nr * 8)
                    channel_type = stream.read(1)[0]
                    name = get_text_v4(name_addr, stream, mapped=mapped, tx_map=self._interned_strings)

                    if use_display_names:
                        comment = get_text_v4(comment_addr, stream, mapped=mapped, tx_map=self._interned_strings)
                        display_names = extract_display_names(comment)
                    else:
                        display_names = {}
                        comment = None

                if id_ != b"##CN":
                    message = f'Expected "##CN" block @{hex(ch_addr)} but found "{id_!r}"'
                    raise MdfException(message)

                if self._remove_source_from_channel_names:
                    name = name.split(path_separator, 1)[0]
                    display_names = {_name.split(path_separator, 1)[0]: val for _name, val in display_names.items()}

                if (
                    parent_channel
                    or channel_type in v4c.MASTER_TYPES
                    or name in self.load_filter
                    or (use_display_names and any(dsp_name in self.load_filter for dsp_name in display_names))
                ):
                    if comment is None:
                        comment = get_text_v4(comment_addr, stream, mapped=mapped, tx_map=self._interned_strings)
                    channel = Channel(
                        address=ch_addr,
                        stream=stream,
                        cc_map=self._cc_map,
                        si_map=self._si_map,
                        at_map=self._attachments_map,
                        use_display_names=use_display_names,
                        mapped=mapped,
                        tx_map=self._interned_strings,
                        parsed_strings=(name, display_names, comment),
                    )

                    si_path = getattr(channel.source, "path", "")
                    if si_path and not self._remove_source_from_channel_names:
                        path_name = f"{si_path}.{channel.name}"
                        if path_name not in channel.display_names:
                            channel.display_names[path_name] = "source_path"

                elif not component_addr:
                    ch_addr = next_ch_addr
                    continue
                else:
                    if (component_addr + v4c.CC_ALG_BLOCK_SIZE) > self.file_limit:
                        logger.warning(
                            f"Channel component address {component_addr:X} is outside the file size {self.file_limit}"
                        )
                        break
                    # check if it is a CABLOCK or CNBLOCK
                    stream.seek(component_addr)
                    blk_id = stream.read(4)
                    if blk_id == b"##CN":
                        (
                            ch_cntr,
                            _1,
                            _2,
                        ) = self._read_channels(
                            component_addr,
                            grp,
                            stream,
                            dg_cntr,
                            ch_cntr,
                            None,
                            mapped=mapped,
                        )

                    ch_addr = next_ch_addr
                    continue

            else:
                channel = Channel(
                    address=ch_addr,
                    stream=stream,
                    cc_map=self._cc_map,
                    si_map=self._si_map,
                    at_map=self._attachments_map,
                    use_display_names=use_display_names,
                    mapped=mapped,
                    tx_map=self._interned_strings,
                    parsed_strings=None,
                )

            if channel.data_type not in VALID_DATA_TYPES:
                ch_addr = channel.next_ch_addr
                continue

            if channel.channel_type == v4c.CHANNEL_TYPE_SYNC:
                channel.attachment = self._attachments_map.get(
                    channel.data_block_addr,
                    None,
                )

            if channel.source and channel.source.bus_type in v4c.BUS_LOGGING_TYPES and channel.source.path:
                path_name = f"{channel.source.path}.{channel.name}"
                if path_name not in channel.display_names:
                    channel.display_names[path_name] = "source_path"

            if self._remove_source_from_channel_names:
                channel.name = channel.name.split(path_separator, 1)[0]
                channel.display_names = {
                    _name.split(path_separator, 1)[0]: val for _name, val in channel.display_names.items()
                }

            entry = (dg_cntr, ch_cntr)
            self._ch_map[ch_addr] = entry

            channels.append(channel)
            if composition is not None and composition_channels is not None:
                composition.append(entry)
                composition_channels.append(channel)

            for _name in channel.display_names:
                self.channels_db.add(_name, entry)
            self.channels_db.add(channel.name, entry)

            # signal data
            cn_data_addr = channel.data_block_addr
            if cn_data_addr:
                grp.signal_data.append(([], self._get_signal_data_blocks_info(cn_data_addr, stream)))
            else:
                grp.signal_data.append(None)

            if cn_data_addr:
                self._cn_data_map[cn_data_addr] = entry

            if channel.channel_type in MASTER_CHANNELS:
                self.masters_db[dg_cntr] = ch_cntr

            ch_cntr += 1

            component_addr = channel.component_addr

            if component_addr:
                if (component_addr + 4) > self.file_limit:
                    logger.warning(
                        f"Channel component address {component_addr:X} is outside the file size {self.file_limit}"
                    )
                    break

                index = ch_cntr - 1
                dependencies.append(None)

                # check if it is a CABLOCK or CNBLOCK
                stream.seek(component_addr)
                blk_id = stream.read(4)
                if blk_id == b"##CN":
                    (
                        ch_cntr,
                        ret_composition,
                        ret_composition_dtype,
                    ) = self._read_channels(
                        component_addr,
                        grp,
                        stream,
                        dg_cntr,
                        ch_cntr,
                        channel,
                        mapped=mapped,
                    )
                    dependencies[index] = ret_composition

                    channel.dtype_fmt = ret_composition_dtype

                else:
                    # only channel arrays with storage=CN_TEMPLATE are
                    # supported so far
                    channel.dtype_fmt = np.dtype(
                        get_fmt_v4(
                            channel.data_type,
                            channel.bit_offset + channel.bit_count,
                            channel.channel_type,
                        )
                    )

                    first_dep = ca_block = ChannelArrayBlock(address=component_addr, stream=stream, mapped=mapped)
                    ca_dependencies = [first_dep]

                    while ca_block.composition_addr:
                        stream.seek(ca_block.composition_addr)
                        blk_id = stream.read(4)
                        if blk_id == b"##CA":
                            ca_block = ChannelArrayBlock(
                                address=ca_block.composition_addr,
                                stream=stream,
                                mapped=mapped,
                            )
                            ca_dependencies.append(ca_block)

                        elif channel.data_type == v4c.DATA_TYPE_BYTEARRAY:
                            # read CA-CN nested structure
                            (
                                ch_cntr,
                                ret_composition,
                                ret_composition_dtype,
                            ) = self._read_channels(
                                ca_block.composition_addr,
                                grp,
                                stream,
                                dg_cntr,
                                ch_cntr,
                                channel,
                                mapped=mapped,
                            )
                            ret_composition = typing.cast(list[ChannelArrayBlock], ret_composition)

                            channel.dtype_fmt = ret_composition_dtype

                            ca_cnt = len(ca_dependencies)
                            if ret_composition:
                                ca_dependencies.extend(ret_composition)

                            byte_offset_factors: list[int] = []
                            bit_pos_inval_factors: list[int] = []
                            dimensions: list[int] = []
                            total_elem = 1

                            for ca_blck in ca_dependencies[:ca_cnt]:
                                # only consider CN templates
                                if ca_blck.ca_type != v4c.CA_STORAGE_TYPE_CN_TEMPLATE:
                                    logger.warning("Only CN template arrays are supported")
                                    continue

                                # 1D array with dimensions
                                for i in range(ca_blck.dims):
                                    dim_size = typing.cast(int, ca_blck[f"dim_size_{i}"])
                                    dimensions.append(dim_size)
                                    total_elem *= dim_size

                                # 1D arrays for byte offset and invalidation bit pos calculations
                                byte_offset_factors.extend(ca_blck.get_byte_offset_factors())
                                bit_pos_inval_factors.extend(ca_blck.get_bit_pos_inval_factors())

                            multipliers = [1] * len(dimensions)
                            for i in range(len(dimensions) - 2, -1, -1):
                                multipliers[i] = multipliers[i + 1] * dimensions[i + 1]

                            def _get_nd_coords(index: int, factors: list[int]) -> list[int]:
                                """Convert 1D index to CA's nD coordinates."""
                                coords = [0] * len(factors)
                                for i, factor in enumerate(factors):
                                    coords[i] = index // factor
                                    index %= factor
                                return coords

                            def _get_name_with_indices(ch_name: str, ch_parent_name: str, indices: list[int]) -> str:
                                coords = "[" + "][".join(str(coord) for coord in indices) + "]"
                                m = re.match(ch_parent_name, ch_name)
                                n = re.search(r"\[\d+\]", ch_name)
                                if m:
                                    name = ch_name[: m.end()] + coords + ch_name[m.end() :]
                                elif n:
                                    name = ch_name[: n.start()] + coords + ch_name[n.start() :]
                                else:
                                    name = ch_name + coords
                                return name

                            ch_len = len(channels)
                            for elem_id in range(1, total_elem):
                                for cn_id in range(index, ch_len):
                                    nd_coords = _get_nd_coords(elem_id, multipliers)

                                    # copy composition block
                                    new_block = deepcopy(channels[cn_id])

                                    # update byte offset & position of invalidation bit
                                    byte_offset = bit_offset = 0
                                    for coord, byte_factor, bit_factor in zip(
                                        nd_coords, byte_offset_factors, bit_pos_inval_factors, strict=False
                                    ):
                                        byte_offset += coord * byte_factor
                                        bit_offset += coord * bit_factor
                                    new_block.byte_offset += byte_offset
                                    new_block.pos_invalidation_bit += bit_offset

                                    # update channel name
                                    new_block.name = _get_name_with_indices(new_block.name, channel.name, nd_coords)

                                    # append to channel list
                                    channels.append(new_block)

                                    # update channel dependencies
                                    if (deps := dependencies[cn_id]) is not None:
                                        cn_deps: list[tuple[int, int]] = []
                                        for dep in deps:
                                            if not isinstance(dep, ChannelArrayBlock):
                                                dep_entry = (dep[0], dep[1] + (ch_len - index) * elem_id)
                                                cn_deps.append(dep_entry)
                                        dependencies.append(cn_deps)
                                    else:
                                        dependencies.append(None)

                                    # update channels db
                                    entry = (dg_cntr, ch_cntr)
                                    self.channels_db.add(new_block.name, entry)
                                    ch_cntr += 1

                            # modify channels' names found recursively in-place
                            orig_name = channel.name
                            for cn_id in range(index, ch_len):
                                nd_coords = _get_nd_coords(0, multipliers)
                                name = _get_name_with_indices(channels[cn_id].name, orig_name, nd_coords)
                                entries = self.channels_db.pop(channels[cn_id].name)
                                channels[cn_id].name = name
                                # original channel entry will only contain single source tuple
                                self.channels_db.add(name, entries[0])

                            break

                        else:
                            logger.warning(
                                "skipping CN block; Nested CA structure should be contained within BYTEARRAY data type"
                            )
                            break
                    dependencies[index] = ca_dependencies

            else:
                dependencies.append(None)

                channel.dtype_fmt = np.dtype(
                    get_fmt_v4(
                        channel.data_type,
                        channel.bit_offset + channel.bit_count,
                        channel.channel_type,
                    )
                )

            # go to next channel of the current channel group
            ch_addr = channel.next_ch_addr

        if composition_channels is not None and parent_channel:
            composition_channels.sort(key=lambda x: x.byte_offset)
            padding = 0
            dtype_fields: list[DTypeLike] = []
            offset = parent_channel.byte_offset

            for comp_channel in composition_channels:
                if (delta := (comp_channel.byte_offset - offset)) > 0:
                    dtype_fields.append((f"__padding_{padding}__", f"V{delta}"))
                    padding += 1
                dtype_fields.append((unique_names.get_unique_name(comp_channel.name), comp_channel.dtype_fmt))
                offset = comp_channel.byte_offset + comp_channel.dtype_fmt.itemsize

            composition_dtype = np.dtype(dtype_fields)

        else:
            composition_dtype = None

        return ch_cntr, composition, composition_dtype

    def _load_signal_data(
        self,
        group: Group | None = None,
        index: int | None = None,
        offsets: tuple[int, int] | None = None,
    ) -> bytes:
        """This method is used to get the channel signal data, usually for VLSD
        channels.

        Returns
        -------
        data : bytes
            Signal data bytes.
        """

        if group is not None and index is not None:
            info_blocks = group.signal_data[index]

            if info_blocks is not None:
                stream: tempfile._TemporaryFileWrapper[bytes] | FileLike | mmap.mmap
                if offsets is None:
                    data_list: list[bytes] = []

                    for info in group.get_signal_data_blocks(index):
                        address, original_size, compressed_size, block_type, param = (
                            info.address,
                            info.original_size,
                            info.compressed_size,
                            info.block_type,
                            info.param,
                        )

                        if not info.original_size:
                            continue
                        if info.location == v4c.LOCATION_TEMPORARY_FILE:
                            stream = self._tempfile
                        else:
                            if self._file is None:
                                raise RuntimeError("'_file' is None")
                            stream = self._file

                        stream.seek(address)
                        new_data = stream.read(compressed_size)
                        match block_type:
                            case v4c.DZ_BLOCK_DEFLATE:
                                new_data = decompress(new_data, bufsize=original_size)
                            case v4c.DZ_BLOCK_TRANSPOSED:
                                new_data = decompress(new_data, bufsize=original_size)
                                cols = param
                                lines = original_size // cols

                                matrix_size = lines * cols

                                if matrix_size != original_size:
                                    new_data = (
                                        frombuffer(new_data[:matrix_size], dtype=uint8)
                                        .reshape((cols, lines))
                                        .T.ravel()
                                        .tobytes()
                                        + new_data[matrix_size:]
                                    )
                                else:
                                    new_data = (
                                        frombuffer(new_data, dtype=uint8).reshape((cols, lines)).T.ravel().tobytes()
                                    )

                            case v4c.DZ_BLOCK_LZ:
                                new_data = lz_decompress(new_data)

                        data_list.append(new_data)

                    data = b"".join(data_list)

                else:
                    start_offset, end_offset = offsets

                    data_array = bytearray()

                    start_offset = int(start_offset)
                    end_offset = int(end_offset)
                    last_sample_start = end_offset - start_offset

                    current_offset = 0

                    for info in group.get_signal_data_blocks(index):
                        address, original_size, compressed_size, block_type, param = (
                            info.address,
                            info.original_size,
                            info.compressed_size,
                            info.block_type,
                            info.param,
                        )

                        if not info.original_size:
                            continue
                        if info.location == v4c.LOCATION_TEMPORARY_FILE:
                            stream = self._tempfile
                        else:
                            if self._file is None:
                                raise RuntimeError("'_file' is None")
                            stream = self._file

                        if current_offset + original_size < start_offset:
                            current_offset += original_size
                            continue

                        stream.seek(address)
                        new_data = stream.read(compressed_size)
                        match block_type:
                            case v4c.DZ_BLOCK_DEFLATE:
                                new_data = decompress(new_data, bufsize=original_size)
                            case v4c.DZ_BLOCK_TRANSPOSED:
                                new_data = decompress(new_data, bufsize=original_size)
                                cols = param
                                lines = original_size // cols

                                matrix_size = lines * cols

                                if matrix_size != original_size:
                                    new_data = (
                                        frombuffer(new_data[:matrix_size], dtype=uint8)
                                        .reshape((cols, lines))
                                        .T.ravel()
                                        .tobytes()
                                        + new_data[matrix_size:]
                                    )
                                else:
                                    new_data = (
                                        frombuffer(new_data, dtype=uint8).reshape((cols, lines)).T.ravel().tobytes()
                                    )

                            case v4c.DZ_BLOCK_LZ:
                                new_data = lz_decompress(new_data)

                        if start_offset > current_offset:
                            data_array.extend(new_data[start_offset - current_offset :])
                        else:
                            data_array.extend(new_data)

                        current_offset += original_size

                        if (current_data_size := len(data_array)) >= last_sample_start + 4:
                            (last_sample_size,) = UINT32_uf(data_array, last_sample_start)
                            required_size = last_sample_start + 4 + last_sample_size
                            if required_size <= current_data_size:
                                data = bytes(data_array[:required_size])
                                break

                    else:
                        data = bytes(data_array)

            else:
                data = b""
        else:
            data = b""

        return data

    def _load_data(
        self,
        group: Group,
        record_offset: int = 0,
        record_count: int | None = None,
        optimize_read: bool = False,
    ) -> Iterator[Fragment]:
        """Get group's data block bytes."""

        from time import perf_counter

        cc = 0

        offset = 0
        invalidation_offset = 0
        has_yielded = False
        _count = 0

        data_blocks_info_generator = group.data_blocks_info_generator
        channel_group = group.channel_group

        stream: FileLike | mmap.mmap | tempfile._TemporaryFileWrapper[bytes]
        if group.data_location == v4c.LOCATION_ORIGINAL_FILE:
            stream = typing.cast(FileLike | mmap.mmap, self._file)
        else:
            stream = self._tempfile

        read = stream.read
        seek = stream.seek

        if group.uses_ld:
            samples_size = channel_group.samples_byte_nr
            invalidation_size = channel_group.invalidation_bytes_nr
            invalidation_record_offset = record_offset * invalidation_size
            rm = True
        else:
            rm = False
            samples_size = channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr
            invalidation_size = channel_group.invalidation_bytes_nr

        record_offset *= samples_size

        finished = False
        if record_count is not None:
            invalidation_record_count = record_count * invalidation_size
            record_count *= samples_size
            max_size = record_count + invalidation_record_count
        else:
            max_size = (invalidation_size + samples_size) * channel_group.cycles_nr

        if not samples_size:
            if rm:
                yield Fragment(b"", offset, _count, b"")
            else:
                yield Fragment(b"", offset, _count, None)
        else:
            if group.read_split_count:
                split_size = group.read_split_count * samples_size
                invalidation_split_size = group.read_split_count * invalidation_size
            else:
                if self._read_fragment_size:
                    split_size = self._read_fragment_size // samples_size
                    invalidation_split_size = split_size * invalidation_size
                    split_size *= samples_size

                else:
                    channels_nr = len(group.channels)

                    y_axis = CONVERT

                    idx = int(searchsorted(CHANNEL_COUNT, channels_nr, side="right") - 1)
                    idx = max(idx, 0)
                    split_size = y_axis[idx]

                    split_size = split_size // samples_size
                    invalidation_split_size = split_size * invalidation_size
                    split_size *= samples_size

            if split_size == 0:
                split_size = samples_size
                invalidation_split_size = invalidation_size

            split_size = int(split_size)
            if split_size > max_size:
                invalidation_split_size = (max_size // samples_size) * invalidation_size
                split_size = max_size

            buffer = bytearray(split_size)
            buffer_view = memoryview(buffer)

            invalidation_split_size = int(invalidation_split_size)

            blocks = iter(group.data_blocks)

            cur_size = 0
            data: list[object] = []

            cur_invalidation_size = 0
            invalidation_data: list[bytes] = []

            tt = perf_counter()
            ss = 0
            cc = 0

            while True:
                try:
                    info = next(blocks)
                    (
                        address,
                        original_size,
                        compressed_size,
                        block_type,
                        param,
                        block_limit,
                    ) = (
                        info.address,
                        typing.cast(int, info.original_size),
                        info.compressed_size,
                        info.block_type,
                        info.param,
                        info.block_limit,
                    )

                    if rm and invalidation_size:
                        invalidation_info = info.invalidation_block
                    else:
                        invalidation_info = None
                except StopIteration:
                    try:
                        info = next(data_blocks_info_generator)
                        (
                            address,
                            original_size,
                            compressed_size,
                            block_type,
                            param,
                            block_limit,
                        ) = (
                            info.address,
                            typing.cast(int, info.original_size),
                            info.compressed_size,
                            info.block_type,
                            info.param,
                            info.block_limit,
                        )

                        if rm and invalidation_size:
                            invalidation_info = info.invalidation_block
                        else:
                            invalidation_info = None
                        group.data_blocks.append(info)
                    except StopIteration:
                        break

                if offset + original_size < record_offset + 1:
                    offset += original_size
                    if rm and invalidation_size:
                        if invalidation_info is None:
                            raise RuntimeError(
                                "'invalidation_info' cannot be None if 'rm and invalidation_size' is True"
                            )
                        if invalidation_info.all_valid:
                            count = original_size // samples_size
                            invalidation_offset += count * invalidation_size
                        else:
                            invalidation_offset += typing.cast(int, invalidation_info.original_size)
                    continue

                seek(address)
                new_data: bytes | memoryview[int] = read(typing.cast(int, compressed_size))

                cc += 1
                ss += original_size

                match block_type:
                    case v4c.DZ_BLOCK_DEFLATE:
                        new_data = decompress(new_data, bufsize=original_size)
                    case v4c.DZ_BLOCK_TRANSPOSED:
                        new_data = decompress(new_data, bufsize=original_size)
                        cols = typing.cast(int, param)
                        lines = original_size // cols
                        matrix_size = lines * cols

                        if matrix_size != original_size:
                            new_data = (
                                frombuffer(new_data[:matrix_size], dtype=uint8)
                                .reshape((cols, lines))
                                .T.ravel()
                                .tobytes()
                                + new_data[matrix_size:]
                            )
                        else:
                            new_data = frombuffer(new_data, dtype=uint8).reshape((cols, lines)).T.ravel().tobytes()

                    case v4c.DZ_BLOCK_LZ:
                        new_data = lz_decompress(new_data)

                if block_limit is not None:
                    new_data = new_data[:block_limit]

                if len(new_data) > split_size - cur_size:
                    new_data = memoryview(new_data)

                if rm and invalidation_size:
                    if invalidation_info is None:
                        raise RuntimeError("'invalidation_info' cannot be None if 'rm and invalidation_size' is True")

                    if invalidation_info.all_valid:
                        count = original_size // samples_size
                        new_invalidation_data = b"\0" * (count * invalidation_size)

                    else:
                        seek(invalidation_info.address)
                        compressed_size = typing.cast(int, invalidation_info.compressed_size)
                        new_invalidation_data = read(compressed_size)
                        original_size = typing.cast(int, invalidation_info.original_size)
                        if invalidation_info.block_type == v4c.DZ_BLOCK_DEFLATE:
                            new_invalidation_data = decompress(
                                new_invalidation_data,
                                bufsize=original_size,
                            )
                        elif invalidation_info.block_type == v4c.DZ_BLOCK_TRANSPOSED:
                            new_invalidation_data = decompress(
                                new_invalidation_data,
                                bufsize=original_size,
                            )
                            cols = typing.cast(int, invalidation_info.param)
                            lines = original_size // cols

                            nd = frombuffer(new_invalidation_data[: lines * cols], dtype=uint8)
                            nd = nd.reshape((cols, lines))
                            new_invalidation_data = nd.T.ravel().tobytes() + new_invalidation_data[lines * cols :]
                        if invalidation_info.block_limit is not None:
                            new_invalidation_data = new_invalidation_data[: invalidation_info.block_limit]

                    inv_size = len(new_invalidation_data)

                if offset < record_offset:
                    delta = record_offset - offset
                    new_data = new_data[delta:]
                    original_size -= delta
                    offset = record_offset

                    if rm and invalidation_size:
                        delta = invalidation_record_offset - invalidation_offset
                        new_invalidation_data = new_invalidation_data[delta:]
                        inv_size -= delta
                        invalidation_offset = invalidation_record_offset

                while original_size >= split_size - cur_size:
                    if cur_size:
                        buffer_view[cur_size:] = new_data[: split_size - cur_size]
                        new_data = new_data[split_size - cur_size :]

                        if rm and invalidation_size:
                            invalidation_data.append(
                                new_invalidation_data[: invalidation_split_size - cur_invalidation_size]
                            )
                            new_invalidation_data = new_invalidation_data[
                                invalidation_split_size - cur_invalidation_size :
                            ]
                            invalidation_data_ = b"".join(invalidation_data)

                        if record_count is not None:
                            if rm and invalidation_size:
                                __data = buffer[:record_count]
                                _count = len(__data) // samples_size
                                yield Fragment(
                                    __data,
                                    offset // samples_size,
                                    _count,
                                    invalidation_data_[:invalidation_record_count],
                                )
                                invalidation_record_count -= len(invalidation_data_)
                            else:
                                __data = buffer[:record_count]
                                _count = len(__data) // samples_size
                                yield Fragment(__data, offset // samples_size, _count, None)
                            has_yielded = True
                            record_count -= split_size
                            if record_count <= 0:
                                finished = True
                                break
                        else:
                            if rm and invalidation_size:
                                _count = split_size // samples_size
                                yield Fragment(buffer, offset // samples_size, _count, invalidation_data_)
                            else:
                                _count = split_size // samples_size
                                yield Fragment(buffer, offset // samples_size, _count, None)
                            has_yielded = True

                    else:
                        buffer_view[:] = new_data[:split_size]
                        new_data = new_data[split_size:]
                        if rm and invalidation_size:
                            invalidation_data_ = new_invalidation_data[:invalidation_split_size]
                            new_invalidation_data = new_invalidation_data[invalidation_split_size:]

                        if record_count is not None:
                            if rm and invalidation_size:
                                yield Fragment(
                                    buffer[:record_count],
                                    offset // samples_size,
                                    _count,
                                    invalidation_data_[:invalidation_record_count],
                                )
                                invalidation_record_count -= len(invalidation_data_)
                            else:
                                __data = buffer[:record_count]
                                _count = len(__data) // samples_size
                                yield Fragment(__data, offset // samples_size, _count, None)
                            has_yielded = True
                            record_count -= split_size
                            if record_count <= 0:
                                finished = True
                                cur_size = 0
                                break
                        else:
                            if rm and invalidation_size:
                                _count = split_size // samples_size
                                yield Fragment(buffer, offset // samples_size, _count, invalidation_data_)
                            else:
                                _count = split_size // samples_size
                                yield Fragment(buffer, offset // samples_size, _count, None)
                            has_yielded = True

                    offset += split_size
                    original_size -= split_size - cur_size
                    data = []
                    cur_size = 0

                    if rm and invalidation_size:
                        invalidation_offset += invalidation_split_size
                        invalidation_data = []
                        cur_invalidation_size = 0
                        inv_size -= invalidation_split_size - cur_invalidation_size

                if finished:
                    cur_size = 0
                    original_size = 0
                    if rm and invalidation_size:
                        invalidation_data = []
                    break

                if original_size > 0:
                    buffer_view[cur_size : cur_size + original_size] = new_data
                    cur_size += original_size

                    if rm and invalidation_size:
                        invalidation_data.append(new_invalidation_data)
                        cur_invalidation_size += inv_size

                    if record_count is not None and cur_size >= record_count:
                        finished = True
                        break

                if 0 and (vv := (perf_counter() - tt)) > 10:
                    print(f"{ss / 1024 / 1024 / vv:.6f} MB/s {cc=} {vv=}")
                    cc = 0
                    ss = 0
                    tt = perf_counter()

            if cur_size:
                data_ = buffer[:cur_size]
                if rm and invalidation_size:
                    invalidation_data_ = b"".join(invalidation_data)
                if record_count is not None:
                    if rm and invalidation_size:
                        __data = data_[:record_count]
                        _count = len(__data) // samples_size
                        yield Fragment(
                            __data, offset // samples_size, _count, invalidation_data_[:invalidation_record_count]
                        )
                        invalidation_record_count -= len(invalidation_data_)
                    else:
                        __data = data_[:record_count]
                        _count = len(__data) // samples_size
                        yield Fragment(__data, offset // samples_size, _count, None)
                    has_yielded = True
                    record_count -= len(data_)
                else:
                    if rm and invalidation_size:
                        _count = len(data_) // samples_size
                        yield Fragment(data_, offset // samples_size, _count, invalidation_data_)
                    else:
                        _count = len(data_) // samples_size
                        yield Fragment(data_, offset // samples_size, _count, None)
                    has_yielded = True

            if not has_yielded:
                if rm and invalidation_size:
                    yield Fragment(b"", 0, 0, b"")
                else:
                    yield Fragment(b"", 0, 0, None)

    def _prepare_record(self, group: Group) -> list[tuple[np.dtype[Any], int, int, int] | None]:
        """Compute record.

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
            channels = group.channels

            record: list[tuple[np.dtype[Any], int, int, int] | None] = []

            for idx, new_ch in enumerate(channels):
                start_offset = new_ch.byte_offset
                bit_offset = new_ch.bit_offset
                data_type = new_ch.data_type
                bit_count = new_ch.bit_count
                ch_type = new_ch.channel_type
                dependency_list = group.channel_dependencies[idx]

                if ch_type not in v4c.VIRTUAL_TYPES:
                    # adjust size to 1, 2, 4 or 8 bytes
                    size = bit_offset + bit_count

                    byte_size, rem = divmod(size, 8)
                    if rem:
                        byte_size += 1
                    bit_size = byte_size * 8

                    if data_type in (
                        v4c.DATA_TYPE_SIGNED_MOTOROLA,
                        v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                    ):
                        if size > 32:
                            bit_offset += 64 - bit_size
                        elif size > 16:
                            bit_offset += 32 - bit_size
                        elif size > 8:
                            bit_offset += 16 - bit_size

                    if new_ch.dtype_fmt == np.dtype(np.void):
                        new_ch.dtype_fmt = np.dtype(get_fmt_v4(data_type, size, ch_type))

                    if (
                        bit_offset
                        or dependency_list
                        or (new_ch.dtype_fmt.kind in "ui" and size < 64 and size not in (8, 16, 32))
                    ):
                        new_ch.standard_C_size = False

                    record.append(
                        (
                            new_ch.dtype_fmt,
                            new_ch.dtype_fmt.itemsize,
                            start_offset,
                            bit_offset,
                        )
                    )
                else:
                    record.append(None)

            group.record = record

        return group.record

    def _uses_ld(
        self,
        address: int,
        stream: FileLike | mmap.mmap,
        block_type: bytes = b"##DT",
        mapped: bool = False,
    ) -> bool:
        mapped = mapped or not is_file_like(stream)
        uses_ld = False

        if utils.stream_is_mmap(stream, mapped):
            if address:
                id_string, block_len = COMMON_SHORT_uf(stream, address)

                if id_string == b"##LD":
                    uses_ld = True
                # or a header list
                elif id_string == b"##HL":
                    hl = HeaderList(address=address, stream=stream, mapped=mapped)
                    address = hl.first_dl_addr

                    uses_ld = self._uses_ld(
                        address,
                        stream,
                        block_type,
                        mapped,
                    )
        else:
            if address:
                stream.seek(address)
                id_string, block_len = COMMON_SHORT_u(stream.read(COMMON_SHORT_SIZE))

                # can be a DataBlock
                if id_string == b"##LD":
                    uses_ld = True

                # or a header list
                elif id_string == b"##HL":
                    hl = HeaderList(address=address, stream=stream)
                    address = hl.first_dl_addr

                    uses_ld = self._uses_ld(
                        address,
                        stream,
                        block_type,
                        mapped,
                    )

        return uses_ld

    def _get_data_blocks_info(
        self,
        address: int,
        stream: FileLike | mmap.mmap,
        block_type: bytes = b"##DT",
        mapped: bool = False,
        total_size: int = 0,
        inval_total_size: int = 0,
        record_size: int = 0,
    ) -> Iterator[DataBlockInfo]:
        mapped = mapped or not is_file_like(stream)

        if record_size > 4 * 1024 * 1024:
            READ_CHUNK_SIZE = record_size
        elif record_size:
            READ_CHUNK_SIZE = 4 * 1024 * 1024 // record_size * record_size
        else:
            READ_CHUNK_SIZE = 4 * 1024 * 1024

        READ_CHUNK_SIZE = min(READ_CHUNK_SIZE, total_size)

        if utils.stream_is_mmap(stream, mapped):
            if address:
                id_string, block_len = COMMON_SHORT_uf(stream, address)

                # can be a DataBlock
                if id_string == block_type:
                    size = block_len - 24
                    if size:
                        size = min(size, total_size)

                        address = address + COMMON_SIZE

                        # split the DTBLOCK into chucks of up to 32MB
                        while True:
                            if size > READ_CHUNK_SIZE:
                                total_size -= READ_CHUNK_SIZE
                                size -= READ_CHUNK_SIZE

                                yield DataBlockInfo(
                                    address=address,
                                    block_type=v4c.DT_BLOCK,
                                    original_size=READ_CHUNK_SIZE,
                                    compressed_size=READ_CHUNK_SIZE,
                                    param=0,
                                    block_limit=None,
                                )
                                address += READ_CHUNK_SIZE
                            else:
                                if total_size < size:
                                    block_limit = total_size
                                else:
                                    block_limit = None

                                yield DataBlockInfo(
                                    address=address,
                                    block_type=v4c.DT_BLOCK,
                                    original_size=size,
                                    compressed_size=size,
                                    param=0,
                                    block_limit=block_limit,
                                )
                                break

                # or a DataZippedBlock
                elif id_string == b"##DZ":
                    (
                        zip_type,
                        param,
                        original_size,
                        zip_size,
                    ) = v4c.DZ_COMMON_INFO_uf(stream, address + v4c.DZ_INFO_COMMON_OFFSET)

                    if original_size:
                        if zip_type == v4c.FLAG_DZ_DEFLATE:
                            block_type_ = v4c.DZ_BLOCK_DEFLATE
                            param = 0
                        else:
                            block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                        if total_size < original_size:
                            block_limit = total_size
                        else:
                            block_limit = None
                        total_size -= original_size
                        yield DataBlockInfo(
                            address=address + v4c.DZ_COMMON_SIZE,
                            block_type=block_type_,
                            original_size=original_size,
                            compressed_size=zip_size,
                            param=param,
                            block_limit=block_limit,
                        )

                # or a DataList
                elif id_string == b"##DL":
                    while address:
                        dl = DataList(address=address, stream=stream, mapped=mapped)
                        for i in range(dl.data_block_nr):
                            addr = getattr(dl, f"data_block_addr{i}")

                            id_string, block_len = COMMON_SHORT_uf(stream, addr)
                            # can be a DataBlock
                            if id_string == block_type:
                                size = block_len - 24
                                if size:
                                    size = min(size, total_size)

                                    addr += COMMON_SIZE

                                    # split the DTBLOCK into chucks of up to 32MB
                                    while True:
                                        if size > READ_CHUNK_SIZE:
                                            total_size -= READ_CHUNK_SIZE
                                            size -= READ_CHUNK_SIZE

                                            yield DataBlockInfo(
                                                address=addr,
                                                block_type=v4c.DT_BLOCK,
                                                original_size=READ_CHUNK_SIZE,
                                                compressed_size=READ_CHUNK_SIZE,
                                                param=0,
                                                block_limit=None,
                                            )
                                            addr += READ_CHUNK_SIZE
                                        else:
                                            if total_size < size:
                                                block_limit = total_size
                                            else:
                                                block_limit = None

                                            total_size -= size

                                            yield DataBlockInfo(
                                                address=addr,
                                                block_type=v4c.DT_BLOCK,
                                                original_size=size,
                                                compressed_size=size,
                                                param=0,
                                                block_limit=block_limit,
                                            )
                                            break

                            # or a DataZippedBlock
                            else:
                                (
                                    zip_type,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_INFO_uf(stream, addr + v4c.DZ_INFO_COMMON_OFFSET)

                                if original_size:
                                    if zip_type == v4c.FLAG_DZ_DEFLATE:
                                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                                        param = 0
                                    else:
                                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                    if total_size < original_size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= original_size
                                    yield DataBlockInfo(
                                        address=addr + v4c.DZ_COMMON_SIZE,
                                        block_type=block_type_,
                                        original_size=original_size,
                                        compressed_size=zip_size,
                                        param=param,
                                        block_limit=block_limit,
                                    )
                        address = dl.next_dl_addr

                # or a ListData
                elif id_string == b"##LD":
                    uses_ld = True
                    while address:
                        ld = ListData(address=address, stream=stream, mapped=mapped)
                        has_invalidation = ld.flags & v4c.FLAG_LD_INVALIDATION_PRESENT
                        for i in range(ld.data_block_nr):
                            addr = getattr(ld, f"data_block_addr_{i}")

                            id_string, block_len = COMMON_SHORT_uf(stream, addr)
                            # can be a DataBlock
                            if id_string == b"##DV":
                                size = block_len - 24
                                if size:
                                    if total_size < size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= size
                                    data_info = DataBlockInfo(
                                        address=addr + COMMON_SIZE,
                                        block_type=v4c.DT_BLOCK,
                                        original_size=size,
                                        compressed_size=size,
                                        param=0,
                                        block_limit=block_limit,
                                    )

                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                (
                                    zip_type,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_INFO_uf(stream, addr + v4c.DZ_INFO_COMMON_OFFSET)

                                if original_size:
                                    if zip_type == v4c.FLAG_DZ_DEFLATE:
                                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                                        param = 0
                                    else:
                                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                    if total_size < original_size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= original_size
                                    data_info = DataBlockInfo(
                                        address=addr + v4c.DZ_COMMON_SIZE,
                                        block_type=block_type_,
                                        original_size=original_size,
                                        compressed_size=zip_size,
                                        param=param,
                                        block_limit=block_limit,
                                    )

                            if has_invalidation:
                                inval_addr = typing.cast(int, ld[f"invalidation_bits_addr_{i}"])
                                if inval_addr:
                                    id_string, block_len = COMMON_SHORT_uf(stream, inval_addr)
                                    if id_string == b"##DI":
                                        size = block_len - 24
                                        if size:
                                            if inval_total_size < size:
                                                block_limit = inval_total_size
                                            else:
                                                block_limit = None
                                            inval_total_size -= size
                                            data_info.invalidation_block = InvalidationBlockInfo(
                                                address=inval_addr + COMMON_SIZE,
                                                block_type=v4c.DT_BLOCK,
                                                original_size=size,
                                                compressed_size=size,
                                                param=0,
                                                block_limit=block_limit,
                                            )
                                    else:
                                        (
                                            zip_type,
                                            param,
                                            original_size,
                                            zip_size,
                                        ) = v4c.DZ_COMMON_INFO_uf(
                                            stream,
                                            inval_addr + v4c.DZ_INFO_COMMON_OFFSET,
                                        )

                                        if original_size:
                                            if zip_type == v4c.FLAG_DZ_DEFLATE:
                                                block_type_ = v4c.DZ_BLOCK_DEFLATE
                                                param = 0
                                            else:
                                                block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                            if inval_total_size < original_size:
                                                block_limit = inval_total_size
                                            else:
                                                block_limit = None
                                            inval_total_size -= original_size
                                            data_info.invalidation_block = InvalidationBlockInfo(
                                                address=inval_addr + v4c.DZ_COMMON_SIZE,
                                                block_type=block_type_,
                                                original_size=original_size,
                                                compressed_size=zip_size,
                                                param=param,
                                                block_limit=block_limit,
                                            )
                                else:
                                    data_info.invalidation_block = InvalidationBlockInfo(
                                        address=0,
                                        block_type=v4c.DT_BLOCK,
                                        original_size=None,
                                        compressed_size=None,
                                        param=None,
                                        all_valid=True,
                                    )

                            yield data_info

                        address = ld.next_ld_addr

                # or a header list
                elif id_string == b"##HL":
                    hl = HeaderList(address=address, stream=stream, mapped=mapped)
                    address = hl.first_dl_addr

                    yield from self._get_data_blocks_info(
                        address,
                        stream,
                        block_type,
                        mapped,
                        total_size,
                        inval_total_size,
                        record_size,
                    )
        else:
            if address:
                stream.seek(address)
                id_string, block_len = COMMON_SHORT_u(stream.read(COMMON_SHORT_SIZE))

                # can be a DataBlock
                if id_string == block_type:
                    size = block_len - 24
                    if size:
                        size = min(size, total_size)
                        address = address + COMMON_SIZE

                        # split the DTBLOCK into chucks of up to 32MB
                        while True:
                            if size > READ_CHUNK_SIZE:
                                total_size -= READ_CHUNK_SIZE
                                size -= READ_CHUNK_SIZE

                                yield DataBlockInfo(
                                    address=address,
                                    block_type=v4c.DT_BLOCK,
                                    original_size=READ_CHUNK_SIZE,
                                    compressed_size=READ_CHUNK_SIZE,
                                    param=0,
                                    block_limit=None,
                                )
                                address += READ_CHUNK_SIZE
                            else:
                                if total_size < size:
                                    block_limit = total_size
                                else:
                                    block_limit = None

                                yield DataBlockInfo(
                                    address=address,
                                    block_type=v4c.DT_BLOCK,
                                    original_size=size,
                                    compressed_size=size,
                                    param=0,
                                    block_limit=block_limit,
                                )
                                break

                # or a DataZippedBlock
                elif id_string == b"##DZ":
                    stream.seek(address + v4c.DZ_INFO_COMMON_OFFSET)
                    (
                        zip_type,
                        param,
                        original_size,
                        zip_size,
                    ) = v4c.DZ_COMMON_INFO_u(stream.read(v4c.DZ_COMMON_INFO_SIZE))

                    if original_size:
                        if zip_type == v4c.FLAG_DZ_DEFLATE:
                            block_type_ = v4c.DZ_BLOCK_DEFLATE
                            param = 0
                        else:
                            block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                        if total_size < original_size:
                            block_limit = total_size
                        else:
                            block_limit = None
                        total_size -= original_size
                        yield DataBlockInfo(
                            address=address + v4c.DZ_COMMON_SIZE,
                            block_type=block_type_,
                            original_size=original_size,
                            compressed_size=zip_size,
                            param=param,
                            block_limit=block_limit,
                        )

                # or a DataList
                elif id_string == b"##DL":
                    while address:
                        dl = DataList(address=address, stream=stream)
                        for i in range(dl.data_block_nr):
                            addr = getattr(dl, f"data_block_addr{i}")

                            stream.seek(addr)
                            id_string, block_len = COMMON_SHORT_u(stream.read(COMMON_SHORT_SIZE))

                            # can be a DataBlock
                            if id_string == block_type:
                                size = block_len - 24
                                if size:
                                    addr = addr + COMMON_SIZE

                                    # split the DTBLOCK into chucks of up to 32MB
                                    while True:
                                        if size > READ_CHUNK_SIZE:
                                            total_size -= READ_CHUNK_SIZE
                                            size -= READ_CHUNK_SIZE

                                            yield DataBlockInfo(
                                                address=addr,
                                                block_type=v4c.DT_BLOCK,
                                                original_size=READ_CHUNK_SIZE,
                                                compressed_size=READ_CHUNK_SIZE,
                                                param=0,
                                                block_limit=None,
                                            )
                                            addr += READ_CHUNK_SIZE
                                        else:
                                            if total_size < size:
                                                block_limit = total_size
                                            else:
                                                block_limit = None

                                            total_size -= size

                                            yield DataBlockInfo(
                                                address=addr,
                                                block_type=v4c.DT_BLOCK,
                                                original_size=size,
                                                compressed_size=size,
                                                param=0,
                                                block_limit=block_limit,
                                            )
                                            break

                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                stream.seek(addr + v4c.DZ_INFO_COMMON_OFFSET)
                                (
                                    zip_type,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_INFO_u(stream.read(v4c.DZ_COMMON_INFO_SIZE))

                                if original_size:
                                    if zip_type == v4c.FLAG_DZ_DEFLATE:
                                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                                        param = 0
                                    else:
                                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                    if total_size < original_size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= original_size
                                    yield DataBlockInfo(
                                        address=addr + v4c.DZ_COMMON_SIZE,
                                        block_type=block_type_,
                                        original_size=original_size,
                                        compressed_size=zip_size,
                                        param=param,
                                        block_limit=block_limit,
                                    )

                        address = dl.next_dl_addr

                # or a DataList
                elif id_string == b"##LD":
                    uses_ld = True
                    while address:
                        ld = ListData(address=address, stream=stream)
                        has_invalidation = ld.flags & v4c.FLAG_LD_INVALIDATION_PRESENT
                        for i in range(ld.data_block_nr):
                            addr = getattr(ld, f"data_block_addr{i}")

                            stream.seek(addr)
                            id_string, block_len = COMMON_SHORT_u(stream.read(COMMON_SHORT_SIZE))
                            # can be a DataBlock
                            if id_string == b"##DV":
                                size = block_len - 24
                                if size:
                                    if total_size < size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= size
                                    data_info = DataBlockInfo(
                                        address=addr + COMMON_SIZE,
                                        block_type=v4c.DT_BLOCK,
                                        original_size=size,
                                        compressed_size=size,
                                        param=0,
                                        block_limit=block_limit,
                                    )

                            # or a DataZippedBlock
                            elif id_string == b"##DZ":
                                stream.seek(addr + v4c.DZ_INFO_COMMON_OFFSET)
                                (
                                    zip_type,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_INFO_u(stream.read(v4c.DZ_COMMON_INFO_SIZE))

                                if original_size:
                                    if zip_type == v4c.FLAG_DZ_DEFLATE:
                                        block_type_ = v4c.DZ_BLOCK_DEFLATE
                                        param = 0
                                    else:
                                        block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                    if total_size < original_size:
                                        block_limit = total_size
                                    else:
                                        block_limit = None
                                    total_size -= original_size
                                    data_info = DataBlockInfo(
                                        address=addr + v4c.DZ_COMMON_SIZE,
                                        block_type=block_type_,
                                        original_size=original_size,
                                        compressed_size=zip_size,
                                        param=param,
                                        block_limit=block_limit,
                                    )

                            if has_invalidation:
                                inval_addr = typing.cast(int, ld[f"invalidation_bits_addr_{i}"])
                                if inval_addr:
                                    stream.seek(inval_addr)
                                    id_string, block_len = COMMON_SHORT_u(stream.read(COMMON_SHORT_SIZE))
                                    if id_string == b"##DI":
                                        size = block_len - 24
                                        if size:
                                            if inval_total_size < size:
                                                block_limit = inval_total_size
                                            else:
                                                block_limit = None
                                            inval_total_size -= size
                                            data_info.invalidation_block = InvalidationBlockInfo(
                                                address=inval_addr + COMMON_SIZE,
                                                block_type=v4c.DT_BLOCK,
                                                original_size=size,
                                                compressed_size=size,
                                                param=0,
                                                block_limit=block_limit,
                                            )
                                    else:
                                        stream.seek(inval_addr + v4c.DZ_INFO_COMMON_OFFSET)
                                        (
                                            zip_type,
                                            param,
                                            original_size,
                                            zip_size,
                                        ) = v4c.DZ_COMMON_INFO_u(stream.read(v4c.DZ_COMMON_INFO_SIZE))

                                        if original_size:
                                            if zip_type == v4c.FLAG_DZ_DEFLATE:
                                                block_type_ = v4c.DZ_BLOCK_DEFLATE
                                                param = 0
                                            else:
                                                block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                                            if inval_total_size < original_size:
                                                block_limit = inval_total_size
                                            else:
                                                block_limit = None
                                            inval_total_size -= original_size
                                            data_info.invalidation_block = InvalidationBlockInfo(
                                                address=inval_addr + v4c.DZ_COMMON_SIZE,
                                                block_type=block_type_,
                                                original_size=original_size,
                                                compressed_size=zip_size,
                                                param=param,
                                                block_limit=block_limit,
                                            )
                                else:
                                    data_info.invalidation_block = InvalidationBlockInfo(
                                        address=0,
                                        block_type=v4c.DT_BLOCK,
                                        original_size=0,
                                        compressed_size=0,
                                        param=0,
                                        all_valid=True,
                                    )

                            yield data_info
                        address = ld.next_ld_addr

                # or a header list
                elif id_string == b"##HL":
                    hl = HeaderList(address=address, stream=stream)
                    address = hl.first_dl_addr

                    yield from self._get_data_blocks_info(
                        address,
                        stream,
                        block_type,
                        mapped,
                        total_size,
                        inval_total_size,
                        record_size,
                    )

    def _get_signal_data_blocks_info(
        self,
        address: int,
        stream: FileLike | mmap.mmap,
    ) -> Iterator[SignalDataBlockInfo]:
        if not address:
            raise MdfException(f"Expected non-zero SDBLOCK address but got 0x{address:X}")

        stream.seek(address)
        id_string, block_len = COMMON_SHORT_u(stream.read(COMMON_SHORT_SIZE))

        # can be a DataBlock
        if id_string == b"##SD":
            size = block_len - 24
            if size:
                yield SignalDataBlockInfo(
                    address=address + COMMON_SIZE,
                    compressed_size=size,
                    original_size=size,
                    block_type=v4c.DT_BLOCK,
                )

        # or a DataZippedBlock
        elif id_string == b"##DZ":
            stream.seek(address + v4c.DZ_INFO_COMMON_OFFSET)
            (
                zip_type,
                param,
                original_size,
                zip_size,
            ) = v4c.DZ_COMMON_INFO_u(stream.read(v4c.DZ_COMMON_INFO_SIZE))

            if original_size:
                if zip_type == v4c.FLAG_DZ_DEFLATE:
                    block_type_ = v4c.DZ_BLOCK_DEFLATE
                    param = 0
                else:
                    block_type_ = v4c.DZ_BLOCK_TRANSPOSED

                yield SignalDataBlockInfo(
                    address=address + v4c.DZ_COMMON_SIZE,
                    block_type=block_type_,
                    original_size=original_size,
                    compressed_size=zip_size,
                    param=param,
                )

        # or a DataList
        elif id_string == b"##DL":
            while address:
                dl = DataList(address=address, stream=stream)
                for i in range(dl.data_block_nr):
                    addr = typing.cast(int, dl[f"data_block_addr{i}"])

                    stream.seek(addr)
                    id_string, block_len = COMMON_SHORT_u(stream.read(COMMON_SHORT_SIZE))

                    # can be a DataBlock
                    if id_string == b"##SD":
                        size = block_len - 24
                        if size:
                            yield SignalDataBlockInfo(
                                address=addr + COMMON_SIZE,
                                compressed_size=size,
                                original_size=size,
                                block_type=v4c.DT_BLOCK,
                            )

                    # or a DataZippedBlock
                    elif id_string == b"##DZ":
                        stream.seek(addr + v4c.DZ_INFO_COMMON_OFFSET)
                        (
                            zip_type,
                            param,
                            original_size,
                            zip_size,
                        ) = v4c.DZ_COMMON_INFO_u(stream.read(v4c.DZ_COMMON_INFO_SIZE))

                        if original_size:
                            if zip_type == v4c.FLAG_DZ_DEFLATE:
                                block_type_ = v4c.DZ_BLOCK_DEFLATE
                                param = 0
                            else:
                                block_type_ = v4c.DZ_BLOCK_TRANSPOSED
                            yield SignalDataBlockInfo(
                                address=addr + v4c.DZ_COMMON_SIZE,
                                block_type=block_type_,
                                original_size=original_size,
                                compressed_size=zip_size,
                                param=param,
                            )

                address = dl.next_dl_addr

        # or a header list
        elif id_string == b"##HL":
            hl = HeaderList(address=address, stream=stream)
            address = hl.first_dl_addr

            yield from self._get_signal_data_blocks_info(
                address,
                stream,
            )

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
                if ((source := self.groups[gp_idx].channels[cn_idx].source) is not None and source.name == source_name)
                or (
                    (acq_source := self.groups[gp_idx].channel_group.acq_source) is not None
                    and acq_source.name == source_name
                )
            )

        if source_path is not None:
            occurrences = (
                (gp_idx, cn_idx)
                for gp_idx, cn_idx in occurrences
                if ((source := self.groups[gp_idx].channels[cn_idx].source) is not None and source.path == source_path)
                or (
                    (acq_source := self.groups[gp_idx].channel_group.acq_source) is not None
                    and acq_source.path == source_path
                )
            )

        if acq_name is not None:
            occurrences = (
                (gp_idx, cn_idx)
                for gp_idx, cn_idx in occurrences
                if self.groups[gp_idx].channel_group.acq_name == acq_name
            )

        return occurrences

    def get_invalidation_bits(
        self,
        group_index: int,
        pos_invalidation_bit: int,
        fragment: Fragment,
    ) -> InvalidationArray:
        """Get invalidation indexes of the channels in the given group.

        Parameters
        ----------
        group_index : int
            Group index.
        pos_invalidation_bit : int
            Channel invalidation bit position.
        fragment : Fragment
            Data bytes as a Fragment.

        Returns
        -------
        invalidation_bits : iterable
            Iterable of valid channel indexes; if all are valid `None` is
            returned.
        """
        group = self.groups[group_index]

        data_bytes, offset, _count, invalidation_bytes = (
            fragment.data,
            fragment.record_offset,
            fragment.record_count,
            fragment.invalidation_data,
        )

        if invalidation_bytes is None:
            invalidation_bytes_nr = group.channel_group.invalidation_bytes_nr
            samples_byte_nr = group.channel_group.samples_byte_nr
            record = group.record
            if record is None:
                self._prepare_record(group)

            invalidation_bytes = get_channel_raw_bytes(
                data_bytes,
                samples_byte_nr + invalidation_bytes_nr,
                samples_byte_nr,
                invalidation_bytes_nr,
            )

        key = (group_index, offset, _count, pos_invalidation_bit)
        if key not in self._invalidation_cache:
            self._invalidation_cache[key] = InvalidationArray(
                get_invalidation_bits_array(
                    invalidation_bytes, group.channel_group.invalidation_bytes_nr, pos_invalidation_bit
                ),
                (group_index, pos_invalidation_bit),
            )

        return self._invalidation_cache[key]

    @overload
    def append(
        self,
        signals: DataFrame,
        acq_name: str | None = None,
        acq_source: Source | None = None,
        comment: str = "Python",
        common_timebase: bool = False,
        units: dict[str, str] | None = None,
    ) -> None: ...

    @overload
    def append(
        self,
        signals: list[Signal] | Signal,
        acq_name: str | None = None,
        acq_source: Source | None = None,
        comment: str = "Python",
        common_timebase: bool = False,
        units: dict[str, str] | None = None,
    ) -> int: ...

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
            *utf-8* encoded.
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
        >>> mdf = MDF(version='4.10')
        >>> mdf.append([s1, s2, s3], comment='created by asammdf')

        Case 2: VTAB conversions from channels inside another file.

        >>> mdf1 = MDF('in.mf4')
        >>> ch1 = mdf1.get("Channel1_VTAB")
        >>> ch2 = mdf1.get("Channel2_VTABR")
        >>> mdf2 = MDF('out.mf4')
        >>> mdf2.append([ch1, ch2], comment='created by asammdf')
        >>> mdf2.append(ch1, comment='just a single channel')
        >>> df = pd.DataFrame.from_dict({'s1': np.array([1, 2, 3, 4, 5]), 's2': np.array([-1, -2, -3, -4, -5])})
        >>> units = {'s1': 'V', 's2': 'A'}
        >>> mdf2.append(df, units=units)
        """
        source_block = SourceInformation.from_common_source(acq_source) if acq_source else None

        if isinstance(signals, Signal):
            signals = [signals]
        elif isinstance(signals, DataFrame):
            self._append_dataframe(
                signals,
                acq_name=acq_name,
                acq_source=source_block,
                comment=comment,
                units=units,
            )
            return None

        if not signals:
            return None

        # check if the signals have a common timebase
        # if not interpolate the signals using the union of all timebases

        supports_virtual_channels = self.version >= "4.10"
        virtual_master = False
        virtual_master_conversion = None

        if signals:
            t_ = signals[0].timestamps

            if (
                supports_virtual_channels
                and all(sig.flags & sig.Flags.virtual_master for sig in signals)
                and all(np.array_equal(sig.timestamps, t_) for sig in signals)
            ):
                virtual_master = True
                virtual_master_conversion = signals[0].virtual_master_conversion
                t = t_

            else:
                if not common_timebase:
                    for s in signals[1:]:
                        if not array_equal(s.timestamps, t_):
                            different = True
                            break
                    else:
                        different = False

                    if different:
                        times = [s.timestamps for s in signals]
                        t = unique(concatenate(times))
                        if t.dtype != float64:
                            t = t.astype(float64)
                        signals = [
                            s.interp(
                                t,
                                integer_interpolation_mode=self._integer_interpolation,
                                float_interpolation_mode=self._float_interpolation,
                            )
                            for s in signals
                        ]
                    else:
                        t = t_
                else:
                    t = t_
        else:
            t = np.array([])

        if self.version >= "4.20" and self._column_storage:
            return self._append_column_oriented(signals, acq_name=acq_name, acq_source=source_block, comment=comment)

        dg_cntr = len(self.groups)

        gp = Group(DataGroup())
        gp_sdata = gp.signal_data = []
        gp_channels = gp.channels = []
        gp_dep = gp.channel_dependencies = []
        gp_sig_types = gp.signal_types = []

        cycles_nr = len(t)

        # channel group
        cg_kwargs: ChannelGroupKwargs = {"cycles_nr": cycles_nr, "samples_byte_nr": 0}
        gp.channel_group = ChannelGroup(**cg_kwargs)
        gp.channel_group.acq_name = acq_name
        gp.channel_group.acq_source = source_block
        gp.channel_group.comment = comment
        record = gp.record = []
        inval_bits: dict[tuple[int, int], list[InvalidationArray] | InvalidationArray] = {
            InvalidationArray.ORIGIN_UNKNOWN: []
        }

        if any(sig.invalidation_bits is not None for sig in signals):
            invalidation_bytes_nr = 1
            gp.channel_group.invalidation_bytes_nr = invalidation_bytes_nr

        else:
            invalidation_bytes_nr = 0

        self.groups.append(gp)

        fields: list[tuple[bytes | NDArray[Any], int]] = []

        ch_cntr = 0
        offset = 0

        defined_texts: dict[str, int] = {}
        si_map = self._si_map

        # setup all blocks related to the time master channel

        file = self._tempfile
        tell = file.tell
        seek = file.seek

        seek(0, 2)

        if signals:
            master_metadata = signals[0].master_metadata
        else:
            master_metadata = None
        if master_metadata:
            time_name, sync_type = master_metadata
            match sync_type:
                case 0 | 1:
                    time_unit = "s"
                case 2:
                    time_unit = "deg"
                case 3:
                    time_unit = "m"
                case 4:
                    time_unit = "index"
        else:
            time_name, sync_type = "time", v4c.SYNC_TYPE_TIME
            time_unit = "s"

        gp.channel_group.acq_source = source_block

        if signals:
            # time channel

            if virtual_master:
                cn_kwargs: ChannelKwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VIRTUAL_MASTER,
                    "data_type": v4c.DATA_TYPE_UNSIGNED_INTEL,
                    "sync_type": sync_type,
                    "byte_offset": 0,
                    "bit_offset": 0,
                    "bit_count": 0,
                }

                ch = Channel(**cn_kwargs)
                ch.unit = time_unit
                ch.name = time_name
                ch.source = source_block
                ch.dtype_fmt = t.dtype
                ch.conversion = conversion_transfer(virtual_master_conversion, version=4)
                name = time_name

                gp_channels.append(ch)

                gp_sdata.append(None)
                self.channels_db.add(name, (dg_cntr, ch_cntr))
                self.masters_db[dg_cntr] = 0

                # time channel doesn't have channel dependencies
                gp_dep.append(None)

                ch_cntr += 1

                gp_sig_types.append((v4c.SIGNAL_TYPE_VIRTUAL, 0))

            else:
                t_type, t_size = fmt_to_datatype_v4(t.dtype, t.shape)
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_MASTER,
                    "data_type": t_type,
                    "sync_type": sync_type,
                    "byte_offset": 0,
                    "bit_offset": 0,
                    "bit_count": t_size,
                }

                ch = Channel(**cn_kwargs)
                ch.unit = time_unit
                ch.name = time_name
                ch.source = source_block
                ch.dtype_fmt = t.dtype
                name = time_name

                gp_channels.append(ch)

                gp_sdata.append(None)
                self.channels_db.add(name, (dg_cntr, ch_cntr))
                self.masters_db[dg_cntr] = 0

                record.append(
                    (
                        t.dtype,
                        t.dtype.itemsize,
                        0,
                        0,
                    )
                )

                # time channel doesn't have channel dependencies
                gp_dep.append(None)

                if not t.flags["C_CONTIGUOUS"]:
                    t = np.ascontiguousarray(t)
                fields.append((t, t.itemsize))

                offset += t_size // 8
                ch_cntr += 1

                gp_sig_types.append((v4c.SIGNAL_TYPE_SCALAR, t_size // 8))

        for signal in signals:
            sig = signal
            samples = sig.samples
            sig_dtype = samples.dtype
            sig_shape = samples.shape
            names = sig_dtype.names
            name = signal.name

            if names is None:
                if supports_virtual_channels and sig.flags & sig.Flags.virtual:
                    sig_type = v4c.SIGNAL_TYPE_VIRTUAL
                else:
                    sig_type = v4c.SIGNAL_TYPE_SCALAR
                    if sig_dtype.kind in "SV":
                        sig_type = v4c.SIGNAL_TYPE_STRING
            else:
                prepare_record = False
                if names in (v4c.CANOPEN_TIME_FIELDS, v4c.CANOPEN_DATE_FIELDS):
                    sig_type = v4c.SIGNAL_TYPE_CANOPEN
                elif names[0] != sig.name:
                    sig_type = v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v4c.SIGNAL_TYPE_ARRAY

            # first add the signals in the simple signal list
            if sig_type == v4c.SIGNAL_TYPE_SCALAR:
                # compute additional byte offset for large records size
                s_type, s_size = fmt_to_datatype_v4(sig_dtype, sig_shape)

                if (s_type, s_size) == (v4c.DATA_TYPE_BYTEARRAY, 0):
                    offsets = typing.cast(NDArray[np.uint64], arange(len(samples), dtype=uint64) * (sig_shape[1] + 4))

                    values = [
                        full(len(samples), sig_shape[1], dtype=uint32),
                        samples,
                    ]

                    types_ = [("o", uint32), ("s", sig_dtype, sig_shape[1:])]

                    records = np.rec.fromarrays(values, dtype=types_)

                    data_size = len(records) * records.itemsize
                    if data_size:
                        data_addr = tell()
                        info = SignalDataBlockInfo(
                            address=data_addr,
                            compressed_size=data_size,
                            original_size=data_size,
                            location=v4c.LOCATION_TEMPORARY_FILE,
                        )
                        gp_sdata.append(
                            (
                                [info],
                                iter(EMPTY_TUPLE),
                            )
                        )
                        records.tofile(file)
                    else:
                        data_addr = 0
                        gp_sdata.append(
                            (
                                [],
                                iter(EMPTY_TUPLE),
                            )
                        )

                    byte_size = 8
                    cn_kwargs = {
                        "channel_type": v4c.CHANNEL_TYPE_VLSD,
                        "bit_count": 64,
                        "byte_offset": offset,
                        "bit_offset": 0,
                        "data_type": s_type,
                        "flags": 0,
                    }

                    if invalidation_bytes_nr and signal.invalidation_bits is not None:
                        if (origin := signal.invalidation_bits.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                            invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits[origin])
                            invalidation_arrays.append(signal.invalidation_bits)
                        else:
                            inval_bits[origin] = signal.invalidation_bits
                        cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                        cn_kwargs["pos_invalidation_bit"] = origin[1]

                    ch = Channel(**cn_kwargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_names = signal.display_names
                    ch.dtype_fmt = np.dtype("<u8")

                    # conversions for channel
                    conversion = conversion_transfer(signal.conversion, version=4)
                    if signal.raw:
                        ch.conversion = conversion

                    # source for channel
                    source = signal.source
                    if source:
                        if source in si_map:
                            ch.source = si_map[source]
                        else:
                            new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                            new_source.name = source.name
                            new_source.path = source.path
                            new_source.comment = source.comment

                            si_map[source] = new_source

                            ch.source = new_source

                    gp_channels.append(ch)

                    record.append(
                        (
                            np.dtype(uint64),
                            8,
                            offset,
                            0,
                        )
                    )
                    gp_sig_types.append((sig_type, 8))

                    offset += byte_size

                    entry = (dg_cntr, ch_cntr)
                    self.channels_db.add(name, entry)
                    for _name in ch.display_names:
                        self.channels_db.add(_name, entry)

                    if not offsets.flags["C_CONTIGUOUS"]:
                        offsets = np.ascontiguousarray(offsets)
                    fields.append((offsets, 8))

                    ch_cntr += 1

                    # simple channels don't have channel dependencies
                    gp_dep.append(None)

                else:
                    byte_size = s_size // 8 or 1
                    data_block_addr = 0

                    gp_sig_types.append((sig_type, byte_size))

                    if sig_dtype.kind == "u" and signal.bit_count <= 4:
                        s_size = signal.bit_count

                    if signal.flags & signal.Flags.stream_sync:
                        channel_type = v4c.CHANNEL_TYPE_SYNC
                        if signal.attachment:
                            at_data, at_name, hash_sum = typing.cast(tuple[bytes, Path, bytes], signal.attachment)
                            attachment_index = self.attach(
                                at_data,
                                at_name,
                                hash_sum,
                                mime="video/avi",
                                embedded=False,
                            )
                            attachment = attachment_index
                        else:
                            attachment = None

                        sync_type = v4c.SYNC_TYPE_TIME
                    else:
                        channel_type = v4c.CHANNEL_TYPE_VALUE
                        sync_type = v4c.SYNC_TYPE_NONE

                        if signal.attachment:
                            at_data, at_name, hash_sum = typing.cast(tuple[bytes, Path, bytes], signal.attachment)

                            attachment_index = self.attach(at_data, at_name, hash_sum)
                            attachment = attachment_index
                        else:
                            attachment = None

                    cn_kwargs = {
                        "channel_type": channel_type,
                        "sync_type": sync_type,
                        "bit_count": s_size,
                        "byte_offset": offset,
                        "bit_offset": 0,
                        "data_type": s_type,
                        "flags": 0,
                    }

                    if attachment is not None:
                        cn_kwargs["attachment_addr"] = 0

                    if invalidation_bytes_nr and signal.invalidation_bits is not None:
                        if (origin := signal.invalidation_bits.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                            invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits[origin])
                            invalidation_arrays.append(signal.invalidation_bits)
                        else:
                            inval_bits[origin] = signal.invalidation_bits
                        cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                        cn_kwargs["pos_invalidation_bit"] = origin[1]

                    ch = Channel(**cn_kwargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_names = signal.display_names
                    if len(sig_shape) > 1:
                        ch.dtype_fmt = np.dtype((sig_dtype, sig_shape[1:]))
                    else:
                        ch.dtype_fmt = sig_dtype
                    ch.attachment = attachment

                    # conversions for channel
                    if signal.raw:
                        ch.conversion = conversion_transfer(signal.conversion, version=4)

                    # source for channel

                    source = signal.source
                    if source:
                        if source in si_map:
                            ch.source = si_map[source]
                        else:
                            new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                            new_source.name = source.name
                            new_source.path = source.path
                            new_source.comment = source.comment

                            si_map[source] = new_source

                            ch.source = new_source

                    gp_channels.append(ch)

                    record.append(
                        (
                            ch.dtype_fmt,
                            ch.dtype_fmt.itemsize,
                            offset,
                            0,
                        )
                    )

                    offset += byte_size

                    if not samples.flags["C_CONTIGUOUS"]:
                        samples = np.ascontiguousarray(samples)
                    fields.append((samples, byte_size))

                    gp_sdata.append(None)
                    entry = (dg_cntr, ch_cntr)
                    self.channels_db.add(name, entry)
                    for _name in ch.display_names:
                        self.channels_db.add(_name, entry)

                    ch_cntr += 1

                    # simple channels don't have channel dependencies
                    gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_VIRTUAL:
                channel_type = v4c.CHANNEL_TYPE_VIRTUAL
                sync_type = v4c.SYNC_TYPE_NONE

                cn_kwargs = {
                    "channel_type": channel_type,
                    "sync_type": sync_type,
                    "bit_count": 0,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": v4c.DATA_TYPE_UNSIGNED_INTEL,
                    "flags": 0,
                }

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_names = signal.display_names

                # conversions for channel
                ch.conversion = conversion_transfer(signal.virtual_conversion, version=4)

                # source for channel

                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)
                gp_sig_types.append((sig_type, 0))

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in ch.display_names:
                    self.channels_db.add(_name, entry)

                ch_cntr += 1

                # virtual channels don't have channel dependencies
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_CANOPEN:
                vals: bytes | NDArray[Any]
                if names == v4c.CANOPEN_TIME_FIELDS:
                    record.append(
                        (
                            np.dtype("V6"),
                            6,
                            offset,
                            0,
                        )
                    )

                    if not signal.samples.flags["C_CONTIGUOUS"]:
                        vals = np.ascontiguousarray(signal.samples)
                    else:
                        vals = signal.samples

                    fields.append((vals, 6))
                    byte_size = 6
                    s_type = v4c.DATA_TYPE_CANOPEN_TIME
                    s_dtype = np.void(6)
                    gp_sig_types.append((sig_type, 6))

                else:
                    record.append(
                        (
                            np.dtype("V7"),
                            7,
                            offset,
                            0,
                        )
                    )

                    arrays: list[NDArray[Any]] = []
                    for field in ("ms", "min", "hour", "day", "month", "year"):
                        if field == "hour":
                            arrays.append(signal.samples[field] + (signal.samples["summer_time"] << 7))
                        elif field == "day":
                            arrays.append(signal.samples[field] + (signal.samples["day_of_week"] << 4))
                        else:
                            arrays.append(signal.samples[field])
                    vals = np.rec.fromarrays(arrays)

                    if not vals.flags["C_CONTIGUOUS"]:
                        vals = np.ascontiguousarray(vals)

                    fields.append((vals, 7))
                    byte_size = 7
                    s_type = v4c.DATA_TYPE_CANOPEN_DATE
                    s_dtype = np.void(7)
                    gp_sig_types.append((sig_type, 7))

                s_size = byte_size * 8

                # there is no channel dependency
                gp_dep.append(None)

                # add channel block
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }
                if invalidation_bytes_nr and signal.invalidation_bits is not None:
                    if (origin := signal.invalidation_bits.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                        invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits[origin])
                        invalidation_arrays.append(signal.invalidation_bits)
                    else:
                        inval_bits[origin] = signal.invalidation_bits
                    cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    cn_kwargs["pos_invalidation_bit"] = origin[1]

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_names = signal.display_names
                ch.dtype_fmt = np.dtype(s_dtype)

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                offset += byte_size

                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in ch.display_names:
                    self.channels_db.add(_name, entry)

                gp_sdata.append(None)

                ch_cntr += 1

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                (
                    offset,
                    dg_cntr,
                    ch_cntr,
                    struct_self,
                    new_fields,
                ) = self._append_structure_composition(
                    gp,
                    signal,
                    offset,
                    dg_cntr,
                    ch_cntr,
                    defined_texts,
                    invalidation_bytes_nr,
                    inval_bits,
                )
                fields.extend(new_fields)
                gp_sig_types.append((sig_type, signal.samples.dtype.itemsize))

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                if names is None:
                    raise RuntimeError("'names' is None")
                # here we have channel arrays or mdf v3 channel dependencies
                samples = signal.samples[names[0]]
                shape = samples.shape[1:]

                if len(names) > 1 or len(shape) > 1:
                    # add channel dependency block for composed parent channel
                    dims_nr = len(shape)
                    names_nr = len(names)

                    if names_nr == 0:
                        ca_kwargs: ChannelArrayBlockKwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_FIXED_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    elif len(names) == 1:
                        ca_kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_ARRAY,
                            "flags": 0,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    else:
                        ca_kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    parent_dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([parent_dep])

                else:
                    # add channel dependency block for composed parent channel
                    ca_kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_ARRAY,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    parent_dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([parent_dep])

                # first we add the structure channel
                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape, True)

                # add channel block
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }

                if invalidation_bytes_nr and signal.invalidation_bits is not None:
                    if (origin := signal.invalidation_bits.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                        invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits[origin])
                        invalidation_arrays.append(signal.invalidation_bits)
                    else:
                        inval_bits[origin] = signal.invalidation_bits
                    cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    cn_kwargs["pos_invalidation_bit"] = origin[1]

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_names = signal.display_names
                ch.dtype_fmt = samples.dtype

                record.append(
                    (
                        samples.dtype,
                        samples.dtype.itemsize,
                        offset,
                        0,
                    )
                )

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                size = s_size // 8
                for dim in shape:
                    size *= dim
                offset += size

                if not samples.flags["C_CONTIGUOUS"]:
                    samples = np.ascontiguousarray(samples)
                fields.append((samples, size))
                gp_sig_types.append((sig_type, size))

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in ch.display_names:
                    self.channels_db.add(_name, entry)

                ch_cntr += 1

                for name in names[1:]:
                    samples = signal.samples[name]
                    shape = samples.shape[1:]

                    # add channel dependency block
                    ca_kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([dep])

                    # add components channel
                    s_type, s_size = fmt_to_datatype_v4(samples.dtype, ())
                    byte_size = s_size // 8 or 1
                    cn_kwargs = {
                        "channel_type": v4c.CHANNEL_TYPE_VALUE,
                        "bit_count": s_size,
                        "byte_offset": offset,
                        "bit_offset": 0,
                        "data_type": s_type,
                        "flags": 0,
                    }

                    if invalidation_bytes_nr and signal.invalidation_bits is not None:
                        if (origin := signal.invalidation_bits.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                            invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits[origin])
                            invalidation_arrays.append(signal.invalidation_bits)
                        else:
                            inval_bits[origin] = signal.invalidation_bits
                        cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                        cn_kwargs["pos_invalidation_bit"] = origin[1]

                    ch = Channel(**cn_kwargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_names = signal.display_names
                    ch.dtype_fmt = samples.dtype

                    record.append(
                        (
                            samples.dtype,
                            samples.dtype.itemsize,
                            offset,
                            0,
                        )
                    )

                    gp_channels.append(ch)

                    entry = dg_cntr, ch_cntr
                    parent_dep.axis_channels.append(entry)
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    if not samples.flags["C_CONTIGUOUS"]:
                        samples = np.ascontiguousarray(samples)

                    fields.append((samples, byte_size))

                    gp_sdata.append(None)
                    self.channels_db.add(name, entry)

                    ch_cntr += 1

            else:
                encoding = signal.encoding
                samples = signal.samples
                sig_dtype = samples.dtype

                match encoding:
                    case "utf-8":
                        data_type = v4c.DATA_TYPE_STRING_UTF_8
                    case "latin-1":
                        data_type = v4c.DATA_TYPE_STRING_LATIN_1
                    case "utf-16-be":
                        data_type = v4c.DATA_TYPE_STRING_UTF_16_BE
                    case "utf-16-le":
                        data_type = v4c.DATA_TYPE_STRING_UTF_16_LE
                    case _:
                        raise MdfException(f'wrong encoding "{encoding}" for string signal')

                if self.compact_vlsd:
                    buffers: list[bytes] = []
                    offsets_list: list[int] = []
                    off = 0
                    if encoding == "utf-16-le":
                        for elem in samples:
                            offsets_list.append(off)
                            size = len(elem)
                            if size % 2:
                                size += 1
                                elem = elem + b"\0"
                            buffers.append(UINT32_p(size))
                            buffers.append(elem)
                            off += size + 4
                    else:
                        for elem in samples:
                            offsets_list.append(off)
                            size = len(elem)
                            buffers.append(UINT32_p(size))
                            buffers.append(elem)
                            off += size + 4

                    data_size = off
                    offsets = array(offsets_list, dtype=uint64)
                    if data_size:
                        data_addr = tell()
                        info = SignalDataBlockInfo(
                            address=data_addr,
                            compressed_size=data_size,
                            original_size=data_size,
                            location=v4c.LOCATION_TEMPORARY_FILE,
                        )
                        gp_sdata.append(
                            (
                                [info],
                                iter(EMPTY_TUPLE),
                            )
                        )
                        file.seek(0, 2)
                        file.write(b"".join(buffers))
                    else:
                        data_addr = 0
                        gp_sdata.append(
                            (
                                [],
                                iter(EMPTY_TUPLE),
                            )
                        )
                else:
                    offsets = typing.cast(
                        NDArray[np.uint64], arange(len(samples), dtype=uint64) * (signal.samples.itemsize + 4)
                    )

                    values = [
                        full(len(samples), samples.itemsize, dtype=uint32),
                        samples,
                    ]

                    types_ = [("o", uint32), ("s", sig_dtype)]

                    records = np.rec.fromarrays(values, dtype=types_)

                    data_size = len(records) * records.itemsize
                    if data_size:
                        data_addr = tell()
                        info = SignalDataBlockInfo(
                            address=data_addr,
                            compressed_size=data_size,
                            original_size=data_size,
                            location=v4c.LOCATION_TEMPORARY_FILE,
                        )
                        gp_sdata.append(
                            (
                                [info],
                                iter(EMPTY_TUPLE),
                            )
                        )
                        records.tofile(file)
                    else:
                        data_addr = 0
                        gp_sdata.append(
                            (
                                [],
                                iter(EMPTY_TUPLE),
                            )
                        )

                # compute additional byte offset for large records size
                byte_size = 8
                gp_sig_types.append((sig_type, 8))
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VLSD,
                    "bit_count": 64,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": data_type,
                    "flags": 0,
                }

                if invalidation_bytes_nr:
                    if signal.invalidation_bits is not None:
                        if (origin := signal.invalidation_bits.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                            invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits[origin])
                            invalidation_arrays.append(signal.invalidation_bits)
                        else:
                            inval_bits[origin] = signal.invalidation_bits
                        cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                        cn_kwargs["pos_invalidation_bit"] = origin[1]

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_names = signal.display_names
                ch.dtype_fmt = np.dtype("<u8")

                # conversions for channel
                conversion = conversion_transfer(signal.conversion, version=4)
                if signal.raw:
                    ch.conversion = conversion

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                record.append(
                    (
                        np.dtype(uint64),
                        8,
                        offset,
                        0,
                    )
                )

                offset += byte_size

                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in ch.display_names:
                    self.channels_db.add(_name, entry)

                if not offsets.flags["C_CONTIGUOUS"]:
                    offsets = np.ascontiguousarray(offsets)

                fields.append((offsets, 8))

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

        if invalidation_bytes_nr:
            unknown_origin = typing.cast(list[InvalidationArray], inval_bits.pop(InvalidationArray.ORIGIN_UNKNOWN))
            inval_arrays = typing.cast(dict[tuple[int, int], InvalidationArray], inval_bits)

            _pos_map = {key: idx for idx, (_, key) in enumerate(inval_arrays)}

            _unknown_pos_map = deque(list(range(len(inval_arrays), len(inval_arrays) + len(unknown_origin))))

            invalidation_bits_list = typing.cast(list[NDArray[np.bool]], list(inval_arrays.values()) + unknown_origin)
            invalidation_bytes_nr = len(invalidation_bits_list)

            for _ in range(8 - invalidation_bytes_nr % 8):
                invalidation_bits_list.append(zeros(cycles_nr, dtype=bool))

            invalidation_bits_list.reverse()
            invalidation_bytes_nr = len(invalidation_bits_list) // 8

            gp.channel_group.invalidation_bytes_nr = invalidation_bytes_nr

            bytes_array = np.fliplr(
                np.packbits(array(invalidation_bits_list).T).reshape((cycles_nr, invalidation_bytes_nr))
            ).ravel()

            if self.version < "4.20":
                fields.append((bytes_array, invalidation_bytes_nr))

            for ch in gp.channels:
                if ch.flags & v4c.FLAG_CN_INVALIDATION_PRESENT:
                    if (pos_invalidation_bit := ch.pos_invalidation_bit) == InvalidationArray.ORIGIN_UNKNOWN[1]:
                        ch.pos_invalidation_bit = _unknown_pos_map.popleft()
                    else:
                        ch.pos_invalidation_bit = _pos_map[pos_invalidation_bit]
        else:
            bytes_array = None

        gp.channel_group.cycles_nr = cycles_nr
        gp.channel_group.samples_byte_nr = offset

        virtual_group = VirtualChannelGroup()
        self.virtual_groups[dg_cntr] = virtual_group
        self.virtual_groups_map[dg_cntr] = dg_cntr
        virtual_group.groups.append(dg_cntr)
        virtual_group.record_size = offset + invalidation_bytes_nr
        virtual_group.cycles_nr = cycles_nr

        # data group
        gp.sorted = True

        data_block = data_block_from_arrays(fields, cycles_nr, THREAD_COUNT)
        size = len(data_block)
        data_block_view = memoryview(data_block)

        del fields

        record_size = offset + invalidation_bytes_nr

        if size:
            if self.version < "4.20":
                block_size = 32 * 1024 * 1024 // record_size * record_size

                count = ceil(size / block_size)

                for i in range(count):
                    data_ = data_block_view[i * block_size : (i + 1) * block_size]
                    raw_size = len(data_)
                    data_ = lz_compress(data_, store_size=True)

                    size = len(data_)
                    data_address = self._tempfile.tell()
                    self._tempfile.write(data_)

                    gp.data_blocks.append(
                        DataBlockInfo(
                            address=data_address,
                            block_type=v4c.DZ_BLOCK_LZ,
                            original_size=raw_size,
                            compressed_size=size,
                            param=0,
                        )
                    )

            else:
                gp.uses_ld = True
                data_address = tell()

                raw_size = len(data_block_view)
                data = lz_compress(data_block_view, store_size=True)

                size = len(data)
                self._tempfile.write(data)

                gp.data_blocks.append(
                    DataBlockInfo(
                        address=data_address,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=0,
                    )
                )

                if bytes_array is not None:
                    addr = tell()
                    data = bytes_array.tobytes()
                    raw_size = len(data)
                    data = lz_compress(data, store_size=True)
                    size = len(data)
                    self._tempfile.write(data)

                    gp.data_blocks[-1].invalidation_block = InvalidationBlockInfo(
                        address=addr,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=None,
                    )

        gp.data_location = v4c.LOCATION_TEMPORARY_FILE

        return dg_cntr

    def _append_column_oriented(
        self,
        signals: list[Signal],
        acq_name: str | None = None,
        acq_source: SourceInformation | None = None,
        comment: str = "",
    ) -> int:
        defined_texts: dict[str, int] = {}
        si_map = self._si_map

        # setup all blocks related to the time master channel

        file = self._tempfile
        tell = file.tell
        seek = file.seek
        write = file.write

        seek(0, 2)

        dg_cntr = initial_dg_cntr = len(self.groups)

        # add the master group

        gp = Group(DataGroup())
        gp_sdata = gp.signal_data = []
        gp_channels = gp.channels = []
        gp_dep = gp.channel_dependencies = []
        gp_sig_types: list[int] = []
        gp.signal_types = gp_sig_types
        gp.uses_ld = True
        gp.sorted = True
        record = gp.record = []

        timestamps = signals[0].timestamps

        cycles_nr = len(timestamps)

        # channel group
        cg_kwargs: ChannelGroupKwargs = {"cycles_nr": cycles_nr, "samples_byte_nr": 0}
        gp.channel_group = remote_master_channel_group = ChannelGroup(**cg_kwargs)
        gp.channel_group.acq_name = acq_name
        gp.channel_group.acq_source = acq_source
        gp.channel_group.comment = comment

        self.groups.append(gp)

        ch_cntr = 0
        types: list[tuple[str, np.dtype[Any], tuple[int, ...]] | DTypeLike] = []
        ch_cntr = 0
        offset = 0

        prepare_record = True
        source_block = acq_source

        master_metadata = signals[0].master_metadata
        if master_metadata:
            time_name, sync_type = master_metadata
            match sync_type:
                case 0 | 1:
                    time_unit = "s"
                case 2:
                    time_unit = "deg"
                case 3:
                    time_unit = "m"
                case 4:
                    time_unit = "index"
                case _:
                    raise RuntimeError(f"unexpected sync_type '{sync_type}'")
        else:
            time_name, sync_type = "time", v4c.SYNC_TYPE_TIME
            time_unit = "s"

        gp.channel_group.acq_source = source_block
        # time channel
        t_type, t_size = fmt_to_datatype_v4(timestamps.dtype, timestamps.shape)
        cn_kwargs: ChannelKwargs = {
            "channel_type": v4c.CHANNEL_TYPE_MASTER,
            "data_type": t_type,
            "sync_type": sync_type,
            "byte_offset": 0,
            "bit_offset": 0,
            "bit_count": t_size,
        }

        ch = Channel(**cn_kwargs)
        ch.unit = time_unit
        ch.name = time_name
        ch.source = source_block
        ch.dtype_fmt = timestamps.dtype
        name = time_name

        gp_channels.append(ch)

        gp_sdata.append(None)
        self.channels_db.add(name, (dg_cntr, ch_cntr))
        self.masters_db[dg_cntr] = 0

        record.append(
            (
                timestamps.dtype,
                timestamps.dtype.itemsize,
                offset,
                0,
            )
        )

        # time channel doesn't have channel dependencies
        gp_dep.append(None)

        types.append((name, timestamps.dtype))

        offset += t_size // 8
        ch_cntr += 1

        gp_sig_types.append(0)

        gp.channel_group.samples_byte_nr = offset

        # data block
        gp.sorted = True

        size = cycles_nr * timestamps.itemsize

        cg_master_index = dg_cntr

        virtual_group = VirtualChannelGroup()
        self.virtual_groups[cg_master_index] = virtual_group
        self.virtual_groups_map[dg_cntr] = dg_cntr
        virtual_group.groups.append(dg_cntr)
        virtual_group.record_size = offset
        virtual_group.cycles_nr = cycles_nr

        dg_cntr += 1

        if size:
            data_address = tell()
            gp.data_location = v4c.LOCATION_TEMPORARY_FILE
            write(timestamps.tobytes())

            chunk = self._write_fragment_size // timestamps.itemsize
            chunk *= timestamps.itemsize

            while size:
                if size > chunk:
                    gp.data_blocks.append(
                        DataBlockInfo(
                            address=data_address,
                            block_type=v4c.DT_BLOCK,
                            original_size=chunk,
                            compressed_size=chunk,
                            param=0,
                        )
                    )
                    data_address += chunk
                    size -= chunk
                else:
                    gp.data_blocks.append(
                        DataBlockInfo(
                            address=data_address,
                            block_type=v4c.DT_BLOCK,
                            original_size=size,
                            compressed_size=size,
                            param=0,
                        )
                    )
                    size = 0
        else:
            gp.data_location = v4c.LOCATION_TEMPORARY_FILE

        for signal in signals:
            gp = Group(DataGroup())
            gp_sdata = gp.signal_data = []
            gp_channels = gp.channels = []
            gp_dep = gp.channel_dependencies = []
            gp.signal_types = gp_sig_types = []
            gp.sorted = True
            gp.uses_ld = True
            record = gp.record = []

            # channel group
            cg_kwargs = {
                "cycles_nr": cycles_nr,
                "samples_byte_nr": 0,
                "flags": v4c.FLAG_CG_REMOTE_MASTER,
            }
            gp.channel_group = ChannelGroup(**cg_kwargs)
            gp.channel_group.acq_name = acq_name
            gp.channel_group.acq_source = acq_source
            gp.channel_group.comment = remote_master_channel_group.comment
            gp.channel_group.cg_master_index = cg_master_index

            self.groups.append(gp)

            types = []
            ch_cntr = 0
            offset = 0
            field_names = UniqueDB()

            sig = signal
            samples = sig.samples
            sig_dtype = samples.dtype
            sig_shape = samples.shape
            names = sig_dtype.names
            name = signal.name

            if names is None:
                sig_type = v4c.SIGNAL_TYPE_SCALAR
                if sig_dtype.kind in "SV":
                    sig_type = v4c.SIGNAL_TYPE_STRING
            else:
                if names in (v4c.CANOPEN_TIME_FIELDS, v4c.CANOPEN_DATE_FIELDS):
                    sig_type = v4c.SIGNAL_TYPE_CANOPEN
                elif names[0] != sig.name:
                    sig_type = v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v4c.SIGNAL_TYPE_ARRAY

            gp_sig_types.append(sig_type)

            # first add the signals in the simple signal list
            if sig_type == v4c.SIGNAL_TYPE_SCALAR:
                # compute additional byte offset for large records size
                s_type, s_size = fmt_to_datatype_v4(sig_dtype, sig_shape)

                byte_size = s_size // 8 or 1

                if sig_dtype.kind == "u" and signal.bit_count <= 4:
                    s_size = signal.bit_count

                if signal.flags & signal.Flags.stream_sync:
                    channel_type = v4c.CHANNEL_TYPE_SYNC
                    if signal.attachment:
                        at_data, at_name, hash_sum = typing.cast(tuple[bytes, Path, bytes], signal.attachment)
                        attachment_addr = self.attach(at_data, at_name, hash_sum, mime="video/avi", embedded=False)
                        data_block_addr = attachment_addr
                    else:
                        data_block_addr = 0

                    sync_type = v4c.SYNC_TYPE_TIME
                else:
                    channel_type = v4c.CHANNEL_TYPE_VALUE
                    data_block_addr = 0
                    sync_type = v4c.SYNC_TYPE_NONE

                cn_kwargs = {
                    "channel_type": channel_type,
                    "sync_type": sync_type,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }

                if signal.invalidation_bits is not None:
                    invalidation_bits = signal.invalidation_bits
                    cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    cn_kwargs["pos_invalidation_bit"] = 0
                else:
                    invalidation_bits = None

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_names = signal.display_names

                # conversions for channel
                if signal.raw:
                    ch.conversion = conversion_transfer(signal.conversion, version=4)

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in ch.display_names:
                    self.channels_db.add(_name, entry)

                _shape = sig_shape[1:]
                types.append((name, sig_dtype, _shape))
                gp.single_channel_dtype = ch.dtype_fmt = np.dtype((sig_dtype, _shape))

                record.append(
                    (
                        ch.dtype_fmt,
                        ch.dtype_fmt.itemsize,
                        0,
                        0,
                    )
                )

                offset = byte_size

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_CANOPEN:
                if names == v4c.CANOPEN_TIME_FIELDS:
                    record.append(
                        (
                            np.dtype("V6"),
                            6,
                            0,
                            0,
                        )
                    )

                    types.append((name, "V6"))
                    gp.single_channel_dtype = np.dtype("V6")
                    byte_size = 6
                    s_type = v4c.DATA_TYPE_CANOPEN_TIME

                else:
                    record.append(
                        (
                            np.dtype("V7"),
                            7,
                            0,
                            0,
                        )
                    )
                    vals: list[Any] = []
                    for field in ("ms", "min", "hour", "day", "month", "year"):
                        if field == "hour":
                            vals.append(signal.samples[field] + (signal.samples["summer_time"] << 7))
                        elif field == "day":
                            vals.append(signal.samples[field] + (signal.samples["day_of_week"] << 4))
                        else:
                            vals.append(signal.samples[field])
                    samples = np.rec.fromarrays(vals)

                    types.append((name, "V7"))
                    gp.single_channel_dtype = np.dtype("V7")
                    byte_size = 7
                    s_type = v4c.DATA_TYPE_CANOPEN_DATE

                s_size = byte_size * 8

                # there is no channel dependency
                gp_dep.append(None)

                # add channel block
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }
                if signal.invalidation_bits is not None:
                    invalidation_bits = signal.invalidation_bits
                    cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    cn_kwargs["pos_invalidation_bit"] = 0
                else:
                    invalidation_bits = None

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_names = signal.display_names
                ch.dtype_fmt = gp.single_channel_dtype

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                offset = byte_size

                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in ch.display_names:
                    self.channels_db.add(_name, entry)

                gp_sdata.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                (
                    offset,
                    dg_cntr,
                    ch_cntr,
                    struct_self,
                    new_fields,
                    new_types,
                ) = self._append_structure_composition_column_oriented(
                    gp,
                    signal,
                    field_names,
                    offset,
                    dg_cntr,
                    ch_cntr,
                    defined_texts,
                )

                if signal.invalidation_bits is not None:
                    invalidation_bits = signal.invalidation_bits
                else:
                    invalidation_bits = None

                gp.signal_types = np.dtype(new_types)
                offset = gp.signal_types.itemsize

                samples = signal.samples

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                fields: list[NDArray[Any]] = []

                if names is None:
                    raise RuntimeError("'names' is None")

                # here we have channel arrays or mdf v3 channel dependencies
                samples = signal.samples[names[0]]
                shape = samples.shape[1:]

                if len(names) > 1 or len(shape) > 1:
                    # add channel dependency block for composed parent channel
                    dims_nr = len(shape)
                    names_nr = len(names)

                    if names_nr == 0:
                        ca_kwargs: ChannelArrayBlockKwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_FIXED_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    elif len(names) == 1:
                        ca_kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_ARRAY,
                            "flags": 0,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    else:
                        ca_kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    parent_dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([parent_dep])

                else:
                    # add channel dependency block for composed parent channel
                    ca_kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    parent_dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([parent_dep])

                field_name = field_names.get_unique_name(name)

                if not samples.flags["C_CONTIGUOUS"]:
                    samples = np.ascontiguousarray(samples)

                fields.append(samples)
                dtype_pair = field_name, samples.dtype, shape
                types.append(dtype_pair)

                record.append(
                    (
                        samples.dtype,
                        samples.dtype.itemsize,
                        offset,
                        0,
                    )
                )

                # first we add the structure channel
                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape, True)

                # add channel block
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }

                if signal.invalidation_bits is not None:
                    invalidation_bits = signal.invalidation_bits
                    cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    cn_kwargs["pos_invalidation_bit"] = 0
                else:
                    invalidation_bits = None

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_names = signal.display_names
                ch.dtype_fmt = samples.dtype

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                size = s_size // 8
                for dim in shape:
                    size *= dim
                offset += size

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in ch.display_names:
                    self.channels_db.add(_name, entry)

                ch_cntr += 1

                for name in names[1:]:
                    field_name = field_names.get_unique_name(name)

                    samples = signal.samples[name]
                    shape = samples.shape[1:]

                    if not samples.flags["C_CONTIGUOUS"]:
                        samples = np.ascontiguousarray(samples)
                    fields.append(samples)
                    types.append((field_name, samples.dtype, shape))

                    record.append(
                        (
                            samples.dtype,
                            samples.dtype.itemsize,
                            offset,
                            0,
                        )
                    )

                    # add channel dependency block
                    ca_kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([dep])

                    # add components channel
                    s_type, s_size = fmt_to_datatype_v4(samples.dtype, ())
                    byte_size = s_size // 8 or 1
                    cn_kwargs = {
                        "channel_type": v4c.CHANNEL_TYPE_VALUE,
                        "bit_count": s_size,
                        "byte_offset": offset,
                        "bit_offset": 0,
                        "data_type": s_type,
                        "flags": 0,
                    }

                    if signal.invalidation_bits is not None:
                        invalidation_bits = signal.invalidation_bits
                        cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                        cn_kwargs["pos_invalidation_bit"] = 0
                    else:
                        invalidation_bits = None

                    ch = Channel(**cn_kwargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_names = signal.display_names
                    ch.dtype_fmt = samples.dtype

                    gp_channels.append(ch)

                    entry = dg_cntr, ch_cntr
                    parent_dep.axis_channels.append(entry)
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    gp_sdata.append(None)
                    self.channels_db.add(name, entry)

                    ch_cntr += 1

                gp.signal_types = np.dtype(types)

                samples = signal.samples

            else:
                encoding = signal.encoding
                samples = signal.samples
                sig_dtype = samples.dtype

                match encoding:
                    case "utf-8":
                        data_type = v4c.DATA_TYPE_STRING_UTF_8
                    case "latin-1":
                        data_type = v4c.DATA_TYPE_STRING_LATIN_1
                    case "utf-16-be":
                        data_type = v4c.DATA_TYPE_STRING_UTF_16_BE
                    case "utf-16-le":
                        data_type = v4c.DATA_TYPE_STRING_UTF_16_LE
                    case _:
                        raise MdfException(f'wrong encoding "{encoding}" for string signal')

                offsets = arange(len(samples), dtype=uint64) * (signal.samples.itemsize + 4)

                values = [full(len(samples), samples.itemsize, dtype=uint32), samples]

                types_ = [("o", uint32), ("s", sig_dtype)]

                array = np.rec.fromarrays(values, dtype=types_)

                data_size = len(array) * array.itemsize
                if data_size:
                    data_addr = tell()
                    info = SignalDataBlockInfo(
                        address=data_addr,
                        compressed_size=data_size,
                        original_size=data_size,
                        location=v4c.LOCATION_TEMPORARY_FILE,
                    )
                    gp_sdata.append(
                        (
                            [info],
                            iter(EMPTY_TUPLE),
                        )
                    )
                    array.tofile(file)
                else:
                    data_addr = 0
                    gp_sdata.append(
                        (
                            [],
                            iter(EMPTY_TUPLE),
                        )
                    )

                # compute additional byte offset for large records size
                byte_size = 8
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VLSD,
                    "bit_count": 64,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": data_type,
                    "flags": 0,
                }

                if signal.invalidation_bits is not None:
                    invalidation_bits = signal.invalidation_bits
                    cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                    cn_kwargs["pos_invalidation_bit"] = 0
                else:
                    invalidation_bits = None

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_names = signal.display_names

                # conversions for channel
                conversion = conversion_transfer(signal.conversion, version=4)
                if signal.raw:
                    ch.conversion = conversion

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                record.append(
                    (
                        np.dtype(uint64),
                        8,
                        offset,
                        0,
                    )
                )

                offset = byte_size

                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in ch.display_names:
                    self.channels_db.add(_name, entry)

                types.append((name, uint64))
                gp.single_channel_dtype = ch.dtype_fmt = np.dtype(uint64)

                samples = offsets

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            gp.channel_group.samples_byte_nr = offset
            if invalidation_bits is not None:
                gp.channel_group.invalidation_bytes_nr = 1

            virtual_group.groups.append(dg_cntr)
            self.virtual_groups_map[dg_cntr] = cg_master_index

            virtual_group.record_size += offset
            if signal.invalidation_bits:
                virtual_group.record_size += 1

            dg_cntr += 1
            size = cycles_nr * samples.itemsize
            if size:
                data_address = tell()

                data = samples.tobytes()
                raw_size = len(data)
                data = lz_compress(data, store_size=True)

                size = len(data)
                write(data)

                gp.data_blocks.append(
                    DataBlockInfo(
                        address=data_address,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=0,
                    )
                )

                if invalidation_bits is not None:
                    addr = tell()
                    data = invalidation_bits.tobytes()
                    raw_size = len(data)
                    data = lz_compress(data, store_size=True)
                    size = len(data)
                    write(data)

                    gp.data_blocks[-1].invalidation_block = InvalidationBlockInfo(
                        address=addr,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=None,
                    )

            gp.data_location = v4c.LOCATION_TEMPORARY_FILE

        return initial_dg_cntr

    def _append_dataframe(
        self,
        df: DataFrame,
        acq_name: str | None = None,
        acq_source: SourceInformation | None = None,
        comment: str = "",
        units: dict[str, str] | None = None,
    ) -> None:
        """Append a new data group from a pandas DataFrame."""
        units = units or {}

        if df.shape == (0, 0):
            return

        t: NDArray[Any] = df.index.values
        index_name = df.index.name
        time_name = index_name if isinstance(index_name, str) and index_name else "time"
        sync_type = v4c.SYNC_TYPE_TIME
        time_unit = "s"

        dg_cntr = len(self.groups)

        gp = Group(DataGroup())
        gp_sdata = gp.signal_data = []
        gp_channels = gp.channels = []
        gp_dep = gp.channel_dependencies = []
        gp_sig_types = gp.signal_types = []
        record = gp.record = []

        cycles_nr = len(t)

        # channel group
        cg_kwargs: ChannelGroupKwargs = {"cycles_nr": cycles_nr, "samples_byte_nr": 0}
        gp.channel_group = ChannelGroup(**cg_kwargs)
        gp.channel_group.acq_name = acq_name
        gp.channel_group.acq_source = acq_source
        gp.channel_group.comment = comment

        self.groups.append(gp)

        fields: list[NDArray[Any]] = []
        types: list[DTypeLike | tuple[str, np.dtype[Any], tuple[int, ...]]] = []
        ch_cntr = 0
        offset = 0
        field_names = UniqueDB()

        # setup all blocks related to the time master channel

        file = self._tempfile
        tell = file.tell
        seek = file.seek

        seek(0, 2)

        virtual_group = VirtualChannelGroup()
        self.virtual_groups[dg_cntr] = virtual_group
        self.virtual_groups_map[dg_cntr] = dg_cntr
        virtual_group.groups.append(dg_cntr)
        virtual_group.cycles_nr = cycles_nr

        # time channel
        t_type, t_size = fmt_to_datatype_v4(t.dtype, t.shape)
        cn_kwargs: ChannelKwargs = {
            "channel_type": v4c.CHANNEL_TYPE_MASTER,
            "data_type": t_type,
            "sync_type": sync_type,
            "byte_offset": 0,
            "bit_offset": 0,
            "bit_count": t_size,
            "min_raw_value": t[0] if cycles_nr else 0,
            "max_raw_value": t[-1] if cycles_nr else 0,
            "lower_limit": t[0] if cycles_nr else 0,
            "upper_limit": t[-1] if cycles_nr else 0,
            "flags": v4c.FLAG_PHY_RANGE_OK | v4c.FLAG_VAL_RANGE_OK,
        }
        ch = Channel(**cn_kwargs)
        ch.unit = time_unit
        ch.name = time_name
        ch.dtype_fmt = t.dtype
        name = time_name
        gp_channels.append(ch)

        gp_sdata.append(None)
        self.channels_db.add(name, (dg_cntr, ch_cntr))
        self.masters_db[dg_cntr] = 0

        record.append(
            (
                t.dtype,
                t.dtype.itemsize,
                offset,
                0,
            )
        )

        # time channel doesn't have channel dependencies
        gp_dep.append(None)

        fields.append(t)
        types.append((name, t.dtype))
        field_names.get_unique_name(name)

        offset += t_size // 8
        ch_cntr += 1

        gp_sig_types.append(0)

        for signal in df.columns:
            if index_name == signal:
                continue

            sig: Series[Any] = df[signal]
            name = signal

            sig_type = v4c.SIGNAL_TYPE_SCALAR
            if sig.dtype.kind in "SV":
                sig_type = v4c.SIGNAL_TYPE_STRING

            gp_sig_types.append(sig_type)

            # first add the signals in the simple signal list
            if sig_type == v4c.SIGNAL_TYPE_SCALAR:
                # compute additional byte offset for large records size
                if sig.dtype.kind == "O":
                    array = encode(sig.array.astype(str), "utf-8")
                else:
                    array = sig.to_numpy()

                s_type, s_size = fmt_to_datatype_v4(array.dtype, array.shape)

                byte_size = s_size // 8 or 1

                channel_type = v4c.CHANNEL_TYPE_VALUE
                data_block_addr = 0
                sync_type = v4c.SYNC_TYPE_NONE

                cn_kwargs = {
                    "channel_type": channel_type,
                    "sync_type": sync_type,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                }

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = units.get(name, "")
                ch.dtype_fmt = np.dtype((array.dtype, array.shape[1:]))

                record.append(
                    (
                        ch.dtype_fmt,
                        ch.dtype_fmt.itemsize,
                        offset,
                        0,
                    )
                )

                gp_channels.append(ch)

                offset += byte_size

                gp_sdata.append(None)
                self.channels_db.add(name, (dg_cntr, ch_cntr))

                field_name = field_names.get_unique_name(name)

                fields.append(array)
                types.append((field_name, array.dtype, array.shape[1:]))

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_STRING:
                array = sig.to_numpy()

                offsets = arange(len(array), dtype=uint64) * (array.dtype.itemsize + 4)

                values = [full(len(array), array.dtype.itemsize, dtype=uint32), array]

                types_: list[DTypeLike] = [("", uint32), ("", array.dtype)]

                data = np.rec.fromarrays(values, dtype=types_)

                data_size = len(data) * data.itemsize
                if data_size:
                    data_addr = tell()
                    info = SignalDataBlockInfo(
                        address=data_addr,
                        compressed_size=data_size,
                        original_size=data_size,
                        location=v4c.LOCATION_TEMPORARY_FILE,
                    )
                    gp_sdata.append(
                        (
                            [info],
                            iter(EMPTY_TUPLE),
                        )
                    )
                    data.tofile(file)
                else:
                    data_addr = 0
                    gp_sdata.append(
                        (
                            [],
                            iter(EMPTY_TUPLE),
                        )
                    )

                # compute additional byte offset for large records size
                byte_size = 8
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VLSD,
                    "bit_count": 64,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": v4c.DATA_TYPE_STRING_UTF_8,
                    "min_raw_value": 0,
                    "max_raw_value": 0,
                    "lower_limit": 0,
                    "upper_limit": 0,
                    "flags": 0,
                }

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = units.get(name, "")
                ch.dtype_fmt = np.dtype("<u8")

                gp_channels.append(ch)

                record.append(
                    (
                        np.dtype(np.uint64),
                        8,
                        offset,
                        0,
                    )
                )

                offset += byte_size

                self.channels_db.add(name, (dg_cntr, ch_cntr))

                field_name = field_names.get_unique_name(name)

                fields.append(offsets)
                types.append((field_name, uint64))

                ch_cntr += 1

                # simple channels don't have channel dependencies
                gp_dep.append(None)

        virtual_group.record_size = offset
        virtual_group.cycles_nr = cycles_nr

        gp.channel_group.cycles_nr = cycles_nr
        gp.channel_group.samples_byte_nr = offset

        # data block
        gp.sorted = True

        samples: NDArray[Any]
        if df.shape[0]:
            samples = np.rec.fromarrays(fields, dtype=np.dtype(types))
        else:
            samples = np.array([])

        size = len(samples) * samples.itemsize
        if size:
            data_address = self._tempfile.tell()
            gp.data_location = v4c.LOCATION_TEMPORARY_FILE

            samples.tofile(self._tempfile)

            self._tempfile.write(samples.tobytes())

            gp.data_blocks.append(
                DataBlockInfo(
                    address=data_address,
                    block_type=v4c.DT_BLOCK,
                    original_size=size,
                    compressed_size=size,
                    param=0,
                )
            )
        else:
            gp.data_location = v4c.LOCATION_TEMPORARY_FILE

    def _append_structure_composition(
        self,
        grp: Group,
        signal: Signal,
        offset: int,
        dg_cntr: int,
        ch_cntr: int,
        defined_texts: dict[str, int],
        invalidation_bytes_nr: int,
        inval_bits: dict[tuple[int, int], list[InvalidationArray] | InvalidationArray],
    ) -> tuple[int, int, int, tuple[int, int], list[tuple[NDArray[Any], int]]]:
        si_map = self._si_map

        fields: list[tuple[NDArray[Any], int]] = []

        file = self._tempfile
        seek = file.seek
        seek(0, 2)

        gp = grp
        gp_sdata = gp.signal_data
        gp_channels = gp.channels
        gp_dep = gp.channel_dependencies
        record = self._prepare_record(gp)

        name = signal.name
        names = signal.samples.dtype.names

        if names is None:
            raise RuntimeError("'names' is None")

        # first we add the structure channel

        if signal.attachment and signal.attachment[0]:
            at_data, at_name, hash_sum = typing.cast(tuple[bytes, Path, bytes], signal.attachment)
            if at_name is not None:
                suffix = Path(at_name).suffix.lower().strip(".")
            else:
                suffix = "dbc"
            if suffix == "a2l":
                mime = "application/A2L"
            else:
                mime = f"application/x-{suffix}"
            attachment_index = self.attach(at_data, at_name, hash_sum=hash_sum, mime=mime)
            attachment = attachment_index
        else:
            attachment = None

        # add channel block
        cn_kwargs: ChannelKwargs = {
            "channel_type": v4c.CHANNEL_TYPE_VALUE,
            "bit_count": signal.samples.dtype.itemsize * 8,
            "byte_offset": offset,
            "bit_offset": 0,
            "data_type": v4c.DATA_TYPE_BYTEARRAY,
            "precision": 0,
        }

        if attachment is not None:
            cn_kwargs["attachment_addr"] = 0

        def source_bus(source: Source | None) -> TypeIs[Source]:
            return source is not None and source.source_type == v4c.SOURCE_BUS

        if source_bus(signal.source):
            cn_kwargs["flags"] = v4c.FLAG_CN_BUS_EVENT
            flags_ = v4c.FLAG_CN_BUS_EVENT
            grp.channel_group.flags |= v4c.FLAG_CG_BUS_EVENT | v4c.FLAG_CG_PLAIN_BUS_EVENT
        else:
            cn_kwargs["flags"] = 0
            flags_ = 0

        if invalidation_bytes_nr and signal.invalidation_bits is not None:
            if (origin := signal.invalidation_bits.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits[origin])
                invalidation_arrays.append(signal.invalidation_bits)
            else:
                inval_bits[origin] = signal.invalidation_bits
            cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
            cn_kwargs["pos_invalidation_bit"] = origin[1]

        ch = Channel(**cn_kwargs)
        ch.name = name
        ch.unit = signal.unit
        ch.comment = signal.comment
        ch.display_names = signal.display_names
        ch.attachment = attachment
        ch.dtype_fmt = signal.samples.dtype

        record.append((ch.dtype_fmt, ch.dtype_fmt.itemsize, offset, 0))

        if source_bus(signal.source) and grp.channel_group.acq_source is None:
            grp.channel_group.acq_source = SourceInformation.from_common_source(signal.source)

            match signal.source.bus_type:
                case v4c.BUS_TYPE_CAN:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "CAN"
                case v4c.BUS_TYPE_FLEXRAY:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "FLEXRAY"
                case v4c.BUS_TYPE_ETHERNET:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "ETHERNET"
                case v4c.BUS_TYPE_K_LINE:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "K_LINE"
                case v4c.BUS_TYPE_MOST:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "MOST"
                case v4c.BUS_TYPE_LIN:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "LIN"

        # source for channel
        source = signal.source
        if source:
            if source in si_map:
                ch.source = si_map[source]
            else:
                new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                new_source.name = source.name
                new_source.path = source.path
                new_source.comment = source.comment

                si_map[source] = new_source

                ch.source = new_source

        entry = dg_cntr, ch_cntr
        gp_channels.append(ch)
        struct_self = entry

        gp_sdata.append(None)
        self.channels_db.add(name, entry)
        for _name in ch.display_names:
            self.channels_db.add(_name, entry)

        ch_cntr += 1

        dep_list: list[tuple[int, int]] = []
        gp_dep.append(dep_list)

        # then we add the fields
        for name in names:
            samples = signal.samples[name]
            fld_names = samples.dtype.names

            if fld_names is None:
                sig_type = v4c.SIGNAL_TYPE_SCALAR
                if samples.dtype.kind in "SV":
                    sig_type = v4c.SIGNAL_TYPE_STRING
            else:
                if fld_names in (v4c.CANOPEN_TIME_FIELDS, v4c.CANOPEN_DATE_FIELDS):
                    sig_type = v4c.SIGNAL_TYPE_CANOPEN
                elif fld_names[0] != name:
                    sig_type = v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v4c.SIGNAL_TYPE_ARRAY

            if sig_type in (v4c.SIGNAL_TYPE_SCALAR, v4c.SIGNAL_TYPE_STRING):
                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape)
                byte_size = s_size // 8 or 1

                # add channel block
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": flags_,
                }

                if invalidation_bytes_nr:
                    if signal.invalidation_bits is not None:
                        if (origin := signal.invalidation_bits.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                            invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits[origin])
                            invalidation_arrays.append(signal.invalidation_bits)
                        else:
                            inval_bits[origin] = signal.invalidation_bits
                        cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                        cn_kwargs["pos_invalidation_bit"] = origin[1]

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.dtype_fmt = np.dtype((samples.dtype, samples.shape[1:]))

                record.append(
                    (
                        ch.dtype_fmt,
                        ch.dtype_fmt.itemsize,
                        offset,
                        0,
                    )
                )

                entry = (dg_cntr, ch_cntr)
                gp_channels.append(ch)
                dep_list.append(entry)

                offset += byte_size

                if not samples.flags["C_CONTIGUOUS"]:
                    samples = np.ascontiguousarray(samples)

                fields.append((samples, byte_size))

                gp_sdata.append(None)
                self.channels_db.add(name, entry)

                ch_cntr += 1
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                # here we have channel arrays or mdf v3 channel dependencies
                array_samples = samples
                names = samples.dtype.names
                if names is None:
                    raise RuntimeError("'names' is None")
                samples = array_samples[names[0]]
                shape = samples.shape[1:]

                if len(names) > 1:
                    # add channel dependency block for composed parent channel
                    dims_nr = len(shape)
                    names_nr = len(names)

                    if names_nr == 0:
                        ca_kwargs: ChannelArrayBlockKwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_FIXED_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    elif len(names) == 1:
                        ca_kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_ARRAY,
                            "flags": 0,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    else:
                        ca_kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    parent_dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([parent_dep])

                else:
                    # add channel dependency block for composed parent channel
                    ca_kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    parent_dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([parent_dep])

                record.append(
                    (
                        samples.dtype,
                        samples.dtype.itemsize,
                        offset,
                        0,
                    )
                )

                # first we add the structure channel
                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape, True)

                # add channel block
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }

                if invalidation_bytes_nr:
                    if signal.invalidation_bits is not None:
                        if (origin := signal.invalidation_bits.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                            invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits[origin])
                            invalidation_arrays.append(signal.invalidation_bits)
                        else:
                            inval_bits[origin] = signal.invalidation_bits
                        cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                        cn_kwargs["pos_invalidation_bit"] = origin[1]

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_names = signal.display_names
                ch.dtype_fmt = samples.dtype

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                size = s_size // 8
                for dim in shape:
                    size *= dim
                offset += size

                if not samples.flags["C_CONTIGUOUS"]:
                    samples = np.ascontiguousarray(samples)

                fields.append((samples, size))

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in ch.display_names:
                    self.channels_db.add(_name, entry)

                ch_cntr += 1

                for name in names[1:]:
                    samples = array_samples[name]
                    shape = samples.shape[1:]

                    record.append(
                        (
                            samples.dtype,
                            samples.dtype.itemsize,
                            offset,
                            0,
                        )
                    )

                    # add channel dependency block
                    ca_kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([dep])

                    # add components channel
                    s_type, s_size = fmt_to_datatype_v4(samples.dtype, ())
                    byte_size = s_size // 8 or 1
                    cn_kwargs = {
                        "channel_type": v4c.CHANNEL_TYPE_VALUE,
                        "bit_count": s_size,
                        "byte_offset": offset,
                        "bit_offset": 0,
                        "data_type": s_type,
                        "flags": 0,
                    }

                    if invalidation_bytes_nr:
                        if signal.invalidation_bits is not None:
                            if (origin := signal.invalidation_bits.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                                invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits[origin])
                                invalidation_arrays.append(signal.invalidation_bits)
                            else:
                                inval_bits[origin] = signal.invalidation_bits
                            cn_kwargs["flags"] = v4c.FLAG_CN_INVALIDATION_PRESENT
                            cn_kwargs["pos_invalidation_bit"] = origin[1]

                    ch = Channel(**cn_kwargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_names = signal.display_names
                    ch.dtype_fmt = samples.dtype

                    gp_channels.append(ch)

                    entry = dg_cntr, ch_cntr
                    parent_dep.axis_channels.append(entry)
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    if not samples.flags["C_CONTIGUOUS"]:
                        samples = np.ascontiguousarray(samples)

                    fields.append((samples, byte_size))

                    gp_sdata.append(None)
                    self.channels_db.add(name, entry)

                    ch_cntr += 1

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                struct = Signal(
                    samples,
                    samples,
                    name=name,
                    invalidation_bits=signal.invalidation_bits,
                )
                (
                    offset,
                    dg_cntr,
                    ch_cntr,
                    sub_structure,
                    new_fields,
                ) = self._append_structure_composition(
                    grp,
                    struct,
                    offset,
                    dg_cntr,
                    ch_cntr,
                    defined_texts,
                    invalidation_bytes_nr,
                    inval_bits,
                )
                dep_list.append(sub_structure)
                fields.extend(new_fields)

        return offset, dg_cntr, ch_cntr, struct_self, fields

    def _append_structure_composition_column_oriented(
        self,
        grp: Group,
        signal: Signal,
        field_names: UniqueDB,
        offset: int,
        dg_cntr: int,
        ch_cntr: int,
        defined_texts: dict[str, int],
    ) -> tuple[
        int,
        int,
        int,
        tuple[int, int],
        list[NDArray[Any]],
        list[tuple[str, np.dtype[Any], tuple[int, ...]]],
    ]:
        si_map = self._si_map

        fields: list[NDArray[Any]] = []
        types: list[tuple[str, np.dtype[Any], tuple[int, ...]]] = []

        file = self._tempfile
        seek = file.seek
        seek(0, 2)

        gp = grp
        gp_sdata = gp.signal_data
        gp_channels = gp.channels
        gp_dep = gp.channel_dependencies
        record = self._prepare_record(gp)

        name = signal.name
        names = signal.samples.dtype.names

        if names is None:
            raise RuntimeError("'names' is None")

        field_name = field_names.get_unique_name(name)

        # first we add the structure channel

        if signal.attachment and signal.attachment[0]:
            at_data, at_name, hash_sum = typing.cast(tuple[bytes, Path, bytes], signal.attachment)
            if at_name is not None:
                suffix = Path(at_name).suffix.strip(".")
            else:
                suffix = "dbc"
            attachment_index = self.attach(at_data, at_name, hash_sum=hash_sum, mime=f"application/x-{suffix}")
            attachment = attachment_index
        else:
            attachment = None

        # add channel block
        cn_kwargs: ChannelKwargs = {
            "channel_type": v4c.CHANNEL_TYPE_VALUE,
            "bit_count": signal.samples.dtype.itemsize * 8,
            "byte_offset": offset,
            "bit_offset": 0,
            "data_type": v4c.DATA_TYPE_BYTEARRAY,
            "precision": 0,
        }

        if attachment is not None:
            cn_kwargs["attachment_addr"] = 0

        def source_bus(source: Source | None) -> TypeIs[Source]:
            return source is not None and source.source_type == v4c.SOURCE_BUS

        if source_bus(signal.source):
            cn_kwargs["flags"] = v4c.FLAG_CN_BUS_EVENT
            flags_ = v4c.FLAG_CN_BUS_EVENT
            grp.channel_group.flags |= v4c.FLAG_CG_BUS_EVENT
        else:
            cn_kwargs["flags"] = 0
            flags_ = 0

        if signal.invalidation_bits is not None:
            cn_kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
            cn_kwargs["pos_invalidation_bit"] = 0

        ch = Channel(**cn_kwargs)
        ch.name = name
        ch.unit = signal.unit
        ch.comment = signal.comment
        ch.display_names = signal.display_names
        ch.attachment = attachment
        ch.dtype_fmt = signal.samples.dtype

        if source_bus(signal.source):
            grp.channel_group.acq_source = SourceInformation.from_common_source(signal.source)

            match signal.source.bus_type:
                case v4c.BUS_TYPE_CAN:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "CAN"
                case v4c.BUS_TYPE_FLEXRAY:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "FLEXRAY"
                case v4c.BUS_TYPE_ETHERNET:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "ETHERNET"
                case v4c.BUS_TYPE_K_LINE:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "K_LINE"
                case v4c.BUS_TYPE_MOST:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "MOST"
                case v4c.BUS_TYPE_LIN:
                    grp.channel_group.path_separator = 46
                    grp.channel_group.acq_name = "LIN"

        # source for channel
        source = signal.source
        if source:
            if source in si_map:
                ch.source = si_map[source]
            else:
                new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                new_source.name = source.name
                new_source.path = source.path
                new_source.comment = source.comment

                si_map[source] = new_source

                ch.source = new_source

        entry = dg_cntr, ch_cntr
        gp_channels.append(ch)
        struct_self = entry

        gp_sdata.append(None)
        self.channels_db.add(name, entry)
        for _name in ch.display_names:
            self.channels_db.add(_name, entry)

        ch_cntr += 1

        dep_list: list[tuple[int, int]] = []
        gp_dep.append(dep_list)

        record.append((ch.dtype_fmt, ch.dtype_fmt.itemsize, offset, 0))

        # then we add the fields

        for name in names:
            field_name = field_names.get_unique_name(name)

            samples = signal.samples[name]
            fld_names = samples.dtype.names

            if fld_names is None:
                sig_type = v4c.SIGNAL_TYPE_SCALAR
                if samples.dtype.kind in "SV":
                    sig_type = v4c.SIGNAL_TYPE_STRING
            else:
                if fld_names in (v4c.CANOPEN_TIME_FIELDS, v4c.CANOPEN_DATE_FIELDS):
                    sig_type = v4c.SIGNAL_TYPE_CANOPEN
                elif fld_names[0] != name:
                    sig_type = v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION
                else:
                    sig_type = v4c.SIGNAL_TYPE_ARRAY

            if sig_type in (v4c.SIGNAL_TYPE_SCALAR, v4c.SIGNAL_TYPE_STRING):
                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape)
                byte_size = s_size // 8 or 1

                fields.append(samples)
                types.append((field_name, samples.dtype, samples.shape[1:]))

                # add channel block
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": flags_,
                }

                if signal.invalidation_bits is not None:
                    cn_kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                    cn_kwargs["pos_invalidation_bit"] = 0

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.dtype_fmt = np.dtype((samples.dtype, samples.shape[1:]))

                record.append(
                    (
                        ch.dtype_fmt,
                        ch.dtype_fmt.itemsize,
                        offset,
                        0,
                    )
                )

                entry = (dg_cntr, ch_cntr)
                gp_channels.append(ch)
                dep_list.append(entry)

                offset += byte_size

                gp_sdata.append(None)
                self.channels_db.add(name, entry)

                ch_cntr += 1
                gp_dep.append(None)

            elif sig_type == v4c.SIGNAL_TYPE_ARRAY:
                # here we have channel arrays or mdf v3 channel dependencies
                array_samples = samples
                names = samples.dtype.names
                if names is None:
                    raise RuntimeError("'names' is None")
                samples = array_samples[names[0]]
                shape = samples.shape[1:]

                record.append(
                    (
                        samples.dtype,
                        samples.dtype.itemsize,
                        offset,
                        0,
                    )
                )

                if len(names) > 1:
                    # add channel dependency block for composed parent channel
                    dims_nr = len(shape)
                    names_nr = len(names)

                    if names_nr == 0:
                        ca_kwargs: ChannelArrayBlockKwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_FIXED_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    elif len(names) == 1:
                        ca_kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_ARRAY,
                            "flags": 0,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    else:
                        ca_kwargs = {
                            "dims": dims_nr,
                            "ca_type": v4c.CA_TYPE_LOOKUP,
                            "flags": v4c.FLAG_CA_AXIS,
                            "byte_offset_base": samples.dtype.itemsize,
                        }
                        for i in range(dims_nr):
                            ca_kwargs[f"dim_size_{i}"] = shape[i]  # type: ignore[literal-required]

                    parent_dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([parent_dep])

                else:
                    # add channel dependency block for composed parent channel
                    ca_kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    parent_dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([parent_dep])

                field_name = field_names.get_unique_name(name)

                fields.append(samples)
                dtype_pair = field_name, samples.dtype, shape
                types.append(dtype_pair)

                # first we add the structure channel
                s_type, s_size = fmt_to_datatype_v4(samples.dtype, samples.shape, True)

                # add channel block
                cn_kwargs = {
                    "channel_type": v4c.CHANNEL_TYPE_VALUE,
                    "bit_count": s_size,
                    "byte_offset": offset,
                    "bit_offset": 0,
                    "data_type": s_type,
                    "flags": 0,
                }

                if signal.invalidation_bits is not None:
                    cn_kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                    cn_kwargs["pos_invalidation_bit"] = 0

                ch = Channel(**cn_kwargs)
                ch.name = name
                ch.unit = signal.unit
                ch.comment = signal.comment
                ch.display_names = signal.display_names
                ch.dtype_fmt = samples.dtype

                # source for channel
                source = signal.source
                if source:
                    if source in si_map:
                        ch.source = si_map[source]
                    else:
                        new_source = SourceInformation(source_type=source.source_type, bus_type=source.bus_type)
                        new_source.name = source.name
                        new_source.path = source.path
                        new_source.comment = source.comment

                        si_map[source] = new_source

                        ch.source = new_source

                gp_channels.append(ch)

                size = s_size // 8
                for dim in shape:
                    size *= dim
                offset += size

                gp_sdata.append(None)
                entry = (dg_cntr, ch_cntr)
                self.channels_db.add(name, entry)
                for _name in ch.display_names:
                    self.channels_db.add(_name, entry)

                ch_cntr += 1

                for name in names[1:]:
                    field_name = field_names.get_unique_name(name)

                    samples = array_samples[name]
                    shape = samples.shape[1:]
                    fields.append(samples)
                    types.append((field_name, samples.dtype, shape))

                    record.append(
                        (
                            samples.dtype,
                            samples.dtype.itemsize,
                            offset,
                            0,
                        )
                    )

                    # add channel dependency block
                    ca_kwargs = {
                        "dims": 1,
                        "ca_type": v4c.CA_TYPE_SCALE_AXIS,
                        "flags": 0,
                        "byte_offset_base": samples.dtype.itemsize,
                        "dim_size_0": shape[0],
                    }
                    dep = ChannelArrayBlock(**ca_kwargs)
                    gp_dep.append([dep])

                    # add components channel
                    s_type, s_size = fmt_to_datatype_v4(samples.dtype, ())
                    byte_size = s_size // 8 or 1
                    cn_kwargs = {
                        "channel_type": v4c.CHANNEL_TYPE_VALUE,
                        "bit_count": s_size,
                        "byte_offset": offset,
                        "bit_offset": 0,
                        "data_type": s_type,
                        "flags": 0,
                    }

                    if signal.invalidation_bits is not None:
                        cn_kwargs["flags"] |= v4c.FLAG_CN_INVALIDATION_PRESENT
                        cn_kwargs["pos_invalidation_bit"] = 0

                    ch = Channel(**cn_kwargs)
                    ch.name = name
                    ch.unit = signal.unit
                    ch.comment = signal.comment
                    ch.display_names = signal.display_names
                    ch.dtype_fmt = samples.dtype

                    gp_channels.append(ch)

                    entry = dg_cntr, ch_cntr
                    parent_dep.axis_channels.append(entry)
                    for dim in shape:
                        byte_size *= dim
                    offset += byte_size

                    gp_sdata.append(None)
                    self.channels_db.add(name, entry)

                    ch_cntr += 1

            elif sig_type == v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                struct = Signal(
                    samples,
                    samples,
                    name=name,
                    invalidation_bits=signal.invalidation_bits,
                )
                (
                    offset,
                    dg_cntr,
                    ch_cntr,
                    sub_structure,
                    new_fields,
                    new_types,
                ) = self._append_structure_composition_column_oriented(
                    grp,
                    struct,
                    field_names,
                    offset,
                    dg_cntr,
                    ch_cntr,
                    defined_texts,
                )
                dep_list.append(sub_structure)
                fields.extend(new_fields)
                types.extend(new_types)

        return offset, dg_cntr, ch_cntr, struct_self, fields, types

    def extend(self, index: int, signals: Sequence[tuple[NDArray[Any], NDArray[np.bool] | None]]) -> None:
        """Extend a group with new samples.

        `signals` contains (values, invalidation_bits) pairs for each extended
        signal. The first pair is the master channel's pair, and the next pairs
        must respect the same order in which the signals were appended. The
        samples must have raw or physical values according to the signals used
        for the initial append.

        Parameters
        ----------
        index : int
            Group index.
        signals : sequence
            Sequence of (np.ndarray, np.ndarray) tuples.

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
        >>> mdf = MDF(version='4.10')
        >>> mdf.append([s1, s2, s3], comment='created by asammdf')
        >>> t = np.array([0.006, 0.007, 0.008, 0.009, 0.010])

        Extend without invalidation bits.

        >>> mdf.extend(0, [(t, None), (s1.samples, None), (s2.samples, None), (s3.samples, None)])

        Extend with some invalidation bits.

        >>> s1_inv = np.array([0, 0, 0, 1, 1], dtype=bool)
        >>> mdf.extend(0, [(t, None), (s1.samples, s1_inv), (s2.samples, None), (s3.samples, None)])
        """
        if self.version >= "4.20" and (self._column_storage or 1):
            return self._extend_column_oriented(index, signals)
        gp = self.groups[index]
        if not signals:
            message = '"append" requires a non-empty list of Signal objects'
            raise MdfException(message)

        stream = self._tempfile

        fields: list[tuple[bytes | NDArray[Any], int]] = []
        inval_bits_map: dict[tuple[int, int], list[InvalidationArray] | InvalidationArray] = {
            InvalidationArray.ORIGIN_UNKNOWN: []
        }

        added_cycles = len(signals[0][0])

        invalidation_bytes_nr = gp.channel_group.invalidation_bytes_nr
        signal_types = typing.cast(list[tuple[int, int]], gp.signal_types)
        for i, ((signal, invalidation_bits), (sig_type, sig_size)) in enumerate(
            zip(signals, signal_types, strict=False)
        ):
            if invalidation_bytes_nr:
                if invalidation_bits is not None:
                    if not isinstance(invalidation_bits, InvalidationArray):
                        invalidation_array = InvalidationArray(invalidation_bits)
                    else:
                        invalidation_array = invalidation_bits
                    if (origin := invalidation_array.origin) == InvalidationArray.ORIGIN_UNKNOWN:
                        invalidation_arrays = typing.cast(list[InvalidationArray], inval_bits_map[origin])
                        invalidation_arrays.append(invalidation_array)
                    else:
                        inval_bits_map[origin] = invalidation_array

            # first add the signals in the simple signal list
            match sig_type:
                case v4c.SIGNAL_TYPE_SCALAR:
                    if not signal.flags["C_CONTIGUOUS"]:
                        signal = np.ascontiguousarray(signal)

                    fields.append((signal, sig_size))

                case v4c.SIGNAL_TYPE_CANOPEN:
                    names = signal.dtype.names

                    if names == v4c.CANOPEN_TIME_FIELDS:
                        if not signal.flags["C_CONTIGUOUS"]:
                            signal = np.ascontiguousarray(signal)

                        fields.append((signal, sig_size))

                    else:
                        arrays: list[Any] = []
                        for field in ("ms", "min", "hour", "day", "month", "year"):
                            arrays.append(signal[field])
                        vals = np.rec.fromarrays(arrays).tobytes()

                        fields.append((vals, sig_size))

                case v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                    if not signal.flags["C_CONTIGUOUS"]:
                        signal = np.ascontiguousarray(signal)

                    fields.append((signal, sig_size))

                case v4c.SIGNAL_TYPE_ARRAY:
                    names = signal.dtype.names

                    if names is None:
                        raise RuntimeError("'names' is None")

                    samples = signal[names[0]]

                    if not samples.flags["C_CONTIGUOUS"]:
                        samples = np.ascontiguousarray(samples)

                    fields.append((samples, sig_size))

                    for name in names[1:]:
                        samples = signal[name]
                        shape = samples.shape[1:]
                        s_type, s_size = fmt_to_datatype_v4(samples.dtype, ())
                        size = s_size // 8
                        for dim in shape:
                            size *= dim

                        if not samples.flags["C_CONTIGUOUS"]:
                            samples = np.ascontiguousarray(samples)

                        fields.append((samples, size))

                case _:
                    if self.compact_vlsd:
                        cur_offset = sum(blk.original_size for blk in gp.get_signal_data_blocks(i))

                        pairs: list[bytes] = []
                        offsets: list[int] = []
                        off = 0
                        if gp.channels[i].data_type == v4c.DATA_TYPE_STRING_UTF_16_LE:
                            for elem in signal:
                                offsets.append(off)
                                size = len(elem)
                                if size % 2:
                                    size += 1
                                    elem = elem + b"\0"
                                pairs.extend((UINT32_p(size), elem))
                                off += size + 4
                        else:
                            for elem in signal:
                                offsets.append(off)
                                size = len(elem)
                                pairs.extend((UINT32_p(size), elem))
                                off += size + 4

                        offsets_arr: NDArray[np.integer[Any]] = array(offsets, dtype=uint64)

                        stream.seek(0, 2)
                        addr = stream.tell()

                        data_size = off
                        if data_size:
                            info = SignalDataBlockInfo(
                                address=addr,
                                compressed_size=data_size,
                                original_size=data_size,
                                location=v4c.LOCATION_TEMPORARY_FILE,
                            )
                            signal_data = typing.cast(
                                list[tuple[list[SignalDataBlockInfo], Iterator[SignalDataBlockInfo]]], gp.signal_data
                            )
                            signal_data[i][0].append(info)
                            stream.write(b"".join(pairs))

                        offsets_arr += cur_offset  # type: ignore[misc, unused-ignore]
                        if not offsets_arr.flags["C_CONTIGUOUS"]:
                            offsets_arr = np.ascontiguousarray(offsets_arr)
                        fields.append((offsets_arr, 8))

                    else:
                        cur_offset = sum(blk.original_size for blk in gp.get_signal_data_blocks(i))

                        offsets_arr = arange(len(signal), dtype=uint64) * (signal.itemsize + 4)

                        values = [full(len(signal), signal.itemsize, dtype=uint32), signal]

                        types_ = [("", uint32), ("", signal.dtype)]

                        values_arr = np.rec.fromarrays(values, dtype=types_)

                        stream.seek(0, 2)
                        addr = stream.tell()
                        block_size = len(values_arr) * values_arr.itemsize
                        if block_size:
                            info = SignalDataBlockInfo(
                                address=addr,
                                compressed_size=block_size,
                                original_size=block_size,
                                location=v4c.LOCATION_TEMPORARY_FILE,
                            )
                            signal_data = typing.cast(
                                list[tuple[list[SignalDataBlockInfo], Iterator[SignalDataBlockInfo]]], gp.signal_data
                            )
                            signal_data[i][0].append(info)
                            values_arr.tofile(stream)

                        offsets_arr += cur_offset
                        if not offsets_arr.flags["C_CONTIGUOUS"]:
                            offsets_arr = np.ascontiguousarray(offsets_arr)
                        fields.append((offsets_arr, 8))

        if invalidation_bytes_nr:
            unknown_origin = typing.cast(list[InvalidationArray], inval_bits_map.pop(InvalidationArray.ORIGIN_UNKNOWN))
            inval_array_map = typing.cast(dict[tuple[int, int], InvalidationArray], inval_bits_map)

            inval_bits = typing.cast(list[NDArray[np.bool]], list(inval_array_map.values()) + unknown_origin)
            cycles_nr = len(inval_bits[0])
            new_invalidation_bytes_nr = len(inval_bits)

            for _ in range(8 - new_invalidation_bytes_nr % 8):
                inval_bits.append(zeros(cycles_nr, dtype=bool))

            inval_bits.reverse()
            new_invalidation_bytes_nr = len(inval_bits) // 8

            if new_invalidation_bytes_nr != invalidation_bytes_nr:
                raise MdfException(
                    "The invalidation bytes number in the extend methods differs from the one from the append"
                )

            inval_bits_vals = np.fliplr(
                np.packbits(array(inval_bits).T).reshape((cycles_nr, invalidation_bytes_nr))
            ).ravel()

            if self.version < "4.20":
                fields.append((inval_bits_vals, invalidation_bytes_nr))

        samples_bytes = data_block_from_arrays(fields, added_cycles, THREAD_COUNT)
        size = len(samples_bytes)
        samples_view = memoryview(samples_bytes)

        del fields

        stream.seek(0, 2)
        addr = stream.tell()

        record_size = gp.channel_group.samples_byte_nr + gp.channel_group.invalidation_bytes_nr

        if size:
            if self.version < "4.20":
                block_size = 32 * 1024 * 1024 // record_size * record_size

                count = ceil(size / block_size)

                for i in range(count):
                    data_ = samples_view[i * block_size : (i + 1) * block_size]
                    raw_size = len(data_)
                    data_ = lz_compress(data_, store_size=True)

                    size = len(data_)
                    data_address = self._tempfile.tell()
                    self._tempfile.write(data_)

                    gp.data_blocks.append(
                        DataBlockInfo(
                            address=data_address,
                            block_type=v4c.DZ_BLOCK_LZ,
                            original_size=raw_size,
                            compressed_size=size,
                            param=0,
                        )
                    )

                gp.channel_group.cycles_nr += added_cycles
                self.virtual_groups[index].cycles_nr += added_cycles

            else:
                raw_size = size
                data = lz_compress(samples, store_size=True)
                size = len(data)
                stream.write(data)

                gp.data_blocks.append(
                    DataBlockInfo(
                        address=addr,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=0,
                    )
                )

                gp.channel_group.cycles_nr += added_cycles
                self.virtual_groups[index].cycles_nr += added_cycles

                if invalidation_bytes_nr:
                    addr = stream.tell()

                    data = inval_bits_vals.tobytes()
                    raw_size = len(data)
                    data = lz_compress(data, store_size=True)
                    size = len(data)
                    stream.write(data)

                    gp.data_blocks[-1].invalidation_block = InvalidationBlockInfo(
                        address=addr,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=0,
                    )

    def _extend_column_oriented(
        self, index: int, signals: Sequence[tuple[NDArray[Any], NDArray[np.bool] | None]]
    ) -> None:
        """Extend a group with new samples.

        `signals` contains (values, invalidation_bits) pairs for each extended
        signal. The first pair is the master channel's pair, and the next pairs
        must respect the same order in which the signals were appended. The
        samples must have raw or physical values according to the signals used
        for the initial append.

        Parameters
        ----------
        index : int
            Group index.
        signals : sequence
            Sequence of (np.ndarray, np.ndarray) tuples.

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
        >>> mdf = MDF(version='4.10')
        >>> mdf.append([s1, s2, s3], comment='created by asammdf')
        >>> t = np.array([0.006, 0.007, 0.008, 0.009, 0.010])

        Extend without invalidation bits.

        >>> mdf.extend(0, [(t, None), (s1.samples, None), (s2.samples, None), (s3.samples, None)])

        Extend with some invalidation bits.

        >>> s1_inv = np.array([0, 0, 0, 1, 1], dtype=bool)
        >>> mdf.extend(0, [(t, None), (s1.samples, s1_inv), (s2.samples, None), (s3.samples, None)])
        """
        gp = self.groups[index]
        if not signals:
            message = '"append" requires a non-empty list of Signal objects'
            raise MdfException(message)

        stream = self._tempfile
        stream.seek(0, 2)
        write = stream.write
        tell = stream.tell

        added_cycles = len(signals[0][0])

        self.virtual_groups[index].cycles_nr += added_cycles

        for i, (signal, invalidation_bits) in enumerate(signals):
            gp = self.groups[index + i]
            sig_type = typing.cast(int, gp.signal_types[0])

            # first add the signals in the simple signal list
            match sig_type:
                case v4c.SIGNAL_TYPE_SCALAR:
                    samples = signal

                case v4c.SIGNAL_TYPE_CANOPEN:
                    names = signal.dtype.names

                    if names == v4c.CANOPEN_TIME_FIELDS:
                        samples = signal

                    else:
                        vals = []
                        for field in ("ms", "min", "hour", "day", "month", "year"):
                            vals.append(signal[field])
                        samples = np.rec.fromarrays(vals)

                case v4c.SIGNAL_TYPE_STRUCTURE_COMPOSITION:
                    samples = signal

                case v4c.SIGNAL_TYPE_ARRAY:
                    samples = signal

                case _:
                    cur_offset = sum(blk.original_size for blk in gp.get_signal_data_blocks(0))

                    offsets = arange(len(signal), dtype=uint64) * (signal.itemsize + 4)

                    arrays = [full(len(signal), signal.itemsize, dtype=uint32), signal]

                    types_ = [("", uint32), ("", signal.dtype)]

                    values = np.rec.fromarrays(arrays, dtype=types_)

                    addr = tell()
                    block_size = len(values) * values.itemsize
                    if block_size:
                        info = SignalDataBlockInfo(
                            address=addr,
                            compressed_size=block_size,
                            original_size=block_size,
                            location=v4c.LOCATION_TEMPORARY_FILE,
                        )
                        sd_block_infos = typing.cast(
                            tuple[list[SignalDataBlockInfo], Iterator[SignalDataBlockInfo]], gp.signal_data[i]
                        )
                        sd_block_infos[0].append(info)
                        write(values.tobytes())

                    offsets += cur_offset

                    samples = offsets

            addr = tell()

            if added_cycles:
                data = samples.tobytes()
                raw_size = len(data)
                data = lz_compress(data, store_size=True)

                size = len(data)
                write(data)

                gp.data_blocks.append(
                    DataBlockInfo(
                        address=addr,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=0,
                    )
                )

                gp.channel_group.cycles_nr += added_cycles

                if invalidation_bits is not None:
                    addr = tell()
                    data = invalidation_bits.tobytes()
                    raw_size = len(data)
                    data = lz_compress(data, store_size=True)
                    size = len(data)
                    write(data)

                    gp.data_blocks[-1].invalidation_block = InvalidationBlockInfo(
                        address=addr,
                        block_type=v4c.DZ_BLOCK_LZ,
                        original_size=raw_size,
                        compressed_size=size,
                        param=None,
                    )

    def attach(
        self,
        data: bytes,
        file_name: StrPath | None = None,
        hash_sum: bytes | None = None,
        comment: str = "",
        compression: bool = True,
        mime: str = r"application/octet-stream",
        embedded: bool = True,
        password: str | bytes | None = None,
    ) -> int:
        """Attach embedded attachment as application/octet-stream.

        Parameters
        ----------
        data : bytes
            Data to be attached.
        file_name : str | path-like, optional
            File name.
        hash_sum : bytes, optional
            MD5 of the data.
        comment : str, optional
            Attachment comment.
        compression : bool, default True
            Use compression for embedded attachment data.
        mime : str, default 'application/octet-stream'
            Mime type.
        embedded : bool, default True
            Attachment is embedded in the file.
        password : str | bytes, optional
            Password used to encrypt the data using AES256 encryption.

            .. versionadded:: 7.0.0

        Returns
        -------
        index : int
            New attachment index.
        """
        if self._force_attachment_encryption:
            password = password or self._password

        if password and not CRYPTOGRAPHY_AVAILABLE:
            raise MdfException("cryptography must be installed for attachment encryption")

        hash_digest: bytes | str
        if hash_sum is None:
            worker = md5()
            worker.update(data)
            hash_digest = worker.hexdigest()
        else:
            hash_digest = hash_sum

        if hash_digest in self._attachments_cache:
            return self._attachments_cache[hash_digest]

        hash_sum_encrypted: bytes | str
        if password:
            if isinstance(password, str):
                password = password.encode("utf-8")

            size = len(password)
            if size < 32:
                password = password + bytes(32 - size)
            else:
                password = password[:32]

            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(password), modes.CBC(iv))
            encryptor = cipher.encryptor()

            original_size = len(data)

            rem = original_size % 16
            if rem:
                data += os.urandom(16 - rem)

            data = iv + encryptor.update(data) + encryptor.finalize()
            worker = md5()
            worker.update(data)
            hash_sum_encrypted = worker.hexdigest()

            comment = f"""<ATcomment>
    <TX>{comment}</TX>
    <extensions>
		<extension>
			<encrypted>true</encrypted>
			<algorithm>AES256</algorithm>
			<original_md5_sum>{hash_sum_encrypted}</original_md5_sum>
			<original_size>{original_size}</original_size>
		</extension>
	</extensions>
</ATcomment>
"""
        else:
            hash_sum_encrypted = hash_digest

        if hash_sum_encrypted in self._attachments_cache:
            return self._attachments_cache[hash_digest]

        creator_index = len(self.file_history)
        fh = FileHistory()
        fh.comment = """<FHcomment>
    <TX>Added new embedded attachment from {file_name}</TX>
    <tool_id>{tool}</tool_id>
    <tool_vendor>{vendor}</tool_vendor>
    <tool_version>{version}</tool_version>
</FHcomment>""".format(
            file_name=file_name if file_name else "bin.bin",
            version=tool.__version__,
            tool=tool.__tool__,
            vendor=tool.__vendor__,
        )

        self.file_history.append(fh)

        file_name = file_name or "bin.bin"

        at_block = AttachmentBlock(
            data=data,
            compression=compression,
            embedded=embedded,
            file_name=file_name,
            comment=comment,
        )
        at_block.comment = comment
        at_block.creator_index = creator_index

        self.attachments.append(at_block)

        file_path = Path(file_name)

        mime_type, _ = guess_type(Path(file_path))
        if mime_type is None:
            suffix = file_path.suffix.lower().strip(".")
            if suffix == "a2l":
                mime = "application/A2L"
            elif not mime:
                mime = f"application/x-{suffix}"
        else:
            mime = mime_type

        at_block.mime = mime

        index = len(self.attachments) - 1
        self._attachments_cache[hash_digest] = index
        self._attachments_cache[hash_sum_encrypted] = index

        return index

    def close(self) -> None:
        """Call this just before the object is not used anymore to clean up the
        temporary file and close the file object.
        """

        if self._closed:
            return
        else:
            self._closed = True

        self._parent = None
        if self._tempfile is not None:
            self._tempfile.close()
        if not self._from_filelike and self._file is not None:
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
                    Path(self.name).unlink()
                except:
                    pass

        for gp in self.groups:
            gp.clear()
        self.groups.clear()
        del self.header
        del self.identification
        self.file_history.clear()
        self.channels_db.clear()
        self.masters_db.clear()
        self.attachments.clear()
        self._attachments_cache.clear()
        self.events.clear()

        self._ch_map.clear()
        self._master_channel_metadata.clear()
        self._invalidation_cache.clear()
        self._external_dbc_cache.clear()
        self._si_map.clear()
        self._file_si_map.clear()
        self._cc_map.clear()
        self._file_cc_map.clear()
        self._cg_map.clear()
        self._cn_data_map.clear()
        self._dbc_cache.clear()
        self.virtual_groups.clear()

    def _extract_attachment(
        self,
        index: int | None = None,
        password: str | bytes | None = None,
    ) -> tuple[bytes | str, Path, bytes | str]:
        """Extract attachment data by index. If it is an embedded attachment,
        then this method creates the new file according to the attachment file
        name information.

        Parameters
        ----------
        index : int, optional
            Attachment index.
        password : str | bytes, optional
            Password used to encrypt the data using AES256 encryption.

            .. versionadded:: 7.0.0

        Returns
        -------
        data : (bytes, pathlib.Path, bytes)
            Tuple of attachment data and path.
        """
        password = password or self._password
        if index is None:
            return b"", Path(""), md5().digest()

        attachment = self.attachments[index]

        current_path = Path.cwd()
        file_path = Path(attachment.file_name or "embedded")

        data: bytes | str
        md5_sum: bytes | str

        try:
            os.chdir(self.name.resolve().parent)

            flags = attachment.flags

            # for embedded attachments extract data and create new files
            if flags & v4c.FLAG_AT_EMBEDDED:
                data = attachment.extract()
                md5_worker = md5()
                md5_worker.update(data)
                md5_sum = md5_worker.digest()

                encryption_info = extract_encryption_information(attachment.comment)
                if encryption_info.get("encrypted", False):
                    if not password:
                        raise MdfException("the password must be provided for encrypted attachments")

                    if isinstance(password, str):
                        password = password.encode("utf-8")

                    size = len(password)
                    if size < 32:
                        password = password + bytes(32 - size)
                    else:
                        password = password[:32]

                    if encryption_info["algorithm"] == "aes256":
                        md5_worker = md5()
                        md5_worker.update(data)
                        md5_sum = md5_worker.hexdigest().lower()

                        if md5_sum != encryption_info["original_md5_sum"]:
                            raise MdfException(
                                f"MD5 sum mismatch for encrypted attachment: original={encryption_info['original_md5_sum']} and computed={md5_sum}"
                            )

                        iv, data = data[:16], data[16:]
                        cipher = Cipher(algorithms.AES(password), modes.CBC(iv))
                        decryptor = cipher.decryptor()
                        data = decryptor.update(data) + decryptor.finalize()

                        data = data[: encryption_info["original_size"]]

                    else:
                        raise MdfException(
                            f"not implemented attachment encryption algorithm <{encryption_info['algorithm']}>"
                        )

            else:
                # for external attachments read the file and return the content

                if file_path.exists() and file_path.is_file():
                    data = file_path.read_bytes()
                else:
                    if self.original_name is None:
                        raise RuntimeError("'original_name' is None")
                    file_path = Path(self.original_name).parent / file_path.name
                    if file_path.exists() and file_path.is_file():
                        data = file_path.read_bytes()
                    else:
                        raise Exception(f"External attachment file {attachment.file_name} was not found")

                md5_worker = md5()
                md5_worker.update(data)
                md5_sum = md5_worker.digest()

                if attachment.mime.startswith("text"):
                    data = data.decode("utf-8", errors="replace")

                if flags & v4c.FLAG_AT_MD5_VALID and attachment.md5_sum != md5_sum:
                    message = (
                        f'ATBLOCK md5sum="{attachment.md5_sum!r}" '
                        f"and external attachment data ({file_path}) "
                        f'md5sum="{md5_sum!r}"'
                    )
                    logger.warning(message)

        except Exception as err:
            os.chdir(current_path)
            message = f'Exception during attachment "{attachment.file_name}" extraction: {err!r}'
            logger.warning(message)
            data = b""
            md5_sum = md5().digest()
        finally:
            os.chdir(current_path)

        return data, file_path, md5_sum

    @overload
    def get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: RasterType | None = ...,
        samples_only: Literal[False] = ...,
        data: Fragment | None = ...,
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
        data: Fragment | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: Literal[False] = ...,
        record_offset: int = ...,
        record_count: int | None = ...,
        skip_channel_validation: bool = ...,
    ) -> tuple[NDArray[Any], NDArray[np.bool] | None]: ...

    @overload
    def get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: RasterType | None = ...,
        *,
        samples_only: Literal[True],
        data: Fragment | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: Literal[True],
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
        *,
        samples_only: Literal[True],
        data: Fragment | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = ...,
        record_count: int | None = ...,
        skip_channel_validation: bool = ...,
    ) -> tuple[NDArray[Any], NDArray[np.bool] | None]: ...

    @overload
    def get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: RasterType | None = ...,
        samples_only: bool = ...,
        data: Fragment | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = ...,
        record_count: int | None = ...,
        skip_channel_validation: bool = ...,
    ) -> Signal | tuple[NDArray[Any], NDArray[np.bool] | None]: ...

    def get(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
        raster: RasterType | None = None,
        samples_only: bool = False,
        data: Fragment | None = None,
        raw: bool = False,
        ignore_invalidation_bits: bool = False,
        record_offset: int = 0,
        record_count: int | None = None,
        skip_channel_validation: bool = False,
    ) -> Signal | tuple[NDArray[Any], NDArray[np.bool] | None]:
        """Get channel samples. The raw data group samples are not loaded to
        memory so it is advised to use `filter` or `select` instead of
        performing several `get` calls.

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
        data : Fragment, optional
            Prevent redundant data read by providing the raw data group samples.
        raw : bool, default False
            Return channel samples without applying the conversion rule.
        ignore_invalidation_bits : bool, default False
            Option to ignore invalidation bits.
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
        res : (np.ndarray, np.ndarray) | Signal
            Returns `Signal` if `samples_only=False` (default option),
            otherwise returns a (np.ndarray, np.ndarray) tuple of samples and
            invalidation bits. If invalidation bits are not used or if
            `ignore_invalidation_bits` is False, then the second item will be
            None.

            The `Signal` samples are:

            * np.recarray for channels that have composition/channel
              array address or for channels of type CANOPENDATE, CANOPENTIME
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
        >>> mdf = MDF(version='4.10')
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
            if group is None:
                raise RuntimeError("'group' cannot be None if 'skip_channel_validation' is True")
            if index is None:
                raise RuntimeError("'index' cannot be None if 'skip_channel_validation' is True")
            gp_nr, ch_nr = group, index
        else:
            gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]

        # get the channel object
        channel = grp.channels[ch_nr]
        dependency_list = grp.channel_dependencies[ch_nr]

        master_is_required = not samples_only or bool(raster)

        vals: NDArray[Any] | None = None
        all_invalid = False

        if channel.byte_offset + (channel.bit_offset + channel.bit_count) / 8 > grp.channel_group.samples_byte_nr:
            all_invalid = True
            logger.warning(
                "\n\t".join(
                    [
                        f"Channel {channel.name} byte offset too high:",
                        f"byte offset = {channel.byte_offset}",
                        f"bit offset = {channel.bit_offset}",
                        f"bit count = {channel.bit_count}",
                        f"group record size = {grp.channel_group.samples_byte_nr}",
                        f"group index = {gp_nr}",
                        f"channel index = {ch_nr}",
                    ]
                )
            )

            if (channel.bit_offset + channel.bit_count) / 8 > grp.channel_group.samples_byte_nr:
                vals = np.array([])
            else:
                channel = deepcopy(channel)
                channel.byte_offset = 0

        if vals is None:
            if dependency_list:
                if not isinstance(dependency_list[0], ChannelArrayBlock):
                    dependency_list = typing.cast(list[tuple[int, int]], dependency_list)
                    samples, timestamps, invalidation_bits, encoding = self._get_structure(
                        channel=channel,
                        group=grp,
                        group_index=gp_nr,
                        channel_index=ch_nr,
                        dependency_list=dependency_list,
                        raster=raster,
                        data=data,
                        ignore_invalidation_bits=ignore_invalidation_bits,
                        record_offset=record_offset,
                        record_count=record_count,
                        master_is_required=master_is_required,
                        raw=raw,
                    )
                else:
                    dependency_list = typing.cast(list[ChannelArrayBlock], dependency_list)
                    samples, timestamps, invalidation_bits, encoding = self._get_array(
                        channel=channel,
                        group=grp,
                        group_index=gp_nr,
                        channel_index=ch_nr,
                        dependency_list=dependency_list,
                        raster=raster,
                        data=data,
                        ignore_invalidation_bits=ignore_invalidation_bits,
                        record_offset=record_offset,
                        record_count=record_count,
                        master_is_required=master_is_required,
                    )

            else:
                if (
                    data
                    and (fast_path := channel.fast_path) is not None
                    and not master_is_required
                    and ignore_invalidation_bits
                    and not raster
                ):
                    samples, timestamps, invalidation_bits, encoding = self._fast_scalar_path(*fast_path, data)
                else:
                    samples, timestamps, invalidation_bits, encoding = self._get_scalar(
                        channel=channel,
                        group=grp,
                        group_index=gp_nr,
                        channel_index=ch_nr,
                        dependency_list=dependency_list,
                        raster=raster,
                        data=data,
                        ignore_invalidation_bits=ignore_invalidation_bits,
                        record_offset=record_offset,
                        record_count=record_count,
                        master_is_required=master_is_required,
                    )
        else:
            samples = vals
            timestamps = np.array([], dtype=np.float64)
            invalidation_bits = None
            encoding = None

        if all_invalid:
            invalidation_bits = np.ones(len(samples), dtype=bool)

        res: tuple[NDArray[Any], NDArray[np.bool] | None] | Signal
        if samples_only:
            if not raw:
                conversion = channel.conversion
                if conversion:
                    samples = conversion.convert(samples)

            res = samples, invalidation_bits
        else:
            if timestamps is None:
                raise RuntimeError("'timestamps' cannot be None if 'samples_only' is False")

            conversion = channel.conversion
            if not raw:
                if conversion:
                    samples = conversion.convert(samples)
                    conversion = None

                if samples.dtype.kind == "S":
                    encoding = "utf-8"

            channel_type = channel.channel_type

            if name is None:
                name = channel.name

            unit = (conversion and conversion.unit) or channel.unit

            comment = channel.comment

            source_information = channel.source

            if source_information:
                source = Source.from_source(source_information)
            else:
                cg_source = grp.channel_group.acq_source
                if cg_source:
                    source = Source.from_source(cg_source)
                else:
                    source = None

            if channel.attachment is not None:
                attachment = self.extract_attachment(
                    channel.attachment,
                )
            else:
                attachment = None

            master_metadata = self._master_channel_metadata.get(gp_nr, None)

            if channel_type == v4c.CHANNEL_TYPE_SYNC:
                flags = Signal.Flags.stream_sync
            else:
                flags = Signal.Flags.no_flags

            try:
                res = Signal(
                    samples=samples,
                    timestamps=timestamps,
                    unit=unit,
                    name=name,
                    comment=comment,
                    conversion=conversion,
                    raw=raw,
                    master_metadata=master_metadata,
                    attachment=attachment,
                    source=source,
                    display_names=channel.display_names,
                    bit_count=channel.bit_count,
                    flags=flags,
                    invalidation_bits=invalidation_bits,
                    encoding=encoding,
                    group_index=gp_nr,
                    channel_index=ch_nr,
                )
            except:
                debug_channel(self, grp, channel, dependency_list)
                raise

        if data is None:
            self._invalidation_cache.clear()

        return res

    @overload
    def _get_structure(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[tuple[int, int]],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[False],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[False],
        raw: bool,
    ) -> tuple[NDArray[Any], None, NDArray[np.bool] | None, None]: ...

    @overload
    def _get_structure(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[tuple[int, int]],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[True],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[False],
        raw: bool,
    ) -> tuple[NDArray[Any], None, None, None]: ...

    @overload
    def _get_structure(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[tuple[int, int]],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[False],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[True],
        raw: bool,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.bool] | None, None]: ...

    @overload
    def _get_structure(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[tuple[int, int]],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[True],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[True],
        raw: bool,
    ) -> tuple[NDArray[Any], NDArray[Any], None, None]: ...

    @overload
    def _get_structure(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[tuple[int, int]],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: bool,
        record_offset: int,
        record_count: int | None,
        master_is_required: bool,
        raw: bool,
    ) -> tuple[NDArray[Any], NDArray[Any] | None, NDArray[np.bool] | None, None]: ...

    def _get_structure(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[tuple[int, int]],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: bool,
        record_offset: int,
        record_count: int | None,
        master_is_required: bool,
        raw: bool,
    ) -> tuple[NDArray[Any], NDArray[Any] | None, NDArray[np.bool] | None, None]:
        grp = group
        gp_nr = group_index
        # get data group record
        self._prepare_record(grp)

        # get group data
        if data is None:
            fragments = self._load_data(grp, record_offset=record_offset, record_count=record_count)
        else:
            fragments = iter((data,))

        groups = self.groups

        channel_invalidation_present = channel.flags & (v4c.FLAG_CN_ALL_INVALID | v4c.FLAG_CN_INVALIDATION_PRESENT)

        _dtype = np.dtype(channel.dtype_fmt)
        conditions = [
            _dtype.itemsize == channel.bit_count // 8,
            all(
                groups[dg_nr].channels[ch_nr].channel_type != v4c.CHANNEL_TYPE_VLSD
                and (
                    (conversion := groups[dg_nr].channels[ch_nr].conversion) is None
                    or not conversion.conversion_type == v4c.CONVERSION_TYPE_NON
                )
                for (dg_nr, ch_nr) in dependency_list
            ),
        ]
        channel_values: list[NDArray[Any]] | list[list[NDArray[Any]]]
        if all(conditions):
            fast_path = True
            flat_channel_values: list[NDArray[Any]] = []
            masters: list[NDArray[Any]] = []
            invalidation_arrays: list[InvalidationArray] = []

            byte_offset = channel.byte_offset
            record_size = grp.channel_group.samples_byte_nr + grp.channel_group.invalidation_bytes_nr

            count = 0
            for fragment in fragments:
                bts = fragment.data

                buffer = get_channel_raw_bytes(bts, record_size, byte_offset, _dtype.itemsize)

                flat_channel_values.append(frombuffer(buffer, dtype=_dtype))

                if master_is_required:
                    masters.append(self.get_master(gp_nr, fragment, one_piece=True))
                if channel_invalidation_present:
                    invalidation_arrays.append(
                        self.get_invalidation_bits(gp_nr, channel.pos_invalidation_bit, fragment)
                    )

                count += 1

            channel_values = flat_channel_values
        else:
            unique_names = UniqueDB()
            fast_path = False
            names = [unique_names.get_unique_name(grp.channels[ch_nr].name) for _, ch_nr in dependency_list]

            channel_values_list: list[list[NDArray[Any]]] = [[] for _ in dependency_list]
            masters = []
            invalidation_arrays = []

            count = 0
            for fragment in fragments:
                for i, (dg_nr, ch_nr) in enumerate(dependency_list):
                    vals = self.get(
                        group=dg_nr,
                        index=ch_nr,
                        samples_only=True,
                        data=fragment,
                        ignore_invalidation_bits=ignore_invalidation_bits,
                        record_offset=record_offset,
                        record_count=record_count,
                        raw=False,
                    )[0]
                    channel_values_list[i].append(vals)
                if master_is_required:
                    masters.append(self.get_master(gp_nr, fragment, one_piece=True))
                if channel_invalidation_present:
                    invalidation_arrays.append(
                        self.get_invalidation_bits(gp_nr, channel.pos_invalidation_bit, fragment)
                    )

                count += 1

            channel_values = channel_values_list

        if fast_path:
            flat_channel_values = typing.cast(list[NDArray[Any]], channel_values)
            total_size = sum(len(_) for _ in flat_channel_values)
            shape = (total_size,) + flat_channel_values[0].shape[1:]

            if count > 1:
                out = empty(shape, dtype=flat_channel_values[0].dtype)
                vals = concatenate(flat_channel_values, out=out)
            else:
                vals = flat_channel_values[0]
        else:
            channel_values_list = typing.cast(list[list[NDArray[Any]]], channel_values)
            total_size = sum(len(_) for _ in channel_values_list[0])

            if count > 1:
                arrays: list[NDArray[Any]] = []
                for lst in channel_values_list:
                    shape_len = len(lst[0].shape)

                    # fix bytearray signals if the length changes between the chunks
                    if shape_len == 2:
                        shape = (total_size,)
                        vlsd_max_size = max(l.shape[1] for l in lst)
                        shape = (*shape, vlsd_max_size)

                        max_vlsd_arrs = []
                        for arr in lst:
                            if arr.shape[1] < vlsd_max_size:
                                arr = np.hstack(
                                    (
                                        arr,
                                        np.zeros(
                                            (
                                                arr.shape[0],
                                                vlsd_max_size - arr.shape[1],
                                            ),
                                            dtype=arr.dtype,
                                        ),
                                    )
                                )
                            max_vlsd_arrs.append(arr)

                        arr = concatenate(
                            max_vlsd_arrs,
                            out=empty(shape, dtype=max_vlsd_arrs[0].dtype),
                        )
                        arrays.append(arr)
                    elif shape_len == 1:
                        arr = concatenate(
                            lst,
                            out=empty((total_size,), dtype=lst[0].dtype),
                        )
                        arrays.append(arr)

                    else:
                        arrays.append(
                            concatenate(
                                lst,
                                out=empty((total_size,) + lst[0].shape[1:], dtype=lst[0].dtype),
                            )
                        )
            else:
                arrays = [lst[0] for lst in channel_values_list]
            types: DTypeLike = [(name_, arr.dtype, arr.shape[1:]) for name_, arr in zip(names, arrays, strict=False)]
            types = np.dtype(types)

            vals = np.rec.fromarrays(arrays, dtype=types)

        if master_is_required:
            if count > 1:
                out = empty(total_size, dtype=masters[0].dtype)
                timestamps = concatenate(masters, out=out)
            else:
                timestamps = masters[0]
        else:
            timestamps = None

        if channel_invalidation_present:
            if count > 1:
                out = empty(total_size, dtype=invalidation_arrays[0].dtype)
                invalidation_array = concatenate(invalidation_arrays, out=out)
            else:
                invalidation_array = invalidation_arrays[0]
            if not ignore_invalidation_bits:
                vals = vals[nonzero(~invalidation_array)[0]]
                if master_is_required:
                    if timestamps is None:
                        raise RuntimeError("'timestamps' cannot be None if 'master_is_required' is True")
                    timestamps = timestamps[nonzero(~invalidation_array)[0]]
                invalidation_bits = None
            else:
                invalidation_bits = invalidation_array
        else:
            invalidation_bits = None

        if raster:
            if timestamps is None:
                raise RuntimeError("'timestamps' cannot be None if 'raster' is not None")
            if len(timestamps) > 1:
                t = arange(timestamps[0], timestamps[-1], raster, dtype=np.float64)

                signal = Signal(vals, timestamps, name="_", invalidation_bits=invalidation_bits).interp(
                    t,
                    integer_interpolation_mode=self._integer_interpolation,
                    float_interpolation_mode=self._float_interpolation,
                )

                vals, timestamps, invalidation_bits = (
                    signal.samples,
                    signal.timestamps,
                    signal.invalidation_bits,
                )

        return vals, timestamps, invalidation_bits, None

    @overload
    def _get_array(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[False],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[False],
    ) -> tuple[NDArray[Any], None, NDArray[np.bool] | None, None]: ...

    @overload
    def _get_array(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[True],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[False],
    ) -> tuple[NDArray[Any], None, None, None]: ...

    @overload
    def _get_array(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[False],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[True],
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.bool] | None, None]: ...

    @overload
    def _get_array(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[True],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[True],
    ) -> tuple[NDArray[Any], NDArray[Any], None, None]: ...

    @overload
    def _get_array(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: bool,
        record_offset: int,
        record_count: int | None,
        master_is_required: bool,
    ) -> tuple[NDArray[Any], NDArray[Any] | None, NDArray[np.bool] | None, None]: ...

    def _get_array(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock],
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: bool,
        record_offset: int,
        record_count: int | None,
        master_is_required: bool,
    ) -> tuple[NDArray[Any], NDArray[Any] | None, NDArray[np.bool] | None, None]:
        grp = group
        gp_nr = group_index
        ch_nr = channel_index
        # get data group record
        self._prepare_record(grp)

        # get group data
        if data is None:
            fragments = self._load_data(grp, record_offset=record_offset, record_count=record_count)
        else:
            fragments = iter((data,))

        dep = ca_block = dependency_list[0]
        shape = tuple(typing.cast(int, ca_block[f"dim_size_{i}"]) for i in range(ca_block.dims))
        shape = tuple(dim for dim in shape if dim > 1)
        shape = shape or (1,)

        dim = 1
        for d in shape:
            dim *= d
        size = ca_block.byte_offset_base * dim

        if group.uses_ld:
            record_size = group.channel_group.samples_byte_nr
        else:
            record_size = group.channel_group.samples_byte_nr + group.channel_group.invalidation_bytes_nr

        channel_dtype = get_fmt_v4(
            channel.data_type,
            channel.bit_count,
            channel.channel_type,
        )

        byte_offset = channel.byte_offset

        dtype_fmt: np.dtype[Any] = np.dtype(
            [
                ("", f"S{byte_offset}"),
                ("vals", channel_dtype, shape),
                ("", f"S{record_size - size - byte_offset}"),
            ]
        )

        channel_invalidation_present = channel.flags & (v4c.FLAG_CN_ALL_INVALID | v4c.FLAG_CN_INVALIDATION_PRESENT)

        channel_group = grp.channel_group
        samples_size = channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr

        channel_values: list[NDArray[Any]] = []
        masters: list[NDArray[Any]] = []
        invalidation_arrays: list[InvalidationArray] = []
        count = 0

        timestamps: NDArray[Any] | None
        for fragment in fragments:
            arrays: list[NDArray[Any]] = []
            types: list[tuple[str, np.dtype[Any], tuple[int, ...]]] = []

            data_bytes, offset, _count, invalidation_bytes = (
                fragment.data,
                fragment.record_offset,
                fragment.record_count,
                fragment.invalidation_data,
            )

            cycles = len(data_bytes) // samples_size

            vals = frombuffer(data_bytes, dtype=dtype_fmt)["vals"]

            if dep.flags & v4c.FLAG_CA_INVERSE_LAYOUT:
                shape = vals.shape
                shape = (shape[0],) + shape[1:][::-1]
                vals = vals.reshape(shape)

                axes = (0, *reversed(range(1, len(shape))))
                vals = transpose(vals, axes=axes)

            cycles_nr = len(vals)

            dtype_pair: tuple[str, np.dtype[Any], tuple[int, ...]]
            for ca_block in dependency_list[:1]:
                if not isinstance(ca_block, ChannelArrayBlock):
                    break

                dims_nr = ca_block.dims

                if ca_block.ca_type == v4c.CA_TYPE_SCALE_AXIS:
                    shape = (ca_block.dim_size_0,)
                    arrays.append(vals)
                    dtype_pair = channel.name, vals.dtype, shape
                    types.append(dtype_pair)

                elif ca_block.ca_type == v4c.CA_TYPE_LOOKUP:
                    shape = vals.shape[1:]
                    arrays.append(vals)
                    dtype_pair = channel.name, vals.dtype, shape
                    types.append(dtype_pair)

                    if ca_block.flags & v4c.FLAG_CA_FIXED_AXIS:
                        for i in range(dims_nr):
                            shape = (typing.cast(int, ca_block[f"dim_size_{i}"]),)
                            axis_list: list[float] = []
                            for j in range(shape[0]):
                                key = f"axis_{i}_value_{j}"
                                axis_list.append(typing.cast(float, ca_block[key]))
                            axis = array(axis_list)
                            axis = array([axis for _ in range(cycles_nr)])
                            arrays.append(axis)
                            dtype_pair = (f"axis_{i}", axis.dtype, shape)
                            types.append(dtype_pair)
                    else:
                        for i in range(dims_nr):
                            axis_channel = ca_block.axis_channels[i]
                            shape = (typing.cast(int, ca_block[f"dim_size_{i}"]),)

                            if axis_channel is None:
                                axisname = f"axis_{i}"
                                axis_values = array(
                                    [arange(shape[0])] * cycles,
                                    dtype=f"({shape[0]},)f8",
                                )

                            else:
                                try:
                                    (ref_dg_nr, ref_ch_nr) = axis_channel
                                except:
                                    debug_channel(self, grp, channel, dependency_list)
                                    raise

                                axisname = self.groups[ref_dg_nr].channels[ref_ch_nr].name

                                if ref_dg_nr == gp_nr:
                                    axis_values = self.get(
                                        group=ref_dg_nr,
                                        index=ref_ch_nr,
                                        samples_only=True,
                                        data=fragment,
                                        ignore_invalidation_bits=ignore_invalidation_bits,
                                        record_offset=record_offset,
                                        record_count=cycles,
                                        raw=True,
                                    )[0]
                                else:
                                    channel_group = grp.channel_group
                                    record_size = channel_group.samples_byte_nr
                                    record_size += channel_group.invalidation_bytes_nr
                                    start = offset // record_size
                                    end = start + len(data_bytes) // record_size + 1
                                    ref = self.get(
                                        group=ref_dg_nr,
                                        index=ref_ch_nr,
                                        samples_only=True,
                                        ignore_invalidation_bits=ignore_invalidation_bits,
                                        record_offset=record_offset,
                                        record_count=cycles,
                                        raw=True,
                                    )[0]
                                    axis_values = ref[start:end].copy()

                                axis_values = axis_values[axisname]
                                if len(axis_values) == 0 and cycles:
                                    axis_values = array([arange(shape[0])] * cycles)

                            arrays.append(axis_values)
                            dtype_pair = (axisname, axis_values.dtype, shape)
                            types.append(dtype_pair)

                elif ca_block.ca_type == v4c.CA_TYPE_ARRAY:
                    shape = vals.shape[1:]
                    arrays.append(vals)
                    dtype_pair = channel.name, vals.dtype, shape
                    types.append(dtype_pair)

            for ca_block in dependency_list[1:]:
                if not isinstance(ca_block, ChannelArrayBlock):
                    break

                dims_nr = ca_block.dims

                if ca_block.flags & v4c.FLAG_CA_FIXED_AXIS:
                    for i in range(dims_nr):
                        shape = (typing.cast(int, ca_block[f"dim_size_{i}"]),)
                        axis_list = []
                        for j in range(shape[0]):
                            key = f"axis_{i}_value_{j}"
                            axis_list.append(typing.cast(float, ca_block[key]))

                        axis = array([axis_list for _ in range(cycles_nr)], dtype=f"{shape}f8")
                        arrays.append(axis)
                        types.append((f"axis_{i}", axis.dtype, shape))
                else:
                    for i in range(dims_nr):
                        axis_channel = ca_block.axis_channels[i]
                        shape = (typing.cast(int, ca_block[f"dim_size_{i}"]),)

                        if axis_channel is None:
                            axisname = f"axis_{i}"
                            axis_values = array([arange(shape[0])] * cycles, dtype=f"({shape[0]},)f8")

                        else:
                            try:
                                ref_dg_nr, ref_ch_nr = axis_channel
                            except:
                                debug_channel(self, grp, channel, dependency_list)
                                raise

                            axisname = self.groups[ref_dg_nr].channels[ref_ch_nr].name

                            if ref_dg_nr == gp_nr:
                                axis_values = self.get(
                                    group=ref_dg_nr,
                                    index=ref_ch_nr,
                                    samples_only=True,
                                    data=fragment,
                                    ignore_invalidation_bits=ignore_invalidation_bits,
                                    record_offset=record_offset,
                                    record_count=cycles,
                                    raw=False,
                                )[0]
                            else:
                                channel_group = grp.channel_group
                                record_size = channel_group.samples_byte_nr
                                record_size += channel_group.invalidation_bytes_nr
                                start = offset // record_size
                                end = start + len(data_bytes) // record_size + 1
                                ref = self.get(
                                    group=ref_dg_nr,
                                    index=ref_ch_nr,
                                    samples_only=True,
                                    ignore_invalidation_bits=ignore_invalidation_bits,
                                    record_offset=record_offset,
                                    record_count=cycles,
                                    raw=False,
                                )[0]
                                axis_values = ref[start:end].copy()
                            axis_values = axis_values[axisname]
                            if len(axis_values) == 0 and cycles:
                                axis_values = array([arange(shape[0])] * cycles)

                        arrays.append(axis_values)
                        dtype_pair = (axisname, axis_values.dtype, shape)
                        types.append(dtype_pair)

            vals = np.rec.fromarrays(arrays, np.dtype(types))

            if master_is_required:
                masters.append(self.get_master(gp_nr, fragment, one_piece=True))
            if channel_invalidation_present:
                invalidation_arrays.append(self.get_invalidation_bits(gp_nr, channel.pos_invalidation_bit, fragment))

            channel_values.append(vals)
            count += 1

        if count > 1:
            total_size = sum(len(_) for _ in channel_values)
            shape = (total_size,) + channel_values[0].shape[1:]

        if count > 1:
            out = empty(shape, dtype=channel_values[0].dtype)
            vals = concatenate(channel_values, out=out)
        elif count == 1:
            vals = channel_values[0]
        else:
            vals = np.array([])

        if master_is_required:
            if count > 1:
                out = empty(total_size, dtype=masters[0].dtype)
                timestamps = concatenate(masters, out=out)
            else:
                timestamps = masters[0]
        else:
            timestamps = None

        if channel_invalidation_present:
            if count > 1:
                out = empty(total_size, dtype=invalidation_arrays[0].dtype)
                invalidation_array = concatenate(invalidation_arrays, out=out)
            else:
                invalidation_array = invalidation_arrays[0]
            if not ignore_invalidation_bits:
                vals = vals[nonzero(~invalidation_array)[0]]
                if master_is_required:
                    if timestamps is None:
                        raise RuntimeError("'timestamps' cannot be None is 'master_is_required' is True")
                    timestamps = timestamps[nonzero(~invalidation_array)[0]]
                invalidation_bits = None
            else:
                invalidation_bits = invalidation_array
        else:
            invalidation_bits = None

        if raster:
            if timestamps is None:
                raise RuntimeError("'timestamps' cannot be None is 'raster' is not None")
            if len(timestamps) > 1:
                t = arange(timestamps[0], timestamps[-1], raster, dtype=np.float64)

                signal = Signal(vals, timestamps, name="_", invalidation_bits=invalidation_bits).interp(
                    t,
                    integer_interpolation_mode=self._integer_interpolation,
                    float_interpolation_mode=self._float_interpolation,
                )

                vals, timestamps, invalidation_bits = (
                    signal.samples,
                    signal.timestamps,
                    signal.invalidation_bits,
                )

        return vals, timestamps, invalidation_bits, None

    def _fast_scalar_path(
        self,
        gp_nr: int,
        record_size: int,
        byte_offset: int,
        byte_size: int,
        pos_invalidation_bit: int,
        # data_type,
        # channel_type,
        # bit_count,
        dtype: np.dtype[Any],
        fragment: Fragment,
    ) -> tuple[NDArray[Any], None, InvalidationArray | None, None]:
        buffer: Buffer
        if fragment.is_record:
            buffer = get_channel_raw_bytes(
                fragment.data,
                record_size,
                byte_offset,
                byte_size,
            )
        else:
            buffer = fragment.data

        vals = frombuffer(buffer, dtype=dtype)

        if pos_invalidation_bit >= 0:
            invalidation_bits = self.get_invalidation_bits(gp_nr, pos_invalidation_bit, fragment)
        else:
            invalidation_bits = None

        return vals, None, invalidation_bits, None

    @overload
    def _get_scalar(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock] | list[tuple[int, int]] | None,
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[False],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[False],
        skip_vlsd: bool = ...,
    ) -> tuple[NDArray[Any], None, NDArray[np.bool] | None, str | None]: ...

    @overload
    def _get_scalar(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock] | list[tuple[int, int]] | None,
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[True],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[False],
        skip_vlsd: bool = ...,
    ) -> tuple[NDArray[Any], None, None, str | None]: ...

    @overload
    def _get_scalar(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock] | list[tuple[int, int]] | None,
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[False],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[True],
        skip_vlsd: bool = ...,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.bool] | None, str | None]: ...

    @overload
    def _get_scalar(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock] | list[tuple[int, int]] | None,
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: Literal[True],
        record_offset: int,
        record_count: int | None,
        master_is_required: Literal[True],
        skip_vlsd: bool = ...,
    ) -> tuple[NDArray[Any], NDArray[Any], None, str | None]: ...

    @overload
    def _get_scalar(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock] | list[tuple[int, int]] | None,
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: bool,
        record_offset: int,
        record_count: int | None,
        master_is_required: bool,
        skip_vlsd: bool = ...,
    ) -> tuple[NDArray[Any], NDArray[Any] | None, NDArray[np.bool] | None, str | None]: ...

    def _get_scalar(
        self,
        channel: Channel,
        group: Group,
        group_index: int,
        channel_index: int,
        dependency_list: list[ChannelArrayBlock] | list[tuple[int, int]] | None,
        raster: RasterType | None,
        data: Fragment | None,
        ignore_invalidation_bits: bool,
        record_offset: int,
        record_count: int | None,
        master_is_required: bool,
        skip_vlsd: bool = False,
    ) -> tuple[NDArray[Any], NDArray[Any] | None, NDArray[np.bool] | None, str | None]:
        grp = group
        # get group data
        if data is None:
            fragments = self._load_data(grp, record_offset=record_offset, record_count=record_count)
            one_piece = False
        else:
            fragments = iter((data,))
            one_piece = True

        gp_nr = group_index
        ch_nr = channel_index

        channel_invalidation_present = channel.flags & (v4c.FLAG_CN_ALL_INVALID | v4c.FLAG_CN_INVALIDATION_PRESENT)

        data_type = channel.data_type
        channel_type = channel.channel_type
        bit_count = channel.bit_count

        encoding: str | None = None

        channel_dtype = channel.dtype_fmt

        vals: NDArray[Any]
        timestamps: NDArray[Any] | None = None
        invalidation_bits: NDArray[np.bool] | None = None

        # get channel values
        if channel_type in {
            v4c.CHANNEL_TYPE_VIRTUAL,
            v4c.CHANNEL_TYPE_VIRTUAL_MASTER,
        }:
            if channel.dtype_fmt == np.dtype(np.void):
                channel.dtype_fmt = np.dtype(get_fmt_v4(data_type, 64))
            ch_dtype = channel.dtype_fmt

            channel_values: list[NDArray[Any]] = []
            masters: list[NDArray[Any]] = []
            invalidation_arrays: list[InvalidationArray] = []

            channel_group = grp.channel_group
            record_size = channel_group.samples_byte_nr
            record_size += channel_group.invalidation_bytes_nr

            count = 0

            for fragment in fragments:
                data_bytes, offset, _count, invalidation_bytes = (
                    fragment.data,
                    fragment.record_offset,
                    fragment.record_count,
                    fragment.invalidation_data,
                )
                offset = offset // record_size

                vals = arange(len(data_bytes) // record_size, dtype=ch_dtype)
                vals += offset

                if master_is_required:
                    masters.append(
                        self.get_master(
                            gp_nr,
                            fragment,
                            record_offset=offset,
                            record_count=_count,
                            one_piece=True,
                        )
                    )
                if channel_invalidation_present:
                    invalidation_arrays.append(
                        self.get_invalidation_bits(gp_nr, channel.pos_invalidation_bit, fragment)
                    )

                channel_values.append(vals)
                count += 1

            if count > 1:
                total_size = sum(len(_) for _ in channel_values)
                shape = (total_size,) + channel_values[0].shape[1:]

            if count > 1:
                out = empty(shape, dtype=channel_values[0].dtype)
                vals = concatenate(channel_values, out=out)
            elif count == 1:
                vals = channel_values[0]
            else:
                vals = np.array([])

            if master_is_required:
                if count > 1:
                    out = empty(total_size, dtype=masters[0].dtype)
                    timestamps = concatenate(masters, out=out)
                else:
                    timestamps = masters[0]
            else:
                timestamps = None

            if channel_invalidation_present:
                if count > 1:
                    out = empty(total_size, dtype=invalidation_arrays[0].dtype)
                    invalidation_array = concatenate(invalidation_arrays, out=out)
                else:
                    invalidation_array = invalidation_arrays[0]
                if not ignore_invalidation_bits:
                    vals = vals[nonzero(~invalidation_array)[0]]
                    if master_is_required:
                        if timestamps is None:
                            raise RuntimeError("'timestamps' cannot be None if 'master_is_required' is True")
                        timestamps = timestamps[nonzero(~invalidation_array)[0]]
                    invalidation_bits = None
                else:
                    invalidation_bits = invalidation_array
            else:
                invalidation_bits = None

            if raster:
                if timestamps is None:
                    raise RuntimeError("'timestamps' cannot be None if 'raster' is not None")
                if len(timestamps) > 1:
                    num = float(float32((timestamps[-1] - timestamps[0]) / raster))
                    if num.is_integer():
                        t = linspace(timestamps[0], timestamps[-1], int(num))
                    else:
                        t = arange(timestamps[0], timestamps[-1], raster)

                    signal = Signal(vals, timestamps, name="_", invalidation_bits=invalidation_bits).interp(
                        t,
                        integer_interpolation_mode=self._integer_interpolation,
                        float_interpolation_mode=self._float_interpolation,
                    )

                    vals, timestamps, invalidation_bits = (
                        signal.samples,
                        signal.timestamps,
                        signal.invalidation_bits,
                    )

            if channel.conversion:
                vals = channel.conversion.convert(vals)

        else:
            channel_group = grp.channel_group

            record_size = channel_group.samples_byte_nr

            if one_piece:
                fragment = next(fragments)
                data_bytes, rec_offset, rec_count = fragment.data, fragment.record_offset, fragment.record_count

                record = self._prepare_record(grp)
                info = record[ch_nr]

                if info is not None:
                    dtype_, byte_size, byte_offset, bit_offset = info
                    buffer: Buffer
                    if ch_nr == 0 and len(grp.channels) == 1 and channel.dtype_fmt.itemsize == record_size:
                        buffer = bytearray(data_bytes)
                    else:
                        if fragment.is_record:
                            buffer = get_channel_raw_bytes(
                                data_bytes,
                                record_size + channel_group.invalidation_bytes_nr,
                                byte_offset,
                                byte_size,
                            )
                        else:
                            buffer = data_bytes

                    vals = frombuffer(buffer, dtype=dtype_)

                    if not channel.standard_C_size:
                        size = byte_size

                        view: DTypeLike
                        if channel_dtype.byteorder == "=" and data_type in (
                            v4c.DATA_TYPE_SIGNED_MOTOROLA,
                            v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                        ):
                            view = np.dtype(f">u{vals.itemsize}")
                        else:
                            view = np.dtype(f"{channel_dtype.byteorder}u{vals.itemsize}")

                        if view != vals.dtype:
                            vals = vals.view(view)

                        if bit_offset:
                            vals >>= bit_offset

                        if bit_count != size * 8:
                            if data_type in v4c.SIGNED_INT:
                                vals = as_non_byte_sized_signed_int(vals, bit_count)
                            else:
                                mask = (1 << bit_count) - 1
                                vals &= mask
                        elif data_type in v4c.SIGNED_INT:
                            view = f"{channel_dtype.byteorder}i{vals.itemsize}"
                            if np.dtype(view) != vals.dtype:
                                vals = vals.view(view)

                    elif channel_type == v4c.CHANNEL_TYPE_VALUE and channel.fast_path is None:
                        channel.fast_path = (
                            gp_nr,
                            record_size + channel_group.invalidation_bytes_nr,
                            byte_offset,
                            byte_size,
                            channel.pos_invalidation_bit if channel_invalidation_present else -1,
                            # data_type,
                            # channel_type,
                            # bit_count,
                            dtype_,
                        )

                else:
                    vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                if bit_count == 1 and self._single_bit_uint_as_bool:
                    vals = array(vals, dtype=bool)

                if master_is_required:
                    timestamps = self.get_master(gp_nr, fragment, one_piece=True)
                else:
                    timestamps = None

                if channel_invalidation_present:
                    invalidation_array = self.get_invalidation_bits(gp_nr, channel.pos_invalidation_bit, fragment)

                    if not ignore_invalidation_bits:
                        vals = vals[nonzero(~invalidation_array)[0]]
                        if master_is_required:
                            if timestamps is None:
                                raise RuntimeError("'timestamps' cannot be None if 'master_is_required' is True")
                            timestamps = timestamps[nonzero(~invalidation_array)[0]]
                        invalidation_bits = None
                    else:
                        invalidation_bits = invalidation_array
                else:
                    invalidation_bits = None
            else:
                channel_values = []
                masters = []
                invalidation_arrays = []

                record = self._prepare_record(grp)
                info = record[ch_nr]

                if info is None:
                    for count, fragment in enumerate(fragments, 1):
                        data_bytes, offset, _count, invalidation_bytes = (
                            fragment.data,
                            fragment.record_offset,
                            fragment.record_count,
                            fragment.invalidation_data,
                        )

                        vals = self._get_not_byte_aligned_data(data_bytes, grp, ch_nr)

                        if bit_count == 1 and self._single_bit_uint_as_bool:
                            vals = array(vals, dtype=bool)

                        if master_is_required:
                            masters.append(self.get_master(gp_nr, fragment, one_piece=True))
                        if channel_invalidation_present:
                            invalidation_arrays.append(
                                self.get_invalidation_bits(gp_nr, channel.pos_invalidation_bit, fragment)
                            )

                        channel_values.append(vals)
                    vals = concatenate(channel_values)
                else:
                    dtype_, byte_size, byte_offset, bit_offset = info

                    buffers: list[bytearray] = []
                    count = 0

                    for count, fragment in enumerate(fragments, 1):
                        data_bytes = fragment.data

                        if ch_nr == 0 and len(grp.channels) == 1 and channel.dtype_fmt.itemsize == record_size:
                            buffers.append(bytearray(data_bytes))
                        else:
                            buffers.append(
                                get_channel_raw_bytes(
                                    data_bytes,
                                    record_size + channel_group.invalidation_bytes_nr,
                                    byte_offset,
                                    byte_size,
                                )
                            )

                        if master_is_required:
                            masters.append(self.get_master(gp_nr, fragment, one_piece=True))
                        if channel_invalidation_present:
                            invalidation_arrays.append(
                                self.get_invalidation_bits(gp_nr, channel.pos_invalidation_bit, fragment)
                            )

                    if count > 1:
                        buffer = bytearray().join(buffers)
                    elif count == 1:
                        buffer = buffers[0]
                    else:
                        buffer = bytearray()

                    vals = frombuffer(buffer, dtype=dtype_)

                    if not channel.standard_C_size:
                        size = dtype_.itemsize

                        if channel_dtype.byteorder == "=" and data_type in (
                            v4c.DATA_TYPE_SIGNED_MOTOROLA,
                            v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                        ):
                            view = f">u{vals.itemsize}"
                        else:
                            view = f"{channel_dtype.byteorder}u{vals.itemsize}"

                        if np.dtype(view) != dtype_:
                            vals = vals.view(view)

                        if bit_offset:
                            vals >>= bit_offset

                        if bit_count != size * 8:
                            if data_type in v4c.SIGNED_INT:
                                vals = as_non_byte_sized_signed_int(vals, bit_count)
                            else:
                                mask = (1 << bit_count) - 1
                                vals &= mask
                        elif data_type in v4c.SIGNED_INT:
                            view = f"{channel_dtype.byteorder}i{vals.itemsize}"
                            if np.dtype(view) != vals.dtype:
                                vals = vals.view(view)

                    if bit_count == 1 and self._single_bit_uint_as_bool:
                        vals = array(vals, dtype=bool)

                    total_size = len(vals)

                    if master_is_required:
                        if count > 1:
                            out = empty(total_size, dtype=masters[0].dtype)
                            timestamps = concatenate(masters, out=out)
                        elif count == 1:
                            timestamps = masters[0]
                        else:
                            timestamps = np.array([], dtype=np.float64)
                    else:
                        timestamps = None

                    if channel_invalidation_present:
                        if count > 1:
                            out = empty(total_size, dtype=invalidation_arrays[0].dtype)
                            invalidation_array = concatenate(invalidation_arrays, out=out)
                        elif count == 1:
                            invalidation_array = invalidation_arrays[0]
                        else:
                            invalidation_array = typing.cast(
                                np.ndarray[tuple[int], np.dtype[np.bool]], np.array([], dtype=np.bool)
                            )
                        if not ignore_invalidation_bits:
                            vals = vals[nonzero(~invalidation_array)[0]]
                            if master_is_required:
                                if timestamps is None:
                                    raise RuntimeError("'timestamps' cannot be None if 'master_is_required' is True")
                                timestamps = timestamps[nonzero(~invalidation_array)[0]]
                            invalidation_bits = None
                        else:
                            invalidation_bits = invalidation_array
                    else:
                        invalidation_bits = None

            if raster:
                if timestamps is None:
                    raise RuntimeError("'timestamps' cannot be None if 'raster' is not None")
                if len(timestamps) > 1:
                    num = float(float32((timestamps[-1] - timestamps[0]) / raster))
                    if num.is_integer():
                        t = linspace(timestamps[0], timestamps[-1], int(num))
                    else:
                        t = arange(timestamps[0], timestamps[-1], raster)

                    signal = Signal(vals, timestamps, name="_", invalidation_bits=invalidation_bits).interp(
                        t,
                        integer_interpolation_mode=self._integer_interpolation,
                        float_interpolation_mode=self._float_interpolation,
                    )

                    vals, timestamps, invalidation_bits = (
                        signal.samples,
                        signal.timestamps,
                        signal.invalidation_bits,
                    )

        if channel_type == v4c.CHANNEL_TYPE_VLSD and not skip_vlsd:
            count_ = len(vals)

            if count_:
                signal_data = self._load_signal_data(group=grp, index=ch_nr, offsets=(vals[0], vals[-1]))
            else:
                signal_data = b""

            max_vlsd_size = self.determine_max_vlsd_sample_size(group_index, channel_index)

            if signal_data:
                if data_type in (
                    v4c.DATA_TYPE_BYTEARRAY,
                    v4c.DATA_TYPE_UNSIGNED_INTEL,
                    v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                    v4c.DATA_TYPE_MIME_SAMPLE,
                    v4c.DATA_TYPE_MIME_STREAM,
                ):
                    vals = extract(signal_data, True, vals - vals[0])
                    if vals.shape[1] < max_vlsd_size:
                        vals = np.hstack(
                            (
                                vals,
                                np.zeros(
                                    (
                                        vals.shape[0],
                                        max_vlsd_size - vals.shape[1],
                                    ),
                                    dtype=vals.dtype,
                                ),
                            )
                        )
                else:
                    vals = extract(signal_data, False, vals - vals[0])

                if data_type not in (
                    v4c.DATA_TYPE_BYTEARRAY,
                    v4c.DATA_TYPE_UNSIGNED_INTEL,
                    v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                    v4c.DATA_TYPE_MIME_SAMPLE,
                    v4c.DATA_TYPE_MIME_STREAM,
                ):
                    match data_type:
                        case v4c.DATA_TYPE_STRING_UTF_16_BE:
                            encoding = "utf-16-be"

                        case v4c.DATA_TYPE_STRING_UTF_16_LE:
                            encoding = "utf-16-le"

                        case v4c.DATA_TYPE_STRING_UTF_8:
                            encoding = "utf-8"
                            vals = np.array(
                                [e.rsplit(b"\0")[0] for e in typing.cast(list[bytes], vals.tolist())],
                                dtype=vals.dtype,
                            )

                        case v4c.DATA_TYPE_STRING_LATIN_1:
                            encoding = "latin-1"
                            vals = np.array(
                                [e.rsplit(b"\0")[0] for e in typing.cast(list[bytes], vals.tolist())],
                                dtype=vals.dtype,
                            )

                        case _:
                            raise MdfException(f'wrong data type "{data_type}" for vlsd channel "{channel.name}"')

                    vals = vals.astype(f"S{max_vlsd_size}")
            else:
                if len(vals):
                    raise MdfException(
                        f'Wrong signal data block refence (0x{channel.data_block_addr:X}) for VLSD channel "{channel.name}"'
                    )
                # no VLSD signal data samples
                if data_type != v4c.DATA_TYPE_BYTEARRAY:
                    vals = array([], dtype=f"S{max_vlsd_size}")

                    match data_type:
                        case v4c.DATA_TYPE_STRING_UTF_16_BE:
                            encoding = "utf-16-be"

                        case v4c.DATA_TYPE_STRING_UTF_16_LE:
                            encoding = "utf-16-le"

                        case v4c.DATA_TYPE_STRING_UTF_8:
                            encoding = "utf-8"

                        case v4c.DATA_TYPE_STRING_LATIN_1:
                            encoding = "latin-1"

                        case _:
                            raise MdfException(f'wrong data type "{data_type}" for vlsd channel "{channel.name}"')

                else:
                    vals = array([], dtype=f"({max_vlsd_size},)u1")

        elif v4c.DATA_TYPE_STRING_LATIN_1 <= data_type <= v4c.DATA_TYPE_STRING_UTF_16_BE:
            if channel_type in (v4c.CHANNEL_TYPE_VALUE, v4c.CHANNEL_TYPE_MLSD):
                match data_type:
                    case v4c.DATA_TYPE_STRING_UTF_16_BE:
                        encoding = "utf-16-be"

                    case v4c.DATA_TYPE_STRING_UTF_16_LE:
                        encoding = "utf-16-le"

                    case v4c.DATA_TYPE_STRING_UTF_8:
                        encoding = "utf-8"

                    case v4c.DATA_TYPE_STRING_LATIN_1:
                        encoding = "latin-1"

                    case _:
                        raise MdfException(f'wrong data type "{data_type}" for string channel')

        elif data_type in (v4c.DATA_TYPE_CANOPEN_TIME, v4c.DATA_TYPE_CANOPEN_DATE):
            # CANopen date
            if data_type == v4c.DATA_TYPE_CANOPEN_DATE:
                types: np.dtype[Any] = np.dtype(
                    [
                        ("ms", "<u2"),
                        ("min", "<u1"),
                        ("hour", "<u1"),
                        ("day", "<u1"),
                        ("month", "<u1"),
                        ("year", "<u1"),
                    ]
                )
                vals = vals.view(types)

                arrays = [
                    vals["ms"],
                    vals["min"] & 0x3F,  # bit 6 and 7 of minutes are reserved
                    vals["hour"] & 0xF,  # only first 4 bits of hour are used
                    vals["day"] & 0xF,  # the first 4 bits are the day number
                    vals["month"] & 0x3F,  # bit 6 and 7 of month are reserved
                    vals["year"] & 0x7F,  # bit 7 of year is reserved
                    (vals["hour"] & 0x80) >> 7,  # add summer or standard time information for hour
                    (vals["day"] & 0xF0) >> 4,  # add day of week information
                ]

                names = [
                    "ms",
                    "min",
                    "hour",
                    "day",
                    "month",
                    "year",
                    "summer_time",
                    "day_of_week",
                ]
                vals = np.rec.fromarrays(arrays, names=names)  # type: ignore[call-overload]

            # CANopen time
            elif data_type == v4c.DATA_TYPE_CANOPEN_TIME:
                types = np.dtype([("ms", "<u4"), ("days", "<u2")])
                vals = vals.view(types)

        return vals, timestamps, invalidation_bits, encoding

    def _get_not_byte_aligned_data(self, data: bytes | bytearray, group: Group, ch_nr: int) -> NDArray[Any]:
        big_endian_types = (
            v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
            v4c.DATA_TYPE_REAL_MOTOROLA,
            v4c.DATA_TYPE_SIGNED_MOTOROLA,
        )

        if group.uses_ld:
            record_size = group.channel_group.samples_byte_nr
        else:
            record_size = group.channel_group.samples_byte_nr + group.channel_group.invalidation_bytes_nr

        channel = group.channels[ch_nr]

        bit_offset = channel.bit_offset
        byte_offset = channel.byte_offset
        bit_count = channel.bit_count

        if ch_nr >= 0:
            dependencies = group.channel_dependencies[ch_nr]
            if dependencies and isinstance(dependencies[0], ChannelArrayBlock):
                ca_block = dependencies[0]

                size = bit_count // 8

                shape = tuple(typing.cast(int, ca_block[f"dim_size_{i}"]) for i in range(ca_block.dims))
                if ca_block.byte_offset_base // size > 1 and len(shape) == 1:
                    shape += (ca_block.byte_offset_base // size,)
                dim = 1
                for d in shape:
                    dim *= d
                size *= dim
                bit_count = size * 8

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
        elif byte_size < 8:
            extra_bytes = 4 - (byte_size % 4)
        else:
            extra_bytes = 0

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

        if data_type in v4c.SIGNED_INT:
            return as_non_byte_sized_signed_int(vals, bit_count)
        elif data_type in v4c.FLOATS:
            return vals.view(get_fmt_v4(data_type, bit_count))
        else:
            return vals

    def _determine_max_vlsd_sample_size(self, group_idx: int, index: int) -> int:
        group_index = group_idx
        channel_index = index
        group = self.groups[group_idx]
        ch = group.channels[index]

        if ch.channel_type != v4c.CHANNEL_TYPE_VLSD:
            return 0

        if (group_index, ch.name) in self.vlsd_max_length:
            return self.vlsd_max_length[(group_index, ch.name)]
        else:
            offsets, *_ = self._get_scalar(
                ch,
                group,
                group_index,
                channel_index,
                group.channel_dependencies[channel_index],
                raster=None,
                data=None,
                ignore_invalidation_bits=True,
                record_offset=0,
                record_count=None,
                master_is_required=False,
                skip_vlsd=True,
            )
            offsets = offsets.astype("u8")

            data = self._load_signal_data(group, channel_index)

            max_size = get_vlsd_max_sample_size(data, offsets, len(offsets))

            return max_size

    def included_channels(
        self,
        index: int | None = None,
        channels: ChannelsType | None = None,
        skip_master: bool = True,
        minimal: bool = True,
    ) -> dict[int, dict[int, list[int]]]:
        if channels is None:
            if index is None:
                raise ValueError("index cannot be None if channels is None")
            virtual_channel_group = self.virtual_groups[index]
            groups = virtual_channel_group.groups

            gps: dict[int, list[int]] = {}

            for gp_index in groups:
                group = self.groups[gp_index]

                included_channels = set(range(len(group.channels)))
                master_index = self.masters_db.get(gp_index, None)
                if master_index is not None:
                    included_channels.remove(master_index)

                for dependencies in group.channel_dependencies:
                    if dependencies is None:
                        continue

                    if all(not isinstance(dep, ChannelArrayBlock) for dep in dependencies):
                        dependencies = typing.cast(list[tuple[int, int]], dependencies)
                        for _, ch_nr in dependencies:
                            try:
                                included_channels.remove(ch_nr)
                            except KeyError:
                                pass
                    else:
                        dependencies = typing.cast(list[ChannelArrayBlock], dependencies)
                        for dep in dependencies:
                            for referenced_channels in (
                                dep.axis_channels,
                                dep.dynamic_size_channels,
                                dep.input_quantity_channels,
                            ):
                                for gp_nr, ch_nr in filter(None, referenced_channels):
                                    if gp_nr == gp_index:
                                        try:
                                            included_channels.remove(ch_nr)
                                        except KeyError:
                                            pass

                            if dep.output_quantity_channel:
                                gp_nr, ch_nr = dep.output_quantity_channel
                                if gp_nr == gp_index:
                                    try:
                                        included_channels.remove(ch_nr)
                                    except KeyError:
                                        pass

                            if dep.comparison_quantity_channel:
                                gp_nr, ch_nr = dep.comparison_quantity_channel
                                if gp_nr == gp_index:
                                    try:
                                        included_channels.remove(ch_nr)
                                    except KeyError:
                                        pass

                gps[gp_index] = sorted(included_channels)

            result = {index: gps}
        else:
            gps_set: dict[int, set[int]] = {}
            for item in channels:
                if isinstance(item, list | tuple):
                    if len(item) not in (2, 3):
                        raise MdfException(
                            "The items used for filtering must be strings, "
                            "or they must match the first 3 arguments of the get "
                            "method"
                        )
                    else:
                        gp_nr, idx = self._validate_channel_selection(*item)
                        gps_idx = gps_set.setdefault(gp_nr, set())
                        gps_idx.add(idx)
                else:
                    name = item
                    gp_nr, idx = self._validate_channel_selection(name)
                    gps_idx = gps_set.setdefault(gp_nr, set())
                    gps_idx.add(idx)

            result = {}
            for gp_index, gps_idx in gps_set.items():
                master = self.virtual_groups_map[gp_index]
                group = self.groups[gp_index]

                if minimal:
                    channel_dependencies = [group.channel_dependencies[ch_nr] for ch_nr in gps_idx]

                    for dependencies in channel_dependencies:
                        if dependencies is None:
                            continue

                        if all(not isinstance(dep, ChannelArrayBlock) for dep in dependencies):
                            dependencies = typing.cast(list[tuple[int, int]], dependencies)
                            for _, ch_nr in dependencies:
                                try:
                                    gps_idx.remove(ch_nr)
                                except KeyError:
                                    pass
                        else:
                            dependencies = typing.cast(list[ChannelArrayBlock], dependencies)
                            for dep in dependencies:
                                for referenced_channels in (
                                    dep.axis_channels,
                                    dep.dynamic_size_channels,
                                    dep.input_quantity_channels,
                                ):
                                    for gp_nr, ch_nr in filter(None, referenced_channels):
                                        if gp_nr == gp_index:
                                            try:
                                                gps_idx.remove(ch_nr)
                                            except KeyError:
                                                pass

                                if dep.output_quantity_channel:
                                    gp_nr, ch_nr = dep.output_quantity_channel
                                    if gp_nr == gp_index:
                                        try:
                                            gps_idx.remove(ch_nr)
                                        except KeyError:
                                            pass

                                if dep.comparison_quantity_channel:
                                    gp_nr, ch_nr = dep.comparison_quantity_channel
                                    if gp_nr == gp_index:
                                        try:
                                            gps_idx.remove(ch_nr)
                                        except KeyError:
                                            pass

                    gp_master = self.masters_db.get(gp_index, None)
                    if gp_master is not None and gp_master in gps_idx:
                        gps_idx.remove(gp_master)

                if master not in result:
                    result[master] = {}
                    result[master][master] = [self.masters_db.get(master, None)]  # type: ignore[list-item]

                result[master][gp_index] = sorted(gps_idx)

        return result

    def _yield_selected_signals(
        self,
        index: int,
        groups: dict[int, list[int]] | None = None,
        record_offset: int = 0,
        record_count: int | None = None,
        skip_master: bool = True,
        version: str | None = None,
    ) -> Iterator[list[Signal] | list[tuple[NDArray[Any], None]]]:
        version = version or self.version
        virtual_channel_group = self.virtual_groups[index]
        record_size = virtual_channel_group.record_size

        from time import perf_counter

        tt = perf_counter()

        if groups is None:
            groups = self.included_channels(index, skip_master=skip_master)[index]

        record_size = 0
        for group_index in groups:
            grp = self.groups[group_index]
            record_size += grp.channel_group.samples_byte_nr + grp.channel_group.invalidation_bytes_nr

        record_size = record_size or 1

        if self._read_fragment_size:
            count = self._read_fragment_size // record_size or 1
        else:
            if version < "4.20":
                count = 64 * 1024 * 1024 // record_size or 1
            else:
                count = 128 * 1024 * 1024 // record_size or 1

        data_streams: list[Iterator[Fragment]] = []
        for idx, group_index in enumerate(groups):
            grp = self.groups[group_index]
            grp.read_split_count = count
            data_streams.append(self._load_data(grp, record_offset=record_offset, record_count=record_count))
            if group_index == index:
                master_index = idx

        encodings: dict[int, list[tuple[str | None, np.dtype[Any]] | None]] = {
            group_index: [None] for group_index in groups
        }

        self._set_temporary_master(None)
        idx = 0

        group_info: dict[int, list[list[int]]] = {}
        for group_index, channels in groups.items():
            grp = self.groups[group_index]
            if not grp.single_channel_dtype:
                self._prepare_record(grp)

            ch_info: list[list[int]] = []
            group_info[group_index] = ch_info
            dependency_list = grp.channel_dependencies
            info = self._prepare_record(grp)

            for channel_index in channels:
                channel = grp.channels[channel_index]

                if (
                    channel.byte_offset + (channel.bit_offset + channel.bit_count) / 8
                    > grp.channel_group.samples_byte_nr
                ):
                    ch_info.append([0, 0])
                elif dependency_list[channel_index]:
                    ch_info.append([0, 0])
                else:
                    if (record := info[channel_index]) is not None:
                        _, byte_size, byte_offset, _ = record
                        ch_info.append([byte_offset, byte_size])
                    else:
                        ch_info.append([0, 0])

        while True:
            try:
                fragments = [next(stream) for stream in data_streams]
            except:
                break
            #
            # if perf_counter() - tt > 120:
            #     x = 1 / 0

            # prepare the master
            _master = self.get_master(index, data=fragments[master_index], one_piece=True)
            self._set_temporary_master(_master)

            signals: list[Signal] | list[tuple[NDArray[Any], None]]
            if idx == 0:
                signals = []
            else:
                signals = [(_master, None)]

            for fragment, (group_index, channels) in zip(fragments, groups.items(), strict=False):
                grp = self.groups[group_index]
                if not grp.single_channel_dtype:
                    self._prepare_record(grp)

                self._invalidation_cache.clear()

                if 1 and len(channels) >= 100:
                    # prepare the invalidation bytes for this group and fragment
                    invalidation_bytes = get_channel_raw_bytes(
                        fragment.data,
                        grp.channel_group.samples_byte_nr + grp.channel_group.invalidation_bytes_nr,
                        grp.channel_group.samples_byte_nr,
                        grp.channel_group.invalidation_bytes_nr,
                    )

                    channels_raw_data = get_channel_raw_bytes_parallel(
                        fragment.data,
                        grp.channel_group.samples_byte_nr + grp.channel_group.invalidation_bytes_nr,
                        group_info[group_index],
                        THREAD_COUNT,
                    )

                    if idx == 0:
                        sigs = typing.cast(list[Signal], signals)
                        for channel_index, raw_data in zip(channels, channels_raw_data, strict=False):
                            signal = self.get(
                                group=group_index,
                                index=channel_index,
                                data=(
                                    Fragment(
                                        data=raw_data,
                                        record_offset=fragment.record_offset,
                                        record_count=fragment.record_count,
                                        invalidation_data=invalidation_bytes,
                                        is_record=False,
                                    )
                                    if raw_data
                                    else fragment
                                ),
                                raw=True,
                                ignore_invalidation_bits=True,
                                samples_only=False,
                                skip_channel_validation=True,
                            )

                            sigs.append(signal)

                    else:
                        for channel_index, raw_data in zip(channels, channels_raw_data, strict=False):
                            samples, invalidation_bits = self.get(
                                group=group_index,
                                index=channel_index,
                                data=(
                                    Fragment(
                                        data=raw_data,
                                        record_offset=fragment.record_offset,
                                        record_count=fragment.record_count,
                                        invalidation_data=invalidation_bytes,
                                        is_record=False,
                                    )
                                    if raw_data
                                    else fragment
                                ),
                                raw=True,
                                ignore_invalidation_bits=True,
                                samples_only=True,
                                skip_channel_validation=True,
                            )

                            signals.append((samples, invalidation_bits))
                else:
                    if idx == 0:
                        sigs = typing.cast(list[Signal], signals)
                        for channel_index in channels:
                            signal = self.get(
                                group=group_index,
                                index=channel_index,
                                data=fragment,
                                raw=True,
                                ignore_invalidation_bits=True,
                                samples_only=False,
                            )

                            sigs.append(signal)

                    else:
                        for channel_index in channels:
                            samples, invalidation_bits = self.get(
                                group=group_index,
                                index=channel_index,
                                data=fragment,
                                raw=True,
                                ignore_invalidation_bits=True,
                                samples_only=True,
                            )

                            signals.append((samples, invalidation_bits))

                if version < "4.00":
                    if idx == 0:
                        sigs = typing.cast(list[Signal], signals)
                        for sig, channel_index in zip(sigs, channels, strict=False):
                            if sig.samples.dtype.kind == "S":
                                strsig = self.get(
                                    group=group_index,
                                    index=channel_index,
                                    samples_only=True,
                                    ignore_invalidation_bits=True,
                                )[0]

                                _dtype = strsig.dtype
                                sig.samples = sig.samples.astype(_dtype)
                                encodings[group_index].append((sig.encoding, _dtype))
                                del strsig
                                if sig.encoding != "latin-1":
                                    if sig.encoding == "utf-16-le":
                                        sig.samples = sig.samples.view(uint16).byteswap().view(sig.samples.dtype)
                                        sig.samples = encode(decode(sig.samples, "utf-16-be"), "latin-1")
                                    else:
                                        sig.samples = encode(
                                            decode(sig.samples, sig.encoding),
                                            "latin-1",
                                        )
                                sig.samples = sig.samples.astype(_dtype)
                            else:
                                encodings[group_index].append(None)
                    else:
                        for i, (sig_samples, encoding_tuple) in enumerate(
                            zip(signals, encodings[group_index], strict=False)
                        ):
                            if encoding_tuple:
                                encoding, _dtype = encoding_tuple
                                samples = sig_samples[0]
                                if encoding != "latin-1":
                                    if encoding == "utf-16-le":
                                        samples = samples.view(uint16).byteswap().view(samples.dtype)
                                        samples = encode(decode(samples, "utf-16-be"), "latin-1")
                                    else:
                                        samples = encode(decode(samples, encoding), "latin-1")
                                samples = samples.astype(_dtype)
                                signals[i] = (samples, sig_samples[1])

            self._set_temporary_master(None)
            idx += 1
            yield signals

        grp.read_split_count = 0
        self._invalidation_cache.clear()

    def get_master(
        self,
        index: int,
        data: Fragment | None = None,
        record_offset: int = 0,
        record_count: int | None = None,
        one_piece: bool = False,
    ) -> NDArray[Any]:
        """Get master channel samples for the given group.

        Parameters
        ----------
        index : int
            Group index.
        data : Fragment, optional
            Data bytes as a Fragment.
        record_offset : int, optional
            If `data=None`, use this to select the record offset from which the
            group data should be loaded.
        record_count : int, optional
            Number of records to read; default is None and in this case all
            available records are used.

        Returns
        -------
        t, virtual_master_conversion : (np.ndarray, ChannelConversion | None)
            Master channel samples and virtual master conversion.
        """

        if self._master is not None:
            return self._master

        group = self.groups[index]
        if group.channel_group.flags & v4c.FLAG_CG_REMOTE_MASTER:
            if data is not None:
                record_offset = data.record_offset
                record_count = data.record_count
            return self.get_master(
                typing.cast(int, group.channel_group.cg_master_index),
                record_offset=record_offset,
                record_count=record_count,
            )

        time_ch_nr = self.masters_db.get(index, None)
        channel_group = group.channel_group
        record_size = channel_group.samples_byte_nr
        record_size += channel_group.invalidation_bytes_nr
        if record_count is not None:
            cycles_nr = record_count
        else:
            cycles_nr = group.channel_group.cycles_nr

        fragment = data
        if fragment:
            data_bytes, offset, _, invalidation_bytes = (
                fragment.data,
                fragment.record_offset,
                fragment.record_count,
                fragment.invalidation_data,
            )
            cycles_nr = len(data_bytes) // record_size if record_size else 0
            record_offset = fragment.record_offset
            record_count = fragment.record_count
        else:
            offset = 0

        t: NDArray[Any]
        if time_ch_nr is None:
            if record_size:
                t = arange(cycles_nr, dtype=float64)
                t += offset
            else:
                t = array([], dtype=float64)
            metadata = ("timestamps", v4c.SYNC_TYPE_TIME)

        else:
            time_ch = group.channels[time_ch_nr]
            time_conv = time_ch.conversion
            time_name = time_ch.name

            metadata = (time_name, time_ch.sync_type)

            if time_ch.channel_type == v4c.CHANNEL_TYPE_VIRTUAL_MASTER:
                if record_count is None:
                    t = arange(record_offset, cycles_nr, 1, dtype=float64)
                else:
                    t = arange(record_offset, record_offset + record_count, 1, dtype=float64)

                if time_conv is not None:
                    t = time_conv.convert(t)

            else:
                # check if the channel group contains just the master channel
                # and that there are no padding bytes
                if len(group.channels) == 1 and time_ch.dtype_fmt.itemsize == record_size:
                    if one_piece:
                        if data is None:
                            raise RuntimeError("data is None")

                        t = frombuffer(data.data, dtype=time_ch.dtype_fmt)
                    else:
                        # get data
                        fragments: Iterable[Fragment]
                        if fragment is None:
                            fragments = self._load_data(
                                group,
                                record_offset=record_offset,
                                record_count=record_count,
                            )
                        else:
                            fragments = (fragment,)

                        buffer = bytearray().join([fragment.data for fragment in fragments])

                        t = frombuffer(buffer, dtype=time_ch.dtype_fmt)

                else:
                    record = typing.cast(list[tuple[np.dtype[Any], int, int, int]], self._prepare_record(group))
                    dtype_, byte_size, byte_offset, bit_offset = record[time_ch_nr]

                    if one_piece:
                        if data is None:
                            raise RuntimeError("data is None")

                        data_bytes = data.data

                        buffer = get_channel_raw_bytes(
                            data_bytes,
                            record_size,
                            byte_offset,
                            byte_size,
                        )

                        t = frombuffer(buffer, dtype=dtype_)

                    else:
                        # get data
                        if fragment is None:
                            fragments = self._load_data(
                                group,
                                record_offset=record_offset,
                                record_count=record_count,
                            )
                        else:
                            fragments = (fragment,)

                        buffer = bytearray().join(
                            [
                                get_channel_raw_bytes(
                                    fragment.data,
                                    record_size,
                                    byte_offset,
                                    byte_size,
                                )
                                for fragment in fragments
                            ]
                        )

                        t = frombuffer(buffer, dtype=dtype_)

                    if not time_ch.standard_C_size:
                        channel_dtype = time_ch.dtype_fmt
                        bit_count = time_ch.bit_count
                        data_type = time_ch.data_type

                        size = byte_size

                        if channel_dtype.byteorder == "=" and time_ch.data_type in (
                            v4c.DATA_TYPE_SIGNED_MOTOROLA,
                            v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
                        ):
                            dtype = f">u{t.itemsize}"
                        else:
                            dtype = f"{channel_dtype.byteorder}u{t.itemsize}"

                        if np.dtype(dtype) != t.dtype:
                            t = t.view(dtype)

                        if bit_offset:
                            t >>= bit_offset

                        if bit_count != size * 8:
                            if data_type in v4c.SIGNED_INT:
                                t = as_non_byte_sized_signed_int(t, bit_count)
                            else:
                                mask = (1 << bit_count) - 1
                                t &= mask
                        elif data_type in v4c.SIGNED_INT:
                            dtype = f"{channel_dtype.byteorder}i{t.itemsize}"
                            if np.dtype(dtype) != t.dtype:
                                t = t.view(dtype)

                # get timestamps
                if time_conv:
                    t = time_conv.convert(t)

        self._master_channel_metadata[index] = metadata

        if t.dtype != float64:
            t = t.astype(float64)

        return t

    def get_bus_signal(
        self,
        bus: BusType,
        name: str,
        database: CanMatrix | StrPath | None = None,
        ignore_invalidation_bits: bool = False,
        data: Fragment | None = None,
        raw: bool = False,
        ignore_value2text_conversion: bool = True,
    ) -> Signal:
        """Get a signal decoded from a raw bus logging. The currently supported
        buses are CAN and LIN (LDF databases are not supported, they need to be
        converted to DBC and fed to this function).

        .. versionadded:: 6.0.0

        Parameters
        ----------
        bus : str
            "CAN" or "LIN".
        name : str
            Signal name.
        database : str | path-like | CanMatrix, optional
            Path of external CAN/LIN database file (.dbc or .arxml) or
            canmatrix.CanMatrix.

            .. versionchanged:: 6.0.0
                `db` and `database` arguments were merged into this single
                argument.

        ignore_invalidation_bits : bool, default False
            Option to ignore invalidation bits.
        data : Fragment, optional
            Data bytes as a Fragment.
        raw : bool, default False
            Return channel samples without applying the conversion rule.
        ignore_value2text_conversion : bool, default True
            Return channel samples without values that have a description in
            .dbc or .arxml file.

        Returns
        -------
        sig : Signal
            Signal object with the physical values.
        """

        match bus:
            case "CAN":
                return self.get_can_signal(
                    name,
                    database=database,
                    ignore_invalidation_bits=ignore_invalidation_bits,
                    data=data,
                    raw=raw,
                    ignore_value2text_conversion=ignore_value2text_conversion,
                )
            case "LIN":
                return self.get_lin_signal(
                    name,
                    database=database,
                    ignore_invalidation_bits=ignore_invalidation_bits,
                    data=data,
                    raw=raw,
                    ignore_value2text_conversion=ignore_value2text_conversion,
                )

    def get_can_signal(
        self,
        name: str,
        database: CanMatrix | StrPath | None = None,
        ignore_invalidation_bits: bool = False,
        data: Fragment | None = None,
        raw: bool = False,
        ignore_value2text_conversion: bool = True,
    ) -> Signal:
        """Get CAN message signal. You can specify an external CAN database path
        or a canmatrix database object that has already been loaded from a file.

        The signal name can be specified in the following ways:

        * ``CAN<ID>.<MESSAGE_NAME>.<SIGNAL_NAME>`` - the `ID` value starts from 1
          and must match the ID found in the measurement (the source CAN bus ID).
          Example: CAN1.Wheels.FL_WheelSpeed

        * ``CAN<ID>.CAN_DataFrame_<MESSAGE_ID>.<SIGNAL_NAME>`` - the `ID` value
          starts from 1 and the `MESSAGE_ID` is the decimal message ID as found
          in the database. Example: CAN1.CAN_DataFrame_218.FL_WheelSpeed

        * ``<MESSAGE_NAME>.<SIGNAL_NAME>`` - in this case the first occurrence
          of the message name and signal are returned (the same message could
          be found on multiple CAN buses; for example on CAN1 and CAN3).
          Example: Wheels.FL_WheelSpeed

        * ``CAN_DataFrame_<MESSAGE_ID>.<SIGNAL_NAME>`` - in this case the first
          occurrence of the message name and signal are returned (the same
          message could be found on multiple CAN buses; for example on CAN1 and
          CAN3). Example: CAN_DataFrame_218.FL_WheelSpeed

        * ``<SIGNAL_NAME>`` - in this case the first occurrence of the signal
          name is returned (the same signal name could be found in multiple
          messages and on multiple CAN buses). Example: FL_WheelSpeed

        Parameters
        ----------
        name : str
            Signal name.
        database : str | path-like | CanMatrix, optional
            Path of external CAN database file (.dbc or .arxml) or
            canmatrix.CanMatrix.

            .. versionchanged:: 6.0.0
                `db` and `database` arguments were merged into this single
                argument.

        ignore_invalidation_bits : bool, default False
            Option to ignore invalidation bits.
        data : Fragment, optional
            Data bytes as a Fragment.
        raw : bool, default False
            Return channel samples without applying the conversion rule.
        ignore_value2text_conversion : bool, default True
            Return channel samples without values that have a description in
            .dbc or .arxml file.

        Returns
        -------
        sig : Signal
            Signal object with the physical values.
        """

        if database is None:
            return self.get(name)

        if isinstance(database, (str, Path)):
            database_path = Path(database)
            if database_path.suffix.lower() not in (".arxml", ".dbc"):
                message = f'Expected .dbc or .arxml file as CAN channel attachment but got "{database_path}"'
                logger.exception(message)
                raise MdfException(message)
            else:
                db_string = database_path.read_bytes()
                md5_sum = md5(db_string).digest()

                if md5_sum in self._external_dbc_cache:
                    db = self._external_dbc_cache[md5_sum]
                else:
                    db = load_can_database(database_path, contents=db_string)
                    if db is None:
                        raise MdfException("failed to load database")
        else:
            db = database

        is_j1939 = db.contains_j1939

        name_ = name.split(".")

        message_id: str | int
        if len(name_) == 3:
            can_id_str, message_id_str, signal_name = name_

            match = v4c.CAN_ID_PATTERN.search(can_id_str)
            if match is None:
                raise MdfException(f'CAN id "{can_id_str}" of signal name "{name}" is not recognised by this library')
            else:
                can_id = int(match.group("id"))

            match = v4c.CAN_DATA_FRAME_PATTERN.search(message_id_str)
            if match is None:
                message_id = message_id_str
            else:
                message_id = int(match.group("id"))

            if isinstance(message_id, str):
                frame = db.frame_by_name(message_id)
            else:
                frame = db.frame_by_id(message_id)

        elif len(name_) == 2:
            message_id_str, signal_name = name_

            can_id = None

            match = v4c.CAN_DATA_FRAME_PATTERN.search(message_id_str)
            if match is None:
                message_id = message_id_str
            else:
                message_id = int(match.group("id"))

            if isinstance(message_id, str):
                frame = db.frame_by_name(message_id)
            else:
                frame = db.frame_by_id(message_id)

        else:
            frame = None
            for msg in db:
                for signal in msg:
                    if signal.name == name:
                        frame = msg

            can_id = None
            signal_name = name

        if frame is None:
            raise MdfException(f"Could not find signal {name} in {database}")

        for signal in frame.signals:
            if signal.name == signal_name:
                break
        else:
            raise MdfException(f'Signal "{signal_name}" not found in message "{frame.name}" of "{database}"')

        test_ids: Collection[int]
        if can_id is None:
            index = None
            for _can_id, messages in self.bus_logging_map["CAN"].items():
                if is_j1939:
                    test_ids = [
                        canmatrix.ArbitrationId(id_, extended=True).pgn for id_ in self.bus_logging_map["CAN"][_can_id]
                    ]

                    id_ = frame.arbitration_id.pgn

                else:
                    id_ = frame.arbitration_id.id
                    test_ids = self.bus_logging_map["CAN"][_can_id]

                if id_ in test_ids:
                    if is_j1939:
                        for id__, idx in self.bus_logging_map["CAN"][_can_id].items():
                            if canmatrix.ArbitrationId(id__, extended=True).pgn == id_:
                                index = idx
                                break
                    else:
                        index = self.bus_logging_map["CAN"][_can_id][frame.arbitration_id.id]

                if index is not None:
                    break
            else:
                raise MdfException(
                    f'Message "{frame.name}" (ID={hex(frame.arbitration_id.id)}) not found in the measurement'
                )
        else:
            if can_id in self.bus_logging_map["CAN"]:
                if is_j1939:
                    test_ids = [
                        canmatrix.ArbitrationId(id_, extended=True).pgn for id_ in self.bus_logging_map["CAN"][can_id]
                    ]
                    id_ = frame.arbitration_id.pgn

                else:
                    id_ = frame.arbitration_id.id
                    test_ids = self.bus_logging_map["CAN"][can_id]

                if id_ in test_ids:
                    if is_j1939:
                        for id__, idx in self.bus_logging_map["CAN"][can_id].items():
                            if canmatrix.ArbitrationId(id__, extended=True).pgn == id_:
                                index = idx
                                break
                    else:
                        index = self.bus_logging_map["CAN"][can_id][frame.arbitration_id.id]
                else:
                    raise MdfException(
                        f'Message "{frame.name}" (ID={hex(frame.arbitration_id.id)}) not found in the measurement'
                    )
            else:
                raise MdfException(f'No logging from "{can_id}" was found in the measurement')

        can_ids = self.get(
            "CAN_DataFrame.ID",
            group=index,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
        )
        can_ids.samples = can_ids.samples.astype("<u4") & 0x1FFFFFFF

        payload = self.get(
            "CAN_DataFrame.DataBytes",
            group=index,
            samples_only=True,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
        )[0]

        if is_j1939:
            tmp_pgn = can_ids.samples >> 8
            ps = tmp_pgn & 0xFF
            pf = (can_ids.samples >> 16) & 0xFF
            _pgn = tmp_pgn & 0x3FF00
            can_ids.samples = where(pf >= 240, _pgn + ps, _pgn)

            indexes = argwhere(can_ids.samples == frame.arbitration_id.pgn).ravel()
        else:
            indexes = argwhere(can_ids.samples == frame.arbitration_id.id).ravel()

        payload = payload[indexes]
        t = can_ids.timestamps[indexes].copy()

        if can_ids.invalidation_bits is not None:
            invalidation_bits = can_ids.invalidation_bits[indexes]
        else:
            invalidation_bits = None

        if not ignore_invalidation_bits and invalidation_bits is not None:
            payload = payload[nonzero(~invalidation_bits)[0]]
            t = t[nonzero(~invalidation_bits)[0]]

        extracted_signals = bus_logging_utils.extract_mux(
            payload,
            frame,
            None,
            None,
            t,
            original_message_id=None,
            ignore_value2text_conversion=ignore_value2text_conversion,
            raw=raw,
        )

        comment = signal.comment or ""

        for entry, signals in extracted_signals.items():
            for signal_name, extracted_signal in signals.items():
                if signal_name == signal.name:
                    sig = Signal(
                        samples=extracted_signal["samples"],
                        timestamps=extracted_signal["t"],
                        name=signal_name,
                        unit=signal.unit or "",
                        comment=comment,
                        raw=raw,
                        conversion=extracted_signal["conversion"],
                    )
                    if len(sig):
                        return sig
                    else:
                        raise MdfException(f'No logging from "{signal}" was found in the measurement')

        raise MdfException(f'No logging from "{signal}" was found in the measurement')

    def get_lin_signal(
        self,
        name: str,
        database: CanMatrix | str | Path | None = None,
        ignore_invalidation_bits: bool = False,
        data: Fragment | None = None,
        raw: bool = False,
        ignore_value2text_conversion: bool = True,
    ) -> Signal:
        """Get LIN message signal. You can specify an external LIN database path
        or a canmatrix database object that has already been loaded from a file.

        The signal name can be specified in the following ways:

        * ``LIN_Frame_<MESSAGE_ID>.<SIGNAL_NAME>`` - Example: LIN_Frame_218.FL_WheelSpeed

        * ``<MESSAGE_NAME>.<SIGNAL_NAME>`` - Example: Wheels.FL_WheelSpeed

        * ``<SIGNAL_NAME>`` - Example: FL_WheelSpeed

        .. versionadded:: 6.0.0

        Parameters
        ----------
        name : str
            Signal name.
        database : str | path-like | CanMatrix, optional
            Path of external LIN database file (.dbc, .arxml or .ldf) or
            canmatrix.CanMatrix.
        ignore_invalidation_bits : bool, default False
            Option to ignore invalidation bits.
        data : Fragment, optional
            Data bytes as a Fragment.
        raw : bool, default False
            Return channel samples without applying the conversion rule.
        ignore_value2text_conversion : bool, default True
            Return channel samples without values that have a description in
            .dbc, .arxml or .ldf file.

        Returns
        -------
        sig : Signal
            Signal object with the physical values.
        """

        if database is None:
            return self.get(name)

        if isinstance(database, (str, Path)):
            database_path = Path(database)
            if database_path.suffix.lower() not in (".arxml", ".dbc", ".ldf"):
                message = f'Expected .dbc, .arxml or .ldf file as LIN channel attachment but got "{database_path}"'
                logger.exception(message)
                raise MdfException(message)
            else:
                db_string = database_path.read_bytes()
                md5_sum = md5(db_string).digest()

                if md5_sum in self._external_dbc_cache:
                    db = self._external_dbc_cache[md5_sum]
                else:
                    contents = None if database_path.suffix.lower() == ".ldf" else db_string
                    db = load_can_database(database_path, contents=contents)
                    if db is None:
                        raise MdfException("failed to load database")
        else:
            db = database

        name_ = name.split(".")

        if len(name_) == 2:
            message_id_str, signal_name = name_

            match = v4c.LIN_DATA_FRAME_PATTERN.search(message_id_str)
            message_id: str | int
            if match is None:
                message_id = message_id_str
            else:
                message_id = int(match.group("id"))

            if isinstance(message_id, str):
                frame = db.frame_by_name(message_id)
            else:
                frame = db.frame_by_id(message_id)

        else:
            frame = None
            for msg in db:
                for signal in msg:
                    if signal.name == name:
                        frame = msg

            signal_name = name

        if frame is None:
            raise MdfException(f"Could not find signal {name} in {database}")

        for signal in frame.signals:
            if signal.name == signal_name:
                break
        else:
            raise MdfException(f'Signal "{signal_name}" not found in message "{frame.name}" of "{database}"')

        id_ = frame.arbitration_id.id

        if id_ in self.bus_logging_map["LIN"]:
            index = self.bus_logging_map["LIN"][id_]
        else:
            raise MdfException(
                f'Message "{frame.name}" (ID={hex(frame.arbitration_id.id)}) not found in the measurement'
            )

        can_ids = self.get(
            "LIN_Frame.ID",
            group=index,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
        )
        can_ids.samples = can_ids.samples.astype("<u4") & 0x1FFFFFFF
        payload = self.get(
            "LIN_Frame.DataBytes",
            group=index,
            samples_only=True,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
        )[0]

        idx = argwhere(can_ids.samples == frame.arbitration_id.id).ravel()

        payload = payload[idx]
        t = can_ids.timestamps[idx].copy()

        if can_ids.invalidation_bits is not None:
            invalidation_bits = can_ids.invalidation_bits[idx]
        else:
            invalidation_bits = None

        if not ignore_invalidation_bits and invalidation_bits is not None:
            payload = payload[nonzero(~invalidation_bits)[0]]
            t = t[nonzero(~invalidation_bits)[0]]

        extracted_signals = bus_logging_utils.extract_mux(
            payload,
            frame,
            None,
            None,
            t,
            original_message_id=None,
            ignore_value2text_conversion=ignore_value2text_conversion,
            raw=raw,
        )

        comment = signal.comment or ""

        for entry, signals in extracted_signals.items():
            for signal_name, extracted_signal in signals.items():
                if signal_name == signal.name:
                    sig = Signal(
                        samples=extracted_signal["samples"],
                        timestamps=extracted_signal["t"],
                        name=signal_name,
                        unit=signal.unit or "",
                        comment=comment,
                        conversion=extracted_signal["conversion"],
                        raw=raw,
                    )
                    if len(sig):
                        return sig
                    else:
                        raise MdfException(f'No logging from "{signal}" was found in the measurement')

        raise MdfException(f'No logging from "{signal}" was found in the measurement')

    def info(self) -> dict[str, object]:
        """Get MDF information as a dict.

        Examples
        --------
        >>> mdf = MDF('test.mdf')
        >>> mdf.info()
        """
        info: dict[str, object] = {
            "version": self.version,
            "program": self.identification.program_identification.decode("utf-8").strip(" \0\n\r\t"),
            "comment": self.header.comment,
        }
        info["groups"] = len(self.groups)
        for i, gp in enumerate(self.groups):
            inf: dict[str, object] = {}
            info[f"group {i}"] = inf
            inf["cycles"] = gp.channel_group.cycles_nr
            inf["comment"] = gp.channel_group.comment
            inf["channels count"] = len(gp.channels)
            for j, channel in enumerate(gp.channels):
                name = channel.name

                ch_type = v4c.CHANNEL_TYPE_TO_DESCRIPTION[channel.channel_type]
                inf[f"channel {j}"] = f'name="{name}" type={ch_type}'

        return info

    def _extract_can_logging(
        self,
        output_file: "MDF4",
        dbc_files: Iterable[DbcFileType],
        ignore_value2text_conversion: bool = True,
        prefix: str = "",
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> "MDF4":
        out = output_file

        max_flags: list[list[list[bool]]] = []

        valid_dbc_files: list[tuple[CanMatrix, StrPath, int]] = []
        unique_name = UniqueDB()
        for dbc_name, bus_channel in dbc_files:
            if isinstance(dbc_name, CanMatrix):
                valid_dbc_files.append(
                    (
                        dbc_name,
                        unique_name.get_unique_name("UserProvidedCanMatrix"),
                        bus_channel,
                    )
                )
            else:
                dbc = load_can_database(Path(dbc_name))
                if dbc is None:
                    continue
                else:
                    valid_dbc_files.append((dbc, dbc_name, bus_channel))

        count = sum(
            1
            for group in self.groups
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT
            and group.channel_group.acq_source
            and group.channel_group.acq_source.bus_type == v4c.BUS_TYPE_CAN
        )
        count *= len(valid_dbc_files)

        if progress is not None:
            if callable(progress):
                progress(0, count)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(count)

                if progress.stop:
                    raise Terminated

        cntr = 0

        total_unique_ids: set[tuple[int, bool]] = set()
        found_ids: defaultdict[StrPath, set[tuple[tuple[int, int, bool], str]]] = defaultdict(set)
        not_found_ids: defaultdict[StrPath, list[tuple[tuple[int, bool] | int, str]]] = defaultdict(list)
        unknown_ids: defaultdict[int | tuple[int, bool], list[bool]] = defaultdict(list)

        for dbc, dbc_name, bus_channel in valid_dbc_files:
            messages = {(message.arbitration_id.id, message.arbitration_id.extended): message for message in dbc}

            global_is_j1939 = dbc.attributes.get("ProtocolType", "").lower() == "j1939"
            not_extended = [msg for msg in dbc if not msg.arbitration_id.extended]
            if global_is_j1939 and not_extended:
                logger.warning(
                    f"Not all j1939 messages in <{dbc_name}> seem to use extended addressing. Disabling global j1939 flag..."
                )
                for msg in not_extended:
                    logger.warning(f"  {msg} with id {msg.arbitration_id}")
                global_is_j1939 = False  # Relax req on j1939 adressing

            j1939_messages = {
                (
                    message.arbitration_id.pgn,
                    message.arbitration_id.j1939_source,
                ): message
                for message in dbc
                if message.is_j1939 or global_is_j1939
            }

            current_not_found = {
                (
                    (
                        (message.arbitration_id.id, message.arbitration_id.extended)
                        if not message.is_j1939 and not global_is_j1939
                        else message.arbitration_id.pgn
                    ),
                    message.name,
                )
                for msg_id, message in messages.items()
            }

            msg_map: dict[tuple[int | None, int | None, bool, int | None, str | None, int, int], int] = {}

            for i, group in enumerate(self.groups):
                if (
                    not group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT
                    or (group.channel_group.acq_source and group.channel_group.acq_source.bus_type != v4c.BUS_TYPE_CAN)
                    or not "CAN_DataFrame" in [ch.name for ch in group.channels]
                ):
                    continue

                self._prepare_record(group)
                data = self._load_data(group, optimize_read=False)

                for fragment in data:
                    self._set_temporary_master(None)
                    self._set_temporary_master(self.get_master(i, data=fragment, one_piece=True))

                    bus_ids = self.get(
                        "CAN_DataFrame.BusChannel",
                        group=i,
                        data=fragment,
                    ).samples.astype("<u1")

                    msg_ids = self.get("CAN_DataFrame.ID", group=i, data=fragment).astype("<u4")
                    try:
                        msg_ide = self.get("CAN_DataFrame.IDE", group=i, data=fragment).samples.astype("<u1")
                    except:
                        msg_ide = ((msg_ids & 0x80000000) >> 31).samples

                    msg_ids &= 0x1FFFFFFF

                    data_bytes = self.get(
                        "CAN_DataFrame.DataBytes",
                        group=i,
                        data=fragment,
                    ).samples

                    buses = np.unique(bus_ids)

                    for bus in buses:
                        if bus_channel and bus != bus_channel:
                            continue

                        idx = np.argwhere(bus_ids == bus).ravel()
                        bus_t = msg_ids.timestamps[idx]
                        bus_msg_ids = msg_ids.samples[idx]
                        bus_msg_ide = msg_ide[idx]
                        bus_data_bytes = data_bytes[idx]

                        tmp_pgns = bus_msg_ids >> 8
                        pss = tmp_pgns & 0xFF
                        pfs = (bus_msg_ids >> 16) & 0xFF
                        _pgns = tmp_pgns & 0x3FF00
                        j1939_msg_pgns = np.where(pfs >= 240, _pgns + pss, _pgns)
                        j9193_msg_sa = bus_msg_ids & 0xFF

                        unique_ids = set(
                            zip(
                                typing.cast(list[int], bus_msg_ids.tolist()),
                                typing.cast(list[bool], bus_msg_ide.tolist()),
                                strict=False,
                            )
                        )

                        total_unique_ids = total_unique_ids | set(unique_ids)

                        for msg_id, is_extended in sorted(unique_ids):
                            message = messages.get((msg_id, is_extended), None)

                            if message is None:
                                tmp_pgn = msg_id >> 8
                                ps = tmp_pgn & 0xFF
                                pf = (msg_id >> 16) & 0xFF
                                _pgn = tmp_pgn & 0x3FF00
                                msg_pgn = _pgn + ps if pf >= 240 else _pgn

                                for (_pgn, _sa), _msg in j1939_messages.items():
                                    if _pgn == msg_pgn:
                                        message = _msg
                                        break
                                else:
                                    unknown_ids[msg_id].append(True)
                                    continue

                            is_j1939 = message.is_j1939 or global_is_j1939
                            if is_j1939:
                                source_address = msg_id & 0xFF
                                pgn_number = message.arbitration_id.pgn
                                key = (pgn_number, source_address, True)
                                found_ids[dbc_name].add((key, message.name))

                                try:
                                    current_not_found.remove((pgn_number, message.name))
                                except KeyError:
                                    pass

                            else:
                                key = msg_id, bool(is_extended), False

                                found_ids[dbc_name].add((key, message.name))
                                try:
                                    current_not_found.remove(((msg_id, is_extended), message.name))
                                except KeyError:
                                    pass

                            unknown_ids[(msg_id, is_extended)].append(False)

                            if is_j1939:
                                idx = np.argwhere(
                                    (j1939_msg_pgns == pgn_number) & (j9193_msg_sa == source_address)
                                ).ravel()
                            else:
                                idx = np.argwhere((bus_msg_ids == msg_id) & (bus_msg_ide == is_extended)).ravel()

                            payload = bus_data_bytes[idx]
                            t = bus_t[idx]

                            try:
                                extracted_signals = bus_logging_utils.extract_mux(
                                    payload,
                                    message,
                                    msg_id,
                                    bus,
                                    t,
                                    original_message_id=source_address if is_j1939 else None,
                                    ignore_value2text_conversion=ignore_value2text_conversion,
                                    is_j1939=is_j1939,
                                    is_extended=is_extended,
                                    raw=True,
                                )
                            except:
                                print(format_exc())
                                raise

                            for entry, signals in extracted_signals.items():
                                if len(next(iter(signals.values()))["samples"]) == 0:
                                    continue

                                if entry not in msg_map:
                                    sigs: list[Signal] = []

                                    index = len(out.groups)
                                    msg_map[entry] = index

                                    for name_, signal in signals.items():
                                        signal_name = f"{prefix}{signal['name']}"
                                        sig = Signal(
                                            samples=signal["samples"],
                                            timestamps=signal["t"],
                                            name=signal_name,
                                            comment=signal["comment"],
                                            unit=signal["unit"],
                                            invalidation_bits=signal["invalidation_bits"],
                                            display_names={
                                                f"CAN{bus}.{message.name}.{signal_name}": "bus",
                                                f"{message.name}.{signal_name}": "message",
                                            },
                                            raw=True,
                                            conversion=signal["conversion"],
                                        )

                                        sigs.append(sig)

                                    if is_j1939:
                                        if prefix:
                                            comment = f"{prefix}: CAN{bus} ID=0x{msg_id:X} {message} PGN=0x{pgn_number:X} SA=0x{source_address:X}"
                                        else:
                                            comment = f"CAN{bus} ID=0x{msg_id:X} {message} PGN=0x{pgn_number:X} SA=0x{source_address:X}"
                                        acq_name = f"SourceAddress = 0x{source_address}"
                                    else:
                                        if prefix:
                                            acq_name = (
                                                f"{prefix}: CAN{bus} message ID=0x{msg_id:X} EXT={bool(is_extended)}"
                                            )
                                            comment = f'{prefix}: CAN{bus} - message "{message}" 0x{msg_id:X} EXT={bool(is_extended)}'
                                        else:
                                            acq_name = f"CAN{bus} message ID=0x{msg_id:X} EXT={bool(is_extended)}"
                                            comment = (
                                                f"CAN{bus} - message {message} 0x{msg_id:X} EXT={bool(is_extended)}"
                                            )

                                    acq_source = Source(
                                        name=acq_name,
                                        path=f"CAN{int(bus)}.CAN_DataFrame.ID=0x{message.arbitration_id.id:X} EXT={bool(is_extended)}",
                                        comment=f"""\
<SIcomment>
    <TX>CAN{bus} data frame 0x{message.arbitration_id.id:X} EXT={bool(is_extended)} - {message.name}</TX>
    <bus name="CAN{int(bus)}"/>
    <common_properties>
        <e name="ChannelNo" type="integer">{int(bus)}</e>
    </common_properties>
</SIcomment>""",
                                        source_type=v4c.SOURCE_BUS,
                                        bus_type=v4c.BUS_TYPE_CAN,
                                    )

                                    for sig in sigs:
                                        sig.source = acq_source

                                    cg_nr = out.append(
                                        sigs,
                                        acq_name=acq_name,
                                        acq_source=acq_source,
                                        comment=comment,
                                        common_timebase=True,
                                    )

                                    out.groups[cg_nr].channel_group.flags = v4c.FLAG_CG_BUS_EVENT

                                    if is_j1939:
                                        max_flags.append([[False]])
                                        for ch_index, sig in enumerate(sigs, 1):
                                            max_flags[cg_nr].append(
                                                [
                                                    (
                                                        bool(np.all(sig.invalidation_bits))
                                                        if sig.invalidation_bits is not None
                                                        else False
                                                    )
                                                ]
                                            )
                                    else:
                                        max_flags.append([[False]] * (len(sigs) + 1))

                                else:
                                    index = msg_map[entry]

                                    signal_samples: list[tuple[NDArray[Any], NDArray[np.bool] | None]] = []

                                    for name_, signal in signals.items():
                                        signal_samples.append(
                                            (
                                                signal["samples"],
                                                signal["invalidation_bits"],
                                            )
                                        )

                                        t = signal["t"]

                                    if is_j1939:
                                        for ch_index, sig_sample in enumerate(signal_samples, 1):
                                            max_flags[index][ch_index].append(
                                                bool(np.all(sig_sample[1])) if sig_sample[1] is not None else False
                                            )

                                    signal_samples.insert(0, (t, None))

                                    out.extend(index, signal_samples)
                    self._set_temporary_master(None)

                cntr += 1
                if progress is not None:
                    if callable(progress):
                        progress(cntr, count)
                    else:
                        progress.signals.setValue.emit(cntr)

                        if progress.stop:
                            raise Terminated

            if current_not_found:
                not_found_ids[dbc_name] = list(current_not_found)

        unknown_id_set = {msg_id for msg_id, not_found in unknown_ids.items() if all(not_found)}

        self.last_call_info["CAN"] = {
            "dbc_files": dbc_files,
            "total_unique_ids": total_unique_ids,
            "unknown_id_count": len(unknown_id_set),
            "not_found_ids": not_found_ids,
            "found_ids": found_ids,
            "unknown_ids": unknown_id_set,
            "max_flags": max_flags,
        }

        if not out.groups:
            logger.warning(f'No CAN signals could be extracted from "{self.name}". The output file will be empty.')

        return out

    def _extract_lin_logging(
        self,
        output_file: "MDF4",
        dbc_files: Iterable[DbcFileType],
        ignore_value2text_conversion: bool = True,
        prefix: str = "",
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> "MDF4":
        out = output_file

        valid_dbc_files: list[tuple[CanMatrix, StrPath, int]] = []
        unique_name = UniqueDB()
        for dbc_name, bus_channel in dbc_files:
            if isinstance(dbc_name, CanMatrix):
                valid_dbc_files.append(
                    (
                        dbc_name,
                        unique_name.get_unique_name("UserProvidedCanMatrix"),
                        bus_channel,
                    )
                )
            else:
                dbc = load_can_database(Path(dbc_name))
                if dbc is None:
                    continue
                else:
                    valid_dbc_files.append((dbc, dbc_name, bus_channel))

        count = sum(
            1
            for group in self.groups
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT
            and group.channel_group.acq_source
            and group.channel_group.acq_source.bus_type == v4c.BUS_TYPE_LIN
        )
        count *= len(valid_dbc_files)

        if progress is not None:
            if callable(progress):
                progress(0, count)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(count)

                if progress.stop:
                    raise Terminated

        cntr = 0

        total_unique_ids: set[tuple[int, ...]] = set()
        found_ids: defaultdict[StrPath, set[tuple[tuple[int, bool, bool], str]]] = defaultdict(set)
        not_found_ids: defaultdict[StrPath, list[tuple[int, str]]] = defaultdict(list)
        unknown_ids: defaultdict[int, list[bool]] = defaultdict(list)

        for dbc, dbc_name, bus_channel in valid_dbc_files:
            messages = {message.arbitration_id.id: message for message in dbc}

            current_not_found_ids = {(msg_id, message.name) for msg_id, message in messages.items()}

            msg_map = {}

            for i, group in enumerate(self.groups):
                if (
                    not group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT
                    or (group.channel_group.acq_source and group.channel_group.acq_source.bus_type != v4c.BUS_TYPE_LIN)
                    or not "LIN_Frame" in [ch.name for ch in group.channels]
                ):
                    continue

                self._prepare_record(group)
                data = self._load_data(group, optimize_read=False)

                for fragment in data:
                    self._set_temporary_master(None)
                    self._set_temporary_master(self.get_master(i, data=fragment, one_piece=True))

                    msg_ids = self.get("LIN_Frame.ID", group=i, data=fragment).astype("<u4") & 0x1FFFFFFF

                    original_ids = msg_ids.samples.copy()

                    data_bytes = self.get(
                        "LIN_Frame.DataBytes",
                        group=i,
                        data=fragment,
                    ).samples

                    try:
                        bus_ids = self.get(
                            "LIN_Frame.BusChannel",
                            group=i,
                            data=fragment,
                        ).samples.astype("<u1")
                    except:
                        bus_ids = np.ones(len(original_ids), dtype="u1")

                    bus_t = msg_ids.timestamps
                    bus_msg_ids = msg_ids.samples
                    bus_data_bytes = data_bytes
                    original_msg_ids = original_ids

                    unique_ids = np.unique(np.rec.fromarrays([bus_msg_ids, bus_msg_ids]))

                    total_unique_ids = total_unique_ids | {tuple(int(e) for e in f) for f in unique_ids}

                    buses = np.unique(bus_ids)

                    for bus in buses:
                        if bus_channel and bus != bus_channel:
                            continue

                        for msg_id_record in sorted(unique_ids.tolist()):
                            msg_id = int(msg_id_record[0])
                            original_msg_id = int(msg_id_record[1])
                            message = messages.get(msg_id, None)
                            if message is None:
                                unknown_ids[msg_id].append(True)
                                continue

                            found_ids[dbc_name].add(((msg_id, False, False), message.name))
                            try:
                                current_not_found_ids.remove((msg_id, message.name))
                            except KeyError:
                                pass

                            unknown_ids[msg_id].append(False)

                            idx = np.argwhere(bus_msg_ids == msg_id).ravel()
                            payload = bus_data_bytes[idx]
                            t = bus_t[idx]

                            extracted_signals = bus_logging_utils.extract_mux(
                                payload,
                                message,
                                msg_id,
                                bus,
                                t,
                                original_message_id=None,
                                ignore_value2text_conversion=ignore_value2text_conversion,
                                raw=True,
                            )

                            for entry, signals in extracted_signals.items():
                                if len(next(iter(signals.values()))["samples"]) == 0:
                                    continue
                                if entry not in msg_map:
                                    sigs: list[Signal] = []

                                    index = len(out.groups)
                                    msg_map[entry] = index

                                    for name_, signal in signals.items():
                                        signal_name = f"{prefix}{signal['name']}"
                                        sig = Signal(
                                            samples=signal["samples"],
                                            timestamps=signal["t"],
                                            name=signal_name,
                                            comment=signal["comment"],
                                            unit=signal["unit"],
                                            invalidation_bits=signal["invalidation_bits"],
                                            display_names={
                                                f"LIN{bus}.{message.name}.{signal_name}": "bus",
                                                f"{message.name}.{signal_name}": "message",
                                            },
                                            raw=True,
                                            conversion=signal["conversion"],
                                        )

                                        sigs.append(sig)

                                    if prefix:
                                        acq_name = f"{prefix}: from LIN{bus} message ID=0x{msg_id:X}"
                                    else:
                                        acq_name = f"from LIN{bus} message ID=0x{msg_id:X}"

                                    acq_source = Source(
                                        name=acq_name,
                                        path=f"LIN{int(bus)}.LIN_Frame.ID=0x{message.arbitration_id.id:X}",
                                        comment=f"""\
<SIcomment>
    <TX>LIN{bus} data frame 0x{message.arbitration_id.id:X} - {message.name}</TX>
    <bus name="LIN{int(bus)}"/>
    <common_properties>
        <e name="ChannelNo" type="integer">{int(bus)}</e>
    </common_properties>
</SIcomment>""",
                                        source_type=v4c.SOURCE_BUS,
                                        bus_type=v4c.BUS_TYPE_LIN,
                                    )

                                    for sig in sigs:
                                        sig.source = acq_source

                                    cg_nr = out.append(
                                        sigs,
                                        acq_name=acq_name,
                                        acq_source=acq_source,
                                        comment=f"from LIN{bus} - message {message} 0x{msg_id:X}",
                                        common_timebase=True,
                                    )

                                    out.groups[cg_nr].channel_group.flags = v4c.FLAG_CG_BUS_EVENT

                                else:
                                    index = msg_map[entry]

                                    signal_samples: list[tuple[NDArray[Any], NDArray[np.bool] | None]] = []

                                    for name_, signal in signals.items():
                                        signal_samples.append(
                                            (
                                                signal["samples"],
                                                signal["invalidation_bits"],
                                            )
                                        )

                                        t = signal["t"]

                                    signal_samples.insert(0, (t, None))

                                    out.extend(index, signal_samples)
                    self._set_temporary_master(None)

                cntr += 1
                if progress is not None:
                    if callable(progress):
                        progress(cntr, count)
                    else:
                        progress.signals.setValue.emit(cntr)

                        if progress.stop:
                            raise Terminated

            if current_not_found_ids:
                not_found_ids[dbc_name] = list(current_not_found_ids)

        unknown_id_set = {msg_id for msg_id, not_found in unknown_ids.items() if all(not_found)}

        self.last_call_info["LIN"] = {
            "dbc_files": dbc_files,
            "total_unique_ids": total_unique_ids,
            "unknown_id_count": len(unknown_id_set),
            "not_found_ids": not_found_ids,
            "found_ids": found_ids,
            "unknown_ids": unknown_id_set,
        }

        if not out.groups:
            logger.warning(f'No LIN signals could be extracted from "{self.name}". The output file will be empty.')

        return out

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
        dst: FileLike | StrPath,
        overwrite: bool = False,
        compression: CompressionType = v4c.CompressionAlgorithm.NO_COMPRESSION,
        progress: Any | None = None,
        add_history_block: bool = True,
    ) -> Path:
        """Save `MDF` to `dst`. If `overwrite` is True, then the destination
        file is overwritten, otherwise the file name is appended with '.<cntr>',
        where '<cntr>' is the first counter that produces a new file name that
        does not already exist in the filesystem.

        Parameters
        ----------
        dst : str | path-like | file-like
            Destination file name.
        overwrite : bool, default False
            Overwrite flag.
        compression : int, default 0
            Use compressed data blocks; valid since MDF version 4.10.

            * 0 - no compression
            * 1 - deflate (slower, but produces smaller files)
            * 2 - transposition + deflate (slowest, but produces
              the smallest files)

        add_history_block : bool, default True
            Option to add file history block.

        Returns
        -------
        output_file : pathlib.Path
            Path to saved file.
        """

        if is_file_like(dst):
            dst_ = dst
            file_like = True
            if hasattr(dst, "name"):
                dst = Path(dst.name)
            else:
                dst = Path("__file_like.mf4")
            dst_.seek(0)
            suffix = ".mf4"
        else:
            file_like = False
            suffix = Path(dst).suffix.lower()

            dst = Path(dst).with_suffix(".mf4")

            destination_dir = dst.parent
            destination_dir.mkdir(parents=True, exist_ok=True)

            if overwrite is False:
                if dst.is_file():
                    cntr = 0
                    while True:
                        name = dst.with_suffix(f".{cntr}.mf4")
                        if not name.exists():
                            break
                        else:
                            cntr += 1
                    message = (
                        f'Destination file "{dst}" already exists and "overwrite" is False. Saving MDF file as "{name}"'
                    )
                    logger.warning(message)
                    dst = name

            if dst == self.name:
                destination = dst.with_suffix(".savetemp")
            else:
                destination = dst

            dst_ = open(destination, "wb+")

        if not self.file_history:
            comment = "created"
        else:
            comment = "updated"

        if add_history_block:
            fh = FileHistory()
            fh.comment = f"""<FHcomment>
    <TX>{comment}</TX>
    <tool_id>{tool.__tool__}</tool_id>
    <tool_vendor>{tool.__vendor__}</tool_vendor>
    <tool_version>{tool.__version__}</tool_version>
</FHcomment>"""

            self.file_history.append(fh)

        cg_map = {}

        try:
            defined_texts: dict[bytes | str, int] = {"": 0, b"": 0}
            cc_map: dict[bytes | int, int] = {}
            si_map: dict[bytes | int, int] = {}

            groups_nr = len(self.groups)

            write = dst_.write
            tell = dst_.tell
            seek = dst_.seek

            blocks: list[bytes | SupportsBytes] = []

            write(bytes(self.identification))

            self.header.to_blocks(dst_.tell(), blocks)
            for block in blocks:
                write(bytes(block))

            original_data_addresses = []

            if self.version < "4.30":
                if compression > v4c.CompressionAlgorithm.TRANSPOSED_DEFLATE:
                    zip_type = v4c.FLAG_DZ_TRANSPOSED_DEFLATE
                elif compression == v4c.CompressionAlgorithm.DEFLATE:
                    zip_type = v4c.FLAG_DZ_DEFLATE
                else:
                    zip_type = v4c.FLAG_DZ_TRANSPOSED_DEFLATE
            else:
                match compression:
                    case v4c.CompressionAlgorithm.DEFLATE:
                        zip_type = v4c.FLAG_DZ_DEFLATE
                    case v4c.CompressionAlgorithm.TRANSPOSED_DEFLATE:
                        zip_type = v4c.FLAG_DZ_TRANSPOSED_DEFLATE
                    case v4c.CompressionAlgorithm.LZ4:
                        zip_type = v4c.FLAG_DZ_LZ4
                    case v4c.CompressionAlgorithm.TRANSPOSED_LZ4:
                        zip_type = v4c.FLAG_DZ_TRANSPOSED_LZ4
                    case v4c.CompressionAlgorithm.ZSTD:
                        zip_type = v4c.FLAG_DZ_ZSTD
                    case v4c.CompressionAlgorithm.TRANSPOSED_ZSTD:
                        zip_type = v4c.FLAG_DZ_TRANSPOSED_ZSTD

            # write DataBlocks first
            for gp_nr, gp in enumerate(self.groups):
                original_data_addresses.append(gp.data_group.data_block_addr)

                if gp.channel_group.flags & v4c.FLAG_CG_VLSD:
                    continue

                address = tell()

                total_size = (
                    gp.channel_group.samples_byte_nr + gp.channel_group.invalidation_bytes_nr
                ) * gp.channel_group.cycles_nr

                if total_size:
                    if self._write_fragment_size:
                        samples_size = gp.channel_group.samples_byte_nr + gp.channel_group.invalidation_bytes_nr
                        if samples_size:
                            split_size = self._write_fragment_size // samples_size
                            split_size *= samples_size
                            if split_size == 0:
                                split_size = samples_size
                            chunks = float(total_size) / split_size
                            chunks = ceil(chunks)

                            self._read_fragment_size = split_size
                        else:
                            chunks = 1
                    else:
                        chunks = 1

                    data = self._load_data(gp)

                    if chunks == 1:
                        fragment = next(data)
                        data_, inval_ = fragment.data, fragment.invalidation_data
                        if self.version >= "4.20" and gp.uses_ld:
                            if compression:
                                if gp.channel_group.samples_byte_nr > 1:
                                    current_zip_type = zip_type
                                    if compression == 1:
                                        param = 0
                                    else:
                                        param = gp.channel_group.samples_byte_nr
                                else:
                                    current_zip_type = v4c.FLAG_DZ_DEFLATE
                                    param = 0

                                dz_kwargs: DataZippedBlockKwargs = {
                                    "data": data_,
                                    "zip_type": current_zip_type,
                                    "param": param,
                                    "original_type": b"DV",
                                }
                                data_block: DataZippedBlock | DataBlock = DataZippedBlock(**dz_kwargs)
                            else:
                                data_block = DataBlock(data=data_, type="DV")
                            write(bytes(data_block))
                            data_address = address

                            align = data_block.block_len % 8
                            if align:
                                write(b"\0" * (8 - align))

                            if inval_ is not None:
                                inval_address = address = tell()
                                if compression:
                                    if compression == 1:
                                        param = 0
                                    else:
                                        param = gp.channel_group.invalidation_bytes_nr
                                    dz_kwargs = {
                                        "data": inval_,
                                        "zip_type": zip_type,
                                        "param": param,
                                        "original_type": b"DI",
                                    }
                                    inval_block: DataZippedBlock | DataBlock = DataZippedBlock(**dz_kwargs)
                                else:
                                    inval_block = DataBlock(data=inval_, type="DI")
                                write(bytes(inval_block))

                                align = inval_block.block_len % 8
                                if align:
                                    write(b"\0" * (8 - align))

                            address = tell()

                            ld_kwargs: ListDataKwargs = {  # type: ignore[typeddict-unknown-key]
                                "flags": v4c.FLAG_LD_EQUAL_LENGHT,
                                "data_block_nr": 1,
                                "data_block_len": gp.channel_group.cycles_nr,
                                "data_block_addr_0": data_address,
                            }
                            if inval_:
                                ld_kwargs["flags"] |= v4c.FLAG_LD_INVALIDATION_PRESENT
                                ld_kwargs["invalidation_bits_addr_0"] = inval_address  # type: ignore[typeddict-unknown-key]
                            ld_block = ListData(**ld_kwargs)
                            write(bytes(ld_block))

                            align = ld_block.block_len % 8
                            if align:
                                write(b"\0" * (8 - align))

                            if gp.channel_group.cycles_nr:
                                gp.data_group.data_block_addr = address
                            else:
                                gp.data_group.data_block_addr = 0

                        else:
                            if compression and self.version >= "4.10":
                                if compression == 1:
                                    param = 0
                                else:
                                    param = gp.channel_group.samples_byte_nr + gp.channel_group.invalidation_bytes_nr
                                dz_kwargs = {
                                    "data": data_,
                                    "zip_type": zip_type,
                                    "param": param,
                                }
                                data_block = DataZippedBlock(**dz_kwargs)
                            else:
                                data_block = DataBlock(data=data_)
                            write(bytes(data_block))

                            align = data_block.block_len % 8
                            if align:
                                write(b"\0" * (8 - align))

                            if gp.channel_group.cycles_nr:
                                gp.data_group.data_block_addr = address
                            else:
                                gp.data_group.data_block_addr = 0
                    else:
                        if self.version >= "4.20" and gp.uses_ld:
                            dv_addr = []
                            di_addr = []
                            block_size = 0
                            for i, fragment in enumerate(data):
                                data_, inval_ = fragment.data, fragment.invalidation_data
                                if i == 0:
                                    block_size = len(data_)
                                if compression:
                                    if compression == 1:
                                        param = 0
                                    else:
                                        param = gp.channel_group.samples_byte_nr
                                    dz_kwargs = {
                                        "data": data_,
                                        "zip_type": zip_type,
                                        "param": param,
                                        "original_type": b"DV",
                                    }
                                    data_block = DataZippedBlock(**dz_kwargs)
                                else:
                                    data_block = DataBlock(data=data_, type="DV")
                                dv_addr.append(tell())
                                write(bytes(data_block))

                                align = data_block.block_len % 8
                                if align:
                                    write(b"\0" * (8 - align))

                                if inval_ is not None:
                                    if compression:
                                        if compression == 1:
                                            param = 0
                                        else:
                                            param = gp.channel_group.invalidation_bytes_nr
                                        dz_kwargs = {
                                            "data": inval_,
                                            "zip_type": zip_type,
                                            "param": param,
                                            "original_type": b"DI",
                                        }
                                        inval_block = DataZippedBlock(**dz_kwargs)
                                    else:
                                        inval_block = DataBlock(data=inval_, type="DI")
                                    di_addr.append(tell())
                                    write(bytes(inval_block))

                                    align = inval_block.block_len % 8
                                    if align:
                                        write(b"\0" * (8 - align))

                            address = tell()

                            ld_kwargs = {
                                "flags": v4c.FLAG_LD_EQUAL_LENGHT,
                                "data_block_nr": len(dv_addr),
                                "data_block_len": block_size // gp.channel_group.samples_byte_nr,
                            }
                            for i, addr in enumerate(dv_addr):
                                ld_kwargs[f"data_block_addr_{i}"] = addr  # type: ignore[literal-required]

                            if di_addr:
                                ld_kwargs["flags"] |= v4c.FLAG_LD_INVALIDATION_PRESENT
                                for i, addr in enumerate(di_addr):
                                    ld_kwargs[f"invalidation_bits_addr_{i}"] = addr  # type: ignore[literal-required]

                            ld_block = ListData(**ld_kwargs)
                            write(bytes(ld_block))

                            align = ld_block.block_len % 8
                            if align:
                                write(b"\0" * (8 - align))

                            if gp.channel_group.cycles_nr:
                                gp.data_group.data_block_addr = address
                            else:
                                gp.data_group.data_block_addr = 0

                        else:
                            hl_kwargs: HeaderListKwargs = {
                                "flags": v4c.FLAG_DL_EQUAL_LENGHT,
                                "zip_type": zip_type,
                            }
                            hl_block = HeaderList(**hl_kwargs)

                            dl_kwargs: DataListKwargs = {
                                "flags": v4c.FLAG_DL_EQUAL_LENGHT,
                                "links_nr": chunks + 1,
                                "data_block_nr": chunks,
                                "data_block_len": split_size,
                            }
                            dl_block = DataList(**dl_kwargs)

                            for i, fragment in enumerate(data):
                                data_ = fragment.data

                                if compression and self.version >= "4.10":
                                    if compression == 1:
                                        zip_type = v4c.FLAG_DZ_DEFLATE
                                    else:
                                        zip_type = v4c.FLAG_DZ_TRANSPOSED_DEFLATE
                                    if compression == 1:
                                        param = 0
                                    else:
                                        param = (
                                            gp.channel_group.samples_byte_nr + gp.channel_group.invalidation_bytes_nr
                                        )
                                    dz_kwargs = {
                                        "data": data_,
                                        "zip_type": zip_type,
                                        "param": param,
                                    }
                                    block = DataZippedBlock(**dz_kwargs)
                                else:
                                    block = DataBlock(data=data_)
                                address = tell()
                                block.address = address

                                write(bytes(block))

                                align = block.block_len % 8
                                if align:
                                    write(b"\0" * (8 - align))
                                dl_block[f"data_block_addr{i}"] = address

                            address = tell()
                            dl_block.address = address
                            write(bytes(dl_block))

                            if compression and self.version != "4.00":
                                hl_block.first_dl_addr = address
                                address = tell()
                                hl_block.address = address
                                write(bytes(hl_block))

                            gp.data_group.data_block_addr = address
                else:
                    gp.data_group.data_block_addr = 0

                if progress is not None:
                    progress.signals.setValue.emit(int(50 * (gp_nr + 1) / groups_nr))

                    if progress.stop:
                        dst_.close()
                        self.close()

                        raise Terminated

            address = tell()

            blocks = []

            # file history blocks
            for fh in self.file_history:
                address = fh.to_blocks(address, blocks, defined_texts)

            for i, fh in enumerate(self.file_history[:-1]):
                fh.next_fh_addr = self.file_history[i + 1].address
            self.file_history[-1].next_fh_addr = 0

            # data groups
            gp_rec_ids = []
            valid_data_groups = []
            for gp in self.groups:
                if gp.channel_group.flags & v4c.FLAG_CG_VLSD:
                    continue

                valid_data_groups.append(gp.data_group)
                gp_rec_ids.append(gp.data_group.record_id_len)

                address = gp.data_group.to_blocks(address, blocks, defined_texts)

            if valid_data_groups:
                for i, dg in enumerate(valid_data_groups[:-1]):
                    addr_ = valid_data_groups[i + 1].address
                    dg.next_dg_addr = addr_
                valid_data_groups[-1].next_dg_addr = 0

            # go through each data group and append the rest of the blocks
            for i, gp in enumerate(self.groups):
                channels = gp.channels

                for j, channel in enumerate(channels):
                    if channel.attachment is not None:
                        channel.attachment_addr = self.attachments[channel.attachment].address
                    elif channel.attachment_nr:
                        channel.attachment_addr = 0

                    address = channel.to_blocks(address, blocks, defined_texts, cc_map, si_map)

                    if channel.channel_type == v4c.CHANNEL_TYPE_SYNC:
                        if channel.attachment is not None:
                            channel.data_block_addr = self.attachments[channel.attachment].address
                    else:
                        sdata = self._load_signal_data(group=gp, index=j)
                        if sdata:
                            split_size = self._write_fragment_size
                            if self._write_fragment_size:
                                chunks = float(len(sdata)) / split_size
                                chunks = ceil(chunks)
                            else:
                                chunks = 1

                            if chunks == 1:
                                if compression and self.version > "4.00":
                                    signal_data: DataZippedBlock | DataBlock = DataZippedBlock(
                                        data=sdata,
                                        zip_type=v4c.FLAG_DZ_DEFLATE,
                                        original_type=b"SD",
                                    )
                                    signal_data.address = address
                                    address += signal_data.block_len
                                    blocks.append(signal_data)
                                    align = signal_data.block_len % 8
                                    if align:
                                        blocks.append(b"\0" * (8 - align))
                                        address += 8 - align
                                else:
                                    signal_data = DataBlock(data=sdata, type="SD")
                                    signal_data.address = address
                                    address += signal_data.block_len
                                    blocks.append(signal_data)
                                    align = signal_data.block_len % 8
                                    if align:
                                        blocks.append(b"\0" * (8 - align))
                                        address += 8 - align

                                channel.data_block_addr = signal_data.address
                            else:
                                dl_kwargs = {
                                    "flags": v4c.FLAG_DL_EQUAL_LENGHT,
                                    "links_nr": chunks + 1,
                                    "data_block_nr": chunks,
                                    "data_block_len": self._write_fragment_size,
                                }
                                dl_block = DataList(**dl_kwargs)

                                for k in range(chunks):
                                    data_ = sdata[k * split_size : (k + 1) * split_size]
                                    if compression and self.version > "4.00":
                                        zip_type = v4c.FLAG_DZ_DEFLATE
                                        param = 0

                                        dz_kwargs = {
                                            "data": data_,
                                            "zip_type": zip_type,
                                            "param": param,
                                            "original_type": b"SD",
                                        }
                                        block = DataZippedBlock(**dz_kwargs)
                                    else:
                                        block = DataBlock(data=data_, type="SD")
                                    blocks.append(block)
                                    block.address = address
                                    address += block.block_len

                                    align = block.block_len % 8
                                    if align:
                                        blocks.append(b"\0" * (8 - align))
                                        address += 8 - align
                                    dl_block[f"data_block_addr{k}"] = block.address

                                dl_block.address = address
                                blocks.append(dl_block)

                                address += dl_block.block_len

                                if compression and self.version > "4.00":
                                    hl_kwargs = {
                                        "flags": v4c.FLAG_DL_EQUAL_LENGHT,
                                        "zip_type": v4c.FLAG_DZ_DEFLATE,
                                        "first_dl_addr": dl_block.address,
                                    }
                                    hl_block = HeaderList(**hl_kwargs)
                                    hl_block.address = address
                                    address += hl_block.block_len

                                    blocks.append(hl_block)

                                    channel.data_block_addr = hl_block.address
                                else:
                                    channel.data_block_addr = dl_block.address

                        else:
                            channel.data_block_addr = 0

                    dep_list = gp.channel_dependencies[j]
                    if dep_list:
                        if all(isinstance(dep, ChannelArrayBlock) for dep in dep_list):
                            dep_list = typing.cast(list[ChannelArrayBlock], dep_list)
                            for dep in dep_list:
                                dep.address = address
                                address += dep.block_len
                                blocks.append(dep)
                            for k, dep in enumerate(dep_list[:-1]):
                                dep.composition_addr = dep_list[k + 1].address
                            dep_list[-1].composition_addr = 0

                            channel.component_addr = dep_list[0].address

                        else:
                            dep_list = typing.cast(list[tuple[int, int]], dep_list)
                            index = dep_list[0][1]
                            addr_ = gp.channels[index].address

                group_channels = gp.channels
                if group_channels:
                    for j, channel in enumerate(group_channels[:-1]):
                        channel.next_ch_addr = group_channels[j + 1].address
                    group_channels[-1].next_ch_addr = 0

                # channel dependencies
                j = len(channels) - 1
                while j >= 0:
                    dep_list = gp.channel_dependencies[j]
                    if dep_list and all(isinstance(dep, tuple) for dep in dep_list):
                        dep_list = typing.cast(list[tuple[int, int]], dep_list)
                        index = dep_list[0][1]
                        channels[j].component_addr = channels[index].address
                        index = dep_list[-1][1]
                        channels[j].next_ch_addr = channels[index].next_ch_addr
                        channels[index].next_ch_addr = 0

                        for _, ch_nr in dep_list:
                            channels[ch_nr].source_addr = 0
                    j -= 1

                # channel group
                if gp.channel_group.flags & v4c.FLAG_CG_VLSD:
                    continue

                gp.channel_group.first_sample_reduction_addr = 0

                if channels:
                    gp.channel_group.first_ch_addr = gp.channels[0].address
                else:
                    gp.channel_group.first_ch_addr = 0
                gp.channel_group.next_cg_addr = 0

                address = gp.channel_group.to_blocks(address, blocks, defined_texts, si_map)
                gp.data_group.first_cg_addr = gp.channel_group.address

                cg_map[i] = gp.channel_group.address

                if progress is not None:
                    progress.signals.setValue.emit(int(50 * (i + 1) / groups_nr) + 25)

                    if progress.stop:
                        dst_.close()
                        self.close()

                        raise Terminated

            for gp in self.groups:
                for dep_list in gp.channel_dependencies:
                    if dep_list:
                        if all(isinstance(dep, ChannelArrayBlock) for dep in dep_list):
                            dep_list = typing.cast(list[ChannelArrayBlock], dep_list)
                            for dep in dep_list:
                                for i, (gp_nr, ch_nr) in enumerate(filter(None, dep.dynamic_size_channels)):
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep[f"dynamic_size_{i}_dg_addr"] = grp.data_group.address
                                    dep[f"dynamic_size_{i}_cg_addr"] = grp.channel_group.address
                                    dep[f"dynamic_size_{i}_ch_addr"] = ch.address

                                for i, (gp_nr, ch_nr) in enumerate(filter(None, dep.input_quantity_channels)):
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep[f"input_quantity_{i}_dg_addr"] = grp.data_group.address
                                    dep[f"input_quantity_{i}_cg_addr"] = grp.channel_group.address
                                    dep[f"input_quantity_{i}_ch_addr"] = ch.address

                                for i, conversion in enumerate(dep.axis_conversions):
                                    if conversion:
                                        address = conversion.to_blocks(address, blocks, defined_texts, cc_map)
                                        dep[f"axis_conversion_{i}"] = conversion.address
                                    else:
                                        dep[f"axis_conversion_{i}"] = 0

                                if dep.output_quantity_channel:
                                    gp_nr, ch_nr = dep.output_quantity_channel
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep.output_quantity_dg_addr = grp.data_group.address
                                    dep.output_quantity_cg_addr = grp.channel_group.address
                                    dep.output_quantity_ch_addr = ch.address

                                if dep.comparison_quantity_channel:
                                    gp_nr, ch_nr = dep.comparison_quantity_channel
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep.comparison_quantity_dg_addr = grp.data_group.address
                                    dep.comparison_quantity_cg_addr = grp.channel_group.address
                                    dep.comparison_quantity_ch_addr = ch.address

                                for i, (gp_nr, ch_nr) in enumerate(filter(None, dep.axis_channels)):
                                    grp = self.groups[gp_nr]
                                    ch = grp.channels[ch_nr]
                                    dep[f"scale_axis_{i}_dg_addr"] = grp.data_group.address
                                    dep[f"scale_axis_{i}_cg_addr"] = grp.channel_group.address
                                    dep[f"scale_axis_{i}_ch_addr"] = ch.address

            position = tell()

            for gp in self.groups:
                gp.data_group.record_id_len = 0

                cg_master_index = gp.channel_group.cg_master_index
                if cg_master_index is not None:
                    gp.channel_group.cg_master_addr = cg_map[cg_master_index]
                    seek(gp.channel_group.address)
                    write(bytes(gp.channel_group))

            seek(position)

            ev_map = []

            if self.events:
                for event in self.events:
                    for i, ref in enumerate(event.scopes):
                        if isinstance(ref, tuple):
                            dg_cntr, ch_cntr = ref
                            event[f"scope_{i}_addr"] = self.groups[dg_cntr].channels[ch_cntr].address
                        else:
                            dg_cntr = ref
                            event[f"scope_{i}_addr"] = self.groups[dg_cntr].channel_group.address

                    blocks.append(event)
                    ev_map.append(address)
                    event.address = address
                    address += event.block_len

                    if event.name:
                        tx_block = TextBlock(text=event.name)
                        tx_block.address = address
                        blocks.append(tx_block)
                        address += tx_block.block_len
                        event.name_addr = tx_block.address
                    else:
                        event.name_addr = 0

                    if event.comment:
                        meta = event.comment.startswith("<EVcomment")
                        tx_block = TextBlock(text=event.comment, meta=meta)
                        tx_block.address = address
                        blocks.append(tx_block)
                        address += tx_block.block_len
                        event.comment_addr = tx_block.address
                    else:
                        event.comment_addr = 0

                    if event.parent is not None:
                        event.parent_ev_addr = ev_map[event.parent]
                    if event.range_start is not None:
                        event.range_start_ev_addr = ev_map[event.range_start]

                for i in range(len(self.events) - 1):
                    self.events[i].next_ev_addr = self.events[i + 1].address
                self.events[-1].next_ev_addr = 0

                self.header.first_event_addr = self.events[0].address

            if progress is not None and progress.stop:
                dst_.close()
                self.close()
                raise Terminated

            # attachments
            at_map: dict[int, int] = {}
            if self.attachments:
                # put the attachment texts before the attachments
                for at_block in self.attachments:
                    for text in (at_block.file_name, at_block.mime, at_block.comment):
                        if text not in defined_texts:
                            tx_block = TextBlock(text=str(text))
                            defined_texts[text] = address
                            tx_block.address = address
                            address += tx_block.block_len
                            blocks.append(tx_block)

                for at_block in self.attachments:
                    address = at_block.to_blocks(address, blocks, defined_texts)

                for i in range(len(self.attachments) - 1):
                    at_block = self.attachments[i]
                    at_block.next_at_addr = self.attachments[i + 1].address
                self.attachments[-1].next_at_addr = 0

                if self.events:
                    for event in self.events:
                        for i in range(event.attachment_nr):
                            key = f"attachment_{i}_addr"
                            addr = typing.cast(int, event[key])
                            event[key] = at_map[addr]

                for i, gp in enumerate(self.groups):
                    for j, channel in enumerate(gp.channels):
                        if channel.attachment is not None:
                            channel.attachment_addr = self.attachments[channel.attachment].address
                        elif channel.attachment_nr:
                            channel.attachment_addr = 0

                        if channel.channel_type == v4c.CHANNEL_TYPE_SYNC and channel.attachment is not None:
                            channel.data_block_addr = self.attachments[channel.attachment].address

            if progress is not None:
                blocks_nr = len(blocks)
                threshold = blocks_nr / 25
                count = 1
                for i, block in enumerate(blocks):
                    write(bytes(block))
                    if i >= threshold:
                        progress.signals.setValue.emit(75 + count)

                        count += 1
                        threshold += blocks_nr / 25
            else:
                for block in blocks:
                    write(bytes(block))

            for gp, rec_id in zip(self.groups, gp_rec_ids, strict=False):
                gp.data_group.record_id_len = rec_id

            if valid_data_groups:
                addr_ = valid_data_groups[0].address
                self.header.first_dg_addr = addr_
            else:
                self.header.first_dg_addr = 0
            self.header.file_history_addr = self.file_history[0].address
            if self.attachments:
                first_attachment = self.attachments[0]
                addr_ = first_attachment.address
                self.header.first_attachment_addr = addr_
            else:
                self.header.first_attachment_addr = 0

            seek(v4c.IDENTIFICATION_BLOCK_SIZE)
            write(bytes(self.header))

            for orig_addr, gp in zip(original_data_addresses, self.groups, strict=False):
                gp.data_group.data_block_addr = orig_addr

            at_map = {value: key for key, value in at_map.items()}

            for event in self.events:
                for i in range(event.attachment_nr):
                    key = f"attachment_{i}_addr"
                    addr = typing.cast(int, event[key])
                    event[key] = at_map[addr]

        except:
            if not file_like:
                dst_.close()
            raise
        else:
            if not file_like:
                dst_.close()

        if suffix in (".zip", ".mf4z"):
            output_fname = dst.with_suffix(suffix)
            try:
                zipped_mf4 = ZipFile(output_fname, "w", compression=ZIP_DEFLATED)
                zipped_mf4.write(
                    str(dst),
                    dst.name,
                    compresslevel=1,
                )
                zipped_mf4.close()
                os.remove(destination)
                dst = output_fname
            except:
                pass

        if dst == self.name:
            self.close()
            try:
                Path.unlink(self.name)
                Path.rename(destination, self.name)
            except:
                pass

            self.groups.clear()
            del self.header
            del self.identification
            self.file_history.clear()
            self.channels_db.clear()
            self.masters_db.clear()
            self.attachments.clear()

            self._ch_map.clear()

            self._tempfile = NamedTemporaryFile(dir=self.temporary_folder)
            self._file = open(self.name, "rb")
            self._read(self._file)

        return dst

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

        return self.groups[gp_nr].channels[ch_nr].name

    def get_channel_metadata(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
    ) -> Channel:
        gp_nr, ch_nr = self._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]

        channel = grp.channels[ch_nr]

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

        conversion = channel.conversion

        unit = (conversion and conversion.unit) or channel.unit or ""

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

        return extract_xml_comment(channel.comment)

    def _finalize(self, stream: FileLike | mmap.mmap) -> None:
        """
        Attempt finalization of the file.
        :return:    None
        """

        flags = self.identification.unfinalized_standard_flags

        blocks, block_groups, addresses = all_blocks_addresses(stream)

        stream.seek(0, 2)
        limit = stream.tell()
        mapped = self._mapped

        if flags & v4c.FLAG_UNFIN_UPDATE_LAST_DL:
            for dg_addr in block_groups[b"##DG"]:
                group = DataGroup(address=dg_addr, stream=stream, mapped=mapped)
                data_addr = group.data_block_addr
                if not data_addr:
                    continue

                stream.seek(data_addr)
                blk_id = stream.read(4)
                if blk_id == b"##DT":
                    continue
                elif blk_id in (b"##DL", b"##HL"):
                    if blk_id == b"##HL":
                        hl = HeaderList(address=data_addr, stream=stream, mapped=mapped)
                        data_addr = hl.first_dl_addr

                    while True:
                        dl = DataList(address=data_addr, stream=stream, mapped=mapped)
                        if not dl.next_dl_addr:
                            break

                    kwargs: DataListKwargs = {}

                    count = dl.links_nr - 1
                    valid_count = 0
                    for i in range(count):
                        dt_addr = dl[f"data_block_addr{i}"]
                        if dt_addr:
                            valid_count += 1
                            kwargs[f"data_block_addr{i}"] = dt_addr  # type: ignore[literal-required]
                        else:
                            break

                    starting_address = dl.address
                    next_block_position = bisect.bisect_right(addresses, starting_address)
                    # search for data blocks after the DLBLOCK
                    for j in range(i, count):
                        if next_block_position >= len(addresses):
                            break

                        next_block_address = addresses[next_block_position]
                        next_block_type = blocks[next_block_address]

                        if next_block_type not in {b"##DZ", b"##DT", b"##DV", b"##DI"}:
                            break
                        else:
                            stream.seek(next_block_address + v4c.DZ_INFO_COMMON_OFFSET)

                            if next_block_type == b"##DZ":
                                (
                                    zip_type,
                                    param,
                                    original_size,
                                    zip_size,
                                ) = v4c.DZ_COMMON_INFO_uf(stream.read(v4c.DZ_COMMON_INFO_SIZE))

                                exceeded = limit - (next_block_address + v4c.DZ_COMMON_SIZE + zip_size) < 0

                            else:
                                id_string, block_len = COMMON_SHORT_uf(stream.read(v4c.COMMON_SIZE))
                                original_size = block_len - 24

                                exceeded = limit - (next_block_address + block_len) < 0

                            # update the data block size in case all links were NULL before
                            if i == 0 and (dl.flags & v4c.FLAG_DL_EQUAL_LENGHT):
                                kwargs["data_block_len"] = original_size

                            # check if the file limit is exceeded
                            if exceeded:
                                break
                            else:
                                next_block_position += 1
                                valid_count += 1
                                kwargs[f"data_block_addr{j}"] = next_block_address  # type: ignore[literal-required]

                    kwargs["links_nr"] = valid_count + 1
                    kwargs["flags"] = dl.flags
                    if dl.flags & v4c.FLAG_DL_EQUAL_LENGHT:
                        kwargs["data_block_len"] = dl.data_block_len
                    else:
                        for i in range(valid_count):
                            kwargs[f"offset_{i}"] = dl[f"offset_{i}"]  # type: ignore[literal-required]

                    stream.seek(data_addr)
                    stream.write(bytes(DataList(**kwargs)))

            self.identification.unfinalized_standard_flags -= v4c.FLAG_UNFIN_UPDATE_LAST_DL

        if flags & v4c.FLAG_UNFIN_UPDATE_LAST_DT_LENGTH:
            try:
                for dg_addr in block_groups[b"##DG"]:
                    group = DataGroup(address=dg_addr, stream=stream, mapped=mapped)
                    data_addr = group.data_block_addr
                    if not data_addr:
                        continue

                    stream.seek(data_addr)
                    blk_id = stream.read(4)
                    if blk_id == b"##DT":
                        blk = DataBlock(address=data_addr, stream=stream, mapped=mapped)
                    elif blk_id == b"##DL":
                        while True:
                            dl = DataList(address=data_addr, stream=stream, mapped=mapped)
                            if not dl.next_dl_addr:
                                break

                        data_addr = typing.cast(int, dl[f"data_block_addr{dl.links_nr - 2}"])
                        blk = DataBlock(address=data_addr, stream=stream, mapped=mapped)

                    elif blk_id == b"##HL":
                        hl = HeaderList(address=data_addr, stream=stream, mapped=mapped)

                        data_addr = hl.first_dl_addr
                        while True:
                            dl = DataList(address=data_addr, stream=stream, mapped=mapped)
                            if not dl.next_dl_addr:
                                break

                        data_addr = typing.cast(int, dl[f"data_block_addr{dl.links_nr - 2}"])
                        blk = DataBlock(address=data_addr, stream=stream, mapped=mapped)

                    next_block = bisect.bisect_right(addresses, data_addr)
                    if next_block == len(addresses):
                        block_len = limit - data_addr
                    else:
                        block_len = addresses[next_block] - data_addr

                    blk.block_len = block_len
                    stream.seek(data_addr)
                    stream.write(bytes(blk))
            except:
                print(format_exc())
                raise

            self.identification.unfinalized_standard_flags -= v4c.FLAG_UNFIN_UPDATE_LAST_DT_LENGTH
        self.identification.file_identification = b"MDF     "

    def _sort(
        self,
        stream: FileLike | mmap.mmap,
        compress: bool = True,
        current_progress_index: int = 0,
        max_progress_count: int = 0,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        flags = self.identification.unfinalized_standard_flags

        common: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for i, group in enumerate(self.groups):
            if group.sorted:
                continue

            try:
                data_block = next(group.get_data_blocks())
                common[data_block.address].append((i, group.channel_group.record_id))
            except:
                continue

        read = stream.read
        seek = stream.seek

        self._tempfile.seek(0, 2)

        tell = self._tempfile.tell
        write = self._tempfile.write

        for address, groups in common.items():
            cg_map = {rec_id: self.groups[index_].channel_group for index_, rec_id in groups}

            final_records: dict[int, list[DataBlockInfo]] = {id_: [] for (_, id_) in groups}

            for rec_id, channel_group in cg_map.items():
                if channel_group.address in self._cn_data_map:
                    gp_idx, cn_idx = self._cn_data_map[channel_group.address]
                    self.groups[gp_idx].signal_data[cn_idx] = ([], iter(EMPTY_TUPLE))

            group = self.groups[groups[0][0]]

            record_id_nr = group.data_group.record_id_len
            cg_size = group.record_size

            match record_id_nr:
                case 1:
                    _unpack_stuct = UINT8_uf
                case 2:
                    _unpack_stuct = UINT16_uf
                case 4:
                    _unpack_stuct = UINT32_uf
                case 8:
                    _unpack_stuct = UINT64_uf
                case _:
                    message = f"invalid record id size {record_id_nr}"
                    raise MdfException(message)

            rem = b""
            blocks = list(group.get_data_blocks())  # might be expensive ?
            # most of the steps are for sorting, but the last 2 are after we've done sorting
            # so remove the 2 steps that are not related to sorting from the count
            step = float(SORT_STEPS - 2) / len(blocks) / len(common)
            count = float(current_progress_index)
            previous = count
            for block_info in blocks:
                dtblock_address, dtblock_raw_size, dtblock_size, block_type, param = (
                    block_info.address,
                    typing.cast(int, block_info.original_size),
                    block_info.compressed_size,
                    block_info.block_type,
                    block_info.param,
                )

                count += step

                # if we've been told to notify about progress
                # and we've been given a max progress count (only way we can do progress updates)
                # and there's a tick update (at least 1 integer between the last update and the current index)
                # then we can notify about the callback progress

                if callable(progress) and max_progress_count and floor(previous) < floor(count):
                    progress(floor(count), max_progress_count)
                    previous = count

                seek(dtblock_address)

                if block_type != v4c.DT_BLOCK:
                    partial_records: dict[int, list[bytes]] = {id_: [] for _, id_ in groups}
                    new_data = read(dtblock_size)

                    if block_type == v4c.DZ_BLOCK_DEFLATE:
                        new_data = decompress(new_data, bufsize=dtblock_raw_size)
                    elif block_type == v4c.DZ_BLOCK_TRANSPOSED:
                        new_data = decompress(new_data, bufsize=dtblock_raw_size)
                        cols = typing.cast(int, param)
                        lines = dtblock_raw_size // cols

                        nd: NDArray[np.record] = np.rec.fromstring(new_data[: lines * cols], dtype=uint8)
                        nd = nd.reshape((cols, lines))
                        new_data = nd.T.ravel().tobytes() + new_data[lines * cols :]

                    new_data = rem + new_data

                    try:
                        rem = sort_data_block(
                            new_data,
                            partial_records,
                            cg_size,
                            record_id_nr,
                            _unpack_stuct,
                        )
                    except:
                        print(format_exc())
                        raise

                    for rec_id, records in partial_records.items():
                        channel_group = cg_map[rec_id]

                        dg_cntr: int | None
                        ch_cntr: int | None
                        if channel_group.address in self._cn_data_map:
                            dg_cntr, ch_cntr = self._cn_data_map[channel_group.address]
                        else:
                            dg_cntr, ch_cntr = None, None

                        if records:
                            tempfile_address = tell()

                            new_data = b"".join(records)
                            original_size = len(new_data)
                            if original_size:
                                if compress:
                                    new_data = lz_compress(new_data, store_size=True)
                                    compressed_size = len(new_data)

                                    write(new_data)

                                    if dg_cntr is not None and ch_cntr is not None:
                                        info = SignalDataBlockInfo(
                                            address=tempfile_address,
                                            compressed_size=compressed_size,
                                            original_size=original_size,
                                            block_type=v4c.DZ_BLOCK_LZ,
                                            location=v4c.LOCATION_TEMPORARY_FILE,
                                        )
                                        signal_data = typing.cast(
                                            list[tuple[list[SignalDataBlockInfo], Iterator[SignalDataBlockInfo]]],
                                            self.groups[dg_cntr].signal_data,
                                        )
                                        signal_data[ch_cntr][0].append(info)

                                    else:
                                        block_info = DataBlockInfo(
                                            address=tempfile_address,
                                            block_type=v4c.DZ_BLOCK_LZ,
                                            compressed_size=compressed_size,
                                            original_size=original_size,
                                            param=0,
                                        )
                                        final_records[rec_id].append(block_info)
                                else:
                                    write(new_data)

                                    if dg_cntr is not None and ch_cntr is not None:
                                        info = SignalDataBlockInfo(
                                            address=tempfile_address,
                                            compressed_size=original_size,
                                            original_size=original_size,
                                            block_type=v4c.DT_BLOCK,
                                            location=v4c.LOCATION_TEMPORARY_FILE,
                                        )
                                        signal_data = typing.cast(
                                            list[tuple[list[SignalDataBlockInfo], Iterator[SignalDataBlockInfo]]],
                                            self.groups[dg_cntr].signal_data,
                                        )
                                        signal_data[ch_cntr][0].append(info)

                                    else:
                                        block_info = DataBlockInfo(
                                            address=tempfile_address,
                                            block_type=v4c.DT_BLOCK,
                                            compressed_size=original_size,
                                            original_size=original_size,
                                            param=0,
                                        )
                                        final_records[rec_id].append(block_info)

                else:  # DTBLOCK
                    seek(dtblock_address)
                    limit = 32 * 1024 * 1024  # 32MB
                    while dtblock_size:
                        if dtblock_size > limit:
                            dtblock_size -= limit
                            new_data = rem + read(limit)
                        else:
                            new_data = rem + read(dtblock_size)
                            dtblock_size = 0
                        partial_records = {id_: [] for _, id_ in groups}

                        rem = sort_data_block(
                            new_data,
                            partial_records,
                            cg_size,
                            record_id_nr,
                            _unpack_stuct,
                        )

                        for rec_id, records in partial_records.items():
                            channel_group = cg_map[rec_id]

                            if channel_group.address in self._cn_data_map:
                                dg_cntr, ch_cntr = self._cn_data_map[channel_group.address]
                            else:
                                dg_cntr, ch_cntr = None, None

                            if records:
                                tempfile_address = tell()
                                new_data = b"".join(records)

                                original_size = len(new_data)
                                if original_size:
                                    if compress:
                                        new_data = lz_compress(new_data, store_size=True)
                                        compressed_size = len(new_data)

                                        write(new_data)

                                        if dg_cntr is not None and ch_cntr is not None:
                                            info = SignalDataBlockInfo(
                                                address=tempfile_address,
                                                compressed_size=compressed_size,
                                                original_size=original_size,
                                                block_type=v4c.DZ_BLOCK_LZ,
                                                location=v4c.LOCATION_TEMPORARY_FILE,
                                            )
                                            signal_data = typing.cast(
                                                list[tuple[list[SignalDataBlockInfo], Iterator[SignalDataBlockInfo]]],
                                                self.groups[dg_cntr].signal_data,
                                            )
                                            signal_data[ch_cntr][0].append(info)

                                        else:
                                            block_info = DataBlockInfo(
                                                address=tempfile_address,
                                                block_type=v4c.DZ_BLOCK_LZ,
                                                compressed_size=compressed_size,
                                                original_size=original_size,
                                                param=None,
                                            )

                                            final_records[rec_id].append(block_info)
                                    else:
                                        write(new_data)

                                        if dg_cntr is not None and ch_cntr is not None:
                                            info = SignalDataBlockInfo(
                                                address=tempfile_address,
                                                compressed_size=original_size,
                                                original_size=original_size,
                                                block_type=v4c.DT_BLOCK,
                                                location=v4c.LOCATION_TEMPORARY_FILE,
                                            )
                                            signal_data = typing.cast(
                                                list[tuple[list[SignalDataBlockInfo], Iterator[SignalDataBlockInfo]]],
                                                self.groups[dg_cntr].signal_data,
                                            )
                                            signal_data[ch_cntr][0].append(info)

                                        else:
                                            block_info = DataBlockInfo(
                                                address=tempfile_address,
                                                block_type=v4c.DT_BLOCK,
                                                compressed_size=original_size,
                                                original_size=original_size,
                                                param=None,
                                            )

                                            final_records[rec_id].append(block_info)

            # after we read all DTBLOCKs in the original file,
            # we assign freshly created blocks from temporary file to
            # corresponding groups.
            for idx, rec_id in groups:
                group = self.groups[idx]
                group.data_location = v4c.LOCATION_TEMPORARY_FILE
                group.set_blocks_info(final_records[rec_id])
                group.sorted = True

        for i, group in enumerate(self.groups):
            if flags & v4c.FLAG_UNFIN_UPDATE_CG_COUNTER:
                channel_group = group.channel_group

                if channel_group.flags & v4c.FLAG_CG_VLSD:
                    continue

                if self.version >= "4.20" and channel_group.flags & v4c.FLAG_CG_REMOTE_MASTER:
                    index = typing.cast(int, channel_group.cg_master_index)
                else:
                    index = i

                if group.uses_ld:
                    samples_size = channel_group.samples_byte_nr
                else:
                    samples_size = channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr

                total_size = sum(typing.cast(int, blk.original_size) for blk in group.get_data_blocks())

                cycles_nr = total_size // samples_size
                virtual_channel_group = self.virtual_groups[index]
                virtual_channel_group.cycles_nr = cycles_nr
                channel_group.cycles_nr = cycles_nr

        if self.identification.unfinalized_standard_flags & v4c.FLAG_UNFIN_UPDATE_CG_COUNTER:
            self.identification.unfinalized_standard_flags -= v4c.FLAG_UNFIN_UPDATE_CG_COUNTER
        if self.identification.unfinalized_standard_flags & v4c.FLAG_UNFIN_UPDATE_VLSD_BYTES:
            self.identification.unfinalized_standard_flags -= v4c.FLAG_UNFIN_UPDATE_VLSD_BYTES

    def _process_bus_logging(self) -> None:
        groups_count = len(self.groups)
        for index in range(groups_count):
            group = self.groups[index]
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                source = group.channel_group.acq_source

                if (
                    source
                    and source.bus_type in (v4c.BUS_TYPE_CAN, v4c.BUS_TYPE_OTHER)
                    and "CAN_DataFrame" in [ch.name for ch in group.channels]
                ):
                    try:
                        self._process_can_logging(index, group)
                    except Exception:
                        message = f"Error during CAN logging processing: {format_exc()}"
                        logger.error(message)

                if (
                    source
                    and source.bus_type in (v4c.BUS_TYPE_LIN, v4c.BUS_TYPE_OTHER)
                    and "LIN_Frame" in [ch.name for ch in group.channels]
                ):
                    try:
                        self._process_lin_logging(index, group)
                    except Exception as e:
                        message = f"Error during LIN logging processing: {e}"
                        logger.error(message)

    def _process_can_logging(self, group_index: int, grp: Group) -> None:
        channels = grp.channels
        group = grp

        dbc = None

        for channel in channels:
            if channel.name == "CAN_DataFrame":
                attachment_addr = channel.attachment

                if attachment_addr is not None:
                    if attachment_addr not in self._dbc_cache:
                        attachment, at_name, md5_sum = self.extract_attachment(
                            index=attachment_addr,
                        )
                        if at_name.suffix.lower() not in (".arxml", ".dbc"):
                            message = f'Expected .dbc or .arxml file as CAN channel attachment but got "{at_name}"'
                            logger.warning(message)
                        elif not attachment:
                            message = f'Attachment "{at_name}" not found'
                            logger.warning(message)
                        else:
                            dbc = load_can_database(at_name, contents=attachment)
                            if dbc:
                                self._dbc_cache[attachment_addr] = dbc
                    else:
                        dbc = self._dbc_cache[attachment_addr]
                break

        if not group.channel_group.flags & v4c.FLAG_CG_PLAIN_BUS_EVENT:
            self._prepare_record(group)
            data = self._load_data(group, record_offset=0, record_count=1)

            for fragment in data:
                self._set_temporary_master(None)
                self._set_temporary_master(self.get_master(group_index, data=fragment, one_piece=True))

                bus_ids = self.get(
                    "CAN_DataFrame.BusChannel",
                    group=group_index,
                    data=fragment,
                    samples_only=True,
                )[0].astype("<u1")

                msg_ids = (
                    self.get(
                        "CAN_DataFrame.ID",
                        group=group_index,
                        data=fragment,
                        samples_only=True,
                    )[0].astype("<u4")
                    & 0x1FFFFFFF
                )

                if len(bus_ids) == 0:
                    continue

                bus = bus_ids[0]
                msg_id = msg_ids[0]

                bus_map = self.bus_logging_map["CAN"].setdefault(bus, {})
                bus_map[int(msg_id)] = group_index

            self._set_temporary_master(None)

        elif dbc is None:
            self._prepare_record(group)
            data = self._load_data(group, optimize_read=False)

            for fragment in data:
                self._set_temporary_master(None)
                self._set_temporary_master(self.get_master(group_index, data=fragment, one_piece=True))

                bus_ids = self.get(
                    "CAN_DataFrame.BusChannel",
                    group=group_index,
                    data=fragment,
                    samples_only=True,
                )[0].astype("<u1")

                msg_ids = (
                    self.get(
                        "CAN_DataFrame.ID",
                        group=group_index,
                        data=fragment,
                        samples_only=True,
                    )[0].astype("<u4")
                    & 0x1FFFFFFF
                )

                if len(bus_ids) == 0:
                    continue

                buses = unique(bus_ids)

                for bus in buses:
                    bus_msg_ids = msg_ids[bus_ids == bus]
                    unique_id_array = unique(bus_msg_ids)
                    unique_id_array.sort()
                    unique_ids = typing.cast(list[int], unique_id_array.tolist())

                    bus_map = self.bus_logging_map["CAN"].setdefault(bus, {})

                    for msg_id in unique_ids:
                        bus_map[int(msg_id)] = group_index

            self._set_temporary_master(None)

        else:
            is_j1939 = dbc.contains_j1939

            if is_j1939:
                messages = {frame.arbitration_id.pgn: frame for frame in dbc}
            else:
                messages = {frame.arbitration_id.id: frame for frame in dbc}

            msg_map = {}

            self._prepare_record(group)
            data = self._load_data(group, optimize_read=False)

            for fragment in data:
                self._set_temporary_master(None)
                self._set_temporary_master(self.get_master(group_index, data=fragment, one_piece=True))

                data_bytes = self.get(
                    "CAN_DataFrame.DataBytes",
                    group=group_index,
                    data=fragment,
                    samples_only=True,
                )[0]

                bus_ids = self.get(
                    "CAN_DataFrame.BusChannel",
                    group=group_index,
                    data=fragment,
                    samples_only=True,
                )[0].astype("<u1")

                msg_id_signal = (
                    self.get("CAN_DataFrame.ID", group=group_index, data=fragment).astype("<u4") & 0x1FFFFFFF
                )

                if is_j1939:
                    tmp_pgn = msg_id_signal.samples >> 8
                    ps = tmp_pgn & 0xFF
                    pf = (msg_id_signal.samples >> 16) & 0xFF
                    _pgn = tmp_pgn & 0x3FF00
                    msg_id_signal.samples = where(pf >= 240, _pgn + ps, _pgn)

                buses = unique(bus_ids)
                if len(bus_ids) == 0:
                    continue

                for bus in buses:
                    idx_ = bus_ids == bus
                    bus_msg_ids = msg_id_signal.samples[idx_]

                    bus_t = msg_id_signal.timestamps[idx_]

                    bus_data_bytes = data_bytes[idx_]

                    unique_ids = sorted(unique(bus_msg_ids).astype("<u8"))

                    bus_map = self.bus_logging_map["CAN"].setdefault(bus, {})

                    for msg_id in unique_ids:
                        bus_map[int(msg_id)] = group_index

                    for msg_id in unique_ids:
                        frame = messages.get(msg_id, None)
                        if frame is None:
                            continue

                        idx = bus_msg_ids == msg_id
                        payload = bus_data_bytes[idx]
                        t = bus_t[idx]

                        extracted_signals = bus_logging_utils.extract_mux(payload, frame, msg_id, bus, t, raw=True)

                        for msg, signals in extracted_signals.items():
                            if len(next(iter(signals.values()))["samples"]) == 0:
                                continue
                            if msg not in msg_map:
                                sigs: list[Signal] = []

                                for name_, signal in signals.items():
                                    sig = Signal(
                                        samples=signal["samples"],
                                        timestamps=signal["t"],
                                        name=signal["name"],
                                        comment=signal["comment"],
                                        unit=signal["unit"],
                                        invalidation_bits=signal["invalidation_bits"],
                                        display_names={
                                            f"{frame.name}.{signal['name']}": "message_name",
                                            f"CAN{bus}.{frame.name}.{signal['name']}": "bus_name",
                                        },
                                        conversion=signal["conversion"],
                                        raw=True,
                                    )

                                    sigs.append(sig)

                                cg_nr = self.append(
                                    sigs,
                                    acq_name=f"from CAN{bus} message ID=0x{msg_id:X}",
                                    comment=f"{frame} 0x{msg_id:X}",
                                    common_timebase=True,
                                )

                                msg_map[msg] = cg_nr

                                for ch_index, ch in enumerate(self.groups[cg_nr].channels):
                                    if ch_index == 0:
                                        continue

                                    entry = cg_nr, ch_index

                                    name_ = f"{frame}.{ch.name}"
                                    self.channels_db.add(name_, entry)

                                    name_ = f"CAN{bus}.{frame}.{ch.name}"
                                    self.channels_db.add(name_, entry)

                                    name_ = f"CAN_DataFrame_{msg_id}.{ch.name}"
                                    self.channels_db.add(name_, entry)

                                    name_ = f"CAN{bus}.CAN_DataFrame_{msg_id}.{ch.name}"
                                    self.channels_db.add(name_, entry)

                            else:
                                index = msg_map[msg]

                                sigs_samples: list[tuple[NDArray[Any], NDArray[np.bool] | None]] = []

                                for name_, signal in signals.items():
                                    sigs_samples.append((signal["samples"], signal["invalidation_bits"]))

                                    t = signal["t"]

                                sigs_samples.insert(0, (t, None))

                                self.extend(index, sigs_samples)
                self._set_temporary_master(None)

    def _process_lin_logging(self, group_index: int, grp: Group) -> None:
        channels = grp.channels
        group = grp

        dbc = None

        for channel in channels:
            if channel.name == "LIN_Frame":
                attachment_addr = channel.attachment
                if attachment_addr is not None:
                    if attachment_addr not in self._dbc_cache:
                        attachment, at_name, md5_sum = self.extract_attachment(
                            index=attachment_addr,
                        )
                        if at_name.suffix.lower() not in (".arxml", ".dbc", ".ldf"):
                            message = (
                                f'Expected .dbc, .arxml or .ldf file as LIN channel attachment but got "{at_name}"'
                            )
                            logger.warning(message)
                        elif not attachment:
                            message = f'Attachment "{at_name}" not found'
                            logger.warning(message)
                        else:
                            contents = None if at_name.suffix.lower() == ".ldf" else attachment
                            dbc = load_can_database(at_name, contents=contents)
                            if dbc:
                                self._dbc_cache[attachment_addr] = dbc
                    else:
                        dbc = self._dbc_cache[attachment_addr]
                break

        if dbc is None:
            self._prepare_record(group)
            data = self._load_data(group, optimize_read=False)

            for fragment in data:
                self._set_temporary_master(None)
                self._set_temporary_master(self.get_master(group_index, data=fragment, one_piece=True))

                msg_ids = (
                    self.get(
                        "LIN_Frame.ID",
                        group=group_index,
                        data=fragment,
                        samples_only=True,
                    )[0].astype("<u4")
                    & 0x1FFFFFFF
                )

                unique_ids = sorted(unique(msg_ids).astype("<u8"))

                lin_map = self.bus_logging_map["LIN"]

                for msg_id in unique_ids:
                    lin_map[int(msg_id)] = group_index

            self._set_temporary_master(None)

        else:
            messages = {frame.arbitration_id.id: frame for frame in dbc}

            msg_map: dict[tuple[int | None, int | None, bool, int | None, str | None, int, int], int] = {}

            self._prepare_record(group)
            data = self._load_data(group, optimize_read=False)

            for fragment in data:
                self._set_temporary_master(None)
                self._set_temporary_master(self.get_master(group_index, data=fragment, one_piece=True))

                sig = self.get("LIN_Frame.ID", group=group_index, data=fragment).astype("<u4") & 0x1FFFFFFF

                data_bytes = self.get(
                    "LIN_Frame.DataBytes",
                    group=group_index,
                    data=fragment,
                    samples_only=True,
                )[0]

                bus_msg_ids = sig.samples

                bus_t = sig.timestamps
                bus_data_bytes = data_bytes

                unique_ids = sorted(unique(bus_msg_ids).astype("<u8"))

                lin_map = self.bus_logging_map["LIN"]

                for msg_id in unique_ids:
                    lin_map[int(msg_id)] = group_index

                for msg_id in unique_ids:
                    frame = messages.get(msg_id, None)
                    if frame is None:
                        continue

                    idx = bus_msg_ids == msg_id
                    payload = bus_data_bytes[idx]
                    t = bus_t[idx]

                    extracted_signals = bus_logging_utils.extract_mux(payload, frame, msg_id, 0, t, raw=True)

                    for msg, signals in extracted_signals.items():
                        if len(next(iter(signals.values()))["samples"]) == 0:
                            continue
                        if msg not in msg_map:
                            sigs = []

                            for name_, signal in signals.items():
                                sig = Signal(
                                    samples=signal["samples"],
                                    timestamps=signal["t"],
                                    name=signal["name"],
                                    comment=signal["comment"],
                                    unit=signal["unit"],
                                    invalidation_bits=signal["invalidation_bits"],
                                    display_names={
                                        f"{frame.name}.{signal['name']}": "message_name",
                                        f"LIN.{frame.name}.{signal['name']}": "bus_name",
                                    },
                                    conversion=signal["conversion"],
                                    raw=True,
                                )

                                sigs.append(sig)

                            cg_nr = self.append(
                                sigs,
                                acq_name=f"from LIN message ID=0x{msg_id:X}",
                                comment=f"{frame} 0x{msg_id:X}",
                                common_timebase=True,
                            )

                            msg_map[msg] = cg_nr

                            for ch_index, ch in enumerate(self.groups[cg_nr].channels):
                                if ch_index == 0:
                                    continue

                                entry = cg_nr, ch_index

                                name_ = f"{frame}.{ch.name}"
                                self.channels_db.add(name_, entry)

                                name_ = f"LIN.{frame}.{ch.name}"
                                self.channels_db.add(name_, entry)

                                name_ = f"LIN_Frame_{msg_id}.{ch.name}"
                                self.channels_db.add(name_, entry)

                                name_ = f"LIN.LIN_Frame_{msg_id}.{ch.name}"
                                self.channels_db.add(name_, entry)

                        else:
                            index = msg_map[msg]

                            sigs_samples: list[tuple[NDArray[Any], NDArray[np.bool] | None]] = []

                            for name_, signal in signals.items():
                                sigs_samples.append((signal["samples"], signal["invalidation_bits"]))

                                t = signal["t"]

                            sigs_samples.insert(0, (t, None))

                            self.extend(index, sigs_samples)
                self._set_temporary_master(None)

    def reload_header(self) -> None:
        if not self._file:
            raise RuntimeError("self._file is None")
        self.header = HeaderBlock(address=0x40, stream=self._file)


def debug_channel(
    mdf: MDF4,
    group: Group,
    channel: Channel,
    dependency: list[ChannelArrayBlock] | list[tuple[int, int]] | None,
    file: StringIO | None = None,
) -> None:
    """Use this to print debug information in case of errors.

    Parameters
    ----------
    mdf : MDF
        Source MDF object.
    group : dict
        Group.
    channel : Channel
        Channel object.
    dependency : ChannelDependency
        Channel dependency object.
    """
    print("MDF", "=" * 76, file=file)
    print("name:", mdf.name, file=file)
    print("version:", mdf.version, file=file)
    print("read fragment size:", mdf._read_fragment_size, file=file)
    print("write fragment size:", mdf._write_fragment_size, file=file)
    print()

    record = mdf._prepare_record(group)
    print("GROUP", "=" * 74, file=file)
    print("sorted:", group.sorted, file=file)
    print("data location:", group.data_location, file=file)
    print("data blocks:", group.data_blocks, file=file)
    print("dependencies", group.channel_dependencies, file=file)
    print("record:", record, file=file)
    print(file=file)

    cg = group.channel_group
    print("CHANNEL GROUP", "=" * 66, file=file)
    print(cg, cg.cycles_nr, cg.samples_byte_nr, cg.invalidation_bytes_nr, file=file)
    print(file=file)

    print("CHANNEL", "=" * 72, file=file)
    print(channel, file=file)
    print(file=file)

    print("CHANNEL ARRAY", "=" * 66, file=file)
    print(dependency, file=file)
    print(file=file)

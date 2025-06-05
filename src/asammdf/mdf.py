"""Common MDF file format module"""

import bz2
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from copy import deepcopy
import csv
from datetime import datetime, timezone
from enum import Enum
from functools import reduce
import gzip
from io import BufferedIOBase, BytesIO
import logging
import mmap
import os
from pathlib import Path
import re
from shutil import copy, move
import sys
from tempfile import gettempdir, mkdtemp
from traceback import format_exc
from types import TracebackType
import typing
from typing import Literal, Optional, Union
import xml.etree.ElementTree as ET
import zipfile

from canmatrix import CanMatrix
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from typing_extensions import Any, LiteralString, Never, overload, TypedDict, Unpack

from . import tool
from .blocks import mdf_v2, mdf_v3, mdf_v4
from .blocks import v2_v3_blocks as v3b
from .blocks import v2_v3_constants as v3c
from .blocks import v4_blocks as v4b
from .blocks import v4_constants as v4c
from .blocks.conversion_utils import from_dict
from .blocks.cutils import get_channel_raw_bytes_complete
from .blocks.mdf_common import (
    LastCallInfo,
    MdfCommonKwargs,
    MdfKwargs,
)
from .blocks.mdf_v4 import BusLoggingMap
from .blocks.options import FloatInterpolation, IntegerInterpolation
from .blocks.source_utils import Source
from .blocks.types import (
    BusType,
    ChannelsType,
    CompressionType,
    DbcFileType,
    EmptyChannelsType,
    FloatInterpolationModeType,
    IntInterpolationModeType,
    RasterType,
    StrPath,
)
from .blocks.utils import (
    as_non_byte_sized_signed_int,
    ChannelsDB,
    components,
    csv_bytearray2hex,
    csv_int2hex,
    downcast,
    FileLike,
    Fragment,
    is_file_like,
    load_can_database,
    matlab_compatible,
    MDF2_VERSIONS,
    MDF3_VERSIONS,
    MDF4_VERSIONS,
    MdfException,
    plausible_timestamps,
    randomized_string,
    SignalDataBlockInfo,
    SUPPORTED_VERSIONS,
    Terminated,
    THREAD_COUNT,
    UINT16_u,
    UINT64_u,
    UniqueDB,
    validate_version_argument,
    VirtualChannelGroup,
)
from .blocks.v2_v3_blocks import ChannelExtension
from .blocks.v4_blocks import (
    AttachmentBlock,
    EventBlock,
    FileHistory,
    FileIdentificationBlock,
    SourceInformation,
)
from .signal import InvalidationArray, Signal

try:
    import fsspec

    FSSPEF_AVAILABLE = True
except:
    FSSPEF_AVAILABLE = False

try:
    import polars as pl

    POLARS_AVAILABLE = True
except:
    POLARS_AVAILABLE = False

logger = logging.getLogger("asammdf")
LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo


target_byte_order = "<=" if sys.byteorder == "little" else ">="


__all__ = ["MDF", "SUPPORTED_VERSIONS"]

Version = Literal[v3c.Version2, v3c.Version, v4c.Version]


class SearchMode(Enum):
    plain = "plain"
    regex = "regex"
    wildcard = "wildcard"


if sys.version_info >= (3, 12):
    _Quoting = Literal["ALL", "MINIMAL", "NONE", "NONNUMERIC", "NOTNULL", "STRINGS"]
else:
    _Quoting = Literal["ALL", "MINIMAL", "NONE", "NONNUMERIC"]


class _ExportKwargs(TypedDict, total=False):
    single_time_base: bool
    raster: RasterType | None
    time_from_zero: bool
    use_display_names: bool
    empty_channels: EmptyChannelsType
    format: Literal["4", "5", "7.3"]
    oned_as: Literal["row", "column"]
    reduce_memory_usage: bool
    compression: LiteralString | bool
    time_as_date: bool
    ignore_value2text_conversions: bool
    raw: bool
    delimiter: str
    quotechar: str
    escapechar: str | None
    doublequote: bool
    lineterminator: str
    quoting: _Quoting
    add_units: bool


class _ConcatenateKwargs(TypedDict, total=False):
    process_bus_logging: bool
    use_display_names: bool


class _StackKwargs(TypedDict, total=False):
    process_bus_logging: bool
    use_display_names: bool


def get_measurement_timestamp_and_version(
    mdf: FileLike | mmap.mmap,
) -> tuple[datetime, str]:
    id_block = FileIdentificationBlock(address=0, stream=mdf)

    mdf_version = id_block.mdf_version
    header_cls: type[v4b.HeaderBlock] | type[v3b.HeaderBlock]
    if mdf_version >= 400:
        header_cls = v4b.HeaderBlock
    else:
        header_cls = v3b.HeaderBlock

    header = header_cls(address=64, stream=mdf)
    main_version, revision = divmod(mdf_version, 100)
    version = f"{main_version}.{revision}"

    return header.start_time, version


def get_temporary_filename(path: Path = Path("temporary.mf4"), dir: StrPath | None = None) -> Path:
    folder: StrPath
    if not dir:
        folder = gettempdir()
    else:
        folder = dir
    mf4_path = path.with_suffix(".mf4")
    idx = 0
    while True:
        tmp_path = (Path(folder) / mf4_path.name).with_suffix(f".{idx}.mf4")
        if not tmp_path.exists():
            break
        else:
            idx += 1

    return tmp_path


class MDF:
    r"""Unified access to MDF v3 and v4 files. Underlying _mdf's attributes and
    methods are linked to the `MDF` object via `setattr`. This is done to expose
    them to the user code and for performance considerations.

    Parameters
    ----------
    name : str | BytesIO | zipfile.ZipFile | bz2.BZ2File | gzip.GzipFile, optional
        MDF file name (if provided it must be a real file name), file-like
        object or compressed file opened as a Python object.

        .. versionchanged:: 6.2.0

            Added support for zipfile.ZipFile, bz2.BZ2File and gzip.GzipFile.

    version : str, default '4.10'
        MDF file version from ('2.00', '2.10', '2.14', '3.00', '3.10', '3.20',
        '3.30', '4.00', '4.10', '4.11', '4.20'). This argument is only used for
        MDF objects created from scratch; for MDF objects created from a file
        the version is set to file version.

    channels : iterable, optional
        Channel names that will be used for selective loading. This can
        dramatically improve the file loading time. Default is None -> load all
        channels.

        .. versionadded:: 6.1.0

        .. versionchanged:: 6.3.0 Make the default None.

    use_display_names : bool, default True
        For MDF v4 files, parse the XML channel comment to search for the
        display name; XML parsing is quite expensive so setting this to False
        can decrease the loading times very much.
    remove_source_from_channel_names : bool, default False
        Remove source from channel names ("Speed\XCP3" -> "Speed").
    raise_on_multiple_occurrences : bool, default True
        Raise MdfException when there are multiple channel occurrences in the
        file and the `get` call is ambiguous.

        .. versionadded:: 7.0.0

    temporary_folder : str | path-like, optional
        Folder to use for temporary files.

        .. versionadded:: 7.0.0

    process_bus_logging : bool, default True
        Controls whether the bus processing of MDF v4 files is done when the
        file is loaded.

        .. versionadded:: 8.0.0

    Examples
    --------
    >>> mdf = MDF(version='3.30')  # new MDF object with version 3.30
    >>> mdf = MDF('path/to/file.mf4')  # MDF loaded from file
    >>> mdf = MDF(BytesIO(data))  # MDF from file contents
    >>> mdf = MDF(zipfile.ZipFile('data.zip'))  # MDF creating using the first valid MDF from archive
    >>> mdf = MDF(bz2.BZ2File('path/to/data.bz2', 'rb'))  # MDF from bz2 object
    >>> mdf = MDF(gzip.GzipFile('path/to/data.gzip', 'rb'))  # MDF from gzip object
    """

    def __init__(
        self,
        name: StrPath | FileLike | zipfile.ZipFile | None = None,
        version: str | Version = "4.10",
        channels: list[str] | None = None,
        **kwargs: Unpack[MdfKwargs],
    ) -> None:
        if "callback" in kwargs:
            kwargs["progress"] = kwargs["callback"]
            del kwargs["callback"]

        temporary_folder = kwargs.get("temporary_folder", None)
        if temporary_folder:
            try:
                os.makedirs(temporary_folder, exist_ok=True)
            except:
                kwargs["temporary_folder"] = None

        self._mdf: mdf_v2.MDF2 | mdf_v3.MDF3 | mdf_v4.MDF4
        if name:
            original_name: str | Path | None
            if is_file_like(name):
                if isinstance(name, (BytesIO, BufferedIOBase)):
                    original_name = None
                    file_stream = name
                    do_close = False

                elif isinstance(name, bz2.BZ2File):
                    original_name = Path(name.name)
                    tmp_name = get_temporary_filename(original_name, dir=temporary_folder)
                    tmp_name.write_bytes(name.read())
                    file_stream = open(tmp_name, "rb")
                    name = tmp_name

                    do_close = True

                elif isinstance(name, gzip.GzipFile):
                    original_name = Path(name.name)
                    tmp_name = get_temporary_filename(original_name, dir=temporary_folder)
                    tmp_name.write_bytes(name.read())
                    file_stream = open(tmp_name, "rb")
                    name = tmp_name

                    do_close = True

                elif FSSPEF_AVAILABLE and isinstance(name, fsspec.spec.AbstractBufferedFile):
                    original_name = "AzureFile"
                    file_stream = name
                    do_close = False

                else:
                    raise MdfException(f"{type(name)} is not supported as input for the MDF class")

            elif isinstance(name, zipfile.ZipFile):
                archive = name
                files = archive.namelist()

                for fname in files:
                    if Path(fname).suffix.lower() in (".mdf", ".dat", ".mf4"):
                        original_name = fname
                        break
                else:
                    raise Exception("invalid zipped MF4: no supported file found in the archive")

                name = get_temporary_filename(Path(original_name), dir=temporary_folder)

                tmpdir = mkdtemp()
                output = archive.extract(fname, tmpdir)
                move(output, name)

                file_stream = open(name, "rb")
                do_close = True

            else:
                name = original_name = Path(name)
                if not name.is_file() or not name.exists():
                    raise MdfException(f'File "{name}" does not exist')

                if original_name.suffix.lower() in (".mf4z", ".zip"):
                    name = get_temporary_filename(original_name, dir=temporary_folder)
                    with zipfile.ZipFile(original_name, allowZip64=True) as archive:
                        files = archive.namelist()
                        for fname in files:
                            if Path(fname).suffix.lower() in (".mdf", ".dat", ".mf4"):
                                break
                        else:
                            raise Exception("invalid zipped MF4: no supported file found in the archive")

                        tmpdir = mkdtemp()
                        output = archive.extract(fname, tmpdir)

                        move(output, name)

                file_stream = open(name, "rb")
                do_close = True

            file_stream.seek(0)
            magic_header = file_stream.read(8)

            if magic_header.strip() not in (b"MDF", b"UnFinMF"):
                if do_close:
                    file_stream.close()
                raise MdfException(f'"{name}" is not a valid ASAM MDF file: magic header is {magic_header!r}')

            file_stream.seek(8)
            version = file_stream.read(4).decode("ascii").strip(" \0")
            if not version:
                _, version = get_measurement_timestamp_and_version(file_stream)

            if do_close:
                file_stream.close()

            common_kwargs = MdfCommonKwargs({**kwargs, "original_name": original_name, "__internal__": True})

            if version in MDF3_VERSIONS:
                self._mdf = mdf_v3.MDF3(name, channels=channels, **common_kwargs)
            elif version in MDF4_VERSIONS:
                self._mdf = mdf_v4.MDF4(name, channels=channels, **common_kwargs)
            elif version in MDF2_VERSIONS:
                self._mdf = mdf_v2.MDF2(name, channels=channels, **common_kwargs)
            else:
                message = f'"{name}" is not a supported MDF file; "{version}" file version was found'
                raise MdfException(message)

        else:
            common_kwargs = MdfCommonKwargs({**kwargs, "original_name": None, "__internal__": True})
            version = validate_version_argument(version)
            if version in MDF2_VERSIONS:
                version = typing.cast(v3c.Version2, version)
                self._mdf = mdf_v2.MDF2(version=version, **common_kwargs)
            elif version in MDF3_VERSIONS:
                version = typing.cast(v3c.Version, version)
                self._mdf = mdf_v3.MDF3(version=version, **common_kwargs)
            elif version in MDF4_VERSIONS:
                version = typing.cast(v4c.Version, version)
                self._mdf = mdf_v4.MDF4(version=version, **common_kwargs)
            else:
                message = (
                    f'"{version}" is not a supported MDF file version; Supported versions are {SUPPORTED_VERSIONS}'
                )
                raise MdfException(message)

        # we need a backreference to the MDF object to avoid it being garbage
        # collected in code like this:
        # MDF(filename).convert('4.10')
        self._mdf._parent = self

    def __enter__(self) -> "MDF":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            self.close()
        except:
            print(format_exc())
        return None

    def __del__(self) -> None:
        try:
            self.close()
        except:
            pass

    def __lt__(self, other: "MDF") -> bool:
        if self.header.start_time < other.header.start_time:
            return True
        elif self.header.start_time > other.header.start_time:
            return False
        else:
            t_min: list[float] = []
            for i, group in enumerate(self.groups):
                group = typing.cast(mdf_v3.Group | mdf_v4.Group, group)
                cycles_nr = group.channel_group.cycles_nr
                if cycles_nr and i in self.masters_db:
                    master_min = self._mdf.get_master(i, record_offset=0, record_count=1)
                    if len(master_min):
                        t_min.append(master_min[0])

            other_t_min: list[float] = []
            for i, group in enumerate(other.groups):
                group = typing.cast(mdf_v3.Group | mdf_v4.Group, group)
                cycles_nr = group.channel_group.cycles_nr
                if cycles_nr and i in other.masters_db:
                    master_min = other._mdf.get_master(i, record_offset=0, record_count=1)
                    if len(master_min):
                        other_t_min.append(master_min[0])

            if not t_min or not other_t_min:
                return True
            else:
                return min(t_min) < min(other_t_min)

    def _transfer_events(self, other: "MDF") -> None:
        def get_scopes(event: EventBlock, events: list[EventBlock]) -> list[tuple[int, int] | int]:
            if event.scopes:
                return event.scopes
            else:
                if event.parent is not None:
                    return get_scopes(events[event.parent], events)
                elif event.range_start is not None:
                    return get_scopes(events[event.range_start], events)
                else:
                    return event.scopes

        if isinstance(other._mdf, mdf_v4.MDF4):
            for event in other._mdf.events:
                if isinstance(self._mdf, mdf_v4.MDF4):
                    new_event = deepcopy(event)
                    event_valid = True
                    for i, ref in enumerate(new_event.scopes):
                        if not isinstance(ref, int):
                            dg_cntr, ch_cntr = ref
                            try:
                                self.groups[dg_cntr].channels[ch_cntr]
                            except:
                                event_valid = False
                        else:
                            dg_cntr = ref
                            try:
                                self.groups[dg_cntr]
                            except:
                                event_valid = False
                    # ignore attachments for now
                    for i in range(new_event.attachment_nr):
                        key = f"attachment_{i}_addr"
                        event[key] = 0
                    if event_valid:
                        self._mdf.events.append(new_event)
                else:
                    ev_type = event.event_type
                    ev_range = event.range_type
                    ev_base = event.sync_base
                    ev_factor = event.sync_factor

                    timestamp = ev_base * ev_factor

                    try:
                        comment_elem = ET.fromstring(event.comment.replace(' xmlns="http://www.asam.net/mdf/v4"', ""))
                        pre_elem = comment_elem.find(".//pre_trigger_interval")
                        if pre_elem is not None:
                            pre = float(pre_elem.text) if pre_elem.text else 0.0
                        else:
                            pre = 0.0
                        post_elem = comment_elem.find(".//post_trigger_interval")
                        if post_elem is not None:
                            post = float(post_elem.text) if post_elem.text else 0.0
                        else:
                            post = 0.0
                        tx_elem = comment_elem.find(".//TX")
                        if tx_elem is not None:
                            comment = tx_elem.text if tx_elem.text else ""
                        else:
                            comment = ""

                    except:
                        pre = 0.0
                        post = 0.0
                        comment = event.comment

                    if comment:
                        comment += ": "

                    if ev_range == v4c.EVENT_RANGE_TYPE_BEGINNING:
                        comment += "Begin of "
                    elif ev_range == v4c.EVENT_RANGE_TYPE_END:
                        comment += "End of "
                    else:
                        comment += "Single point "

                    if ev_type == v4c.EVENT_TYPE_RECORDING:
                        comment += "recording"
                    elif ev_type == v4c.EVENT_TYPE_RECORDING_INTERRUPT:
                        comment += "recording interrupt"
                    elif ev_type == v4c.EVENT_TYPE_ACQUISITION_INTERRUPT:
                        comment += "acquisition interrupt"
                    elif ev_type == v4c.EVENT_TYPE_START_RECORDING_TRIGGER:
                        comment += "measurement start trigger"
                    elif ev_type == v4c.EVENT_TYPE_STOP_RECORDING_TRIGGER:
                        comment += "measurement stop trigger"
                    elif ev_type == v4c.EVENT_TYPE_TRIGGER:
                        comment += "trigger"
                    else:
                        comment += "marker"

                    scopes = get_scopes(event, other._mdf.events)
                    if scopes:
                        for i, ref in enumerate(scopes):
                            event_valid = True
                            if not isinstance(ref, int):
                                dg_cntr, ch_cntr = ref
                                try:
                                    (self.groups[dg_cntr])
                                except:
                                    event_valid = False
                            else:
                                dg_cntr = ref
                                try:
                                    (self.groups[dg_cntr])
                                except:
                                    event_valid = False
                            if event_valid:
                                self._mdf.add_trigger(
                                    dg_cntr,
                                    timestamp,
                                    pre_time=pre,
                                    post_time=post,
                                    comment=comment,
                                )
                    else:
                        for i, _ in enumerate(self.groups):
                            self._mdf.add_trigger(
                                i,
                                timestamp,
                                pre_time=pre,
                                post_time=post,
                                comment=comment,
                            )

        else:
            for trigger_info in other._mdf.iter_get_triggers():
                comment = trigger_info["comment"]
                timestamp = trigger_info["time"]
                group = trigger_info["group"]

                if not isinstance(self._mdf, mdf_v4.MDF4):
                    self._mdf.add_trigger(
                        group,
                        timestamp,
                        pre_time=trigger_info["pre_time"],
                        post_time=trigger_info["post_time"],
                        comment=comment,
                    )
                else:
                    if timestamp:
                        ev_type = v4c.EVENT_TYPE_TRIGGER
                    else:
                        ev_type = v4c.EVENT_TYPE_START_RECORDING_TRIGGER
                    event = EventBlock(
                        event_type=ev_type,
                        sync_base=int(timestamp * 10**9),
                        sync_factor=10**-9,
                        scope_0_addr=0,  # type: ignore[call-arg]
                    )
                    event.comment = comment
                    event.scopes.append(group)
                    self._mdf.events.append(event)

    def _transfer_header_data(self, other: "MDF", message: str = "") -> None:
        self.header.author = other.header.author
        self.header.department = other.header.department
        self.header.project = other.header.project
        self.header.subject = other.header.subject
        self.header.comment = other.header.comment
        if isinstance(self._mdf, mdf_v4.MDF4) and message:
            fh = FileHistory()
            fh.comment = f"""<FHcomment>
    <TX>{message}</TX>
    <tool_id>{tool.__tool__}</tool_id>
    <tool_vendor>{tool.__vendor__}</tool_vendor>
    <tool_version>{tool.__version__}</tool_version>
</FHcomment>"""

            self._mdf.file_history = [fh]

    @staticmethod
    def _transfer_channel_group_data(
        out_group: v3b.ChannelGroup | v4b.ChannelGroup, source_group: v3b.ChannelGroup | v4b.ChannelGroup
    ) -> None:
        if not isinstance(out_group, v4b.ChannelGroup) or not isinstance(source_group, v4b.ChannelGroup):
            out_group.comment = source_group.comment
        else:
            out_group.flags = source_group.flags
            out_group.path_separator = source_group.path_separator
            out_group.comment = source_group.comment
            out_group.acq_name = source_group.acq_name
            acq_source = source_group.acq_source
            if acq_source:
                out_group.acq_source = acq_source.copy()

    def _transfer_metadata(self, other: "MDF", message: str = "") -> None:
        self._transfer_events(other)
        self._transfer_header_data(other, message)

    def __contains__(self, channel: str) -> bool:
        """If *'channel name'* in *'mdf file'*"""
        return channel in self.channels_db

    def __iter__(self) -> Iterator[Signal]:
        """Iterate over all the channels found in the file; master channels
        are skipped from iteration.
        """
        yield from self.iter_channels()

    def configure(
        self,
        *,
        from_other: Optional["MDF"] = None,
        read_fragment_size: int | None = None,
        write_fragment_size: int | None = None,
        use_display_names: bool | None = None,
        single_bit_uint_as_bool: bool | None = None,
        integer_interpolation: IntInterpolationModeType | IntegerInterpolation | None = None,
        float_interpolation: FloatInterpolationModeType | FloatInterpolation | None = None,
        raise_on_multiple_occurrences: bool | None = None,
        temporary_folder: str | None = None,
        fill_0_for_missing_computation_channels: bool | None = None,
    ) -> None:
        """Configure `MDF` parameters.

        The default values for the options are the following:

        * read_fragment_size = 256 MB
        * write_fragment_size = 4 MB
        * use_display_names = True
        * single_bit_uint_as_bool = False
        * integer_interpolation = 0 (repeat previous sample)
        * float_interpolation = 1 (linear interpolation)
        * raise_on_multiple_occurrences = True
        * temporary_folder = ""
        * fill_0_for_missing_computation_channels = False

        Parameters
        ----------
        from_other : MDF, optional
            Copy configuration options from other MDF.

            .. versionadded:: 6.2.0

        read_fragment_size : int, optional
            Size hint of split data blocks. If the initial size is smaller, then
            no data list is used. The actual split size depends on the data
            groups' records size.
        write_fragment_size : int, optional
            Size hint of split data blocks. If the initial size is smaller, then
            no data list is used. The actual split size depends on the data
            groups' records size. Maximum size is 4 MB to ensure compatibility
            with CANape.
        use_display_names : bool, optional
            Search for display name in the Channel XML comment.
        single_bit_uint_as_bool : bool, optional
            Return single bit channels as np.bool arrays.
        integer_interpolation : int, optional
            Interpolation mode for integer channels.

            * 0 - repeat previous sample
            * 1 - linear interpolation
            * 2 - hybrid interpolation: channels with integer data type (raw
              values) that have a conversion that outputs float values will use
              linear interpolation, otherwise the previous sample is used

            .. versionchanged:: 6.2.0
                Added hybrid interpolation mode.

        float_interpolation : int, optional
            Interpolation mode for float channels.

            * 0 - repeat previous sample
            * 1 - linear interpolation

            .. versionadded:: 6.2.0

        raise_on_multiple_occurrences : bool, optional
            Raise MdfException when there are multiple channel occurrences in
            the file and the `get` call is ambiguous.

            .. versionadded:: 6.2.0

        temporary_folder : str, optional
            Default folder for temporary files.

            .. versionadded:: 7.0.0

        fill_0_for_missing_computation_channels : bool, optional
            When a channel required by a computed channel is missing, then fill
            with 0 values. If False, then the computation will fail and the
            computed channel will be marked as not existing.

            .. versionadded:: 7.1.0
        """

        if from_other is not None:
            self._mdf._read_fragment_size = from_other._mdf._read_fragment_size
            self._mdf._write_fragment_size = from_other._mdf._write_fragment_size
            self._mdf._use_display_names = from_other._mdf._use_display_names
            self._mdf._single_bit_uint_as_bool = from_other._mdf._single_bit_uint_as_bool
            self._mdf._integer_interpolation = from_other._mdf._integer_interpolation
            self._mdf._float_interpolation = from_other._mdf._float_interpolation
            self._mdf._raise_on_multiple_occurrences = from_other._mdf._raise_on_multiple_occurrences

        if read_fragment_size is not None:
            self._mdf._read_fragment_size = int(read_fragment_size)

        if write_fragment_size is not None:
            self._mdf._write_fragment_size = min(int(write_fragment_size), 4 * 1024 * 1024)

        if use_display_names is not None:
            self._mdf._use_display_names = bool(use_display_names)

        if single_bit_uint_as_bool is not None:
            self._mdf._single_bit_uint_as_bool = bool(single_bit_uint_as_bool)

        if integer_interpolation is not None:
            self._mdf._integer_interpolation = IntegerInterpolation(integer_interpolation)

        if float_interpolation is not None:
            self._mdf._float_interpolation = FloatInterpolation(float_interpolation)

        if temporary_folder is not None:
            try:
                os.makedirs(temporary_folder, exist_ok=True)
                self._mdf.temporary_folder = temporary_folder
            except:
                self._mdf.temporary_folder = None

        if raise_on_multiple_occurrences is not None:
            self._mdf._raise_on_multiple_occurrences = bool(raise_on_multiple_occurrences)

    @property
    def original_name(self) -> str | Path | None:
        return self._mdf.original_name

    @original_name.setter
    def original_name(self, value: str | Path | None) -> None:
        self._mdf.original_name = value

    @property
    def name(self) -> Path:
        return self._mdf.name

    @property
    def identification(self) -> v3b.FileIdentificationBlock | v4b.FileIdentificationBlock:
        return self._mdf.identification

    @property
    def version(self) -> str:
        return self._mdf.version

    @property
    def header(self) -> v3b.HeaderBlock | v4b.HeaderBlock:
        return self._mdf.header

    @property
    def groups(self) -> list[mdf_v3.Group] | list[mdf_v4.Group]:
        return self._mdf.groups

    @property
    def channels_db(self) -> ChannelsDB:
        return self._mdf.channels_db

    @property
    def masters_db(self) -> dict[int, int]:
        return self._mdf.masters_db

    @property
    def attachments(self) -> list[AttachmentBlock]:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("Attachments are only supported in MDF4 files")
        return self._mdf.attachments

    @attachments.setter
    def attachments(self, attachments_list: list[AttachmentBlock]) -> None:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("Attachments are only supported in MDF4 files")
        self._mdf.attachments = attachments_list

    @property
    def events(self) -> list[EventBlock]:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("Events are only supported in MDF4 files")
        return self._mdf.events

    @events.setter
    def events(self, events_list: list[EventBlock]) -> None:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("Events are only supported in MDF4 files")
        self._mdf.events = events_list

    @property
    def bus_logging_map(self) -> BusLoggingMap:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("Bus logging is only supported in MDF4 files")
        return self._mdf.bus_logging_map

    @property
    def last_call_info(self) -> LastCallInfo:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("last_call_info is only supported in MDF4 files")
        return self._mdf.last_call_info

    @property
    def file_history(self) -> list[FileHistory]:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("file_history is only supported in MDF4 files")
        return self._mdf.file_history

    @file_history.setter
    def file_history(self, file_history_list: list[FileHistory]) -> None:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("file_history is only supported in MDF4 files")
        self._mdf.file_history = file_history_list

    @property
    def virtual_groups(self) -> dict[int, VirtualChannelGroup]:
        return self._mdf.virtual_groups

    @virtual_groups.setter
    def virtual_groups(self, groups: dict[int, VirtualChannelGroup]) -> None:
        self._mdf.virtual_groups = groups

    @property
    def virtual_groups_map(self) -> dict[int, int]:
        return self._mdf.virtual_groups_map

    @virtual_groups_map.setter
    def virtual_groups_map(self, groups_map: dict[int, int]) -> None:
        self._mdf.virtual_groups_map = groups_map

    def add_trigger(
        self,
        group: int,
        timestamp: float,
        pre_time: float = 0,
        post_time: float = 0,
        comment: str = "",
    ) -> None:
        if isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("add_trigger is only supported in MDF2 and MDF3 files")
        return self._mdf.add_trigger(
            group,
            timestamp,
            pre_time=pre_time,
            post_time=post_time,
            comment=comment,
        )

    def get_invalidation_bits(
        self, group_index: int, pos_invalidation_bit: int, fragment: Fragment
    ) -> InvalidationArray:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("get_invalidation_bits is only supported in MDF4 file")
        return self._mdf.get_invalidation_bits(group_index, pos_invalidation_bit, fragment)

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
        return self._mdf.append(
            signals,
            acq_name=acq_name,
            acq_source=acq_source,
            comment=comment,
            common_timebase=common_timebase,
            units=units,
        )

    def extend(self, index: int, signals: Sequence[tuple[NDArray[Any], NDArray[np.bool] | None]]) -> None:
        return self._mdf.extend(index, signals)

    def get_channel_name(self, group: int, index: int) -> str:
        return self._mdf.get_channel_name(group, index)

    def get_channel_metadata(
        self, name: str | None = None, group: int | None = None, index: int | None = None
    ) -> v3b.Channel | v4b.Channel:
        return self._mdf.get_channel_metadata(name=name, group=group, index=index)

    def get_channel_unit(self, name: str | None = None, group: int | None = None, index: int | None = None) -> str:
        return self._mdf.get_channel_unit(name=name, group=group, index=index)

    def get_channel_comment(self, name: str | None = None, group: int | None = None, index: int | None = None) -> str:
        return self._mdf.get_channel_comment(name=name, group=group, index=index)

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
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("Attachments are only supported in MDF4 files")
        return self._mdf.attach(
            data,
            file_name=file_name,
            hash_sum=hash_sum,
            comment=comment,
            compression=compression,
            mime=mime,
            embedded=embedded,
            password=password,
        )

    def close(self) -> None:
        return self._mdf.close()

    def extract_attachment(
        self, index: int | None = None, password: str | bytes | None = None
    ) -> tuple[bytes | str, Path, bytes | str]:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("Attachments are only supported in MDF4 files")
        return self._mdf.extract_attachment(index=index, password=password)

    def convert(self, version: str | Version, progress: Any | None = None) -> "MDF":
        """Convert `MDF` to other version.

        Parameters
        ----------
        version : str
            New MDF file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20').

        Returns
        -------
        out : MDF
            New `MDF` object.
        """
        version = validate_version_argument(version)

        out = MDF(version=version, **self._mdf._kwargs)

        out.configure(from_other=self)

        out.header.start_time = self.header.start_time

        groups_nr = len(self.virtual_groups)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

                if progress.stop:
                    raise Terminated

        # walk through all groups and get all channels
        for i, virtual_group in enumerate(self.virtual_groups):
            for idx, sigs in enumerate(self._mdf._yield_selected_signals(virtual_group, version=version)):
                if idx == 0:
                    sigs = typing.cast(list[Signal], sigs)
                    if sigs:
                        cg = self.groups[virtual_group].channel_group
                        cg_nr = out.append(
                            sigs,
                            common_timebase=True,
                            comment=cg.comment,
                        )
                        MDF._transfer_channel_group_data(out.groups[cg_nr].channel_group, cg)
                    else:
                        break
                else:
                    sigs = typing.cast(list[tuple[NDArray[Any], None]], sigs)
                    out.extend(cg_nr, sigs)

                if progress and progress.stop:
                    raise Terminated

            if progress is not None:
                if callable(progress):
                    progress(i + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1)
                    progress.signals.setMaximum.emit(groups_nr)

                    if progress.stop:
                        raise Terminated

        out._transfer_metadata(self, message=f"Converted from {self.name}")

        return out

    def cut(
        self,
        start: float | None = None,
        stop: float | None = None,
        whence: int = 0,
        version: str | Version | None = None,
        include_ends: bool = True,
        time_from_zero: bool = False,
        progress: Any | None = None,
    ) -> "MDF":
        """Cut `MDF`. `start` and `stop` are absolute values or values relative
        to the first timestamp depending on the `whence` argument.

        Parameters
        ----------
        start : float, optional
            Start time; default is None. If None, the start of the measurement
            is used.
        stop : float, optional
            Stop time; default is None. If None, the end of the measurement is
            used.
        whence : int, default 0
            How to search for the start and stop values.

            * 0 : absolute
            * 1 : relative to first timestamp

        version : str, optional
            New MDF file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default is None
            and in this case the original file version is used.
        include_ends : bool, default True
            Include the `start` and `stop` timestamps after cutting the signal.
            If `start` and `stop` are not found in the original timestamps,
            then the new samples will be computed using interpolation.
        time_from_zero : bool, default False
            Start timestamps from 0s in the cut measurement.

        Returns
        -------
        out : MDF
            New `MDF` object.
        """

        if version is None:
            version = self.version
        else:
            version = validate_version_argument(version)

        out = MDF(
            version=version,
            **self._mdf._kwargs,
        )

        integer_interpolation_mode = self._mdf._integer_interpolation
        float_interpolation_mode = self._mdf._float_interpolation
        out.configure(from_other=self)

        if whence == 1:
            timestamps: list[float] = []
            for group in self.virtual_groups:
                master = self._mdf.get_master(group, record_offset=0, record_count=1)
                if master.size:
                    timestamps.append(master[0])

            if timestamps:
                first_timestamp = np.amin(timestamps)
            else:
                first_timestamp = 0

            if start is not None:
                start += first_timestamp
            if stop is not None:
                stop += first_timestamp

        if time_from_zero:
            delta = start if start else 0.0
            t_epoch = self.header.start_time.timestamp() + delta
            out.header.start_time = datetime.fromtimestamp(t_epoch)
        else:
            delta = 0
            out.header.start_time = self.header.start_time

        groups_nr = len(self.virtual_groups)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

        # walk through all groups and get all channels
        for i, (group_index, virtual_group) in enumerate(self.virtual_groups.items()):
            included_channels = self.included_channels(group_index)[group_index]
            if not included_channels:
                continue

            idx = 0
            signals: list[Signal] = []
            for j, sigs in enumerate(self._mdf._yield_selected_signals(group_index, groups=included_channels)):
                if not sigs:
                    break
                if j == 0:
                    sigs = typing.cast(list[Signal], sigs)
                    master = sigs[0].timestamps
                    signals = sigs
                else:
                    sigs = typing.cast(list[tuple[NDArray[Any], None]], sigs)
                    master = sigs[0][0]

                if not len(master):
                    continue

                needs_cutting = True

                # check if this fragmement is within the cut interval or
                # if the cut interval has ended
                if start is None and stop is None:
                    fragment_start = None
                    fragment_stop = None
                    start_index = 0
                    stop_index = len(master)
                    needs_cutting = False
                elif start is None:
                    fragment_start = None
                    start_index = 0
                    if master[0] > stop:
                        break
                    else:
                        fragment_stop = min(stop, master[-1])
                        stop_index = np.searchsorted(master, fragment_stop, side="right")
                        if stop_index == len(master):
                            needs_cutting = False

                elif stop is None:
                    fragment_stop = None
                    if master[-1] < start:
                        continue
                    else:
                        fragment_start = max(start, master[0])
                        start_index = np.searchsorted(master, fragment_start, side="left")
                        stop_index = len(master)
                        if start_index == 0:
                            needs_cutting = False
                else:
                    if master[0] > stop:
                        break
                    elif master[-1] < start:
                        continue
                    else:
                        fragment_start = max(start, master[0])
                        start_index = np.searchsorted(master, fragment_start, side="left")
                        fragment_stop = min(stop, master[-1])
                        stop_index = np.searchsorted(master, fragment_stop, side="right")
                        if start_index == 0 and stop_index == len(master):
                            needs_cutting = False

                # update the signal if this is not the first yield
                if j:
                    sigs = typing.cast(list[tuple[NDArray[Any], None]], sigs)
                    for signal, (samples, invalidation) in zip(signals, sigs[1:], strict=False):
                        signal.samples = samples
                        signal.timestamps = master
                        signal.invalidation_bits = invalidation

                if needs_cutting:
                    master = (
                        Signal(master, master, name="_")
                        .cut(
                            fragment_start,
                            fragment_stop,
                            include_ends,
                            integer_interpolation_mode=integer_interpolation_mode,
                            float_interpolation_mode=float_interpolation_mode,
                        )
                        .timestamps
                    )

                    if not len(master):
                        continue

                    signals = [
                        sig.cut(
                            master[0],
                            master[-1],
                            include_ends=include_ends,
                            integer_interpolation_mode=integer_interpolation_mode,
                            float_interpolation_mode=float_interpolation_mode,
                        )
                        for sig in signals
                    ]

                if time_from_zero:
                    master = master - delta
                    for sig in signals:
                        sig.timestamps = master

                if idx == 0:
                    if start:
                        start_ = f"{start}s"
                    else:
                        start_ = "start of measurement"
                    if stop:
                        stop_ = f"{stop}s"
                    else:
                        stop_ = "end of measurement"
                    cg = self.groups[group_index].channel_group
                    cg_nr = out.append(
                        signals,
                        common_timebase=True,
                        comment=cg.comment,
                    )
                    MDF._transfer_channel_group_data(out.groups[cg_nr].channel_group, cg)

                else:
                    signals_samples = [(sig.samples, sig.invalidation_bits) for sig in signals]
                    signals_samples.insert(0, (master, None))
                    out.extend(cg_nr, signals_samples)

                idx += 1

                if progress and progress.stop:
                    raise Terminated

            # if the cut interval is not found in the measurement
            # then append a data group with 0 cycles
            if idx == 0 and signals:
                for sig in signals:
                    sig.samples = sig.samples[:0]
                    sig.timestamps = sig.timestamps[:0]
                    if sig.invalidation_bits is not None:
                        sig.invalidation_bits = InvalidationArray(sig.invalidation_bits[:0])

                if start:
                    start_ = f"{start}s"
                else:
                    start_ = "start of measurement"
                if stop:
                    stop_ = f"{stop}s"
                else:
                    stop_ = "end of measurement"
                cg = self.groups[group_index].channel_group
                cg_nr = out.append(
                    signals,
                    common_timebase=True,
                )
                MDF._transfer_channel_group_data(out.groups[cg_nr].channel_group, cg)

            if progress is not None:
                if callable(progress):
                    progress(i + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1)

                    if progress.stop:
                        print("return terminated")
                        raise Terminated

        out._transfer_metadata(self, message=f"Cut from {start_} to {stop_}")

        return out

    @overload
    def get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: RasterType | None = ...,
        samples_only: Literal[False] = ...,
        data: tuple[bytes, int, int | None] | Fragment | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = 0,
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
        data: tuple[bytes, int, int | None] | Fragment | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: Literal[True],
        record_offset: int = 0,
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
        data: tuple[bytes, int, int | None] | Fragment | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = 0,
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
        data: tuple[bytes, int, int | None] | Fragment | None = ...,
        raw: bool = ...,
        ignore_invalidation_bits: bool = ...,
        record_offset: int = 0,
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
        data: tuple[bytes, int, int | None] | Fragment | None = None,
        raw: bool = False,
        ignore_invalidation_bits: bool = False,
        record_offset: int = 0,
        record_count: int | None = None,
        skip_channel_validation: bool = False,
    ) -> Signal | tuple[NDArray[Any], NDArray[np.bool] | None]:
        if isinstance(self._mdf, mdf_v4.MDF4):
            if data is not None and not isinstance(data, Fragment):
                raise TypeError("'data' must be of type Fragment")

            return self._mdf.get(
                name=name,
                group=group,
                index=index,
                raster=raster,
                samples_only=samples_only,
                data=data,
                raw=raw,
                ignore_invalidation_bits=ignore_invalidation_bits,
                record_offset=record_offset,
                record_count=record_count,
                skip_channel_validation=skip_channel_validation,
            )

        if data is not None and not isinstance(data, tuple):
            raise TypeError("'data' must be of type tuple[bytes, int, int | None]")

        return self._mdf.get(
            name=name,
            group=group,
            index=index,
            raster=raster,
            samples_only=samples_only,
            data=data,
            raw=raw,
            ignore_invalidation_bits=ignore_invalidation_bits,
            record_offset=record_offset,
            record_count=record_count,
            skip_channel_validation=skip_channel_validation,
        )

    def included_channels(
        self,
        index: int | None = None,
        channels: ChannelsType | None = None,
        skip_master: bool = True,
        minimal: bool = True,
    ) -> dict[int, dict[int, list[int]]]:
        return self._mdf.included_channels(
            index=index,
            channels=channels,
            skip_master=skip_master,
            minimal=minimal,
        )

    def reload_header(self) -> None:
        return self._mdf.reload_header()

    def get_master(
        self,
        index: int,
        data: tuple[bytes, int, int | None] | Fragment | None = None,
        record_offset: int = 0,
        record_count: int | None = None,
        one_piece: bool = False,
    ) -> NDArray[Any]:
        if isinstance(self._mdf, mdf_v4.MDF4):
            if data and not isinstance(data, Fragment):
                raise TypeError("'data' must be of type Fragment")

            return self._mdf.get_master(
                index,
                data=data,
                record_offset=record_offset,
                record_count=record_count,
                one_piece=one_piece,
            )

        if data is not None and not isinstance(data, tuple):
            raise TypeError("'data' must be of type tuple[bytes, int, int | None]")

        return self._mdf.get_master(
            index,
            data=data,
            record_offset=record_offset,
            record_count=record_count,
            one_piece=one_piece,
        )

    def iter_get_triggers(self) -> Iterator[mdf_v3.TriggerInfoDict]:
        if isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("iter_get_triggers is not supported in MDF4 files")
        return self._mdf.iter_get_triggers()

    def info(self) -> dict[str, object]:
        return self._mdf.info()

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
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("get_bus_signal is only supported in MDF4 files")
        return self._mdf.get_bus_signal(
            bus,
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
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("get_can_signal is only supported in MDF4 files")
        return self._mdf.get_can_signal(
            name,
            database=database,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
            raw=raw,
            ignore_value2text_conversion=ignore_value2text_conversion,
        )

    def get_lin_signal(
        self,
        name: str,
        database: CanMatrix | str | Path | None = None,
        ignore_invalidation_bits: bool = False,
        data: Fragment | None = None,
        raw: bool = False,
        ignore_value2text_conversion: bool = True,
    ) -> Signal:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("get_lin_signal is only supported in MDF4 files")
        return self._mdf.get_lin_signal(
            name,
            database=database,
            ignore_invalidation_bits=ignore_invalidation_bits,
            data=data,
            raw=raw,
            ignore_value2text_conversion=ignore_value2text_conversion,
        )

    def export(
        self,
        fmt: Literal["asc", "csv", "hdf5", "mat", "parquet"],
        filename: StrPath | None = None,
        progress: Any | None = None,
        **kwargs: Unpack[_ExportKwargs],
    ) -> None:
        r"""Export `MDF` to other formats. The `MDF` file name is used if
        available, otherwise the `filename` argument must be provided.

        The pandas export option was removed. You should use the method
        `to_dataframe` instead.

        Parameters
        ----------
        fmt : str
            Can be one of the following:

            * `csv` : CSV export that uses the ',' delimiter. This option
              will generate a new csv file for each data group
              (<MDFNAME>_DataGroup_<cntr>.csv)

            * `hdf5` : HDF5 file output; each `MDF` data group is mapped to
              an HDF5 group with the name 'DataGroup_<cntr>'
              (where <cntr> is the index)

            * `mat` : Matlab .mat version 4, 5 or 7.3 export. If
              `single_time_base=False`, the channels will be renamed in the mat
              file to 'D<cntr>_<channel name>'. The channel group master will
              be renamed to 'DM<cntr>_<channel name>'
              (*<cntr>* is the data group index starting from 0)

            * `parquet` : export to Apache parquet format

            * `asc` : Vector ASCII format for bus logging

                .. versionadded:: 7.3.3

        filename : str | path-like, optional
            Export file name.

        Other Parameters
        ----------------
        single_time_base : bool, default False
            Resample all channels to common time base.
        raster : float | array-like | str, optional
            Time raster for resampling. Valid if `single_time_base` is True.
            It can be:

            * a float step value
            * a channel name whose timestamps will be used as raster (starting
              with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            See `resample` for examples of using this argument.

        time_from_zero : bool, default True
            Adjust time channel to start from 0.
        use_display_names : bool, default True
            Use display name instead of standard channel name, if available.
        empty_channels : {'skip', 'zeros'}, default 'skip'
            Behaviour for channels without samples.
        format : {'5', '4', '7.3'}, default '5'
            Only valid for *mat* export.
        oned_as : {'row', 'column'}, default 'row'
            Only valid for *mat* export.
        reduce_memory_usage : bool, default False
            Reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can represent the values found
            in integer columns.
        compression : str | bool, optional
            Compression to be used.

            * for `parquet` : 'GZIP', 'SNAPPY' or 'LZ4'
            * for `hfd5` : 'gzip', 'lzf' or 'szip'
            * for `mat` : bool

            .. versionadded:: 8.1.0

                Added LZ4 compression after changing to pyarrow.

        time_as_date : bool, default False
            Export time as local timezone datetime; only valid for CSV export.

            .. versionadded:: 5.8.0

        ignore_value2text_conversions : bool, default False
            Valid only for the channels that have value to text conversions and
            if `raw=False`. If this is True, then the raw numeric values will
            be used, and the conversion will not be applied.

            .. versionadded:: 5.8.0

        raw : bool, default False
            Export all channels using the raw values.

            .. versionadded:: 6.0.0

        delimiter : str, default ','
            Only valid for CSV: see cpython documentation for csv.Dialect.delimiter.

            .. versionadded:: 6.2.0

        doublequote : bool, default True
            Only valid for CSV: see cpython documentation for csv.Dialect.doublequote.

            .. versionadded:: 6.2.0

        escapechar : str, default '"'
            Only valid for CSV: see cpython documentation for csv.Dialect.escapechar.

            .. versionadded:: 6.2.0

        lineterminator : str, default '\\r\\n'
            Only valid for CSV: see cpython documentation for csv.Dialect.lineterminator.

            .. versionadded:: 6.2.0

        quotechar : str, default '"'
            Only valid for CSV: see cpython documentation for csv.Dialect.quotechar.

            .. versionadded:: 6.2.0

        quoting : str, default 'MINIMAL'
            Only valid for CSV: see cpython documentation for csv.Dialect.quoting.
            Use the last part of the quoting constant name.

            .. versionadded:: 6.2.0

        add_units : bool, default False
            Only valid for CSV: add the channel units on the second row of the
            CSV file.

            .. versionadded:: 7.1.0
        """

        header_items = (
            "date",
            "time",
            "author_field",
            "department_field",
            "project_field",
            "subject_field",
        )

        fmt = typing.cast(Literal["asc", "csv", "hdf5", "mat", "parquet"], fmt.lower())

        if filename is None:
            message = "Must specify filename for export if MDF was created without a file name"
            logger.warning(message)
            return None

        single_time_base = kwargs.get("single_time_base", False)
        raster = kwargs.get("raster", None)
        time_from_zero = kwargs.get("time_from_zero", True)
        use_display_names = kwargs.get("use_display_names", True)
        empty_channels = kwargs.get("empty_channels", "skip")
        format = kwargs.get("format", "5")
        oned_as = kwargs.get("oned_as", "row")
        reduce_memory_usage = kwargs.get("reduce_memory_usage", False)
        compression = kwargs.get("compression")
        time_as_date = kwargs.get("time_as_date", False)
        ignore_value2text_conversions = kwargs.get("ignore_value2text_conversions", False)
        raw = bool(kwargs.get("raw", False))

        if isinstance(compression, str) and compression.lower() == "snappy":
            try:
                import snappy  # noqa: F401
            except ImportError:
                logger.warning("snappy compressor is not installed; compression will be set to gzip")
                compression = "gzip"

        filename = Path(filename) if filename else self.name

        if fmt == "parquet":
            try:
                import pyarrow as pa
                from pyarrow.parquet import write_table as write_parquet

            except ImportError:
                logger.warning("pyarrow not found; export to parquet is unavailable")
                return None

        elif fmt == "hdf5":
            try:
                from h5py import File as HDF5
            except ImportError:
                logger.warning("h5py not found; export to HDF5 is unavailable")
                return None

        elif fmt == "mat":
            if format == "7.3":
                try:
                    from hdf5storage import savemat
                except ImportError:
                    logger.warning("hdf5storage not found; export to mat v7.3 is unavailable")
                    return None
            else:
                try:
                    from scipy.io import savemat
                except ImportError:
                    logger.warning("scipy not found; export to mat v4 and v5 is unavailable")
                    return None

        elif fmt not in ("csv", "asc"):
            raise MdfException(f"Export to {fmt} is not implemented")

        if progress is not None:
            if callable(progress):
                progress(0, 100)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(100)

                if progress.stop:
                    raise Terminated

        if fmt == "asc":
            self._asc_export(filename.with_suffix(".asc"))
            return None

        if single_time_base or fmt == "parquet":
            df = self.to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                use_display_names=use_display_names,
                empty_channels=empty_channels,
                reduce_memory_usage=reduce_memory_usage,
                ignore_value2text_conversions=ignore_value2text_conversions,
                raw=raw,
                numeric_1D_only=fmt == "parquet",
            )
            units: dict[Hashable, str] = {}
            comments: dict[Hashable, str] = {}
            used_names = UniqueDB()

            groups_nr = len(self.groups)
            if progress is not None:
                if callable(progress):
                    progress(0, groups_nr * 2)
                else:
                    progress.signals.setMaximum.emit(groups_nr * 2)

                    if progress.stop:
                        raise Terminated

            for i, grp in enumerate(self.groups):
                grp = typing.cast(mdf_v3.Group | mdf_v4.Group, grp)
                if progress is not None and progress.stop:
                    raise Terminated

                for ch in grp.channels:
                    if use_display_names:
                        channel_name = list(ch.display_names)[0] if ch.display_names else ch.name
                    else:
                        channel_name = ch.name

                    channel_name = used_names.get_unique_name(channel_name)

                    if hasattr(ch, "unit"):
                        unit = ch.unit
                        if ch.conversion:
                            unit = unit or ch.conversion.unit
                    else:
                        unit = ""
                    comment = ch.comment

                    units[channel_name] = unit
                    comments[channel_name] = comment

                if progress is not None:
                    if callable(progress):
                        progress(i + 1, groups_nr * 2)
                    else:
                        progress.signals.setValue.emit(i + 1)

                        if progress.stop:
                            raise Terminated

        if fmt == "hdf5":
            filename = filename.with_suffix(".hdf")

            if single_time_base:
                with HDF5(str(filename), "w") as hdf:
                    # header information
                    group = hdf.create_group(str(filename))

                    if self.version in MDF2_VERSIONS + MDF3_VERSIONS:
                        for item in header_items:
                            group.attrs[item] = getattr(self.header, item).replace(b"\0", b"")

                    # save each data group in a HDF5 group called
                    # "DataGroup_<cntr>" with the index starting from 1
                    # each HDF5 group will have a string attribute "master"
                    # that will hold the name of the master channel

                    count = len(df.columns)

                    if progress is not None:
                        if callable(progress):
                            progress(0, count * 2)
                        else:
                            progress.signals.setValue.emit(0)
                            progress.signals.setMaximum.emit(count * 2)

                            if progress.stop:
                                raise Terminated

                    samples: NDArray[Any] | pd.Series[Any]
                    for i, channel in enumerate(df):
                        samples = df[channel]
                        unit = units.get(channel, "")
                        comment = comments.get(channel, "")

                        if samples.dtype.kind == "O":
                            if isinstance(samples[0], np.ndarray):
                                samples = np.vstack(list(samples))
                            else:
                                continue

                        if compression:
                            dataset = group.create_dataset(channel, data=samples, compression=compression)
                        else:
                            dataset = group.create_dataset(channel, data=samples)
                        unit = unit.replace("\0", "")
                        if unit:
                            dataset.attrs["unit"] = unit
                        comment = comment.replace("\0", "")
                        if comment:
                            dataset.attrs["comment"] = comment

                        if progress is not None:
                            if callable(progress):
                                progress(i + 1, count * 2)
                            else:
                                progress.signals.setValue.emit(i + 1)

                                if progress.stop:
                                    raise Terminated

            else:
                with HDF5(str(filename), "w") as hdf:
                    # header information
                    group = hdf.create_group(str(filename))

                    if self.version in MDF2_VERSIONS + MDF3_VERSIONS:
                        for item in header_items:
                            group.attrs[item] = getattr(self.header, item).replace(b"\0", b"")

                    # save each data group in a HDF5 group called
                    # "DataGroup_<cntr>" with the index starting from 1
                    # each HDF5 group will have a string attribute "master"
                    # that will hold the name of the master channel

                    groups_nr = len(self.virtual_groups)

                    if progress is not None:
                        if callable(progress):
                            progress(0, groups_nr)
                        else:
                            progress.signals.setValue.emit(0)
                            progress.signals.setMaximum.emit(groups_nr)

                            if progress.stop:
                                raise Terminated

                    for i, (group_index, virtual_group) in enumerate(self.virtual_groups.items()):
                        included_channels = self.included_channels(group_index)[group_index]

                        if not included_channels:
                            continue

                        unique_names = UniqueDB()
                        if progress is not None and progress.stop:
                            raise Terminated

                        if len(virtual_group.groups) == 1:
                            comment = self.groups[virtual_group.groups[0]].channel_group.comment
                        else:
                            comment = "Virtual group i"

                        group_name = r"/" + f"ChannelGroup_{i}"
                        group = hdf.create_group(group_name)

                        group.attrs["comment"] = comment

                        master_index = self.masters_db.get(group_index, -1)

                        if master_index >= 0:
                            group.attrs["master"] = self.groups[group_index].channels[master_index].name
                            master = self._mdf.get(group.attrs["master"], group_index)
                            if reduce_memory_usage:
                                master.timestamps = downcast(master.timestamps)
                            if compression:
                                dataset = group.create_dataset(
                                    group.attrs["master"],
                                    data=master.timestamps,
                                    compression=compression,
                                )
                            else:
                                dataset = group.create_dataset(
                                    group.attrs["master"],
                                    data=master.timestamps,
                                    dtype=master.timestamps.dtype,
                                )
                            unit = master.unit.replace("\0", "")
                            if unit:
                                dataset.attrs["unit"] = unit
                            comment = master.comment.replace("\0", "")
                            if comment:
                                dataset.attrs["comment"] = comment

                        channels = [
                            (None, gp_index, ch_index)
                            for gp_index, channel_indexes in included_channels.items()
                            for ch_index in channel_indexes
                        ]

                        if not channels:
                            continue

                        signals = self.select(channels, raw=raw)

                        for j, sig in enumerate(signals):
                            if use_display_names:
                                name = list(sig.display_names)[0] if sig.display_names else sig.name
                            else:
                                name = sig.name
                            name = name.replace("\\", "_").replace("/", "_")
                            name = unique_names.get_unique_name(name)
                            if reduce_memory_usage:
                                sig.samples = downcast(sig.samples)
                            if compression:
                                dataset = group.create_dataset(name, data=sig.samples, compression=compression)
                            else:
                                dataset = group.create_dataset(name, data=sig.samples, dtype=sig.samples.dtype)
                            unit = sig.unit.replace("\0", "")
                            if unit:
                                dataset.attrs["unit"] = unit
                            comment = sig.comment.replace("\0", "")
                            if comment:
                                dataset.attrs["comment"] = comment

                        if progress is not None:
                            if callable(progress):
                                progress(i + 1, groups_nr)
                            else:
                                progress.signals.setValue.emit(i + 1)

                                if progress.stop:
                                    raise Terminated

        elif fmt == "csv":
            delimiter = kwargs.get("delimiter", ",")[0]
            if delimiter == "\\t":
                delimiter = "\t"

            doublequote = kwargs.get("doublequote", True)
            lineterminator = kwargs.get("lineterminator", "\r\n")
            quotechar = kwargs.get("quotechar", '"')[0]

            quoting_name = kwargs.get("quoting", "MINIMAL").upper()
            quoting: Literal[0, 1, 2, 3, 4, 5] = getattr(csv, f"QUOTE_{quoting_name}")

            escapechar = kwargs.get("escapechar", '"')
            if escapechar is not None:
                escapechar = escapechar[0]

            if single_time_base:
                filename = filename.with_suffix(".csv")
                message = f'Writing csv export to file "{filename}"'
                logger.info(message)

                if time_as_date:
                    index = (
                        pd.to_datetime(df.index + self.header.start_time.timestamp(), unit="s")
                        .tz_localize("UTC")
                        .tz_convert(LOCAL_TIMEZONE)
                        .astype(str)
                    )
                    df.index = index
                    df.index.name = "timestamps"

                    units["timestamps"] = ""
                else:
                    units["timestamps"] = "s"

                if hasattr(self, "can_logging_db") and self.can_logging_db:
                    dropped = {}

                    for name_ in df.columns:
                        if name_.endswith("CAN_DataFrame.ID"):
                            dropped[name_] = pd.Series(
                                csv_int2hex(df[name_].astype(np.dtype("<u4")) & 0x1FFFFFFF),
                                index=df.index,
                            )

                        elif name_.endswith("CAN_DataFrame.DataBytes"):
                            dropped[name_] = pd.Series(csv_bytearray2hex(df[name_]), index=df.index)

                    df = df.drop(columns=list(dropped))
                    for name, s in dropped.items():
                        df[name] = s

                with open(filename, "w", newline="") as csvfile:
                    writer = csv.writer(
                        csvfile,
                        delimiter=delimiter,
                        quotechar=quotechar,
                        escapechar=escapechar,
                        doublequote=doublequote,
                        lineterminator=lineterminator,
                        quoting=quoting,
                    )

                    names_row = [df.index.name, *df.columns]
                    writer.writerow(names_row)

                    if kwargs.get("add_units", False):
                        units_row = [units[name] for name in names_row]
                        writer.writerow(units_row)

                    for col in df:
                        if df[col].dtype.kind == "S":
                            for encoding, errors in (
                                ("utf-8", "strict"),
                                ("latin-1", "strict"),
                                ("utf-8", "replace"),
                                ("latin-1", "replace"),
                            ):
                                try:
                                    df[col] = df[col] = df[col].str.decode(encoding, errors)
                                    break
                                except:
                                    continue

                    vals: list[object]
                    if reduce_memory_usage:
                        vals = [df.index, *(df[name] for name in df)]
                    else:
                        vals = [
                            df.index.to_list(),
                            *(df[name].to_list() for name in df),
                        ]
                    count = len(df.index)

                    if progress is not None:
                        if callable(progress):
                            progress(0, count)
                        else:
                            progress.signals.setValue.emit(0)
                            progress.signals.setMaximum.emit(count)

                            if progress.stop:
                                raise Terminated

                    for i, row in enumerate(zip(*vals, strict=False)):
                        writer.writerow(row)

                        if progress is not None:
                            if callable(progress):
                                progress(i + 1, count)
                            else:
                                progress.signals.setValue.emit(i + 1)
                                if progress.stop:
                                    raise Terminated

            else:
                add_units = kwargs.get("add_units", False)

                filename = filename.with_suffix(".csv")

                gp_count = len(self.virtual_groups)

                if progress is not None:
                    if callable(progress):
                        progress(0, gp_count)
                    else:
                        progress.signals.setValue.emit(0)
                        progress.signals.setMaximum.emit(gp_count)

                        if progress.stop:
                            raise Terminated

                for i, (group_index, virtual_group) in enumerate(self.virtual_groups.items()):
                    if progress is not None and progress.stop:
                        raise Terminated

                    message = f"Exporting group {i + 1} of {gp_count}"
                    logger.info(message)

                    if len(virtual_group.groups) == 1:
                        comment = self.groups[virtual_group.groups[0]].channel_group.comment
                    else:
                        comment = ""

                    if comment:
                        for char in '\n\t\r\b <>\\/:"?*|':
                            comment = comment.replace(char, "_")
                        group_csv_name = filename.parent / f"{filename.stem}.ChannelGroup_{i}_{comment}.csv"
                    else:
                        group_csv_name = filename.parent / f"{filename.stem}.ChannelGroup_{i}.csv"

                    df = self.get_group(
                        group_index,
                        raster=raster,
                        time_from_zero=time_from_zero,
                        use_display_names=use_display_names,
                        reduce_memory_usage=reduce_memory_usage,
                        ignore_value2text_conversions=ignore_value2text_conversions,
                        raw=raw,
                    )

                    if add_units:
                        units = {}
                        used_names = UniqueDB()

                        for gp_index, channel_indexes in self.included_channels(group_index)[group_index].items():
                            for ch_index in channel_indexes:
                                ch = self.groups[gp_index].channels[ch_index]

                                if use_display_names:
                                    channel_name = list(ch.display_names)[0] if ch.display_names else ch.name
                                else:
                                    channel_name = ch.name

                                channel_name = used_names.get_unique_name(channel_name)

                                if hasattr(ch, "unit"):
                                    unit = ch.unit
                                    if ch.conversion:
                                        unit = unit or ch.conversion.unit
                                else:
                                    unit = ""

                                units[channel_name] = unit
                    else:
                        units = {}

                    if time_as_date:
                        index = (
                            pd.to_datetime(df.index + self.header.start_time.timestamp(), unit="s")
                            .tz_localize("UTC")
                            .tz_convert(LOCAL_TIMEZONE)
                            .astype(str)
                        )
                        df.index = index
                        df.index.name = "timestamps"

                        units["timestamps"] = ""
                    else:
                        units["timestamps"] = "s"

                    with open(group_csv_name, "w", newline="") as csvfile:
                        writer = csv.writer(
                            csvfile,
                            delimiter=delimiter,
                            quotechar=quotechar,
                            escapechar=escapechar,
                            doublequote=doublequote,
                            lineterminator=lineterminator,
                            quoting=quoting,
                        )

                        if hasattr(self, "can_logging_db") and self.can_logging_db:
                            dropped = {}

                            for name_ in df.columns:
                                if name_.endswith("CAN_DataFrame.ID"):
                                    dropped[name_] = pd.Series(
                                        csv_int2hex(df[name_] & 0x1FFFFFFF),
                                        index=df.index,
                                    )

                                elif name_.endswith("CAN_DataFrame.DataBytes"):
                                    dropped[name_] = pd.Series(csv_bytearray2hex(df[name_]), index=df.index)

                            df = df.drop(columns=list(dropped))
                            for name_, s in dropped.items():
                                df[name_] = s

                        names_row = [df.index.name, *df.columns]
                        writer.writerow(names_row)

                        if add_units:
                            units_row = [units[name] for name in names_row]
                            writer.writerow(units_row)

                        if reduce_memory_usage:
                            vals = [df.index, *(df[name] for name in df)]
                        else:
                            vals = [
                                df.index.to_list(),
                                *(df[name].to_list() for name in df),
                            ]

                        for i, row in enumerate(zip(*vals, strict=False)):
                            writer.writerow(row)

                    if progress is not None:
                        if callable(progress):
                            progress(i + 1, gp_count)
                        else:
                            progress.signals.setValue.emit(i + 1)

                            if progress.stop:
                                raise Terminated

        elif fmt == "mat":
            filename = filename.with_suffix(".mat")

            if not single_time_base:

                def decompose(samples: NDArray[Any]) -> dict[str, NDArray[Any]]:
                    dct: dict[str, NDArray[Any]] = {}

                    for name in samples.dtype.names or ():
                        vals = samples[name]

                        if vals.dtype.names:
                            dct.update(decompose(vals))
                        else:
                            dct[name] = vals

                    return dct

                mdict: dict[str, NDArray[Any]] = {}

                master_name_template = "DGM{}_{}"
                channel_name_template = "DG{}_{}"
                used_names = UniqueDB()

                groups_nr = len(self.virtual_groups)

                if progress is not None:
                    if callable(progress):
                        progress(0, groups_nr)
                    else:
                        progress.signals.setValue.emit(0)
                        progress.signals.setMaximum.emit(groups_nr + 1)

                        if progress.stop:
                            raise Terminated

                for i, (group_index, virtual_group) in enumerate(self.virtual_groups.items()):
                    if progress is not None and progress.stop:
                        raise Terminated

                    included_channels = self.included_channels(group_index)[group_index]

                    if not included_channels:
                        continue

                    channels = [
                        (None, gp_index, ch_index)
                        for gp_index, channel_indexes in included_channels.items()
                        for ch_index in channel_indexes
                    ]

                    if not channels:
                        continue

                    signals = self.select(
                        channels,
                        ignore_value2text_conversions=ignore_value2text_conversions,
                        raw=raw,
                    )

                    master = signals[0].copy()
                    master.samples = master.timestamps

                    signals.insert(0, master)

                    for j, sig in enumerate(signals):
                        if j == 0:
                            channel_name = master_name_template.format(i, "timestamps")
                        else:
                            if use_display_names:
                                channel_name = list(sig.display_names)[0] if sig.display_names else sig.name
                            else:
                                channel_name = sig.name
                            channel_name = channel_name_template.format(i, channel_name)

                        channel_name = matlab_compatible(channel_name)
                        channel_name = used_names.get_unique_name(channel_name)

                        if names := sig.samples.dtype.names:
                            sig.samples.dtype.names = tuple(matlab_compatible(name) for name in names)

                            sigs = decompose(sig.samples)

                            sigs = {
                                channel_name_template.format(i, channel_name): _v for channel_name, _v in sigs.items()
                            }

                            mdict.update(sigs)

                        else:
                            mdict[channel_name] = sig.samples

                    if progress is not None:
                        if callable(progress):
                            progress(i + 1, groups_nr)
                        else:
                            progress.signals.setValue.emit(i + 1)

                            if progress.stop:
                                raise Terminated

            else:
                used_names = UniqueDB()
                mdict = {}

                count = len(df.columns)

                if progress is not None:
                    if callable(progress):
                        progress(0, count)
                    else:
                        progress.signals.setValue.emit(0)
                        progress.signals.setMaximum.emit(count)

                        if progress.stop:
                            raise Terminated

                for i, name in enumerate(df.columns):
                    channel_name = matlab_compatible(name)
                    channel_name = used_names.get_unique_name(channel_name)

                    mdict[channel_name] = df[name].to_numpy()

                    if hasattr(mdict[channel_name].dtype, "categories"):
                        mdict[channel_name] = np.array(mdict[channel_name], dtype="S")

                    if progress is not None:
                        if callable(progress):
                            progress(i + 1, groups_nr)
                        else:
                            progress.signals.setValue.emit(i + 1)
                            progress.signals.setMaximum.emit(count)

                            if progress.stop:
                                raise Terminated

                mdict["timestamps"] = df.index.values

            if progress is not None:
                if callable(progress):
                    progress(80, 100)
                else:
                    progress.signals.setValue.emit(0)
                    progress.signals.setMaximum.emit(100)
                    progress.signals.setValue.emit(80)

                    if progress.stop:
                        raise Terminated

            if format == "7.3":
                savemat(
                    str(filename),
                    mdict,
                    long_field_names=True,
                    format="7.3",
                    delete_unused_variables=False,
                    oned_as=oned_as,
                    structured_numpy_ndarray_as_struct=True,
                    store_python_metadata=False,
                )
            else:
                savemat(
                    str(filename),
                    mdict,
                    long_field_names=True,
                    oned_as=oned_as,
                    do_compression=bool(compression),
                )

            if progress is not None:
                if callable(progress):
                    progress(100, 100)
                else:
                    progress.signals.setValue.emit(100)

                    if progress.stop:
                        raise Terminated

        elif fmt == "parquet":
            filename = filename.with_suffix(".parquet")
            table = pa.table(df)
            if compression:
                write_parquet(table, filename, compression=compression)  # type: ignore[arg-type]
            else:
                write_parquet(table, filename)

        else:
            message = 'Unsupported export type "{}". Please select "csv", "excel", "hdf5", "mat" or "pandas"'
            message.format(fmt)
            logger.warning(message)

        return None

    def filter(
        self,
        channels: ChannelsType,
        version: str | Version | None = None,
        progress: Any | None = None,
    ) -> "MDF":
        """Return new `MDF` object that contains only the channels listed in the
        `channels` argument.

        Parameters
        ----------
        channels : list
            List of items to be selected; each item can be:

            * a channel name string
            * (channel name, group index, channel index) list or tuple
            * (channel name, group index) list or tuple
            * (None, group index, channel index) list or tuple

        version : str, optional
            New MDF file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default is None
            and in this case the original file version is used.

        Returns
        -------
        mdf : MDF
            New `MDF` object.

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> t = np.arange(5)
        >>> s = np.ones(5)
        >>> mdf = MDF()
        >>> mdf.configure(raise_on_multiple_occurrences=False)
        >>> for i in range(4):
        ...     sigs = [Signal(s * (i * 10 + j), t, name='SIG') for j in range(1, 4)]
        ...     mdf.append(sigs)

        Select channel "SIG" (the first occurrence, which is group 0 index 1),
        channel "SIG" from group 3 index 1, channel "SIG" from group 2 (the
        first occurrence, which is index 1), and channel from group 1 index 2.

        >>> filtered = mdf.filter(['SIG', ('SIG', 3, 1), ['SIG', 2], (None, 1, 2)])
        >>> for gp_nr, ch_nr in filtered.channels_db['SIG']:
        ...     print(filtered.get(group=gp_nr, index=ch_nr))
        <Signal SIG:
                samples=[ 1.  1.  1.  1.  1.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">
        <Signal SIG:
                samples=[ 31.  31.  31.  31.  31.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">
        <Signal SIG:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">
        <Signal SIG:
                samples=[ 12.  12.  12.  12.  12.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">
        """
        if version is None:
            version = self.version
        else:
            version = validate_version_argument(version)

        _raise_on_multiple_occurrences = self._mdf._raise_on_multiple_occurrences
        self._mdf._raise_on_multiple_occurrences = False

        names_map = {}
        for item in channels:
            name: str | None
            if isinstance(item, str):
                entry = self._mdf._validate_channel_selection(item)
                name = item
            else:
                if name := item[0]:
                    entry = self._mdf._validate_channel_selection(*item)
                else:
                    continue
            names_map[entry] = name
        self._mdf._raise_on_multiple_occurrences = _raise_on_multiple_occurrences

        # group channels by group index
        gps = self.included_channels(channels=channels)

        mdf = MDF(
            version=version,
            **self._mdf._kwargs,
        )

        mdf.configure(from_other=self)
        mdf.header.start_time = self.header.start_time

        groups_nr = len(gps)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

                if progress.stop:
                    raise Terminated

        for i, (group_index, groups) in enumerate(gps.items()):
            for idx, sigs in enumerate(self._mdf._yield_selected_signals(group_index, groups=groups, version=version)):
                if not sigs:
                    break

                if idx == 0:
                    sigs = typing.cast(list[Signal], sigs)
                    if sigs:
                        for sig in sigs:
                            entry = sig.group_index, sig.channel_index
                            if entry in names_map:
                                sig.name = names_map[entry]
                        cg = self.groups[group_index].channel_group
                        cg_nr = mdf.append(
                            sigs,
                            common_timebase=True,
                            comment=cg.comment,
                            acq_name=getattr(cg, "acq_name", None),
                            acq_source=getattr(cg, "acq_source", None),
                        )
                        MDF._transfer_channel_group_data(mdf.groups[cg_nr].channel_group, cg)
                    else:
                        break

                else:
                    sigs = typing.cast(list[tuple[NDArray[Any], None]], sigs)
                    mdf.extend(cg_nr, sigs)

                if progress and progress.stop:
                    raise Terminated

            if progress is not None:
                if callable(progress):
                    progress(i + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1)

                    if progress.stop:
                        raise Terminated

        mdf._transfer_metadata(self, message=f"Filtered from {self.name}")

        return mdf

    @overload
    def iter_get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: float | None = ...,
        samples_only: Literal[False] = ...,
        raw: bool = ...,
    ) -> Iterator[Signal]: ...

    @overload
    def iter_get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: float | None = ...,
        *,
        samples_only: Literal[True],
        raw: bool = ...,
    ) -> Iterator[tuple[NDArray[Any], NDArray[Any] | None]]: ...

    @overload
    def iter_get(
        self,
        name: str | None = ...,
        group: int | None = ...,
        index: int | None = ...,
        raster: float | None = ...,
        samples_only: bool = ...,
        raw: bool = ...,
    ) -> Iterator[Signal] | Iterator[tuple[NDArray[Any], NDArray[Any] | None]]: ...

    def iter_get(
        self,
        name: str | None = None,
        group: int | None = None,
        index: int | None = None,
        raster: float | None = None,
        samples_only: bool = False,
        raw: bool = False,
    ) -> Iterator[Signal] | Iterator[tuple[NDArray[Any], NDArray[Any] | None]]:
        """Iterator over a channel.

        This is usefull in case of large files with a small number of channels.

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
        raw : bool, default False
            Return channel samples without applying the conversion rule.
        """

        gp_nr, ch_nr = self._mdf._validate_channel_selection(name, group, index)

        grp = self.groups[gp_nr]

        data = self._mdf._load_data(grp)  # type: ignore[arg-type]

        for fragment in data:
            yield self.get(
                group=gp_nr,
                index=ch_nr,
                raster=raster,
                samples_only=samples_only,
                ignore_invalidation_bits=samples_only,
                data=fragment,
                raw=raw,
            )

    @staticmethod
    def concatenate(
        files: Sequence[Union["MDF", FileLike, StrPath]],
        version: str | Version = "4.10",
        sync: bool = True,
        add_samples_origin: bool = False,
        direct_timestamp_continuation: bool = False,
        progress: Any | None = None,
        **kwargs: Unpack[_ConcatenateKwargs],
    ) -> "MDF":
        """Concatenate several files. The files must have the same internal
        structure (same number of groups, and same channels in each group).

        The order of the input files is always preserved, only the samples'
        timestamps are influenced by the `sync` argument.

        Parameters
        ----------
        files : list | tuple
            List of MDF file names or `MDF`, zipfile.ZipFile, bz2.BZ2File or
            gzip.GzipFile instances.

            .. versionchanged:: 6.2.0

                Added support for zipfile.ZipFile, bz2.BZ2File and gzip.GzipFile.

        version : str, default '4.10'
            Merged file version.
        sync : bool, default True
            Sync the files based on the start of measurement. The order of the
            input files is preserved, only the samples' timestamps are
            influenced by this argument.
        add_samples_origin : bool, default False
            Option to create a new "__samples_origin" channel that will hold
            the index of the measurement from where each timestamp originated.
        direct_timestamp_continuation : bool, default False
            The timestamps from the next file will be added right after the last
            timestamp from the previous file.

            .. versionadded:: 6.0.0

        use_display_names : bool, default False
            Use display name instead of standard channel name, if available.
        process_bus_logging : bool, default True
            Controls whether the bus processing of MDF v4 files is done when the
            file is loaded.

            .. versionadded:: 8.1.0

        Returns
        -------
        concatenated : MDF
            New `MDF` object with concatenated channels.

        Raises
        ------
        MdfException
            If there are inconsistencies between the files.

        Examples
        --------
        >>> conc = MDF.concatenate(
            [
                'path/to/file.mf4',
                MDF(BytesIO(data)),
                MDF(zipfile.ZipFile('data.zip')),
                MDF(bz2.BZ2File('path/to/data.bz2', 'rb')),
                MDF(gzip.GzipFile('path/to/data.gzip', 'rb')),
            ],
            version='4.00',
            sync=False,
        )
        """

        if not files:
            raise MdfException("No files given for merge")

        if progress is not None:
            if callable(progress):
                progress(0, 100)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(100)

                progress.signals.setWindowTitle.emit("Concatenating measurements")

                if progress.stop:
                    raise Terminated

        mdf_nr = len(files)
        use_display_names = kwargs.get("use_display_names", False)

        input_types = [isinstance(mdf, MDF) for mdf in files]

        versions: list[str] = []
        if sync:
            start_times: list[datetime] = []
            for file in files:
                if isinstance(file, MDF):
                    start_times.append(file.header.start_time)
                    versions.append(file.version)
                else:
                    if is_file_like(file):
                        ts, version = get_measurement_timestamp_and_version(file)
                        start_times.append(ts)
                        versions.append(version)
                    else:
                        with open(file, "rb") as bytes_io:
                            ts, version = get_measurement_timestamp_and_version(bytes_io)
                            start_times.append(ts)
                            versions.append(version)

            try:
                oldest = min(start_times)
            except TypeError:
                start_times = [timestamp.astimezone(timezone.utc) for timestamp in start_times]
                oldest = min(start_times)

            offsets = [(timestamp - oldest).total_seconds() for timestamp in start_times]
            offsets = [max(0, offset) for offset in offsets]

        else:
            file = files[0]
            if isinstance(file, MDF):
                timestamp = file.header.start_time
                version = file.version
            else:
                if is_file_like(file):
                    timestamp, version = get_measurement_timestamp_and_version(file)
                else:
                    with open(file, "rb") as bytes_io:
                        timestamp, version = get_measurement_timestamp_and_version(bytes_io)

            oldest = timestamp
            versions.append(version)

            offsets = [0 for _ in files]

        included_channel_names: list[list[str]] = []
        cg_map: dict[int, int] = {}

        if add_samples_origin:
            dict_conversion: dict[str, object] = {}
            for i, file in enumerate(files):
                dict_conversion[f"val_{i}"] = i
                dict_conversion[f"text_{i}"] = str(file._mdf.original_name if isinstance(file, MDF) else str(file))
            origin_conversion = from_dict(dict_conversion)

        for mdf_index, (offset, file) in enumerate(zip(offsets, files, strict=False)):
            if not isinstance(file, MDF):
                mdf = MDF(file, use_display_names=use_display_names)
                close = True
            else:
                mdf = file
                close = False

            if progress is not None and not callable(progress):
                progress.signals.setLabelText.emit(
                    f"Concatenating the file {mdf_index + 1} of {mdf_nr}\n{mdf._mdf.original_name}"
                )

            if mdf_index == 0:
                version = validate_version_argument(version)
                first_version = mdf.version

                kwargs = dict(mdf._mdf._kwargs)  # type: ignore[assignment]

                merged = MDF(
                    version=version,
                    **mdf._mdf._kwargs.copy(),
                )

                merged.configure(from_other=mdf)

                merged.header.start_time = oldest

            reorder_channel_groups = False
            cg_translations: dict[int, int | None] = {}

            vlsd_max_length: dict[tuple[int, str], int] = {}

            if mdf_index == 0:
                last_timestamps = [None for gp in mdf.virtual_groups]
                groups_nr = len(last_timestamps)
                first_mdf = mdf

                if progress is not None:
                    if callable(progress):
                        progress(0, groups_nr * mdf_nr)
                    else:
                        progress.signals.setValue.emit(0)
                        progress.signals.setMaximum.emit(groups_nr * mdf_nr)

                        if progress.stop:
                            raise Terminated

                if isinstance(first_mdf._mdf, mdf_v4.MDF4):
                    w_mdf = first_mdf

                    vlds_channels: list[tuple[str, int, int]] = []

                    for _gp_idx, _gp in enumerate(first_mdf._mdf.groups):
                        for _ch_idx, _ch in enumerate(_gp.channels):
                            if _ch.channel_type == v4c.CHANNEL_TYPE_VLSD:
                                vlds_channels.append((_ch.name, _gp_idx, _ch_idx))

                                vlsd_max_length[(_gp_idx, _ch.name)] = 0

                    if vlsd_max_length:
                        for i, _file in enumerate(files):
                            if not isinstance(_file, MDF):
                                _close = True
                                _file = MDF(_file)
                            else:
                                _close = False

                            _file._mdf.determine_max_vlsd_sample_size.cache_clear()

                            for _ch_name, _gp_idx, _ch_idx in vlds_channels:
                                key = (_gp_idx, _ch_name)
                                for _second_gp_idx, _second_ch_idx in w_mdf.whereis(_ch_name):
                                    if _second_gp_idx == _gp_idx:
                                        vlsd_max_length[key] = max(
                                            vlsd_max_length[key],
                                            _file._mdf.determine_max_vlsd_sample_size(_second_gp_idx, _second_ch_idx),
                                        )
                                        break
                                else:
                                    raise MdfException(
                                        f"internal structure of file {i} is different; different channels"
                                    )

                            if _close:
                                _file.close()

            else:
                if len(mdf.virtual_groups) != groups_nr:
                    raise MdfException(
                        f"internal structure of file <{mdf.name}> is different; different channel groups count"
                    )
                else:
                    cg_translations = dict.fromkeys(range(groups_nr))

                    make_translation = False

                    # check if the order of the channel groups is the same
                    for i, group_index in enumerate(mdf.virtual_groups):
                        included_channels = mdf.included_channels(group_index)[group_index]
                        names = [
                            mdf.groups[gp_index].channels[ch_index].name
                            for gp_index, channels in included_channels.items()
                            for ch_index in channels
                        ]

                        if names != included_channel_names[i]:
                            if sorted(names) != sorted(included_channel_names[i]):
                                make_translation = reorder_channel_groups = True
                                break

                    # Make a channel group translation dictionary if the order is different
                    if make_translation:
                        first_mdf._mdf = typing.cast(mdf_v4.MDF4, first_mdf._mdf)
                        mdf._mdf = typing.cast(mdf_v4.MDF4, mdf._mdf)
                        for i, org_group in enumerate(first_mdf._mdf.groups):
                            org_group_source = org_group.channel_group.acq_source
                            for j, new_group in enumerate(mdf._mdf.groups):
                                new_group_source = new_group.channel_group.acq_source
                                if (
                                    new_group.channel_group.acq_name == org_group.channel_group.acq_name
                                    and (new_group_source and org_group_source)
                                    and new_group_source.name == org_group_source.name
                                    and new_group_source.path == org_group_source.path
                                    and new_group.channel_group.samples_byte_nr
                                    == org_group.channel_group.samples_byte_nr
                                ):
                                    new_included_channels = mdf.included_channels(j)[j]

                                    new_names = [
                                        mdf.groups[gp_index].channels[ch_index].name
                                        for gp_index, channels in new_included_channels.items()
                                        for ch_index in channels
                                    ]

                                    if sorted(new_names) == sorted(included_channel_names[i]):
                                        cg_translations[i] = j
                                        break

            for i, group_index in enumerate(mdf.virtual_groups):
                # save original group index for extension
                # replace with the translated group index
                if reorder_channel_groups:
                    cg_trans = typing.cast(dict[int, int], cg_translations)
                    origin_gp_idx = group_index
                    group_index = cg_trans[group_index]

                included_channels = mdf.included_channels(group_index)[group_index]

                if mdf_index == 0:
                    included_channel_names.append(
                        [
                            mdf.groups[gp_index].channels[ch_index].name
                            for gp_index, channels in included_channels.items()
                            for ch_index in channels
                        ]
                    )
                    different_channel_order = False
                else:
                    names = [
                        mdf.groups[gp_index].channels[ch_index].name
                        for gp_index, channels in included_channels.items()
                        for ch_index in channels
                    ]
                    different_channel_order = False
                    if names != included_channel_names[i]:
                        if sorted(names) != sorted(included_channel_names[i]):
                            raise MdfException(
                                f"internal structure of file {mdf_index} is different; different channels"
                            )
                        else:
                            original_names = included_channel_names[i]
                            different_channel_order = True
                            remap = [original_names.index(name) for name in names]

                if not included_channels:
                    continue

                last_timestamp = last_timestamps[i]
                first_timestamp = None
                original_first_timestamp = None

                mdf._mdf.vlsd_max_length.clear()
                mdf._mdf.vlsd_max_length.update(vlsd_max_length)

                for idx, signals in enumerate(mdf._mdf._yield_selected_signals(group_index, groups=included_channels)):
                    if not signals:
                        break
                    if mdf_index == 0 and idx == 0:
                        signals = typing.cast(list[Signal], signals)
                        first_signal = signals[0]
                        if len(first_signal):
                            if offset > 0:
                                timestamps = first_signal.timestamps + offset
                                for sig in signals:
                                    sig.timestamps = timestamps
                            last_timestamp = first_signal.timestamps[-1]
                            first_timestamp = first_signal.timestamps[0]
                            original_first_timestamp = first_timestamp

                        if add_samples_origin:
                            signals.append(
                                Signal(
                                    samples=np.ones(len(first_signal), dtype="<u2") * mdf_index,
                                    timestamps=first_signal.timestamps,
                                    conversion=origin_conversion,
                                    name="__samples_origin",
                                )
                            )

                        cg = mdf.groups[group_index].channel_group
                        cg_nr = merged.append(
                            signals,
                            common_timebase=True,
                        )
                        MDF._transfer_channel_group_data(merged.groups[cg_nr].channel_group, cg)
                        cg_map[group_index] = cg_nr

                    else:
                        if different_channel_order:
                            new_signals = [
                                typing.cast(Signal | tuple[NDArray[Any], None] | None, None) for _ in signals
                            ]
                            if idx == 0:
                                signals = typing.cast(list[Signal], signals)
                                for new_index, sig in zip(remap, signals, strict=False):
                                    new_signals[new_index] = sig
                            else:
                                signals = typing.cast(list[tuple[NDArray[Any], None]], signals)
                                for new_index, signal_samples in zip(remap, signals[1:], strict=False):
                                    new_signals[new_index + 1] = signal_samples
                                new_signals[0] = signals[0]

                            signals = typing.cast(list[Signal] | list[tuple[NDArray[Any], None]], new_signals)

                        if idx == 0:
                            signals = typing.cast(list[Signal], signals)
                            signals_samples = [(signals[0].timestamps, typing.cast(NDArray[np.bool] | None, None))] + [
                                (sig.samples, sig.invalidation_bits) for sig in signals
                            ]
                        else:
                            signals_samples = typing.cast(list[tuple[NDArray[Any], NDArray[np.bool] | None]], signals)

                        master = signals_samples[0][0]
                        _copied = False

                        if len(master):
                            if original_first_timestamp is None:
                                original_first_timestamp = master[0]
                            if offset > 0:
                                master = master + offset
                                _copied = True
                            if last_timestamp is None:
                                last_timestamp = master[-1]
                            else:
                                if last_timestamp >= master[0] or direct_timestamp_continuation:
                                    if len(master) >= 2:
                                        delta = master[1] - master[0]
                                    else:
                                        delta = 0.001
                                    if _copied:
                                        master -= master[0]
                                    else:
                                        master = master - master[0]
                                        _copied = True
                                    master += last_timestamp + delta
                                last_timestamp = master[-1]

                            signals_samples[0] = master, None

                            if add_samples_origin:
                                signals_samples.append(
                                    (
                                        np.ones(len(master), dtype="<u2") * mdf_index,
                                        None,
                                    )
                                )
                            cg_nr = cg_map[group_index]
                            # set the original channel group number back for extension
                            if reorder_channel_groups:
                                cg_nr = cg_map[origin_gp_idx]
                            merged.extend(cg_nr, signals_samples)

                            if first_timestamp is None:
                                first_timestamp = master[0]

                    if progress and progress.stop:
                        raise Terminated

                last_timestamps[i] = last_timestamp

            if mdf_index == 0:
                merged._transfer_metadata(mdf)

            if progress is not None:
                if callable(progress):
                    progress(i + 1 + mdf_index * groups_nr, mdf_nr * groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1 + mdf_index * groups_nr)

                    if progress.stop:
                        raise Terminated

            if close and mdf_index:
                mdf.close()

        if not isinstance(files[0], MDF):
            first_mdf.close()

        try:
            if kwargs.get("process_bus_logging", True):
                if not isinstance(merged._mdf, mdf_v4.MDF4):
                    raise MdfException("Bus logging processing is only available for MDF4 files")
                merged._mdf._process_bus_logging()
        except:
            pass

        return merged

    @staticmethod
    def stack(
        files: Sequence[Union["MDF", FileLike, StrPath]],
        version: str | Version = "4.10",
        sync: bool = True,
        progress: Any | None = None,
        **kwargs: Unpack[_StackKwargs],
    ) -> "MDF":
        """Stack several files and return the stacked `MDF` object.

        Parameters
        ----------
        files : list | tuple
            List of MDF file names or `MDF`, zipfile.ZipFile, bz2.BZ2File or
            gzip.GzipFile instances.

            .. versionchanged:: 6.2.0

                Added support for zipfile.ZipFile, bz2.BZ2File and gzip.GzipFile.

        version : str, default '4.10'
            Merged file version.
        sync : bool, default True
            Sync the files based on the start of measurement.
        use_display_names : bool, default False
            Use display name instead of standard channel name, if available.
        process_bus_logging : bool, default True
            Controls whether the bus processing of MDF v4 files is done when the
            file is loaded.

            .. versionadded:: 8.1.0

        Returns
        -------
        stacked : MDF
            New `MDF` object with stacked channels.

        Examples
        --------
        >>> stacked = MDF.stack(
            [
                'path/to/file.mf4',
                MDF(BytesIO(data)),
                MDF(zipfile.ZipFile('data.zip')),
                MDF(bz2.BZ2File('path/to/data.bz2', 'rb')),
                MDF(gzip.GzipFile('path/to/data.gzip', 'rb')),
            ],
            version='4.00',
            sync=False,
        )
        """
        if not files:
            raise MdfException("No files given for stack")

        version = validate_version_argument(version)

        use_display_names = kwargs.get("use_display_names", False)

        files_nr = len(files)

        input_types = [isinstance(mdf, MDF) for mdf in files]

        if progress is not None:
            if callable(progress):
                progress(0, files_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(files_nr)

                if progress.stop:
                    raise Terminated

        if sync:
            start_times: list[datetime] = []
            for file in files:
                if isinstance(file, MDF):
                    start_times.append(file.header.start_time)
                else:
                    if is_file_like(file):
                        ts, version = get_measurement_timestamp_and_version(file)
                        start_times.append(ts)
                    else:
                        with open(file, "rb") as bytes_io:
                            ts, version = get_measurement_timestamp_and_version(bytes_io)
                            start_times.append(ts)

            try:
                oldest = min(start_times)
            except TypeError:
                start_times = [timestamp.astimezone(timezone.utc) for timestamp in start_times]
                oldest = min(start_times)

            offsets = [(timestamp - oldest).total_seconds() for timestamp in start_times]

        else:
            offsets = [0 for file in files]

        for mdf_index, (offset, mdf) in enumerate(zip(offsets, files, strict=False)):
            if not isinstance(mdf, MDF):
                mdf = MDF(mdf, use_display_names=use_display_names)

            if progress is not None:
                progress.signals.setLabelText.emit(f"Stacking file {mdf_index + 1} of {files_nr}\n{mdf.name.name}")

            if mdf_index == 0:
                version = validate_version_argument(version)

                kwargs = dict(mdf._mdf._kwargs)  # type: ignore[assignment]

                stacked = MDF(
                    version=version,
                    **mdf._mdf._kwargs.copy(),
                )

                stacked.configure(from_other=mdf)

                if sync:
                    stacked.header.start_time = oldest
                else:
                    stacked.header.start_time = mdf.header.start_time

            for i, group in enumerate(mdf.virtual_groups):
                included_channels = mdf.included_channels(group)[group]
                if not included_channels:
                    continue

                for idx, signals in enumerate(
                    mdf._mdf._yield_selected_signals(group, groups=included_channels, version=version)
                ):
                    if not signals:
                        break
                    if idx == 0:
                        signals = typing.cast(list[Signal], signals)
                        if sync:
                            timestamps = signals[0].timestamps + offset
                            for sig in signals:
                                sig.timestamps = timestamps
                        cg = mdf.groups[group].channel_group
                        dg_cntr = stacked.append(
                            signals,
                            common_timebase=True,
                        )
                        MDF._transfer_channel_group_data(stacked.groups[dg_cntr].channel_group, cg)
                    else:
                        signals_samples = typing.cast(list[tuple[NDArray[Any], None]], signals)
                        master = signals_samples[0][0]
                        if sync:
                            master = master + offset
                            signals_samples[0] = master, None

                        stacked.extend(dg_cntr, signals_samples)

                    if progress and progress.stop:
                        raise Terminated

                if dg_cntr is not None:
                    for index in range(dg_cntr, len(stacked.groups)):
                        stacked.groups[
                            index
                        ].channel_group.comment = f'stacked from channel group {i} of "{mdf.name.parent}"'

            if progress is not None:
                if callable(progress):
                    progress(mdf_index, files_nr)
                else:
                    progress.signals.setValue.emit(mdf_index)

                    if progress.stop:
                        raise Terminated

            if mdf_index == 0:
                stacked._transfer_metadata(mdf)

            if not input_types[mdf_index]:
                mdf.close()

            if progress is not None and progress.stop:
                raise Terminated

        try:
            if kwargs.get("process_bus_logging", True):
                if not isinstance(stacked._mdf, mdf_v4.MDF4):
                    raise MdfException("process_bus_logging is only supported for MDF4 files")
                stacked._mdf._process_bus_logging()
        except:
            pass

        return stacked

    def iter_channels(
        self,
        skip_master: bool = True,
        copy_master: bool = True,
        raw: bool | dict[str, bool] = False,
    ) -> Iterator[Signal]:
        """Generator that yields a `Signal` for each non-master channel.

        Parameters
        ----------
        skip_master : bool, default True
            Do not yield master channels.
        copy_master : bool, default True
            Copy master for each yielded channel.
        raw : bool | dict, default False
            Return raw channels instead of converted.

            .. versionchanged:: 8.0.0

                Provide individual raw mode based on a dict. The dict keys are
                channel names and each value is a boolean that sets whether to
                return raw samples for that channel. The key '__default__' is
                mandatory and sets the raw mode for all channels not specified.
        """

        if isinstance(raw, dict):
            if "__default__" not in raw:
                raise MdfException("The raw argument given as dict must contain the __default__ key")

        for index in self.virtual_groups:
            channels = [
                (None, gp_index, ch_index)
                for gp_index, channel_indexes in self.included_channels(index)[index].items()
                for ch_index in channel_indexes
            ]

            signals = self.select(channels, copy_master=copy_master, raw=raw)

            yield from signals

    def iter_groups(
        self,
        raster: RasterType | None = None,
        time_from_zero: bool = True,
        empty_channels: EmptyChannelsType = "skip",
        use_display_names: bool = False,
        time_as_date: bool = False,
        reduce_memory_usage: bool = False,
        raw: bool | dict[str, bool] = False,
        ignore_value2text_conversions: bool = False,
        only_basenames: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Generator that yields channel groups as pandas DataFrames. If there
        are multiple occurrences for the same channel name inside a channel
        group, then a counter will be used to make the names unique
        (<original_name>_<counter>).

        Parameters
        ----------
        raster : float | array-like | str, optional
            New raster that can be:

            * a float step value
            * a channel name whose timestamps will be used as raster (starting
              with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            See `resample` for examples of using this argument.

            .. versionadded:: 5.21.0

        time_from_zero : bool, default True
            Adjust time channel to start from 0.

        empty_channels : {'skip', 'zeros'}, default 'skip'
            Behaviour for channels without samples.

            .. versionadded:: 5.21.0

        use_display_names : bool, default False
            Use display name instead of standard channel name, if available.

            .. versionadded:: 5.21.0

        time_as_date : bool, default False
            The DataFrame index will contain the datetime timestamps according
            to the measurement start time. If True, then the argument
            `time_from_zero` will be ignored.
        reduce_memory_usage : bool, default False
            Reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can represent the values found
            in integer columns.

            .. versionadded:: 5.21.0

        raw : bool | dict, default False
            The DataFrame will contain the raw channel values.

            .. versionadded:: 5.21.0

            .. versionchanged:: 8.0.0

                Provide individual raw mode based on a dict. The dict keys are
                channel names and each value is a boolean that sets whether to
                return raw samples for that channel. The key '__default__' is
                mandatory and sets the raw mode for all channels not specified.

        ignore_value2text_conversions : bool, default False
            Valid only for the channels that have value to text conversions and
            if `raw=False`. If this is True, then the raw numeric values will
            be used, and the conversion will not be applied.

            .. versionadded:: 5.21.0

        only_basenames : bool, default False
            Use just the field names, without prefix, for structures and channel
            arrays.

            .. versionadded:: 5.21.0
        """

        for i in self.virtual_groups:
            yield self.get_group(
                i,
                raster=raster,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                use_display_names=use_display_names,
                time_as_date=time_as_date,
                reduce_memory_usage=reduce_memory_usage,
                raw=raw,
                ignore_value2text_conversions=ignore_value2text_conversions,
                only_basenames=only_basenames,
            )

    def master_using_raster(self, raster: RasterType, endpoint: bool = False) -> NDArray[Any]:
        """Get single master based on the raster.

        Parameters
        ----------
        raster : float
            New raster.
        endpoint : bool, default False
            Include maximum timestamp in the new master.

        Returns
        -------
        master : np.ndarray
            New master.
        """
        if not raster:
            master = np.array([], dtype="<f8")
        else:
            t_min_list: list[float] = []
            t_max_list: list[float] = []
            for group_index in self.virtual_groups:
                group = self.groups[group_index]
                cycles_nr = group.channel_group.cycles_nr
                if cycles_nr:
                    master_min = self.get_master(group_index, record_offset=0, record_count=1)
                    if len(master_min):
                        t_min_list.append(master_min[0])
                    master_max = self.get_master(group_index, record_offset=cycles_nr - 1, record_count=1)
                    if len(master_max):
                        t_max_list.append(master_max[0])

            if t_min_list:
                t_min = np.amin(t_min_list)
                t_max = np.amax(t_max_list)

                num = float(np.float64((t_max - t_min) / raster))
                if num.is_integer():
                    master = np.linspace(t_min, t_max, int(num) + 1)
                else:
                    master = np.arange(t_min, t_max, raster)
                    if endpoint:
                        master = np.concatenate([master, [t_max]])

            else:
                master = np.array([], dtype="<f8")

        return master

    def resample(
        self,
        raster: RasterType,
        version: str | Version | None = None,
        time_from_zero: bool = False,
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> "MDF":
        """Resample all channels using the given raster. See `configure` to
        select the interpolation method for integer and float channels.

        Parameters
        ----------
        raster : float | array-like | str
            New raster that can be:

            * a float step value
            * a channel name whose timestamps will be used as raster (starting
              with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

        version : str, optional
            New MDF file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default is None
            and in this case the original file version is used.

        time_from_zero : bool, default False
            Start timestamps from 0s in the resampled measurement.

        Returns
        -------
        mdf : MDF
            New `MDF` object with resampled channels.

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> mdf = MDF()
        >>> sig = Signal(name='S1', samples=[1, 2, 3, 4], timestamps=[1, 2, 3, 4])
        >>> mdf.append(sig)
        >>> sig = Signal(name='S2', samples=[1, 2, 3, 4], timestamps=[1.1, 3.5, 3.7, 3.9])
        >>> mdf.append(sig)

        Resample to a float step value.

        >>> resampled = mdf.resample(raster=0.1)
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 4]
                timestamps=[1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7
          2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4. ]
                unit=""
                comment="">,
         <Signal S2:
                samples=[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 3 3 4 4]
                timestamps=[1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7
          2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4. ]
                unit=""
                comment="">
        ]

        Resample to the timestamps of one of the channels.

        >>> resampled = mdf.resample(raster='S2')
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 3 3 3]
                timestamps=[1.1 3.5 3.7 3.9]
                unit=""
                comment="">,
         <Signal S2:
                samples=[1 2 3 4]
                timestamps=[1.1 3.5 3.7 3.9]
                unit=""
                comment="">
        ]

        Resample to an arbitrary array of timestamps.

        >>> resampled = mdf.resample(raster=[1.9, 2.0, 2.1])
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 2 2]
                timestamps=[1.9 2.  2.1]
                unit=""
                comment="">,
         <Signal S2:
                samples=[1 1 1]
                timestamps=[1.9 2.  2.1]
                unit=""
                comment="">
        ]

        Resample to the timestamps of one of the channels, and adjust the
        timestamps to start at 0.

        >>> resampled = mdf.resample(raster='S2', time_from_zero=True)
        >>> resampled.select(['S1', 'S2'])
        [<Signal S1:
                samples=[1 3 3 3]
                timestamps=[0.  2.4 2.6 2.8]
                unit=""
                comment="">,
         <Signal S2:
                samples=[1 2 3 4]
                timestamps=[0.  2.4 2.6 2.8]
                unit=""
                comment="">
        ]
        """

        if version is None:
            version = self.version
        else:
            version = validate_version_argument(version)

        mdf = MDF(
            version=version,
            **self._mdf._kwargs,
        )

        integer_interpolation_mode = self._mdf._integer_interpolation
        float_interpolation_mode = self._mdf._float_interpolation
        mdf.configure(from_other=self)

        mdf.header.start_time = self.header.start_time

        groups_nr = len(self.virtual_groups)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

                if progress.stop:
                    raise Terminated

        if isinstance(raster, (int, float)):
            raster = float(raster)
            if raster <= 0:
                raise MdfException("The raster value must be > 0")
            raster = self.master_using_raster(raster)
        elif isinstance(raster, str):
            raster = self._mdf.get(raster, raw=True, ignore_invalidation_bits=True).timestamps
        else:
            raster = np.array(raster)

        if time_from_zero and len(raster):
            delta = raster[0]
            new_raster = raster - delta
            t_epoch = self.header.start_time.timestamp() + delta
            mdf.header.start_time = datetime.fromtimestamp(t_epoch)
        else:
            new_raster = None
            mdf.header.start_time = self.header.start_time

        for i, (group_index, virtual_group) in enumerate(self.virtual_groups.items()):
            channels = [
                (None, gp_index, ch_index)
                for gp_index, channel_indexes in self.included_channels(group_index)[group_index].items()
                for ch_index in channel_indexes
            ]

            if not channels:
                continue

            sigs = self.select(channels, raw=True)

            sigs = [
                sig.interp(
                    raster,
                    integer_interpolation_mode=integer_interpolation_mode,
                    float_interpolation_mode=float_interpolation_mode,
                )
                for sig in sigs
            ]

            if new_raster is not None:
                for sig in sigs:
                    if len(sig):
                        sig.timestamps = new_raster

            cg = self.groups[group_index].channel_group
            dg_cntr = mdf.append(
                sigs,
                common_timebase=True,
                comment=cg.comment,
            )
            MDF._transfer_channel_group_data(mdf.groups[dg_cntr].channel_group, cg)

            if progress is not None:
                if callable(progress):
                    progress(i + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1)

                    if progress.stop:
                        raise Terminated

        mdf._transfer_metadata(self, message=f"Resampled from {self.name}")

        return mdf

    def select(
        self,
        channels: ChannelsType,
        record_offset: int = 0,
        raw: bool | dict[str, bool] = False,
        copy_master: bool = True,
        ignore_value2text_conversions: bool = False,
        record_count: int | None = None,
        validate: bool = False,
    ) -> list[Signal]:
        """Retrieve the channels listed in the `channels` argument as `Signal`
        objects.

        .. note:: The `dataframe` argument was removed in version 5.8.0,
                  use the `to_dataframe` method instead.

        Parameters
        ----------
        channels : list
            List of items to be selected; each item can be:

            * a channel name string
            * (channel name, group index, channel index) list or tuple
            * (channel name, group index) list or tuple
            * (None, group index, channel index) list or tuple

        record_offset : int, optional
            Record number offset; optimization to get the last part of signal
            samples.
        raw : bool | dict, default False
            Get raw channel samples.

            .. versionchanged:: 8.0.0

                Provide individual raw mode based on a dict. The dict keys are
                channel names and each value is a boolean that sets whether to
                return raw samples for that channel. The key '__default__' is
                mandatory and sets the raw mode for all channels not specified.

        copy_master : bool, default True
            Option to get a new timestamps array for each selected Signal or to
            use a shared array for channels of the same channel group.
        ignore_value2text_conversions : bool, default False
            Valid only for the channels that have value to text conversions and
            if `raw=False`. If this is True, then the raw numeric values will
            be used, and the conversion will not be applied.

            .. versionchanged:: 5.8.0

        record_count : int, optional
            Number of records to read; default is None and in this case all
            available records are used.
        validate : bool, default False
            Consider the invalidation bits.

            .. versionadded:: 5.16.0

        Returns
        -------
        signals : list
            List of `Signal` objects based on the input channel list.

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> t = np.arange(5)
        >>> s = np.ones(5)
        >>> mdf = MDF()
        >>> mdf.configure(raise_on_multiple_occurrences=False)
        >>> for i in range(4):
        ...     sigs = [Signal(s * (i * 10 + j), t, name='SIG') for j in range(1, 4)]
        ...     mdf.append(sigs)

        Select channel "SIG" (the first occurrence, which is group 0 index 1),
        channel "SIG" from group 3 index 1, channel "SIG" from group 2 (the
        first occurrence, which is index 1), and channel from group 1 index 2.

        >>> mdf.select(['SIG', ('SIG', 3, 1), ['SIG', 2], (None, 1, 2)])
        [<Signal SIG:
                samples=[ 1.  1.  1.  1.  1.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">,
         <Signal SIG:
                samples=[ 31.  31.  31.  31.  31.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">,
         <Signal SIG:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">,
         <Signal SIG:
                samples=[ 12.  12.  12.  12.  12.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">
        ]
        """

        def validate_blocks(blocks: list[SignalDataBlockInfo], record_size: int) -> bool:
            for block in blocks:
                if block.original_size % record_size:
                    return False

            return True

        if (
            not isinstance(self._mdf, mdf_v4.MDF4)
            or not self._mdf._mapped_file
            or record_offset
            or record_count is not None
            or True  # disable for now
        ):
            return self._select_fallback(
                channels, record_offset, raw, copy_master, ignore_value2text_conversions, record_count, validate
            )

        if isinstance(raw, dict):
            if "__default__" not in raw:
                raise MdfException("The raw argument given as dict must contain the __default__ key")

            __default__ = raw["__default__"]
        else:
            __default__ = raw

        virtual_groups = self.included_channels(channels=channels, minimal=False, skip_master=False)
        for virtual_group, groups in virtual_groups.items():
            if len(self._mdf.virtual_groups[virtual_group].groups) > 1:
                return self._select_fallback(
                    channels, record_offset, raw, copy_master, ignore_value2text_conversions, record_count, validate
                )

        output_signals: dict[tuple[int, int], Signal] = {}

        for virtual_group, groups in virtual_groups.items():
            group_index = virtual_group
            grp = self._mdf.groups[group_index]
            grp.load_all_data_blocks()
            blocks = grp.data_blocks
            record_size = grp.channel_group.samples_byte_nr + grp.channel_group.invalidation_bytes_nr
            cycles_nr = grp.channel_group.cycles_nr
            channel_indexes = groups[group_index]

            pairs = [(group_index, ch_index) for ch_index in channel_indexes]

            master_index = self.masters_db.get(group_index, None)
            if master_index is None or grp.record[master_index] is None:
                return self._select_fallback(
                    channels, record_offset, raw, copy_master, ignore_value2text_conversions, record_count, validate
                )

            channel = grp.channels[master_index]
            master_dtype, byte_size, byte_offset, _ = typing.cast(
                tuple[np.dtype[Any], int, int, int], grp.record[master_index]
            )
            signals = [(byte_offset, byte_size, channel.pos_invalidation_bit)]

            for ch_index in channel_indexes:
                channel = grp.channels[ch_index]

                if (info := grp.record[ch_index]) is None:
                    print("NASOl")
                    return self._select_fallback(
                        channels, record_offset, raw, copy_master, ignore_value2text_conversions, record_count, validate
                    )
                else:
                    _, byte_size, byte_offset, _ = info
                    signals.append((byte_offset, byte_size, channel.pos_invalidation_bit))

            raw_and_invalidation = get_channel_raw_bytes_complete(
                blocks,
                signals,
                self._mdf._mapped_file.name,
                cycles_nr,
                record_size,
                grp.channel_group.invalidation_bytes_nr,
                THREAD_COUNT,
            )
            master_bytes, _ = raw_and_invalidation[0]
            raw_and_invalidation = raw_and_invalidation[1:]

            # prepare the master
            master = np.frombuffer(master_bytes, dtype=master_dtype)

            for pair, (raw_data, invalidation_bits) in zip(pairs, raw_and_invalidation, strict=False):
                ch_index = pair[-1]
                channel = grp.channels[ch_index]
                channel_dtype, byte_size, byte_offset, bit_offset = grp.record[ch_index]
                vals = np.frombuffer(raw_data, dtype=channel_dtype)

                data_type = channel.data_type

                if not channel.standard_C_size:
                    size = byte_size

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

                    if channel.bit_count != size * 8:
                        if data_type in v4c.SIGNED_INT:
                            vals = as_non_byte_sized_signed_int(vals, channel.bit_count)
                        else:
                            mask = (1 << channel.bit_count) - 1
                            vals &= mask
                    elif data_type in v4c.SIGNED_INT:
                        view = f"{channel_dtype.byteorder}i{vals.itemsize}"
                        if np.dtype(view) != vals.dtype:
                            vals = vals.view(view)

                conversion = channel.conversion
                unit = (conversion and conversion.unit) or channel.unit

                source = channel.source

                if source:
                    source = Source.from_source(source)
                else:
                    cg_source = grp.channel_group.acq_source
                    if cg_source:
                        source = Source.from_source(cg_source)
                    else:
                        source = None

                master_metadata = self._mdf._master_channel_metadata.get(group_index, None)

                output_signals[pair] = Signal(
                    samples=vals,
                    timestamps=master,
                    unit=unit,
                    name=channel.name,
                    comment=channel.comment,
                    conversion=conversion,
                    raw=True,
                    master_metadata=master_metadata,
                    attachment=None,
                    source=source,
                    display_names=channel.display_names,
                    bit_count=channel.bit_count,
                    flags=Signal.Flags.no_flags,
                    invalidation_bits=invalidation_bits,
                    encoding=None,
                    group_index=group_index,
                    channel_index=ch_index,
                )

        indexes = []

        for item in channels:
            if not isinstance(item, (list, tuple)):
                item = [item]
            indexes.append(self._mdf._validate_channel_selection(*item))

        signals = [output_signals[pair] for pair in indexes]

        if copy_master:
            for signal in signals:
                signal.timestamps = signal.timestamps.copy()

        for signal in signals:
            if (isinstance(raw, dict) and not raw.get(signal.name, __default__)) or not raw:
                conversion = signal.conversion
                if conversion:
                    samples = conversion.convert(
                        signal.samples, ignore_value2text_conversions=ignore_value2text_conversions
                    )
                    signal.samples = samples

                signal.raw = False
                signal.conversion = None
                if signal.samples.dtype.kind == "S":
                    signal.encoding = "utf-8" if self.version >= "4.00" else "latin-1"

        if validate:
            signals = [sig.validate(copy=False) for sig in signals]

        for signal, channel in zip(signals, channels, strict=False):
            if isinstance(channel, str):
                signal.name = channel
            else:
                name = channel[0]
                if name is not None:
                    signal.name = name

        unique = set()
        for i, signal in enumerate(signals):
            obj_id = id(signal)
            if id(signal) in unique:
                signals[i] = signal.copy()
            unique.add(obj_id)

        return signals

    def _select_fallback(
        self,
        channels: ChannelsType,
        record_offset: int = 0,
        raw: bool | dict[str, bool] = False,
        copy_master: bool = True,
        ignore_value2text_conversions: bool = False,
        record_count: int | None = None,
        validate: bool = False,
    ) -> list[Signal]:
        """Retrieve the channels listed in the `channels` argument as `Signal`
        objects.

        .. note:: The `dataframe` argument was removed in version 5.8.0,
                  use the `to_dataframe` method instead.

        Parameters
        ----------
        channels : list
            List of items to be selected; each item can be:

            * a channel name string
            * (channel name, group index, channel index) list or tuple
            * (channel name, group index) list or tuple
            * (None, group index, channel index) list or tuple

        record_offset : int, optional
            Record number offset; optimization to get the last part of signal
            samples.
        raw : bool | dict, default False
            Get raw channel samples.

            .. versionchanged:: 8.0.0

                Provide individual raw mode based on a dict. The dict keys are
                channel names and each value is a boolean that sets whether to
                return raw samples for that channel. The key '__default__' is
                mandatory and sets the raw mode for all channels not specified.

        copy_master : bool, default True
            Option to get a new timestamps array for each selected Signal or to
            use a shared array for channels of the same channel group.
        ignore_value2text_conversions : bool, default False
            Valid only for the channels that have value to text conversions and
            if `raw=False`. If this is True, then the raw numeric values will
            be used, and the conversion will not be applied.

            .. versionchanged:: 5.8.0

        record_count : int, optional
            Number of records to read; default is None and in this case all
            available records are used.
        validate : bool, default False
            Consider the invalidation bits.

            .. versionadded:: 5.16.0

        Returns
        -------
        signals : list
            List of `Signal` objects based on the input channel list.

        Examples
        --------
        >>> from asammdf import MDF, Signal
        >>> import numpy as np
        >>> t = np.arange(5)
        >>> s = np.ones(5)
        >>> mdf = MDF()
        >>> mdf.configure(raise_on_multiple_occurrences=False)
        >>> for i in range(4):
        ...     sigs = [Signal(s * (i * 10 + j), t, name='SIG') for j in range(1, 4)]
        ...     mdf.append(sigs)

        Select channel "SIG" (the first occurrence, which is group 0 index 1),
        channel "SIG" from group 3 index 1, channel "SIG" from group 2 (the
        first occurrence, which is index 1), and channel from group 1 index 2.

        >>> mdf.select(['SIG', ('SIG', 3, 1), ['SIG', 2], (None, 1, 2)])
        [<Signal SIG:
                samples=[ 1.  1.  1.  1.  1.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">,
         <Signal SIG:
                samples=[ 31.  31.  31.  31.  31.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">,
         <Signal SIG:
                samples=[ 21.  21.  21.  21.  21.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">,
         <Signal SIG:
                samples=[ 12.  12.  12.  12.  12.]
                timestamps=[ 0.  1.  2.  3.  4.]
                unit=""
                comment="">
        ]
        """

        if isinstance(raw, dict):
            if "__default__" not in raw:
                raise MdfException("The raw argument given as dict must contain the __default__ key")

            __default__ = raw["__default__"]
        else:
            __default_ = raw

        virtual_groups = self.included_channels(channels=channels, minimal=False, skip_master=False)

        output_signals: dict[tuple[int, int], Signal] = {}

        for virtual_group, groups in virtual_groups.items():
            cycles_nr = self._mdf.virtual_groups[virtual_group].cycles_nr
            pairs = [
                (gp_index, ch_index) for gp_index, channel_indexes in groups.items() for ch_index in channel_indexes
            ]

            if record_count is None:
                cycles = cycles_nr - record_offset
            else:
                if cycles_nr < record_count + record_offset:
                    cycles = cycles_nr - record_offset
                else:
                    cycles = record_count

            signals: list[Signal] = []

            current_pos = 0

            for idx, sigs in enumerate(
                self._mdf._yield_selected_signals(
                    virtual_group,
                    groups=groups,
                    record_offset=record_offset,
                    record_count=record_count,
                )
            ):
                if not sigs:
                    break
                if idx == 0:
                    sigs = typing.cast(list[Signal], sigs)
                    next_pos = current_pos + len(sigs[0])

                    master = np.empty(cycles, dtype=sigs[0].timestamps.dtype)
                    master[current_pos:next_pos] = sigs[0].timestamps

                    for sig in sigs:
                        shape = (cycles,) + sig.samples.shape[1:]
                        samples = np.empty(shape, dtype=sig.samples.dtype)
                        samples[current_pos:next_pos] = sig.samples
                        sig.samples = samples
                        signals.append(sig)

                        if sig.invalidation_bits is not None:
                            inval_array = np.empty(cycles, dtype=sig.invalidation_bits.dtype)
                            inval_array[current_pos:next_pos] = sig.invalidation_bits
                            sig.invalidation_bits = InvalidationArray(inval_array)

                else:
                    sigs = typing.cast(list[tuple[NDArray[Any], None]], sigs)
                    samples, _ = sigs[0]
                    next_pos = current_pos + len(samples)
                    master[current_pos:next_pos] = samples

                    for signal, (samples, inval) in zip(signals, sigs[1:], strict=False):
                        signal.samples[current_pos:next_pos] = samples
                        if signal.invalidation_bits is not None:
                            signal.invalidation_bits[current_pos:next_pos] = inval

                current_pos = next_pos

            for signal, pair in zip(signals, pairs, strict=False):
                signal.timestamps = master
                output_signals[pair] = signal

        indexes: list[tuple[int, int]] = []

        for item in channels:
            name: str | None
            if not isinstance(item, (list, tuple)):
                name = item
                group = index = None
            elif len(item) == 2:
                name, group = item
                index = None
            else:
                name, group, index = item
            indexes.append(self._mdf._validate_channel_selection(name=name, group=group, index=index))

        signals = [output_signals[pair] for pair in indexes]

        if copy_master:
            for signal in signals:
                signal.timestamps = signal.timestamps.copy()

        for signal in signals:
            if (isinstance(raw, dict) and not raw.get(signal.name, __default__)) or not raw:
                conversion = signal.conversion
                if conversion:
                    samples = conversion.convert(
                        signal.samples, ignore_value2text_conversions=ignore_value2text_conversions
                    )
                    signal.samples = samples

                signal.raw = False
                signal.conversion = None
                if signal.samples.dtype.kind == "S":
                    signal.encoding = "utf-8" if self.version >= "4.00" else "latin-1"

        if validate:
            signals = [sig.validate(copy=False) for sig in signals]

        unique = set()
        for i, signal in enumerate(signals):
            obj_id = id(signal)
            if id(signal) in unique:
                signals[i] = signal.copy()
            unique.add(obj_id)

        for signal, channel in zip(signals, channels, strict=False):
            if isinstance(channel, str):
                signal.name = channel
            else:
                name = channel[0]
                if name is not None:
                    signal.name = name

        return signals

    @staticmethod
    def scramble(
        name: StrPath,
        skip_attachments: bool = False,
        progress: Callable[[int, int], None] | Any | None = None,
        **kwargs: Never,
    ) -> Path:
        """Scramble text blocks and keep original file structure.

        Parameters
        ----------
        name : str | path-like
            File name.
        skip_attachments : bool, default False
            Skip scrambling of attachments data if True.

            .. versionadded:: 5.9.0

        Returns
        -------
        name : pathlib.Path
            Name of scrambled file.
        """

        name = Path(name)

        mdf = MDF(name)
        texts: dict[int, bytes] = {}

        if progress is not None:
            if callable(progress):
                progress(0, 100)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(100)

                if progress.stop:
                    raise Terminated

        count = len(mdf.groups)

        if isinstance(mdf._mdf, mdf_v4.MDF4):
            try:
                stream = mdf._mdf._file

                if not stream:
                    raise ValueError("stream is None")

                if mdf.header.comment_addr:
                    stream.seek(mdf.header.comment_addr + 8)
                    size = UINT64_u(stream.read(8))[0] - 24
                    texts[mdf.header.comment_addr] = randomized_string(size)

                for fh in mdf._mdf.file_history:
                    addr = fh.comment_addr
                    if addr and addr not in texts:
                        stream.seek(addr + 8)
                        size = UINT64_u(stream.read(8))[0] - 24
                        texts[addr] = randomized_string(size)

                for ev in mdf._mdf.events:
                    for addr in (ev.comment_addr, ev.name_addr):
                        if addr and addr not in texts:
                            stream.seek(addr + 8)
                            size = UINT64_u(stream.read(8))[0] - 24
                            texts[addr] = randomized_string(size)

                for at in mdf._mdf.attachments:
                    for addr in (at.comment_addr, at.file_name_addr):
                        if addr and addr not in texts:
                            stream.seek(addr + 8)
                            size = UINT64_u(stream.read(8))[0] - 24
                            texts[addr] = randomized_string(size)
                    if not skip_attachments and at.embedded_data:
                        texts[at.address + v4c.AT_COMMON_SIZE] = randomized_string(at.embedded_size)

                for idx, v4_gp in enumerate(mdf._mdf.groups, 1):
                    addr = v4_gp.data_group.comment_addr
                    if addr and addr not in texts:
                        stream.seek(addr + 8)
                        size = UINT64_u(stream.read(8))[0] - 24
                        texts[addr] = randomized_string(size)

                    v4_cg = v4_gp.channel_group
                    for addr in (v4_cg.acq_name_addr, v4_cg.comment_addr):
                        if v4_cg.flags & v4c.FLAG_CG_BUS_EVENT:
                            continue

                        if addr and addr not in texts:
                            stream.seek(addr + 8)
                            size = UINT64_u(stream.read(8))[0] - 24
                            texts[addr] = randomized_string(size)

                        source = v4_cg.acq_source_addr
                        if source:
                            source_information = SourceInformation(
                                address=source, stream=stream, mapped=False, tx_map={}
                            )
                            for addr in (
                                source_information.name_addr,
                                source_information.path_addr,
                                source_information.comment_addr,
                            ):
                                if addr and addr not in texts:
                                    stream.seek(addr + 8)
                                    size = UINT64_u(stream.read(8))[0] - 24
                                    texts[addr] = randomized_string(size)

                    for v4_ch in v4_gp.channels:
                        for addr in (v4_ch.name_addr, v4_ch.unit_addr, v4_ch.comment_addr):
                            if addr and addr not in texts:
                                stream.seek(addr + 8)
                                size = UINT64_u(stream.read(8))[0] - 24
                                texts[addr] = randomized_string(size)

                        source = v4_ch.source_addr
                        if source:
                            source_information = SourceInformation(
                                address=source, stream=stream, mapped=False, tx_map={}
                            )
                            for addr in (
                                source_information.name_addr,
                                source_information.path_addr,
                                source_information.comment_addr,
                            ):
                                if addr and addr not in texts:
                                    stream.seek(addr + 8)
                                    size = UINT64_u(stream.read(8))[0] - 24
                                    texts[addr] = randomized_string(size)

                        conv = v4_ch.conversion_addr
                        if conv:
                            v4_conv = v4b.ChannelConversion(
                                address=conv,
                                stream=stream,
                                mapped=False,
                                tx_map={},
                            )
                            for addr in (
                                v4_conv.name_addr,
                                v4_conv.unit_addr,
                                v4_conv.comment_addr,
                            ):
                                if addr and addr not in texts:
                                    stream.seek(addr + 8)
                                    size = UINT64_u(stream.read(8))[0] - 24
                                    texts[addr] = randomized_string(size)
                            if v4_conv.conversion_type == v4c.CONVERSION_TYPE_ALG:
                                addr = v4_conv.formula_addr
                                if addr and addr not in texts:
                                    stream.seek(addr + 8)
                                    size = UINT64_u(stream.read(8))[0] - 24
                                    texts[addr] = randomized_string(size)

                            if v4_conv.referenced_blocks:
                                for key, v4_block in v4_conv.referenced_blocks.items():
                                    if v4_block:
                                        if isinstance(v4_block, bytes):
                                            addr = typing.cast(int, v4_conv[key])
                                            if addr not in texts:
                                                stream.seek(addr + 8)
                                                size = len(v4_block)
                                                texts[addr] = randomized_string(size)

                    if progress is not None:
                        if callable(progress):
                            progress(int(idx / count * 66), 100)
                        else:
                            progress.signals.setValue.emit(int(idx / count * 66))

                            if progress.stop:
                                raise Terminated

            except:
                print(f"Error while scrambling the file: {format_exc()}.\nWill now use fallback method")
                texts = MDF._fallback_scramble_mf4(name)

            mdf.close()

            dst = name.with_suffix(".scrambled.mf4")

            copy(name, dst)

            with open(dst, "rb+") as bytes_io:
                count = len(texts)
                chunk = max(count // 34, 1)
                idx = 0
                for index, (addr, bts) in enumerate(texts.items()):
                    bytes_io.seek(addr + 24)
                    bytes_io.write(bts)
                    if index % chunk == 0:
                        if progress is not None:
                            if callable(progress):
                                progress(66 + idx, 100)
                            else:
                                progress.signals.setValue.emit(66 + idx)

                                if progress.stop:
                                    raise Terminated

        else:
            stream = mdf._mdf._file

            if not stream:
                raise ValueError("stream is None")

            if mdf.header.comment_addr:
                stream.seek(mdf.header.comment_addr + 2)
                size = UINT16_u(stream.read(2))[0] - 4
                texts[mdf.header.comment_addr + 4] = randomized_string(size)
            texts[36 + 0x40] = randomized_string(32)
            texts[68 + 0x40] = randomized_string(32)
            texts[100 + 0x40] = randomized_string(32)
            texts[132 + 0x40] = randomized_string(32)

            for idx, v3_gp in enumerate(mdf._mdf.groups, 1):
                v3_cg = v3_gp.channel_group
                addr = v3_cg.comment_addr

                if addr and addr not in texts:
                    stream.seek(addr + 2)
                    size = UINT16_u(stream.read(2))[0] - 4
                    texts[addr + 4] = randomized_string(size)

                if v3_gp.trigger:
                    addr = v3_gp.trigger.text_addr
                    if addr:
                        stream.seek(addr + 2)
                        size = UINT16_u(stream.read(2))[0] - 4
                        texts[addr + 4] = randomized_string(size)

                for v3_ch in v3_gp.channels:
                    for key in ("long_name_addr", "display_name_addr", "comment_addr"):
                        if hasattr(v3_ch, key):
                            addr = getattr(v3_ch, key)
                        else:
                            addr = 0
                        if addr and addr not in texts:
                            stream.seek(addr + 2)
                            size = UINT16_u(stream.read(2))[0] - 4
                            texts[addr + 4] = randomized_string(size)

                    texts[v3_ch.address + 26] = randomized_string(32)
                    texts[v3_ch.address + 58] = randomized_string(128)

                    source = v3_ch.source_addr
                    if source:
                        channel_extension = ChannelExtension(address=source, stream=stream)
                        if channel_extension.type == v3c.SOURCE_ECU:
                            texts[channel_extension.address + 12] = randomized_string(80)
                            texts[channel_extension.address + 92] = randomized_string(32)
                        else:
                            texts[channel_extension.address + 14] = randomized_string(36)
                            texts[channel_extension.address + 50] = randomized_string(36)

                    conv = v3_ch.conversion_addr
                    if conv:
                        texts[conv + 22] = randomized_string(20)

                        v3_conv = v3b.ChannelConversion(address=conv, stream=stream)

                        if v3_conv.conversion_type == v3c.CONVERSION_TYPE_FORMULA:
                            texts[conv + 36] = randomized_string(v3_conv.block_len - 36)

                        if v3_conv.referenced_blocks:
                            for key, v3_block in v3_conv.referenced_blocks.items():
                                if v3_block:
                                    if isinstance(v3_block, bytes):
                                        addr = typing.cast(int, v3_conv[key])
                                        if addr and addr not in texts:
                                            stream.seek(addr + 2)
                                            size = UINT16_u(stream.read(2))[0] - 4
                                            texts[addr + 4] = randomized_string(size)

                if progress is not None:
                    if callable(progress):
                        progress(int(idx / count * 100), 100)
                    else:
                        progress.signals.setValue.emit(int(idx / count * 66))

                        if progress.stop:
                            raise Terminated

            mdf.close()

            dst = name.with_suffix(".scrambled.mdf")

            copy(name, dst)

            with open(dst, "rb+") as bytes_io:
                chunk = count // 34
                idx = 0
                for index, (addr, bts) in enumerate(texts.items()):
                    bytes_io.seek(addr)
                    bytes_io.write(bts)
                    if chunk and index % chunk == 0:
                        if progress is not None:
                            if callable(progress):
                                progress(66 + idx, 100)
                            else:
                                progress.signals.setValue.emit(66 + idx)

                                if progress.stop:
                                    raise Terminated

        if progress is not None:
            if callable(progress):
                progress(100, 100)
            else:
                progress.signals.setValue.emit(100)

        return dst

    @staticmethod
    def _fallback_scramble_mf4(name: StrPath | bytes | os.PathLike[bytes]) -> dict[int, bytes]:
        """Scramble text blocks and keep original file structure.

        Parameters
        ----------
        name : str | path-like
            File name.

        Returns
        -------
        name : pathlib.Path
            Name of scrambled file.
        """

        pattern = re.compile(
            rb"(?P<block>##(TX|MD))",
            re.DOTALL | re.MULTILINE,
        )

        texts = {}

        with open(name, "rb") as stream:
            stream.seek(0, 2)
            file_limit = stream.tell()
            stream.seek(0)

            for match in re.finditer(pattern, stream.read()):
                start = match.start()

                if file_limit - start >= 24:
                    stream.seek(start + 8)
                    (size,) = UINT64_u(stream.read(8))

                    if start + size <= file_limit:
                        texts[start + 24] = randomized_string(size - 24)

        return texts

    def get_group(
        self,
        index: int,
        raster: RasterType | None = None,
        time_from_zero: bool = True,
        empty_channels: EmptyChannelsType = "skip",
        use_display_names: bool = False,
        time_as_date: bool = False,
        reduce_memory_usage: bool = False,
        raw: bool | dict[str, bool] = False,
        ignore_value2text_conversions: bool = False,
        only_basenames: bool = False,
    ) -> pd.DataFrame:
        """Get channel group as a pandas DataFrame. If there are multiple
        occurrences for the same channel name, then a counter will be used to
        make the names unique (<original_name>_<counter>).

        Parameters
        ----------
        index : int
            Channel group index.
        raster : float | array-like | str, optional
            New raster that can be:

            * a float step value
            * a channel name whose timestamps will be used as raster (starting
              with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            See `resample` for examples of using this argument.

        time_from_zero : bool, default True
            Adjust time channel to start from 0.

        empty_channels : {'skip', 'zeros'}, default 'skip'
            Behaviour for channels without samples.

            .. versionadded:: 5.8.0

        use_display_names : bool, default False
            Use display name instead of standard channel name, if available.
        time_as_date : bool, default False
            The DataFrame index will contain the datetime timestamps according
            to the measurement start time. If True, then the argument
            `time_from_zero` will be ignored.
        reduce_memory_usage : bool, default False
            Reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can represent the values found
            in integer columns.
        raw : bool | dict, default False
            The DataFrame will contain the raw channel values.

            .. versionadded:: 5.7.0

            .. versionchanged:: 8.0.0

                Provide individual raw mode based on a dict. The dict keys are
                channel names and each value is a boolean that sets whether to
                return raw samples for that channel. The key '__default__' is
                mandatory and sets the raw mode for all channels not specified.

        ignore_value2text_conversions : bool, default False
            Valid only for the channels that have value to text conversions and
            if `raw=False`. If this is True, then the raw numeric values will
            be used, and the conversion will not be applied.

            .. versionadded:: 5.8.0

        only_basenames : bool, default False
            Use just the field names, without prefix, for structures and channel
            arrays.

            .. versionadded:: 5.13.0

        Returns
        -------
        dataframe : pandas.DataFrame
            Channel group data.
        """

        channels = [
            (None, gp_index, ch_index)
            for gp_index, channel_indexes in self.included_channels(index)[index].items()
            for ch_index in channel_indexes
        ]

        return self.to_dataframe(
            channels=channels,
            raster=raster,
            time_from_zero=time_from_zero,
            empty_channels=empty_channels,
            use_display_names=use_display_names,
            time_as_date=time_as_date,
            reduce_memory_usage=reduce_memory_usage,
            raw=raw,
            ignore_value2text_conversions=ignore_value2text_conversions,
            only_basenames=only_basenames,
        )

    def iter_to_dataframe(
        self,
        channels: ChannelsType | None = None,
        raster: RasterType | None = None,
        time_from_zero: bool = True,
        empty_channels: EmptyChannelsType = "skip",
        use_display_names: bool = False,
        time_as_date: bool = False,
        reduce_memory_usage: bool = False,
        raw: bool | dict[str, bool] = False,
        ignore_value2text_conversions: bool = False,
        use_interpolation: bool = True,
        only_basenames: bool = False,
        chunk_ram_size: int = 200 * 1024 * 1024,
        interpolate_outwards_with_nan: bool = False,
        numeric_1D_only: bool = False,
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Generator that yields pandas DataFrames that should not exceed
        200 MB of RAM.

        .. versionadded:: 5.15.0

        Parameters
        ----------
        channels : list, optional
            List of items to be selected; each item can be:

            * a channel name string
            * (channel name, group index, channel index) list or tuple
            * (channel name, group index) list or tuple
            * (None, group index, channel index) list or tuple

            The default is to select all channels.

        raster : float | array-like | str, optional
            New raster that can be:

            * a float step value
            * a channel name whose timestamps will be used as raster (starting
              with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            See `resample` for examples of using this argument.

        time_from_zero : bool, default True
            Adjust time channel to start from 0.
        empty_channels : {'skip', 'zeros'}, default 'skip'
            Behaviour for channels without samples.
        use_display_names : bool, default False
            Use display name instead of standard channel name, if available.
        time_as_date : bool, default False
            The DataFrame index will contain the datetime timestamps according
            to the measurement start time. If True, then the argument
            `time_from_zero` will be ignored.
        reduce_memory_usage : bool, default False
            Reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can represent the values found
            in integer columns.
        raw : bool | dict, default False
            The columns will contain the raw values.

            .. versionchanged:: 8.0.0

                Provide individual raw mode based on a dict. The dict keys are
                channel names and each value is a boolean that sets whether to
                return raw samples for that channel. The key '__default__' is
                mandatory and sets the raw mode for all channels not specified.

        ignore_value2text_conversions : bool, default False
            Valid only for the channels that have value to text conversions and
            if `raw=False`. If this is True, then the raw numeric values will
            be used, and the conversion will not be applied.
        use_interpolation : bool, default True
            Option to perform interpolations when multiple timestamp rasters are
            present. If False, then DataFrame columns will be automatically
            filled with NaNs where the DataFrame index values are not found in
            the current column's timestamps.
        only_basenames : bool, default False
            Use just the field names, without prefix, for structures and channel
            arrays.
        chunk_ram_size : int, default 200 * 1024 * 1024 (= 200 MB)
            Desired DataFrame RAM usage in bytes.
        interpolate_outwards_with_nan : bool, default False
            Use NaN values for the samples that lie outside of the original
            signal's timestamps.
        numeric_1D_only : bool, default False
            Only keep the 1D-columns that have numeric values.

            .. versionadded:: 7.0.0


        Yields
        ------
        dataframe : pandas.DataFrame
            Pandas DataFrames that should not exceed 200 MB of RAM.
        """

        if isinstance(raw, dict):
            if "__default__" not in raw:
                raise MdfException("The raw argument given as dict must contain the __default__ key")

            __default__ = raw["__default__"]
        else:
            __default__ = raw

        if channels:
            mdf = self.filter(channels)

            result = mdf.iter_to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                use_display_names=use_display_names,
                time_as_date=time_as_date,
                reduce_memory_usage=reduce_memory_usage,
                raw=raw,
                ignore_value2text_conversions=ignore_value2text_conversions,
                use_interpolation=use_interpolation,
                only_basenames=only_basenames,
                chunk_ram_size=chunk_ram_size,
                interpolate_outwards_with_nan=interpolate_outwards_with_nan,
                numeric_1D_only=numeric_1D_only,
            )

            for df in result:
                yield df

            mdf.close()
        else:
            # channels is None

            self._mdf._set_temporary_master(None)

            masters = {index: self._mdf.get_master(index) for index in self.virtual_groups}

            if raster is not None:
                if isinstance(raster, (int, float)):
                    raster = float(raster)
                    if raster <= 0:
                        raise MdfException("The raster value must be > 0")
                    raster = self.master_using_raster(raster)
                elif isinstance(raster, str):
                    raster = self._mdf.get(raster, raw=True, ignore_invalidation_bits=True).timestamps
                else:
                    raster = np.array(raster)

                master = raster
            else:
                if masters:
                    master = reduce(np.union1d, masters.values())
                else:
                    master = np.array([], dtype="<f4")

            master_ = master
            if time_from_zero and len(master_):
                master_ -= master_[0]
            channel_count = sum(len(gp.channels) - 1 for gp in self.groups) + 1
            # approximation with all float64 dtype
            itemsize = channel_count * 8
            # use 200MB DataFrame chunks
            chunk_count = chunk_ram_size // itemsize or 1

            chunks, r = divmod(len(master), chunk_count)
            if r:
                chunks += 1

            for i in range(chunks):
                master = master_[chunk_count * i : chunk_count * (i + 1)]
                start = master[0]
                end = master[-1]

                data: dict[str, pd.Series[Any]] = {}
                self._mdf._set_temporary_master(None)

                used_names = UniqueDB()
                used_names.get_unique_name("timestamps")

                groups_nr = len(self.virtual_groups)

                if progress is not None:
                    if callable(progress):
                        progress(0, groups_nr)
                    else:
                        progress.signals.setValue.emit(0)
                        progress.signals.setMaximum.emit(groups_nr)

                for group_index, virtual_group in self.virtual_groups.items():
                    group_cycles = virtual_group.cycles_nr
                    if group_cycles == 0 and empty_channels == "skip":
                        continue

                    record_offset = max(np.searchsorted(masters[group_index], start).flatten()[0] - 1, 0)
                    stop = np.searchsorted(masters[group_index], end).flatten()[0]
                    record_count = min(stop - record_offset + 1, group_cycles)

                    channels = [
                        (None, gp_index, ch_index)
                        for gp_index, channel_indexes in self.included_channels(group_index)[group_index].items()
                        for ch_index in channel_indexes
                    ]
                    signals = self.select(
                        channels,
                        raw=True,
                        copy_master=False,
                        record_offset=record_offset,
                        record_count=record_count,
                        validate=False,
                    )

                    if not signals:
                        continue

                    group_master = signals[0].timestamps

                    for sig in signals:
                        if len(sig) == 0:
                            if empty_channels == "zeros":
                                sig.samples = np.zeros(
                                    len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                                    dtype=sig.samples.dtype,
                                )
                                sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

                    for signal in signals:
                        if (isinstance(raw, dict) and not raw.get(signal.name, __default__)) or not raw:
                            conversion = signal.conversion
                            if conversion:
                                samples = conversion.convert(
                                    signal.samples, ignore_value2text_conversions=ignore_value2text_conversions
                                )
                                signal.samples = samples

                            signal.raw = False
                            signal.conversion = None
                            if signal.samples.dtype.kind == "S":
                                signal.encoding = "utf-8" if self.version >= "4.00" else "latin-1"

                    for s_index, sig in enumerate(signals):
                        sig = sig.validate(copy=False)

                        if len(sig) == 0:
                            if empty_channels == "zeros":
                                sig.samples = np.zeros(
                                    len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                                    dtype=sig.samples.dtype,
                                )
                                sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

                        signals[s_index] = sig

                    if use_interpolation:
                        same_master = np.array_equal(master, group_master)

                        if not same_master and interpolate_outwards_with_nan:
                            idx = np.argwhere((master >= group_master[0]) & (master <= group_master[-1])).flatten()

                        cycles = len(group_master)

                        signals = [
                            (
                                signal.interp(
                                    master,
                                    integer_interpolation_mode=self._mdf._integer_interpolation,
                                    float_interpolation_mode=self._mdf._float_interpolation,
                                )
                                if not same_master or len(signal) != cycles
                                else signal
                            )
                            for signal in signals
                        ]

                        if not same_master and interpolate_outwards_with_nan:
                            for sig in signals:
                                sig.timestamps = sig.timestamps[idx]
                                sig.samples = sig.samples[idx]

                        group_master = master

                    signals = [sig for sig in signals if len(sig)]

                    if signals:
                        diffs = np.diff(group_master, prepend=-np.inf) > 0

                        if group_master.dtype.byteorder not in target_byte_order:
                            group_master = group_master.byteswap().view(group_master.dtype.newbyteorder())

                        if np.all(diffs):
                            index = pd.Index(group_master, tupleize_cols=False)

                        else:
                            idx = np.argwhere(diffs).flatten()
                            group_master = group_master[idx]

                            index = pd.Index(group_master, tupleize_cols=False)

                            for sig in signals:
                                sig.samples = sig.samples[idx]
                                sig.timestamps = sig.timestamps[idx]

                    size = len(index)
                    for sig in signals:
                        if sig.timestamps.dtype.byteorder not in target_byte_order:
                            sig.timestamps = sig.timestamps.byteswap().view(sig.timestamps.dtype.newbyteorder())

                        sig_index = index if len(sig) == size else pd.Index(sig.timestamps, tupleize_cols=False)

                        # byte arrays
                        if len(sig.samples.shape) > 1:
                            if use_display_names:
                                channel_name = list(sig.display_names)[0] if sig.display_names else sig.name
                            else:
                                channel_name = sig.name

                            channel_name = used_names.get_unique_name(channel_name)

                            if sig.samples.dtype.byteorder not in target_byte_order:
                                sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                            data[channel_name] = pd.Series(
                                list(sig.samples),
                                index=sig_index,
                            )

                        # arrays and structures
                        elif sig.samples.dtype.names:
                            for name, series in components(
                                sig.samples,
                                sig.name,
                                used_names,
                                master=sig_index,
                                only_basenames=only_basenames,
                            ):
                                data[name] = series

                        # scalars
                        else:
                            if use_display_names:
                                channel_name = list(sig.display_names)[0] if sig.display_names else sig.name
                            else:
                                channel_name = sig.name

                            channel_name = used_names.get_unique_name(channel_name)

                            if reduce_memory_usage and sig.samples.dtype.kind in "SU":
                                unique = np.unique(sig.samples)

                                if sig.samples.dtype.byteorder not in target_byte_order:
                                    sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                                if len(sig.samples) / len(unique) >= 2:
                                    data[channel_name] = pd.Series(
                                        sig.samples,
                                        index=sig_index,
                                        dtype="category",
                                    )
                                else:
                                    data[channel_name] = pd.Series(
                                        sig.samples,
                                        index=sig_index,
                                    )
                            else:
                                if reduce_memory_usage:
                                    sig.samples = downcast(sig.samples)

                                if sig.samples.dtype.byteorder not in target_byte_order:
                                    sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                                data[channel_name] = pd.Series(
                                    sig.samples,
                                    index=sig_index,
                                )

                    if progress is not None:
                        if callable(progress):
                            progress(group_index + 1, groups_nr)
                        else:
                            progress.signals.setValue.emit(group_index + 1)

                strings: dict[str, pd.Series[Any]] = {}
                nonstrings: dict[str, pd.Series[Any]] = {}

                for col, series in data.items():
                    if series.dtype.kind == "S":
                        strings[col] = series
                    else:
                        nonstrings[col] = series

                if numeric_1D_only:
                    nonstrings = {col: series for col, series in nonstrings.items() if series.dtype.kind in "uif"}
                    strings = {}

                df = pd.DataFrame(nonstrings, index=master)

                if strings:
                    df_strings = pd.DataFrame(strings, index=master)
                    df = pd.concat([df, df_strings], axis=1)

                df.index.name = "timestamps"

                if time_as_date:
                    delta = pd.to_timedelta(df.index, unit="s")

                    new_index = self.header.start_time + delta
                    df.set_index(new_index, inplace=True)

                yield df

    @overload
    def to_dataframe(
        self,
        channels: ChannelsType | None = ...,
        raster: RasterType | None = ...,
        time_from_zero: bool = ...,
        empty_channels: EmptyChannelsType = ...,
        use_display_names: bool = ...,
        time_as_date: bool = ...,
        reduce_memory_usage: bool = ...,
        raw: bool | dict[str, bool] = ...,
        ignore_value2text_conversions: bool = ...,
        use_interpolation: bool = ...,
        only_basenames: bool = ...,
        interpolate_outwards_with_nan: bool = ...,
        numeric_1D_only: bool = ...,
        progress: Callable[[int, int], None] | Any | None = ...,
        use_polars: Literal[False] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def to_dataframe(
        self,
        channels: ChannelsType | None = ...,
        raster: RasterType | None = ...,
        time_from_zero: bool = ...,
        empty_channels: EmptyChannelsType = ...,
        use_display_names: bool = ...,
        time_as_date: bool = ...,
        reduce_memory_usage: bool = ...,
        raw: bool | dict[str, bool] = ...,
        ignore_value2text_conversions: bool = ...,
        use_interpolation: bool = ...,
        only_basenames: bool = ...,
        interpolate_outwards_with_nan: bool = ...,
        numeric_1D_only: bool = ...,
        progress: Callable[[int, int], None] | Any | None = ...,
        use_polars: Literal[True] = ...,
    ) -> "pl.DataFrame": ...

    @overload
    def to_dataframe(
        self,
        channels: ChannelsType | None = ...,
        raster: RasterType | None = ...,
        time_from_zero: bool = ...,
        empty_channels: EmptyChannelsType = ...,
        use_display_names: bool = ...,
        time_as_date: bool = ...,
        reduce_memory_usage: bool = ...,
        raw: bool | dict[str, bool] = ...,
        ignore_value2text_conversions: bool = ...,
        use_interpolation: bool = ...,
        only_basenames: bool = ...,
        interpolate_outwards_with_nan: bool = ...,
        numeric_1D_only: bool = ...,
        progress: Callable[[int, int], None] | Any | None = ...,
        use_polars: bool = ...,
    ) -> Union[pd.DataFrame, "pl.DataFrame"]: ...

    def to_dataframe(
        self,
        channels: ChannelsType | None = None,
        raster: RasterType | None = None,
        time_from_zero: bool = True,
        empty_channels: EmptyChannelsType = "skip",
        use_display_names: bool = False,
        time_as_date: bool = False,
        reduce_memory_usage: bool = False,
        raw: bool | dict[str, bool] = False,
        ignore_value2text_conversions: bool = False,
        use_interpolation: bool = True,
        only_basenames: bool = False,
        interpolate_outwards_with_nan: bool = False,
        numeric_1D_only: bool = False,
        progress: Callable[[int, int], None] | Any | None = None,
        use_polars: bool = False,
    ) -> Union[pd.DataFrame, "pl.DataFrame"]:
        """Generate a pandas DataFrame.

        Parameters
        ----------
        channels : list, optional
            List of items to be selected; each item can be:

            * a channel name string
            * (channel name, group index, channel index) list or tuple
            * (channel name, group index) list or tuple
            * (None, group index, channel index) list or tuple

            The default is to select all channels.

        raster : float | array-like | str, optional
            New raster that can be:

            * a float step value
            * a channel name whose timestamps will be used as raster (starting
              with asammdf 5.5.0)
            * an array (starting with asammdf 5.5.0)

            See `resample` for examples of using this argument.

        time_from_zero : bool, default True
            Adjust time channel to start from 0.
        empty_channels : {'skip', 'zeros'}, default 'skip'
            Behaviour for channels without samples.
        use_display_names : bool, default False
            Use display name instead of standard channel name, if available.
        time_as_date : bool, default False
            The DataFrame index will contain the datetime timestamps according
            to the measurement start time. If True, then the argument
            `time_from_zero` will be ignored.
        reduce_memory_usage : bool, default False
            Reduce memory usage by converting all float columns to float32 and
            searching for minimum dtype that can represent the values found
            in integer columns.
        raw : bool | dict, default False
            The columns will contain the raw values.

            .. versionadded:: 5.7.0

            .. versionchanged:: 8.0.0

                Provide individual raw mode based on a dict. The dict keys are
                channel names and each value is a boolean that sets whether to
                return raw samples for that channel. The key '__default__' is
                mandatory and sets the raw mode for all channels not specified.

        ignore_value2text_conversions : bool, default False
            Valid only for the channels that have value to text conversions and
            if `raw=False`. If this is True, then the raw numeric values will
            be used, and the conversion will not be applied.

            .. versionadded:: 5.8.0

        use_interpolation : bool, default True
            Option to perform interpolations when multiple timestamp rasters are
            present. If False, then DataFrame columns will be automatically
            filled with NaNs where the DataFrame index values are not found in
            the current column's timestamps.

            .. versionadded:: 5.11.0

        only_basenames : bool, default False
            Use just the field names, without prefix, for structures and channel
            arrays.

            .. versionadded:: 5.13.0

        interpolate_outwards_with_nan : bool, default False
            Use NaN values for the samples that lie outside of the original
            signal's timestamps.

            .. versionadded:: 5.15.0

        numeric_1D_only : bool, default False
            Only keep the 1D-columns that have numeric values.
        use_polars : bool, default False
            Return polars.DataFrame instead of pandas.DataFrame.

            .. versionadded:: 8.1.0

        Returns
        -------
        dataframe : pandas.DataFrame or polars.DataFrame
            Channel data.
        """
        if isinstance(raw, dict):
            if "__default__" not in raw:
                raise MdfException("The raw argument given as dict must contain the __default__ key")

            __default__ = raw["__default__"]
        else:
            __default__ = raw

        if channels is not None:
            mdf = self.filter(channels)

            result = mdf.to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                use_display_names=use_display_names,
                time_as_date=time_as_date,
                reduce_memory_usage=reduce_memory_usage,
                raw=raw,
                ignore_value2text_conversions=ignore_value2text_conversions,
                use_interpolation=use_interpolation,
                only_basenames=only_basenames,
                interpolate_outwards_with_nan=interpolate_outwards_with_nan,
                numeric_1D_only=numeric_1D_only,
                use_polars=use_polars,
            )

            mdf.close()
            return result

        target_byte_order = "<=" if sys.byteorder == "little" else ">="

        data: dict[str, NDArray[Any] | pd.Series[Any]] | dict[str, pl.Series] = {}

        self._mdf._set_temporary_master(None)

        if raster is not None:
            if isinstance(raster, (int, float)):
                raster = float(raster)
                if raster <= 0:
                    raise MdfException("The raster value must be > 0")
                raster = self.master_using_raster(raster)
            elif isinstance(raster, str):
                raster = self._mdf.get(raster).timestamps
            else:
                raster = np.array(raster)
            master = raster

        else:
            masters = {index: self._mdf.get_master(index) for index in self.virtual_groups}

            if masters:
                master = reduce(np.union1d, masters.values())
            else:
                master = np.array([], dtype="<f4")

            del masters

        idx = np.argwhere(np.diff(master, prepend=-np.inf) > 0).flatten()
        master = master[idx]

        used_names = UniqueDB()
        used_names.get_unique_name("timestamps")

        groups_nr = len(self.virtual_groups)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

                if progress.stop:
                    raise Terminated

        for group_index, (virtual_group_index, virtual_group) in enumerate(self.virtual_groups.items()):
            if virtual_group.cycles_nr == 0 and empty_channels == "skip":
                continue

            channels = [
                (None, gp_index, ch_index)
                for gp_index, channel_indexes in self.included_channels(virtual_group_index)[
                    virtual_group_index
                ].items()
                for ch_index in channel_indexes
                if ch_index != self.masters_db.get(gp_index, None)
            ]

            signals = self.select(channels, raw=True, copy_master=False, validate=False)

            if not signals:
                continue

            group_master = signals[0].timestamps

            for sig in signals:
                if len(sig) == 0:
                    if empty_channels == "zeros":
                        sig.samples = np.zeros(
                            len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                            dtype=sig.samples.dtype,
                        )
                        sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

            for signal in signals:
                if (isinstance(raw, dict) and not raw.get(signal.name, __default__)) or not raw:
                    conversion = signal.conversion
                    if conversion:
                        samples = conversion.convert(
                            signal.samples, ignore_value2text_conversions=ignore_value2text_conversions
                        )
                        signal.samples = samples

                    signal.raw = False
                    signal.conversion = None
                    if signal.samples.dtype.kind == "S":
                        signal.encoding = "utf-8" if self.version >= "4.00" else "latin-1"

            for s_index, sig in enumerate(signals):
                sig = sig.validate(copy=False)

                if len(sig) == 0:
                    if empty_channels == "zeros":
                        sig.samples = np.zeros(
                            len(master) if virtual_group.cycles_nr == 0 else virtual_group.cycles_nr,
                            dtype=sig.samples.dtype,
                        )
                        sig.timestamps = master if virtual_group.cycles_nr == 0 else group_master

                signals[s_index] = sig

            if use_interpolation or use_polars:
                same_master = np.array_equal(master, group_master)

                if not same_master and interpolate_outwards_with_nan and not use_polars:
                    idx = np.argwhere((master >= group_master[0]) & (master <= group_master[-1])).flatten()

                cycles = len(group_master)

                signals = [
                    (
                        signal.interp(
                            master,
                            integer_interpolation_mode=self._mdf._integer_interpolation,
                            float_interpolation_mode=self._mdf._float_interpolation,
                        )
                        if not same_master or len(signal) != cycles
                        else signal
                    )
                    for signal in signals
                ]

                if not same_master and interpolate_outwards_with_nan and not use_polars:
                    for sig in signals:
                        sig.timestamps = sig.timestamps[idx]
                        sig.samples = sig.samples[idx]

                group_master = master

            if any(len(sig) for sig in signals):
                signals = [sig for sig in signals if len(sig)]

            if group_master.dtype.byteorder not in target_byte_order:
                group_master = group_master.byteswap().view(group_master.dtype.newbyteorder())

            index: NDArray[Any] | pd.Index[Any]

            if signals:
                diffs = np.diff(group_master, prepend=-np.inf) > 0
                if np.all(diffs):
                    if use_polars:
                        index = group_master
                    else:
                        index = pd.Index(group_master, tupleize_cols=False)

                else:
                    idx = np.argwhere(diffs).flatten()
                    group_master = group_master[idx]

                    if use_polars:
                        index = group_master
                    else:
                        index = pd.Index(group_master, tupleize_cols=False)

                    for sig in signals:
                        sig.samples = sig.samples[idx]
                        sig.timestamps = sig.timestamps[idx]
            else:
                if use_polars:
                    index = group_master
                else:
                    index = pd.Index(group_master, tupleize_cols=False)

            size = len(index)
            for sig in signals:
                if sig.timestamps.dtype.byteorder not in target_byte_order:
                    sig.timestamps = sig.timestamps.byteswap().view(sig.timestamps.dtype.newbyteorder())

                if use_polars:
                    sig_index = index
                else:
                    sig_index = index if len(sig) == size else pd.Index(sig.timestamps, tupleize_cols=False)

                # byte arrays
                if len(sig.samples.shape) > 1:
                    if use_display_names:
                        channel_name = list(sig.display_names)[0] if sig.display_names else sig.name
                    else:
                        channel_name = sig.name

                    channel_name = used_names.get_unique_name(channel_name)

                    if sig.samples.dtype.byteorder not in target_byte_order:
                        sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                    if use_polars:
                        data = typing.cast(dict[str, pl.Series], data)
                        data[channel_name] = pl.Series(name=channel_name, values=sig.samples)
                    else:
                        data = typing.cast(dict[str, Union[NDArray[Any], "pd.Series[Any]"]], data)
                        data[channel_name] = pd.Series(list(sig.samples), index=sig_index)

                # arrays and structures
                elif sig.samples.dtype.names:
                    if use_polars:
                        data = typing.cast(dict[str, pl.Series], data)
                        for name, values in components(
                            sig.samples,
                            sig.name,
                            used_names,
                            master=sig_index,
                            only_basenames=only_basenames,
                            use_polars=use_polars,
                        ):
                            data[name] = pl.Series(name=name, values=values)
                    else:
                        data = typing.cast(dict[str, Union[NDArray[Any], "pd.Series[Any]"]], data)
                        for name, pd_series in components(
                            sig.samples,
                            sig.name,
                            used_names,
                            master=sig_index,
                            only_basenames=only_basenames,
                            use_polars=use_polars,
                        ):
                            data[name] = pd_series

                # scalars
                else:
                    if use_display_names:
                        channel_name = list(sig.display_names)[0] if sig.display_names else sig.name
                    else:
                        channel_name = sig.name

                    channel_name = used_names.get_unique_name(channel_name)

                    if reduce_memory_usage and sig.samples.dtype.kind not in "SU":
                        if sig.samples.size > 0:
                            sig.samples = downcast(sig.samples)

                    if sig.samples.dtype.byteorder not in target_byte_order:
                        sig.samples = sig.samples.byteswap().view(sig.samples.dtype.newbyteorder())

                    if use_polars:
                        data = typing.cast(dict[str, pl.Series], data)
                        data[channel_name] = pl.Series(name=channel_name, values=sig.samples)
                    else:
                        data = typing.cast(dict[str, Union[NDArray[Any], "pd.Series[Any]"]], data)
                        data[channel_name] = pd.Series(sig.samples, index=sig_index)

            if progress is not None:
                if callable(progress):
                    progress(group_index + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(group_index + 1)

                    if progress.stop:
                        raise Terminated

        if use_polars:
            data = typing.cast(dict[str, pl.Series], data)

            if not POLARS_AVAILABLE:
                raise MdfException("to_dataframe(use_polars=True) requires polars")

            if numeric_1D_only:
                data = {col: pl_series for col, pl_series in data.items() if pl_series.dtype.is_numeric()}

            if time_as_date:
                # FIXME: something is wrong with the type of timestamps/master
                master = self.header.start_time + pd.to_timedelta(master, unit="s")  # type: ignore[assignment]
            elif time_from_zero and len(master):
                master = master - master[0]

            # FIXME: something is wrong with the type of timestamps/master
            data = {"timestamps": master, **data}  # type: ignore[assignment]
            return pl.DataFrame(data)

        else:
            data = typing.cast(dict[str, Union[NDArray[Any], "pd.Series[Any]"]], data)

            strings: dict[str, NDArray[Any] | pd.Series[Any]] = {}
            nonstrings: dict[str, NDArray[Any] | pd.Series[Any]] = {}

            for col, vals in data.items():
                if vals.dtype.kind == "S":
                    strings[col] = vals
                else:
                    nonstrings[col] = vals

            if numeric_1D_only:
                nonstrings = {col: vals for col, vals in data.items() if vals.dtype.kind in "uif"}
                strings = {}

            df = pd.DataFrame(nonstrings, index=master)

            if strings:
                df_strings = pd.DataFrame(strings, index=master)
                df = pd.concat([df, df_strings], axis=1)

            df.index.name = "timestamps"

            if time_as_date:
                delta = pd.to_timedelta(df.index, unit="s")

                new_index = self.header.start_time + delta
                df.set_index(new_index, inplace=True)

            elif time_from_zero and len(master):
                df.set_index(df.index - df.index[0], inplace=True)

            return df

    def extract_bus_logging(
        self,
        database_files: dict[BusType, Iterable[DbcFileType]],
        version: str | v4c.Version | None = None,
        ignore_value2text_conversion: bool = True,
        prefix: str = "",
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> "MDF":
        """Extract all possible CAN signals using the provided databases.

        .. versionchanged:: 6.0.0 Renamed from `extract_can_logging`.

        Parameters
        ----------
        database_files : dict
            Each key will contain an iterable of database files for that bus
            type. The supported bus types are "CAN" and "LIN". The iterables
            will contain the (database, valid bus) pairs. The database can be a
            str, pathlib.Path or canmatrix.CanMatrix object. The valid bus is
            an integer specifying for which bus channel the database can be
            applied; 0 means any bus channel.

            .. versionchanged:: 6.0.0 Added canmatrix.CanMatrix type.

            .. versionchanged:: 6.3.0 Added bus channel filter.

        version : str, optional
            Output file version.

        ignore_value2text_conversion : bool, default True
            Ignore value to text conversions.

            .. versionadded:: 5.23.0

        prefix : str, default ''
            Prefix that will be added to the channel group names and signal
            names in the output file.

            .. versionadded:: 6.3.0


        Returns
        -------
        mdf : MDF
            New `MDF` object that contains the succesfully extracted signals.

        Examples
        --------
        Extract CAN and LIN bus logging.

        >>> mdf = asammdf.MDF(r'bus_logging.mf4')
        >>> databases = {
        ...     "CAN": [("file1.dbc", 0), ("file2.arxml", 2)],
        ...     "LIN": [("file3.dbc", 0)],
        ... }
        >>> extracted = mdf.extract_bus_logging(database_files=databases)

        Extract just LIN bus logging.

        >>> mdf = asammdf.MDF(r'bus_logging.mf4')
        >>> databases = {
        ...     "LIN": [("file3.dbc", 0)],
        ... }
        >>> extracted = mdf.extract_bus_logging(database_files=databases)
        """
        if not isinstance(self._mdf, mdf_v4.MDF4):
            raise MdfException("extract_bus_logging is only supported in MDF4 files")

        if version is None:
            version = self.version
        else:
            version = validate_version_argument(version)

        out = MDF(
            version=version,
            password=self._mdf._password,
            use_display_names=True,
        )

        if not isinstance(out._mdf, mdf_v4.MDF4):
            raise MdfException("extract_bus_logging is only supported in MDF4 files")

        out.header.start_time = self.header.start_time

        if database_files.get("CAN", None):
            out._mdf = self._mdf._extract_can_logging(
                out._mdf,
                database_files["CAN"],
                ignore_value2text_conversion,
                prefix,
                progress=progress,
            )

            to_keep: list[tuple[None, int, int]] = []
            all_channels: list[tuple[None, int, int]] = []

            for i, group in enumerate(out._mdf.groups):
                for j, channel in enumerate(group.channels[1:], 1):
                    if not all(self._mdf.last_call_info["CAN"]["max_flags"][i][j]):
                        to_keep.append((None, i, j))
                    all_channels.append((None, i, j))

            if to_keep != all_channels:
                tmp = out.filter(to_keep, out.version)
                out.close()
                out = tmp

        if database_files.get("LIN", None):
            out._mdf = self._mdf._extract_lin_logging(
                typing.cast(mdf_v4.MDF4, out._mdf),
                database_files["LIN"],
                ignore_value2text_conversion,
                prefix,
                progress=progress,
            )

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
        if isinstance(self._mdf, mdf_v4.MDF4):
            return self._mdf.save(
                dst,
                overwrite=overwrite,
                compression=compression,
                progress=progress,
                add_history_block=add_history_block,
            )

        if isinstance(dst, FileLike):
            raise TypeError(f"'dst' must be of type '{StrPath}'")

        return self._mdf.save(
            dst,
            overwrite=overwrite,
            compression=compression,
            progress=progress,
            add_history_block=add_history_block,
        )

    def cleanup_timestamps(
        self,
        minimum: float,
        maximum: float,
        exp_min: int = -15,
        exp_max: int = 15,
        version: str | Version | None = None,
        progress: Callable[[int, int], None] | Any | None = None,
    ) -> "MDF":
        """Clean up timestamps and convert `MDF` to other version.

        .. versionadded:: 5.22.0

        Parameters
        ----------
        minimum : float
            Minimum plausible timestamp.
        maximum : float
            Maximum plausible timestamp.
        exp_min : int, default -15
            Minimum plausible exponent used for the timestamps float values.
        exp_max : int, default 15
            Maximum plausible exponent used for the timestamps float values.
        version : str, optional
            New MDF file version from ('2.00', '2.10', '2.14', '3.00', '3.10',
            '3.20', '3.30', '4.00', '4.10', '4.11', '4.20'); default is None
            and in this case the original file version is used.

        Returns
        -------
        out : MDF
            New `MDF` object.
        """

        if version is None:
            version = self.version
        else:
            version = validate_version_argument(version)

        out = MDF(version=version)

        out.header.start_time = self.header.start_time

        groups_nr = len(self.virtual_groups)

        if progress is not None:
            if callable(progress):
                progress(0, groups_nr)
            else:
                progress.signals.setValue.emit(0)
                progress.signals.setMaximum.emit(groups_nr)

                if progress.stop:
                    raise Terminated

        # walk through all groups and get all channels
        for i, virtual_group in enumerate(self.virtual_groups):
            for idx, sigs in enumerate(self._mdf._yield_selected_signals(virtual_group, version=version)):
                if idx == 0:
                    sigs = typing.cast(list[Signal], sigs)
                    if sigs:
                        t = sigs[0].timestamps
                        if len(t):
                            all_ok, indices = plausible_timestamps(t, minimum, maximum, exp_min, exp_max)
                            if not all_ok:
                                t = t[indices]
                                if len(t):
                                    for sig in sigs:
                                        sig.samples = sig.samples[indices]
                                        sig.timestamps = t
                                        if sig.invalidation_bits is not None:
                                            sig.invalidation_bits = InvalidationArray(sig.invalidation_bits[indices])
                        cg = self.groups[virtual_group].channel_group
                        cg_nr = out.append(
                            sigs,
                            acq_name=getattr(cg, "acq_name", None),
                            acq_source=getattr(cg, "acq_source", None),
                            comment=f"Timestamps cleaned up and converted from {self.version} to {version}",
                            common_timebase=True,
                        )
                    else:
                        break
                else:
                    sigs = typing.cast(list[tuple[NDArray[Any], None]], sigs)
                    t, _ = sigs[0]
                    if len(t):
                        all_ok, indices = plausible_timestamps(t, minimum, maximum, exp_min, exp_max)
                        if not all_ok:
                            t = t[indices]
                            if len(t):
                                for i, (samples, invalidation_bits) in enumerate(sigs):
                                    if invalidation_bits is not None:
                                        invalidation_bits = invalidation_bits[indices]
                                    samples = samples[indices]

                                    sigs[i] = (samples, invalidation_bits)

                    out.extend(cg_nr, sigs)

            if progress is not None:
                if callable(progress):
                    progress(i + 1, groups_nr)
                else:
                    progress.signals.setValue.emit(i + 1)

                    if progress.stop:
                        raise Terminated

        out._transfer_metadata(self)

        return out

    def whereis(
        self,
        channel: str,
        source_name: str | None = None,
        source_path: str | None = None,
        acq_name: str | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Get occurrences of channel name in the file.

        Parameters
        ----------
        channel : str
            Channel name string.
        source_name : str, optional
            Filter occurrences on source name.
        source_path : str, optional
            Filter occurrences on source path.
        acq_name : str, optional
            Filter occurrences on channel group acquisition name.

            .. versionadded:: 6.0.0

        Returns
        -------
        tuple[tuple[int, int], ...]
            (gp_idx, cn_idx) pairs.

        Examples
        --------
        >>> mdf = MDF(file_name)
        >>> mdf.whereis('VehicleSpeed')  # "VehicleSpeed" exists in the file
        ((1, 2), (2, 4))
        >>> mdf.whereis('VehicleSPD')  # "VehicleSPD" doesn't exist in the file
        ()
        """
        occurrences = tuple(
            self._mdf._filter_occurrences(
                iter(self.channels_db.get(channel, ())),
                source_name=source_name,
                source_path=source_path,
                acq_name=acq_name,
            )
        )
        return occurrences

    def search(
        self,
        pattern: str,
        mode: Literal["plain", "regex", "wildcard"] | SearchMode = SearchMode.plain,
        case_insensitive: bool = False,
    ) -> list[str]:
        """Search channels.

        .. versionadded:: 7.0.0

        Parameters
        ----------
        pattern : str
            Search pattern.
        mode : {'plain', 'regex', 'wildcard'} or SearchMode, default SearchMode.plain
            Search mode.

            * `plain` : normal name search
            * `regex` : regular expression based search
            * `wildcard` : wildcard based search
        case_insensitive : bool, default False
            Case sensitivity for the channel name search.

        Returns
        -------
        channels : list[str]
            Names of the channels.

        Raises
        ------
        ValueError
            Unsupported search mode.

        Examples
        --------
        >>> mdf = MDF(file_name)
        >>> mdf.search('*veh*speed*', case_insensitive=True, mode='wildcard')  # case insensitive wildcard based search
        ['vehicleAverageSpeed', 'vehicleInstantSpeed', 'targetVehicleAverageSpeed', 'targetVehicleInstantSpeed']
        >>> mdf.search('^vehicle.*Speed$', case_insensitive=False, mode='regex')  # case sensitive regex based search
        ['vehicleAverageSpeed', 'vehicleInstantSpeed']
        """
        search_mode = SearchMode(mode)

        if search_mode is SearchMode.plain:
            if case_insensitive:
                pattern = pattern.casefold()
                channels = [name for name in self.channels_db if pattern in name.casefold()]
            else:
                channels = [name for name in self.channels_db if pattern in name]
        elif search_mode is SearchMode.regex:
            flags = re.IGNORECASE if case_insensitive else 0
            compiled_pattern = re.compile(pattern, flags=flags)
            channels = [name for name in self.channels_db if compiled_pattern.search(name)]
        elif search_mode is SearchMode.wildcard:
            wildcard = f"{os.urandom(6).hex()}_WILDCARD_{os.urandom(6).hex()}"
            pattern = pattern.replace("*", wildcard)
            pattern = re.escape(pattern)
            pattern = pattern.replace(wildcard, ".*")

            flags = re.IGNORECASE if case_insensitive else 0

            compiled_pattern = re.compile(pattern, flags=flags)

            channels = [name for name in self.channels_db if compiled_pattern.search(name)]

        else:
            raise ValueError(f"unsupported mode {search_mode}")

        return channels

    def _asc_export(self, file_name: Path) -> None:
        if not isinstance(self._mdf, mdf_v4.MDF4):
            return

        groups_count = len(self.groups)

        dfs = []

        for idx in range(groups_count):
            group = self._mdf.groups[idx]
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                source = group.channel_group.acq_source

                names = {ch.name for ch in group.channels}

                columns: dict[str, object]
                if source and source.bus_type == v4c.BUS_TYPE_CAN:
                    if "CAN_DataFrame" in names:
                        data = self._mdf.get("CAN_DataFrame", idx)  # , raw=True)

                    elif "CAN_RemoteFrame" in names:
                        data = self._mdf.get("CAN_RemoteFrame", idx, raw=True)

                    elif "CAN_ErrorFrame" in names:
                        data = self._mdf.get("CAN_ErrorFrame", idx, raw=True)

                    else:
                        continue

                    df_index = data.timestamps
                    count = len(df_index)

                    columns = {
                        "timestamps": df_index,
                        "type": np.full(count, "CAN", dtype="O"),
                        "Bus": np.zeros(count, dtype="u1"),
                        "ID": np.full(count, 0xFFFFFFFF, dtype="u4"),
                        "IDE": np.zeros(count, dtype="u1"),
                        "Direction": np.full(count, "Rx", dtype="O"),
                        "Event Type": np.full(count, "CAN Frame", dtype="O"),
                        "Details": np.full(count, "", dtype="O"),
                        "ESI": np.zeros(count, dtype="u1"),
                        "EDL": np.zeros(count, dtype="u1"),
                        "BRS": np.zeros(count, dtype="u1"),
                        "DLC": np.zeros(count, dtype="u1"),
                        "Data Length": np.zeros(count, dtype="u1"),
                        "Data Bytes": np.full(count, "", dtype="O"),
                        "Name": np.full(count, "", dtype="O"),
                    }

                    for string in v4c.CAN_ERROR_TYPES.values():
                        sys.intern(string)

                    frame_map = None
                    if data.attachment and data.attachment[0]:
                        dbc = load_can_database(data.attachment[1], data.attachment[0])
                        if dbc:
                            frame_map = {frame.arbitration_id.id: frame.name for frame in dbc}

                            for name in frame_map.values():
                                sys.intern(name)

                    if data.name == "CAN_DataFrame":
                        columns["Bus"] = data["CAN_DataFrame.BusChannel"].astype("u1")

                        vals = data["CAN_DataFrame.ID"].astype("u4") & 0x1FFFFFFF
                        columns["ID"] = vals
                        if frame_map:
                            columns["Name"] = [frame_map.get(_id, "") for _id in typing.cast(list[int], vals.tolist())]

                        columns["DLC"] = data["CAN_DataFrame.DLC"].astype("u1")
                        columns["Data Length"] = data["CAN_DataFrame.DataLength"].astype("u1")

                        data_bytes = csv_bytearray2hex(
                            pd.Series(list(data["CAN_DataFrame.DataBytes"])).to_numpy(),
                            typing.cast(int, columns["Data Length"]),
                        )
                        columns["Data Bytes"] = data_bytes

                        if "CAN_DataFrame.Dir" in names:
                            if data["CAN_DataFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8").capitalize()
                                    for v in typing.cast(list[bytes], data["CAN_DataFrame.Dir"].tolist())
                                ]
                            else:
                                columns["Direction"] = [
                                    "Tx" if dir else "Rx" for dir in data["CAN_DataFrame.Dir"].astype("u1").tolist()
                                ]

                        if "CAN_DataFrame.ESI" in names:
                            columns["ESI"] = data["CAN_DataFrame.ESI"].astype("u1")

                        if "CAN_DataFrame.EDL" in names:
                            columns["EDL"] = data["CAN_DataFrame.EDL"].astype("u1")

                        if "CAN_DataFrame.BRS" in names:
                            columns["BRS"] = data["CAN_DataFrame.BRS"].astype("u1")

                        if "CAN_DataFrame.IDE" in names:
                            columns["IDE"] = data["CAN_DataFrame.IDE"].astype("u1")

                    elif data.name == "CAN_RemoteFrame":
                        columns["Bus"] = data["CAN_RemoteFrame.BusChannel"].astype("u1")

                        vals = data["CAN_RemoteFrame.ID"].astype("u4") & 0x1FFFFFFF
                        columns["ID"] = vals
                        if frame_map:
                            columns["Name"] = [frame_map.get(_id, "") for _id in typing.cast(list[int], vals.tolist())]

                        columns["DLC"] = data["CAN_RemoteFrame.DLC"].astype("u1")
                        columns["Data Length"] = data["CAN_RemoteFrame.DataLength"].astype("u1")
                        columns["Event Type"] = "Remote Frame"

                        if "CAN_RemoteFrame.Dir" in names:
                            if data["CAN_RemoteFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8").capitalize()
                                    for v in typing.cast(list[bytes], data["CAN_RemoteFrame.Dir"].tolist())
                                ]
                            else:
                                columns["Direction"] = [
                                    "Tx" if dir else "Rx" for dir in data["CAN_RemoteFrame.Dir"].astype("u1").tolist()
                                ]

                        if "CAN_RemoteFrame.IDE" in names:
                            columns["IDE"] = data["CAN_RemoteFrame.IDE"].astype("u1")

                    elif data.name == "CAN_ErrorFrame":
                        if data.samples.dtype.names is None:
                            raise ValueError("names is None")

                        names = set(data.samples.dtype.names)

                        if "CAN_ErrorFrame.BusChannel" in names:
                            columns["Bus"] = data["CAN_ErrorFrame.BusChannel"].astype("u1")

                        if "CAN_ErrorFrame.ID" in names:
                            vals = data["CAN_ErrorFrame.ID"].astype("u4") & 0x1FFFFFFF
                            columns["ID"] = vals
                            if frame_map:
                                columns["Name"] = [
                                    frame_map.get(_id, "") for _id in typing.cast(list[int], vals.tolist())
                                ]

                        if "CAN_ErrorFrame.DLC" in names:
                            columns["DLC"] = data["CAN_ErrorFrame.DLC"].astype("u1")

                        if "CAN_ErrorFrame.DataLength" in names:
                            columns["Data Length"] = data["CAN_ErrorFrame.DataLength"].astype("u1")

                        columns["Event Type"] = "Error Frame"

                        if "CAN_ErrorFrame.ErrorType" in names:
                            error_types = typing.cast(list[int], data["CAN_ErrorFrame.ErrorType"].astype("u1").tolist())
                            details = [v4c.CAN_ERROR_TYPES.get(err, "Other error") for err in error_types]

                            columns["Details"] = details

                        if "CAN_ErrorFrame.Dir" in names:
                            if data["CAN_ErrorFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8").capitalize()
                                    for v in typing.cast(list[bytes], data["CAN_ErrorFrame.Dir"].tolist())
                                ]
                            else:
                                columns["Direction"] = [
                                    "Tx" if dir else "Rx" for dir in data["CAN_ErrorFrame.Dir"].astype("u1").tolist()
                                ]

                    dfs.append(pd.DataFrame(columns, index=df_index))

                elif source and source.bus_type == v4c.BUS_TYPE_FLEXRAY:
                    if "FLX_Frame" in names:
                        data = self._mdf.get("FLX_Frame", idx, raw=True)

                    elif "FLX_NullFrame" in names:
                        data = self._mdf.get("FLX_NullFrame", idx, raw=True)

                    elif "FLX_StartCycle" in names:
                        data = self._mdf.get("FLX_StartCycle", idx, raw=True)

                    elif "FLX_Status" in names:
                        data = self._mdf.get("FLX_Status", idx, raw=True)
                    else:
                        continue

                    df_index = data.timestamps
                    count = len(df_index)

                    columns = {
                        "timestamps": df_index,
                        "type": np.full(count, "FLEXRAY", dtype="O"),
                        "Bus": np.zeros(count, dtype="u1"),
                        "ID": np.full(count, 0xFFFF, dtype="u2"),
                        "ControllerFlags": np.zeros(count, dtype="u2"),
                        "FrameFlags": np.zeros(count, dtype="u4"),
                        "Direction": np.full(count, "Rx", dtype="O"),
                        "Cycle": np.full(count, 0xFF, dtype="u1"),
                        "Event Type": np.full(count, "FlexRay Frame", dtype="O"),
                        "Details": np.full(count, "", dtype="O"),
                        "Data Length": np.zeros(count, dtype="u1"),
                        "Payload Length": np.zeros(count, dtype="u1"),
                        "Data Bytes": np.full(count, "", dtype="O"),
                        "Header CRC": np.full(count, 0xFFFF, dtype="u2"),
                    }

                    if data.name == "FLX_Frame":
                        columns["Bus"] = data["FLX_Frame.FlxChannel"].astype("u1")
                        columns["ID"] = data["FLX_Frame.ID"].astype("u2")
                        columns["Cycle"] = data["FLX_Frame.Cycle"].astype("u1")
                        columns["Data Length"] = data["FLX_Frame.DataLength"].astype("u1")
                        columns["Payload Length"] = data["FLX_Frame.PayloadLength"].astype("u1") * 2

                        data_bytes = csv_bytearray2hex(
                            pd.Series(list(data["FLX_Frame.DataBytes"])),
                            typing.cast(int, columns["Data Length"]),
                        )
                        columns["Data Bytes"] = data_bytes

                        columns["Header CRC"] = data["FLX_Frame.HeaderCRC"].astype("u2")

                        if "FLX_Frame.Dir" in names:
                            if data["FLX_Frame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8").capitalize()
                                    for v in typing.cast(list[bytes], data["FLX_Frame.Dir"].tolist())
                                ]
                            else:
                                columns["Direction"] = [
                                    "Tx" if dir else "Rx" for dir in data["FLX_Frame.Dir"].astype("u1").tolist()
                                ]

                        if "FLX_Frame.ControllerFlags" in names:
                            columns["ControllerFlags"] = np.frombuffer(
                                data["FLX_Frame.ControllerFlags"].tobytes(), dtype="<u2"
                            )
                        if "FLX_Frame.FrameFlags" in names:
                            columns["FrameFlags"] = np.frombuffer(data["FLX_Frame.FrameFlags"].tobytes(), dtype="<u4")

                    elif data.name == "FLX_NullFrame":
                        columns["Bus"] = data["FLX_NullFrame.FlxChannel"].astype("u1")
                        columns["ID"] = data["FLX_NullFrame.ID"].astype("u2")
                        columns["Cycle"] = data["FLX_NullFrame.Cycle"].astype("u1")

                        columns["Event Type"] = "FlexRay NullFrame"
                        columns["Header CRC"] = data["FLX_NullFrame.HeaderCRC"].astype("u2")

                        if "FLX_NullFrame.Dir" in names:
                            if data["FLX_NullFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8").capitalize()
                                    for v in typing.cast(list[bytes], data["FLX_NullFrame.Dir"].tolist())
                                ]
                            else:
                                columns["Direction"] = [
                                    "Tx" if dir else "Rx" for dir in data["FLX_NullFrame.Dir"].astype("u1").tolist()
                                ]

                    elif data.name == "FLX_StartCycle":
                        columns["Cycle"] = data["FLX_StartCycle.Cycle"].astype("u1")
                        columns["Event Type"] = "FlexRay StartCycle"

                    elif data.name == "FLX_Status":
                        vals = data["FLX_Status.StatusType"].astype("u1")
                        columns["Details"] = vals.astype("U").astype("O")

                        columns["Event Type"] = "FlexRay Status"

                    dfs.append(pd.DataFrame(columns, index=df_index))

        if dfs:
            signals = pd.concat(dfs).sort_index()

            index = pd.Index(range(len(signals)))
            signals.set_index(index, inplace=True)
        else:
            signals = pd.DataFrame()

        with open(file_name, "w") as asc:
            start = self.start_time.strftime("%a %b %d %I:%M:%S.%f %p %Y")
            asc.write(f"date {start}\n")
            asc.write("base hex  timestamps absolute\n")
            asc.write("no internal events logged\n")

            for row in signals.to_dict("records"):
                if row["type"] == "CAN":
                    if row["Event Type"] == "CAN Frame":
                        if row["EDL"]:
                            data = row["Data Bytes"].lower()
                            id = f"{row['ID']:x}"

                            if row["IDE"]:
                                id = f"{id}x"

                            bus = row["Bus"]
                            dir = row["Direction"]
                            t = row["timestamps"]

                            flags = 1 << 12
                            brs = row["BRS"]
                            if brs:
                                flags |= 1 << 13

                            esi = row["ESI"]
                            if esi:
                                flags |= 1 << 14

                            name = row["Name"]
                            dlc = row["DLC"]
                            data_length = row["Data Length"]

                            asc.write(
                                f"   {t: 9.6f} CANFD {bus:>3} {dir:<4} {id:>8}  {name:>32} {brs} {esi} {dlc:x} {data_length:>2} {data}        0    0 {flags:>8x}        0        0        0        0        0\n"
                            )

                        else:
                            dlc = row["DLC"]
                            data = row["Data Bytes"]
                            id = f"{row['ID']:x}"

                            if row["IDE"]:
                                id = f"{id}x"

                            bus = row["Bus"]
                            dir = row["Direction"]
                            t = row["timestamps"]

                            asc.write(f"{t: 9.6f} {bus}  {id:<15} {dir:<4} d {dlc:x} {data}\n")

                    elif row["Event Type"] == "Error Frame":
                        asc.write(f"   {row['timestamps']: 9.6f} {row['Bus']} ErrorFrame\n")

                    elif row["Event Type"] == "Remote Frame":
                        dlc = row["DLC"]
                        id = f"{row['ID']:x}"

                        if row["IDE"]:
                            id = f"{id}x"

                        bus = row["Bus"]
                        dir = row["Direction"]
                        t = row["timestamps"]

                        asc.write(f"   {t: 9.6f} {bus}  {id:<15} {dir:<4} r {dlc:x}\n")

                elif row["type"] == "FLEXRAY":
                    if row["Event Type"] == "FlexRay Frame":
                        frame_flags = f"{row['FrameFlags']:x}"
                        controller_flags = f"{row['ControllerFlags']:x}"
                        data = row["Data Bytes"]
                        header_crc = f"{row['Header CRC']:x}"
                        data_length = f"{row['Data Length']:x}"
                        payload_length = f"{row['Payload Length']:x}"
                        bus = f"{row['Bus'] + 1:x}"
                        slot = f"{row['ID']:x}"
                        cycle = f"{row['Cycle']:x}"
                        dir = row["Direction"]
                        t = row["timestamps"]

                        asc.write(
                            f"   {t: 9.6f} Fr RMSG  0 0 1 {bus} {slot} {cycle} {dir} 0 {frame_flags} 5  {controller_flags}  {header_crc} x {payload_length} {data_length} {data} 0  0  0\n"
                        )

                    elif row["Event Type"] == "FlexRay NullFrame":
                        frame_flags = f"{row['FrameFlags']:x}"
                        controller_flags = f"{row['ControllerFlags']:x}"
                        header_crc = f"{row['Header CRC']:x}"
                        payload_length = f"{row['Payload Length']:x}"
                        bus = f"{row['Bus'] + 1:x}"
                        slot = f"{row['ID']:x}"
                        cycle = f"{row['Cycle']:x}"
                        dir = row["Direction"]
                        t = row["timestamps"]

                        asc.write(
                            f"   {t: 9.6f} Fr RMSG  0 0 1 {bus} {slot} {cycle} {dir} 0 {frame_flags} 5  {controller_flags}  {header_crc} x {payload_length} 0 0  0  0\n"
                        )


if __name__ == "__main__":
    pass

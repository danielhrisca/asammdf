"""
asammdf utility functions and classes
"""

from collections import namedtuple
from collections.abc import Callable, Collection, Iterator
from copy import deepcopy
from functools import lru_cache
import json
import logging
import mmap
import multiprocessing
from os import PathLike
from pathlib import Path
from random import randint
import re
import string
from struct import Struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from traceback import format_exc
from types import TracebackType
import typing
from typing import (
    Any,
    Literal,
    Optional,
    overload,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
import xml.etree.ElementTree as ET

from canmatrix.canmatrix import CanMatrix, matrix_class
import canmatrix.formats
import lxml.etree
import numpy as np
from numpy import arange, bool_, dtype, interp, where
from numpy.typing import NDArray
import pandas as pd
from pandas import Series
from typing_extensions import (
    Buffer,
    NotRequired,
    ParamSpec,
    runtime_checkable,
    TypedDict,
    TypeIs,
    Unpack,
)

from . import v2_v3_constants as v3c
from . import v4_constants as v4c

try:
    from pyqtgraph import functions as fn
except ImportError:

    class fn:  # type: ignore[no-redef]
        @classmethod
        def mkColor(cls, color: str) -> str:
            return color

        @classmethod
        def mkPen(cls, color: str) -> str:
            return color

        @classmethod
        def mkBrush(cls, color: str) -> str:
            return color


try:
    from cchardet import detect
except:
    try:
        from chardet import detect
    except:

        class DetectDict(TypedDict):
            encoding: str | None

        def detect(text: bytes) -> DetectDict:
            encoding: str | None
            for encoding in ("utf-8", "latin-1", "cp1250", "cp1252"):
                try:
                    text.decode(encoding)
                    break
                except:
                    continue
            else:
                encoding = None
            return {"encoding": encoding}


if TYPE_CHECKING:
    from PySide6 import QtCore

THREAD_COUNT = max(multiprocessing.cpu_count() - 1, 1)
TERMINATED = object()
NONE = object()
COMPARISON_NAME = re.compile(r"(\s*\d+:)?(?P<name>.+)")
C_FUNCTION = re.compile(r"\s+(?P<function>\S+)\s*\(\s*struct\s+DATA\s+\*data\s*\)")
target_byte_order = "<=" if sys.byteorder == "little" else ">="

COLOR_MAPS = {
    "Accent": ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0", "#f0027f", "#bf5b16", "#666666"],
    "Dark2": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"],
    "Paired": [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
    ],
    "Pastel1": ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc", "#e5d8bd", "#fddaec", "#f2f2f2"],
    "Pastel2": ["#b3e2cd", "#fdcdac", "#cbd5e8", "#f4cae4", "#e6f5c9", "#fff2ae", "#f1e2cc", "#cccccc"],
    "Set1": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"],
    "Set2": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"],
    "Set3": [
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
    ],
    "tab10": [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ],
    "tab20": [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ],
    "tab20b": [
        "#393b79",
        "#5254a3",
        "#6b6ecf",
        "#9c9ede",
        "#637939",
        "#8ca252",
        "#b5cf6b",
        "#cedb9c",
        "#8c6d31",
        "#bd9e39",
        "#e7ba52",
        "#e7cb94",
        "#843c39",
        "#ad494a",
        "#d6616b",
        "#e7969c",
        "#7b4173",
        "#a55194",
        "#ce6dbd",
        "#de9ed6",
    ],
    "tab20c": [
        "#3182bd",
        "#6baed6",
        "#9ecae1",
        "#c6dbef",
        "#e6550d",
        "#fd8d3c",
        "#fdae6b",
        "#fdd0a2",
        "#31a354",
        "#74c476",
        "#a1d99b",
        "#c7e9c0",
        "#756bb1",
        "#9e9ac8",
        "#bcbddc",
        "#dadaeb",
        "#636363",
        "#969696",
        "#bdbdbd",
        "#d9d9d9",
    ],
}

COLORS = COLOR_MAPS["tab10"]
COLORS_COUNT = len(COLORS)

UINT8_u: Callable[[Buffer], tuple[int]] = Struct("<B").unpack
UINT16_u: Callable[[Buffer], tuple[int]] = Struct("<H").unpack
UINT32_p = Struct("<I").pack
UINT32_u: Callable[[Buffer], tuple[int]] = Struct("<I").unpack
UINT64_u: Callable[[Buffer], tuple[int]] = Struct("<Q").unpack
UINT8_uf: Callable[[Buffer, int], tuple[int]] = Struct("<B").unpack_from
UINT16_uf: Callable[[Buffer, int], tuple[int]] = Struct("<H").unpack_from
UINT32_uf: Callable[[Buffer, int], tuple[int]] = Struct("<I").unpack_from
UINT64_uf: Callable[[Buffer, int], tuple[int]] = Struct("<Q").unpack_from
FLOAT64_u: Callable[[Buffer], tuple[float]] = Struct("<d").unpack
FLOAT64_uf: Callable[[Buffer, int], tuple[float]] = Struct("<d").unpack_from
TWO_UINT64_u: Callable[[Buffer], tuple[int, int]] = Struct("<2Q").unpack
TWO_UINT64_uf: Callable[[Buffer, int], tuple[int, int]] = Struct("<2Q").unpack_from
BLK_COMMON_uf: Callable[[Buffer, int], tuple[bytes, int]] = Struct("<4s4xQ").unpack_from
BLK_COMMON_u: Callable[[Buffer], tuple[bytes, int]] = Struct("<4s4xQ8x").unpack

EMPTY_TUPLE = ()

_xmlns_pattern = re.compile(' xmlns="[^"]*"')

logger = logging.getLogger("asammdf")

__all__ = [
    "CHANNEL_COUNT",
    "CONVERT",
    "MDF2_VERSIONS",
    "MDF3_VERSIONS",
    "MDF4_VERSIONS",
    "MERGE",
    "SUPPORTED_VERSIONS",
    "ChannelsDB",
    "MdfException",
    "UniqueDB",
    "extract_xml_comment",
    "fmt_to_datatype_v3",
    "fmt_to_datatype_v4",
    "get_fmt_v3",
    "get_fmt_v4",
    "get_text_v4",
    "matlab_compatible",
    "validate_version_argument",
]

_channel_count = (1000, 2000, 10000, 20000)
CHANNEL_COUNT = arange(0, 20000, 1000, dtype="<u4")

_convert = (10 * 2**20, 20 * 2**20, 30 * 2**20, 40 * 2**20)
CONVERT = interp(CHANNEL_COUNT, _channel_count, _convert).astype("<u4")

_merge = (10 * 2**20, 20 * 2**20, 35 * 2**20, 60 * 2**20)
MERGE = interp(CHANNEL_COUNT, _channel_count, _merge).astype("<u4")

MDF2_VERSIONS = ("2.00", "2.10", "2.14")
MDF3_VERSIONS = ("3.00", "3.10", "3.20", "3.30")
MDF4_VERSIONS = ("4.00", "4.10", "4.11", "4.20")
SUPPORTED_VERSIONS = MDF2_VERSIONS + MDF3_VERSIONS + MDF4_VERSIONS


ALLOWED_MATLAB_CHARS = set(string.ascii_letters + string.digits + "_")


class MdfException(Exception):
    """MDF Exception class"""

    def __repr__(self) -> str:
        return f"asammdf MdfException: {self.args[0]}"


def extract_xml_comment(comment: str) -> str:
    """extract *TX* tag or otherwise the *common_properties* from a xml comment

    Parameters
    ----------
    comment : str
        xml string comment

    Returns
    -------
    comment : str
        extracted string

    """

    comment = comment.replace(' xmlns="http://www.asam.net/mdf/v4"', "")
    try:
        comment_elem = ET.fromstring(comment)
        match = comment_elem.find(".//TX")
        if match is None:
            common_properties = comment_elem.find(".//common_properties")
            if common_properties is not None:
                comments: list[str] = []
                for e in common_properties:
                    field = f'{e.get("name")}: {e.text}'
                    comments.append(field)
                comment = "\n".join(field)
            else:
                comment = ""
        else:
            comment = match.text or ""
    except ET.ParseError:
        pass

    return comment


def matlab_compatible(name: str) -> str:
    """make a channel name compatible with Matlab variable naming

    Parameters
    ----------
    name : str
        channel name

    Returns
    -------
    compatible_name : str
        channel name compatible with Matlab

    """

    compatible_names = [ch if ch in ALLOWED_MATLAB_CHARS else "_" for ch in name]
    compatible_name = "".join(compatible_names)

    if compatible_name[0] not in string.ascii_letters:
        compatible_name = "M_" + compatible_name

    # max variable name is 63 and 3 chars are reserved
    # for get_unique_name in case of multiple channel name occurrence
    return compatible_name[:60]


@runtime_checkable
class FileLike(Protocol):
    def __iter__(self) -> Iterator[bytes]: ...
    def close(self) -> None: ...
    def read(self, size: int | None = -1, /) -> bytes: ...
    def seek(self, target: int, whence: int = 0, /) -> int: ...
    def tell(self) -> int: ...
    def write(self, buffer: Buffer, /) -> int: ...


class BlockKwargs(TypedDict, total=False):
    stream: FileLike | mmap.mmap
    mapped: bool
    address: int


def stream_is_mmap(_stream: FileLike | mmap.mmap, mapped: bool) -> TypeIs[mmap.mmap]:
    return mapped


@overload
def get_text_v3(
    address: int,
    stream: FileLike | mmap.mmap,
    mapped: bool = ...,
    decode: Literal[True] = ...,
) -> str: ...


@overload
def get_text_v3(
    address: int,
    stream: FileLike | mmap.mmap,
    mapped: bool = ...,
    decode: Literal[False] = ...,
) -> bytes: ...


def get_text_v3(address: int, stream: FileLike | mmap.mmap, mapped: bool = False, decode: bool = True) -> bytes | str:
    """faster way to extract strings from mdf versions 2 and 3 TextBlock

    Parameters
    ----------
    address : int
        TextBlock address
    stream : handle
        file IO handle

    Returns
    -------
    text : str | bytes
        unicode string or bytes object depending on the ``decode`` argument

    """

    if address == 0:
        return "" if decode else b""

    if stream_is_mmap(stream, mapped):
        block_id = stream[address : address + 2]
        if block_id != b"TX":
            return "" if decode else b""
        (size,) = UINT16_uf(stream, address + 2)
        text_bytes = stream[address + 4 : address + size].split(b"\0", 1)[0].strip(b" \r\t\n")
    else:
        stream.seek(address)
        block_id = stream.read(2)
        if block_id != b"TX":
            return "" if decode else b""
        size = UINT16_u(stream.read(2))[0] - 4
        text_bytes = stream.read(size).split(b"\0", 1)[0].strip(b" \r\t\n")

    text: bytes | str

    if decode:
        try:
            text = text_bytes.decode("latin-1")
        except UnicodeDecodeError:
            encoding = detect(text_bytes)["encoding"]
            if encoding:
                try:
                    text = text_bytes.decode(encoding, "ignore")
                except:
                    text = "<!text_decode_error>"
            else:
                text = "<!text_decode_error>"
    else:
        text = text_bytes

    return text


MappedText = namedtuple("MappedText", ["raw", "decoded"])


@overload
def get_text_v4(
    address: int,
    stream: FileLike | mmap.mmap,
    mapped: bool = ...,
    decode: Literal[True] = ...,
    tx_map: dict | None = ...,
) -> str: ...


@overload
def get_text_v4(
    address: int,
    stream: FileLike | mmap.mmap,
    mapped: bool = ...,
    *,
    decode: Literal[False],
    tx_map: dict | None,
) -> bytes: ...


def get_text_v4(
    address: int,
    stream: FileLike | mmap.mmap,
    mapped: bool = False,
    decode: bool = True,
    tx_map: dict | None = None,
) -> bytes | str:
    """faster way to extract strings from mdf version 4 TextBlock

    Parameters
    ----------
    address : int
        TextBlock address
    stream : handle
        file IO handle
    mapped: bool
        flag for mapped stream
    decode: bool
        option to return decoded str instead of raw btyes
    tx_map : dict | None
        map that contains interned strings

    Returns
    -------
    text : str | bytes
        unicode string or bytes object depending on the ``decode`` argument

    """

    if mapped_text := tx_map.get(address, None):
        return mapped_text.decoded if decode else mapped_text.raw

    if address == 0:
        tx_map[address] = MappedText(b"", "")
        return "" if decode else b""

    if stream_is_mmap(stream, mapped):
        block_id, size = BLK_COMMON_uf(stream, address)
        if block_id not in (b"##TX", b"##MD"):
            tx_map[address] = MappedText(b"", "")
            return "" if decode else b""
        text_bytes = stream[address + 24 : address + size].split(b"\0", 1)[0].strip(b" \r\t\n")
    else:
        stream.seek(address)
        block_id, size = BLK_COMMON_u(stream.read(24))
        if block_id not in (b"##TX", b"##MD"):
            tx_map[address] = MappedText(b"", "")
            return "" if decode else b""
        text_bytes = stream.read(size - 24).split(b"\0", 1)[0].strip(b" \r\t\n")

    try:
        decoded_text = text_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect(text_bytes)["encoding"]
        if encoding:
            try:
                decoded_text = text_bytes.decode(encoding, "ignore")
            except:
                decoded_text = "<!text_decode_error>"
        else:
            decoded_text = "<!text_decode_error>"

    tx_map[address] = MappedText(text_bytes, decoded_text)

    return decoded_text if decode else text_bytes


def sanitize_xml(text: str) -> str:
    return re.sub(_xmlns_pattern, "", text)


def extract_display_names(comment: str) -> dict[str, str]:
    comment = comment.replace(' xmlns="http://www.asam.net/mdf/v4"', "")
    display_names = {}
    if comment.startswith("<CN") and "<names>" in comment:
        try:
            start = comment.index("<names>")
            end = comment.index("</names>") + 8
            names = ET.fromstring(comment[start:end])
            for i, elem in enumerate(names.iter()):
                if i == 0:
                    continue
                if elem.text is None:
                    raise ValueError("text is None")
                display_names[elem.text.strip(" \t\r\n\v\0")] = elem.tag

        except:
            pass

    return display_names


class EncryptionInfo(TypedDict, total=False):
    encrypted: bool
    algorithm: str
    original_md5_sum: str
    original_size: int


def extract_encryption_information(comment: str) -> EncryptionInfo:
    info: EncryptionInfo = {}
    comment = comment.replace(' xmlns="http://www.asam.net/mdf/v4"', "")
    if comment.startswith("<ATcomment") and "<encrypted>" in comment:
        try:
            comment_elem = ET.fromstring(comment)
            for match in comment_elem.findall(".//extensions/extension"):
                elem = match.find("encrypted")
                if elem is None:
                    raise RuntimeError("cannot find 'encrypted' Element")
                if elem.text is None:
                    raise RuntimeError("text is None")
                encrypted = elem.text.strip().lower() == "true"
                elem = match.find("algorithm")
                if elem is None:
                    raise RuntimeError("cannot find 'algorithm' Element")
                if elem.text is None:
                    raise RuntimeError("text is None")
                algorithm = elem.text.strip().lower()
                elem = match.find("original_md5_sum")
                if elem is None:
                    raise RuntimeError("cannot find 'original_md5_sum' Element")
                if elem.text is None:
                    raise RuntimeError("text is None")
                original_md5_sum = elem.text.strip().lower()
                elem = match.find("original_size")
                if elem is None:
                    raise RuntimeError("cannot find 'original_size' Element")
                if elem.text is None:
                    raise RuntimeError("text is None")
                original_size = int(elem.text)

                info["encrypted"] = encrypted
                info["algorithm"] = algorithm
                info["original_md5_sum"] = original_md5_sum
                info["original_size"] = original_size
                break
        except:
            pass

    return info


def extract_ev_tool(comment: str) -> str:
    tool = ""
    comment = comment.replace(' xmlns="http://www.asam.net/mdf/v4"', "")
    try:
        comment_elem = ET.fromstring(comment)
        match = comment_elem.find(".//tool")
        if match is None:
            tool = ""
        else:
            tool = match.text or ""
    except:
        pass

    return tool


@lru_cache(maxsize=1024)
def get_fmt_v3(data_type: int, size: int, byte_order: int = v3c.BYTE_ORDER_INTEL) -> str:
    """convert mdf versions 2 and 3 channel data type to numpy dtype format
    string

    Parameters
    ----------
    data_type : int
        mdf channel data type
    size : int
        data bit size
    Returns
    -------
    fmt : str
        numpy compatible data type format string

    """
    if data_type in (v3c.DATA_TYPE_STRING, v3c.DATA_TYPE_BYTEARRAY):
        size = size // 8
        if data_type == v3c.DATA_TYPE_STRING:
            fmt = f"S{size}"
        else:
            fmt = f"({size},)u1"
    else:
        if size > 64 and data_type in (
            v3c.DATA_TYPE_UNSIGNED_INTEL,
            v3c.DATA_TYPE_UNSIGNED,
            v3c.DATA_TYPE_UNSIGNED_MOTOROLA,
            v3c.DATA_TYPE_SIGNED_INTEL,
            v3c.DATA_TYPE_SIGNED,
            v3c.DATA_TYPE_SIGNED_MOTOROLA,
        ):
            fmt = f"({size // 8},)u1"
        else:
            if size <= 8:
                size = 1
            elif size <= 16:
                size = 2
            elif size <= 32:
                size = 4
            elif size <= 64:
                size = 8
            else:
                size = size // 8

            match data_type:
                case v3c.DATA_TYPE_UNSIGNED_INTEL:
                    fmt = f"<u{size}"

                case v3c.DATA_TYPE_UNSIGNED:
                    if byte_order == v3c.BYTE_ORDER_INTEL:
                        fmt = f"<u{size}"
                    else:
                        fmt = f">u{size}"

                case v3c.DATA_TYPE_UNSIGNED_MOTOROLA:
                    fmt = f">u{size}"

                case v3c.DATA_TYPE_SIGNED_INTEL:
                    fmt = f"<i{size}"

                case v3c.DATA_TYPE_SIGNED:
                    if byte_order == v3c.BYTE_ORDER_INTEL:
                        fmt = f"<i{size}"
                    else:
                        fmt = f">i{size}"

                case v3c.DATA_TYPE_SIGNED_MOTOROLA:
                    fmt = f">i{size}"

                case v3c.DATA_TYPE_FLOAT_INTEL | v3c.DATA_TYPE_DOUBLE_INTEL:
                    fmt = f"<f{size}"

                case v3c.DATA_TYPE_FLOAT_MOTOROLA | v3c.DATA_TYPE_DOUBLE_MOTOROLA:
                    fmt = f">f{size}"

                case v3c.DATA_TYPE_FLOAT | v3c.DATA_TYPE_DOUBLE:
                    if byte_order == v3c.BYTE_ORDER_INTEL:
                        fmt = f"<f{size}"
                    else:
                        fmt = f">f{size}"

    return fmt


@lru_cache(maxsize=1024)
def get_fmt_v4(data_type: int, size: int, channel_type: int = v4c.CHANNEL_TYPE_VALUE) -> str:
    """convert mdf version 4 channel data type to numpy dtype format string

    Parameters
    ----------
    data_type : int
        mdf channel data type
    size : int
        data bit size
    channel_type: int
        mdf channel type

    Returns
    -------
    fmt : str
        numpy compatible data type format string

    """
    if data_type in v4c.NON_SCALAR_TYPES:
        size = size // 8 or 1

        if data_type in (
            v4c.DATA_TYPE_BYTEARRAY,
            v4c.DATA_TYPE_MIME_STREAM,
            v4c.DATA_TYPE_MIME_SAMPLE,
        ):
            if channel_type == v4c.CHANNEL_TYPE_VALUE:
                fmt = f"({size},)u1"
            else:
                fmt = f"<u{size}"

        elif data_type in v4c.STRING_TYPES:
            if channel_type == v4c.CHANNEL_TYPE_VALUE:
                fmt = f"S{size}"
            else:
                fmt = f"<u{size}"

        elif data_type == v4c.DATA_TYPE_CANOPEN_DATE:
            fmt = "V7"

        elif data_type == v4c.DATA_TYPE_CANOPEN_TIME:
            fmt = "V6"

    elif channel_type in v4c.VIRTUAL_TYPES:
        match data_type:
            case v4c.DATA_TYPE_UNSIGNED_INTEL:
                fmt = "<u8"

            case v4c.DATA_TYPE_UNSIGNED_MOTOROLA:
                fmt = ">u8"

            case v4c.DATA_TYPE_SIGNED_INTEL:
                fmt = "<i8"

            case v4c.DATA_TYPE_SIGNED_MOTOROLA:
                fmt = ">i8"

            case v4c.DATA_TYPE_REAL_INTEL:
                fmt = "<f8"

            case v4c.DATA_TYPE_REAL_MOTOROLA:
                fmt = ">f8"
            case v4c.DATA_TYPE_COMPLEX_INTEL:
                fmt = "<c8"
            case v4c.DATA_TYPE_COMPLEX_MOTOROLA:
                fmt = ">c8"

    else:
        if size > 64 and data_type in (
            v4c.DATA_TYPE_UNSIGNED_INTEL,
            v4c.DATA_TYPE_UNSIGNED_MOTOROLA,
            v4c.DATA_TYPE_SIGNED_INTEL,
            v4c.DATA_TYPE_SIGNED_MOTOROLA,
        ):
            fmt = f"({size // 8},)u1"
        else:
            if size <= 8:
                size = 1
            elif size <= 16:
                size = 2
            elif size <= 32:
                size = 4
            elif size <= 64:
                size = 8
            else:
                size = size // 8

            match data_type:
                case v4c.DATA_TYPE_UNSIGNED_INTEL:
                    fmt = f"<u{size}"

                case v4c.DATA_TYPE_UNSIGNED_MOTOROLA:
                    fmt = f">u{size}"

                case v4c.DATA_TYPE_SIGNED_INTEL:
                    fmt = f"<i{size}"

                case v4c.DATA_TYPE_SIGNED_MOTOROLA:
                    fmt = f">i{size}"

                case v4c.DATA_TYPE_REAL_INTEL:
                    fmt = f"<f{size}"

                case v4c.DATA_TYPE_REAL_MOTOROLA:
                    fmt = f">f{size}"
                case v4c.DATA_TYPE_COMPLEX_INTEL:
                    fmt = f"<c{size}"
                case v4c.DATA_TYPE_COMPLEX_MOTOROLA:
                    fmt = f">c{size}"

    return fmt


@lru_cache(maxsize=1024)
def fmt_to_datatype_v3(fmt: dtype[Any], shape: tuple[int, ...], array: bool = False) -> tuple[int, int]:
    """convert numpy dtype format string to mdf versions 2 and 3
    channel data type and size

    Parameters
    ----------
    fmt : numpy.dtype
        numpy data type
    shape : tuple
        numpy array shape
    array : bool
        disambiguate between bytearray and channel array

    Returns
    -------
    data_type, size : int, int
        integer data type as defined by ASAM MDF and bit size

    """
    byteorder = fmt.byteorder
    if byteorder in "=|":
        byteorder = "<" if sys.byteorder == "little" else ">"
    size = fmt.itemsize * 8
    kind = fmt.kind

    if not array and shape[1:] and fmt.itemsize == 1 and kind == "u":
        data_type = v3c.DATA_TYPE_BYTEARRAY
        for dim in shape[1:]:
            size *= dim
    else:
        match kind:
            case "u":
                if byteorder == "<":
                    data_type = v3c.DATA_TYPE_UNSIGNED_INTEL
                else:
                    data_type = v3c.DATA_TYPE_UNSIGNED_MOTOROLA
            case "i":
                if byteorder == "<":
                    data_type = v3c.DATA_TYPE_SIGNED_INTEL
                else:
                    data_type = v3c.DATA_TYPE_SIGNED_MOTOROLA
            case "f":
                if byteorder == "<":
                    if size == 32:
                        data_type = v3c.DATA_TYPE_FLOAT
                    else:
                        data_type = v3c.DATA_TYPE_DOUBLE
                else:
                    if size == 32:
                        data_type = v3c.DATA_TYPE_FLOAT_MOTOROLA
                    else:
                        data_type = v3c.DATA_TYPE_DOUBLE_MOTOROLA
            case "S" | "V":
                data_type = v3c.DATA_TYPE_STRING
            case "b":
                data_type = v3c.DATA_TYPE_UNSIGNED_INTEL
                size = 1
            case _:
                message = f"Unknown type: dtype={fmt}, shape={shape}"
                logger.exception(message)
                raise MdfException(message)

    return data_type, size


@lru_cache(maxsize=1024)
def info_to_datatype_v4(signed: bool, little_endian: bool) -> int:
    """map CAN signal to MDF integer types

    Parameters
    ----------
    signed : bool
        signal is flagged as signed in the CAN database
    little_endian : bool
        signal is flagged as little endian (Intel) in the CAN database

    Returns
    -------
    datatype : int
        integer code for MDF channel data type

    """

    if signed:
        if little_endian:
            datatype = v4c.DATA_TYPE_SIGNED_INTEL
        else:
            datatype = v4c.DATA_TYPE_SIGNED_MOTOROLA
    else:
        if little_endian:
            datatype = v4c.DATA_TYPE_UNSIGNED_INTEL
        else:
            datatype = v4c.DATA_TYPE_UNSIGNED_MOTOROLA

    return datatype


@lru_cache(maxsize=1024)
def fmt_to_datatype_v4(fmt: dtype[Any], shape: tuple[int, ...], array: bool = False) -> tuple[int, int]:
    """convert numpy dtype format string to mdf version 4 channel data
    type and size

    Parameters
    ----------
    fmt : numpy.dtype
        numpy data type
    shape : tuple
        numpy array shape
    array : bool
        disambiguate between bytearray and channel array

    Returns
    -------
    data_type, size : int, int
        integer data type as defined by ASAM MDF and bit size

    """
    byteorder = fmt.byteorder
    if byteorder in "=|":
        byteorder = "<" if sys.byteorder == "little" else ">"
    size = fmt.itemsize * 8
    kind = fmt.kind

    if not array and len(shape) > 1 and size == 8 and kind == "u":
        data_type = v4c.DATA_TYPE_BYTEARRAY
        for dim in shape[1:]:
            size *= dim

    else:
        match kind:
            case "u":
                if byteorder == "<":
                    data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
                else:
                    data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
            case "i":
                if byteorder == "<":
                    data_type = v4c.DATA_TYPE_SIGNED_INTEL
                else:
                    data_type = v4c.DATA_TYPE_SIGNED_MOTOROLA
            case "f":
                if byteorder == "<":
                    data_type = v4c.DATA_TYPE_REAL_INTEL
                else:
                    data_type = v4c.DATA_TYPE_REAL_MOTOROLA
            case "S" | "V":
                data_type = v4c.DATA_TYPE_STRING_LATIN_1
            case "b":
                data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
                size = 1
            case "c":
                if byteorder == "<":
                    data_type = v4c.DATA_TYPE_COMPLEX_INTEL
                else:
                    data_type = v4c.DATA_TYPE_COMPLEX_MOTOROLA
            case _:
                message = f"Unknown type: dtype={fmt}, shape={shape}"
                logger.exception(message)
                raise MdfException(message)

    return data_type, size


def as_non_byte_sized_signed_int(integer_array: NDArray[Any], bit_length: int) -> NDArray[Any]:
    """
    The MDF spec allows values to be encoded as integers that aren't
    byte-sized. Numpy only knows how to do two's complement on byte-sized
    integers (i.e. int16, int32, int64, etc.), so we have to calculate two's
    complement ourselves in order to handle signed integers with unconventional
    lengths.

    Parameters
    ----------
    integer_array : np.array
        Array of integers to apply two's complement to
    bit_length : int
        Number of bits to sample from the array

    Returns
    -------
    integer_array : np.array
        signed integer array with non-byte-sized two's complement applied

    """

    if integer_array.flags.writeable:
        integer_array &= (1 << bit_length) - 1  # Zero out the unwanted bits
        truncated_integers = integer_array
    else:
        truncated_integers = integer_array & ((1 << bit_length) - 1)  # Zero out the unwanted bits
    return where(
        truncated_integers >> bit_length - 1,  # sign bit as a truth series (True when negative)
        (2**bit_length - truncated_integers) * np.int8(-1),  # when negative, do two's complement
        truncated_integers,  # when positive, return the truncated int
    )


def count_channel_groups(
    stream: FileLike | mmap.mmap, include_channels: bool = False, mapped: bool = False
) -> tuple[int, int]:
    """count all channel groups as fast as possible. This is used to provide
    reliable progress information when loading a file using the GUI

    Parameters
    ----------
    stream : file handle
        opened file handle
    include_channels : bool
        also count channels

    Returns
    -------
    count : int
        channel group count

    """

    count = 0
    ch_count = 0

    stream.seek(0, 2)
    file_limit = stream.tell()

    stream.seek(64)
    blk_id = stream.read(2)
    if blk_id == b"HD":
        version = 3
    else:
        blk_id += stream.read(2)
        if blk_id == b"##HD":
            version = 4
        else:
            raise MdfException(f"'{getattr(stream, 'name', stream)}' is not a valid MDF file")

    if version >= 4:
        if stream_is_mmap(stream, mapped):
            dg_addr = UINT64_uf(stream, 88)[0]
            while dg_addr:
                stream.seek(dg_addr + 32)
                cg_addr = UINT64_uf(stream, dg_addr + 32)[0]
                while cg_addr:
                    count += 1
                    if include_channels:
                        ch_addr = UINT64_uf(stream, cg_addr + 32)[0]
                        while ch_addr:
                            ch_count += 1
                            ch_addr = UINT64_uf(stream, ch_addr + 24)[0]
                            if ch_addr >= file_limit:
                                raise MdfException("File is a corrupted MDF file - Invalid CH block address")

                    cg_addr = UINT64_uf(stream, cg_addr + 24)[0]
                    if cg_addr >= file_limit:
                        raise MdfException("File is a corrupted MDF file - Invalid CG block address")

                dg_addr = UINT64_uf(stream, dg_addr + 24)[0]
                if dg_addr >= file_limit:
                    raise MdfException("File is a corrupted MDF file - Invalid DG block address")
        else:
            stream.seek(88, 0)
            dg_addr = UINT64_u(stream.read(8))[0]
            while dg_addr:
                stream.seek(dg_addr + 32)
                cg_addr = UINT64_u(stream.read(8))[0]
                while cg_addr:
                    count += 1
                    if include_channels:
                        stream.seek(cg_addr + 32)
                        ch_addr = UINT64_u(stream.read(8))[0]
                        while ch_addr:
                            ch_count += 1
                            stream.seek(ch_addr + 24)
                            ch_addr = UINT64_u(stream.read(8))[0]
                            if ch_addr >= file_limit:
                                raise MdfException("File is a corrupted MDF file - Invalid CH block address")

                    stream.seek(cg_addr + 24)
                    cg_addr = UINT64_u(stream.read(8))[0]
                    if cg_addr >= file_limit:
                        raise MdfException("File is a corrupted MDF file - Invalid CG block address")

                stream.seek(dg_addr + 24)
                dg_addr = UINT64_u(stream.read(8))[0]

                if dg_addr >= file_limit:
                    raise MdfException("File is a corrupted MDF file - Invalid DG block address")

    else:
        stream.seek(68, 0)
        dg_addr = UINT32_u(stream.read(4))[0]
        while dg_addr:
            stream.seek(dg_addr + 8)
            cg_addr = UINT32_u(stream.read(4))[0]
            while cg_addr:
                count += 1
                if include_channels:
                    stream.seek(cg_addr + 8)
                    ch_addr = UINT32_u(stream.read(4))[0]
                    while ch_addr:
                        ch_count += 1
                        stream.seek(ch_addr + 4)
                        ch_addr = UINT32_u(stream.read(4))[0]
                        if ch_addr >= file_limit:
                            raise MdfException("File is a corrupted MDF file - Invalid CH block address")

                stream.seek(cg_addr + 4)
                cg_addr = UINT32_u(stream.read(4))[0]
                if cg_addr >= file_limit:
                    raise MdfException("File is a corrupted MDF file - Invalid CG block address")

            stream.seek(dg_addr + 4)
            dg_addr = UINT32_u(stream.read(4))[0]

            if dg_addr >= file_limit:
                raise MdfException("File is a corrupted MDF file - Invalid DG block address")

    return count, ch_count


@overload
def validate_version_argument(version: v3c.Version2, hint: Literal[2] = ...) -> v3c.Version2: ...


@overload
def validate_version_argument(version: v3c.Version, hint: Literal[3] = ...) -> v3c.Version: ...


def validate_version_argument(version: str, hint: int = 4) -> str:
    """validate the version argument against the supported MDF versions. The
    default version used depends on the hint MDF major revision

    Parameters
    ----------
    version : str
        requested MDF version
    hint : int
        MDF revision hint

    Returns
    -------
    valid_version : str
        valid version

    """
    if version not in SUPPORTED_VERSIONS:
        if hint == 2:
            valid_version = "2.14"
        elif hint == 3:
            valid_version = "3.30"
        else:
            valid_version = "4.10"
        message = 'Unknown mdf version "{}".' " The available versions are {};" ' automatically using version "{}"'
        message = message.format(version, SUPPORTED_VERSIONS, valid_version)
        logger.warning(message)
    else:
        valid_version = version
    return valid_version


class ChannelsDB(dict[str, tuple[tuple[int, int], ...]]):
    def __init__(self) -> None:
        super().__init__()

    def add(self, channel_name: str, entry: tuple[int, int]) -> None:
        """add name to channels database and check if it contains a source
        path

        Parameters
        ----------
        channel_name : str
            name that needs to be added to the database
        entry : tuple
            (group index, channel index) pair

        """
        if channel_name:
            if channel_name not in self:
                self[channel_name] = (entry,)
            else:
                self[channel_name] += (entry,)

            if "\\" in channel_name:
                channel_name, _ = channel_name.split("\\", 1)

                if channel_name not in self:
                    self[channel_name] = (entry,)
                elif entry not in self[channel_name]:
                    self[channel_name] += (entry,)


def randomized_string(size: int) -> bytes:
    """get a \0 terminated string of size length

    Parameters
    ----------
    size : int
        target string length

    Returns
    -------
    string : bytes
        randomized string

    """
    return bytes(randint(65, 90) for _ in range(size - 1)) + b"\0"


def is_file_like(obj: object) -> TypeIs[FileLike]:
    """
    Check if the object is a file-like object.

    For objects to be considered file-like, they must
    be an iterator AND have a 'read' and 'seek' method
    as an attribute.

    Note: file-like objects must be iterable, but
    iterable objects need not be file-like.

    Parameters
    ----------
    obj : The object to check.

    Returns
    -------
    is_file_like : bool
        Whether `obj` has file-like properties.

    Examples
    --------
    >>> buffer = BytesIO(b"data")
    >>> is_file_like(buffer)
    True
    >>> is_file_like([1, 2, 3])
    False
    """
    return isinstance(obj, FileLike) or isinstance(getattr(obj, "file", None), FileLike)


class UniqueDB:
    def __init__(self) -> None:
        self._db: dict[str, int] = {}

    def get_unique_name(self, name: str) -> str:
        """returns an available unique name

        Parameters
        ----------
        name : str
            name to be made unique

        Returns
        -------
        unique_name : str
            new unique name

        """

        if name not in self._db:
            self._db[name] = 0
            return name
        else:
            index = self._db[name]
            self._db[name] = index + 1
            return f"{name}_{index}"


def cut_video_stream(stream: bytes, start: float, end: float, fmt: str) -> bytes:
    """cut video stream from `start` to `end` time

    Parameters
    ----------
    stream : bytes
        video file content
    start : float
        start time
    end : float
        end time

    Returns
    -------
    result : bytes
        content of cut video

    """
    with TemporaryDirectory() as tmp:
        in_file = Path(tmp) / f"in{fmt}"
        out_file = Path(tmp) / f"out{fmt}"

        in_file.write_bytes(stream)

        try:
            ret = subprocess.run(
                [
                    "ffmpeg",
                    "-ss",
                    f"{start}",
                    "-i",
                    f"{in_file}",
                    "-to",
                    f"{end}",
                    "-c",
                    "copy",
                    f"{out_file}",
                ],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            result = stream
        else:
            if ret.returncode:
                result = stream
            else:
                result = out_file.read_bytes()

    return result


def get_video_stream_duration(stream: bytes) -> float | None:
    with TemporaryDirectory() as tmp:
        in_file = Path(tmp) / "in"
        in_file.write_bytes(stream)

        try:
            process = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    f"{in_file}",
                ],
                capture_output=True,
                check=False,
            )
            result = float(process.stdout)
        except FileNotFoundError:
            result = None
    return result


class VirtualChannelGroup:
    """starting with MDF v4.20 it is possible to use remote masters and column
    oriented storage. This means we now have virtual channel groups that can
    span over multiple regular channel groups. This class facilitates the
    handling of this virtual groups"""

    __slots__ = (
        "cycles_nr",
        "groups",
        "record_size",
    )

    def __init__(self) -> None:
        self.groups: list[int] = []
        self.record_size = 0
        self.cycles_nr = 0

    def __repr__(self) -> str:
        return f"VirtualChannelGroup(groups={self.groups}, records_size={self.record_size}, cycles_nr={self.cycles_nr})"


def block_fields(obj: object) -> list[str]:
    fields = []
    for attr in dir(obj):
        if attr[:2] + attr[-2:] == "____":
            continue
        try:
            if callable(getattr(obj, attr)):
                continue
            fields.append(f"{attr}:{getattr(obj, attr)}")
        except AttributeError:
            continue

    return fields


@overload
def components(
    channel: NDArray[Any],
    channel_name: str,
    unique_names: UniqueDB,
    prefix: str = ...,
    master: Optional["pd.Index[float]"] = ...,
    only_basenames: bool = ...,
    use_polars: Literal[False] = ...,
) -> Iterator[tuple[str, "pd.Series[Any]"]]: ...


@overload
def components(
    channel: NDArray[Any],
    channel_name: str,
    unique_names: UniqueDB,
    prefix: str = ...,
    master: Optional["pd.Index[float]"] = ...,
    only_basenames: bool = ...,
    *,
    use_polars: Literal[True],
) -> Iterator[tuple[str, "list[Any]"]]: ...


def components(
    channel: NDArray[Any],
    channel_name: str,
    unique_names: UniqueDB,
    prefix: str = "",
    master: Optional["pd.Index[float]"] = None,
    only_basenames: bool = False,
    use_polars: bool = False,
) -> Iterator[tuple[str, Union["pd.Series[Any]", list[Any]]]]:
    """yield pandas Series and unique name based on the ndarray object

    Parameters
    ----------
    channel : numpy.ndarray
        channel to be used for Series
    channel_name : str
        channel name
    unique_names : UniqueDB
        unique names object
    prefix : str
        prefix used in case of nested recarrays
    master : pd.Index
        optional index for the Series
    only_basenames (False) : bool
        use just the field names, without prefix, for structures and channel
        arrays

        .. versionadded:: 5.13.0

    use_polars (False) : bool
        use polars

        .. versionadded:: 8.1.0

    Returns
    -------
    name, series : (str, values)
        tuple of unique name and values
    """
    names = channel.dtype.names

    # channel arrays
    if names and names[0] == channel_name:
        name = names[0]

        if not only_basenames:
            if prefix:
                name_ = unique_names.get_unique_name(f"{prefix}.{name}")
            else:
                name_ = unique_names.get_unique_name(name)
        else:
            name_ = unique_names.get_unique_name(name)

        samples = channel[name]
        if samples.dtype.byteorder not in target_byte_order:
            samples = samples.byteswap().view(samples.dtype.newbyteorder())

        if len(samples.shape) > 1:
            values = (
                list(samples)
                if use_polars
                else Series(
                    list(samples),
                    index=master,
                )
            )
        elif not use_polars:
            values = Series(
                samples,
                index=master,
            )

        yield name_, values

        for name in names[1:] if names else ():
            samples = channel[name]

            if samples.dtype.byteorder not in target_byte_order:
                samples = samples.byteswap().view(samples.dtype.newbyteorder())

            if not only_basenames:
                axis_name = unique_names.get_unique_name(f"{name_}.{name}")
            else:
                axis_name = unique_names.get_unique_name(name)
            if len(samples.shape) > 1:
                values = (
                    list(samples)
                    if use_polars
                    else Series(
                        list(samples),
                        index=master,
                    )
                )
            elif not use_polars:
                values = Series(
                    samples,
                    index=master,
                )

            yield axis_name, values

    # structure composition
    else:
        for name in channel.dtype.names or ():
            samples = channel[name]

            if samples.dtype.names:
                yield from components(
                    samples,
                    name,
                    unique_names,
                    prefix=f"{prefix}.{channel_name}" if prefix else f"{channel_name}",
                    master=master,
                    only_basenames=only_basenames,
                )

            else:
                if samples.dtype.byteorder not in target_byte_order:
                    samples = samples.byteswap().view(samples.dtype.newbyteorder())

                if not only_basenames:
                    name_ = unique_names.get_unique_name(
                        f"{prefix}.{channel_name}.{name}" if prefix else f"{channel_name}.{name}"
                    )
                else:
                    name_ = unique_names.get_unique_name(name)
                if len(samples.shape) > 1:
                    values = (
                        list(samples)
                        if use_polars
                        else Series(
                            list(samples),
                            index=master,
                        )
                    )
                elif not use_polars:
                    values = Series(
                        samples,
                        index=master,
                    )

                yield name_, values


class DataBlockInfo:
    __slots__ = (
        "address",
        "block_limit",
        "block_type",
        "compressed_size",
        "first_timestamp",
        "invalidation_block",
        "last_timestamp",
        "original_size",
        "param",
    )

    def __init__(
        self,
        address: int,
        block_type: int,
        original_size: int,
        compressed_size: int,
        param: int,
        invalidation_block: Optional["InvalidationBlockInfo"] = None,
        block_limit: int | None = None,
        first_timestamp: bytes | None = None,
        last_timestamp: bytes | None = None,
    ) -> None:
        self.address = address
        self.block_type = block_type
        self.original_size = original_size
        self.compressed_size = compressed_size
        self.param = param
        self.invalidation_block = invalidation_block
        self.block_limit = block_limit
        self.first_timestamp = first_timestamp
        self.last_timestamp = last_timestamp

    def __repr__(self) -> str:
        return (
            f"DataBlockInfo(address=0x{self.address:X}, "
            f"block_type={self.block_type}, "
            f"original_size={self.original_size}, "
            f"compressed_size={self.compressed_size}, "
            f"param={self.param}, "
            f"invalidation_block={self.invalidation_block}, "
            f"block_limit={self.block_limit}, "
            f"first_timestamp={self.first_timestamp!r}, "
            f"last_timestamp={self.last_timestamp!r})"
        )


class Fragment:
    def __init__(
        self,
        data: bytes,
        record_offset: int = -1,
        record_count: int = -1,
        invalidation_data: bytes | None = None,
        is_record: bool = True,
    ) -> None:
        self.data = data
        self.record_count = record_count
        self.record_offset = record_offset
        self.invalidation_data = invalidation_data
        self.is_record = is_record

    def __repr__(self) -> str:
        return (
            f"FragmentInfo({len(self.data)} bytes, "
            f"record_offset={self.record_offset}, "
            f"record_count={self.record_count}, "
            f"is_record={self.is_record})"
        )


class InvalidationBlockInfo(DataBlockInfo):
    __slots__ = ("all_valid",)

    def __init__(
        self,
        address: int,
        block_type: int,
        original_size: int,
        compressed_size: int,
        param: int,
        all_valid: bool = False,
        block_limit: int | None = None,
    ) -> None:
        super().__init__(address, block_type, original_size, compressed_size, param, block_limit=block_limit)
        self.all_valid = all_valid

    def __repr__(self) -> str:
        return (
            f"InvalidationBlockInfo(address=0x{self.address:X}, "
            f"block_type={self.block_type}, "
            f"original_size={self.original_size}, "
            f"compressed_size={self.compressed_size}, "
            f"param={self.param}, "
            f"all_valid={self.all_valid}, "
            f"block_limit={self.block_limit})"
        )


class SignalDataBlockInfo:
    __slots__ = (
        "address",
        "block_type",
        "compressed_size",
        "location",
        "original_size",
        "param",
    )

    def __init__(
        self,
        address: int,
        original_size: int,
        block_type: int = v4c.DT_BLOCK,
        param: int = 0,
        compressed_size: int | None = None,
        location: int = v4c.LOCATION_ORIGINAL_FILE,
    ) -> None:
        self.address = address
        self.compressed_size = compressed_size or original_size
        self.block_type = block_type
        self.original_size = original_size
        self.param = param
        self.location = location

    def __repr__(self) -> str:
        return (
            f"SignalDataBlockInfo(address=0x{self.address:X}, "
            f"original_size={self.original_size}, "
            f"compressed_size={self.compressed_size}, "
            f"block_type={self.block_type})"
        )


def get_fields(obj: object) -> list[str]:
    fields: list[str] = []
    for attr in dir(obj):
        if attr[:2] + attr[-2:] == "____":
            continue
        try:
            if callable(getattr(obj, attr)):
                continue
            fields.append(attr)
        except AttributeError:
            continue
    return fields


# code snippet taken from https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def downcast(array: NDArray[Any]) -> NDArray[Any]:
    kind = array.dtype.kind
    if kind == "f":
        array = array.astype(np.float32)
    elif kind in "ui":
        min_ = array.min()
        max_ = array.max()
        if min_ >= 0:
            if max_ < 255:
                array = array.astype(np.uint8)
            elif max_ < 65535:
                array = array.astype(np.uint16)
            elif max_ < 4294967295:
                array = array.astype(np.uint32)
            else:
                array = array.astype(np.uint64)
        else:
            if min_ > np.iinfo(np.int8).min and max_ < np.iinfo(np.int8).max:
                array = array.astype(np.int8)
            elif min_ > np.iinfo(np.int16).min and max_ < np.iinfo(np.int16).max:
                array = array.astype(np.int16)
            elif min_ > np.iinfo(np.int32).min and max_ < np.iinfo(np.int32).max:
                array = array.astype(np.int32)
            elif min_ > np.iinfo(np.int64).min and max_ < np.iinfo(np.int64).max:
                array = array.astype(np.int64)

    return array


def csv_int2bin(val: int) -> str:
    """format CAN id as bin

    100 -> 1100100

    """

    return f"{val:b}"


csv_int2bin = np.vectorize(csv_int2bin, otypes=[str])


def csv_int2hex(val: "pd.Series[bool]") -> str:
    """format CAN id as hex

    100 -> 64

    """

    return f"{val:X}"


csv_int2hex = np.vectorize(csv_int2hex, otypes=[str])


def csv_bytearray2hex(val: NDArray[Any], size: int | None = None) -> str:
    """format CAN payload as hex strings

    b'\xa2\xc3\x08' -> A2 C3 08

    """
    if size is not None:
        hex_val = typing.cast(bytes, val.tobytes())[:size].hex(" ", 1).upper()  # type: ignore[redundant-cast,unused-ignore]
    else:
        try:
            hex_val = typing.cast(bytes, val.tobytes()).hex(" ", 1).upper()  # type: ignore[redundant-cast,unused-ignore]
        except:
            hex_val = "●"

    return hex_val


csv_bytearray2hex = np.vectorize(csv_bytearray2hex, otypes=[str])


def pandas_query_compatible(name: str) -> str:
    """adjust column name for usage in dataframe query string"""

    for c in ".$[]: ":
        name = name.replace(c, "_")

    if name.startswith(tuple(string.digits)):
        name = "file_" + name
    try:
        exec(f"from pandas import {name}")
    except ImportError:
        pass
    else:
        name = f"{name}__"
    return name


class _Kwargs(TypedDict, total=False):
    fd: bool
    load_flat: bool
    cluster_name: str


def load_can_database(
    path: str | PathLike[str], contents: bytes | str | None = None, **kwargs: Unpack[_Kwargs]
) -> CanMatrix | None:
    """


    Parameters
    ----------
    path : StrPathType
        database path
    contents: bytes | str | None = None
        optional database content
    kwargs : dict

        fd : bool = False
            if supplied, only buses with the same FD kind will be loaded

        load_flat : bool = False
            if supplied all the CAN messages found in multiple buses will be contained
            in the CAN database object. By default the first bus will be returned

        cluster_name : str
            if supplied load just the clusters with this name

    Returns
    -------
    db : canmatrix.CanMatrix | None
        CAN database object or None

    """
    path = Path(path)
    import_type = path.suffix.lstrip(".").lower()

    try:
        if contents is None:
            dbs = canmatrix.formats.loadp(str(path), import_type=import_type, key="db", **kwargs)
        else:
            dbs = canmatrix.formats.loads(contents, import_type=import_type, key="db", **kwargs)
    except UnicodeDecodeError:
        if contents is None:
            contents = path.read_bytes()

        encoding = detect(contents)["encoding"]

        if encoding:
            try:
                dbs = canmatrix.formats.loads(
                    contents,
                    import_type=import_type,
                    key="db",
                    encoding=encoding,
                    **kwargs,
                )
            except:
                dbs = None

    if dbs:
        # filter only CAN clusters
        dbs = {name: db for name, db in dbs.items() if db.type == matrix_class.CAN}

    if dbs:
        cluster_name = kwargs.get("cluster_name", None)
        if cluster_name is not None:
            dbs = {name: db for name, db in dbs.items() if name == cluster_name}

        if "fd" in kwargs:
            fd = kwargs["fd"]
            dbs = {name: db for name, db in dbs.items() if db.contains_fd == fd}

        if kwargs.get("load_flat", False):
            can_matrix, *rest = list(dbs.values())
            can_matrix.merge(rest)

        else:
            first_bus = list(dbs)[0]
            can_matrix = dbs[first_bus]
    else:
        can_matrix = None

    return can_matrix


def all_blocks_addresses(obj: FileLike | mmap.mmap) -> tuple[dict[int, bytes], dict[bytes, list[int]], list[int]]:
    DG = "DG\x00\x00\x00\x00\x40\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00"
    others = "(D[VTZIL]|AT|C[AGHNC]|EV|FH|HL|LD|MD|R[DVI]|S[IRD]|TX|GD)\x00\x00\x00\x00"
    pattern = re.compile(
        f"(?P<block>##({DG}|{others}))".encode("ascii"),
        re.DOTALL | re.MULTILINE,
    )

    try:
        obj.seek(0)
    except:
        pass

    source: Buffer | bytes
    try:
        re.search(pattern, obj)  # type: ignore[arg-type]
        source = typing.cast(Buffer, obj)
    except TypeError:
        source = obj.read()

    addresses: list[int] = []
    block_groups: dict[bytes, list[int]] = {}
    blocks: dict[int, bytes] = {}

    for match in re.finditer(pattern, source):
        btype: bytes = match.group("block")[:4]
        start = match.start()

        if start % 8:
            continue

        btype_addresses = block_groups.setdefault(btype, [])
        btype_addresses.append(start)
        addresses.append(start)
        blocks[start] = btype

    return blocks, block_groups, addresses


def plausible_timestamps(
    t: NDArray[Any],
    minimum: float,
    maximum: float,
    exp_min: int = -15,
    exp_max: int = 15,
) -> tuple[bool, NDArray[bool_]]:
    """check if the time stamps are plausible

    Parameters
    ----------
    t : np.array
        time stamps array
    minimum : float
        minimum plausible time stamp
    maximum : float
        maximum plausible time stamp
    exp_min (-15) : int
        minimum plausible exponent used for the time stamps float values
    exp_max (15) : int
        maximum plausible exponent used for the time stamps float values

    Returns
    -------
    all_ok, idx : (bool, np.array)
        the *all_ok* flag to indicate if all the time stamps are ok; this can be checked
        before applying the indexing array.
    """

    exps = np.log10(t)
    idx = (~np.isnan(t)) & (~np.isinf(t)) & (t >= minimum) & (t <= maximum) & (t == 0) | (
        (exps >= exp_min) & (exps <= exp_max)
    )
    if not np.all(idx):
        all_ok = False
        return all_ok, idx
    else:
        all_ok = True
        return all_ok, idx


table = str.maketrans(
    {
        "<": "&lt;",
        ">": "&gt;",
        "&": "&amp;",
        "'": "&apos;",
        '"': "&quot;",
    }
)


def escape_xml_string(string: str) -> str:
    return string.translate(table)


def extract_mime_names(data: "QtCore.QMimeData", disable_new_channels: bool | None = None) -> list[str]:
    def fix_comparison_name(data: Any, disable_new_channels: bool | None = None) -> None:
        for item in data:
            if item["type"] == "channel":
                if disable_new_channels is not None:
                    item["enabled"] = not disable_new_channels

                if (item["group_index"], item["channel_index"]) != (-1, -1):
                    match = COMPARISON_NAME.match(item["name"])
                    if match is None:
                        raise RuntimeError(f"cannot parse '{item['name']}'")
                    name = match.group("name").strip()
                    item["name"] = name
            else:
                if disable_new_channels is not None:
                    item["enabled"] = not disable_new_channels
                fix_comparison_name(item["channels"], disable_new_channels=disable_new_channels)

    names: list[str] = []
    if data.hasFormat("application/octet-stream-asammdf"):
        data_data = data.data("application/octet-stream-asammdf").data()
        data_bytes = data_data.tobytes() if isinstance(data_data, memoryview) else data_data
        text = data_bytes.decode("utf-8")
        obj = json.loads(text)
        fix_comparison_name(obj, disable_new_channels=disable_new_channels)
        names = obj

    return names


def set_mime_enable(mime: list[Any], enable: bool) -> None:
    for item in mime:
        if item["type"] == "channel":
            item["enabled"] = enable
        else:
            set_mime_enable(item["channels"], enable)


class _ChannelBaseDict(TypedDict):
    color: str
    comment: str | None
    common_axis: bool
    enabled: bool
    flags: int
    fmt: str
    individual_axis: bool
    name: str
    origin_uuid: str
    precision: int
    ranges: list[dict[str, object]]
    type: Literal["channel"]
    unit: str


class _ChannelNotComputedDict(_ChannelBaseDict):
    computed: Literal[False]
    conversion: NotRequired[dict[str, object]]
    mode: Literal["phys"]
    y_range: list[float]


class _ChannelComputedDict(_ChannelBaseDict):
    computation: dict[str, object]
    computed: Literal[True]
    conversion: object
    user_defined_name: str | None


_ChannelDict = _ChannelComputedDict | _ChannelNotComputedDict


class _ChannelGroupDict(TypedDict):
    channels: list[Union[_ChannelDict, "_ChannelGroupDict"]]
    enabled: bool
    name: str | None
    origin_uuid: str
    pattern: dict[str, object] | None
    ranges: list[dict[str, object]]
    type: Literal["group"]


def load_dsp(
    file: Path, background: str = "#000000", flat: bool = False, colors_as_string: bool = False
) -> dict[str, object] | list[str]:
    if not colors_as_string and isinstance(background, str):
        background = fn.mkColor(background)

    def parse_conversions(display: lxml.etree._Element | None) -> dict[str | None, dict[str, object]]:
        conversions: dict[str | None, dict[str, object]] = {}

        if display is None:
            return conversions

        for item in display.findall("COMPU_METHOD"):
            try:
                name = item.get("name")
                conv: dict[str, object] = {
                    "name": name,
                    "comment": item.get("description"),
                    "unit": item.get("unit"),
                }

                conversion_type = int(item.attrib["cnv_type"])
                match conversion_type:
                    case 0:
                        conv["conversion_type"] = v4c.CONVERSION_TYPE_LIN

                        coeffs = item.find("COEFFS_LINIAR")

                        if coeffs is None:
                            raise RuntimeError("cannot find 'COEFFS_LINIAR' element")

                        conv["a"] = float(coeffs.attrib["P1"])
                        conv["b"] = float(coeffs.attrib["P2"])

                    case 9:
                        conv["conversion_type"] = v4c.CONVERSION_TYPE_RAT

                        coeffs = item.find("COEFFS")

                        if coeffs is None:
                            raise RuntimeError("cannot find 'COEFFS' element")

                        for i in range(1, 7):
                            conv[f"P{i}"] = float(coeffs.attrib[f"P{i}"])

                    case 11:
                        conv["conversion_type"] = v4c.CONVERSION_TYPE_TABX
                        vtab = item.find("COMPU_VTAB")

                        if vtab is not None:
                            for i, item in enumerate(vtab.findall("tab")):
                                conv[f"val_{i}"] = float(item.attrib["min"])
                                text = item.get("text")
                                if isinstance(text, bytes):
                                    text = text.decode("utf-8", errors="replace")
                                conv[f"text_{i}"] = text

                    case 12:
                        conv["conversion_type"] = v4c.CONVERSION_TYPE_RTABX
                        vtab = item.find("COMPU_VTAB_RANGE")

                        if vtab is not None:
                            text = vtab.get("default")
                            if isinstance(text, bytes):
                                text = text.decode("utf-8", errors="replace")
                            conv["default_addr"] = vtab.get("default")
                            for i, item in enumerate(vtab.findall("tab_range")):
                                conv[f"upper_{i}"] = float(item.attrib["max"])
                                conv[f"lower_{i}"] = float(item.attrib["min"])
                                text = item.get("text")
                                if isinstance(text, bytes):
                                    text = text.decode("utf-8", errors="replace")
                                conv[f"text_{i}"] = text
                    case _:
                        continue

                conversions[name] = conv

            except:
                print(format_exc())
                continue

        return conversions

    def parse_channels(
        display: lxml.etree._Element, conversions: dict[str | None, dict[str, object]]
    ) -> list[_ChannelGroupDict | _ChannelDict]:
        channels: list[_ChannelGroupDict | _ChannelDict] = []
        for elem in display.iterchildren():
            if elem.tag == "CHANNEL":
                channel_name = elem.attrib["name"]

                comment_elem = elem.find("COMMENT")
                if comment_elem is not None:
                    comment = elem.get("text")
                else:
                    comment = ""

                color_ = int(elem.attrib["color"])
                c = 0
                for i in range(3):
                    c = c << 8
                    c += color_ & 0xFF
                    color_ = color_ >> 8

                ch_color = c

                gain = abs(float(elem.attrib["gain"]))
                offset = float(elem.attrib["offset"]) / 100

                multi_color = elem.find("MULTI_COLOR")

                ranges: list[dict[str, object]] = []

                if multi_color is not None:
                    for color in multi_color.findall("color"):
                        some_elem = color.find("min")
                        if some_elem is None:
                            raise RuntimeError("cannot find element 'min'")
                        min_ = float(some_elem.attrib["data"])
                        some_elem = color.find("max")
                        if some_elem is None:
                            raise RuntimeError("cannot find element 'max'")
                        max_ = float(some_elem.attrib["data"])
                        some_elem = color.find("color")
                        if some_elem is None:
                            raise RuntimeError("cannot find element 'color'")
                        color_ = int(some_elem.attrib["data"])
                        c = 0
                        for i in range(3):
                            c = c << 8
                            c += color_ & 0xFF
                            color_ = color_ >> 8
                        font_color = f"#{c:06X}" if colors_as_string else fn.mkColor(f"#{c:06X}")
                        ranges.append(
                            {
                                "background_color": background,
                                "font_color": font_color,
                                "op1": "<=",
                                "op2": "<=",
                                "value1": min_,
                                "value2": max_,
                            }
                        )

                chan: _ChannelNotComputedDict = {
                    "color": f"#{ch_color:06X}",
                    "common_axis": False,
                    "computed": False,
                    "flags": 0,
                    "comment": comment,
                    "enabled": elem.get("on") == "1" and elem.get("trc_fmt") != "2",
                    "fmt": "{}",
                    "individual_axis": False,
                    "name": channel_name,
                    "mode": "phys",
                    "precision": 3,
                    "ranges": ranges,
                    "unit": "",
                    "type": "channel",
                    "y_range": sorted(
                        [
                            -gain * offset,
                            -gain * offset + 19 * gain,
                        ]
                    ),
                    "origin_uuid": "000000000000",
                }

                conv_name = elem.get("cnv_name")
                if conv_name in conversions:
                    chan["conversion"] = deepcopy(conversions[conv_name])

                channels.append(chan)

            elif str(elem.tag).startswith("GROUP"):
                channels.append(
                    {
                        "name": elem.get("data"),
                        "enabled": elem.get("on") == "1",
                        "type": "group",
                        "channels": parse_channels(elem, conversions=conversions),
                        "pattern": None,
                        "origin_uuid": "000000000000",
                        "ranges": [],
                    }
                )

            elif elem.tag == "CHANNEL_PATTERN":
                try:
                    filter_type = elem.get("filter_type")
                    filter_value: float
                    if filter_type in ("None", None):
                        filter_type = "Unspecified"
                        filter_value = 0
                        raw = False
                    else:
                        filter_value = float(elem.attrib["filter_value"])
                        raw = bool(int(elem.attrib["filter_use_raw"]))

                    info: dict[str, object] = {
                        "pattern": elem.get("name_pattern"),
                        "name": elem.get("name_pattern"),
                        "match_type": "Wildcard",
                        "filter_type": filter_type,
                        "filter_value": filter_value,
                        "raw": raw,
                    }

                    multi_color = elem.find("MULTI_COLOR")

                    ranges = []

                    if multi_color is not None:
                        for color in multi_color.findall("color"):
                            some_elem = color.find("min")
                            if some_elem is None:
                                raise RuntimeError("cannot find element 'min'")
                            min_ = float(some_elem.attrib["data"])
                            some_elem = color.find("max")
                            if some_elem is None:
                                raise RuntimeError("cannot find element 'max'")
                            max_ = float(some_elem.attrib["data"])
                            some_elem = color.find("color")
                            if some_elem is None:
                                raise RuntimeError("cannot find element 'color'")
                            color_ = int(some_elem.attrib["data"])
                            c = 0
                            for i in range(3):
                                c = c << 8
                                c += color_ & 0xFF
                                color_ = color_ >> 8
                            font_color = f"#{c:06X}" if colors_as_string else fn.mkColor(f"#{c:06X}")
                            ranges.append(
                                {
                                    "background_color": background,
                                    "font_color": font_color,
                                    "op1": "<=",
                                    "op2": "<=",
                                    "value1": min_,
                                    "value2": max_,
                                }
                            )

                    info["ranges"] = ranges

                    channels.append(
                        {
                            "channels": [],
                            "enabled": True,
                            "name": typing.cast(str, info["pattern"]),
                            "pattern": info,
                            "type": "group",
                            "ranges": [],
                            "origin_uuid": "000000000000",
                        }
                    )

                except:
                    print(format_exc())
                    continue

        return channels

    def parse_virtual_channels(display: lxml.etree._Element | None) -> dict[str | None, dict[str, object]]:
        channels: dict[str | None, dict[str, object]] = {}

        if display is None:
            return channels

        for item in display.findall("V_CHAN"):
            try:
                virtual_channel: dict[str, object] = {}

                parent = item.find("VIR_TIME_CHAN")
                vtab = item.find("COMPU_VTAB")
                if parent is None or vtab is None:
                    continue

                name = item.get("name")

                virtual_channel["name"] = name
                virtual_channel["parent"] = parent.attrib["data"]
                elem = item.find("description")
                if elem is None:
                    raise RuntimeError("cannot find element 'description'")
                virtual_channel["comment"] = elem.attrib["data"]

                conv: dict[str, object] = {}
                for i, item in enumerate(vtab.findall("tab")):
                    conv[f"val_{i}"] = float(item.attrib["min"])
                    text = item.get("text")
                    if isinstance(text, bytes):
                        text = text.decode("utf-8", errors="replace")
                    conv[f"text_{i}"] = text

                virtual_channel["vtab"] = conv

                channels[name] = virtual_channel
            except:
                continue

        return channels

    def parse_c_functions(display: lxml.etree._Element | None) -> Collection[str]:
        c_functions: set[str] = set()

        if display is None:
            return c_functions

        for item in display.findall("CALC_FUNC"):
            string = item.text

            if string is None:
                raise RuntimeError("element text is None")

            for match in C_FUNCTION.finditer(string):
                c_functions.add(match.group("function"))

        return sorted(c_functions)

    dsp = lxml.etree.fromstring(Path(file).read_bytes().replace(b"\0", b""), parser=lxml.etree.XMLParser(recover=True))

    conversions = parse_conversions(dsp.find("COMPU_METHODS"))

    elem = dsp.find("DISPLAY_INFO")

    if elem is None:
        raise RuntimeError("cannot find element 'DISPLAY_INFO'")

    channels = parse_channels(elem, conversions)
    c_functions = parse_c_functions(dsp)

    functions: dict[str, object] = {}
    virtual_channels: list[_ChannelGroupDict | _ChannelDict] = []

    for i, ch in enumerate(parse_virtual_channels(dsp.find("VIRTUAL_CHANNEL")).values()):
        virtual_channels.append(
            {
                "color": COLORS[i % len(COLORS)],
                "common_axis": False,
                "computed": True,
                "computation": {
                    "args": {"arg1": []},
                    "type": "python_function",
                    "channel_comment": ch["comment"],
                    "channel_name": ch["name"],
                    "channel_unit": "",
                    "function": f"f_{ch['name']}",
                    "triggering": "triggering_on_all",
                    "triggering_value": "all",
                },
                "flags": int(SignalFlags.computed | SignalFlags.user_defined_conversion),
                "enabled": True,
                "fmt": "{}",
                "individual_axis": False,
                "name": typing.cast(str, ch["parent"]),
                "precision": 3,
                "ranges": [],
                "unit": "",
                "conversion": ch["vtab"],
                "user_defined_name": typing.cast(str | None, ch["name"]),
                "comment": f"Datalyser virtual channel: {ch['comment']}",
                "origin_uuid": "000000000000",
                "type": "channel",
            }
        )

        functions[f"f_{ch['name']}"] = f"def f_{ch['name']}(arg1=0, t=0):\n    return arg1"

    if virtual_channels:
        channels.append(
            {
                "name": "Datalyser Virtual Channels",
                "enabled": False,
                "type": "group",
                "channels": virtual_channels,
                "pattern": None,
                "origin_uuid": "000000000000",
                "ranges": [],
            }
        )

    windows: list[dict[str, object]] = []
    info: dict[str, object] | list[str] = {
        "selected_channels": [],
        "windows": windows,
        "has_virtual_channels": bool(virtual_channels),
        "c_functions": c_functions,
        "functions": functions,
    }

    if flat:
        info = flatten_dsp(channels)
    else:
        plot: dict[str, object] = {
            "type": "Plot",
            "title": "Display channels",
            "maximized": True,
            "minimized": False,
            "configuration": {
                "channels": channels,
                "locked": True,
                "pattern": {},
            },
        }

        windows.append(plot)

    return info


def flatten_dsp(channels: list[_ChannelGroupDict | _ChannelDict]) -> list[str]:
    res: list[str] = []

    for item in channels:
        if item["type"] == "group":
            res.extend(flatten_dsp(item["channels"]))
        else:
            res.append(item["name"])

    return res


def load_channel_names_from_file(file_name: str, lab_section: str = "") -> list[str]:
    file_path = Path(file_name)
    channels: Collection[str]
    extension = file_path.suffix.lower()
    match extension:
        case ".dsp":
            channels = load_dsp(file_path, flat=True)

        case ".dspf":
            with open(file_path) as infile:
                info = json.load(infile)

            channels = []
            for window in info["windows"]:
                if window["type"] == "Plot":
                    channels.extend(flatten_dsp(window["configuration"]["channels"]))
                elif window["type"] == "Numeric":
                    channels.extend([item["name"] for item in window["configuration"]["channels"]])
                elif window["type"] == "Tabular":
                    channels.extend(window["configuration"]["channels"])

        case ".lab":
            info = load_lab(file_path)
            if info:
                if len(info) > 1 and lab_section:
                    channels = info[lab_section]
                else:
                    channels = list(info.values())[0]

                channels = [name.split(";")[0] for name in channels]

        case ".cfg":
            with open(file_path) as infile:
                info = json.load(infile)
            channels = info.get("selected_channels", [])
        case ".txt":
            try:
                with open(file_path) as infile:
                    info = json.load(infile)
                channels = info.get("selected_channels", [])
            except:
                with open(file_path) as infile:
                    channels = [line.strip() for line in infile.readlines()]
                    channels = [name for name in channels if name]

    return sorted(set(channels))


def load_lab(file: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    with open(file) as lab:
        for line in lab:
            line = line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                section_name = line.strip("[]")
                s: list[str] = []
                sections[section_name] = s

            else:
                if "s" in locals():
                    s.append(line)

    return {name: channels for name, channels in sections.items() if channels if name != "SETTINGS"}


class SignalFlags:
    no_flags = 0x0
    user_defined_comment = 0x1
    user_defined_conversion = 0x2
    user_defined_unit = 0x4
    user_defined_name = 0x8
    stream_sync = 0x10
    computed = 0x20
    virtual = 0x40
    virtual_master = 0x80


_Params = ParamSpec("_Params")
_Ret = TypeVar("_Ret")


def timeit(func: Callable[_Params, _Ret]) -> Callable[_Params, _Ret]:
    def timed(*args: _Params.args, **kwargs: _Params.kwargs) -> _Ret:
        t1 = perf_counter()
        ret = func(*args, **kwargs)
        t2 = perf_counter()
        delta = t2 - t1
        if delta >= 1e-3:
            print(f"CALL {func.__qualname__}: {delta*1e3:.3f} ms")
        else:
            print(f"CALL {func.__qualname__}: {delta*1e6:.3f} us")
        return ret

    return timed


class Timer:

    def __init__(self, name: str = "") -> None:
        self.name = name or str(id(self))
        self.count = 0
        self.total_time = 0.0

    def __enter__(self) -> "Timer":
        now = perf_counter()
        self.start = now
        return self

    def __exit__(
        self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        now = perf_counter()
        self.total_time += now - self.start
        self.count += 1

    def display(self) -> None:
        if self.count:
            for factor, r, unit in ((1e3, 3, "ms"), (1e6, 6, "us"), (1e9, 9, "ns")):
                tpi = round(self.total_time / self.count, r)
                if tpi:
                    break
            print(
                f"""TIMER {self.name}:
\t* {self.count} iterations in {self.total_time * 1000:.3f}ms
\t* {self.count / self.total_time:.3f} iter/s
\t* {self.total_time / self.count * factor:.3f} {unit}/iter"""
            )
        else:
            print(f"TIMER {self.name}:\n\t* inactive")

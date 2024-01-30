"""
asammdf utility functions and classes
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from functools import lru_cache
from io import StringIO
import json
import logging
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
from typing import Any, Dict, overload, Tuple
import xml.etree.ElementTree as ET

import lxml
from typing_extensions import Literal, TypedDict

TERMINATED = object()
COMPARISON_NAME = re.compile(r"(\s*\d+:)?(?P<name>.+)")
C_FUNCTION = re.compile(r"\s+(?P<function>\S+)\s*\(\s*struct\s+DATA\s+\*data\s*\)")
target_byte_order = "<=" if sys.byteorder == "little" else ">="

COLORS = [
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
]
COLORS_COUNT = len(COLORS)

try:
    from pyqtgraph import functions as fn
except ImportError:

    class fn:
        @classmethod
        def mkColor(cls, color):
            return color

        @classmethod
        def mkPen(cls, color):
            return color

        @classmethod
        def mkBrush(cls, color):
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
            for encoding in ("utf-8", "latin-1", "cp1250", "cp1252"):
                try:
                    text.decode(encoding)
                    break
                except:
                    continue
            else:
                encoding = None
            return {"encoding": encoding}


from canmatrix.canmatrix import CanMatrix, matrix_class
import canmatrix.formats
import numpy as np
from numpy import arange, bool_, dtype, interp, where
from numpy.typing import NDArray
from pandas import Series

from ..types import (
    ChannelType,
    DataGroupType,
    MDF_v2_v3_v4,
    RasterType,
    ReadableBufferType,
    StrPathType,
)
from . import v2_v3_constants as v3c
from . import v4_constants as v4c

UINT8_u = Struct("<B").unpack
UINT16_u = Struct("<H").unpack
UINT32_p = Struct("<I").pack
UINT32_u = Struct("<I").unpack
UINT64_u = Struct("<Q").unpack
UINT8_uf = Struct("<B").unpack_from
UINT16_uf = Struct("<H").unpack_from
UINT32_uf = Struct("<I").unpack_from
UINT64_uf = Struct("<Q").unpack_from
FLOAT64_u = Struct("<d").unpack
FLOAT64_uf = Struct("<d").unpack_from
TWO_UINT64_u = Struct("<2Q").unpack
TWO_UINT64_uf = Struct("<2Q").unpack_from
BLK_COMMON_uf = Struct("<4s4xQ").unpack_from
BLK_COMMON_u = Struct("<4s4xQ8x").unpack

EMPTY_TUPLE = ()

_xmlns_pattern = re.compile(' xmlns="[^"]*"')

logger = logging.getLogger("asammdf")

__all__ = [
    "CHANNEL_COUNT",
    "CONVERT",
    "MERGE",
    "ChannelsDB",
    "UniqueDB",
    "MdfException",
    "get_fmt_v3",
    "get_fmt_v4",
    "get_text_v4",
    "fmt_to_datatype_v3",
    "fmt_to_datatype_v4",
    "matlab_compatible",
    "extract_xml_comment",
    "validate_version_argument",
    "MDF2_VERSIONS",
    "MDF3_VERSIONS",
    "MDF4_VERSIONS",
    "SUPPORTED_VERSIONS",
]

CHANNEL_COUNT = (1000, 2000, 10000, 20000)
_channel_count = arange(0, 20000, 1000, dtype="<u4")

CONVERT = (10 * 2**20, 20 * 2**20, 30 * 2**20, 40 * 2**20)
CONVERT = interp(_channel_count, CHANNEL_COUNT, CONVERT).astype("<u4")

MERGE = (10 * 2**20, 20 * 2**20, 35 * 2**20, 60 * 2**20)
MERGE = interp(_channel_count, CHANNEL_COUNT, MERGE).astype("<u4")

CHANNEL_COUNT = _channel_count

MDF2_VERSIONS = ("2.00", "2.10", "2.14")
MDF3_VERSIONS = ("3.00", "3.10", "3.20", "3.30")
MDF4_VERSIONS = ("4.00", "4.10", "4.11", "4.20")
SUPPORTED_VERSIONS = MDF2_VERSIONS + MDF3_VERSIONS + MDF4_VERSIONS


ALLOWED_MATLAB_CHARS = set(string.ascii_letters + string.digits + "_")


class MdfException(Exception):
    """MDF Exception class"""

    def __repr__(self):
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
        comment = ET.fromstring(comment)
        match = comment.find(".//TX")
        if match is None:
            common_properties = comment.find(".//common_properties")
            if common_properties is not None:
                comment = []
                for e in common_properties:
                    field = f'{e.get("name")}: {e.text}'
                    comment.append(field)
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

    compatible_name = [ch if ch in ALLOWED_MATLAB_CHARS else "_" for ch in name]
    compatible_name = "".join(compatible_name)

    if compatible_name[0] not in string.ascii_letters:
        compatible_name = "M_" + compatible_name

    # max variable name is 63 and 3 chars are reserved
    # for get_unique_name in case of multiple channel name occurrence
    return compatible_name[:60]


@overload
def get_text_v3(
    address: int,
    stream: ReadableBufferType,
    mapped: bool = ...,
    decode: Literal[True] = ...,
) -> str: ...


@overload
def get_text_v3(
    address: int,
    stream: ReadableBufferType,
    mapped: bool = ...,
    decode: Literal[False] = ...,
) -> bytes: ...


def get_text_v3(address: int, stream: ReadableBufferType, mapped: bool = False, decode: bool = True) -> str | bytes:
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

    if mapped:
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
    if decode:
        try:
            text = text_bytes.decode("latin-1")
        except UnicodeDecodeError:
            try:
                encoding = detect(text_bytes)["encoding"]
                text = text_bytes.decode(encoding, "ignore")
            except:
                text = "<!text_decode_error>"
    else:
        text = text_bytes

    return text


@overload
def get_text_v4(
    address: int,
    stream: ReadableBufferType,
    mapped: bool = ...,
    decode: Literal[True] = ...,
) -> str: ...


@overload
def get_text_v4(
    address: int,
    stream: ReadableBufferType,
    mapped: bool = ...,
    decode: Literal[False] = ...,
) -> bytes: ...


def get_text_v4(address: int, stream: ReadableBufferType, mapped: bool = False, decode: bool = True) -> str | bytes:
    """faster way to extract strings from mdf version 4 TextBlock

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

    if mapped:
        block_id, size = BLK_COMMON_uf(stream, address)
        if block_id not in (b"##TX", b"##MD"):
            return "" if decode else b""
        text_bytes = stream[address + 24 : address + size].split(b"\0", 1)[0].strip(b" \r\t\n")
    else:
        stream.seek(address)
        block_id, size = BLK_COMMON_u(stream.read(24))
        if block_id not in (b"##TX", b"##MD"):
            return "" if decode else b""
        text_bytes = stream.read(size - 24).split(b"\0", 1)[0].strip(b" \r\t\n")

    if decode:
        try:
            text = text_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                encoding = detect(text_bytes)["encoding"]
                text = text_bytes.decode(encoding, "ignore")
            except:
                text = "<!text_decode_error>"
    else:
        text = text_bytes

    return text


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
                display_names[elem.text.strip(" \t\r\n\v\0")] = elem.tag

        except:
            pass

    return display_names


def extract_encryption_information(comment: str) -> dict[str, str]:
    info = {}
    comment = comment.replace(' xmlns="http://www.asam.net/mdf/v4"', "")
    if comment.startswith("<ATcomment") and "<encrypted>" in comment:
        try:
            comment = ET.fromstring(comment)
            for match in comment.findall(".//extensions/extension"):
                encrypted = match.find("encrypted").text.strip().lower() == "true"
                algorithm = match.find("algorithm").text.strip().lower()
                original_md5_sum = match.find("original_md5_sum").text.strip().lower()
                original_size = int(match.find("original_size").text)

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
        comment = ET.fromstring(comment)
        match = comment.find(".//tool")
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

            if data_type == v3c.DATA_TYPE_UNSIGNED_INTEL:
                fmt = f"<u{size}"

            elif data_type == v3c.DATA_TYPE_UNSIGNED:
                if byte_order == v3c.BYTE_ORDER_INTEL:
                    fmt = f"<u{size}"
                else:
                    fmt = f">u{size}"

            elif data_type == v3c.DATA_TYPE_UNSIGNED_MOTOROLA:
                fmt = f">u{size}"

            elif data_type == v3c.DATA_TYPE_SIGNED_INTEL:
                fmt = f"<i{size}"

            elif data_type == v3c.DATA_TYPE_SIGNED:
                if byte_order == v3c.BYTE_ORDER_INTEL:
                    fmt = f"<i{size}"
                else:
                    fmt = f">i{size}"

            elif data_type == v3c.DATA_TYPE_SIGNED_MOTOROLA:
                fmt = f">i{size}"

            elif data_type in (v3c.DATA_TYPE_FLOAT_INTEL, v3c.DATA_TYPE_DOUBLE_INTEL):
                fmt = f"<f{size}"

            elif data_type in (
                v3c.DATA_TYPE_FLOAT_MOTOROLA,
                v3c.DATA_TYPE_DOUBLE_MOTOROLA,
            ):
                fmt = f">f{size}"

            elif data_type in (v3c.DATA_TYPE_FLOAT, v3c.DATA_TYPE_DOUBLE):
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
        size = size // 8

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
        if data_type == v4c.DATA_TYPE_UNSIGNED_INTEL:
            fmt = "<u8"

        elif data_type == v4c.DATA_TYPE_UNSIGNED_MOTOROLA:
            fmt = ">u8"

        elif data_type == v4c.DATA_TYPE_SIGNED_INTEL:
            fmt = "<i8"

        elif data_type == v4c.DATA_TYPE_SIGNED_MOTOROLA:
            fmt = ">i8"

        elif data_type == v4c.DATA_TYPE_REAL_INTEL:
            fmt = "<f8"

        elif data_type == v4c.DATA_TYPE_REAL_MOTOROLA:
            fmt = ">f8"
        elif data_type == v4c.DATA_TYPE_COMPLEX_INTEL:
            fmt = "<c8"
        elif data_type == v4c.DATA_TYPE_COMPLEX_MOTOROLA:
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

            if data_type == v4c.DATA_TYPE_UNSIGNED_INTEL:
                fmt = f"<u{size}"

            elif data_type == v4c.DATA_TYPE_UNSIGNED_MOTOROLA:
                fmt = f">u{size}"

            elif data_type == v4c.DATA_TYPE_SIGNED_INTEL:
                fmt = f"<i{size}"

            elif data_type == v4c.DATA_TYPE_SIGNED_MOTOROLA:
                fmt = f">i{size}"

            elif data_type == v4c.DATA_TYPE_REAL_INTEL:
                fmt = f"<f{size}"

            elif data_type == v4c.DATA_TYPE_REAL_MOTOROLA:
                fmt = f">f{size}"
            elif data_type == v4c.DATA_TYPE_COMPLEX_INTEL:
                fmt = f"<c{size}"
            elif data_type == v4c.DATA_TYPE_COMPLEX_MOTOROLA:
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
        if kind == "u":
            if byteorder == "<":
                data_type = v3c.DATA_TYPE_UNSIGNED_INTEL
            else:
                data_type = v3c.DATA_TYPE_UNSIGNED_MOTOROLA
        elif kind == "i":
            if byteorder == "<":
                data_type = v3c.DATA_TYPE_SIGNED_INTEL
            else:
                data_type = v3c.DATA_TYPE_SIGNED_MOTOROLA
        elif kind == "f":
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
        elif kind in "SV":
            data_type = v3c.DATA_TYPE_STRING
        elif kind == "b":
            data_type = v3c.DATA_TYPE_UNSIGNED_INTEL
            size = 1
        else:
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
        if kind == "u":
            if byteorder == "<":
                data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            else:
                data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
        elif kind == "i":
            if byteorder == "<":
                data_type = v4c.DATA_TYPE_SIGNED_INTEL
            else:
                data_type = v4c.DATA_TYPE_SIGNED_MOTOROLA
        elif kind == "f":
            if byteorder == "<":
                data_type = v4c.DATA_TYPE_REAL_INTEL
            else:
                data_type = v4c.DATA_TYPE_REAL_MOTOROLA
        elif kind in "SV":
            data_type = v4c.DATA_TYPE_STRING_LATIN_1
        elif kind == "b":
            data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            size = 1
        elif kind == "c":
            if byteorder == "<":
                data_type = v4c.DATA_TYPE_COMPLEX_INTEL
            else:
                data_type = v4c.DATA_TYPE_COMPLEX_MOTOROLA
        else:
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
        (2**bit_length - truncated_integers) * -1,  # when negative, do two's complement
        truncated_integers,  # when positive, return the truncated int
    )


def debug_channel(
    mdf: MDF_v2_v3_v4,
    group: Group,
    channel: ChannelType,
    dependency: list[tuple[int, int]],
    file: StringIO | None = None,
) -> None:
    """use this to print debug information in case of errors

    Parameters
    ----------
    mdf : MDF
        source MDF object
    group : dict
        group
    channel : Channel
        channel object
    dependency : ChannelDependency
        channel dependency object

    """
    print("MDF", "=" * 76, file=file)
    print("name:", mdf.name, file=file)
    print("version:", mdf.version, file=file)
    print("read fragment size:", mdf._read_fragment_size, file=file)
    print("write fragment size:", mdf._write_fragment_size, file=file)
    print()

    record = mdf._prepare_record(group)
    print("GROUP", "=" * 74, file=file)
    print("sorted:", group["sorted"], file=file)
    print("data location:", group["data_location"], file=file)
    print("data blocks:", group.data_blocks, file=file)
    print("dependencies", group["channel_dependencies"], file=file)
    print("record:", record, file=file)
    print(file=file)

    cg = group["channel_group"]
    print("CHANNEL GROUP", "=" * 66, file=file)
    print(cg, cg.cycles_nr, cg.samples_byte_nr, cg.invalidation_bytes_nr, file=file)
    print(file=file)

    print("CHANNEL", "=" * 72, file=file)
    print(channel, file=file)
    print(file=file)

    print("CHANNEL ARRAY", "=" * 66, file=file)
    print(dependency, file=file)
    print(file=file)


def count_channel_groups(
    stream: ReadableBufferType, include_channels: bool = False, mapped: bool = False
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
            raise MdfException(f'"{stream.name}" is not a valid MDF file')

    if version >= 4:
        if mapped:
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


class ChannelsDB(Dict[str, Tuple[Tuple[int, int], ...]]):
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


def is_file_like(obj: object) -> bool:
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
    >>> buffer(StringIO("data"))
    >>> is_file_like(buffer)
    True
    >>> is_file_like([1, 2, 3])
    False
    """
    if not (hasattr(obj, "read") and hasattr(obj, "seek")):
        return False

    if not hasattr(obj, "__iter__"):
        return False

    return True


class UniqueDB:
    def __init__(self) -> None:
        self._db = {}

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
            result = subprocess.run(
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
            result = float(result.stdout)
        except FileNotFoundError:
            result = None
    return result


class Group:
    __slots__ = (
        "channels",
        "channel_dependencies",
        "signal_data",
        "channel_group",
        "record_size",
        "sorted",
        "data_group",
        "data_location",
        "data_blocks",
        "data_blocks_info_generator",
        "record_size",
        "record",
        "signal_types",
        "trigger",
        "string_dtypes",
        "single_channel_dtype",
        "uses_ld",
        "read_split_count",
    )

    def __init__(self, data_group: DataGroupType) -> None:
        self.data_group = data_group
        self.channels = []
        self.channel_dependencies = []
        self.signal_data = []
        self.record = None
        self.trigger = None
        self.string_dtypes = None
        self.data_blocks = []
        self.single_channel_dtype = None
        self.uses_ld = False
        self.read_split_count = 0
        self.data_blocks_info_generator = iter(EMPTY_TUPLE)

    def __getitem__(self, item: str) -> Any:
        return self.__getattribute__(item)

    def __setitem__(self, item: str, value: Any) -> None:
        self.__setattr__(item, value)

    def set_blocks_info(self, info: list[DataBlockInfo]) -> None:
        self.data_blocks = info

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def clear(self) -> None:
        self.data_blocks.clear()
        self.channels.clear()
        self.channel_dependencies.clear()
        self.signal_data.clear()
        self.data_blocks_info_generator = None

    def get_data_blocks(self) -> Iterator[DataBlockInfo]:
        yield from self.data_blocks

        while True:
            try:
                info = next(self.data_blocks_info_generator)
                self.data_blocks.append(info)
                yield info
            except StopIteration:
                break

    def get_signal_data_blocks(self, index: int) -> Iterator[SignalDataBlockInfo]:
        signal_data = self.signal_data[index]
        if signal_data is not None:
            signal_data, signal_generator = signal_data
            yield from signal_data

            while True:
                try:
                    info = next(signal_generator)
                    signal_data.append(info)
                    yield info
                except StopIteration:
                    break


class VirtualChannelGroup:
    """starting with MDF v4.20 it is possible to use remote masters and column
    oriented storage. This means we now have virtual channel groups that can
    span over multiple regular channel groups. This class facilitates the
    handling of this virtual groups"""

    __slots__ = (
        "groups",
        "record_size",
        "cycles_nr",
    )

    def __init__(self) -> None:
        self.groups = []
        self.record_size = 0
        self.cycles_nr = 0

    def __repr__(self) -> None:
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


def components(
    channel: NDArray[Any],
    channel_name: str,
    unique_names: UniqueDB,
    prefix: str = "",
    master: NDArray[Any] | None = None,
    only_basenames: bool = False,
) -> tuple[str, Series[Any]]:
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
    master : np.array
        optional index for the Series
    only_basenames (False) : bool
        use just the field names, without prefix, for structures and channel
        arrays

        .. versionadded:: 5.13.0

    Returns
    -------
    name, series : (str, values)
        tuple of unique name and values
    """
    names = channel.dtype.names

    # channel arrays
    if names[0] == channel_name:
        name = names[0]

        if not only_basenames:
            if prefix:
                name_ = unique_names.get_unique_name(f"{prefix}.{name}")
            else:
                name_ = unique_names.get_unique_name(name)
        else:
            name_ = unique_names.get_unique_name(name)

        values = channel[name]
        if values.dtype.byteorder not in target_byte_order:
            values = values.byteswap().view(values.dtype.newbyteorder())

        if len(values.shape) > 1:
            values = Series(
                list(values),
                index=master,
            )
        else:
            values = Series(
                values,
                index=master,
            )

        yield name_, values

        for name in names[1:]:
            values = channel[name]

            if values.dtype.byteorder not in target_byte_order:
                values = values.byteswap().view(values.dtype.newbyteorder())

            if not only_basenames:
                axis_name = unique_names.get_unique_name(f"{name_}.{name}")
            else:
                axis_name = unique_names.get_unique_name(name)
            if len(values.shape) > 1:
                values = Series(
                    list(values),
                    index=master,
                )
            else:
                values = Series(
                    values,
                    index=master,
                )

            yield axis_name, values

    # structure composition
    else:
        for name in channel.dtype.names:
            values = channel[name]

            if values.dtype.names:
                yield from components(
                    values,
                    name,
                    unique_names,
                    prefix=f"{prefix}.{channel_name}" if prefix else f"{channel_name}",
                    master=master,
                    only_basenames=only_basenames,
                )

            else:
                if values.dtype.byteorder not in target_byte_order:
                    values = values.byteswap().view(values.dtype.newbyteorder())

                if not only_basenames:
                    name_ = unique_names.get_unique_name(
                        f"{prefix}.{channel_name}.{name}" if prefix else f"{channel_name}.{name}"
                    )
                else:
                    name_ = unique_names.get_unique_name(name)
                if len(values.shape) > 1:
                    values = Series(
                        list(values),
                        index=master,
                    )
                else:
                    values = Series(
                        values,
                        index=master,
                    )

                yield name_, values


class DataBlockInfo:
    __slots__ = (
        "address",
        "block_type",
        "original_size",
        "compressed_size",
        "param",
        "invalidation_block",
        "block_limit",
    )

    def __init__(
        self,
        address: int,
        block_type: int,
        original_size: int,
        compressed_size: int,
        param: int,
        invalidation_block=None,
        block_limit: int | None = None,
    ) -> None:
        self.address = address
        self.block_type = block_type
        self.original_size = original_size
        self.compressed_size = compressed_size
        self.param = param
        self.invalidation_block = invalidation_block
        self.block_limit = block_limit

    def __repr__(self) -> str:
        return (
            f"DataBlockInfo(address=0x{self.address:X}, "
            f"block_type={self.block_type}, "
            f"original_size={self.original_size}, "
            f"compressed_size={self.compressed_size}, "
            f"param={self.param}, "
            f"invalidation_block={self.invalidation_block}, "
            f"block_limit={self.block_limit})"
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
        super().__init__(address, block_type, original_size, compressed_size, param, block_limit)
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
        "original_size",
        "compressed_size",
        "param",
        "block_type",
        "location",
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


def get_fields(obj: object) -> list[Any]:
    fields = []
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


def master_using_raster(mdf: MDF_v2_v3_v4, raster: RasterType, endpoint: bool = False) -> NDArray[Any]:
    """get single master based on the raster

    Parameters
    ----------
    mdf : asammdf.MDF
        measurement object
    raster : float
        new raster
    endpoint=False : bool
        include maximum time stamp in the new master

    Returns
    -------
    master : np.array
        new master

    """
    if not raster:
        master = np.array([], dtype="<f8")
    else:
        t_min = []
        t_max = []
        for group_index in mdf.virtual_groups:
            group = mdf.groups[group_index]
            cycles_nr = group.channel_group.cycles_nr
            if cycles_nr:
                master_min = mdf.get_master(group_index, record_offset=0, record_count=1)
                if len(master_min):
                    t_min.append(master_min[0])
                master_max = mdf.get_master(group_index, record_offset=cycles_nr - 1, record_count=1)
                if len(master_max):
                    t_max.append(master_max[0])

        if t_min:
            t_min = np.amin(t_min)
            t_max = np.amax(t_max)

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


def csv_int2bin(val) -> str:
    """format CAN id as bin

    100 -> 1100100

    """

    return f"{val:b}"


csv_int2bin = np.vectorize(csv_int2bin, otypes=[str])


def csv_int2hex(val) -> str:
    """format CAN id as hex

    100 -> 64

    """

    return f"{val:X}"


csv_int2hex = np.vectorize(csv_int2hex, otypes=[str])


def csv_bytearray2hex(val, size: int | None = None) -> str:
    """format CAN payload as hex strings

    b'\xa2\xc3\x08' -> A2 C3 08

    """
    if size is not None:
        val = val.tobytes()[:size].hex(" ", 1).upper()
    else:
        val = val.tobytes().hex(" ", 1).upper()

    return val


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


def load_can_database(path: StrPathType, contents: bytes | str | None = None, **kwargs) -> CanMatrix | None:
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
    if contents is None:
        func = canmatrix.formats.loadp
        arg = path
    else:
        func = canmatrix.formats.loads
        arg = contents

    try:
        dbs = func(arg, import_type=import_type, key="db", **kwargs)
    except UnicodeDecodeError:
        if contents is None:
            contents = path.read_bytes()

        encoding = detect(contents)["encoding"]

        try:
            dbs = func(
                arg,
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


def all_blocks_addresses(obj: ReadableBufferType):
    pattern = re.compile(
        rb"(?P<block>##(D[GVTZIL]|AT|C[AGHNC]|EV|FH|HL|LD|MD|R[DVI]|S[IRD]|TX)\x00\x00)",
        re.DOTALL | re.MULTILINE,
    )

    try:
        obj.seek(0)
    except:
        pass

    try:
        re.search(pattern, obj)
        source = obj
    except TypeError:
        source = obj.read()

    addresses = []
    block_groups = {}
    blocks = {}

    for match in re.finditer(pattern, source):
        btype = match.group("block")
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


def extract_mime_names(data, disable_new_channels=None):
    def fix_comparison_name(data, disable_new_channels=None):
        for item in data:
            if item["type"] == "channel":
                if disable_new_channels is not None:
                    item["enabled"] = not disable_new_channels

                if (item["group_index"], item["channel_index"]) != (-1, -1):
                    name = COMPARISON_NAME.match(item["name"]).group("name").strip()
                    item["name"] = name
            else:
                if disable_new_channels is not None:
                    item["enabled"] = not disable_new_channels
                fix_comparison_name(item["channels"], disable_new_channels=disable_new_channels)

    names = []
    if data.hasFormat("application/octet-stream-asammdf"):
        data = bytes(data.data("application/octet-stream-asammdf")).decode("utf-8")
        data = json.loads(data)
        fix_comparison_name(data, disable_new_channels=disable_new_channels)
        names = data

    return names


def set_mime_enable(mime, enable):
    for item in mime:
        if item["type"] == "channel":
            item["enabled"] = enable
        else:
            set_mime_enable(item["channels"], enable)


def load_dsp(file, background="#000000", flat=False):
    if isinstance(background, str):
        background = fn.mkColor(background)

    def parse_conversions(display):
        conversions = {}

        if display is None:
            return conversions

        for item in display.findall("COMPU_METHOD"):
            try:
                conv = {
                    "name": item.get("name"),
                    "comment": item.get("description"),
                    "unit": item.get("unit"),
                }

                conversion_type = int(item.get("cnv_type"))
                if conversion_type == 0:
                    conv["conversion_type"] = v4c.CONVERSION_TYPE_LIN

                    coeffs = item.find("COEFFS_LINIAR")

                    conv["a"] = float(coeffs.get("P1"))
                    conv["b"] = float(coeffs.get("P2"))

                elif conversion_type == 9:
                    conv["conversion_type"] = v4c.CONVERSION_TYPE_RAT

                    coeffs = item.find("COEFFS")
                    for i in range(1, 7):
                        conv[f"P{i}"] = float(coeffs.get(f"P{i}"))

                elif conversion_type == 11:
                    conv["conversion_type"] = v4c.CONVERSION_TYPE_TABX
                    vtab = item.find("COMPU_VTAB")

                    if vtab is not None:
                        for i, item in enumerate(vtab.findall("tab")):
                            conv[f"val_{i}"] = float(item.get("min"))
                            text = item.get("text")
                            if isinstance(text, bytes):
                                text = text.decode("utf-8", errors="replace")
                            conv[f"text_{i}"] = text

                elif conversion_type == 12:
                    conv["conversion_type"] = v4c.CONVERSION_TYPE_RTABX
                    vtab = item.find("COMPU_VTAB_RANGE")

                    if vtab is not None:
                        text = vtab.get("default")
                        if isinstance(text, bytes):
                            text = text.decode("utf-8", errors="replace")
                        conv["default_addr"] = vtab.get("default")
                        for i, item in enumerate(vtab.findall("tab_range")):
                            conv[f"upper_{i}"] = float(item.get("max"))
                            conv[f"lower_{i}"] = float(item.get("min"))
                            text = item.get("text")
                            if isinstance(text, bytes):
                                text = text.decode("utf-8", errors="replace")
                            conv[f"text_{i}"] = text
                else:
                    continue

                conversions[conv["name"]] = conv

            except:
                print(format_exc())
                continue

        return conversions

    def parse_channels(display, conversions):
        channels = []
        for elem in display.iterchildren():
            if elem.tag == "CHANNEL":
                channel_name = elem.get("name")

                comment = elem.find("COMMENT")
                if comment is not None:
                    comment = elem.get("text")
                else:
                    comment = ""

                color_ = int(elem.get("color"))
                c = 0
                for i in range(3):
                    c = c << 8
                    c += color_ & 0xFF
                    color_ = color_ >> 8

                ch_color = c

                gain = abs(float(elem.get("gain")))
                offset = float(elem.get("offset")) / 100

                multi_color = elem.find("MULTI_COLOR")

                ranges = []

                if multi_color is not None:
                    for color in multi_color.findall("color"):
                        min_ = float(color.find("min").get("data"))
                        max_ = float(color.find("max").get("data"))
                        color_ = int(color.find("color").get("data"))
                        c = 0
                        for i in range(3):
                            c = c << 8
                            c += color_ & 0xFF
                            color_ = color_ >> 8
                        color = fn.mkColor(f"#{c:06X}")
                        ranges.append(
                            {
                                "background_color": background,
                                "font_color": color,
                                "op1": "<=",
                                "op2": "<=",
                                "value1": min_,
                                "value2": max_,
                            }
                        )

                chan = {
                    "color": f"#{ch_color:06X}",
                    "common_axis": False,
                    "computed": False,
                    "flags": 0,
                    "comment": comment,
                    "enabled": elem.get("on") == "1",
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

            elif elem.tag.startswith("GROUP"):
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
                    if filter_type in ("None", None):
                        filter_type = "Unspecified"
                        filter_value = 0
                        raw = False
                    else:
                        filter_value = float(elem.get("filter_value"))
                        raw = bool(int(elem.get("filter_use_raw")))

                    info = {
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
                            min_ = float(color.find("min").get("data"))
                            max_ = float(color.find("max").get("data"))
                            color_ = int(color.find("color").get("data"))
                            c = 0
                            for i in range(3):
                                c = c << 8
                                c += color_ & 0xFF
                                color_ = color_ >> 8
                            color = fn.mkColor(f"#{c:06X}")
                            ranges.append(
                                {
                                    "background_color": background,
                                    "font_color": color,
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
                            "name": info["pattern"],
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

    def parse_virtual_channels(display):
        channels = {}

        if display is None:
            return channels

        for item in display.findall("V_CHAN"):
            try:
                virtual_channel = {}

                parent = item.find("VIR_TIME_CHAN")
                vtab = item.find("COMPU_VTAB")
                if parent is None or vtab is None:
                    continue

                name = item.get("name")

                virtual_channel["name"] = name
                virtual_channel["parent"] = parent.get("data")
                virtual_channel["comment"] = item.find("description").get("data")

                conv = {}
                for i, item in enumerate(vtab.findall("tab")):
                    conv[f"val_{i}"] = float(item.get("min"))
                    text = item.get("text")
                    if isinstance(text, bytes):
                        text = text.decode("utf-8", errors="replace")
                    conv[f"text_{i}"] = text

                virtual_channel["vtab"] = conv

                channels[name] = virtual_channel
            except:
                continue

        return channels

    def parse_c_functions(display):
        c_functions = set()

        if display is None:
            return c_functions

        for item in display.findall("CALC_FUNC"):
            string = item.text

            for match in C_FUNCTION.finditer(string):
                c_functions.add(match.group("function"))

        return sorted(c_functions)

    dsp = Path(file).read_bytes().replace(b"\0", b"")
    dsp = lxml.etree.fromstring(dsp)

    conversions = parse_conversions(dsp.find("COMPU_METHODS"))

    channels = parse_channels(dsp.find("DISPLAY_INFO"), conversions)
    c_functions = parse_c_functions(dsp)

    functions = {}
    virtual_channels = []

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
                "name": ch["parent"],
                "precision": 3,
                "ranges": [],
                "unit": "",
                "conversion": ch["vtab"],
                "user_defined_name": ch["name"],
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

    info = {
        "selected_channels": [],
        "windows": [],
        "has_virtual_channels": bool(virtual_channels),
        "c_functions": c_functions,
        "functions": functions,
    }

    if flat:
        info = flatten_dsp(channels)
    else:
        plot = {
            "type": "Plot",
            "title": "Display channels",
            "maximized": True,
            "configuration": {
                "channels": channels,
                "locked": True,
            },
        }

        info["windows"].append(plot)

    return info


def flatten_dsp(channels):
    res = []

    for item in channels:
        if item["type"] == "group":
            res.extend(flatten_dsp(item["channels"]))
        else:
            res.append(item["name"])

    return res


def load_channel_names_from_file(file_name, lab_section=""):
    file_name = Path(file_name)

    extension = file_name.suffix.lower()
    if extension == ".dsp":
        channels = load_dsp(file_name, flat=True)

    elif extension == ".dspf":
        with open(file_name) as infile:
            info = json.load(infile)

        channels = []
        for window in info["windows"]:
            if window["type"] == "Plot":
                channels.extend(flatten_dsp(window["configuration"]["channels"]))
            elif window["type"] == "Numeric":
                channels.extend([item["name"] for item in window["configuration"]["channels"]])
            elif window["type"] == "Tabular":
                channels.extend(window["configuration"]["channels"])

    elif extension == ".lab":
        info = load_lab(file_name)
        if info:
            if len(info) > 1 and lab_section:
                channels = info[lab_section]
            else:
                channels = list(info.values())[0]

            channels = [name.split(";")[0] for name in channels]

    elif extension == ".cfg":
        with open(file_name) as infile:
            info = json.load(infile)
        channels = info.get("selected_channels", [])
    elif extension == ".txt":
        try:
            with open(file_name) as infile:
                info = json.load(infile)
            channels = info.get("selected_channels", [])
        except:
            with open(file_name) as infile:
                channels = [line.strip() for line in infile.readlines()]
                channels = [name for name in channels if name]

    return sorted(set(channels))


def load_lab(file):
    sections = {}
    with open(file) as lab:
        for line in lab:
            line = line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                section_name = line.strip("[]")
                s = []
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


def timeit(func):
    def timed(*args, **kwargs):
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

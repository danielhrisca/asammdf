# -*- coding: utf-8 -*-
"""
asammdf utility functions and classes
"""

import logging
import string
import xml.etree.ElementTree as ET
import re
import subprocess

from collections import namedtuple
from random import randint
from struct import Struct
from tempfile import TemporaryDirectory
from pathlib import Path

from numpy import where, arange, interp
from numpy.core.records import fromarrays
from pandas import Series

from . import v2_v3_constants as v3c
from . import v4_constants as v4c

UINT8_u = Struct("<B").unpack
UINT16_u = Struct("<H").unpack
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

_xmlns_pattern = re.compile(' xmlns="[^"]*"')

logger = logging.getLogger("asammdf")

__all__ = [
    "CHANNEL_COUNT",
    "CONVERT",
    "MERGE",
    "ChannelsDB",
    "UniqueDB",
    "MdfException",
    "SignalSource",
    "get_fmt_v3",
    "get_fmt_v4",
    "get_text_v4",
    "fmt_to_datatype_v3",
    "fmt_to_datatype_v4",
    "matlab_compatible",
    "extract_cncomment_xml",
    "validate_version_argument",
    "MDF2_VERSIONS",
    "MDF3_VERSIONS",
    "MDF4_VERSIONS",
    "SUPPORTED_VERSIONS",
]

CHANNEL_COUNT = (1000, 2000, 10000, 20000)
_channel_count = arange(0, 20000, 1000, dtype="<u4")

CONVERT = (10 * 2 ** 20, 20 * 2 ** 20, 30 * 2 ** 20, 40 * 2 ** 20)
CONVERT = interp(_channel_count, CHANNEL_COUNT, CONVERT).astype("<u4")

MERGE = (10 * 2 ** 20, 20 * 2 ** 20, 35 * 2 ** 20, 60 * 2 ** 20)
MERGE = interp(_channel_count, CHANNEL_COUNT, MERGE).astype("<u4")

CHANNEL_COUNT = _channel_count

MDF2_VERSIONS = ("2.00", "2.10", "2.14")
MDF3_VERSIONS = ("3.00", "3.10", "3.20", "3.30")
MDF4_VERSIONS = ("4.00", "4.10", "4.11")
SUPPORTED_VERSIONS = MDF2_VERSIONS + MDF3_VERSIONS + MDF4_VERSIONS


ALLOWED_MATLAB_CHARS = set(string.ascii_letters + string.digits + "_")


SignalSource = namedtuple(
    "SignalSource", ["name", "path", "comment", "source_type", "bus_type"]
)
""" Commons reprezentation for source information

Attributes
----------
name : str
    source name
path : str
    source path
comment : str
    source comment
source_type : int
    source type code
bus_type : int
    source bus code

"""


class MdfException(Exception):
    """MDF Exception class"""

    pass


def extract_cncomment_xml(comment):
    """extract *TX* tag or otherwise the *common_properties* from a xml comment

    Paremeters
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


def matlab_compatible(name):
    """ make a channel name compatible with Matlab variable naming

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
    # for get_unique_name in case of multiple channel name occurence
    return compatible_name[:60]


def get_text_v3(address, stream, mapped=False):
    """ faster way to extract strings from mdf versions 2 and 3 TextBlock

    Parameters
    ----------
    address : int
        TextBlock address
    stream : handle
        file IO handle

    Returns
    -------
    text : str
        unicode string

    """

    if address == 0:
        return ""

    if mapped:
        size, = UINT16_uf(stream, address + 2)
        text_bytes = stream[address + 4: address + size]
    else:
        stream.seek(address + 2)
        size = UINT16_u(stream.read(2))[0] - 4
        text_bytes = stream.read(size)
    try:
        text = text_bytes.strip(b" \r\t\n\0").decode("latin-1")
    except UnicodeDecodeError as err:
        try:
            from cchardet import detect

            encoding = detect(text_bytes)["encoding"]
            text = text_bytes.strip(b" \r\t\n\0").decode(encoding)
        except ImportError:
            logger.warning(
                'Unicode exception occured and "cChardet" package is '
                'not installed. Mdf version 3 expects "latin-1" '
                "strings and this package may detect if a different"
                " encoding was used"
            )
            raise err

    return text


def get_text_v4(address, stream, mapped=False):
    """ faster way to extract strings from mdf version 4 TextBlock

    Parameters
    ----------
    address : int
        TextBlock address
    stream : handle
        file IO handle

    Returns
    -------
    text : str
        unicode string

    """

    if address == 0:
        return ""

    if mapped:
        size, _ = TWO_UINT64_uf(stream, address + 8)
        text_bytes = stream[address + 24: address + size]
    else:
        stream.seek(address + 8)
        size, _ = TWO_UINT64_u(stream.read(16))
        text_bytes = stream.read(size - 24)
    try:
        text = text_bytes.strip(b" \r\t\n\0").decode("utf-8")
    except UnicodeDecodeError as err:
        try:
            from cchardet import detect

            encoding = detect(text_bytes)["encoding"]
            text = text_bytes.decode(encoding).strip(" \r\t\n\0")
        except ImportError:
            logger.warning(
                'Unicode exception occured and "cChardet" package is '
                'not installed. Mdf version 4 expects "utf-8" '
                "strings and this package may detect if a different"
                " encoding was used"
            )
            raise err

    return text


def sanitize_xml(text):
    return re.sub(_xmlns_pattern, "", text)


def get_fmt_v3(data_type, size):
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
    if data_type in {v3c.DATA_TYPE_STRING, v3c.DATA_TYPE_BYTEARRAY}:
        size = size // 8
        if data_type == v3c.DATA_TYPE_STRING:
            fmt = f"S{size}"
        elif data_type == v3c.DATA_TYPE_BYTEARRAY:
            fmt = f"({size},)u1"
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

        if data_type in (v3c.DATA_TYPE_UNSIGNED_INTEL, v3c.DATA_TYPE_UNSIGNED):
            fmt = f"<u{size}".format()

        elif data_type == v3c.DATA_TYPE_UNSIGNED_MOTOROLA:
            fmt = f">u{size}"

        elif data_type in (v3c.DATA_TYPE_SIGNED_INTEL, v3c.DATA_TYPE_SIGNED):
            fmt = f"<i{size}"

        elif data_type == v3c.DATA_TYPE_SIGNED_MOTOROLA:
            fmt = f">i{size}"

        elif data_type in {
            v3c.DATA_TYPE_FLOAT,
            v3c.DATA_TYPE_DOUBLE,
            v3c.DATA_TYPE_FLOAT_INTEL,
            v3c.DATA_TYPE_DOUBLE_INTEL,
        }:
            fmt = f"<f{size}"

        elif data_type in (v3c.DATA_TYPE_FLOAT_MOTOROLA, v3c.DATA_TYPE_DOUBLE_MOTOROLA):
            fmt = f">f{size}"

    return fmt


def get_fmt_v4(data_type, size, channel_type=v4c.CHANNEL_TYPE_VALUE):
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

        if data_type == v4c.DATA_TYPE_BYTEARRAY:
            if channel_type == v4c.CHANNEL_TYPE_VALUE:
                fmt = f"({size},)u1"
            else:
                if size == 4:
                    fmt = "<u4"
                elif size == 8:
                    fmt = "<u8"

        elif data_type in v4c.STRING_TYPES:
            if channel_type == v4c.CHANNEL_TYPE_VALUE:
                fmt = f"S{size}"
            else:
                if size == 4:
                    fmt = "<u4"
                elif size == 8:
                    fmt = "<u8"

        elif data_type == v4c.DATA_TYPE_CANOPEN_DATE:
            fmt = "V7"

        elif data_type == v4c.DATA_TYPE_CANOPEN_TIME:
            fmt = "V6"

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

    return fmt


def fmt_to_datatype_v3(fmt, shape, array=False):
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
    size = fmt.itemsize * 8

    if not array and shape[1:] and fmt.itemsize == 1 and fmt.kind == "u":
        data_type = v3c.DATA_TYPE_BYTEARRAY
        for dim in shape[1:]:
            size *= dim
    else:
        if fmt.kind == "u":
            if fmt.byteorder in "=<|":
                data_type = v3c.DATA_TYPE_UNSIGNED
            else:
                data_type = v3c.DATA_TYPE_UNSIGNED_MOTOROLA
        elif fmt.kind == "i":
            if fmt.byteorder in "=<|":
                data_type = v3c.DATA_TYPE_SIGNED
            else:
                data_type = v3c.DATA_TYPE_SIGNED_MOTOROLA
        elif fmt.kind == "f":
            if fmt.byteorder in "=<":
                if size == 32:
                    data_type = v3c.DATA_TYPE_FLOAT
                else:
                    data_type = v3c.DATA_TYPE_DOUBLE
            else:
                if size == 32:
                    data_type = v3c.DATA_TYPE_FLOAT_MOTOROLA
                else:
                    data_type = v3c.DATA_TYPE_DOUBLE_MOTOROLA
        elif fmt.kind in "SV":
            data_type = v3c.DATA_TYPE_STRING
        elif fmt.kind == "b":
            data_type = v3c.DATA_TYPE_UNSIGNED
            size = 1
        else:
            message = f"Unknown type: dtype={fmt}, shape={shape}"
            logger.exception(message)
            raise MdfException(message)

    return data_type, size


def info_to_datatype_v4(signed, little_endian):
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


def fmt_to_datatype_v4(fmt, shape, array=False):
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
    size = fmt.itemsize * 8

    if not array and shape[1:] and fmt.itemsize == 1 and fmt.kind == "u":
        data_type = v4c.DATA_TYPE_BYTEARRAY
        for dim in shape[1:]:
            size *= dim

    else:
        if fmt.kind == "u":
            if fmt.byteorder in "=<|":
                data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            else:
                data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
        elif fmt.kind == "i":
            if fmt.byteorder in "=<|":
                data_type = v4c.DATA_TYPE_SIGNED_INTEL
            else:
                data_type = v4c.DATA_TYPE_SIGNED_MOTOROLA
        elif fmt.kind == "f":
            if fmt.byteorder in "=<":
                data_type = v4c.DATA_TYPE_REAL_INTEL
            else:
                data_type = v4c.DATA_TYPE_REAL_MOTOROLA
        elif fmt.kind in "SV":
            data_type = v4c.DATA_TYPE_STRING_LATIN_1
        elif fmt.kind == "b":
            data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            size = 1
        else:
            message = f"Unknown type: dtype={fmt}, shape={shape}"
            logger.exception(message)
            raise MdfException(message)

    return data_type, size


def as_non_byte_sized_signed_int(integer_array, bit_length):
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
        truncated_integers = integer_array & (
            (1 << bit_length) - 1
        )  # Zero out the unwanted bits
    return where(
        truncated_integers >> bit_length - 1,  # sign bit as a truth series (True when negative)
        (2 ** bit_length - truncated_integers) * -1,  # when negative, do two's complement
        truncated_integers,  # when positive, return the truncated int
    )


def debug_channel(mdf, group, channel, dependency, file=None):
    """ use this to print debug information in case of errors

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

    parents, dtypes = mdf._prepare_record(group)
    print("GROUP", "=" * 74, file=file)
    print("sorted:", group["sorted"], file=file)
    print("data location:", group["data_location"], file=file)
    print("data block type:", group["data_block_type"], file=file)
    print("param:", group["param"], file=file)
    print("data size:", group["data_size"], file=file)
    print("data block size:", group["data_block_size"], file=file)
    print("data block addr:", group["data_block_addr"], file=file)
    print("data block:", group["data_block"], file=file)
    print("record_size:", group["record_size"], file=file)
    print("dependencies", group["channel_dependencies"], file=file)
    print("parents:", parents, file=file)
    print("dtypes:", dtypes, file=file)
    print(file=file)

    cg = group["channel_group"]
    print("CHANNEL GROUP", "=" * 66, file=file)
    print(cg, file=file)
    print(file=file)

    print("CHANNEL", "=" * 72, file=file)
    print(channel, file=file)
    print(file=file)

    print("CHANNEL ARRAY", "=" * 66, file=file)
    print(dependency, file=file)
    print(file=file)

    print("MASTER CACHE", "=" * 67, file=file)
    print(
        [(key, len(val)) for key, val in mdf._master_channel_cache.items()], file=file
    )


def count_channel_groups(stream, include_channels=False):
    """ count all channel groups as fast as possible. This is used to provide
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
                stream.seek(cg_addr + 24)
                cg_addr = UINT64_u(stream.read(8))[0]

            stream.seek(dg_addr + 24)
            dg_addr = UINT64_u(stream.read(8))[0]

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
                stream.seek(cg_addr + 4)
                cg_addr = UINT32_u(stream.read(4))[0]

            stream.seek(dg_addr + 4)
            dg_addr = UINT32_u(stream.read(4))[0]

    return count, ch_count


def validate_version_argument(version, hint=4):
    """ validate the version argument against the supported MDF versions. The
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
        message = (
            'Unknown mdf version "{}".'
            " The available versions are {};"
            ' automatically using version "{}"'
        )
        message = message.format(version, SUPPORTED_VERSIONS, valid_version)
        logger.warning(message)
    else:
        valid_version = version
    return valid_version


class ChannelsDB(dict):
    def __init__(self, version=4):

        super().__init__()
        if version == 4:
            self.encoding = "utf-8"
        else:
            self.encoding = "latin-1"

    def add(self, channel_name, entry):
        """ add name to channels database and check if it contains a source
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
                self[channel_name] = [entry]
            else:
                self[channel_name].append(entry)

            if "\\" in channel_name:
                channel_name = channel_name.split("\\")[0]

                if channel_name not in self:
                    self[channel_name] = [entry]
                else:
                    self[channel_name].append(entry)


def randomized_string(size):
    """ get a \0 terminated string of size length

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


def is_file_like(obj):
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


class UniqueDB(object):
    def __init__(self):
        self._db = {}

    def get_unique_name(self, name):
        """ returns an available unique name

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


def cut_video_stream(stream, start, end, fmt):
    """ cut video stream from `start` to `end` time

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
            )
        except FileNotFoundError:
            result = stream
        else:
            if ret.returncode:
                result = stream
            else:
                result = out_file.read_bytes()

    return result


def get_video_stream_duration(stream):
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
            )
            result = float(result.stdout)
        except FileNotFoundError:
            result = None
    return result


class Group:

    __slots__ = (
        "channels",
        "logging_channels",
        "channel_dependencies",
        "signal_data_size",
        "signal_data",
        "channel_group",
        "record_size",
        "sorted",
        "data_group",
        "data_location",
        "data_blocks",
        "record_size",
        "CAN_logging",
        "CAN_id",
        "CAN_database",
        "dbc_addr",
        "raw_can",
        "extended_id",
        "message_name",
        "message_id",
        "record",
        "parents",
        "types",
        "signal_types",
        "trigger",
        "string_dtypes",
    )

    def __init__(self, data_group):
        self.data_group = data_group
        self.channels = []
        self.logging_channels = []
        self.channel_dependencies = []
        self.signal_data = []
        self.CAN_logging = False
        self.CAN_id = None
        self.CAN_database = False
        self.raw_can = False
        self.extended_id = False
        self.message_name = ""
        self.message_id = None
        self.CAN_database = False
        self.dbc_addr = None
        self.parents = None
        self.types = None
        self.record = None
        self.trigger = None
        self.string_dtypes = None
        self.data_blocks = []

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def __setitem__(self, item, value):
        self.__setattr__(item, value)

    def set_blocks_info(self, info):
        self.data_blocks = info

    def __contains__(self, item):
        return hasattr(self, item)


def block_fields(obj):
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


def components(channel, channel_name, unique_names, prefix="", master=None):
    """ yield pandas Series and unique name based on the ndarray object

    Parameters
    ----------
    channel : numpy.ndarray
        channel to be used foir Series
    channel_name : str
        channel name
    unique_names : UniqueDB
        unique names object
    prefix : str
        prefix used in case of nested recarrays

    Returns
    -------
    name, series : (str, pandas.Series)
        tuple of unqiue name and Series object
    """
    names = channel.dtype.names

    # channel arrays
    if names[0] == channel_name:
        name = names[0]

        if prefix:
            name_ = unique_names.get_unique_name(f"{prefix}.{name}")
        else:
            name_ = unique_names.get_unique_name(name)

        values = channel[name]
        if len(values.shape) > 1:
            arr = [values]
            types = [("", values.dtype, values.shape[1:])]
            values = fromarrays(arr, dtype=types)
            del arr
        yield name_, Series(values, index=master, dtype="O")

        for name in names[1:]:
            values = channel[name]
            axis_name = unique_names.get_unique_name(f"{name_}.{name}")
            if len(values.shape) > 1:
                arr = [values]
                types = [("", values.dtype, values.shape[1:])]
                values = fromarrays(arr, dtype=types)
                del arr
            yield axis_name, Series(values, index=master, dtype="O")

    # structure composition
    else:
        for name in channel.dtype.names:
            values = channel[name]

            if values.dtype.names:
                yield from components(
                    values, name, unique_names,
                    prefix=f"{prefix}.{channel_name}" if prefix else f"{channel_name}",
                    master=master
                )

            else:
                name_ = unique_names.get_unique_name(
                    f"{prefix}.{channel_name}.{name}" if prefix else f"{channel_name}.{name}"
                )
                if len(values.shape) > 1:
                    arr = [values]
                    types = [("", values.dtype, values.shape[1:])]
                    values = fromarrays(arr, dtype=types)
                    del arr
                yield name_, Series(values, index=master)


class DataBlockInfo:

    __slots__ = (
        'address',
        'block_type',
        'raw_size',
        'size',
        'param',
    )

    def __init__(self, address, block_type, raw_size, size, param):
        self.address = address
        self.block_type = block_type
        self.raw_size = raw_size
        self.size = size
        self.param = param

    def __repr__(self):
        return (
            f"DataBlockInfo(address=0x{self.address:X}, "
            f"block_type={self.block_type}, "
            f"raw_size={self.raw_size}, "
            f"size={self.size}, "
            f"param={self.param})"
        )


def get_fields(obj):
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

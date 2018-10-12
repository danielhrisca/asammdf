# -*- coding: utf-8 -*-
'''
asammdf utility functions and classes
'''
from __future__ import division, print_function

import logging
import string
import xml.etree.ElementTree as ET

from collections import namedtuple
from random import randint
from struct import Struct
from warnings import warn

from numpy import (
    amin,
    amax,
    where,
)

import sys
PYVERSION = sys.version_info[0]

from . import v2_v3_constants as v3c
from . import v4_constants as v4c

UINT16 = Struct('<H').unpack
UINT32 = Struct('<I').unpack
UINT64 = Struct('<Q').unpack
logger = logging.getLogger('asammdf')

__all__ = [
    'CHANNEL_COUNT',
    'CONVERT_LOW',
    'CONVERT_MINIMUM',
    'MERGE_LOW',
    'MERGE_MINIMUM',
    'ChannelsDB',
    'MdfException',
    'SignalSource',
    'get_fmt_v3',
    'get_fmt_v4',
    'get_min_max',
    'get_unique_name',
    'get_text_v4',
    'fix_dtype_fields',
    'fmt_to_datatype_v3',
    'fmt_to_datatype_v4',
    'bytes',
    'matlab_compatible',
    'extract_cncomment_xml',
    'validate_memory_argument',
    'validate_version_argument',
    'MDF2_VERSIONS',
    'MDF3_VERSIONS',
    'MDF4_VERSIONS',
    'SUPPORTED_VERSIONS',
]

CHANNEL_COUNT = (
    0,
    200,
    2000,
    10000,
    20000,
    400000,
)

CONVERT_LOW = (
    10 * 2**20,
    10 * 2**20,
    20 * 2**20,
    30 * 2**20,
    40 * 2**20,
    100 * 2**20,
)

CONVERT_MINIMUM = (
    10 * 2**20,
    10 * 2**20,
    30 * 2**20,
    30 * 2**20,
    40 * 2**20,
    100 * 2**20,
)

MERGE_LOW = (
    10 * 2**20,
    10 * 2**20,
    20 * 2**20,
    35 * 2**20,
    60 * 2**20,
    100 * 2**20,
)

MERGE_MINIMUM = (
    10 * 2**20,
    10 * 2**20,
    30 * 2**20,
    50 * 2**20,
    60 * 2**20,
    100 * 2**20,
)

MDF2_VERSIONS = ('2.00', '2.10', '2.14')
MDF3_VERSIONS = ('3.00', '3.10', '3.20', '3.30')
MDF4_VERSIONS = ('4.00', '4.10', '4.11')
SUPPORTED_VERSIONS = MDF2_VERSIONS + MDF3_VERSIONS + MDF4_VERSIONS
VALID_MEMORY_ARGUMENT_VALUES = ('full', 'low', 'minimum')


ALLOWED_MATLAB_CHARS = set(string.ascii_letters + string.digits + '_')


SignalSource = namedtuple(
    'SignalSource',
    ['name', 'path', 'comment', 'source_type', 'bus_type'],
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


if PYVERSION < 3:
    # pylint: disable=W0622
    def bytes(obj):
        """ Python 2 compatibility function """
        try:
            return obj.__bytes__()
        except AttributeError:
            if isinstance(obj, str):
                return obj
            else:
                raise
    # pylint: enable=W0622


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

    comment = comment.replace(' xmlns="http://www.asam.net/mdf/v4"', '')
    try:
        comment = ET.fromstring(comment)
        match = comment.find('.//TX')
        if match is None:
            common_properties = comment.find('.//common_properties')
            if common_properties is not None:
                comment = []
                for e in common_properties:
                    field = '{}: {}'.format(e.get('name'), e.text)
                    comment.append(field)
                comment = '\n'.join(field)
            else:
                comment = ''
        else:
            comment = match.text or ''
    except ET.ParseError:
        pass
    finally:
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

    compatible_name = [
        ch if ch in ALLOWED_MATLAB_CHARS else '_'
        for ch in name
    ]
    compatible_name = ''.join(compatible_name)

    if compatible_name[0] not in string.ascii_letters:
        compatible_name = 'M_' + compatible_name

    return compatible_name


def get_text_v3(address, stream):
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
        return ''

    stream.seek(address + 2)
    size = UINT16(stream.read(2))[0] - 4
    text_bytes = stream.read(size)
    try:
        text = (
            text_bytes
            .strip(b' \r\t\n\0')
            .decode('latin-1')
        )
    except UnicodeDecodeError as err:
        try:
            from chardet import detect
            encoding = detect(text_bytes)['encoding']
            text = (
                text_bytes
                .strip(b' \r\t\n\0')
                .decode(encoding)
            )
        except ImportError:
            logger.warning(
                'Unicode exception occured and "chardet" package is '
                'not installed. Mdf version 3 expects "latin-1" '
                'strings and this package may detect if a different'
                ' encoding was used'
            )
            raise err

    return text


def get_text_v4(address, stream):
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
        return ''

    stream.seek(address + 8)
    size = UINT64(stream.read(8))[0] - 16
    text_bytes = stream.read(size)[8:]
    try:
        text = (
            text_bytes
            .decode('utf-8')
            .strip(' \r\t\n\0')
        )
    except UnicodeDecodeError as err:
        try:
            from chardet import detect
            encoding = detect(text_bytes)['encoding']
            text = (
                text_bytes
                .decode(encoding)
                .strip(' \r\t\n\0')
            )
        except ImportError:
            logger.warning(
                'Unicode exception occured and "chardet" package is '
                'not installed. Mdf version 4 expects "utf-8" '
                'strings and this package may detect if a different'
                ' encoding was used'
            )
            raise err

    return text


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
    if data_type in {
            v3c.DATA_TYPE_STRING,
            v3c.DATA_TYPE_BYTEARRAY}:
        size = size // 8
        if data_type == v3c.DATA_TYPE_STRING:
            fmt = 'S{}'.format(size)
        elif data_type == v3c.DATA_TYPE_BYTEARRAY:
            fmt = '({},)u1'.format(size)
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

        if data_type in {
                v3c.DATA_TYPE_UNSIGNED_INTEL,
                v3c.DATA_TYPE_UNSIGNED}:
            fmt = '<u{}'.format(size)

        elif data_type == v3c.DATA_TYPE_UNSIGNED_MOTOROLA:
            fmt = '>u{}'.format(size)

        elif data_type in {
                v3c.DATA_TYPE_SIGNED_INTEL,
                v3c.DATA_TYPE_SIGNED}:
            fmt = '<i{}'.format(size)

        elif data_type == v3c.DATA_TYPE_SIGNED_MOTOROLA:
            fmt = '>i{}'.format(size)

        elif data_type in {
                v3c.DATA_TYPE_FLOAT,
                v3c.DATA_TYPE_DOUBLE,
                v3c.DATA_TYPE_FLOAT_INTEL,
                v3c.DATA_TYPE_DOUBLE_INTEL}:
            fmt = '<f{}'.format(size)

        elif data_type in {
                v3c.DATA_TYPE_FLOAT_MOTOROLA,
                v3c.DATA_TYPE_DOUBLE_MOTOROLA}:
            fmt = '>f{}'.format(size)

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
    if data_type in {
            v4c.DATA_TYPE_BYTEARRAY,
            v4c.DATA_TYPE_STRING_UTF_8,
            v4c.DATA_TYPE_STRING_LATIN_1,
            v4c.DATA_TYPE_STRING_UTF_16_BE,
            v4c.DATA_TYPE_STRING_UTF_16_LE,
            v4c.DATA_TYPE_CANOPEN_DATE,
            v4c.DATA_TYPE_CANOPEN_TIME}:
        size = size // 8

        if data_type == v4c.DATA_TYPE_BYTEARRAY:
            if channel_type == v4c.CHANNEL_TYPE_VALUE:
                fmt = '({},)u1'.format(size)
            else:
                if size == 4:
                    fmt = '<u4'
                elif size == 8:
                    fmt = '<u8'

        elif data_type in {
                v4c.DATA_TYPE_STRING_UTF_8,
                v4c.DATA_TYPE_STRING_LATIN_1,
                v4c.DATA_TYPE_STRING_UTF_16_BE,
                v4c.DATA_TYPE_STRING_UTF_16_LE}:
            if channel_type == v4c.CHANNEL_TYPE_VALUE:
                fmt = 'S{}'.format(size)
            else:
                if size == 4:
                    fmt = '<u4'
                elif size == 8:
                    fmt = '<u8'

        elif data_type == v4c.DATA_TYPE_CANOPEN_DATE:
            fmt = 'V7'

        elif data_type == v4c.DATA_TYPE_CANOPEN_TIME:
            fmt = 'V6'

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
            fmt = '<u{}'.format(size)

        elif data_type == v4c.DATA_TYPE_UNSIGNED_MOTOROLA:
            fmt = '>u{}'.format(size)

        elif data_type == v4c.DATA_TYPE_SIGNED_INTEL:
            fmt = '<i{}'.format(size)

        elif data_type == v4c.DATA_TYPE_SIGNED_MOTOROLA:
            fmt = '>i{}'.format(size)

        elif data_type == v4c.DATA_TYPE_REAL_INTEL:
            fmt = '<f{}'.format(size)

        elif data_type == v4c.DATA_TYPE_REAL_MOTOROLA:
            fmt = '>f{}'.format(size)

    return fmt


def fix_dtype_fields(fields, encoding='utf-8'):
    """ convert field names to str in case of Python 2"""
    new_types = []
    for pair_ in fields:
        if not isinstance(pair_[0], unicode):
            new_types.append(pair_)
        else:
            new_pair = [pair_[0].encode(encoding)]
            for item in pair_[1:]:
                new_pair.append(item)
            new_types.append(tuple(new_pair))

    return new_types


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

    if not array and shape[1:] and fmt.itemsize == 1 and fmt.kind == 'u':
        data_type = v3c.DATA_TYPE_BYTEARRAY
        for dim in shape[1:]:
            size *= dim
    else:
        if fmt.kind == 'u':
            if fmt.byteorder in '=<|':
                data_type = v3c.DATA_TYPE_UNSIGNED
            else:
                data_type = v3c.DATA_TYPE_UNSIGNED_MOTOROLA
        elif fmt.kind == 'i':
            if fmt.byteorder in '=<|':
                data_type = v3c.DATA_TYPE_SIGNED
            else:
                data_type = v3c.DATA_TYPE_SIGNED_MOTOROLA
        elif fmt.kind == 'f':
            if fmt.byteorder in '=<':
                if size == 32:
                    data_type = v3c.DATA_TYPE_FLOAT
                else:
                    data_type = v3c.DATA_TYPE_DOUBLE
            else:
                if size == 32:
                    data_type = v3c.DATA_TYPE_FLOAT_MOTOROLA
                else:
                    data_type = v3c.DATA_TYPE_DOUBLE_MOTOROLA
        elif fmt.kind in {'S', 'V'}:
            data_type = v3c.DATA_TYPE_STRING
        elif fmt.kind == 'b':
            data_type = v3c.DATA_TYPE_UNSIGNED
            size = 1
        else:
            message = 'Unknown type: dtype={}, shape={}'
            message = message.format(fmt, shape)
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

    if not array and shape[1:] and fmt.itemsize == 1 and fmt.kind == 'u':
        data_type = v4c.DATA_TYPE_BYTEARRAY
        for dim in shape[1:]:
            size *= dim

    else:
        if fmt.kind == 'u':
            if fmt.byteorder in {'=', '<', '|'}:
                data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            else:
                data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
        elif fmt.kind == 'i':
            if fmt.byteorder in {'=', '<', '|'}:
                data_type = v4c.DATA_TYPE_SIGNED_INTEL
            else:
                data_type = v4c.DATA_TYPE_SIGNED_MOTOROLA
        elif fmt.kind == 'f':
            if fmt.byteorder in {'=', '<'}:
                data_type = v4c.DATA_TYPE_REAL_INTEL
            else:
                data_type = v4c.DATA_TYPE_REAL_MOTOROLA
        elif fmt.kind in {'S', 'V'}:
            data_type = v4c.DATA_TYPE_STRING_LATIN_1
        elif fmt.kind == 'b':
            data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            size = 1
        else:
            message = 'Unknown type: dtype={}, shape={}'
            message = message.format(fmt, shape)
            logger.exception(message)
            raise MdfException(message)

    return data_type, size


def get_unique_name(used_names, name):
    """ returns a list of unique names

    Parameters
    ----------
    used_names : set
        set of already taken names
    name : str
        name to be made unique

    Returns
    -------
    unique_name : str
        new unique name

    """
    i = 0
    unique_name = name
    while unique_name in used_names:
        unique_name = "{}_{}".format(name, i)
        i += 1

    return unique_name


def get_min_max(samples):
    """ return min and max values for samples. If the samples are
    string return min>max

    Parameters
    ----------
    samples : numpy.ndarray
        signal samples

    Returns
    -------
    min_val, max_val : float, float
        samples min and max values

    """

    if samples.shape[0]:
        if samples.dtype.kind in {'u', 'i', 'f'}:
            min_val, max_val = amin(samples), amax(samples)
        else:
            min_val, max_val = 1, 0
    else:
        min_val, max_val = 0, 0
    return min_val, max_val


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

    truncated_integers = integer_array & ((1 << bit_length) - 1)  # Zero out the unwanted bits
    return where(
        truncated_integers >> bit_length - 1,  # sign bit as a truth series (True when negative)
        (2**bit_length - truncated_integers) * -1,  # when negative, do two's complement
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
    print('MDF', '='*76, file=file)
    print('name:', mdf.name, file=file)
    print('version:', mdf.version, file=file)
    print('memory:', mdf.memory, file=file)
    print('read fragment size:', mdf._read_fragment_size, file=file)
    print('write fragment size:', mdf._write_fragment_size, file=file)
    print()

    parents, dtypes = mdf._prepare_record(group)
    print('GROUP', '='*74, file=file)
    print('sorted:', group['sorted'], file=file)
    print('data location:', group['data_location'], file=file)
    print('data block type:', group['data_block_type'], file=file)
    print('param:', group['param'], file=file)
    print('data size:', group['data_size'], file=file)
    print('data block size:', group['data_block_size'], file=file)
    print('data block addr:', group['data_block_addr'], file=file)
    print('data block:', group['data_block'], file=file)
    print('record_size:', group['record_size'], file=file)
    print('dependencies', group['channel_dependencies'], file=file)
    print('parents:', parents, file=file)
    print('dtypes:', dtypes, file=file)
    print(file=file)

    cg = group['channel_group']
    print('CHANNEL GROUP', '='*66, file=file)
    print(cg, file=file)
    print(file=file)

    print('CHANNEL', '='*72, file=file)
    print(channel, file=file)
    print(file=file)

    print('CHANNEL ARRAY', '='*66, file=file)
    print(dependency, file=file)
    print(file=file)

    print('MASTER CACHE', '='*67, file=file)
    print([(key, len(val)) for key, val in mdf._master_channel_cache.items()], file=file)


def count_channel_groups(stream, version=4):
    """ count all channel groups as fast as possible. This is used to provide
    reliable progress information when loading a file using the GUI

    Parameters
    ----------
    stream : file handle
        opened file handle
    version : int
        MDF version

    Returns
    -------
    count : int
        channel group count

    """

    count = 0
    if version >= 4:
        stream.seek(88, 0)
        dg_addr = UINT64(stream.read(8))[0]
        while dg_addr:
            stream.seek(dg_addr + 32)
            cg_addr = UINT64(stream.read(8))[0]
            while cg_addr:
                count += 1
                stream.seek(cg_addr + 24)
                cg_addr = UINT64(stream.read(8))[0]

            stream.seek(dg_addr + 24)
            dg_addr = UINT64(stream.read(8))[0]

    else:
        stream.seek(68, 0)
        dg_addr = UINT32(stream.read(4))[0]
        while dg_addr:
            stream.seek(dg_addr + 8)
            cg_addr = UINT32(stream.read(4))[0]
            while cg_addr:
                count += 1
                stream.seek(cg_addr + 4)
                cg_addr = UINT32(stream.read(4))[0]

            stream.seek(dg_addr + 4)
            dg_addr = UINT32(stream.read(4))[0]

    return count


def validate_memory_argument(memory):
    """ validate the version argument against the supported MDF versions. The
    default version used depends on the hint MDF major revision

    Parameters
    ----------
    memory : memory
        requested memory argument

    Returns
    -------
    valid_memory : str
        valid memory

    """
    if memory not in VALID_MEMORY_ARGUMENT_VALUES:
        message = (
            'The memory argument "{}" is wrong:'
            ' The available memory options are {};'
            ' automatically using "full"'
        )
        warn(message.format(memory, VALID_MEMORY_ARGUMENT_VALUES))
        valid_memory = 'full'
    else:
        valid_memory = memory
    return valid_memory


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
            valid_version = '2.14'
        elif hint == 3:
            valid_version = '3.30'
        else:
            valid_version = '4.10'
        message = (
            'Unknown mdf version "{}".'
            ' The available versions are {};'
            ' automatically using version "{}"'
        )
        message = message.format(
            version,
            SUPPORTED_VERSIONS,
            valid_version,
        )
        logger.warning(message)
    else:
        valid_version = version
    return valid_version


class ChannelsDB(dict):

    def __init__(self, version=4):

        super(ChannelsDB, self).__init__()
        if version == 4:
            self.encoding = 'utf-8'
        else:
            self.encoding = 'latin-1'

    def add(self, channel_name, group_index, channel_index):
        """ add name to channels database and check if it contains a source path

        Parameters
        ----------
        channel_name : str
            name that needs to be added to the database
        group_index : int
            data group index
        channel_index : int
            channel index

        """
        if PYVERSION == 2:
            if isinstance(channel_name, unicode):
                channel_name = channel_name.encode(self.encoding)
        if channel_name:
            entry = (group_index, channel_index)
            if channel_name not in self:
                self[channel_name] = []
            self[channel_name].append(
                entry
            )

            if '\\' in channel_name:
                channel_name = channel_name.split('\\')[0]

                if channel_name not in self:
                    self[channel_name] = []
                self[channel_name].append(
                    entry
                )


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
    if PYVERSION >= 3:
        return bytes(randint(65, 90) for _ in range(size - 1)) + b'\0'
    else:
        return ''.join(chr(randint(65, 90)) for _ in range(size - 1)) + '\0'


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

    if not (hasattr(obj, 'read') and hasattr(obj, 'seek')):
        return False

    if not hasattr(obj, "__iter__"):
        return False

    return True

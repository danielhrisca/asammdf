# -*- coding: utf-8 -*-
'''
asammdf utility functions and classes
'''

import logging
import string
import xml.etree.ElementTree as ET

from collections import namedtuple
from struct import unpack
from warnings import warn

from numpy import (
    amin,
    amax,
    where,
)

from . import v2_v3_constants as v3c
from . import v4_constants as v4c

logger = logging.getLogger('asammdf')

__all__ = [
    'CHANNEL_COUNT',
    'CONVERT_LOW',
    'CONVERT_MINIMUM',
    'MERGE_LOW',
    'MERGE_MINIMUM',
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


ALLOWED_MATLAB_CHARS = string.ascii_letters + string.digits + '_'


SignalSource = namedtuple(
    'SignalSource',
    ['name', 'path', 'comment', 'source_type', 'bus_type'],
)


class MdfException(Exception):
    """MDF Exception class"""
    pass


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
    size = unpack('<H', stream.read(2))[0] - 4
    text_bytes = stream.read(size)
    try:
        text = (
            text_bytes
            .decode('latin-1')
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
    size = unpack('<Q', stream.read(8))[0] - 24
    stream.read(8)
    text_bytes = stream.read(size)
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
    if data_type in (
            v3c.DATA_TYPE_STRING,
            v3c.DATA_TYPE_BYTEARRAY):
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

        if data_type in (
                v3c.DATA_TYPE_UNSIGNED_INTEL,
                v3c.DATA_TYPE_UNSIGNED):
            fmt = '<u{}'.format(size)

        elif data_type == v3c.DATA_TYPE_UNSIGNED_MOTOROLA:
            fmt = '>u{}'.format(size)

        elif data_type in (
                v3c.DATA_TYPE_SIGNED_INTEL,
                v3c.DATA_TYPE_SIGNED):
            fmt = '<i{}'.format(size)

        elif data_type == v3c.DATA_TYPE_SIGNED_MOTOROLA:
            fmt = '>i{}'.format(size)

        elif data_type in (
                v3c.DATA_TYPE_FLOAT,
                v3c.DATA_TYPE_DOUBLE,
                v3c.DATA_TYPE_FLOAT_INTEL,
                v3c.DATA_TYPE_DOUBLE_INTEL):
            fmt = '<f{}'.format(size)

        elif data_type in (
                v3c.DATA_TYPE_FLOAT_MOTOROLA,
                v3c.DATA_TYPE_DOUBLE_MOTOROLA):
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
    if data_type in (
            v4c.DATA_TYPE_BYTEARRAY,
            v4c.DATA_TYPE_STRING_UTF_8,
            v4c.DATA_TYPE_STRING_LATIN_1,
            v4c.DATA_TYPE_STRING_UTF_16_BE,
            v4c.DATA_TYPE_STRING_UTF_16_LE,
            v4c.DATA_TYPE_CANOPEN_DATE,
            v4c.DATA_TYPE_CANOPEN_TIME):
        size = size // 8

        if data_type == v4c.DATA_TYPE_BYTEARRAY:
            if channel_type == v4c.CHANNEL_TYPE_VALUE:
                fmt = '({},)u1'.format(size)
            else:
                if size == 4:
                    fmt = '<u4'
                elif size == 8:
                    fmt = '<u8'

        elif data_type in (
                v4c.DATA_TYPE_STRING_UTF_8,
                v4c.DATA_TYPE_STRING_LATIN_1,
                v4c.DATA_TYPE_STRING_UTF_16_BE,
                v4c.DATA_TYPE_STRING_UTF_16_LE):
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


def fix_dtype_fields(fields):
    """ convert field names to str in case of Python 2"""
    new_types = []
    for pair_ in fields:
        new_pair = [str(pair_[0])]
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
        elif fmt.kind in 'SV':
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
            if fmt.byteorder in '=<|':
                data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            else:
                data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
        elif fmt.kind == 'i':
            if fmt.byteorder in '=<|':
                data_type = v4c.DATA_TYPE_SIGNED_INTEL
            else:
                data_type = v4c.DATA_TYPE_SIGNED_MOTOROLA
        elif fmt.kind == 'f':
            if fmt.byteorder in '=<':
                data_type = v4c.DATA_TYPE_REAL_INTEL
            else:
                data_type = v4c.DATA_TYPE_REAL_MOTOROLA
        elif fmt.kind in 'SV':
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
        try:
            min_val, max_val = amin(samples), amax(samples)
        except TypeError:
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


def debug_channel(mdf, group, channel, conversion, dependency):
    """ use this to print debug infromation in case of errors

    Parameters
    ----------
    mdf : MDF
        source MDF object
    group : dict
        group
    channel : Channel
        channel object
    conversion : Channelonversion
        channel conversion object
    dependency : ChannelDependency
        channel dependecy object

    """
    print('MDF', '='*76)
    print('name:', mdf.name)
    print('version:', mdf.version)
    print('memory:', mdf.memory)
    print('read fragment size:', mdf._read_fragment_size)
    print('write fragment size:', mdf._write_fragment_size)
    print()

    parents, dtypes = mdf._prepare_record(group)
    print('GROUP', '='*74)
    print('sorted:', group['sorted'])
    print('data location:', group['data_location'])
    print('record_size:', group['record_size'])
    print('parets:', parents)
    print('dtypes:', dtypes)
    print()

    cg = group['channel_group']
    print('CHANNEL GROUP', '='*66)
    print('record id:', cg['record_id'])
    print('record size:', cg['samples_byte_nr'])
    print('invalidation bytes:', cg.get('invalidation_bytes_nr', 0))
    print('cycles:', cg['cycles_nr'])
    print()

    print('CHANNEL', '='*72)
    print('channel:', channel)
    print('name:', channel.name)
    print('conversion:', conversion)
    print('conversion ref blocks:', conversion.referenced_blocks if conversion else None)
    print()

    print('CHANNEL ARRAY', '='*66)
    print('array:', bool(dependency))
    print()


def count_channel_groups(stream, version=4):
    count = 0
    if version >= 4:
        stream.seek(88, 0)
        dg_addr = unpack('<Q', stream.read(8))[0]
        while dg_addr:
            stream.seek(dg_addr + 32)
            cg_addr = unpack('<Q', stream.read(8))[0]
            while cg_addr:
                count += 1
                stream.seek(cg_addr + 24)
                cg_addr = unpack('<Q', stream.read(8))[0]

            stream.seek(dg_addr + 24)
            dg_addr = unpack('<Q', stream.read(8))[0]

    else:
        stream.seek(68, 0)
        dg_addr = unpack('<I', stream.read(4))[0]
        while dg_addr:
            stream.seek(dg_addr + 8)
            cg_addr = unpack('<I', stream.read(4))[0]
            while cg_addr:
                count += 1
                stream.seek(cg_addr + 4)
                cg_addr = unpack('<I', stream.read(4))[0]

            stream.seek(dg_addr + 4)
            dg_addr = unpack('<I', stream.read(4))[0]

    return count


def validate_memory_argument(memory):
    """ validate the version argument against the supported MDF versions. The
    default version used depends on the hint MDF major revision

    Parameters
    ----------
    version : memory
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

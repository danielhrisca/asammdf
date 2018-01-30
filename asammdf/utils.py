# -*- coding: utf-8 -*-
'''
asammdf utility functions and classes
'''

import warnings

from struct import unpack

from numpy import (
    amin,
    amax,
    where,
)

from . import v2_v3_constants as v3c
from . import v4_constants as v4c

__all__ = [
    'MdfException',
    'get_fmt_v3',
    'get_fmt_v4',
    'get_min_max',
    'get_unique_name',
    'get_text_v4',
    'fix_dtype_fields',
    'fmt_to_datatype_v3',
    'fmt_to_datatype_v4',
    'bytes',
]


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


def get_text_v3(address, stream):
    """ faster way extract string from mdf versions 2 and 3 TextBlock

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

    stream.seek(address + 2)
    size = unpack('<H', stream.read(2))[0] - 4
    text = (
        stream
        .read(size)
        .decode('latin-1')
        .strip(' \r\t\n\0')
    )
    return text


def get_text_v4(address, stream):
    """ faster way extract string from mdf version 4 TextBlock

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
            warnings.warn('Unicode exception occured and "chardet" package is '
                          'not installed. Mdf version 4 expects "utf-8" '
                          'strings and this package may detect if a different'
                          ' encoding was used')
            raise err

    return text


def get_fmt_v3(data_type, size):
    """convert mdf versions 2 and 3 channel data type to numpy dtype format string

    Parameters
    ----------
    data_type : int
        mdf channel data type
    size : int
        data byte size
    Returns
    -------
    fmt : str
        numpy compatible data type format string

    """
    if size == 0:
        fmt = 'b'
    else:
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
        elif data_type == v3c.DATA_TYPE_STRING:
            fmt = 'V{}'.format(size)
        elif data_type == v3c.DATA_TYPE_BYTEARRAY:
            fmt = 'u1'

    return fmt


def get_fmt_v4(data_type, size):
    """convert mdf version 4 channel data type to numpy dtype format string

    Parameters
    ----------
    data_type : int
        mdf channel data type
    size : int
        data byte size

    Returns
    -------
    fmt : str
        numpy compatible data type format string

    """
    if size == 0:
        fmt = 'b'
    else:
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
        elif data_type == v4c.DATA_TYPE_BYTEARRAY:
            fmt = 'V{}'.format(size)
        elif data_type in (
                v4c.DATA_TYPE_STRING_UTF_8,
                v4c.DATA_TYPE_STRING_LATIN_1,
                v4c.DATA_TYPE_STRING_UTF_16_BE,
                v4c.DATA_TYPE_STRING_UTF_16_LE):
            if size == 4:
                fmt = '<u4'
            elif size == 8:
                fmt = '<u8'
            else:
                fmt = 'V{}'.format(size)
        elif data_type == v4c.DATA_TYPE_CANOPEN_DATE:
            fmt = 'V7'
        elif data_type == v4c.DATA_TYPE_CANOPEN_TIME:
            fmt = 'V6'
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


def fmt_to_datatype_v3(fmt):
    """convert numpy dtype format string to mdf versions 2 and 3
    channel data type and size

    Parameters
    ----------
    fmt : numpy.dtype
        numpy data type

    Returns
    -------
    data_type, size : int, int
        integer data type as defined by ASAM MDF and bit size

    """
    size = fmt.itemsize * 8

    if fmt.kind == 'u':
        if fmt.byteorder in '=<':
            data_type = v3c.DATA_TYPE_UNSIGNED
        else:
            data_type = v3c.DATA_TYPE_UNSIGNED_MOTOROLA
    elif fmt.kind == 'i':
        if fmt.byteorder in '=<':
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
    else:
        # here we have arrays
        data_type = v3c.DATA_TYPE_BYTEARRAY

    return data_type, size


def fmt_to_datatype_v4(fmt):
    """convert numpy dtype format string to mdf version 4 channel data
    type and size

    Parameters
    ----------
    fmt : numpy.dtype
        numpy data type

    Returns
    -------
    data_type, size : int, int
        integer data type as defined by ASAM MDF and bit size

    """
    size = fmt.itemsize * 8

    if fmt.kind == 'u':
        if fmt.byteorder in '=<':
            data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
        else:
            data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
    elif fmt.kind == 'i':
        if fmt.byteorder in '=<':
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
    else:
        # here we have arrays
        data_type = v4c.DATA_TYPE_BYTEARRAY

    return data_type, size


def get_unique_name(used_names, name):
    """ returns a list of unique names

    Parameters
    ----------
    used_names : set
        set of already taken names
    names : str
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
    The MDF spec allows values to be encoded as integers that aren't byte-sized. Numpy only knows how to do two's
    complement on byte-sized integers (i.e. int16, int32, int64, etc.), so we have to calculate two's complement
    ourselves in order to handle signed integers with unconventional lengths.

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
    return where(truncated_integers >> bit_length - 1,  # sign bit as a truth series (True when negative)
                 (2**bit_length - truncated_integers) * -1,  # when negative, do two's complement
                 truncated_integers)  # when positive, return the truncated int

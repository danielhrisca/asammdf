# -*- coding: utf-8 -*-
'''
asammdf utility functions and classes
'''

import itertools

from numpy import (
    amin,
    amax,
)

from . import v2_v3_constants as v3c
from . import v4_constants as v4c

__all__ = [
    'MdfException',
    'get_fmt',
    'get_min_max',
    'get_unique_name',
    'fix_dtype_fields',
    'fmt_to_datatype',
    'pair',
    'bytes',
]


class MdfException(Exception):
    """MDF Exception class"""
    pass


def bytes(obj):
    """ Python 2 compatibility function """
    try:
        return obj.__bytes__()
    except AttributeError:
        if isinstance(obj, str):
            return obj
        else:
            raise


def get_fmt(data_type, size, version=3):
    """convert mdf channel data type to numpy dtype format string

    Parameters
    ----------
    data_type : int
        mdf channel data type
    size : int
        data byte size
    version : int
        mdf version; default 3

    Returns
    -------
    fmt : str
        numpy compatible data type format string

    """
    if version <= 3:
        if size == 0:
            fmt = 'b'
        if data_type in (v3c.DATA_TYPE_UNSIGNED_INTEL, v3c.DATA_TYPE_UNSIGNED):
            fmt = '<u{}'.format(size)
        elif data_type == v3c.DATA_TYPE_UNSIGNED_MOTOROLA:
            fmt = '>u{}'.format(size)
        elif data_type in (v3c.DATA_TYPE_SIGNED_INTEL, v3c.DATA_TYPE_SIGNED):
            fmt = '<i{}'.format(size)
        elif data_type == v3c.DATA_TYPE_SIGNED_MOTOROLA:
            fmt = '>i{}'.format(size)
        elif data_type in (v3c.DATA_TYPE_FLOAT,
                           v3c.DATA_TYPE_DOUBLE,
                           v3c.DATA_TYPE_FLOAT_INTEL,
                           v3c.DATA_TYPE_DOUBLE_INTEL):
            fmt = '<f{}'.format(size)
        elif data_type in (v3c.DATA_TYPE_FLOAT_MOTOROLA,
                           v3c.DATA_TYPE_DOUBLE_MOTOROLA):
            fmt = '>f{}'.format(size)
        elif data_type == v3c.DATA_TYPE_STRING:
            fmt = 'V{}'.format(size)
        elif data_type == v3c.DATA_TYPE_BYTEARRAY:
            fmt = 'u1'

    elif version == 4:
        if size == 0:
            fmt = 'b'
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
        elif data_type in (v4c.DATA_TYPE_STRING_UTF_8,
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


def fmt_to_datatype(fmt, version=3):
    """convert numpy dtype format string to mdf channel data type and size

    Parameters
    ----------
    fmt : numpy.dtype
        numpy data type
    version : int
        MDF version (2, 3 or 4); default is 3

    Returns
    -------
    data_type, size : int, int
        integer data type as defined by ASAM MDF and bit size

    """
    size = fmt.itemsize * 8

    if version < 4:
        if fmt.kind == 'u':
            if fmt.byteorder in ('=<'):
                data_type = v3c.DATA_TYPE_UNSIGNED
            else:
                data_type = v3c.DATA_TYPE_UNSIGNED_MOTOROLA
        elif fmt.kind == 'i':
            if fmt.byteorder in ('=<'):
                data_type = v3c.DATA_TYPE_SIGNED
            else:
                data_type = v3c.DATA_TYPE_SIGNED_MOTOROLA
        elif fmt.kind == 'f':
            if fmt.byteorder in ('=<'):
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

    elif version == 4:

        if fmt.kind == 'u':
            if fmt.byteorder in ('=<'):
                data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
            else:
                data_type = v4c.DATA_TYPE_UNSIGNED_MOTOROLA
        elif fmt.kind == 'i':
            if fmt.byteorder in ('=<'):
                data_type = v4c.DATA_TYPE_SIGNED_INTEL
            else:
                data_type = v4c.DATA_TYPE_SIGNED_MOTOROLA
        elif fmt.kind == 'f':
            if fmt.byteorder in ('=<'):
                data_type = v4c.DATA_TYPE_REAL_INTEL
            else:
                data_type = v4c.DATA_TYPE_REAL_MOTOROLA
        elif fmt.kind in 'SV':
            data_type = v4c.DATA_TYPE_STRING_LATIN_1
        else:
            # here we have arrays
            data_type = v4c.DATA_TYPE_BYTEARRAY

    return data_type, size


def pair(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    current, next_ = itertools.tee(iterable)
    next(next_, None)
    return zip(current, next_)


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

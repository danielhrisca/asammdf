# -*- coding: utf-8 -*-
'''
asammdf utility functions and classes
'''

import itertools
import re

from numpy import (
    issubdtype,
    signedinteger,
    unsignedinteger,
    floating,
    flexible,
    amin,
    amax,
)


from . import v3constants as v3c
from . import v4constants as v4c


__all__ = [
    'MdfException',
    'get_fmt',
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


def dtype_mapping(invalue, outversion=3):
    """ map data types between mdf versions 3 and 4

    Parameters
    ----------
    invalue : int
        original data type
    outversion : int
        mdf version of output data type

    Returns
    -------
    res : int
        mapped data type

    """

    v3tov4 = {v3c.DATA_TYPE_UNSIGNED: v4c.DATA_TYPE_UNSIGNED_INTEL,
              v3c.DATA_TYPE_SIGNED: v4c.DATA_TYPE_SIGNED_INTEL,
              v3c.DATA_TYPE_FLOAT: v4c.DATA_TYPE_REAL_INTEL,
              v3c.DATA_TYPE_DOUBLE: v4c.DATA_TYPE_REAL_INTEL,
              v3c.DATA_TYPE_STRING: v4c.DATA_TYPE_STRING_LATIN_1,
              v3c.DATA_TYPE_UNSIGNED_INTEL: v4c.DATA_TYPE_UNSIGNED_INTEL,
              v3c.DATA_TYPE_UNSIGNED_INTEL: v4c.DATA_TYPE_UNSIGNED_INTEL,
              v3c.DATA_TYPE_SIGNED_INTEL: v4c.DATA_TYPE_SIGNED_INTEL,
              v3c.DATA_TYPE_SIGNED_INTEL: v4c.DATA_TYPE_SIGNED_INTEL,
              v3c.DATA_TYPE_FLOAT_INTEL: v4c.DATA_TYPE_REAL_INTEL,
              v3c.DATA_TYPE_FLOAT_INTEL: v4c.DATA_TYPE_REAL_INTEL,
              v3c.DATA_TYPE_DOUBLE_INTEL: v4c.DATA_TYPE_REAL_INTEL,
              v3c.DATA_TYPE_DOUBLE_INTEL: v4c.DATA_TYPE_REAL_INTEL}

    v4tov3 = {v4c.DATA_TYPE_UNSIGNED_INTEL: v3c.DATA_TYPE_UNSIGNED_INTEL,
              v4c.DATA_TYPE_UNSIGNED_MOTOROLA: v3c.DATA_TYPE_UNSIGNED_MOTOROLA,
              v4c.DATA_TYPE_SIGNED_INTEL: v3c.DATA_TYPE_SIGNED_INTEL,
              v4c.DATA_TYPE_STRING_LATIN_1: v3c.DATA_TYPE_STRING,
              v4c.DATA_TYPE_BYTEARRAY: v3c.DATA_TYPE_STRING,
              v4c.DATA_TYPE_REAL_INTEL: v3c.DATA_TYPE_DOUBLE_INTEL,
              v4c.DATA_TYPE_REAL_MOTOROLA: v3c.DATA_TYPE_DOUBLE_MOTOROLA,
              v4c.DATA_TYPE_SIGNED_MOTOROLA: v3c.DATA_TYPE_SIGNED_MOTOROLA}

    if outversion == 3:
        res = v4tov3[invalue]
    else:
        res = v3tov4[invalue]
    return res


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
    if version == 3:
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
        MDF version; default is 3

    Returns
    -------
    data_type, size : int, int
        integer data type as defined by ASAM MDF and bit size

    """
    size = fmt.itemsize * 8

    if version == 3:
        if issubdtype(fmt, unsignedinteger):
            data_type = v3c.DATA_TYPE_UNSIGNED
        elif issubdtype(fmt, signedinteger):
            data_type = v3c.DATA_TYPE_SIGNED
        elif issubdtype(fmt, floating):
            if size == 32:
                data_type = v3c.DATA_TYPE_FLOAT
            else:
                data_type = v3c.DATA_TYPE_DOUBLE
        elif issubdtype(fmt, flexible):
            data_type = v3c.DATA_TYPE_STRING
        else:
            # here we have arrays
            data_type = v3c.DATA_TYPE_BYTEARRAY

    elif version == 4:
        if issubdtype(fmt, unsignedinteger):
            data_type = v4c.DATA_TYPE_UNSIGNED_INTEL
        elif issubdtype(fmt, signedinteger):
            data_type = v4c.DATA_TYPE_SIGNED_INTEL
        elif issubdtype(fmt, floating):
            data_type = v4c.DATA_TYPE_REAL_INTEL
        elif issubdtype(fmt, flexible):
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


def load_dbc(dbc):
    """ Loads all messages description from DBC

    Parameters
    ----------
    dbc : str
        DBC file path

    Returns
    -------
    messages : dict
        the keys are the message ID's from the dbc

    """

    pattern = r'(?P<msg>^BO_ (.+\n)+)'

    with open(dbc, 'r') as dbc_file:
        string = dbc_file.read()

    messages = {}

    for match_ in re.finditer(pattern, string, flags=re.M):
        msg = match_.group('msg')

        pattern = r'BO_ (?P<can_id>\d+) (?P<name>[^ :]+): (?P<dlc>\d).+'
        match = re.search(pattern, msg)
        can_id = int(match.group('can_id'))
        name = match.group('name')
        dlc = int(match.group('dlc'))

        pattern = (r'SG_ (?P<name>[^ ]+) : '
                   r'(?P<start_bit>\d{1,2})\|(?P<size>\d{1,2})'
                   r'@(?P<byte_order>\d)(?P<signed>[+-])'
                   r' \((?P<factor>[^,]+),(?P<offset>[^)]+)\)'
                   r' \[(?P<min_value>[^|]+)\|(?P<max_value>[^]]+)\]'
                   r' "(?P<unit>[^"]*)"')

        messages[can_id] = {
            'name': name,
            'dlc': dlc,
            'signals': {},
            'can_id': can_id
        }

        signals = messages[can_id]['signals']

        for match in re.finditer(pattern, msg):
            signal_name = match.group('name')
            start_bit = int(match.group('start_bit'))
            size = int(match.group('size'))
            byte_order = match.group('byte_order')
            signed = match.group('signed') == '-'
            factor = float(match.group('factor'))
            offset = float(match.group('offset'))
            min_value = float(match.group('min_value'))
            max_value = float(match.group('max_value'))
            unit = match.group('unit')
            signals[signal_name] = {'start_bit': start_bit,
                                    'size': size,
                                    'byte_order': byte_order,
                                    'signed': signed,
                                    'factor': factor,
                                    'offset': offset,
                                    'min_value': min_value,
                                    'max_value': max_value,
                                    'unit': unit}

    return messages

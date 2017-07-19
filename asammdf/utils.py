'''
utility functions and classes
'''
import itertools
from numpy import issubdtype, signedinteger, unsignedinteger, floating, character
from . import v3constants as v3
from . import v4constants as v4


__all__ = ['MdfException',
           'get_fmt',
           'fmt_to_datatype',
           'pair']


class MdfException(Exception):
    """MDF Exception class"""
    pass


def get_fmt(data_type, size, version=3):
    """"Summary line.

    Extended description of function.

    Parameters
    ----------
    data_type : Type of data_type
        Description of data_type default None
    size : Type of size
        Description of size default None
    version : int
        mdf version; default 3

    Returns
    -------

    Examples
    --------
    >>>>

    """
    if version == 3:
        if size == 0:
            fmt = 'b'
        if data_type in (v3.DATA_TYPE_UNSIGNED_INTEL, v3.DATA_TYPE_UNSIGNED):
            fmt = '<u{}'.format(size)
        elif data_type == v3.DATA_TYPE_UNSIGNED_MOTOROLA:
            fmt = '>u{}'.format(size)
        elif data_type in (v3.DATA_TYPE_SIGNED_INTEL, v3.DATA_TYPE_SIGNED):
            fmt = '<i{}'.format(size)
        elif data_type == v3.DATA_TYPE_SIGNED_MOTOROLA:
            fmt = '>i{}'.format(size)
        elif data_type in (v3.DATA_TYPE_FLOAT,
                           v3.DATA_TYPE_DOUBLE,
                           v3.DATA_TYPE_FLOAT_INTEL,
                           v3.DATA_TYPE_DOUBLE_INTEL):
            fmt = '<f{}'.format(size)
        elif data_type in (v3.DATA_TYPE_FLOAT_MOTOROLA,
                           v3.DATA_TYPE_DOUBLE_MOTOROLA):
            fmt = '>f{}'.format(size)
        elif data_type == v3.DATA_TYPE_STRING:
            fmt = 'a{}'.format(size)
    elif version == 4:
        if size == 0:
            fmt = 'b'
        if data_type == v4.DATA_TYPE_UNSIGNED_INTEL:
            fmt = '<u{}'.format(size)
        elif data_type == v4.DATA_TYPE_UNSIGNED_MOTOROLA:
            fmt = '>u{}'.format(size)
        elif data_type == v4.DATA_TYPE_SIGNED_INTEL:
            fmt = '<i{}'.format(size)
        elif data_type == v4.DATA_TYPE_SIGNED_MOTOROLA:
            fmt = '>i{}'.format(size)
        elif data_type == v4.DATA_TYPE_REAL_INTEL:
            fmt = '<f{}'.format(size)
        elif data_type == v4.DATA_TYPE_REAL_MOTOROLA:
            fmt = '>f{}'.format(size)
        elif data_type in (v4.DATA_TYPE_BYTEARRAY,
                           v4.DATA_TYPE_STRING_UTF_8,
                           v4.DATA_TYPE_STRING_LATIN_1,
                           v4.DATA_TYPE_STRING_UTF_16_BE,
                           v4.DATA_TYPE_STRING_UTF_16_LE):
            fmt = 'a{}'.format(size)
        elif data_type == v4.DATA_TYPE_CANOPEN_DATE:
            fmt = 'a7'
        elif data_type == v4.DATA_TYPE_CANOPEN_TIME:
            fmt = 'a6'
    return fmt


def fmt_to_datatype(fmt, version=3):
    """"Summary line.

    Extended description of function.

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
            data_type = v3.DATA_TYPE_UNSIGNED
        elif issubdtype(fmt, signedinteger):
            data_type = v3.DATA_TYPE_SIGNED
        elif issubdtype(fmt, floating):
            data_type = v3.DATA_TYPE_FLOAT if size == 32 else v3.DATA_TYPE_DOUBLE
        elif issubdtype(fmt, character):
            data_type = v3.DATA_TYPE_STRING
    elif version == 4:
        if issubdtype(fmt, unsignedinteger):
            data_type = v4.DATA_TYPE_UNSIGNED_INTEL
        elif issubdtype(fmt, signedinteger):
            data_type = v4.DATA_TYPE_SIGNED_INTEL
        elif issubdtype(fmt, floating):
            data_type = v4.DATA_TYPE_REAL_INTEL
        elif issubdtype(fmt, character):
            data_type = v4.DATA_TYPE_STRING_UTF_8
    return data_type, size


def pair(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    current, next_ = itertools.tee(iterable)
    next(next_, None)
    return zip(current, next_)

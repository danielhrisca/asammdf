# -*- coding: utf-8 -*-
""" asammdf is a parser and editor for ASAM MDF files """

from .mdf_v2 import MDF2
from .mdf_v3 import MDF3
from .mdf_v4 import MDF4
from .mdf import MDF, SUPPORTED_VERSIONS
from .signal import Signal
from .version import __version__

__all__ = [
    '__version__',
    'configure',
    'MDF',
    'MDF2',
    'MDF3',
    'MDF4',
    'Signal',
    'SUPPORTED_VERSIONS',
]


def configure(
        split_data_blocks=None,
        split_threshold=None,
        overwrite=None):
    """ configure asammdf parameters

    Note
    ----
    this is not thread safe

    Parameters
    ----------
    split_data_blocks : bool
        enable/disable splitting of large data blocks using data lists for
        mdf version 4; default is `True`
    split_treshold : int
        size hint of splitted data blocks, default 8MB; if the initial size is
        smaller, then no data list is used. The actual split size depends on
        the data groups' records size
    overwrite : bool
        default option for save method's overwrite argument

    """

    if split_threshold is not None:
        MDF4._split_threshold = int(split_threshold)

    if split_data_blocks is not None:
        MDF4._split_data_blocks = bool(split_data_blocks)

    if overwrite is not None:
        MDF3._overwrite = bool(overwrite)
        MDF4._overwrite = bool(overwrite)

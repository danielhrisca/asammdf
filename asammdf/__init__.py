# -*- coding: utf-8 -*-
""" asammdf is a parser and editor for ASAM MDF files """
from .mdf3 import MDF3
from .mdf4 import MDF4
from .mdf import MDF
from .signal import Signal
from .version import __version__


__all__ = [
    '__version__',
    'configure',
    'MDF',
    'MDF3',
    'MDF4',
    'Signal',
]


def configure(
        integer_compacting=None,
        split_data_blocks=None,
        split_threshold=None):
    """ configure asammdf parameters

    Parameters
    ----------
    integer_compacting : bool
        enable/disable compacting of integer channels on append. This has the
        potential to greatly reduce file size, but append speed is slower and
        further loading of the resulting file will also be slower.
    split_data_blocks : bool
        enable/disable splitting of large data blocks using data lists for
        mdf version 4
    split_treshold : int
        size of splitted data blocks, default 2MB; if the initial size is
        smaller then no data list is used

    """

    if integer_compacting is not None:
        MDF3._compact_integers_on_append = integer_compacting
        MDF4._compact_integers_on_append = integer_compacting

    if split_threshold is not None:
        MDF4._split_threshold = split_threshold

    if split_data_blocks is not None:
        MDF4._split_data_blocks = split_data_blocks

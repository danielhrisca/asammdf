# -*- coding: utf-8 -*-
""" asammdf is a parser and editor for ASAM MDF files """

from .mdf_v2_v3 import MDF_V2_V3
from .mdf_v4 import MDF_V4
from .mdf import MDF, SUPPORTED_VERSIONS
from .signal import Signal
from .version import __version__

__all__ = [
    '__version__',
    'configure',
    'MDF',
    'MDF_V2_V3',
    'MDF_V4',
    'Signal',
    'SUPPORTED_VERSIONS',
]


def configure(
        integer_compacting=None,
        split_data_blocks=None,
        split_threshold=None,
        overwrite=None,
        iter_channels=None):
    """ configure asammdf parameters

    Note
    ----
    this is not thread safe

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
        size hint of splitted data blocks, default 2MB; if the initial size is
        smaller then no data list is used. The actual split size depends on
        the data groups' records size
    overwrite : bool
        default option for save method's overwrite argument
    iter_channels : bool
        default option to yield channels instead of pandas DataFrame when
        iterating over the MDF objects

    """

    if integer_compacting is not None:
        MDF_V2_V3._compact_integers_on_append = bool(integer_compacting)
        MDF_V4._compact_integers_on_append = bool(integer_compacting)

    if split_threshold is not None:
        MDF_V4._split_threshold = int(split_threshold)

    if split_data_blocks is not None:
        MDF_V4._split_data_blocks = bool(split_data_blocks)

    if overwrite is not None:
        MDF_V2_V3._overwrite = bool(overwrite)
        MDF_V4._overwrite = bool(overwrite)

    if iter_channels is not None:
        MDF._iter_channels = bool(iter_channels)

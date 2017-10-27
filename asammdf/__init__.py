# -*- coding: utf-8 -*-
""" asammdf is a parser and editor for ASAM MDF files """
from .mdf3 import MDF3
from .mdf4 import MDF4
from .mdf import MDF
from .signal import Signal
from .version import __version__


__all__ = [
    '__version__',
    'enable_integer_compacting',
    'MDF',
    'MDF3',
    'MDF4',
    'Signal',
]


def enable_integer_compacting(enable):
    """ enable or disable compacting of integer channels when appending.
    This has the potential to greatly reduce file size, but append speed is
    slower and further loading of the resulting file will also be slower.

    Parameters
    ----------
    enable : bool

    """

    MDF3._enable_integer_compacting(enable)
    MDF4._enable_integer_compacting(enable)

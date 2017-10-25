# -*- coding: utf-8 -*-
""" asammdf is a parser and editor for ASAM MDF files """
from .mdf3 import MDF3
from .mdf4 import MDF4
from .mdf import MDF
from .signal import Signal

__version__ = '2.6.3'

__all__ = [
    '__version__',
    'MDF',
    'MDF3',
    'MDF4',
    'Signal',
]

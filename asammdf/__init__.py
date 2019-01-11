# -*- coding: utf-8 -*-
""" asammdf is a parser and editor for ASAM MDF files """

import logging

logging.getLogger('asammdf').addHandler(logging.NullHandler())

from .blocks.mdf_v2 import MDF2
from .blocks.mdf_v3 import MDF3
from .blocks.mdf_v4 import MDF4
from .mdf import MDF, SUPPORTED_VERSIONS
from .signal import Signal
from .version import __version__


__all__ = ["__version__", "MDF", "MDF2", "MDF3", "MDF4", "Signal", "SUPPORTED_VERSIONS"]

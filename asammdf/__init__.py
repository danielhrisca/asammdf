# -*- coding: utf-8 -*-
""" asammdf is a parser and editor for ASAM MDF files """

import logging

logger = logging.getLogger('asammdf')
logger.setLevel(logging.INFO)

from .mdf_v2 import MDF2
from .mdf_v3 import MDF3
from .mdf_v4 import MDF4
from .mdf import MDF, SUPPORTED_VERSIONS
from .signal import Signal
from .version import __version__


__all__ = [
    '__version__',
    'MDF',
    'MDF2',
    'MDF3',
    'MDF4',
    'Signal',
    'SUPPORTED_VERSIONS',
]



# -*- coding: utf-8 -*-
""" asammdf is a parser and editor for ASAM MDF files """

import logging

logger = logging.getLogger("asammdf")
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.ERROR)

from .blocks.options import get_global_option, set_global_option
from .blocks.source_utils import Source
from .gui import plot
from .mdf import MDF, SUPPORTED_VERSIONS
from .signal import Signal
from .version import __version__

try:
    from .blocks import cutils

    __cextension__ = True
except ImportError:
    __cextension__ = False

__all__ = [
    "__cextension__",
    "__version__",
    "get_global_option",
    "set_global_option",
    "MDF",
    "plot",
    "Signal",
    "Source",
    "SUPPORTED_VERSIONS",
]

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


from .blocks.mdf_v2 import MDF2
from .blocks.mdf_v3 import MDF3
from .blocks.mdf_v4 import MDF4
from .mdf import MDF, SUPPORTED_VERSIONS
from .gui import plot
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
    "MDF",
    "MDF2",
    "MDF3",
    "MDF4",
    "Signal",
    "SUPPORTED_VERSIONS",
    "plot",
]

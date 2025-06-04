"""asammdf is a parser and editor for ASAM MDF files"""

import logging

logger = logging.getLogger("asammdf")
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.ERROR)

# patch for hdf5storage
import numpy as np

if not hasattr(np, "unicode_"):
    setattr(np, "unicode_", np.str_)  # noqa: B010

from .blocks.options import get_global_option, set_global_option
from .blocks.source_utils import Source
from .gui import plot
from .mdf import MDF, SUPPORTED_VERSIONS
from .signal import InvalidationArray, Signal
from .version import __version__

try:
    from .blocks import cutils  # noqa: F401

    __cextension__ = True
except ImportError:
    __cextension__ = False

__all__ = [
    "MDF",
    "SUPPORTED_VERSIONS",
    "InvalidationArray",
    "Signal",
    "Source",
    "__cextension__",
    "__version__",
    "get_global_option",
    "plot",
    "set_global_option",
]

# -*- coding: utf-8 -*-
""" asammdf is a parser and editor for ASAM MDF files """

import logging

logger = logging.getLogger("asammdf.recorder")
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.WARNING)

from .recorder import SignalDescription, Recorder
from .plugin import PluginBase


__all__ = [
    "SignalDescription", "Recorder", "PluginBase",
]

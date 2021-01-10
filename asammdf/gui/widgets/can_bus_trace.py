# -*- coding: utf-8 -*-
import datetime
import logging
from traceback import format_exc

import numpy as np
import numpy.core.defchararray as npchar
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from ...blocks.utils import (
    csv_bytearray2hex,
    csv_int2bin,
    csv_int2hex,
    pandas_query_compatible,
)
from ..ui import resource_rc as resource_rc
from ..utils import run_thread_with_progress
from .tabular_base import TabularTreeItem, TabularBase
from .tabular_filter import TabularFilter

logger = logging.getLogger("asammdf.gui")
LOCAL_TIMEZONE = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo


class CANBusTrace(TabularBase):
    add_channels_request = QtCore.pyqtSignal(list)

    def __init__(self, signals=None, start=0, format="phys", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.signals_descr = {name: 0 for name in signals.columns}
        self.start = start
        self.pattern = {}
        self.format = format
        self.format_selection.setCurrentText(format)

        if signals is None:
            self.signals = pd.DataFrame()
        else:
            self.signals = signals

        self.as_hex = [
            name.endswith(
                (
                    "CAN_DataFrame.ID",
                    "FLX_Frame.ID",
                    "FLX_DataFrame.ID",
                    "LIN_Frame.ID",
                    "MOST_DataFrame.ID",
                    "ETH_Frame.ID",
                )
            )
            for name in self.signals.columns
        ]

        self._original_timestamps = signals["timestamps"]

        self.build(self.signals, True)

        prefixes = set()
        for name in self.signals.columns:
            while "." in name:
                name = name.rsplit(".", 1)[0]
                prefixes.add(f"{name}.")

        self.filters.minimal_menu = True

        self.prefix.insertItems(0, sorted(prefixes, key=lambda x: (-len(x), x)))
        self.prefix.setEnabled(False)

        self.prefix.currentIndexChanged.connect(self.prefix_changed)

        if prefixes:
            self.remove_prefix.setCheckState(QtCore.Qt.Checked)

        self._settings = QtCore.QSettings()
        integer_mode = self._settings.value("tabular_format", "phys")

        self.format_selection.setCurrentText(integer_mode)

    def _display(self, position):
        super()._display(position)
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)

        while iterator.value():
            item = iterator.value()
            if item.text(4) == 'Error Frame':
                item.setForeground(4, QtGui.QBrush(QtCore.Qt.darkRed))
            elif item.text(4) == 'Remote Frame':
                item.setForeground(4, QtGui.QBrush(QtCore.Qt.darkGreen))
            iterator += 1

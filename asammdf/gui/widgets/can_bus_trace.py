# -*- coding: utf-8 -*-
import datetime
import logging

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from .tabular_base import TabularBase

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
        columns = self.tree.columnCount()

        try:
            event_index = self.signals.columns.get_loc("Event Type") + 1
        except:
            event_index = self.signals.columns.get_loc("Event_Type") + 1

        while iterator.value():
            item = iterator.value()
            if item.text(event_index) == "Error Frame":
                for col in range(columns):
                    item.setForeground(col, QtGui.QBrush(QtCore.Qt.darkRed))
            elif item.text(event_index) == "Remote Frame":
                for col in range(columns):
                    item.setForeground(col, QtGui.QBrush(QtCore.Qt.darkGreen))
            iterator += 1

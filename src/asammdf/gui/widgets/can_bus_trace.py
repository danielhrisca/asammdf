import logging

import dateutil.tz
import pandas as pd
from PySide6 import QtCore, QtGui

from .tabular_base import TabularBase

logger = logging.getLogger("asammdf.gui")
LOCAL_TIMEZONE = dateutil.tz.tzlocal()


class CANBusTrace(TabularBase):
    add_channels_request = QtCore.Signal(list)

    def __init__(self, signals=None, start=0, format="phys", ranges=None, *args, **kwargs):
        ranges = ranges or {name: [] for name in signals.columns}
        if not ranges["Event Type"]:
            ranges["Event Type"] = [
                {
                    "background_color": QtGui.QColor("#ff0000"),
                    "font_color": QtGui.QColor("#000000"),
                    "op1": "==",
                    "op2": "==",
                    "value1": "Error Frame",
                    "value2": None,
                },
                {
                    "background_color": QtGui.QColor("#00ff00"),
                    "font_color": QtGui.QColor("#000000"),
                    "op1": "==",
                    "op2": "==",
                    "value1": "Remote Frame",
                    "value2": None,
                },
            ]

        super().__init__(signals, ranges)

        self.signals_descr = {name: 0 for name in signals.columns}
        self.start = start.astimezone(LOCAL_TIMEZONE)
        self.pattern = {}
        self.format = format
        self.format_selection.setCurrentText(format)

        self._original_timestamps = signals["timestamps"]
        self._original_ts_series = pd.Series(
            self._original_timestamps,
            index=self.tree.pgdf.df_unfiltered.index,
        )

        prefixes = set()
        for name in self.tree.pgdf.df_unfiltered.columns:
            while "." in name:
                name = name.rsplit(".", 1)[0]
                prefixes.add(f"{name}.")

        self.filters.minimal_menu = True

        self.prefix.insertItems(0, sorted(prefixes, key=lambda x: (-len(x), x)))
        self.prefix.setEnabled(False)

        self.prefix.currentIndexChanged.connect(self.prefix_changed)

        if prefixes:
            self.remove_prefix.setCheckState(QtCore.Qt.CheckState.Checked)

        self._settings = QtCore.QSettings()
        integer_mode = self._settings.value("tabular_format", "phys")

        self.format_selection.setCurrentText(integer_mode)

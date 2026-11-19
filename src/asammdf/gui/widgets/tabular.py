import logging

import dateutil.tz
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from .tabular_base import TabularBase

logger = logging.getLogger("asammdf.gui")
LOCAL_TIMEZONE = dateutil.tz.tzlocal()


class Tabular(TabularBase):
    add_channels_request = QtCore.Signal(list)

    def __init__(
        self, signals=None, start=None, format="phys", ranges=None, owner=None, *args, channel_selection=None, **kwargs
    ):
        # super().__init__(*args, **kwargs)

        self.signals_descr = {}
        self.start = start.astimezone(LOCAL_TIMEZONE)
        self.pattern = {}
        self.format = format
        self.owner = owner
        self.channel_selection = {} if channel_selection is None else channel_selection

        if signals is None:
            signals = pd.DataFrame()
        else:
            index = pd.Series(np.arange(len(signals), dtype="u8"), index=signals.index)
            signals["Index"] = index

            signals["timestamps"] = signals.index

            if ranges is not None:
                ranges["timestamps"] = []
            signals.set_index(index, inplace=True)
            signals = self._format_channel_values(signals.drop(columns="Index"))
            self.signals_descr.update(dict.fromkeys(signals.columns, 0))

            names = list(signals.columns)
            names = [
                "timestamps",
                *[name for name in names if name.endswith((".ID", ".DataBytes"))],
                *[name for name in names if name != "timestamps" and not name.endswith((".ID", ".DataBytes"))],
            ]

            signals = signals[names]

        super().__init__(signals, ranges)
        self.format_selection.setCurrentText(format)

        self._original_timestamps = signals["timestamps"]
        self._original_ts_series = pd.Series(
            self._original_timestamps,
            index=index,
        )

        # self.build(self.signals, True, ranges=ranges)

        prefixes = set()
        for name in signals.columns:
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
        integer_mode = self._settings.value("tabular/integer_format", "phys")

        self.format_selection.setCurrentText(integer_mode)

        self.tree.dataView.setAcceptDrops(True)
        self.tree.dataView.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.tree.dataView.setDropIndicatorShown(True)
        self.tree.dataView.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)

    def close(self):
        self.owner = None
        super().close()

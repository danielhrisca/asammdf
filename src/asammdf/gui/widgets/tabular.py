import logging
import re

import dateutil.tz
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from ...blocks.utils import (
    csv_bytearray2hex,
)
from .tabular_base import TabularBase

try:
    npchar = np.strings
except:
    npchar = np.char

logger = logging.getLogger("asammdf.gui")
LOCAL_TIMEZONE = dateutil.tz.tzlocal()
DATA_BYTES = re.compile(r"(?P<data>.*?Data ?)Bytes(?P<cntr>_\d+)?")


def data_length_name(match):
    result = f"{match.group('data')}Length"
    if cntr := match.group("cntr"):
        result += cntr

    return result


class Tabular(TabularBase):
    add_channels_request = QtCore.Signal(list)

    def __init__(self, signals=None, start=None, format="phys", ranges=None, owner=None, *args, **kwargs):
        # super().__init__(*args, **kwargs)

        self.signals_descr = {}
        self.start = start.astimezone(LOCAL_TIMEZONE)
        self.pattern = {}
        self.format = format
        self.owner = owner

        if signals is None:
            signals = pd.DataFrame()
        else:
            index = pd.Series(np.arange(len(signals), dtype="u8"), index=signals.index)
            signals["Index"] = index

            signals["timestamps"] = signals.index

            if ranges is not None:
                ranges["timestamps"] = []
            signals.set_index(index, inplace=True)
            dropped = {}

            columns = list(signals.columns)
            columns.remove("Index")

            for name_ in signals.columns:
                col = signals[name_]

                if col.dtype.kind == "O":
                    if match := DATA_BYTES.fullmatch(name_):
                        length_name = data_length_name(match)
                        try:
                            sizes = signals[length_name].fillna(value=65535).astype("u2")
                            sizes = [val if val != 65535 else None for val in sizes.to_list()]
                        except KeyError:
                            sizes = None

                        dropped[name_] = pd.Series(
                            csv_bytearray2hex(
                                col,
                                sizes,
                            ),
                            index=signals.index,
                        )

                    elif col.dtype.name != "category":
                        if len(col) and col.dtype.kind == "u" and col.dtype.itemsize == 1:
                            try:
                                dropped[name_] = pd.Series(csv_bytearray2hex(col), index=signals.index)
                            except:
                                pass

                    self.signals_descr[name_] = 0

                elif col.dtype.kind == "S":
                    try:
                        dropped[name_] = pd.Series(npchar.decode(col, "utf-8"), index=signals.index)
                    except:
                        dropped[name_] = pd.Series(npchar.decode(col, "latin-1"), index=signals.index)
                    self.signals_descr[name_] = 0
                else:
                    self.signals_descr[name_] = 0

            signals = signals.drop(columns=["Index", *list(dropped)])
            for name, s in dropped.items():
                signals[name] = s

            if dropped:
                signals = signals[columns]

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
        integer_mode = self._settings.value("tabular_format", "phys")

        self.format_selection.setCurrentText(integer_mode)

        self.tree.dataView.setAcceptDrops(True)
        self.tree.dataView.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.tree.dataView.setDropIndicatorShown(True)
        self.tree.dataView.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)

    def close(self):
        self.owner = None
        super().close()

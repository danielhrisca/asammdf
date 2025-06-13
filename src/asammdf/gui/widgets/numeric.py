import bisect
from functools import partial
import json
import os
from pathlib import Path
import re
from threading import Lock
from traceback import format_exc

from natsort import natsorted
import numpy as np
from numpy import searchsorted
import pyqtgraph.functions as fn
from PySide6 import QtCore, QtGui, QtWidgets

from asammdf.gui import utils
from asammdf.gui.dialogs.range_editor import RangeEditor
from asammdf.gui.utils import (
    copy_ranges,
    get_colors_using_ranges,
    unique_ranges,
    value_as_str,
)
from asammdf.gui.widgets.plot import PlotSignal
import asammdf.mdf as mdf_module

from .. import serde
from ..serde import ExtendedJsonDecoder, ExtendedJsonEncoder, extract_mime_names
from ..ui.numeric_offline import Ui_NumericDisplay
from ..utils import FONT_SIZE
from .tree import substitude_mime_uuids

HERE = Path(__file__).resolve().parent


OPS = {
    "!=": "__ne__",
    "==": "__eq__",
    ">": "__gt__",
    ">=": "__ge__",
    "<": "__lt__",
    "<=": "__le__",
}


class SignalOnline:
    def __init__(
        self,
        name="",
        raw=None,
        scaled=None,
        unit="",
        entry=(),
        conversion=None,
        exists=True,
        format="phys",
        color="#505050",
        y_range=(0, 100),
    ):
        self.name = name
        self.raw = raw
        self.scaled = scaled
        self.unit = unit
        self.entry = entry
        self.conversion = conversion
        self.exists = exists
        self.configured_from_device = True
        self.format = format
        self.y_range = y_range

        color = color or "#505050"
        self.color = fn.mkColor(color)

    @property
    def origin_mdf(self):
        return ""

    @origin_mdf.setter
    def origin_mdf(self, value):
        pass

    @property
    def origin_uuid(self):
        return self.entry[0]

    @origin_uuid.setter
    def origin_uuid(self, value):
        self.entry = (value, self.name)

    def reset(self):
        self.raw = None
        self.scaled = None
        self.exists = True

    def update_values(self, values):
        self.raw = values[-1]
        if self.conversion:
            self.scaled = self.conversion.convert(values[-1:], as_bytes=True)[0]
        else:
            self.scaled = self.raw

    def __lt__(self, other):
        return self.name < other.name

    def get_value(self, index):
        if index == 0:
            return self.name
        elif index == 1:
            return self.raw
        elif index == 2:
            return self.scaled
        elif index == 3:
            return self.unit
        elif index == 4:
            return self.origin_uuid


class SignalOffline:
    def __init__(
        self,
        signal=None,
        exists=True,
    ):
        self.signal = signal
        self.exists = exists
        self.raw = None
        self.scaled = None
        self.last_timestamp = None
        self.entry = signal.entry
        self.name = signal.name
        self.unit = signal.unit
        self.format = getattr(signal, "format", "phys")

        if not hasattr(signal, "color"):
            self.color = fn.mkColor("#505050")

        if not hasattr(signal, "y_range"):
            self.y_range = (0, 100)

    @property
    def y_range(self):
        return self.signal.y_range

    @y_range.setter
    def y_range(self, value):
        self.signal.y_range = value

    @property
    def color(self):
        return self.signal.color

    @color.setter
    def color(self, value):
        self.signal.color = value

    @property
    def origin_mdf(self):
        return self.signal.origin_mdf

    @origin_mdf.setter
    def origin_mdf(self, value):
        self.signal.origin_mdf = value

    @property
    def origin_uuid(self):
        return self.signal.origin_uuid

    @origin_uuid.setter
    def origin_uuid(self, value):
        self.signal.origin_uuid = value

    def reset(self, exists=True):
        self.signal.samples = self.signal.samples[:0]
        self.signal.timestamps = self.signal.timestamps[:0]
        self.exists = exists
        self.raw = None
        self.scaled = None
        self.last_timestamp = None

    def __lt__(self, other):
        return self.name < other.name

    def set_timestamp(self, timestamp):
        if timestamp is not None and (self.last_timestamp is None or self.last_timestamp != timestamp):
            self.last_timestamp = timestamp

            sig = self.signal
            if sig.samples.size:
                idx = searchsorted(sig.timestamps, timestamp, side="right")
                idx -= 1
                idx = max(idx, 0)

                self.raw = sig.raw_samples[idx]
                self.scaled = sig.phys_samples[idx]

    def get_value(self, index, timestamp=None):
        self.set_timestamp(timestamp)
        if self.signal is not None:
            if index == 0:
                return self.signal.name
            elif index == 1:
                return self.raw
            elif index == 2:
                return self.scaled
            elif index == 3:
                return self.unit
            elif index == 4:
                return self.origin_uuid


class OnlineBackEnd:
    def __init__(self, signals, numeric):
        super().__init__()

        self.signals = signals or []
        self.map = None
        self.numeric = numeric

        self.sorted_column_index = 0
        self.sorting_enabled = True
        self.sort_reversed = False
        self.numeric_viewer = None

        self.update()

    def update_signal_origin_uuid(self, signal, origin_uuid):
        old_entry = signal.entry
        signal.origin_uuid = origin_uuid
        self.map[signal.entry] = signal
        del self.map[old_entry]

        self.numeric_viewer.dataView.ranges[signal.entry] = self.numeric_viewer.dataView.ranges[old_entry]
        del self.numeric_viewer.dataView.ranges[old_entry]

    def update(self, others=()):
        self.map = {signal.entry: signal for signal in self.signals}
        for signal in others:
            if signal.entry not in self.map:
                self.map[signal.entry] = signal
                self.signals.append(signal)

        self.sort()

    def sort_column(self, ix):
        if ix != self.sorted_column_index:
            self.sorted_column_index = ix
            self.sort_reversed = False
        else:
            self.sort_reversed = not self.sort_reversed

        self.sort()

    def color_same_origin_signals(self, origin_uuid="", color=""):
        pass

    def data_changed(self):
        self.refresh_ui()

    def move_rows(self, rows, target_row):
        if target_row == -1:
            sigs = [self.signals.pop(row) for row in rows]
            self.signals.extend(sigs)
        else:
            sig = self.signals[target_row]
            sigs = [self.signals.pop(row) for row in rows]

            idx = self.signals.index(sig)
            for sig in sigs:
                self.signals.insert(idx, sig)

        self.data_changed()

    def refresh_ui(self):
        if self.numeric is not None and self.numeric.mode == "offline":
            numeric = self.numeric
            numeric._min = float("inf")
            numeric._max = -float("inf")

            for sig in self.signals:
                if sig.samples.size:
                    numeric._min = min(self._min, sig.timestamps[0])
                    numeric._max = max(self._max, sig.timestamps[-1])

            if numeric._min == float("inf"):
                numeric._min = numeric._max = 0

            numeric._timestamp = numeric._min

            numeric.timestamp.setRange(numeric._min, numeric._max)
            numeric.min_t.setText(f"{numeric._min:.9f}s")
            numeric.max_t.setText(f"{numeric._max:.9f}s")
            numeric.set_timestamp(numeric._min)

        if self.numeric_viewer is not None:
            self.numeric_viewer.refresh_ui()

    def reorder(self, names):
        try:
            sigs = {sig.name: idx for idx, sig in enumerate(self.signals)}

            if len(sigs) == len(names):
                self.signals = [self.signals[sigs[name]] for name in names]

            self.data_changed()

        except:
            pass

    def sort(self):
        if not self.sorting_enabled:
            self.data_changed()
            return

        sorted_column_index = self.sorted_column_index

        if sorted_column_index == 0:
            self.signals = natsorted(self.signals, key=lambda x: x.name, reverse=self.sort_reversed)

        elif sorted_column_index in (1, 2):
            numeric = []
            string = []
            nones = []

            for signal in self.signals:
                value = signal.get_value(sorted_column_index)
                if value is None:
                    nones.append(signal)
                elif isinstance(value, (np.flexible, bytes)):
                    string.append(signal)
                else:
                    numeric.append(signal)

            self.signals = [
                *sorted(
                    numeric,
                    key=lambda x: x.get_value(sorted_column_index),
                    reverse=self.sort_reversed,
                ),
                *sorted(
                    string,
                    key=lambda x: x.get_value(sorted_column_index),
                    reverse=self.sort_reversed,
                ),
                *natsorted(nones, key=lambda x: x.name, reverse=self.sort_reversed),
            ]

        elif sorted_column_index == 3:
            self.signals = natsorted(self.signals, key=lambda x: x.unit, reverse=self.sort_reversed)

        self.data_changed()

    def set_values(self, values=None):
        map_ = self.map
        if values:
            for entry, vals in values.items():
                sig = map_[entry]
                sig.update_values(vals)

            if self.sorted_column_index in (1, 2):
                self.sort()
            else:
                self.data_changed()

    def shift_same_origin_signals(self, origin_uuid="", delta=0.0, absolute=False):
        pass

    def update_missing_signals(self, uuids=()):
        pass

    def reset(self):
        for sig in self.signals:
            sig.reset()
        self.data_changed()

    def __len__(self):
        return len(self.signals)

    def does_not_exist(self, entry, exists):
        self.map[entry].exists = exists

    def get_signal_value(self, signal, column):
        return signal.get_value(column)

    def set_format(self, fmt, rows):
        for row in rows:
            self.signals[row].format = fmt


class OfflineBackEnd:
    def __init__(self, signals, numeric):
        super().__init__()

        self.timestamp = None

        self.signals = signals or []
        self.map = None
        self.numeric = numeric

        self.sorted_column_index = 0
        self.sorting_enabled = True
        self.sort_reversed = False
        self.numeric_viewer = None

        self.timebase = np.array([])

    def update(self, others=()):
        self.map = {signal.entry: signal for signal in self.signals}
        for signal in others:
            if signal.entry not in self.map:
                self.map[signal.entry] = signal
                self.signals.append(signal)

        if self.signals:
            timestamps = {id(signal.signal.timestamps): signal.signal.timestamps for signal in self.signals}
            timestamps = list(timestamps.values())
            self.timebase = np.unique(np.concatenate(timestamps))
        else:
            self.timebase = np.array([])

        self.sort()

    def sort_column(self, ix):
        if ix != self.sorted_column_index:
            self.sorted_column_index = ix
            self.sort_reversed = False
        else:
            self.sort_reversed = not self.sort_reversed

        self.sort()

    def data_changed(self):
        self.refresh_ui()

    def move_rows(self, rows, target_row):
        if target_row == -1:
            sigs = [self.signals.pop(row) for row in rows]
            self.signals.extend(sigs)
        else:
            sig = self.signals[target_row]
            sigs = [self.signals.pop(row) for row in rows]

            idx = self.signals.index(sig)
            for sig in sigs:
                self.signals.insert(idx, sig)

        self.data_changed()

    def refresh_ui(self):
        if self.numeric_viewer is not None:
            self.numeric_viewer.refresh_ui()

    def reorder(self, names):
        try:
            sigs = {sig.name: idx for idx, sig in enumerate(self.signals)}

            if len(sigs) == len(names):
                self.signals = [self.signals[sigs[name]] for name in names]

            self.data_changed()

        except:
            pass

    def sort(self):
        if not self.sorting_enabled:
            self.data_changed()
            return

        sorted_column_index = self.sorted_column_index

        if sorted_column_index == 0:
            self.signals = natsorted(self.signals, key=lambda x: (x.name, x.origin_uuid), reverse=self.sort_reversed)

        elif sorted_column_index in (1, 2):
            numeric = []
            string = []
            nones = []

            for signal in self.signals:
                value = signal.get_value(sorted_column_index, self.timestamp)
                if value is None:
                    nones.append(signal)
                elif isinstance(value, (np.flexible, bytes)):
                    string.append(signal)
                else:
                    numeric.append(signal)

            self.signals = [
                *sorted(
                    numeric,
                    key=lambda x: x.get_value(sorted_column_index),
                    reverse=self.sort_reversed,
                ),
                *sorted(
                    string,
                    key=lambda x: x.get_value(sorted_column_index),
                    reverse=self.sort_reversed,
                ),
                *natsorted(nones, key=lambda x: x.name, reverse=self.sort_reversed),
            ]

        elif sorted_column_index == 3:
            self.signals = natsorted(self.signals, key=lambda x: x.unit, reverse=self.sort_reversed)

        self.data_changed()

    def color_same_origin_signals(self, origin_uuid="", color=""):
        for signal in self.signals:
            if signal.origin_uuid == origin_uuid:
                signal.color = color

        self.data_changed()

    def get_timestamp(self, stamp):
        max_idx = len(self.timebase) - 1
        if max_idx == -1:
            return stamp

        idx = np.searchsorted(self.timebase, stamp)
        idx = min(idx, max_idx)

        return self.timebase[idx]

    def set_timestamp(self, stamp):
        self.timestamp = stamp
        if self.sorted_column_index in (1, 2):
            self.sort()
        else:
            self.data_changed()

    def shift_same_origin_signals(self, origin_uuid="", delta=0.0, absolute=False):
        for signal in self.signals:
            if signal.origin_uuid == origin_uuid:
                if not absolute:
                    signal.signal.timestamps = signal.signal.timestamps + delta
                else:
                    if len(signal.signal.timestamps):
                        signal.signal.timestamps = signal.signal.timestamps - signal.signal.timestamps[0] + delta

        self.data_changed()

    def update_missing_signals(self, uuids=()):
        for signal in self.signals:
            if signal.origin_uuid not in uuids:
                signal.reset(exists=False)

    def reset(self):
        for sig in self.signals:
            sig.reset()
        self.data_changed()

    def __len__(self):
        return len(self.signals)

    def does_not_exist(self, entry, exists):
        self.map[entry].exists = exists

    def get_signal_value(self, signal, column):
        return signal.get_value(column, self.timestamp)

    def set_format(self, fmt, rows):
        for row in rows:
            self.signals[row].format = fmt


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent, background_color, font_color):
        super().__init__(parent)
        self.numeric_viewer = parent
        self.backend = parent.backend
        self.view = None
        self.format = "Physical"
        self.float_precision = -1
        self.background_color = background_color
        self.font_color = font_color

    def headerData(self, section, orientation, role=None):
        pass

    def columnCount(self, parent=None):
        return 5

    def rowCount(self, parent=None):
        return len(self.backend)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        row = index.row()
        col = index.column()

        signal = self.backend.signals[row]
        cell = self.backend.get_signal_value(signal, col)

        match role:
            case QtCore.Qt.ItemDataRole.DisplayRole:
                if cell is None:
                    return "●"
                elif isinstance(cell, (bytes, np.bytes_)):
                    return cell.decode("utf-8", "replace")
                elif isinstance(cell, str):
                    return cell
                elif isinstance(cell, (np.ndarray, np.record, np.recarray)):
                    return str(cell[0])
                else:
                    if np.isnan(cell):
                        return "NaN"
                    else:
                        return value_as_str(cell, signal.format, None, self.float_precision)

            case QtCore.Qt.ItemDataRole.BackgroundRole:
                channel_ranges = self.view.ranges[signal.entry]
                raw_cell = self.backend.get_signal_value(signal, 1)
                scaled_cell = self.backend.get_signal_value(signal, 2)

                try:
                    scaled_value = float(scaled_cell)
                    value = scaled_value
                except:
                    scaled_value = str(scaled_cell)

                    try:
                        raw_value = float(raw_cell)
                        value = raw_value
                    except:
                        value = scaled_value

                new_background_color, new_font_color = get_colors_using_ranges(
                    value,
                    ranges=channel_ranges,
                    default_background_color=self.background_color,
                    default_font_color=signal.color,
                )

                return new_background_color if new_background_color != self.background_color else None

            case QtCore.Qt.ItemDataRole.ForegroundRole:
                channel_ranges = self.view.ranges[signal.entry]
                raw_cell = self.backend.get_signal_value(signal, 1)
                scaled_cell = self.backend.get_signal_value(signal, 2)

                try:
                    scaled_value = float(scaled_cell)
                    value = scaled_value
                except:
                    scaled_value = str(scaled_cell)

                    try:
                        raw_value = float(raw_cell)
                        value = raw_value
                    except:
                        value = scaled_value

                new_background_color, new_font_color = get_colors_using_ranges(
                    value,
                    ranges=channel_ranges,
                    default_background_color=self.background_color,
                    default_font_color=signal.color,
                )

                return new_font_color

            case QtCore.Qt.ItemDataRole.TextAlignmentRole:
                if col:
                    return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                else:
                    return int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

            case QtCore.Qt.ItemDataRole.DecorationRole:
                if col == 0:
                    if not signal.exists:
                        icon = utils.ERROR_ICON
                        if icon is None:
                            utils.ERROR_ICON = QtGui.QIcon()
                            utils.ERROR_ICON.addPixmap(
                                QtGui.QPixmap(":/error.png"),
                                QtGui.QIcon.Mode.Normal,
                                QtGui.QIcon.State.Off,
                            )

                            utils.NO_ERROR_ICON = QtGui.QIcon()

                            icon = utils.ERROR_ICON
                    else:
                        icon = utils.NO_ERROR_ICON
                        if icon is None:
                            utils.ERROR_ICON = QtGui.QIcon()
                            utils.ERROR_ICON.addPixmap(
                                QtGui.QPixmap(":/error.png"),
                                QtGui.QIcon.Mode.Normal,
                                QtGui.QIcon.State.Off,
                            )

                            utils.NO_ERROR_ICON = QtGui.QIcon()

                            icon = utils.NO_ERROR_ICON

                    return icon

                elif col in (1, 2):
                    has_ranges = bool(self.view.ranges.get(signal.entry, False))
                    if has_ranges:
                        icon = utils.RANGE_INDICATOR_ICON
                        if icon is None:
                            utils.RANGE_INDICATOR_ICON = QtGui.QIcon()
                            utils.RANGE_INDICATOR_ICON.addPixmap(
                                QtGui.QPixmap(":/paint.png"),
                                QtGui.QIcon.Mode.Normal,
                                QtGui.QIcon.State.Off,
                            )

                            utils.NO_ERROR_ICON = QtGui.QIcon()
                            utils.NO_ICON = QtGui.QIcon()

                            icon = utils.RANGE_INDICATOR_ICON
                    else:
                        icon = utils.NO_ERROR_ICON
                        if icon is None:
                            utils.RANGE_INDICATOR_ICON = QtGui.QIcon()
                            utils.RANGE_INDICATOR_ICON.addPixmap(
                                QtGui.QPixmap(":/paint.png"),
                                QtGui.QIcon.Mode.Normal,
                                QtGui.QIcon.State.Off,
                            )

                            utils.NO_ERROR_ICON = QtGui.QIcon()
                            utils.NO_ICON = QtGui.QIcon()

                            icon = utils.NO_ERROR_ICON

                    return icon

            case QtCore.Qt.ItemDataRole.ToolTipRole:
                if signal:
                    return f"Origin = {signal.origin_uuid or 'unknown'}\nMDF = {signal.origin_mdf or 'unknown'}"

    def flags(self, index):
        return (
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            | QtCore.Qt.ItemFlag.ItemIsDropEnabled
        )

    def dropMimeData(self, data, action, row, column, parent):
        def moved_rows(data):
            rows = set()
            ds = QtCore.QDataStream(data.data("application/x-qabstractitemmodeldatalist"))
            while not ds.atEnd():
                row = ds.readInt32()
                ds.readInt32()
                map_items = ds.readInt32()
                for i in range(map_items):
                    ds.readInt32()
                    ds.readQVariant()

                rows.add(row)

            return sorted(rows, reverse=True)

        self.backend.move_rows(moved_rows(data), parent.row())

    def supportedDropActions(self) -> bool:
        return QtCore.Qt.DropAction.MoveAction | QtCore.Qt.DropAction.CopyAction

    def set_format(self, fmt, indexes):
        if fmt not in ("phys", "hex", "bin", "ascii"):
            return

        self.format = fmt

        rows = {index.row() for index in indexes}

        self.backend.set_format(fmt, rows)


class TableView(QtWidgets.QTableView):
    add_channels_request = QtCore.Signal(list)

    def __init__(self, parent):
        super().__init__(parent)
        self.numeric_viewer = parent
        self.backend = parent.backend

        self.ranges = {}

        self._backgrund_color = self.palette().color(QtGui.QPalette.ColorRole.Window)
        self._font_color = self.palette().color(QtGui.QPalette.ColorRole.WindowText)

        model = TableModel(parent, self._backgrund_color, self._font_color)
        self.setModel(model)
        model.view = self

        self.horizontalHeader().hide()
        self.verticalHeader().hide()

        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)

        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)

        self.doubleClicked.connect(self.edit_ranges)

        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)

        self.double_clicked_enabled = True

    def sizeHint(self):
        width = 2 * self.frameWidth()
        for i in range(self.model().columnCount()):
            width += self.columnWidth(i)

        height = 2 * self.frameWidth()
        height += 24 * self.model().rowCount()

        return QtCore.QSize(width, height)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key.Key_Delete and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            event.accept()
            selected_items = {index.row() for index in self.selectedIndexes() if index.isValid()}

            for row in sorted(selected_items, reverse=True):
                signal = self.backend.signals.pop(row)
                del self.backend.map[signal.entry]

            self.backend.update()

        elif key == QtCore.Qt.Key.Key_R and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.accept()

            selected_items = {index.row() for index in self.selectedIndexes() if index.isValid()}

            if selected_items:
                ranges = []

                for row in selected_items:
                    signal = self.backend.signals[row]
                    if self.ranges[signal.entry]:
                        ranges.extend(self.ranges[signal.entry])

                dlg = RangeEditor(
                    "<selected items>",
                    "",
                    ranges=unique_ranges(ranges),
                    parent=self,
                    brush=True,
                )
                dlg.exec_()
                if dlg.pressed_button == "apply":
                    ranges = dlg.result
                    for row in selected_items:
                        signal = self.backend.signals[row]
                        self.ranges[signal.entry] = copy_ranges(ranges)

                    self.backend.update()

        elif (
            modifiers == (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier)
            and key == QtCore.Qt.Key.Key_C
        ):
            event.accept()

            selected_items = {index.row() for index in self.selectedIndexes() if index.isValid()}

            if not selected_items:
                return
            else:
                precision = self.model().float_precision
                row = list(selected_items)[0]
                signal = self.backend.signals[row]

                info = {
                    "format": signal.format,
                    "ranges": self.ranges[signal.entry],
                    "type": "channel",
                    "color": signal.color,
                    "precision": precision,
                    "ylink": False,
                    "individual_axis": False,
                    "y_range": signal.y_range,
                    "origin_uuid": signal.origin_uuid,
                }

                QtWidgets.QApplication.instance().clipboard().setText(json.dumps(info, cls=ExtendedJsonEncoder))

        elif (
            modifiers == (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier)
            and key == QtCore.Qt.Key.Key_V
        ):
            event.accept()

            info = QtWidgets.QApplication.instance().clipboard().text()
            selected_items = {index.row() for index in self.selectedIndexes() if index.isValid()}

            if not selected_items:
                return

            try:
                info = json.loads(info, cls=ExtendedJsonDecoder)
            except:
                print(format_exc())
            else:
                for row in selected_items:
                    signal = self.backend.signals[row]

                    signal.format = info["format"]
                    signal.color = fn.mkColor(info["color"])
                    signal.y_range = info["y_range"]
                    signal.origin_uuid = info["origin_uuid"]
                    self.ranges[signal.entry] = copy_ranges(info["ranges"])

                self.backend.update()

        elif modifiers == QtCore.Qt.KeyboardModifier.NoModifier and key == QtCore.Qt.Key.Key_C:
            event.accept()

            selected_items = [index.row() for index in self.selectedIndexes() if index.isValid()]

            if selected_items:
                signal = self.backend.signals[selected_items[0]]
                color = signal.color

                color = QtWidgets.QColorDialog.getColor(color, parent=self)
                if color.isValid():
                    for row in set(selected_items):
                        signal = self.backend.signals[row]
                        signal.color = color

        elif modifiers == QtCore.Qt.KeyboardModifier.ControlModifier and key == QtCore.Qt.Key.Key_N:
            event.accept()
            selected_items = []

            for index in self.selectedIndexes():
                if not index.isValid():
                    continue

                if (row := index.row()) not in selected_items:
                    selected_items.append(row)

            if not selected_items:
                return
            else:
                text = "\n".join(self.backend.signals[row].name for row in selected_items)

            QtWidgets.QApplication.instance().clipboard().setText(text)

        elif modifiers == QtCore.Qt.KeyboardModifier.ControlModifier and key == QtCore.Qt.Key.Key_C:
            event.accept()

            selected_items = []

            for index in self.selectedIndexes():
                if not index.isValid():
                    continue

                if (row := index.row()) not in selected_items:
                    selected_items.append(row)

            data = []
            numeric_mode = self.backend.numeric.mode

            for row in selected_items:
                signal = self.backend.signals[row]

                entry = signal.entry if numeric_mode == "online" else signal.signal.entry

                group_index, channel_index = entry

                info = {
                    "name": signal.name,
                    "computation": {},
                    "computed": False,
                    "group_index": group_index,
                    "channel_index": channel_index,
                    "ranges": self.ranges[signal.entry],
                    "origin_uuid": str(entry[0]) if numeric_mode == "online" else signal.signal.origin_uuid,
                    "type": "channel",
                    "uuid": os.urandom(6).hex(),
                    "color": signal.color,
                }

                data.append(info)

            data = substitude_mime_uuids(data, None, force=True)
            QtWidgets.QApplication.instance().clipboard().setText(json.dumps(data, cls=ExtendedJsonEncoder))

        elif modifiers == QtCore.Qt.KeyboardModifier.ControlModifier and key == QtCore.Qt.Key.Key_V:
            event.accept()
            try:
                data = QtWidgets.QApplication.instance().clipboard().text()
                data = json.loads(data, cls=ExtendedJsonDecoder)
                data = substitude_mime_uuids(data, random_uuid=True)
                self.add_channels_request.emit(data)
            except:
                pass

        else:
            super().keyPressEvent(event)

    def startDrag(self, supportedActions):
        indexes = self.selectedIndexes()
        if not self.backend.sorting_enabled:
            mime_data = self.model().mimeData(indexes)
        else:
            mime_data = QtCore.QMimeData()

        selected_items = []

        for index in self.selectedIndexes():
            if not index.isValid():
                continue

            if (row := index.row()) not in selected_items:
                selected_items.append(row)

        data = []
        numeric_mode = self.backend.numeric.mode

        for row in selected_items:
            signal = self.backend.signals[row]

            entry = signal.entry if numeric_mode == "online" else signal.signal.entry

            *_, group_index, channel_index = entry

            info = {
                "name": signal.name,
                "computation": {},
                "computed": False,
                "group_index": group_index,
                "channel_index": channel_index,
                "ranges": self.ranges[signal.entry],
                "origin_uuid": str(entry[0]),
                "type": "channel",
                "uuid": os.urandom(6).hex(),
                "color": signal.color,
            }

            data.append(info)

        data = json.dumps(data, cls=ExtendedJsonEncoder).encode("utf-8")

        mime_data.setData("application/octet-stream-asammdf", QtCore.QByteArray(data))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(supportedActions)

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        if e.source() is self:
            if e.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
                e.mimeData().removeFormat("application/octet-stream-asammdf")
                super().dropEvent(e)
            else:
                e.ignore()
            self.clearSelection()
        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
                e.accept()
            else:
                e.ignore()

    def edit_ranges(self, index):
        if not self.double_clicked_enabled or not index.isValid():
            return

        row = index.row()
        signal = self.backend.signals[row]

        dlg = RangeEditor(signal.name, signal.unit, self.ranges[signal.entry], parent=self, brush=True)
        dlg.exec_()
        if dlg.pressed_button == "apply":
            ranges = dlg.result
            self.ranges[signal.entry] = ranges

    def set_format(self, fmt):
        indexes = self.selectedIndexes()
        self.model().set_format(fmt, indexes)


class HeaderModel(QtCore.QAbstractTableModel):
    def __init__(self, parent):
        super().__init__(parent)
        self.backend = parent.backend

    def columnCount(self, parent=None):
        return 5

    def rowCount(self, parent=None):
        return 1  # 1?

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        col = index.column()

        names = ["Name", "Raw", "Scaled", "Unit", "Origin"]

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return names[col]

        elif role == QtCore.Qt.ItemDataRole.DecorationRole:
            if not self.backend.sorting_enabled or col != self.backend.sorted_column_index:
                return
            else:
                if self.backend.sort_reversed:
                    icon = QtGui.QIcon(":/sort-descending.png")
                else:
                    icon = QtGui.QIcon(":/sort-ascending.png")

                return icon

        elif role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter

    def headerData(self, section, orientation, role=None):
        pass


class HeaderView(QtWidgets.QTableView):
    sorting_changed = QtCore.Signal(int)
    NameColumn = 0
    RawColumn = 1
    ScaledColumn = 2
    UnitColumn = 3
    OriginColumn = 4

    def __init__(self, parent):
        super().__init__(parent)
        self.numeric_viewer = parent
        self.backend = parent.backend

        self.table = parent.dataView
        self.setModel(HeaderModel(parent))
        self.padding = 10

        self.header_cell_being_resized = None
        self.header_being_resized = False

        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().installEventFilter(self)

        self.setIconSize(QtCore.QSize(16, 16))
        self.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        )
        self.setWordWrap(False)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        font = QtGui.QFont()
        font.setBold(True)
        self.setFont(font)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.resize(self.sizeHint())

        self.columns_width = {
            self.RawColumn: self.columnWidth(self.RawColumn),
            self.ScaledColumn: self.columnWidth(self.ScaledColumn),
            self.UnitColumn: self.columnWidth(self.UnitColumn),
            self.OriginColumn: self.columnWidth(self.OriginColumn),
        }

    def all_columns_width(self):
        widths = []
        for column in (self.NameColumn, self.RawColumn, self.ScaledColumn, self.UnitColumn, self.OriginColumn):
            if self.isColumnHidden(column):
                widths.append(self.columns_width.get(column, 100))
            else:
                widths.append(self.columnWidth(column))
        return widths

    def columns_visibility(self):
        return {
            "raw": not self.isColumnHidden(self.RawColumn),
            "scaled": not self.isColumnHidden(self.ScaledColumn),
            "unit": not self.isColumnHidden(self.UnitColumn),
            "origin": not self.isColumnHidden(self.OriginColumn),
        }

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        super().showEvent(a0)
        self.initial_size = self.size()

    def sorting(self):
        return {
            "sort_column": self.backend.sorted_column_index,
            "enabled": self.backend.sorting_enabled,
            "reversed": self.backend.sort_reversed,
        }

    def mouseDoubleClickEvent(self, event):
        point = event.pos()
        ix = self.indexAt(point)
        col = ix.column()
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.backend.sorting_enabled:
            self.backend.sort_column(col)
            self.sorting_changed.emit(col)
        else:
            super().mouseDoubleClickEvent(event)

    def eventFilter(self, object: QtCore.QObject, event: QtCore.QEvent):
        if event.type() in [
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QEvent.Type.MouseButtonDblClick,
            QtCore.QEvent.Type.MouseMove,
        ]:
            return self.manage_resizing(object, event)

        return False

    def manage_resizing(self, object: QtCore.QObject, event: QtCore.QEvent):
        def over_header_cell_edge(mouse_position, margin=3):
            x = mouse_position
            if self.columnAt(x - margin) != self.columnAt(x + margin):
                if self.columnAt(x + margin) == 0:
                    return None
                else:
                    return self.columnAt(x - margin)
            else:
                return None

        mouse_position = event.pos().x()
        orthogonal_mouse_position = event.pos().y()

        if over_header_cell_edge(mouse_position) is not None:
            self.viewport().setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.SplitHCursor))

        else:
            self.viewport().setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))

        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if over_header_cell_edge(mouse_position) is not None:
                self.header_cell_being_resized = over_header_cell_edge(mouse_position)
                return True
            else:
                self.header_cell_being_resized = None

        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            self.header_cell_being_resized = None
            self.header_being_resized = False

        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            if over_header_cell_edge(mouse_position) is not None:
                header_index = over_header_cell_edge(mouse_position)
                self.numeric_viewer.auto_size_column(header_index)
                return True

        if event.type() == QtCore.QEvent.Type.MouseMove:
            if self.header_cell_being_resized is not None:
                size = mouse_position - self.columnViewportPosition(self.header_cell_being_resized)
                if size > 10:
                    self.setColumnWidth(self.header_cell_being_resized, size)
                    self.numeric_viewer.dataView.setColumnWidth(self.header_cell_being_resized, size)

                    self.updateGeometry()
                    self.numeric_viewer.dataView.updateGeometry()
                return True

            elif self.header_being_resized:
                size = orthogonal_mouse_position - self.geometry().top()
                self.setFixedHeight(max(size, self.initial_size.height()))

                self.updateGeometry()
                self.numeric_viewer.dataView.updateGeometry()
                return True

        return False

    def sizeHint(self):
        width = self.table.sizeHint().width() + self.verticalHeader().width()
        height = 16 + self.font().pointSize() + 2 * self.frameWidth()

        return QtCore.QSize(width, height)

    def toggle_column(self, checked, column):
        if not checked:
            self.columns_width[column] = self.columnWidth(column)

        self.setColumnHidden(column, not checked)
        self.numeric_viewer.dataView.setColumnHidden(column, not checked)

        if checked:
            self.setColumnWidth(column, self.columns_width[column])
            self.numeric_viewer.dataView.setColumnWidth(column, self.columns_width[column])

        self.updateGeometry()
        self.numeric_viewer.dataView.updateGeometry()

    def toggle_sorting(self, checked):
        self.backend.sorting_enabled = checked
        self.backend.sort()

    def minimumSizeHint(self):
        return QtCore.QSize(50, self.sizeHint().height())


class NumericViewer(QtWidgets.QWidget):
    def __init__(self, backend):
        super().__init__()

        backend.numeric_viewer = self
        self.backend = backend

        self.dataView = TableView(parent=self)

        self.columnHeader = HeaderView(parent=self)

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.setLayout(self.gridLayout)

        self.dataView.horizontalScrollBar().valueChanged.connect(self.columnHeader.horizontalScrollBar().setValue)

        self.columnHeader.horizontalScrollBar().valueChanged.connect(self.dataView.horizontalScrollBar().setValue)

        # self.dataView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.dataView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.gridLayout.addWidget(self.columnHeader, 0, 0)
        self.gridLayout.addWidget(self.dataView, 1, 0)
        # self.gridLayout.addWidget(self.dataView.horizontalScrollBar(), 2, 0, 1, 1)
        # self.gridLayout.addWidget(self.dataView.verticalScrollBar(), 1, 1, 1, 1)

        self.dataView.verticalScrollBar().setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Ignored)
        )
        self.dataView.horizontalScrollBar().setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        )

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setRowStretch(1, 1)

        self.columnHeader.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)

        self.default_row_height = 24
        self.set_styles()

        for column_index in range(self.columnHeader.model().columnCount()):
            self.auto_size_column(column_index)

        self.columnHeader.horizontalHeader().setStretchLastSection(True)

        self.columnHeader.horizontalHeader().sectionResized.connect(self.update_horizontal_scroll)

        self.columnHeader.horizontalHeader().setMinimumSectionSize(1)
        self.dataView.horizontalHeader().setMinimumSectionSize(1)

        self.show()

    def set_styles(self):
        self.dataView.verticalHeader().setDefaultSectionSize(self.default_row_height)
        self.dataView.verticalHeader().setMinimumSectionSize(self.default_row_height)
        self.dataView.verticalHeader().setMaximumSectionSize(self.default_row_height)
        self.dataView.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.columnHeader.verticalHeader().setDefaultSectionSize(self.default_row_height)
        self.columnHeader.verticalHeader().setMinimumSectionSize(self.default_row_height)
        self.columnHeader.verticalHeader().setMaximumSectionSize(self.default_row_height)
        self.columnHeader.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)

    def auto_size_header(self):
        s = 0
        for i in range(self.columnHeader.model().columnCount()):
            s += self.auto_size_column(i)

        delta = int((self.dataView.viewport().size().width() - s) // 4)

        if delta > 0:
            for i in range(self.columnHeader.model().columnCount()):
                self.auto_size_column(i, extra_padding=delta)
            # self.dataView.horizontalScrollBar().hide()
        else:
            self.dataView.horizontalScrollBar().show()

    def update_horizontal_scroll(self, *args):
        return

        s = 0
        for i in range(self.columnHeader.model().columnCount()):
            s += self.dataView.columnWidth(i) + self.dataView.frameWidth()

        if self.dataView.viewport().size().width() < s:
            self.dataView.horizontalScrollBar().show()
        else:
            self.dataView.horizontalScrollBar().hide()

    def auto_size_column(self, column_index, extra_padding=0):
        width = 0

        N = 100
        for i in range(self.dataView.model().rowCount())[:N]:
            mi = self.dataView.model().index(i, column_index)
            text = self.dataView.model().data(mi)
            w = self.dataView.fontMetrics().boundingRect(text.replace("\0", " ")).width()
            width = max(width, w)

        for i in range(self.columnHeader.model().rowCount()):
            mi = self.columnHeader.model().index(i, column_index)
            text = self.columnHeader.model().data(mi)
            w = self.columnHeader.fontMetrics().boundingRect(text.replace("\0", " ")).width()
            width = max(width, w)

        padding = 20
        width += padding + extra_padding

        self.columnHeader.setColumnWidth(column_index, width)
        self.dataView.setColumnWidth(column_index, self.columnHeader.columnWidth(column_index))

        self.dataView.updateGeometry()
        self.columnHeader.updateGeometry()

        return width

    def scroll_to_column(self, column=0):
        index = self.dataView.model().index(0, column)
        self.dataView.scrollTo(index)
        self.columnHeader.selectColumn(column)
        self.columnHeader.on_selectionChanged(force=True)

    def refresh_ui(self):
        self.models = []
        self.models += [
            self.dataView.model(),
            self.columnHeader.model(),
        ]

        for model in self.models:
            model.beginResetModel()
            model.endResetModel()

        for view in [self.columnHeader, self.dataView]:
            view.updateGeometry()


class Numeric(Ui_NumericDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.Signal(list)
    timestamp_changed_signal = QtCore.Signal(object, float)

    def __init__(
        self,
        channels=None,
        format=None,
        mode="offline",
        float_precision=None,
        owner=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.mode = mode
        self.owner = owner

        self.lock = Lock()
        self.visible_entries_modified = True

        self._settings = QtCore.QSettings()

        if mode == "offline":
            backend = OfflineBackEnd(None, self)
        else:
            backend = OnlineBackEnd(None, self)
        self.channels = NumericViewer(backend)
        self.backend = backend

        self.channels.dataView.ranges = {}

        self.main_layout.insertWidget(0, self.channels)
        self.main_layout.setStretch(0, 1)

        self.float_precision.addItems(["Full"] + [f"{i} decimals" for i in range(16)])

        self.float_precision.currentIndexChanged.connect(self.set_float_precision)

        format = format or self._settings.value("numeric_format", "Physical")
        if format not in ("Physical", "Hex", "Binary", "Ascii"):
            format = "Physical"
            self._settings.setValue("numeric_format", format)

        if float_precision is None:
            float_precision = self._settings.value("numeric_float_precision", -1, type=int)
        self.float_precision.setCurrentIndex(float_precision + 1)

        self.timebase = np.array([])
        self._timestamp = None

        if channels:
            self.add_new_channels(channels)

        self.channels.dataView.add_channels_request.connect(self.add_channels_request)
        self.channels.dataView.verticalScrollBar().valueChanged.connect(self.reset_visible_entries)
        self.channels.columnHeader.sorting_changed.connect(self.reset_visible_entries)

        self.channels.auto_size_header()
        self.double_clicked_enabled = True

        self.pattern = {}

        if self.mode == "offline":
            self.timestamp.valueChanged.connect(self._timestamp_changed)
            self.timestamp_slider.valueChanged.connect(self._timestamp_slider_changed)

            self._inhibit = False

            self.forward.clicked.connect(self.search_forward)
            self.backward.clicked.connect(self.search_backward)
            self.op.addItems([">", ">=", "<", "<=", "==", "!="])

            self.time_group.setHidden(True)
            self.search_group.setHidden(True)

            self.toggle_controls_btn.clicked.connect(self.toggle_controls)
        else:
            self.toggle_controls_btn.setHidden(True)
            self.time_group.setHidden(True)
            self.search_group.setHidden(True)

        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_menu)

    def show_menu(self, position):
        count = len(self.channels.backend)

        header = self.channels.columnHeader

        menu = QtWidgets.QMenu()
        menu.addAction(f"{count} rows in the numeric window")
        menu.addSeparator()

        action = QtGui.QAction("Sorting", menu)
        action.setCheckable(True)
        action.setChecked(self.channels.backend.sorting_enabled)
        action.toggled.connect(header.toggle_sorting)
        menu.addAction(action)

        menu.addAction("Automatic set columns width")
        menu.addSeparator()

        action = QtGui.QAction("Raw Column", menu)
        action.setCheckable(True)
        action.setChecked(not header.isColumnHidden(header.RawColumn))
        action.toggled.connect(partial(header.toggle_column, column=header.RawColumn))
        menu.addAction(action)

        action = QtGui.QAction("Scaled Column", menu)
        action.setCheckable(True)
        action.setChecked(not header.isColumnHidden(header.ScaledColumn))
        action.toggled.connect(partial(header.toggle_column, column=header.ScaledColumn))
        menu.addAction(action)

        action = QtGui.QAction("Unit Column", menu)
        action.setCheckable(True)
        action.setChecked(not header.isColumnHidden(header.UnitColumn))
        action.toggled.connect(partial(header.toggle_column, column=header.UnitColumn))
        menu.addAction(action)

        action = QtGui.QAction("Origin Column", menu)
        action.setCheckable(True)
        action.setChecked(not header.isColumnHidden(header.OriginColumn))
        action.toggled.connect(partial(header.toggle_column, column=header.OriginColumn))
        menu.addAction(action)

        action = QtGui.QAction("Hide header and controls", menu)
        action.setCheckable(True)
        action.setChecked(header.isHidden())
        menu.addAction(action)

        menu.addSeparator()

        submenu = QtWidgets.QMenu("Copy names")
        submenu.setIcon(QtGui.QIcon(":/copy.png"))
        action = QtGui.QAction("Copy names", submenu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+N"))
        submenu.addAction(action)
        submenu.addAction("Copy names and values")
        menu.addMenu(submenu)

        submenu = QtWidgets.QMenu("Display structure")
        submenu.setIcon(QtGui.QIcon(":/structure.png"))
        action = QtGui.QAction("Copy display properties", submenu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+C"))
        submenu.addAction(action)
        action = QtGui.QAction("Paste display properties", submenu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+V"))
        submenu.addAction(action)
        action = QtGui.QAction("Copy channel structure", submenu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+C"))
        submenu.addAction(action)
        action = QtGui.QAction("Paste channel structure", submenu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+V"))
        submenu.addAction(action)
        menu.addMenu(submenu)

        menu.addSeparator()

        submenu = QtWidgets.QMenu("Edit")
        submenu.setIcon(QtGui.QIcon(":/edit.png"))

        action = QtGui.QAction("Set color", submenu)
        action.setShortcut(QtGui.QKeySequence("C"))
        submenu.addAction(action)
        submenu.addAction("Set random color")
        action = QtGui.QAction("Set color ranges", submenu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        submenu.addAction(action)
        menu.addMenu(submenu)

        menu.addSeparator()

        submenu = QtWidgets.QMenu("Display mode")
        action = QtGui.QAction("Ascii", submenu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+T"))
        submenu.addAction(action)
        action = QtGui.QAction("Bin", submenu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        submenu.addAction(action)
        action = QtGui.QAction("Hex", submenu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        submenu.addAction(action)
        action = QtGui.QAction("Physical", submenu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+P"))
        submenu.addAction(action)
        menu.addMenu(submenu)

        menu.addSeparator()
        action = QtGui.QAction("Delete", menu)
        action.setShortcut(QtGui.QKeySequence("Delete"))
        menu.addAction(action)

        action = menu.exec_(self.mapToGlobal(position))

        if action is None:
            return

        action_text = action.text()

        if action_text == "Copy names":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_N, QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self.keyPressEvent(event)

        elif action_text == "Copy names and values":
            texts = []
            precision = self.float_precision.currentIndex() - 1

            t = self.timestamp.value()

            if precision == -1:
                t = str(t)
            else:
                template = f"{{:.{precision}f}}"
                t = template.format(t)

            selected_items = []

            for index in self.channels.dataView.selectedIndexes():
                if not index.isValid():
                    continue

                if (row := index.row()) not in selected_items:
                    selected_items.append(row)

            model = self.channels.dataView.model()

            for row in selected_items:
                texts.append(
                    ", ".join(
                        [
                            model.data(model.createIndex(row, HeaderView.NameColumn)),
                            t,
                            f"{model.data(model.createIndex(row, HeaderView.RawColumn))}",
                            f"{model.data(model.createIndex(row, HeaderView.ScaledColumn))}{model.data(model.createIndex(row, HeaderView.UnitColumn))}",
                        ]
                    )
                )

            QtWidgets.QApplication.instance().clipboard().setText("\n".join(texts))

        elif action_text == "Copy channel structure":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_C,
                QtCore.Qt.KeyboardModifier.ControlModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Paste channel structure":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_V,
                QtCore.Qt.KeyboardModifier.ControlModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Copy display properties":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_C,
                QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Paste display properties":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_V,
                QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Copy display properties":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_C,
                QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Paste display properties":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_V,
                QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Automatic set columns width":
            header.numeric_viewer.auto_size_header()
        elif action_text == "Hide header and controls":
            if action.isChecked():
                header.hide()
                self.controls.hide()
            else:
                header.show()
                self.controls.show()

        elif action_text == "Set color":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_C,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Set color ranges":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_R,
                QtCore.Qt.KeyboardModifier.ControlModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Set random color":
            selected_items = {index.row() for index in self.channels.dataView.selectedIndexes() if index.isValid()}

            for row in selected_items:
                while True:
                    rgb = os.urandom(3)
                    if 100 <= sum(rgb) <= 650:
                        break

                self.backend.signals[row].color = fn.mkColor(f"#{rgb.hex()}")

        elif action_text == "Ascii":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_T, QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self.keyPressEvent(event)
        elif action_text == "Bin":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_B, QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self.keyPressEvent(event)
        elif action_text == "Hex":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_H, QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self.keyPressEvent(event)
        elif action_text == "Physical":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_P, QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self.keyPressEvent(event)

        elif action_text == "Delete":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Delete, QtCore.Qt.KeyboardModifier.NoModifier
            )
            self.keyPressEvent(event)

    def add_new_channels(self, channels, mime_data=None):
        if self.mode == "online":
            others = []
            for index, sig in enumerate(channels):
                if sig is not None:
                    entry = (sig.origin_uuid, sig.name)

                    if getattr(sig, "color", None):
                        color = sig.color or serde.COLORS[index % serde.COLORS_COUNT]
                    else:
                        color = serde.COLORS[index % serde.COLORS_COUNT]

                    others.append(
                        SignalOnline(
                            name=sig.name,
                            conversion=sig.conversion,
                            entry=entry,
                            unit=sig.unit,
                            format=getattr(sig, "format", "phys"),
                            color=color,
                        )
                    )

                    sig.ranges = copy_ranges(sig.ranges)

                    self.channels.dataView.ranges[entry] = sig.ranges

        else:
            others = []
            for index, sig in enumerate(channels):
                if sig is not None:
                    sig.flags &= ~sig.Flags.computed
                    sig.computation = None
                    exists = getattr(sig, "exists", True)
                    ranges = sig.ranges
                    sig = PlotSignal(sig, index=index, allow_trim=False, allow_nans=True)
                    if sig.conversion:
                        sig.phys_samples = sig.conversion.convert(sig.raw_samples, as_bytes=True)
                    sig.entry = sig.origin_uuid, sig.group_index, sig.channel_index

                    others.append(
                        SignalOffline(
                            signal=sig,
                            exists=exists,
                        )
                    )

                    self.channels.dataView.ranges[sig.entry] = ranges

        self.channels.backend.update(others)
        self.update_timebase()

    def reset(self):
        self.channels.backend.reset()
        self.channels.dataView.double_clicked_enabled = True

    def set_values(self, values=None):
        selection = self.channels.dataView.selectedIndexes()
        self.channels.backend.set_values(values)

        selection_model = self.channels.dataView.selectionModel()
        for index in selection:
            selection_model.select(index, QtCore.QItemSelectionModel.SelectionFlag.Select)

    def to_config(self):
        channels = []

        pattern = self.pattern
        if not pattern:
            for signal in self.channels.backend.signals:
                channels.append(
                    {
                        "origin_uuid": str(signal.entry[0]),
                        "name": signal.name,
                        "ranges": self.channels.dataView.ranges[signal.entry],
                        "format": signal.format,
                        "color": signal.color,
                    }
                )

        config = {
            "format": "Physical",
            "mode": self.mode,
            "channels": channels,
            "pattern": pattern,
            "float_precision": self.float_precision.currentIndex() - 1,
            "header_sections_width": self.channels.columnHeader.all_columns_width(),
            "font_size": self.font().pointSize(),
            "columns_visibility": self.channels.columnHeader.columns_visibility(),
            "sorting": self.channels.columnHeader.sorting(),
            "header_and_controls_visible": not self.controls.isHidden(),
        }

        return config

    def does_not_exist(self, entry, exists=False):
        self.channels.backend.does_not_exist(entry, exists)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.visible_entries_modified = True

    def reset_visible_entries(self, arg):
        self.visible_entries_modified = True

    def set_format(self, fmt):
        fmt = fmt.lower()
        if fmt in ("phys", "physical"):
            fmt_s = "Physical"
            fmt = "phys"
        elif fmt in ("bin", "binary"):
            fmt_s = "Bin"
            fmt = "bin"
        elif fmt == "hex":
            fmt_s = "Hex"
        elif fmt == "ascii":
            fmt_s = "Ascii"
        else:
            fmt_s = "Physical"
            fmt = "phys"

        self.channels.dataView.set_format(fmt)
        self._settings.setValue("numeric_format", fmt_s)
        self.channels.backend.data_changed()

    def set_float_precision(self, index):
        self._settings.setValue("numeric_float_precision", index - 1)
        self.channels.dataView.model().float_precision = index - 1
        self.channels.backend.data_changed()

    def visible_entries(self):
        visible = set()

        if self.channels.backend.sorted_column_index in (1, 2):
            visible = set(self.channels.backend.map)

        else:
            rect = self.channels.dataView.viewport().rect()

            top = self.channels.dataView.indexAt(rect.topLeft()).row()
            bottom = self.channels.dataView.indexAt(rect.bottomLeft()).row()
            if top == -1:
                pass
            elif bottom == -1:
                visible = set(self.channels.backend.map)
            else:
                for row in range(top, bottom + 1):
                    visible.add(self.channels.backend.signals[row].entry)

        self.visible_entries_modified = True

        return visible

    def _timestamp_changed(self, stamp):
        if not self._inhibit:
            self.set_timestamp(stamp, spinbox=True)

    def _timestamp_slider_changed(self, idx):
        if not self._inhibit:
            if not len(self.timebase):
                return

            self.set_timestamp(self.timebase[idx])

    def set_timestamp(self, stamp=None, emit=True, spinbox=False):
        if stamp is None:
            if self._timestamp is None:
                if len(self.timebase):
                    stamp = self.timebase[0]
                else:
                    return
            else:
                stamp = self._timestamp

        if not len(self.timebase):
            return

        idx = np.searchsorted(self.timebase, stamp, side="right") - 1

        new_stamp = self.timebase[idx]

        if spinbox:
            if new_stamp == self._timestamp and stamp > new_stamp:
                idx += 1
                new_stamp = self.timebase[idx]

        self._timestamp = new_stamp

        self.channels.backend.set_timestamp(new_stamp)

        self._inhibit = True
        self.timestamp_slider.setValue(idx)
        self.timestamp.setValue(new_stamp)
        self._inhibit = False

        if emit:
            self.timestamp_changed_signal.emit(self, new_stamp)

    def search_forward(self):
        if self.op.currentIndex() < 0 or not self.target.text().strip() or not self.pattern_match.text().strip():
            self.match.setText("invalid input values")
            return

        operator = self.op.currentText()

        if self.match_type.currentText() == "Wildcard":
            wildcard = f"{os.urandom(6).hex()}_WILDCARD_{os.urandom(6).hex()}"
            text = self.pattern_match.text().strip()
            pattern = text.replace("*", wildcard)
            pattern = re.escape(pattern)
            pattern = pattern.replace(wildcard, ".*")
        else:
            pattern = self.pattern_match.text().strip()

        if self.case_sensitivity.currentText() == "Case sensitive":
            pattern = re.compile(pattern)
        else:
            pattern = re.compile(f"(?i){pattern}")

        matches = [sig for sig in self.channels.backend.signals if pattern.fullmatch(sig.name)]

        mode = self.match_mode.currentText()

        if not matches:
            self.match.setText("the pattern does not match any channel name")
            return

        try:
            target = float(self.target.text().strip())
        except:
            self.match.setText("the target must a numeric value")
        else:
            if target.is_integer():
                target = int(target)

            start = self.timestamp.value()

            timestamp = None
            signal_name = ""
            for sig in matches:
                sig = sig.signal.cut(start=start)
                if mode == "Raw" or sig.conversion is None:
                    samples = sig.raw_samples
                else:
                    samples = sig.phys_samples

                op = getattr(samples, OPS[operator])
                try:
                    idx = np.argwhere(op(target)).flatten()
                    if len(idx):
                        if len(idx) == 1 or sig.timestamps[idx[0]] != start:
                            timestamp_ = sig.timestamps[idx[0]]
                        else:
                            timestamp_ = sig.timestamps[idx[1]]

                        if timestamp is None or timestamp_ < timestamp:
                            timestamp = timestamp_
                            signal_name = sig.name
                except:
                    continue

            if timestamp is not None:
                self.timestamp.setValue(timestamp)
                self.match.setText(f"condition found for {signal_name}")
            else:
                self.match.setText("condition not found")

    def search_backward(self):
        if self.op.currentIndex() < 0 or not self.target.text().strip() or not self.pattern_match.text().strip():
            self.match.setText("invalid input values")
            return

        operator = self.op.currentText()

        if self.match_type.currentText() == "Wildcard":
            wildcard = f"{os.urandom(6).hex()}_WILDCARD_{os.urandom(6).hex()}"
            text = self.pattern_match.text().strip()
            pattern = text.replace("*", wildcard)
            pattern = re.escape(pattern)
            pattern = pattern.replace(wildcard, ".*")
        else:
            pattern = self.pattern_match.text().strip()

        if self.case_sensitivity.currentText() == "Case sensitive":
            pattern = re.compile(pattern)
        else:
            pattern = re.compile(f"(?i){pattern}")

        matches = [sig for sig in self.channels.backend.signals if pattern.fullmatch(sig.name)]

        mode = self.match_mode.currentText()

        if not matches:
            self.match.setText("the pattern does not match any channel name")
            return

        try:
            target = float(self.target.text().strip())
        except:
            self.match.setText("the target must a numeric value")
        else:
            if target.is_integer():
                target = int(target)

            stop = self.timestamp.value()

            timestamp = None
            signal_name = ""
            for sig in matches:
                sig = sig.signal.cut(stop=stop)
                if mode == "raw values" or sig.conversion is None:
                    samples = sig.raw_samples[:-1]
                else:
                    samples = sig.phys_samples[:-1]

                op = getattr(samples, OPS[operator])
                try:
                    idx = np.argwhere(op(target)).flatten()
                    if len(idx):
                        if len(idx) == 1 or sig.timestamps[idx[-1]] != stop:
                            timestamp_ = sig.timestamps[idx[-1]]
                        else:
                            timestamp_ = sig.timestamps[idx[-2]]

                        if timestamp is None or timestamp_ > timestamp:
                            timestamp = timestamp_
                            signal_name = sig.name
                except:
                    continue

            if timestamp is not None:
                self.timestamp.setValue(timestamp)
                self.match.setText(f"condition found for {signal_name}")
            else:
                self.match.setText("condition not found")

    def color_same_origin_signals(self, origin_uuid="", color=""):
        self.backend.shift_same_origin_signals(origin_uuid=origin_uuid, color=color)

    def shift_same_origin_signals(self, origin_uuid="", delta=0.0):
        self.backend.shift_same_origin_signals(origin_uuid=origin_uuid, delta=delta)

    def update_missing_signals(self, uuids=()):
        self.channels.backend.signals = [
            signal for signal in self.channels.backend.signals if signal.origin_uuid in uuids
        ]

        self.channels.backend.data_changed()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if (
            key in (QtCore.Qt.Key.Key_H, QtCore.Qt.Key.Key_B, QtCore.Qt.Key.Key_P, QtCore.Qt.Key.Key_T)
            and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            event.accept()

            if key == QtCore.Qt.Key.Key_H:
                self.set_format("Hex")
            elif key == QtCore.Qt.Key.Key_B:
                self.set_format("Bin")
            elif key == QtCore.Qt.Key.Key_T:
                self.set_format("Ascii")
            else:
                self.set_format("Physical")

        elif (
            key
            in (
                QtCore.Qt.Key.Key_Left,
                QtCore.Qt.Key.Key_Right,
                QtCore.Qt.Key.Key_PageUp,
                QtCore.Qt.Key.Key_PageDown,
                QtCore.Qt.Key.Key_Home,
                QtCore.Qt.Key.Key_End,
            )
            and modifiers == QtCore.Qt.KeyboardModifier.NoModifier
            and self.mode == "offline"
        ):
            self.timestamp_slider.keyPressEvent(event)

        elif (
            key == QtCore.Qt.Key.Key_S
            and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier
            and self.mode == "offline"
        ):
            event.accept()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save as measurement file",
                "",
                "MDF version 4 files (*.mf4)",
            )

            if file_name:
                signals = [offline_signal.signal for offline_signal in self.channels.dataView.backend.signals]
                if signals:
                    with mdf_module.MDF() as mdf:
                        groups = {}
                        for sig in signals:
                            id_ = id(sig.timestamps)
                            group_ = groups.setdefault(id_, [])
                            group_.append(sig)

                        for signals in groups.values():
                            sigs = []
                            for signal in signals:
                                if ":" in signal.name:
                                    sig = signal.copy()
                                    sig.name = sig.name.split(":")[-1].strip()
                                    sigs.append(sig)
                                else:
                                    sigs.append(signal)
                            mdf.append(sigs, common_timebase=True)
                        mdf.save(file_name, overwrite=True)

        elif key == QtCore.Qt.Key.Key_BracketLeft and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.accept()
            self.decrease_font()

        elif key == QtCore.Qt.Key.Key_BracketRight and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.accept()
            self.increase_font()

        elif (
            key == QtCore.Qt.Key.Key_G
            and modifiers == QtCore.Qt.KeyboardModifier.ShiftModifier
            and self.mode == "offline"
        ):
            event.accept()

            value, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Go to time stamp",
                "Time stamp",
                value=self.timestamp_slider.value(),
                decimals=9,
            )

            if ok:
                self.set_timestamp(value)

        else:
            self.channels.dataView.keyPressEvent(event)

    def close(self):
        self.owner = None
        super().close()

    def decrease_font(self):
        font = self.font()
        size = font.pointSize()
        pos = bisect.bisect_left(FONT_SIZE, size) - 1
        pos = max(pos, 0)
        new_size = FONT_SIZE[pos]

        self.set_font_size(new_size)

    def increase_font(self):
        font = self.font()
        size = font.pointSize()
        pos = bisect.bisect_right(FONT_SIZE, size)
        if pos == len(FONT_SIZE):
            pos -= 1
        new_size = FONT_SIZE[pos]

        self.set_font_size(new_size)

    def set_font_size(self, size):
        self.hide()
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)
        self.show()
        self.channels.default_row_height = 12 + size
        self.channels.set_styles()

    def toggle_controls(self, event=None):
        if self.toggle_controls_btn.text() == "Show controls":
            self.toggle_controls_btn.setText("Hide controls")
            self.time_group.setHidden(False)
            self.search_group.setHidden(False)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/up.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            self.toggle_controls_btn.setIcon(icon)
        else:
            self.toggle_controls_btn.setText("Show controls")
            self.time_group.setHidden(True)
            self.search_group.setHidden(True)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/down.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            self.toggle_controls_btn.setIcon(icon)

    def update_timebase(self):
        if self.mode == "online":
            return

        self.timebase = self.channels.backend.timebase

        count = len(self.timebase)

        if count:
            min_, max_ = self.timebase[0], self.timebase[-1]
            self.timestamp_slider.setRange(0, count - 1)
            if count >= 2:
                self.timestamp.setSingleStep(0.5 * np.min(np.diff(self.timebase)))

        else:
            min_, max_ = 0.0, 0.0
            self.timestamp_slider.setRange(0, 0)

        self.timestamp.setRange(min_, max_)
        self.timestamp.setSingleStep(0.001)

        self.min_t.setText(f"{min_:.9f}s")
        self.max_t.setText(f"{max_:.9f}s")

        self.set_timestamp(emit=False)

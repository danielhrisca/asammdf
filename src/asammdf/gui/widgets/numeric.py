import bisect
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

from ...blocks.utils import extract_mime_names
from ..ui.numeric_offline import Ui_NumericDisplay
from ..utils import FONT_SIZE

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

    def reset(self):
        self.signal = None
        self.exists = True
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
                if idx < 0:
                    idx = 0

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


class OnlineBackEnd:
    def __init__(self, signals, numeric):
        super().__init__()

        self.signals = signals or []
        self.map = None
        self.numeric = numeric

        self.sorted_column_index = 0
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

    def data_changed(self):
        self.refresh_ui()

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

    def sort(self):
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

    def refresh_ui(self):
        if self.numeric_viewer is not None:
            self.numeric_viewer.refresh_ui()

    def sort(self):
        sorted_column_index = self.sorted_column_index

        if sorted_column_index == 0:
            self.signals = natsorted(self.signals, key=lambda x: x.name, reverse=self.sort_reversed)

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

    def get_timestamp(self, stamp):
        max_idx = len(self.timebase) - 1
        if max_idx == -1:
            return stamp

        idx = np.searchsorted(self.timebase, stamp)
        if idx > max_idx:
            idx = max_idx

        return self.timebase[idx]

    def set_timestamp(self, stamp):
        self.timestamp = stamp
        if self.sorted_column_index in (1, 2):
            self.sort()
        else:
            self.data_changed()

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
        return 4

    def rowCount(self, parent=None):
        return len(self.backend)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        row = index.row()
        col = index.column()

        signal = self.backend.signals[row]
        cell = self.backend.get_signal_value(signal, col)

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
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

        elif role == QtCore.Qt.ItemDataRole.BackgroundRole:
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
                default_font_color=self.font_color,
            )

            return new_background_color if new_background_color != self.background_color else None

        elif role == QtCore.Qt.ItemDataRole.ForegroundRole:
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
                default_font_color=self.font_color,
            )

            return new_font_color if new_font_color != self.font_color else None

        elif role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            if col:
                return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            else:
                return int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        elif role == QtCore.Qt.ItemDataRole.DecorationRole:
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

    def flags(self, index):
        return (
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )

    def setData(self, index, value, role=None):
        pass

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

            for row in reversed(list(selected_items)):
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
                row = list(selected_items)[0]
                signal = self.backend.signals[row]

                info = {
                    "format": signal.format,
                    "ranges": copy_ranges(self.ranges[signal.entry]),
                }

                for range_info in info["ranges"]:
                    range_info["background_color"] = range_info["background_color"].color().name()
                    range_info["font_color"] = range_info["font_color"].color().name()

                QtWidgets.QApplication.instance().clipboard().setText(json.dumps(info))

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
                info = json.loads(info)
                for range_info in info["ranges"]:
                    range_info["background_color"] = QtGui.QBrush(QtGui.QColor(range_info["background_color"]))
                    range_info["font_color"] = QtGui.QBrush(QtGui.QColor(range_info["font_color"]))
            except:
                print(format_exc())
            else:
                for row in selected_items:
                    signal = self.backend.signals[row]

                    signal.format = info["format"]
                    self.ranges[signal.entry] = copy_ranges(info["ranges"])

                self.backend.update()

        else:
            super().keyPressEvent(event)

    def startDrag(self, supportedActions):
        selected_items = [index.row() for index in self.selectedIndexes() if index.isValid()]

        mimeData = QtCore.QMimeData()

        data = []
        numeric_mode = self.backend.numeric.mode

        for row in sorted(set(selected_items)):
            signal = self.backend.signals[row]

            entry = signal.entry if numeric_mode == "online" else signal.signal.entry

            group_index, channel_index = entry

            ranges = copy_ranges(self.ranges[signal.entry])

            for range_info in ranges:
                range_info["font_color"] = range_info["font_color"].color().name()
                range_info["background_color"] = range_info["background_color"].color().name()

            info = {
                "name": signal.name,
                "computation": {},
                "computed": True,
                "group_index": group_index,
                "channel_index": channel_index,
                "ranges": ranges,
                "origin_uuid": str(entry[0]) if numeric_mode == "online" else signal.signal.origin_uuid,
                "type": "channel",
                "uuid": os.urandom(6).hex(),
            }

            data.append(info)

        data = json.dumps(data).encode("utf-8")

        mimeData.setData("application/octet-stream-asammdf", QtCore.QByteArray(data))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec(QtCore.Qt.DropAction.CopyAction)

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        if e.source() is self:
            return
        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
            else:
                return

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
        return 4

    def rowCount(self, parent=None):
        return 1  # 1?

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        col = index.column()

        names = ["Name", "Raw", "Scaled", "Unit"]

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return names[col]

        elif role == QtCore.Qt.ItemDataRole.DecorationRole:
            if col != self.backend.sorted_column_index:
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

        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_menu)

        self.resize(self.sizeHint())

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        super().showEvent(a0)
        self.initial_size = self.size()

    def mouseDoubleClickEvent(self, event):
        point = event.pos()
        ix = self.indexAt(point)
        col = ix.column()
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.backend.sort_column(col)
            self.sorting_changed.emit(col)
        else:
            super().mouseDoubleClickEvent(event)

    def show_menu(self, position):
        count = len(self.backend)

        menu = QtWidgets.QMenu()
        menu.addAction(self.tr(f"{count} rows in the numeric window"))
        menu.addSeparator()

        menu.addAction(self.tr("Automatic set columns width"))

        action = menu.exec_(self.viewport().mapToGlobal(position))

        if action is None:
            return

        if action.text() == "Automatic set columns width":
            self.numeric_viewer.auto_size_header()

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
        self.dataView.verticalHeader().sectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.columnHeader.verticalHeader().setDefaultSectionSize(self.default_row_height)
        self.columnHeader.verticalHeader().setMinimumSectionSize(self.default_row_height)
        self.columnHeader.verticalHeader().setMaximumSectionSize(self.default_row_height)
        self.columnHeader.verticalHeader().sectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)

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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.mode = mode

        self.lock = Lock()
        self.visible_entries_modified = True

        self._settings = QtCore.QSettings()

        if mode == "offline":
            backend = OfflineBackEnd(None, self)
        else:
            backend = OnlineBackEnd(None, self)
        self.channels = NumericViewer(backend)

        self.channels.dataView.ranges = {}

        self.main_layout.insertWidget(0, self.channels)
        self.main_layout.setStretch(0, 1)

        self.float_precision.addItems(["Full float precision"] + [f"{i} float decimals" for i in range(16)])

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

        if self.mode == "offline":
            self.pattern = {}

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

    def add_new_channels(self, channels, mime_data=None):
        if self.mode == "online":
            others = []
            for sig in channels:
                if sig is not None:
                    entry = (sig.origin_uuid, sig.name)

                    others.append(
                        SignalOnline(
                            name=sig.name,
                            conversion=sig.conversion,
                            entry=entry,
                            unit=sig.unit,
                            format=getattr(sig, "format", "phys"),
                        )
                    )

                    ranges = copy_ranges(sig.ranges)
                    for range_info in ranges:
                        range_info["font_color"] = fn.mkBrush(range_info["font_color"])
                        range_info["background_color"] = fn.mkBrush(range_info["background_color"])
                    sig.ranges = ranges

                    self.channels.dataView.ranges[entry] = sig.ranges

        else:
            others = []
            for sig in channels:
                if sig is not None:
                    sig.flags &= ~sig.Flags.computed
                    sig.computation = None
                    ranges = sig.ranges
                    exists = getattr(sig, "exists", True)
                    sig = PlotSignal(sig, allow_trim=False, allow_nans=True)
                    if sig.conversion:
                        sig.phys_samples = sig.conversion.convert(sig.raw_samples, as_bytes=True)
                    sig.entry = sig.group_index, sig.channel_index

                    others.append(
                        SignalOffline(
                            signal=sig,
                            exists=exists,
                        )
                    )

                    for range_info in ranges:
                        range_info["font_color"] = fn.mkBrush(range_info["font_color"])
                        range_info["background_color"] = fn.mkBrush(range_info["background_color"])
                    sig.ranges = ranges

                    self.channels.dataView.ranges[sig.entry] = ranges

        self.channels.backend.update(others)
        self.channels.auto_size_header()
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
        for signal in self.channels.backend.signals:
            ranges = self.channels.dataView.ranges[signal.entry]
            ranges = copy_ranges(ranges)

            for range_info in ranges:
                range_info["font_color"] = range_info["font_color"].color().name()
                range_info["background_color"] = range_info["background_color"].color().name()

            channels.append(
                {
                    "origin_uuid": str(signal.entry[0]),
                    "name": signal.name,
                    "ranges": ranges,
                    "format": signal.format,
                }
            )

        if self.mode == "offline":
            pattern = self.pattern
            if pattern:
                ranges = copy_ranges(pattern["ranges"])

                for range_info in ranges:
                    range_info["font_color"] = range_info["font_color"].color().name()
                    range_info["background_color"] = range_info["background_color"].color().name()

                pattern["ranges"] = ranges
        else:
            pattern = {}

        config = {
            "format": "Physical",
            "mode": self.mode,
            "channels": channels,
            "pattern": pattern,
            "float_precision": self.float_precision.currentIndex() - 1,
            "header_sections_width": [
                self.channels.columnHeader.horizontalHeader().sectionSize(i)
                for i in range(self.channels.columnHeader.horizontalHeader().count())
            ],
            "font_size": self.font().pointSize(),
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
            self.set_timestamp(stamp)

    def _timestamp_slider_changed(self, idx):
        if not self._inhibit:
            if not len(self.timebase):
                return

            self.set_timestamp(self.timebase[idx])

    def set_timestamp(self, stamp=None, emit=True):
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

        stamp = self.timebase[idx]
        self._timestamp = stamp

        self.channels.backend.set_timestamp(stamp)

        self._inhibit = True
        self.timestamp_slider.setValue(idx)
        self.timestamp.setValue(stamp)
        self._inhibit = False

        if emit:
            self.timestamp_changed_signal.emit(self, stamp)

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
            key == QtCore.Qt.Key.Key_Right
            and modifiers == QtCore.Qt.KeyboardModifier.NoModifier
            and self.mode == "offline"
        ):
            event.accept()
            self.timestamp_slider.setValue(self.timestamp_slider.value() + 1)

        elif (
            key == QtCore.Qt.Key.Key_Left
            and modifiers == QtCore.Qt.KeyboardModifier.NoModifier
            and self.mode == "offline"
        ):
            event.accept()
            self.timestamp_slider.setValue(self.timestamp_slider.value() - 1)

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
                signals = [signal for signal in self.signals if signal.enable]
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
        super().close()

    def decrease_font(self):
        font = self.font()
        size = font.pointSize()
        pos = bisect.bisect_left(FONT_SIZE, size) - 1
        if pos < 0:
            pos = 0
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

        else:
            min_, max_ = 0.0, 0.0
            self.timestamp_slider.setRange(0, 0)

        self.timestamp.setRange(min_, max_)

        self.min_t.setText(f"{min_:.9f}s")
        self.max_t.setText(f"{max_:.9f}s")

        self.set_timestamp(emit=False)

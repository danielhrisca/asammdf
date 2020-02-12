# -*- coding: utf-8 -*-
import os

bin_ = bin
import logging
from functools import partial, reduce
from time import perf_counter
from struct import unpack
from uuid import uuid4

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

from ..ui import resource_rc as resource_rc

import pyqtgraph as pg

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from ..utils import COLORS
from .cursor import Cursor
from .formated_axis import FormatedAxis
from ..dialogs.define_channel import DefineChannel
from ...mdf import MDF
from ..utils import extract_mime_names
from .list import ListWidget
from .list_item import ListItem
from .channel_display import ChannelDisplay
from .channel_stats import ChannelStats

if not hasattr(pg.InfiniteLine, "addMarker"):
    logger = logging.getLogger("asammdf")
    message = (
        "Old pyqtgraph package: Please install the latest pyqtgraph from the "
        "github develop branch\n"
        "pip install -I --no-deps "
        "https://github.com/pyqtgraph/pyqtgraph/archive/develop.zip"
    )
    logger.warning(message)


class Plot(QtWidgets.QWidget):

    add_channels_request = QtCore.pyqtSignal(list)
    close_request = QtCore.pyqtSignal()
    clicked = QtCore.pyqtSignal()
    cursor_moved_signal = QtCore.pyqtSignal(object, float)
    cursor_removed_signal = QtCore.pyqtSignal(object)

    def __init__(self, signals, with_dots=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContentsMargins(0, 0, 0, 0)

        self.info_uuid = None

        self._range_start = None
        self._range_stop = None

        main_layout = QtWidgets.QVBoxLayout(self)
        # self.setLayout(main_layout)

        vbox = QtWidgets.QVBoxLayout()
        widget = QtWidgets.QWidget()
        self.channel_selection = ListWidget()
        self.channel_selection.setAlternatingRowColors(False)
        self.channel_selection.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel("Cursor/Range information"))
        self.cursor_info = QtWidgets.QLabel("")
        self.cursor_info.setTextFormat(QtCore.Qt.RichText)
        self.cursor_info.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter
        )
        hbox.addWidget(self.cursor_info)
        vbox.addLayout(hbox)
        vbox.addWidget(self.channel_selection)
        widget.setLayout(vbox)

        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(widget)
        self.splitter.setOpaqueResize(False)

        self.plot = _Plot(with_dots=with_dots, parent=self)
        self.plot.range_modified.connect(self.range_modified)
        self.plot.range_removed.connect(self.range_removed)
        self.plot.range_modified_finished.connect(self.range_modified_finished)
        self.plot.cursor_removed.connect(self.cursor_removed)
        self.plot.cursor_moved.connect(self.cursor_moved)
        self.plot.cursor_move_finished.connect(self.cursor_move_finished)
        self.plot.xrange_changed.connect(self.xrange_changed)
        self.plot.computation_channel_inserted.connect(
            self.computation_channel_inserted
        )
        self.plot.curve_clicked.connect(self.channel_selection.setCurrentRow)

        self.splitter.addWidget(self.plot)

        self.info = ChannelStats()
        self.splitter.addWidget(self.info)

        self.channel_selection.itemsDeleted.connect(self.channel_selection_reduced)
        self.channel_selection.itemPressed.connect(self.channel_selection_modified)
        self.channel_selection.currentRowChanged.connect(
            self.channel_selection_row_changed
        )
        self.channel_selection.add_channels_request.connect(self.add_channels_request)
        self.channel_selection.set_time_offset.connect(self.plot.set_time_offset)
        self.plot.add_channels_request.connect(self.add_channels_request)
        self.setAcceptDrops(True)

        main_layout.addWidget(self.splitter)

        if signals:
            self.add_new_channels(signals)

        self.info.hide()

        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setStretchFactor(2, 0)

        self.keyboard_events = (
            set(
                [
                    (QtCore.Qt.Key_M, QtCore.Qt.NoModifier),
                    (QtCore.Qt.Key_B, QtCore.Qt.ControlModifier),
                    (QtCore.Qt.Key_H, QtCore.Qt.ControlModifier),
                    (QtCore.Qt.Key_P, QtCore.Qt.ControlModifier),
                ]
            )
            | self.plot.keyboard_events
        )

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def channel_selection_modified(self, item):
        if item:
            uuid = self.channel_selection.itemWidget(item).uuid
            self.info_uuid = uuid

            sig, _ = self.plot.signal_by_uuid(uuid)
            if sig.enable:

                self.plot.set_current_uuid(self.info_uuid)
                stats = self.plot.get_stats(
                    self.info_uuid, self.channel_selection.itemWidget(item).fmt
                )
                self.info.set_stats(stats)

    def channel_selection_row_changed(self, row):
        if row >= 0:
            item = self.channel_selection.item(row)
            uuid = self.channel_selection.itemWidget(item).uuid
            self.info_uuid = uuid

            sig, _ = self.plot.signal_by_uuid(uuid)
            if sig.enable:

                self.plot.set_current_uuid(self.info_uuid)
                stats = self.plot.get_stats(
                    self.info_uuid, self.channel_selection.itemWidget(item).fmt
                )
                self.info.set_stats(stats)

    def channel_selection_reduced(self, deleted):

        self.plot.delete_channels(deleted)

        if self.info_uuid in deleted:
            self.info_uuid = None
            self.info.hide()

        rows = self.channel_selection.count()

        if not rows:
            self.close_request.emit()

    def cursor_move_finished(self):
        x = self.plot.timebase
        if x is not None and len(x):
            dim = len(x)
            position = self.plot.cursor1.value()

            right = np.searchsorted(x, position, side="right")
            if right == 0:
                next_pos = x[0]
            elif right == dim:
                next_pos = x[-1]
            else:
                if position - x[right - 1] < x[right] - position:
                    next_pos = x[right - 1]
                else:
                    next_pos = x[right]
            self.plot.cursor1.setPos(next_pos)

        # self.plot.cursor_hint.setData(x=[], y=[])
        self.plot.cursor_hint.hide()

    def cursor_moved(self):
        position = self.plot.cursor1.value()

        x = self.plot.timebase

        if x is not None and len(x):
            dim = len(x)
            position = self.plot.cursor1.value()

            right = np.searchsorted(x, position, side="right")
            if right == 0:
                next_pos = x[0]
            elif right == dim:
                next_pos = x[-1]
            else:
                if position - x[right - 1] < x[right] - position:
                    next_pos = x[right - 1]
                else:
                    next_pos = x[right]

            y = []

            _, (hint_min, hint_max) = self.plot.viewbox.viewRange()

            items = [
                self.channel_selection.item(i)
                for i in range(self.channel_selection.count())
            ]

            uuids = [self.channel_selection.itemWidget(item).uuid for item in items]

            for uuid in uuids:
                sig, idx = self.plot.signal_by_uuid(uuid)

                curve = self.plot.curves[idx]
                viewbox = self.plot.view_boxes[idx]

                if curve.isVisible():
                    index = np.argwhere(sig.timestamps == next_pos).flatten()
                    if len(index):
                        _, (y_min, y_max) = viewbox.viewRange()

                        sample = sig.samples[index[0]]
                        sample = (sample - y_min) / (y_max - y_min) * (
                            hint_max - hint_min
                        ) + hint_min

                        y.append(sample)

            # self.plot.viewbox.setYRange(hint_min, hint_max, padding=0)
            # self.plot.cursor_hint.setData(x=[next_pos] * len(y), y=y)
            # if not self.plot.cursor_hint.isVisible():
            #     self.plot.cursor_hint.show()

        if not self.plot.region:
            self.cursor_info.setText(f"t = {position:.6f}s")
            items = [
                self.channel_selection.item(i)
                for i in range(self.channel_selection.count())
            ]

            uuids = [self.channel_selection.itemWidget(item).uuid for item in items]

            for i, uuid in enumerate(uuids):
                signal, idx = self.plot.signal_by_uuid(uuid)
                cut_sig = signal.cut(position, position)
                if signal.plot_texts is None or len(cut_sig) == 0:
                    samples = cut_sig.samples
                    if signal.conversion and hasattr(signal.conversion, "text_0"):
                        samples = signal.conversion.convert(samples)
                        if samples.dtype.kind == "S":
                            try:
                                samples = [s.decode("utf-8") for s in samples]
                            except:
                                samples = [s.decode("latin-1") for s in samples]
                        else:
                            samples = samples.tolist()
                else:
                    t = np.argwhere(
                        signal.plot_timestamps == cut_sig.timestamps
                    ).flatten()
                    try:
                        samples = [e.decode("utf-8") for e in signal.plot_texts[t]]
                    except:
                        samples = [e.decode("latin-1") for e in signal.plot_texts[t]]

                item = self.channel_selection.item(i)
                item = self.channel_selection.itemWidget(item)

                item.set_prefix("= ")
                item.set_fmt(signal.format)

                if len(samples):
                    item.set_value(samples[0])
                else:
                    item.set_value("n.a.")

        if self.info.isVisible():
            fmt = self.widget_by_uuid(self.info_uuid).fmt
            stats = self.plot.get_stats(self.info_uuid, fmt)
            self.info.set_stats(stats)

        self.cursor_moved_signal.emit(self, position)

    def cursor_removed(self):

        for i in range(self.channel_selection.count()):
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            if not self.plot.region:
                self.cursor_info.setText("")
                item.set_prefix("")
                item.set_value("")
        if self.info.isVisible():
            fmt = self.widget_by_uuid(self.info_uuid).fmt
            stats = self.plot.get_stats(self.info_uuid, fmt)
            self.info.set_stats(stats)

        self.cursor_removed_signal.emit(self)

    def range_modified(self):
        start, stop = self.plot.region.getRegion()

        self.cursor_info.setText(
            (
                "< html > < head / > < body >"
                f"< p >t1 = {start:.6f}s< / p > "
                f"< p >t2 = {stop:.6f}s< / p > "
                f"< p >Δt = {stop - start:.6f}s< / p > "
                "< / body > < / html >"
            )
        )

        for i in range(self.channel_selection.count()):

            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            signal, i = self.plot.signal_by_uuid(item.uuid)
            samples = signal.cut(start, stop).samples

            item.set_prefix("Δ = ")
            item.set_fmt(signal.format)

            if len(samples):
                if samples.dtype.kind in "ui":
                    delta = np.int64(np.float64(samples[-1]) - np.float64(samples[0]))
                else:
                    delta = samples[-1] - samples[0]

                item.set_value(delta)

            else:
                item.set_value("n.a.")

        if self.info.isVisible():
            fmt = self.widget_by_uuid(self.info_uuid).fmt
            stats = self.plot.get_stats(self.info_uuid, fmt)
            self.info.set_stats(stats)

    def xrange_changed(self):
        if self.info.isVisible():
            fmt = self.widget_by_uuid(self.info_uuid).fmt
            stats = self.plot.get_stats(self.info_uuid, fmt)
            self.info.set_stats(stats)

    def range_modified_finished(self):
        start, stop = self.plot.region.getRegion()

        if self.plot.timebase is not None and len(self.plot.timebase):
            timebase = self.plot.timebase
            dim = len(timebase)

            right = np.searchsorted(timebase, start, side="right")
            if right == 0:
                next_pos = timebase[0]
            elif right == dim:
                next_pos = timebase[-1]
            else:
                if start - timebase[right - 1] < timebase[right] - start:
                    next_pos = timebase[right - 1]
                else:
                    next_pos = timebase[right]
            start = next_pos

            right = np.searchsorted(timebase, stop, side="right")
            if right == 0:
                next_pos = timebase[0]
            elif right == dim:
                next_pos = timebase[-1]
            else:
                if stop - timebase[right - 1] < timebase[right] - stop:
                    next_pos = timebase[right - 1]
                else:
                    next_pos = timebase[right]
            stop = next_pos

            self.plot.region.setRegion((start, stop))

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        if key == QtCore.Qt.Key_M and modifiers == QtCore.Qt.NoModifier:

            ch_size, plt_size, info_size = self.splitter.sizes()

            if self.info.isVisible():
                self.info.hide()
                self.splitter.setSizes((ch_size, plt_size + info_size, 0,))

            else:

                self.info.show()
                self.splitter.setSizes(
                    (
                        ch_size,
                        int(0.8 * (plt_size + info_size)),
                        int(0.2 * (plt_size + info_size)),
                    )
                )

        elif key == QtCore.Qt.Key_B and modifiers == QtCore.Qt.ControlModifier:
            selected_items = self.channel_selection.selectedItems()
            if not selected_items:
                signals = self.plot.signals

            else:
                uuids = [
                    self.channel_selection.itemWidget(item).uuid
                    for item in selected_items
                ]

                signals = [self.plot.signal_by_uuid(uuid)[0] for uuid in uuids]

            for signal in signals:
                if signal.samples.dtype.kind in "ui":
                    signal.format = "bin"
                    if self.plot.current_uuid == signal.uuid:
                        self.plot.axis.format = "bin"
                        self.plot.axis.hide()
                        self.plot.axis.show()
            if self.plot.cursor1:
                self.plot.cursor_moved.emit()

        elif key == QtCore.Qt.Key_H and modifiers == QtCore.Qt.ControlModifier:
            selected_items = self.channel_selection.selectedItems()
            if not selected_items:
                signals = self.plot.signals

            else:
                uuids = [
                    self.channel_selection.itemWidget(item).uuid
                    for item in selected_items
                ]

                signals = [self.plot.signal_by_uuid(uuid)[0] for uuid in uuids]

            for signal in signals:
                if signal.samples.dtype.kind in "ui":
                    signal.format = "hex"
                    if self.plot.current_uuid == signal.uuid:
                        self.plot.axis.format = "hex"
                        self.plot.axis.hide()
                        self.plot.axis.show()
            if self.plot.cursor1:
                self.plot.cursor_moved.emit()

        elif key == QtCore.Qt.Key_P and modifiers == QtCore.Qt.ControlModifier:
            selected_items = self.channel_selection.selectedItems()
            if not selected_items:
                signals = self.plot.signals

            else:
                uuids = [
                    self.channel_selection.itemWidget(item).uuid
                    for item in selected_items
                ]

                signals = [self.plot.signal_by_uuid(uuid)[0] for uuid in uuids]

            for signal in signals:
                if signal.samples.dtype.kind in "ui":
                    signal.format = "phys"
                if self.plot.current_uuid == signal.uuid:
                    self.plot.axis.format = "phys"
                    self.plot.axis.hide()
                    self.plot.axis.show()
            if self.plot.cursor1:
                self.plot.cursor_moved.emit()

        elif (key, modifiers) in self.plot.keyboard_events:
            self.plot.keyPressEvent(event)
        else:
            self.parent().keyPressEvent(event)

    def range_removed(self):
        for i in range(self.channel_selection.count()):
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            item.set_prefix("")
            item.set_value("")
            self.cursor_info.setText("")

        self._range_start = None
        self._range_stop = None

        if self.plot.cursor1:
            self.plot.cursor_moved.emit()
        if self.info.isVisible():
            fmt = self.widget_by_uuid(self.info_uuid).fmt
            stats = self.plot.get_stats(self.info_uuid, fmt)
            self.info.set_stats(stats)

    def computation_channel_inserted(self):
        sig = self.plot.signals[-1]

        if sig.empty:
            name, unit = sig.name, sig.unit
        else:
            name, unit = sig.name, sig.unit
        item = ListItem((-1, -1), name, sig.computation, self.channel_selection)
        tooltip = getattr(sig, "tooltip", "")
        it = ChannelDisplay(sig.uuid, unit, sig.samples.dtype.kind, 3, tooltip, self)
        it.setAttribute(QtCore.Qt.WA_StyledBackground)

        it.set_name(name)
        it.set_value("")
        it.set_color(sig.color)
        item.setSizeHint(it.sizeHint())
        self.channel_selection.addItem(item)
        self.channel_selection.setItemWidget(item, it)

        it.color_changed.connect(self.plot.set_color)
        it.enable_changed.connect(self.plot.set_signal_enable)
        it.ylink_changed.connect(self.plot.set_common_axis)

        it.enable_changed.emit(sig.uuid, 1)
        it.enable_changed.emit(sig.uuid, 0)
        it.enable_changed.emit(sig.uuid, 1)

        self.info_uuid = sig.uuid

        self.plot.set_current_uuid(self.info_uuid, True)

    def add_new_channels(self, channels):

        for sig in channels:
            sig.uuid = uuid4()

        invalid = []

        for channel in channels:
            if np.any(np.diff(channel.timestamps) < 0):
                invalid.append(channel.name)

        if invalid:
            QtWidgets.QMessageBox.warning(
                self,
                "The following channels do not have monotonous increasing time stamps:",
                f"The following channels do not have monotonous increasing time stamps:\n{', '.join(invalid)}",
            )
            self.plot._can_trim = False

        self.plot.add_new_channels(channels)

        for sig in channels:

            item = ListItem(
                (sig.group_index, sig.channel_index),
                sig.name,
                sig.computation,
                self.channel_selection,
            )
            item.setData(QtCore.Qt.UserRole, sig.name)
            tooltip = getattr(sig, "tooltip", "")
            it = ChannelDisplay(
                sig.uuid, sig.unit, sig.samples.dtype.kind, 3, tooltip, self
            )
            it.setAttribute(QtCore.Qt.WA_StyledBackground)

            it.set_name(sig.name)
            it.set_value("")
            it.set_color(sig.color)
            item.setSizeHint(it.sizeHint())
            self.channel_selection.addItem(item)
            self.channel_selection.setItemWidget(item, it)

            it.color_changed.connect(self.plot.set_color)
            it.enable_changed.connect(self.plot.set_signal_enable)
            it.ylink_changed.connect(self.plot.set_common_axis)
            it.individual_axis_changed.connect(self.plot.set_individual_axis)

            it.enable_changed.emit(sig.uuid, 1)
            it.enable_changed.emit(sig.uuid, 0)
            it.enable_changed.emit(sig.uuid, 1)

            self.info_uuid = sig.uuid

    def to_config(self):
        count = self.channel_selection.count()

        channels = []
        for i in range(count):
            channel = {}
            item = self.channel_selection.itemWidget(self.channel_selection.item(i))
            name = item._name

            sig = [s for s in self.plot.signals if s.name == name][0]

            channel["name"] = name
            channel["unit"] = sig.unit
            channel["enabled"] = item.display.checkState() == QtCore.Qt.Checked
            channel["individual_axis"] = (
                item.individual_axis.checkState() == QtCore.Qt.Checked
            )
            channel["common_axis"] = item.ylink.checkState() == QtCore.Qt.Checked
            channel["color"] = sig.color
            channel["computed"] = sig.computed
            ranges = [
                {"start": start, "stop": stop, "color": color,}
                for (start, stop), color in item.ranges.items()
            ]
            channel["ranges"] = ranges

            channel["precision"] = item.precision
            channel["fmt"] = item.fmt
            if sig.computed:
                channel["computation"] = sig.computation

            channels.append(channel)

        config = {
            "channels": channels,
        }

        return config

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat("application/octet-stream-asammdf"):
            e.accept()
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        if e.source() is self.channel_selection:
            super().dropEvent(e)
        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
            else:
                super().dropEvent(e)

    def widget_by_uuid(self, uuid):
        for i in range(self.channel_selection.count()):
            item = self.channel_selection.item(i)
            widget = self.channel_selection.itemWidget(item)
            if widget.uuid == uuid:
                break
        else:
            widget = None
        return widget


class _Plot(pg.PlotWidget):
    cursor_moved = QtCore.pyqtSignal()
    cursor_removed = QtCore.pyqtSignal()
    range_removed = QtCore.pyqtSignal()
    range_modified = QtCore.pyqtSignal()
    range_modified_finished = QtCore.pyqtSignal()
    cursor_move_finished = QtCore.pyqtSignal()
    xrange_changed = QtCore.pyqtSignal()
    computation_channel_inserted = QtCore.pyqtSignal()
    curve_clicked = QtCore.pyqtSignal(int)

    add_channels_request = QtCore.pyqtSignal(list)

    def __init__(self, signals=None, with_dots=False, *args, **kwargs):
        super().__init__()

        self._last_update = perf_counter()
        self._can_trim = True

        self.setAcceptDrops(True)

        self._last_size = self.geometry()
        self._settings = QtCore.QSettings()

        self.setContentsMargins(5, 5, 5, 5)
        self.xrange_changed.connect(self.xrange_changed_handle)
        self.with_dots = with_dots
        if self.with_dots:
            self.curvetype = pg.PlotDataItem
        else:
            self.curvetype = pg.PlotCurveItem
        self.info = None
        self.current_uuid = 0

        self.standalone = kwargs.get("standalone", False)

        self.region = None
        self.region_lock = None
        self.cursor1 = None
        self.cursor2 = None
        self.signals = signals or []

        self.axes = []
        self._axes_layout_pos = 2

        self.disabled_keys = set()

        self._timebase_db = {}
        for sig in self.signals:
            if id(sig.timestamps) not in self._timebase_db:
                self._timebase_db[id(sig.timestamps)] = set()
            self._timebase_db[id(sig.timestamps)].add(sig.uuid)

        self._compute_all_timebase()

        self.showGrid(x=True, y=True)

        self.plot_item = self.plotItem
        self.plot_item.hideAxis("left")
        self.layout = self.plot_item.layout
        self.scene_ = self.plot_item.scene()
        self.scene_.sigMouseClicked.connect(self._clicked)
        self.viewbox = self.plot_item.vb
        self.viewbox.sigXRangeChanged.connect(self.xrange_changed.emit)

        self.common_axis_items = set()
        self.common_axis_label = ""
        self.common_viewbox = pg.ViewBox(enableMenu=True)
        self.scene_.addItem(self.common_viewbox)
        self.common_viewbox.setXLink(self.viewbox)

        axis = self.layout.itemAt(2, 0)
        axis.setParent(None)
        self.axis = FormatedAxis("left")
        self.layout.removeItem(axis)
        self.layout.addItem(self.axis, 2, 0)
        self.axis.linkToView(axis.linkedView())
        self.plot_item.axes["left"]["item"] = self.axis

        self.cursor_hint = pg.PlotDataItem(
            [],
            [],
            pen="#000000",
            symbolBrush="#000000",
            symbolPen="w",
            symbol="s",
            symbolSize=8,
        )
        self.viewbox.addItem(self.cursor_hint)

        self.view_boxes = []
        self.curves = []

        self.viewbox.sigResized.connect(self.update_views)
        self._prev_geometry = self.viewbox.sceneBoundingRect()

        self.resizeEvent = self._resizeEvent

        if signals:
            self.add_new_channels(signals)

        self.keyboard_events = set(
            [
                (QtCore.Qt.Key_C, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_F, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_G, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_I, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_O, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_R, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_S, QtCore.Qt.ControlModifier),
                (QtCore.Qt.Key_S, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_Y, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_Left, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_Right, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_H, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_Insert, QtCore.Qt.NoModifier),
            ]
        )

    def update_lines(self, with_dots=None, force=False):
        with_dots_changed = False

        if with_dots is not None and with_dots != self.with_dots:
            self.with_dots = with_dots
            self.curvetype = pg.PlotDataItem if with_dots else pg.PlotCurveItem
            with_dots_changed = True

        if with_dots_changed or force:
            for sig in self.signals:
                _, i = self.signal_by_uuid(sig.uuid)
                color = sig.color
                t = sig.plot_timestamps

                if not force:
                    try:
                        curve = self.curvetype(
                            t,
                            sig.plot_samples,
                            pen=color,
                            symbolBrush=color,
                            symbolPen=color,
                            symbol="o",
                            symbolSize=4,
                            clickable=True,
                            mouseWidth=30,
                        )

                        curve.sigClicked.connect(partial(self.curve_clicked.emit, i))
                    except:
                        message = (
                            "Can't show dots due to old pyqtgraph package: "
                            "Please install the latest pyqtgraph from the "
                            "github develop branch\n"
                            "pip install -I --no-deps "
                            "https://github.com/pyqtgraph/pyqtgraph/archive/develop.zip"
                        )
                        logger.warning(message)
                    self.view_boxes[i].removeItem(self.curves[i])

                    self.curves[i] = curve

                    self.view_boxes[i].addItem(curve)
                else:
                    curve = self.curves[i]

                    if len(t):

                        if self.with_dots:
                            curve.setData(x=t, y=sig.plot_samples)
                        else:
                            curve.invalidateBounds()
                            curve._boundsCache = [
                                [(1, None), (t[0], t[-1])],
                                [(1, None), (sig.min, sig.max)],
                            ]

                            curve.xData = t
                            curve.yData = sig.plot_samples
                            curve.path = None
                            curve.fillPath = None
                            curve._mouseShape = None
                            curve.prepareGeometryChange()
                            curve.informViewBoundsChanged()
                            curve.update()
                            curve.sigPlotChanged.emit(curve)

                if sig.enable:
                    curve.show()
                else:
                    curve.hide()

    def set_color(self, uuid, color):
        _, index = self.signal_by_uuid(uuid)
        self.signals[index].color = color
        self.curves[index].setPen(color)
        if self.curvetype == pg.PlotDataItem:
            self.curves[index].setSymbolPen(color)
            self.curves[index].setSymbolBrush(color)

        if uuid == self.current_uuid:
            self.axis.setPen(color)

    def set_common_axis(self, uuid, state):
        _, index = self.signal_by_uuid(uuid)
        viewbox = self.view_boxes[index]
        if state in (QtCore.Qt.Checked, True, 1):
            viewbox.setYRange(*self.common_viewbox.viewRange()[1], padding=0)
            viewbox.setYLink(self.common_viewbox)
            self.common_axis_items.add(uuid)
        else:
            self.view_boxes[index].setYLink(None)
            self.common_axis_items.remove(uuid)

        self.common_axis_label = ", ".join(
            self.signal_by_uuid(uuid)[0].name for uuid in self.common_axis_items
        )

        self.set_current_uuid(self.current_uuid, True)

    def set_individual_axis(self, uuid, state):

        _, index = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.Checked, True, 1):
            if self.signals[index].enable:
                self.axes[index].show()
            self.signals[index].individual_axis = True
        else:
            self.axes[index].hide()
            self.signals[index].individual_axis = False

    def set_signal_enable(self, uuid, state):

        sig, index = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.Checked, True, 1):
            self.signals[index].enable = True
            self.curves[index].show()
            self.view_boxes[index].setXLink(self.viewbox)
            if self.signals[index].individual_axis:
                self.axes[index].show()

            if id(sig.timestamps) not in self._timebase_db:
                self._timebase_db[id(sig.timestamps)] = {uuid}
                self._compute_all_timebase()
            else:
                self._timebase_db[id(sig.timestamps)].add(uuid)
        else:
            self.signals[index].enable = False
            self.curves[index].hide()
            self.view_boxes[index].setXLink(None)
            self.axes[index].hide()

            self._timebase_db[id(sig.timestamps)].remove(uuid)

            if len(self._timebase_db[id(sig.timestamps)]) == 0:
                del self._timebase_db[id(sig.timestamps)]
                self._compute_all_timebase()

        if self.cursor1:
            self.cursor_move_finished.emit()

    def update_views(self):
        geometry = self.viewbox.sceneBoundingRect()
        if geometry != self._prev_geometry:
            for view_box in self.view_boxes:
                view_box.setGeometry(geometry)
            self._prev_geometry = geometry

    def get_stats(self, uuid, fmt):
        stats = {}
        sig, index = self.signal_by_uuid(uuid)
        x = sig.timestamps
        size = len(x)

        if size:

            if sig.plot_texts is not None:
                stats["overall_min"] = ""
                stats["overall_max"] = ""
                stats["overall_average"] = ""
                stats["overall_rms"] = ""
                stats["overall_start"] = fmt.format(sig.timestamps[0])
                stats["overall_stop"] = fmt.format(sig.timestamps[-1])
                stats["unit"] = ""
                stats["color"] = sig.color
                stats["name"] = sig.name

                if self.cursor1:
                    position = self.cursor1.value()
                    stats["cursor_t"] = position

                    if x[0] <= position <= x[-1]:
                        idx = np.searchsorted(x, position)
                        text = sig.texts[idx]
                        try:
                            text = text.decode("utf-8")
                        except:
                            text = text.decode("latin-1")
                        stats["cursor_value"] = text

                    else:
                        stats["cursor_value"] = "n.a."

                else:
                    stats["cursor_t"] = ""
                    stats["cursor_value"] = ""

                stats["selected_start"] = ""
                stats["selected_stop"] = ""
                stats["selected_delta_t"] = ""
                stats["selected_min"] = ""
                stats["selected_max"] = ""
                stats["selected_average"] = ""
                stats["selected_rms"] = ""
                stats["selected_delta"] = ""
                stats["visible_min"] = ""
                stats["visible_max"] = ""
                stats["visible_average"] = ""
                stats["visible_rms"] = ""
                stats["visible_delta"] = ""
            else:
                stats["overall_min"] = fmt.format(sig.min)
                stats["overall_max"] = fmt.format(sig.max)
                stats["overall_average"] = sig.avg
                stats["overall_rms"] = sig.rms
                stats["overall_start"] = sig.timestamps[0]
                stats["overall_stop"] = sig.timestamps[-1]
                stats["unit"] = sig.unit
                stats["color"] = sig.color
                stats["name"] = sig.name

                if self.cursor1:
                    position = self.cursor1.value()
                    stats["cursor_t"] = position

                    if x[0] <= position <= x[-1]:
                        idx = np.searchsorted(x, position)
                        val = sig.samples[idx]
                        if sig.conversion and hasattr(sig.conversion, "text_0"):
                            vals = np.array([val])
                            vals = sig.conversion.convert(vals)
                            if vals.dtype.kind == "S":
                                try:
                                    vals = [s.decode("utf-8") for s in vals]
                                except UnicodeDecodeError:
                                    vals = [s.decode("latin-1") for s in vals]
                                val = f"{val:.6f}= {vals[0]}"
                            else:
                                val = f"{val:.6f}= {vals[0]:.6f}"

                        stats["cursor_value"] = fmt.format(val)

                    else:
                        stats["cursor_value"] = "n.a."

                else:
                    stats["cursor_t"] = ""
                    stats["cursor_value"] = ""

                if self.region:
                    start, stop = self.region.getRegion()

                    if sig._stats["range"] != (start, stop) or sig._stats["fmt"] != fmt:
                        new_stats = {}
                        new_stats["selected_start"] = start
                        new_stats["selected_stop"] = stop
                        new_stats["selected_delta_t"] = stop - start

                        cut = sig.cut(start, stop)

                        if len(cut):
                            new_stats["selected_min"] = fmt.format(
                                np.nanmin(cut.samples)
                            )
                            new_stats["selected_max"] = fmt.format(
                                np.nanmax(cut.samples)
                            )
                            new_stats["selected_average"] = np.mean(cut.samples)
                            new_stats["selected_rms"] = np.sqrt(
                                np.mean(np.square(cut.samples))
                            )
                            if cut.samples.dtype.kind in "ui":
                                new_stats["selected_delta"] = fmt.format(
                                    int(float(cut.samples[-1]) - (cut.samples[0]))
                                )
                            else:
                                new_stats["selected_delta"] = fmt.format(
                                    (cut.samples[-1] - cut.samples[0])
                                )

                        else:
                            new_stats["selected_min"] = "n.a."
                            new_stats["selected_max"] = "n.a."
                            new_stats["selected_average"] = "n.a."
                            new_stats["selected_rms"] = "n.a."
                            new_stats["selected_delta"] = "n.a."

                        sig._stats["range"] = (start, stop)
                        sig._stats["range_stats"] = new_stats

                    stats.update(sig._stats["range_stats"])

                else:
                    stats["selected_start"] = ""
                    stats["selected_stop"] = ""
                    stats["selected_delta_t"] = ""
                    stats["selected_min"] = ""
                    stats["selected_max"] = ""
                    stats["selected_average"] = ""
                    stats["selected_rms"] = ""
                    stats["selected_delta"] = ""

                (start, stop), _ = self.viewbox.viewRange()

                if sig._stats["visible"] != (start, stop) or sig._stats["fmt"] != fmt:
                    new_stats = {}
                    new_stats["visible_start"] = start
                    new_stats["visible_stop"] = stop
                    new_stats["visible_delta_t"] = stop - start

                    cut = sig.cut(start, stop)

                    if len(cut):
                        new_stats["visible_min"] = fmt.format(np.nanmin(cut.samples))
                        new_stats["visible_max"] = fmt.format(np.nanmax(cut.samples))
                        new_stats["visible_average"] = np.mean(cut.samples)
                        new_stats["visible_rms"] = np.sqrt(
                            np.mean(np.square(cut.samples))
                        )
                        if cut.samples.dtype.kind in "ui":
                            new_stats["visible_delta"] = int(cut.samples[-1]) - int(
                                cut.samples[0]
                            )
                        else:
                            new_stats["visible_delta"] = fmt.format(
                                cut.samples[-1] - cut.samples[0]
                            )

                    else:
                        new_stats["visible_min"] = "n.a."
                        new_stats["visible_max"] = "n.a."
                        new_stats["visible_average"] = "n.a."
                        new_stats["visible_rms"] = "n.a."
                        new_stats["visible_delta"] = "n.a."

                    sig._stats["visible"] = (start, stop)
                    sig._stats["visible_stats"] = new_stats

                stats.update(sig._stats["visible_stats"])

        else:
            stats["overall_min"] = "n.a."
            stats["overall_max"] = "n.a."
            stats["overall_average"] = "n.a."
            stats["overall_rms"] = "n.a."
            stats["overall_start"] = "n.a."
            stats["overall_stop"] = "n.a."
            stats["unit"] = sig.unit
            stats["color"] = sig.color
            stats["name"] = sig.name

            if self.cursor1:
                position = self.cursor1.value()
                stats["cursor_t"] = position

                stats["cursor_value"] = "n.a."

            else:
                stats["cursor_t"] = ""
                stats["cursor_value"] = ""

            if self.region:
                start, stop = self.region.getRegion()

                stats["selected_start"] = start
                stats["selected_stop"] = stop
                stats["selected_delta_t"] = stop - start

                stats["selected_min"] = "n.a."
                stats["selected_max"] = "n.a."
                stats["selected_average"] = "n.a."
                stats["selected_rms"] = "n.a."
                stats["selected_delta"] = "n.a."

            else:
                stats["selected_start"] = ""
                stats["selected_stop"] = ""
                stats["selected_delta_t"] = ""
                stats["selected_min"] = ""
                stats["selected_max"] = ""
                stats["selected_average"] = "n.a."
                stats["selected_rms"] = "n.a."
                stats["selected_delta"] = ""

            (start, stop), _ = self.viewbox.viewRange()

            stats["visible_start"] = start
            stats["visible_stop"] = stop
            stats["visible_delta_t"] = stop - start

            stats["visible_min"] = "n.a."
            stats["visible_max"] = "n.a."
            stats["visible_average"] = "n.a."
            stats["visible_rms"] = "n.a."
            stats["visible_delta"] = "n.a."

        sig._stats["fmt"] = fmt
        return stats

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if key in self.disabled_keys:
            super().keyPressEvent(event)
        else:

            if key == QtCore.Qt.Key_C and modifier == QtCore.Qt.NoModifier:
                if self.cursor1 is None:
                    start, stop = self.viewbox.viewRange()[0]
                    self.cursor1 = Cursor(pos=0, angle=90, movable=True)
                    self.plotItem.addItem(self.cursor1, ignoreBounds=True)
                    self.cursor1.sigPositionChanged.connect(self.cursor_moved.emit)
                    self.cursor1.sigPositionChangeFinished.connect(
                        self.cursor_move_finished.emit
                    )
                    self.cursor1.setPos((start + stop) / 2)
                    self.cursor_move_finished.emit()

                    if self.region is not None:
                        self.cursor1.hide()

                else:
                    self.plotItem.removeItem(self.cursor1)
                    self.cursor1.setParent(None)
                    self.cursor1 = None
                    self.cursor_removed.emit()

            elif key == QtCore.Qt.Key_Y and modifier == QtCore.Qt.NoModifier:
                if self.region is not None:
                    if self.region_lock is not None:
                        self.region_lock = None
                    else:
                        self.region_lock = self.region.getRegion()[0]
                else:
                    self.region_lock = None

            elif key == QtCore.Qt.Key_F and modifier == QtCore.Qt.NoModifier:
                if self.common_axis_items:
                    if any(
                        len(self.signal_by_uuid(uuid)[0].plot_samples)
                        for uuid in self.common_axis_items
                        if self.signal_by_uuid(uuid)[0].enable
                    ):
                        with_common_axis = True

                        common_min = np.nanmin(
                            [
                                np.nanmin(self.signal_by_uuid(uuid)[0].plot_samples)
                                for uuid in self.common_axis_items
                                if len(self.signal_by_uuid(uuid)[0].plot_samples)
                            ]
                        )
                        common_max = np.nanmax(
                            [
                                np.nanmax(self.signal_by_uuid(uuid)[0].plot_samples)
                                for uuid in self.common_axis_items
                                if len(self.signal_by_uuid(uuid)[0].plot_samples)
                            ]
                        )
                    else:
                        with_common_axis = False

                for i, (viewbox, signal) in enumerate(
                    zip(self.view_boxes, self.signals)
                ):
                    if len(signal.plot_samples):
                        if signal.uuid in self.common_axis_items:
                            if with_common_axis:
                                min_ = common_min
                                max_ = common_max
                                with_common_axis = False
                            else:
                                continue
                        else:
                            samples = signal.plot_samples[
                                np.isfinite(signal.plot_samples)
                            ]
                            if len(samples):
                                min_, max_ = (
                                    np.nanmin(samples),
                                    np.nanmax(samples),
                                )
                            else:
                                min_, max_ = 0, 1

                        viewbox.setYRange(min_, max_)

                if self.cursor1:
                    self.cursor_moved.emit()

            elif key == QtCore.Qt.Key_G and modifier == QtCore.Qt.NoModifier:
                if self.plotItem.ctrl.yGridCheck.isChecked():
                    self.showGrid(x=True, y=False)
                else:
                    self.showGrid(x=True, y=True)

            elif (
                key in (QtCore.Qt.Key_I, QtCore.Qt.Key_O)
                and modifier == QtCore.Qt.NoModifier
            ):
                x_range, _ = self.viewbox.viewRange()
                delta = x_range[1] - x_range[0]
                step = delta * 0.05
                if key == QtCore.Qt.Key_I:
                    step = -step
                if self.cursor1:
                    pos = self.cursor1.value()
                    x_range = pos - delta / 2, pos + delta / 2
                self.viewbox.setXRange(x_range[0] - step, x_range[1] + step, padding=0)

            elif key == QtCore.Qt.Key_R and modifier == QtCore.Qt.NoModifier:
                if self.region is None:

                    self.region = pg.LinearRegionItem((0, 0))
                    self.region.setZValue(-10)
                    self.plotItem.addItem(self.region)
                    self.region.sigRegionChanged.connect(self.range_modified.emit)
                    self.region.sigRegionChangeFinished.connect(
                        self.range_modified_finished.emit
                    )
                    for line in self.region.lines:
                        line.addMarker("^", 0)
                        line.addMarker("v", 1)
                    start, stop = self.viewbox.viewRange()[0]
                    start, stop = (
                        start + 0.1 * (stop - start),
                        stop - 0.1 * (stop - start),
                    )
                    self.region.setRegion((start, stop))

                    if self.cursor1 is not None:
                        self.cursor1.hide()

                else:
                    self.region_lock = None
                    self.region.setParent(None)
                    self.region.hide()
                    self.region = None
                    self.range_removed.emit()

                    if self.cursor1 is not None:
                        self.cursor1.show()

            elif key == QtCore.Qt.Key_S and modifier == QtCore.Qt.ControlModifier:
                file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Select output measurement file",
                    "",
                    "MDF version 4 files (*.mf4)",
                )

                if file_name:
                    signals = [
                        signal
                        for signal in self.signals
                        if signal.enable
                    ]
                    if signals:
                        with MDF() as mdf:
                            groups = {}
                            for sig in signals:
                                id_ = id(sig.timestamps)
                                if id_ not in groups:
                                    groups[id_] = []
                                groups[id_].append(sig)

                            for signals in groups.values():
                                sigs = []
                                for signal in signals:
                                    if ':' in signal.name:
                                        sig = signal.copy()
                                        sig.name = sig.name.split(':')[-1].strip()
                                        sigs.append(sig)
                                    else:
                                        sigs.append(signal)
                                mdf.append(sigs, common_timebase=True)
                            mdf.save(file_name, overwrite=True)

            elif key == QtCore.Qt.Key_S and modifier == QtCore.Qt.NoModifier:

                parent = self.parent().parent()
                count = parent.channel_selection.count()
                uuids = []

                for i in range(count):
                    item = parent.channel_selection.item(i)
                    uuids.append(parent.channel_selection.itemWidget(item).uuid)
                uuids = reversed(uuids)

                count = sum(
                    1
                    for i, (sig, curve) in enumerate(zip(self.signals, self.curves))
                    if sig.min != "n.a."
                    and curve.isVisible()
                    and sig.uuid not in self.common_axis_items
                )

                if any(
                    sig.min != "n.a." and curve.isVisible() and sig.uuid in self.common_axis_items
                    for (sig, curve) in zip(self.signals, self.curves)
                ):
                    count += 1
                    with_common_axis = True
                else:
                    with_common_axis = False

                if count:

                    position = 0
                    for uuid in uuids:
                        signal, index = self.signal_by_uuid(uuid)
                        viewbox = self.view_boxes[index]

                        if not signal.empty and signal.enable:
                            if (
                                with_common_axis
                                and signal.uuid in self.common_axis_items
                            ):
                                with_common_axis = False

                                min_ = np.nanmin(
                                    [
                                        np.nanmin(
                                            self.signal_by_uuid(uuid)[0].plot_samples
                                        )
                                        for uuid in self.common_axis_items
                                        if len(
                                            self.signal_by_uuid(uuid)[0].plot_samples
                                        )
                                        and self.signal_by_uuid(uuid)[0].enable
                                    ]
                                )
                                max_ = np.nanmax(
                                    [
                                        np.nanmax(
                                            self.signal_by_uuid(uuid)[0].plot_samples
                                        )
                                        for uuid in self.common_axis_items
                                        if len(
                                            self.signal_by_uuid(uuid)[0].plot_samples
                                        )
                                        and self.signal_by_uuid(uuid)[0].enable
                                    ]
                                )

                            else:
                                if signal.uuid in self.common_axis_items:
                                    continue
                                min_ = signal.min
                                max_ = signal.max

                            if min_ == -float("inf") and max_ == float("inf"):
                                min_ = 0
                                max_ = 1
                            elif min_ == -float("inf"):
                                min_ = max_ - 1
                            elif max_ == float("inf"):
                                max_ = min_ + 1

                            if min_ == max_:
                                min_, max_ = min_ - 1, max_ + 1

                            dim = (float(max_) - min_) * 1.1

                            max_ = min_ + dim * count - 0.05 * dim
                            min_ = min_ - 0.05 * dim

                            min_, max_ = (
                                min_ - dim * position,
                                max_ - dim * position,
                            )

                            viewbox.setYRange(min_, max_, padding=0)

                            position += 1

                else:
                    xrange, _ = self.viewbox.viewRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.setXRange(*xrange, padding=0)
                    self.viewbox.disableAutoRange()
                if self.cursor1:
                    self.cursor_moved.emit()

            elif (
                key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right)
                and modifier == QtCore.Qt.NoModifier
            ):
                if self.cursor1:
                    prev_pos = pos = self.cursor1.value()
                    dim = len(self.timebase)
                    if dim:
                        pos = np.searchsorted(self.timebase, pos)
                        if key == QtCore.Qt.Key_Right:
                            pos += 1
                        else:
                            pos -= 1
                        pos = np.clip(pos, 0, dim - 1)
                        pos = self.timebase[pos]
                    else:
                        if key == QtCore.Qt.Key_Right:
                            pos += 1
                        else:
                            pos -= 1

                    (left_side, right_side), _ = self.viewbox.viewRange()

                    if pos >= right_side:
                        delta = abs(pos - prev_pos)
                        self.viewbox.setXRange(
                            left_side + delta, right_side + delta, padding=0
                        )
                    elif pos <= left_side:
                        delta = abs(pos - prev_pos)
                        self.viewbox.setXRange(
                            left_side - delta, right_side - delta, padding=0
                        )
                    else:
                        delta = 0

                    self.cursor1.set_value(pos)

            elif key == QtCore.Qt.Key_H and modifier == QtCore.Qt.NoModifier:
                if len(self.all_timebase):
                    start_ts = np.amin(self.all_timebase)
                    stop_ts = np.amax(self.all_timebase)

                    self.viewbox.setXRange(start_ts, stop_ts)
                    event_ = QtGui.QKeyEvent(
                        QtCore.QEvent.KeyPress, QtCore.Qt.Key_F, QtCore.Qt.NoModifier
                    )
                    self.keyPressEvent(event_)

                    if self.cursor1:
                        self.cursor_moved.emit()

            elif key == QtCore.Qt.Key_Insert and modifier == QtCore.Qt.NoModifier:
                dlg = DefineChannel(self.signals, self.all_timebase, self)
                dlg.setModal(True)
                dlg.exec_()
                sig = dlg.result

                if sig is not None:
                    sig.uuid = uuid4()
                    self.add_new_channels([sig], computed=True)
                    self.computation_channel_inserted.emit()

            else:
                self.parent().keyPressEvent(event)

    def trim(self):
        if not self._can_trim:
            return
        (start, stop), _ = self.viewbox.viewRange()

        width = self.width() - self.axis.width()

        for sig in self.signals:
            dim = len(sig.samples)
            if dim:

                start_t, stop_t = (
                    sig.timestamps[0],
                    sig.timestamps[-1],
                )
                if start > stop_t or stop < start_t:
                    sig.plot_samples = sig.samples[:0]
                    sig.plot_timestamps = sig.timestamps[:0]
                    if sig.plot_texts is not None:
                        sig.plot_texts = sig.texts[:0]
                else:
                    start_ = max(start, start_t)
                    stop_ = min(stop, stop_t)

                    visible = abs(int((stop_ - start_) / (stop - start) * width))

                    start_ = np.searchsorted(sig.timestamps, start_, side="right")
                    stop_ = np.searchsorted(sig.timestamps, stop_, side="right")

                    if visible:
                        raster = abs((stop_ - start_)) // visible
                    else:
                        raster = 0
                    if raster:
                        rows = (stop_ - start_) // raster
                        stop_2 = start_ + rows * raster

                        samples = sig.samples[start_:stop_2].reshape(rows, raster)
                        max_ = np.nanmax(samples, axis=1)
                        min_ = np.nanmin(samples, axis=1)
                        samples = np.dstack((min_, max_)).ravel()

                        timestamps = sig.timestamps[start_:stop_2].reshape(rows, raster)
                        t_max = timestamps[:, -1]
                        t_min = timestamps[:, -0]

                        timestamps = np.dstack((t_min, t_max)).ravel()

                        if stop_2 != stop_:
                            samples_ = sig.samples[stop_2:stop_]
                            max_ = np.nanmax(samples_)
                            min_ = np.nanmin(samples_)
                            samples = np.concatenate((samples, [min_, max_]))

                            timestamps_ = sig.timestamps[stop_2:stop_]
                            t_max = timestamps_[-1]
                            t_min = timestamps_[0]

                            timestamps = np.concatenate((timestamps, [t_min, t_max]))

                        sig.plot_samples = samples
                        sig.plot_timestamps = timestamps
                        if sig.plot_texts is not None:

                            idx = np.searchsorted(sig.timestamps, timestamps)
                            sig.plot_texts = sig.texts[idx]

                    else:
                        start_ = max(0, start_ - 2)
                        stop_ += 2

                        sig.plot_samples = sig.samples[start_:stop_]
                        sig.plot_timestamps = sig.timestamps[start_:stop_]
                        if sig.plot_texts is not None:
                            sig.plot_texts = sig.texts[start_:stop_]

    def xrange_changed_handle(self):
        self.trim()

        self.update_lines(force=True)

    def _resizeEvent(self, ev):

        new_size, last_size = self.geometry(), self._last_size
        if new_size != last_size:
            self._last_size = new_size
            self.xrange_changed_handle()
            super().resizeEvent(ev)

    def set_current_uuid(self, uuid, force=False):
        axis = self.axis
        viewbox = self.viewbox

        sig, index = self.signal_by_uuid(uuid)

        if sig.conversion and hasattr(sig.conversion, "text_0"):
            axis.text_conversion = sig.conversion
        else:
            axis.text_conversion = None
        axis.format = sig.format

        if uuid in self.common_axis_items:
            if self.current_uuid not in self.common_axis_items or force:
                for sig_, vbox in zip(self.signals, self.view_boxes):
                    if sig_.uuid not in self.common_axis_items:
                        vbox.setYLink(None)

                vbox = self.view_boxes[index]
                viewbox.setYRange(*vbox.viewRange()[1], padding=0)
                self.common_viewbox.setYRange(*vbox.viewRange()[1], padding=0)
                self.common_viewbox.setYLink(viewbox)

                if self._settings.value("plot_background") == "Black":
                    axis.setPen("#FFFFFF")
                else:
                    axis.setPen("#000000")
                axis.setLabel(self.common_axis_label)

        else:
            self.common_viewbox.setYLink(None)
            for sig_, vbox in zip(self.signals, self.view_boxes):
                if sig_.uuid not in self.common_axis_items:
                    vbox.setYLink(None)

            viewbox.setYRange(*self.view_boxes[index].viewRange()[1], padding=0)
            self.view_boxes[index].setYLink(viewbox)
            if len(sig.name) <= 32:
                if sig.unit:
                    axis.setLabel(f"{sig.name} [{sig.unit}]")
                else:
                    axis.setLabel(f"{sig.name}")
            else:
                if sig.unit:
                    axis.setLabel(f"{sig.name[:29]}...  [{sig.unit}]")
                else:
                    axis.setLabel(f"{sig.name[:29]}...")

            axis.setPen(sig.color)
            axis.update()

        self.current_uuid = uuid

    def _clicked(self, event):
        modifiers = QtGui.QApplication.keyboardModifiers()

        if QtCore.Qt.Key_C not in self.disabled_keys:

            if self.region is None:
                pos = self.plot_item.vb.mapSceneToView(event.scenePos())

                if self.cursor1 is not None:
                    self.plotItem.removeItem(self.cursor1)
                    self.cursor1.setParent(None)
                    self.cursor1 = None

                self.cursor1 = Cursor(pos=pos, angle=90, movable=True)
                self.plotItem.addItem(self.cursor1, ignoreBounds=True)
                self.cursor1.sigPositionChanged.connect(self.cursor_moved.emit)
                self.cursor1.sigPositionChangeFinished.connect(
                    self.cursor_move_finished.emit
                )
                self.cursor_move_finished.emit()

            else:
                pos = self.plot_item.vb.mapSceneToView(event.scenePos())
                start, stop = self.region.getRegion()

                if self.region_lock is not None:
                    self.region.setRegion((self.region_lock, pos.x()))
                else:
                    if modifiers == QtCore.Qt.ControlModifier:
                        self.region.setRegion((start, pos.x()))
                    else:
                        self.region.setRegion((pos.x(), stop))

    def add_new_channels(self, channels, computed=False):
        geometry = self.viewbox.sceneBoundingRect()
        initial_index = len(self.signals)

        for i, sig in enumerate(channels):
            index = len(self.signals)

            self.signals.append(sig)

            if sig.samples.dtype.kind == "f":
                sig.format = "phys"
                sig.plot_texts = None
            else:
                sig.format = "phys"
                if sig.samples.dtype.kind in "USV":
                    sig.plot_texts = sig.texts = sig.samples
                    sig.samples = np.zeros(len(sig.samples))
                else:
                    sig.plot_texts = None
            sig.enable = True
            sig.individual_axis = False
            if not hasattr(sig, "computed"):
                sig.computed = computed
                if not computed:
                    sig.computation = {}

            if sig.conversion:
                vals = sig.conversion.convert(sig.samples)
                if vals.dtype.kind != "S":
                    nans = np.isnan(vals)
                    samples = np.where(nans, sig.samples, vals,)
                    sig.samples = samples

            sig.plot_samples = sig.samples
            sig.plot_timestamps = sig.timestamps

            sig._stats = {
                "range": (0, -1),
                "range_stats": {},
                "visible": (0, -1),
                "visible_stats": {},
                "fmt": "",
            }

            if hasattr(sig, 'color'):
                color = sig.color or COLORS[index % 10]
            else:
                color = COLORS[index % 10]
            sig.color = color

            if len(sig.samples):
                samples = sig.samples[np.isfinite(sig.samples)]
                if len(samples):
                    sig.min = np.nanmin(samples)
                    sig.max = np.nanmax(samples)
                    sig.avg = np.mean(samples)
                    sig.rms = np.sqrt(np.mean(np.square(samples)))
                else:
                    sig.min = "n.a."
                    sig.max = "n.a."
                    sig.avg = "n.a."
                    sig.rms = "n.a."

                sig.empty = False

            else:
                sig.empty = True
                sig.min = "n.a."
                sig.max = "n.a."
                sig.rms = "n.a."
                sig.avg = "n.a."

            axis = FormatedAxis("right", pen=color)
            if sig.conversion and hasattr(sig.conversion, "text_0"):
                axis.text_conversion = sig.conversion

            view_box = pg.ViewBox(enableMenu=False)
            view_box.disableAutoRange()

            axis.linkToView(view_box)
            if len(sig.name) <= 32:
                axis.labelText = sig.name
            else:
                axis.labelText = f"{sig.name[:29]}..."
            axis.labelUnits = sig.unit
            axis.labelStyle = {"color": color}

            axis.setLabel(axis.labelText, sig.unit, color=color)

            self.layout.addItem(axis, 2, self._axes_layout_pos)
            self._axes_layout_pos += 1

            self.scene_.addItem(view_box)

            t = sig.plot_timestamps

            curve = self.curvetype(
                t,
                sig.plot_samples,
                pen=color,
                symbolBrush=color,
                symbolPen=color,
                symbol="o",
                symbolSize=4,
                clickable=True,
                mouseWidth=30,
                #                connect='finite',
            )
            curve.hide()

            curve.sigClicked.connect(partial(self.curve_clicked.emit, index))

            view_box.setXLink(self.viewbox)
            self.view_boxes.append(view_box)
            self.curves.append(curve)
            if not sig.empty:
                view_box.setYRange(sig.min, sig.max, padding=0, update=True)
            (start, stop), _ = self.viewbox.viewRange()
            view_box.setXRange(start, stop, padding=0, update=True)
            view_box.setGeometry(geometry)

            self.axes.append(axis)
            axis.hide()
            view_box.addItem(curve)

            if id(sig.timestamps) not in self._timebase_db:
                self._timebase_db[id(sig.timestamps)] = set()
            self._timebase_db[id(sig.timestamps)].add(sig.uuid)

        self.trim()
        self.update_lines(force=True)

        for curve in self.curves[initial_index:]:
            curve.show()

        self._compute_all_timebase()

        if initial_index == 0 and self.signals:
            self.set_current_uuid(self.signals[0].uuid)

            if len(self.all_timebase):
                start_t, stop_t = np.amin(self.all_timebase), np.amax(self.all_timebase)
                self.viewbox.setXRange(start_t, stop_t)

    def _compute_all_timebase(self):
        if self._timebase_db:
            timebases = [sig.timestamps for sig in self.signals if id(sig.timestamps) in self._timebase_db]
            self.all_timebase = self.timebase = reduce(np.union1d, timebases)
        else:
            self.all_timebase = self.timebase = []

    def signal_by_uuid(self, uuid):
        for i, sig in enumerate(self.signals):
            if sig.uuid == uuid:
                return sig, i

        raise Exception("Signal not found")

    def signal_by_name(self, name):
        for i, sig in enumerate(self.signals):
            if sig.name == name:
                return sig, i

        raise Exception(
            f"Signal not found: {name} {[sig.name for sig in self.signals]}"
        )

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat("application/octet-stream-asammdf"):
            e.accept()
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        if e.source() is self.parent().channel_selection:
            super().dropEvent(e)
        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
            else:
                super().dropEvent(e)

    def delete_channels(self, deleted):

        needs_timebase_compute = False

        indexes = sorted(
            [(self.signal_by_uuid(uuid)[1], uuid) for uuid in deleted], reverse=True,
        )

        for i, uuid in indexes:
            item = self.curves.pop(i)
            item.hide()
            item.setParent(None)

            item = self.view_boxes.pop(i)
            item.hide()
            item.setParent(None)

            item = self.axes.pop(i)
            item.hide()
            item.setParent(None)

            sig = self.signals.pop(i)

            if uuid in self.common_axis_items:
                self.common_axis_items.remove(uuid)

            if sig.enable:
                self._timebase_db[id(sig.timestamps)].remove(sig.uuid)

                if len(self._timebase_db[id(sig.timestamps)]) == 0:
                    del self._timebase_db[id(sig.timestamps)]
                    needs_timebase_compute = True

        uuids = [sig.uuid for sig in self.signals]

        if uuids:
            if self.current_uuid in uuids:
                self.set_current_uuid(self.current_uuid, True)
            else:
                self.set_current_uuid(uuids[0], True)
        else:
            self.current_uuid = None

        if needs_timebase_compute:
            self._compute_all_timebase()

    def set_time_offset(self, info):
        absolute, offset, *uuids = info
        signals = [
            sig
            for sig in self.signals
            if sig.uuid in uuids
        ]

        if absolute:
            for sig in signals:
                if not len(sig.timestamps):
                    continue
                id_ = id(sig.timestamps)
                delta = sig.timestamps[0] - offset
                sig.timestamps = sig.timestamps - delta

                if id(sig.timestamps) not in self._timebase_db:
                    self._timebase_db[id(sig.timestamps)] = set()
                self._timebase_db[id(sig.timestamps)].add(sig.uuid)

                self._timebase_db[id_].remove(sig.uuid)
                if len(self._timebase_db[id_]) == 0:
                    del self._timebase_db[id_]
        else:
            for sig in signals:
                if not len(sig.timestamps):
                    continue
                id_ = id(sig.timestamps)

                sig.timestamps = sig.timestamps + offset

                if id(sig.timestamps) not in self._timebase_db:
                    self._timebase_db[id(sig.timestamps)] = set()
                self._timebase_db[id(sig.timestamps)].add(sig.uuid)

                self._timebase_db[id_].remove(sig.uuid)
                if len(self._timebase_db[id_]) == 0:
                    del self._timebase_db[id_]

        self._compute_all_timebase()

        self.xrange_changed_handle()

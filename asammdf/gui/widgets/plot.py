# -*- coding: utf-8 -*-
import os

bin_ = bin
import logging
from functools import reduce, partial

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

from ..ui import resource_qt5 as resource_rc
from .list import ListWidget

try:
    import pyqtgraph as pg

    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *

    from ..utils import COLORS
    from .cursor import Cursor
    from .formated_axis import FormatedAxis
    from ..dialogs.define_channel import DefineChannel
    from ...mdf import MDF
    from .list import ListWidget
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

    class Plot(QWidget):

        close_request = pyqtSignal()
        clicked = pyqtSignal()

        def __init__(self, signals, with_dots=False, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setContentsMargins(0, 0, 0, 0)

            main_layout = QVBoxLayout(self)
            self.setLayout(main_layout)

            vbox = QVBoxLayout()
            widget = QWidget()
            self.channel_selection = ListWidget()
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel("Cursor/Range information"))
            self.cursor_info = QLabel("")
            self.cursor_info.setTextFormat(Qt.RichText)
            self.cursor_info.setAlignment(
                Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter
            )
            hbox.addWidget(self.cursor_info)
            vbox.addLayout(hbox)
            vbox.addWidget(self.channel_selection)
            widget.setLayout(vbox)

            self.splitter = QSplitter()
            self.splitter.addWidget(widget)

            self.plot = _Plot(signals, with_dots, self)
            self.plot.range_modified.connect(self.range_modified)
            self.plot.range_removed.connect(self.range_removed)
            self.plot.range_modified_finished.connect(self.range_modified_finished)
            self.plot.cursor_removed.connect(self.cursor_removed)
            self.plot.cursor_moved.connect(self.cursor_moved)
            self.plot.cursor_move_finished.connect(self.cursor_move_finished)
            self.plot.xrange_changed.connect(self.xrange_changed)
            self.plot.computation_channel_inserted.connect(self.computation_channel_inserted)
            self.plot.curve_clicked.connect(self.channel_selection.setCurrentRow)
            self.plot.show()
            self.channel_selection.show()
            self.splitter.addWidget(self.plot)

            for i, sig in enumerate(self.plot.signals):
                if sig.empty:
                    name, unit = sig.name, "[has no samples]"
                else:
                    name, unit = sig.name, sig.unit
                item = QListWidgetItem(self.channel_selection)
                it = ChannelDisplay(i, unit, self)
                it.setAttribute(Qt.WA_StyledBackground)

                it.setName(name)
                it.setValue("")
                it.setColor(sig.color)
                item.setSizeHint(it.sizeHint())
                self.channel_selection.addItem(item)
                self.channel_selection.setItemWidget(item, it)

                it.color_changed.connect(self.plot.setColor)
                it.enable_changed.connect(self.plot.setSignalEnable)
                it.ylink_changed.connect(self.plot.setCommonAxis)

            self.info = ChannelStats()
            self.splitter.addWidget(self.info)
            self.info.hide()

            self.channel_selection.itemsDeleted.connect(self.channel_selection_reduced)
            self.channel_selection.itemSelectionChanged.connect(
                self.channel_selection_modified
            )

            main_layout.addWidget(self.splitter)

        def mousePressEvent(self, event):
            self.clicked.emit()
            super().mousePressEvent(event)

        def channel_selection_modified(self):
            selected_items = list(self.channel_selection.selectedItems())
            if selected_items:
                self.info_index = self.channel_selection.row(selected_items[0])

                if self.plot.signals[self.info_index].enable:

                    self.plot.set_current_index(self.info_index)
                    stats = self.plot.get_stats(self.info_index)
                    self.info.set_stats(stats)

        def channel_selection_reduced(self, deleted):
            for i in sorted(deleted, reverse=True):
                item = self.plot.curves.pop(i)
                item.hide()
                item.setParent(None)

                self.plot.signals.pop(i)

                if self.info_index >= i:
                    self.info_index -= 1

            rows = self.channel_selection.count()

            if not rows:
                self.close_request.emit()
            else:
                for i in range(rows):
                    item = self.channel_selection.item(i)
                    wid = self.channel_selection.itemWidget(item)
                    wid.index = i

                if self.info_index < 0:
                    self.info_index = 0

                self.plot.set_current_index(self.info_index)
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

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

            self.plot.cursor_hint.setData(x=[], y=[])

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

                for viewbox, sig, curve in zip(
                    self.plot.view_boxes, self.plot.signals, self.plot.curves
                ):
                    if curve.isVisible():
                        index = np.argwhere(sig.timestamps == next_pos).flatten()
                        if len(index):
                            _, (y_min, y_max) = viewbox.viewRange()

                            sample = sig.samples[index[0]]
                            sample = (sample - y_min) / (y_max - y_min) * (
                                hint_max - hint_min
                            ) + hint_min

                            y.append(sample)

                self.plot.viewbox.setYRange(hint_min, hint_max, padding=0)
                self.plot.cursor_hint.setData(x=[next_pos] * len(y), y=y)
                self.plot.cursor_hint.show()

            if not self.plot.region:
                self.cursor_info.setText(f"t = {position:.6f}s")
                for i, signal in enumerate(self.plot.signals):
                    cut_sig = signal.cut(position, position)
                    if signal.plot_texts is None or len(cut_sig) == 0:
                        samples = cut_sig.samples
                        if signal.conversion and hasattr(signal.conversion, "text_0"):
                            samples = signal.conversion.convert(samples)
                            if samples.dtype.kind == 'S':
                                try:
                                    samples = [s.decode("utf-8") for s in samples]
                                except:
                                    samples = [s.decode("latin-1") for s in samples]
                            else:
                                samples = samples.tolist()
                    else:
                        t = np.argwhere(signal.plot_timestamps == cut_sig.timestamps).flatten()
                        try:
                            samples = [e.decode("utf-8") for e in signal.plot_texts[t]]
                        except:
                            samples = [e.decode("latin-1") for e in signal.plot_texts[t]]

                    item = self.channel_selection.item(i)
                    item = self.channel_selection.itemWidget(item)

                    item.setPrefix("= ")
                    item.setFmt(signal.format)

                    if len(samples):
                        item.setValue(samples[0])
                    else:
                        item.setValue("n.a.")

            if self.info.isVisible():
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

        def cursor_removed(self):
            for i, signal in enumerate(self.plot.signals):
                item = self.channel_selection.item(i)
                item = self.channel_selection.itemWidget(item)

                if not self.plot.region:
                    self.cursor_info.setText("")
                    item.setPrefix("")
                    item.setValue("")
            if self.info.isVisible():
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

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

            for i, signal in enumerate(self.plot.signals):
                samples = signal.cut(start, stop).samples
                item = self.channel_selection.item(i)
                item = self.channel_selection.itemWidget(item)

                item.setPrefix("Δ = ")
                item.setFmt(signal.format)

                if len(samples):
                    if samples.dtype.kind in "ui":
                        delta = np.int64(np.float64(samples[-1]) - np.float64(samples[0]))
                    else:
                        delta = samples[-1] - samples[0]

                    item.setValue(delta)

                else:
                    item.setValue("n.a.")

            if self.info.isVisible():
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

        def xrange_changed(self):
            if self.info.isVisible():
                stats = self.plot.get_stats(self.info_index)
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
            if event.key() == Qt.Key_M and event.modifiers() == Qt.NoModifier:

                if self.info.isVisible():
                    self.info.hide()
                else:
                    self.info.show()

            else:
                self.plot.keyPressEvent(event)

        def range_removed(self):
            for i, signal in enumerate(self.plot.signals):
                item = self.channel_selection.item(i)
                item = self.channel_selection.itemWidget(item)

                item.setPrefix("")
                item.setValue("")
                self.cursor_info.setText("")

            if self.plot.cursor1:
                self.plot.cursor_moved.emit()
            if self.info.isVisible():
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

        def computation_channel_inserted(self):
            sig = self.plot.signals[-1]
            index = self.channel_selection.count()
            if sig.empty:
                name, unit = sig.name, "[has no samples]"
            else:
                name, unit = sig.name, sig.unit
            item = QListWidgetItem(self.channel_selection)
            it = ChannelDisplay(index, unit, self)
            it.setAttribute(Qt.WA_StyledBackground)

            it.setName(name)
            it.setValue("")
            it.setColor(sig.color)
            item.setSizeHint(it.sizeHint())
            self.channel_selection.addItem(item)
            self.channel_selection.setItemWidget(item, it)

            it.color_changed.connect(self.plot.setColor)
            it.enable_changed.connect(self.plot.setSignalEnable)
            it.ylink_changed.connect(self.plot.setCommonAxis)

            it.enable_changed.emit(index, 1)
            it.enable_changed.emit(index, 0)
            it.enable_changed.emit(index, 1)


    class _Plot(pg.PlotWidget):
        cursor_moved = pyqtSignal()
        cursor_removed = pyqtSignal()
        range_removed = pyqtSignal()
        range_modified = pyqtSignal()
        range_modified_finished = pyqtSignal()
        cursor_move_finished = pyqtSignal()
        xrange_changed = pyqtSignal()
        computation_channel_inserted = pyqtSignal()
        curve_clicked = pyqtSignal(int)

        def __init__(self, signals, with_dots, *args, **kwargs):
            super().__init__()
            self.setContentsMargins(0, 0, 0, 0)
            self.xrange_changed.connect(self.xrange_changed_handle)
            self.with_dots = with_dots
            if self.with_dots:
                self.curvetype = pg.PlotDataItem
            else:
                self.curvetype = pg.PlotCurveItem
            self.info = None
            self.current_index = 0

            self.standalone = kwargs.get("standalone", False)

            self.region = None
            self.cursor1 = None
            self.cursor2 = None
            self.signals = signals

            self.disabled_keys = set()
            for sig in self.signals:
                if sig.samples.dtype.kind == "f":
                    sig.format = "phys"
                    sig.plot_texts = None
                else:
                    sig.format = "phys"
                    if sig.samples.dtype.kind in "SV":
                        sig.plot_texts = sig.texts = sig.samples
                        sig.samples = np.zeros(len(sig.samples))
                    else:
                        sig.plot_texts = None
                sig.enable = True

                if sig.conversion:
                    vals = sig.conversion.convert(sig.samples)
                    if vals.dtype.kind not in 'SV':
                        nans = np.isnan(vals)
                        samples = np.where(
                            nans,
                            sig.samples,
                            vals,
                        )
                        sig.samples = samples

                sig.plot_samples = sig.samples
                sig.plot_timestamps = sig.timestamps

                sig._stats = {
                    "range": (0, -1),
                    "range_stats": {},
                    "visible": (0, -1),
                    "visible_stats": {},
                }

            if self.signals:
                self.all_timebase = self.timebase = reduce(
                    np.union1d, (sig.timestamps for sig in self.signals)
                )
            else:
                self.all_timebase = self.timebase = None

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

            for i, sig in enumerate(self.signals):
                color = COLORS[i % 10]
                sig.color = color

                if len(sig.samples):
                    if sig.samples.dtype.kind not in 'SV':
                        sig.min = np.amin(sig.samples)
                        sig.max = np.amax(sig.samples)
                        sig.avg = np.mean(sig.samples)
                        sig.rms = np.sqrt(np.mean(np.square(sig.samples)))
                    else:
                        sig.min = 'n.a.'
                        sig.max = 'n.a.'
                        sig.avg = 'n.a.'
                        sig.rms = 'n.a.'

                    sig.empty = False
                else:
                    sig.empty = True

                view_box = pg.ViewBox(enableMenu=True)

                self.scene_.addItem(view_box)

                curve = self.curvetype(
                    sig.plot_timestamps,
                    sig.plot_samples,
                    pen=color,
                    symbolBrush=color,
                    symbolPen=color,
                    symbol="o",
                    symbolSize=4,
                )

                curve.sigClicked.connect(partial(self.curve_clicked.emit, i))

                view_box.addItem(curve)

                view_box.setXLink(self.viewbox)

                self.view_boxes.append(view_box)
                self.curves.append(curve)

            self.set_current_index(0)

            #            self.update_views()
            self.viewbox.sigResized.connect(self.update_views)

            #            self.viewbox.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            self.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_H, Qt.NoModifier))

            self.resizeEvent = self._resizeEvent

        def update_lines(self, with_dots=None, force=False):
            with_dots_changed = False

            if with_dots is not None and with_dots != self.with_dots:
                self.with_dots = with_dots
                self.curvetype = pg.PlotDataItem if with_dots else pg.PlotCurveItem
                with_dots_changed = True

            if with_dots_changed or force:
                for i, sig in enumerate(self.signals):
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
                        curve.setData(x=t, y=sig.plot_samples)

                    if sig.enable:
                        curve.show()
                    else:
                        curve.hide()

        def setColor(self, index, color):
            self.signals[index].color = color
            self.curves[index].setPen(color)
            if self.curvetype == pg.PlotDataItem:
                self.curves[index].setSymbolPen(color)
                self.curves[index].setSymbolBrush(color)

            if index == self.current_index:
                self.axis.setPen(color)

        def setCommonAxis(self, index, state):
            viewbox = self.view_boxes[index]
            if state in (Qt.Checked, True, 1):
                viewbox.setYRange(*self.common_viewbox.viewRange()[1], padding=0)
                viewbox.setYLink(self.common_viewbox)
                self.common_axis_items.add(index)
            else:
                self.view_boxes[index].setYLink(None)
                self.common_axis_items.remove(index)

            self.common_axis_label = ', '.join(
                self.signals[i].name
                for i in sorted(self.common_axis_items)
            )

            self.set_current_index(self.current_index, True)

        def setSignalEnable(self, index, state):

            if state in (Qt.Checked, True, 1):
                self.signals[index].enable = True
                self.curves[index].show()
                self.view_boxes[index].setXLink(self.viewbox)
            else:
                self.signals[index].enable = False
                self.curves[index].hide()
                self.view_boxes[index].setXLink(None)

            if self.cursor1:
                self.cursor_move_finished.emit()

        def update_views(self):
            geometry = self.viewbox.sceneBoundingRect()
            for view_box in self.view_boxes:
                view_box.setGeometry(geometry)

        def get_stats(self, index):
            stats = {}
            sig = self.signals[index]
            x = sig.timestamps
            size = len(x)

            if size:

                if sig.plot_texts is not None:
                    stats["overall_min"] = ""
                    stats["overall_max"] = ""
                    stats["overall_average"] = ""
                    stats["overall_rms"] = ""
                    stats["overall_start"] = sig.timestamps[0]
                    stats["overall_stop"] = sig.timestamps[-1]
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
                    stats["overall_min"] = sig.min
                    stats["overall_max"] = sig.max
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
                            if sig.conversion and hasattr(sig.conversion,"text_0"):
                                vals = np.array([val])
                                vals = sig.conversion.convert(vals)
                                if vals.dtype.kind == 'S':
                                    try:
                                        vals = [s.decode("utf-8") for s in vals]
                                    except UnicodeDecodeError:
                                        vals = [s.decode("latin-1") for s in vals]
                                    val = f"{val:.6f}= {vals[0]}"
                                else:
                                    val = f"{val:.6f}= {vals[0]:.6f}"

                            stats["cursor_value"] = val

                        else:
                            stats["cursor_value"] = "n.a."

                    else:
                        stats["cursor_t"] = ""
                        stats["cursor_value"] = ""

                    if self.region:
                        start, stop = self.region.getRegion()

                        if sig._stats["range"] != (start, stop):
                            new_stats = {}
                            new_stats["selected_start"] = start
                            new_stats["selected_stop"] = stop
                            new_stats["selected_delta_t"] = stop - start

                            cut = sig.cut(start, stop)

                            if len(cut):
                                new_stats["selected_min"] = np.amin(cut.samples)
                                new_stats["selected_max"] = np.amax(cut.samples)
                                new_stats["selected_average"] = np.mean(cut.samples)
                                new_stats["selected_rms"] = (
                                     np.sqrt(np.mean(np.square(cut.samples)))
                                )
                                if cut.samples.dtype.kind in "ui":
                                    new_stats["selected_delta"] = int(
                                        float(cut.samples[-1]) - (cut.samples[0])
                                    )
                                else:
                                    new_stats["selected_delta"] = (
                                        cut.samples[-1] - cut.samples[0]
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

                    if sig._stats["visible"] != (start, stop):
                        new_stats = {}
                        new_stats["visible_start"] = start
                        new_stats["visible_stop"] = stop
                        new_stats["visible_delta_t"] = stop - start

                        cut = sig.cut(start, stop)

                        if len(cut):
                            new_stats["visible_min"] = np.amin(cut.samples)
                            new_stats["visible_max"] = np.amax(cut.samples)
                            new_stats["visible_average"] = np.mean(cut.samples)
                            new_stats["visible_rms"] = (
                                 np.sqrt(np.mean(np.square(cut.samples)))
                            )
                            new_stats["visible_delta"] = (
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

            return stats

        def keyPressEvent(self, event):
            key = event.key()
            modifier = event.modifiers()

            if key in self.disabled_keys:
                super().keyPressEvent(event)
            else:

                if key == Qt.Key_C:
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

                    else:
                        self.plotItem.removeItem(self.cursor1)
                        self.cursor1.setParent(None)
                        self.cursor1 = None
                        self.cursor_removed.emit()

                elif key == Qt.Key_F:
                    for i, (viewbox, signal) in enumerate(zip(self.view_boxes, self.signals)):
                        if len(signal.plot_samples):
                            min_, max_ = (
                                np.amin(signal.plot_samples),
                                np.amax(signal.plot_samples),
                            )
                            viewbox.setYRange(min_, max_, padding=0)

                    if self.cursor1:
                        self.cursor_moved.emit()

                elif key == Qt.Key_G:
                    if self.plotItem.ctrl.yGridCheck.isChecked():
                        self.showGrid(x=True, y=False)
                    else:
                        self.showGrid(x=True, y=True)

                elif key in (Qt.Key_I, Qt.Key_O):
                    x_range, _ = self.viewbox.viewRange()
                    delta = x_range[1] - x_range[0]
                    step = delta * 0.05
                    if key == Qt.Key_I:
                        step = -step
                    if self.cursor1:
                        pos = self.cursor1.value()
                        x_range = pos - delta / 2, pos + delta / 2
                    self.viewbox.setXRange(
                        x_range[0] - step, x_range[1] + step, padding=0
                    )

                elif key == Qt.Key_R:
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

                    else:
                        self.region.setParent(None)
                        self.region.hide()
                        self.region = None
                        self.range_removed.emit()

                elif key == Qt.Key_S and modifier == Qt.ControlModifier:
                    file_name, _ = QFileDialog.getSaveFileName(
                        self,
                        "Select output measurement file", "",
                        "MDF version 4 files (*.mf4)",
                    )

                    if file_name:
                        with MDF() as mdf:
                            mdf.append(self.signals)
                            mdf.save(file_name, overwrite=True)

                elif key == Qt.Key_S:
                    count = len(
                        [
                            sig
                            for (sig, curve) in zip(self.signals, self.curves)
                            if not sig.empty and curve.isVisible()
                        ]
                    )

                    if count:
                        position = 0
                        for signal, viewbox, curve in zip(
                            reversed(self.signals),
                            reversed(self.view_boxes),
                            reversed(self.curves),
                        ):
                            if not signal.empty and curve.isVisible():
                                min_ = signal.min
                                max_ = signal.max
                                if min_ == max_:
                                    min_, max_ = min_ - 1, max_ + 1

                                dim = (max_ - min_) * 1.1

                                max_ = min_ + dim * count
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

                elif key == Qt.Key_H and modifier == Qt.ControlModifier:
                    for i, signal in enumerate(self.signals):
                        if signal.samples.dtype.kind in "ui":
                            signal.format = "hex"
                        if self.current_index == i:
                            self.axis.format = "hex"
                            self.axis.hide()
                            self.axis.show()
                    if self.cursor1:
                        self.cursor_moved.emit()

                elif key == Qt.Key_B and modifier == Qt.ControlModifier:
                    for i, signal in enumerate(self.signals):
                        if signal.samples.dtype.kind in "ui":
                            signal.format = "bin"
                        if self.current_index == i:
                            self.axis.format = "bin"
                            self.axis.hide()
                            self.axis.show()
                    if self.cursor1:
                        self.cursor_moved.emit()

                elif key == Qt.Key_P and modifier == Qt.ControlModifier:
                    for i, signal in enumerate(self.signals):
                        if signal.samples.dtype.kind in "ui":
                            signal.format = "phys"
                        if self.current_index == i:
                            self.axis.format = "phys"
                            self.axis.hide()
                            self.axis.show()
                    if self.cursor1:
                        self.cursor_moved.emit()

                elif key in (Qt.Key_Left, Qt.Key_Right):
                    if self.cursor1:
                        prev_pos = pos = self.cursor1.value()
                        dim = len(self.timebase)
                        if dim:
                            pos = np.searchsorted(self.timebase, pos)
                            if key == Qt.Key_Right:
                                pos += 1
                            else:
                                pos -= 1
                            pos = np.clip(pos, 0, dim - 1)
                            pos = self.timebase[pos]
                        else:
                            if key == Qt.Key_Right:
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

                        self.cursor1.setValue(pos)

                elif key == Qt.Key_H:
                    start_ts = [
                        sig.timestamps[0]
                        for sig in self.signals
                        if len(sig.timestamps)
                    ]

                    stop_ts = [
                        sig.timestamps[-1]
                        for sig in self.signals
                        if len(sig.timestamps)
                    ]

                    if start_ts:
                        start_t, stop_t = min(start_ts), max(stop_ts)

                        self.viewbox.setXRange(start_t, stop_t)
                        self.viewbox.autoRange(padding=0)
                        self.viewbox.disableAutoRange()
                        if self.cursor1:
                            self.cursor_moved.emit()

                elif key == Qt.Key_Insert:
                    dlg = DefineChannel(self.signals, self.all_timebase, self)
                    dlg.setModal(True)
                    dlg.exec_()
                    sig = dlg.result
                    if sig is not None:
                        index = len(self.signals)

                        self.signals.append(sig)

                        if sig.samples.dtype.kind == "f":
                            sig.format = "phys"
                            sig.plot_texts = None
                        else:
                            sig.format = "phys"
                            if sig.samples.dtype.kind in "SV":
                                sig.plot_texts = sig.texts = sig.samples
                                sig.samples = np.zeros(len(sig.samples))
                            else:
                                sig.plot_texts = None
                        sig.enable = True

                        if sig.conversion:
                            vals = sig.conversion.convert(sig.samples)
                            if vals.dtype.kind != 'S':
                                nans = np.isnan(vals)
                                samples = np.where(
                                    nans,
                                    sig.samples,
                                    vals,
                                )
                                sig.samples = samples

                        sig.plot_samples = sig.samples
                        sig.plot_timestamps = sig.timestamps

                        sig._stats = {
                            "range": (0, -1),
                            "range_stats": {},
                            "visible": (0, -1),
                            "visible_stats": {},
                        }

                        color = COLORS[index % 10]
                        sig.color = color

                        if len(sig.samples):
                            sig.min = np.amin(sig.samples)
                            sig.max = np.amax(sig.samples)
                            sig.avg = np.mean(sig.samples)
                            sig.rms = np.sqrt(np.mean(np.square(sig.samples)))
                            sig.empty = False
                        else:
                            sig.empty = True

                        view_box = pg.ViewBox(enableMenu=False)

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
                        )

                        view_box.addItem(curve)

                        view_box.setXLink(self.viewbox)
                        self.view_boxes.append(view_box)
                        self.curves.append(curve)

                        view_box.setYRange(sig.min, sig.max, padding=0, update=True)
                        (start, stop), _ = self.viewbox.viewRange()
                        view_box.setXRange(start, stop, padding=0, update=True)
                        QApplication.processEvents()

                        self.computation_channel_inserted.emit()

                else:
                    super().keyPressEvent(event)

        def trim(self, width, start, stop, signals):
            for sig in signals:
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

                        visible = int((stop_ - start_) / (stop - start) * width)

                        start_ = np.searchsorted(
                            sig.timestamps, start_, side="right"
                        )
                        stop_ = np.searchsorted(
                            sig.timestamps, stop_, side="right"
                        )

                        if visible:
                            raster = max((stop_ - start_) // visible, 1)
                        else:
                            raster = 1
                        if raster >= 10:
                            samples = np.array_split(
                                sig.samples[start_:stop_], visible
                            )
                            max_ = np.array([np.amax(s) for s in samples])
                            min_ = np.array([np.amin(s) for s in samples])
                            samples = np.dstack((min_, max_)).ravel()
                            timestamps = np.array_split(
                                sig.timestamps[start_:stop_], visible
                            )
                            timestamps1 = np.array([s[0] for s in timestamps])
                            timestamps2 = np.array([s[1] for s in timestamps])
                            timestamps = np.dstack((timestamps1, timestamps2)).ravel()

                            sig.plot_samples = samples
                            sig.plot_timestamps = timestamps
                            if sig.plot_texts is not None:
                                sig.plot_texts = sig.texts[start_:stop_:raster]
                        else:
                            start_ = max(0, start_ - 2)
                            stop_ += 2

                            sig.plot_samples = sig.samples[start_:stop_]
                            sig.plot_timestamps = sig.timestamps[start_:stop_]
                            if sig.plot_texts is not None:
                                sig.plot_texts = sig.texts[start_:stop_]

        def xrange_changed_handle(self):
            (start, stop), _ = self.viewbox.viewRange()

            width = self.width()
            self.trim(width, start, stop, self.signals)

            self.update_lines(force=True)

        def _resizeEvent(self, ev):
            self.xrange_changed_handle()
            super().resizeEvent(ev)

        def set_current_index(self, index, force=False):
            axis = self.axis
            viewbox = self.viewbox

            sig = self.signals[index]

            if sig.conversion and hasattr(sig.conversion, "text_0"):
                axis.text_conversion = sig.conversion
            else:
                axis.text_conversion = None
            axis.format = sig.format

            if index in self.common_axis_items:
                if self.current_index not in self.common_axis_items or force:
                    for i, vbox in enumerate(self.view_boxes):
                        if i not in self.common_axis_items:
                            vbox.setYLink(None)

                    vbox = self.view_boxes[index]
                    viewbox.setYRange(*vbox.viewRange()[1], padding=0)
                    self.common_viewbox.setYRange(*vbox.viewRange()[1], padding=0)
                    self.common_viewbox.setYLink(viewbox)

                    axis.setPen("#FFFFFF")
                    axis.setLabel(self.common_axis_label)

            else:
                self.common_viewbox.setYLink(None)
                for i, vbox in enumerate(self.view_boxes):
                    if i not in self.common_axis_items:
                        vbox.setYLink(None)

                viewbox.setYRange(*self.view_boxes[index].viewRange()[1], padding=0)
                self.view_boxes[index].setYLink(viewbox)
                if len(sig.name) <= 32:
                    axis.labelText = sig.name
                else:
                    axis.labelText = f"{sig.name[:29]}..."
                axis.setPen(sig.color)
                axis.update()

            self.current_index = index

        def _clicked(self, event):
            x = self.plot_item.vb.mapSceneToView(event.scenePos())

            if self.cursor1 is not None:
                self.plotItem.removeItem(self.cursor1)
                self.cursor1.setParent(None)
                self.cursor1 = None

            self.cursor1 = Cursor(pos=x, angle=90, movable=True)
            self.plotItem.addItem(self.cursor1, ignoreBounds=True)
            self.cursor1.sigPositionChanged.connect(self.cursor_moved.emit)
            self.cursor1.sigPositionChangeFinished.connect(
                self.cursor_move_finished.emit
            )
            self.cursor_move_finished.emit()


except ImportError:
    PYQTGRAPH_AVAILABLE = False

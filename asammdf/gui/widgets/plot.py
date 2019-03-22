# -*- coding: utf-8 -*-
import os

bin_ = bin
import logging
from functools import reduce

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

from ..ui import resource_qt5 as resource_rc

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

    if not hasattr(pg.InfiniteLine, "addMarker"):
        logger = logging.getLogger("asammdf")
        message = (
            "Old pyqtgraph package: Please install the latest pyqtgraph from the "
            "github develop branch\n"
            "pip install -I --no-deps "
            "https://github.com/pyqtgraph/pyqtgraph/archive/develop.zip"
        )
        logger.warning(message)

    class Plot(pg.PlotWidget):
        cursor_moved = pyqtSignal()
        cursor_removed = pyqtSignal()
        range_removed = pyqtSignal()
        range_modified = pyqtSignal()
        range_modified_finished = pyqtSignal()
        cursor_move_finished = pyqtSignal()
        xrange_changed = pyqtSignal()
        computation_channel_inserted = pyqtSignal()

        def __init__(self, signals, with_dots, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setContentsMargins(0, 0, 0, 0)
            self.xrange_changed.connect(self.xrange_changed_handle)
            self.with_dots = with_dots
            if self.with_dots:
                self.curvetype = pg.PlotDataItem
            else:
                self.curvetype = pg.PlotCurveItem
            self.info = None

            self.standalone = kwargs.get("standalone", False)

            self.singleton = None
            self.region = None
            self.cursor1 = None
            self.cursor2 = None
            self.signals = signals
            self.common_axis_items = set()

            self.disabled_keys = set()
            for sig in self.signals:
                if sig.samples.dtype.kind == "f":
                    sig.format = "{:.6f}"
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
            self.viewbox = self.plot_item.vb
            self.viewbox.sigXRangeChanged.connect(self.xrange_changed.emit)

            self.curve = self.curvetype([], [])

            axis = self.layout.itemAt(2, 0)
            axis.setParent(None)
            self.axis = FormatedAxis("left")
            self.layout.removeItem(axis)
            self.layout.addItem(self.axis, 2, 0)
            self.axis.linkToView(axis.linkedView())
            self.plot_item.axes["left"]["item"] = self.axis
            self.plot_item.hideAxis("left")

            self.viewbox.addItem(self.curve)

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
            self.axes = []

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

                axis = FormatedAxis("right", pen=color)
                if sig.conversion and hasattr(sig.conversion, "text_0"):
                    axis.text_conversion = sig.conversion

                view_box = pg.ViewBox(enableMenu=False)

                axis.linkToView(view_box)
                if len(sig.name) <= 32:
                    axis.labelText = sig.name
                else:
                    axis.labelText = f"{sig.name[:29]}..."
                axis.labelUnits = sig.unit
                axis.labelStyle = {"color": color}

                self.layout.addItem(axis, 2, i + 2)

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

                view_box.addItem(curve)

                view_box.setXLink(self.viewbox)

                self.view_boxes.append(view_box)
                self.curves.append(curve)
                self.axes.append(axis)
                axis.hide()

            if len(signals) == 1:
                self.setSignalEnable(0, 1)

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

                    if sig.enable and self.singleton is None:
                        curve.show()
                    else:
                        curve.hide()

                if self.singleton is not None:
                    sig = self.signals[self.singleton]
                    if sig.enable:
                        color = sig.color
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
                        self.viewbox.removeItem(self.curve)

                        self.curve = curve

                        self.viewbox.addItem(curve)

        def setColor(self, index, color):
            self.signals[index].color = color
            self.curves[index].setPen(color)
            if self.curvetype == pg.PlotDataItem:
                self.curves[index].setSymbolPen(color)
                self.curves[index].setSymbolBrush(color)

            self.axes[index].setPen(color)

        def setCommonAxis(self, index, state):
            if state in (Qt.Checked, True, 1):
                self.view_boxes[index].setYLink(self.viewbox)
                self.common_axis_items.add(index)
            else:
                sig = self.signals[index]

                self.view_boxes[index].setYLink(None)
                self.axes[index].labelUnits = sig.unit
                self.axes[index].setLabel(sig.name)
                self.common_axis_items.remove(index)

            if self.common_axis_items:
                if len(self.common_axis_items) == 1:
                    index = list(self.common_axis_items)[0]
                    sig = self.signals[index]
                    self.axes[index].labelUnits = sig.unit
                    self.axes[index].setLabel(sig.name)
                else:
                    axis_text = ', '.join(
                        self.signals[i].name
                        for i in sorted(self.common_axis_items)
                    )
                    for index in self.common_axis_items:
                        self.axes[index].labelUnits = ''
                        self.axes[index].setLabel(axis_text)

        def setSignalEnable(self, index, state):

            if state in (Qt.Checked, True, 1):
                self.signals[index].enable = True
            else:
                self.signals[index].enable = False

            selected_items = [
                index for index, sig in enumerate(self.signals) if sig.enable
            ]

            if len(selected_items) == 1:
                row = selected_items[0]
                if self.singleton != row:
                    self.singleton = row
                    sig = self.signals[row]
                    color = sig.color

                    self.plotItem.showAxis("left")
                    for axis, viewbox, curve in zip(
                        self.axes, self.view_boxes, self.curves
                    ):
                        if curve.isVisible():
                            axis.hide()
                            viewbox.setXLink(None)
                            curve.hide()

                    sig_axis = self.axes[row]
                    viewbox = self.view_boxes[row]
                    viewbox.setXLink(self.viewbox)
                    axis = self.axis

                    if sig.conversion and hasattr(sig.conversion, "text_0"):
                        axis.text_conversion = sig.conversion
                    else:
                        axis.text_conversion = None

                    axis.setRange(*sig_axis.range)
                    axis.linkedView().setYRange(*sig_axis.range)

                    viewbox.setYLink(axis.linkedView())

                    t = sig.plot_timestamps

                    if isinstance(self.curve, pg.PlotCurveItem):
                        self.curve.updateData(
                            t,
                            sig.plot_samples,
                            pen=color,
                            symbolBrush=color,
                            symbolPen=color,
                            symbol="o",
                            symbolSize=4,
                        )
                    else:
                        self.curve.setData(
                            t,
                            sig.plot_samples,
                            pen=color,
                            symbolBrush=color,
                            symbolPen=color,
                            symbol="o",
                            symbolSize=4,
                        )

                    axis.setLabel(sig.name, sig.unit, color=color)

                self.curve.show()
                self.plotItem.showAxis("left")
                self.showGrid(x=False, y=False)
                self.showGrid(x=True, y=True)
                self.timebase = self.curve.xData

            else:
                self.plotItem.hideAxis("left")
                self.curve.hide()

                if len(selected_items):
                    self.singleton = None

                    for i, sig in enumerate(self.signals):
                        if sig.enable and not self.curves[i].isVisible():
                            if i in self.common_axis_items:
                                self.view_boxes[i].setYLink(self.viewbox)
                            else:
                                self.view_boxes[i].setYLink(None)
                            self.view_boxes[i].setXLink(self.viewbox)
                            self.curves[i].show()
                        elif not sig.enable and self.curves[i].isVisible():
                            self.view_boxes[i].setXLink(None)
                            self.axes[i].hide()
                            self.curves[i].hide()

                self.timebase = self.all_timebase
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
                    for viewbox, signal in zip(self.view_boxes, self.signals):
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
                    for axis in self.axes:
                        if axis.grid is False:
                            axis.setGrid(255)
                        else:
                            axis.setGrid(False)

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
                        mdf = MDF()
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
                    for axis, signal in zip(self.axes, self.signals):
                        if axis.isVisible() and signal.samples.dtype.kind in "ui":
                            axis.format = "hex"
                            signal.format = "hex"
                            axis.hide()
                            axis.show()
                    if (
                        self.axis.isVisible()
                        and self.signals[self.singleton].samples.dtype.kind in "ui"
                    ):
                        self.axis.format = "hex"
                        self.signals[self.singleton].format = "hex"
                        self.axis.hide()
                        self.axis.show()
                    if self.cursor1:
                        self.cursor_moved.emit()

                elif key == Qt.Key_B and modifier == Qt.ControlModifier:
                    for axis, signal in zip(self.axes, self.signals):
                        if axis.isVisible() and signal.samples.dtype.kind in "ui":
                            axis.format = "bin"
                            signal.format = "bin"
                            axis.hide()
                            axis.show()
                    if (
                        self.axis.isVisible()
                        and self.signals[self.singleton].samples.dtype.kind in "ui"
                    ):
                        self.axis.format = "bin"
                        self.signals[self.singleton].format = "bin"
                        self.axis.hide()
                        self.axis.show()
                    if self.cursor1:
                        self.cursor_moved.emit()

                elif key == Qt.Key_P and modifier == Qt.ControlModifier:
                    for axis, signal in zip(self.axes, self.signals):
                        if axis.isVisible() and signal.samples.dtype.kind in "ui":
                            axis.format = "phys"
                            signal.format = "phys"
                            axis.hide()
                            axis.show()
                    if (
                        self.axis.isVisible()
                        and self.signals[self.singleton].samples.dtype.kind in "ui"
                    ):
                        self.axis.format = "phys"
                        self.signals[self.singleton].format = "phys"
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
                            sig.format = "{:.6f}"
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
                            sig.empty = False
                        else:
                            sig.empty = True

                        axis = FormatedAxis("right", pen=color)
                        if sig.conversion and hasattr(sig.conversion, "text_0"):
                            axis.text_conversion = sig.conversion

                        view_box = pg.ViewBox(enableMenu=False)

                        axis.linkToView(view_box)
                        axis.labelText = sig.name
                        axis.labelUnits = sig.unit
                        axis.labelStyle = {"color": color}
                        axis.hide()

                        self.layout.addItem(axis, 2, index + 2)

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
                        self.axes.append(axis)

                        view_box.setYRange(sig.min, sig.max, padding=0, update=True)
                        (start, stop), _ = self.viewbox.viewRange()
                        view_box.setXRange(start, stop, padding=0, update=True)
                        axis.showLabel()
                        axis.show()
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


except ImportError:
    PYQTGRAPH_AVAILABLE = False

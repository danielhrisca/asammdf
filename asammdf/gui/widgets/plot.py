# -*- coding: utf-8 -*-
import sys
import os

PYVERSION = sys.version_info[0]
bin_ = bin
import logging
from functools import reduce, partial

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))

try:
    import pyqtgraph as pg

    try:
        from PyQt5.QtGui import *
        from PyQt5.QtWidgets import *
        from PyQt5.QtCore import *
        from PyQt5 import uic
        from ..ui import resource_qt5 as resource_rc

        QT = 5

    except ImportError:
        from PyQt4.QtCore import *
        from PyQt4.QtGui import *
        from PyQt4 import uic
        from ..ui import resource_qt4 as resource_rc

        QT = 4

    from ..utils import COLORS
    from .cursor import Cursor
    from .formated_axis import FormatedAxis
    from ...version import __version__ as libversion

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

        def __init__(self, signals, with_dots, step_mode, *args, **kwargs):
            super(Plot, self).__init__(*args, **kwargs)
            self.xrange_changed.connect(self.xrange_changed_handle)
            self.with_dots = with_dots
            if self.with_dots:
                self.curvetype = pg.PlotDataItem
            else:
                self.curvetype = pg.PlotCurveItem


            self.step_mode = step_mode
            self.info = None

            self.standalone = kwargs.get("standalone", False)

            self.singleton = None
            self.region = None
            self.cursor1 = None
            self.cursor2 = None
            self.signals = signals

            self.disabled_keys = set()
            for sig in self.signals:
                if sig.samples.dtype.kind == "f":
                    sig.stepmode = False
                    sig.format = "{:.6f}"
                    sig.texts = None
                else:
                    if self.step_mode:
                        sig.stepmode = True
                    else:
                        sig.stepmode = False
                    sig.format = "phys"
                    if sig.samples.dtype.kind in "SV":
                        sig.texts = sig.original_texts = sig.samples
                        sig.samples = np.zeros(len(sig.samples))
                    else:
                        sig.texts = None
                sig.enable = True
                sig.original_samples = sig.samples
                sig.original_timestamps = sig.timestamps

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
            self.viewbox.sigResized.connect(self.update_views)
            self.viewbox.sigXRangeChanged.connect(self.xrange_changed.emit)
            self.viewbox.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

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
                    sig.min = np.amin(sig.samples)
                    sig.max = np.amax(sig.samples)
                    sig.empty = False
                else:
                    sig.empty = True

                axis = FormatedAxis("right")
                axis.setPen(color)
                if sig.conversion and "text_0" in sig.conversion:
                    axis.text_conversion = sig.conversion

                view_box = pg.ViewBox()
                view_box.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

                axis.linkToView(view_box)
                axis.setLabel(sig.name, sig.unit, color=color)

                self.layout.addItem(axis, 2, i + 2)

                self.scene_.addItem(view_box)

                if sig.stepmode:
                    if len(sig.timestamps):
                        to_append = sig.timestamps[-1]
                    else:
                        to_append = 0
                    t = np.append(sig.timestamps, to_append)
                else:
                    t = sig.timestamps

                curve = self.curvetype(
                    t,
                    sig.samples,
                    pen=color,
                    symbolBrush=color,
                    symbolPen=color,
                    symbol="o",
                    symbolSize=4,
                    stepMode=sig.stepmode,
                    # connect='pairs'
                )
                # curve.setDownsampling(ds=100, auto=True, method='peak')

                view_box.addItem(curve)

                view_box.setXLink(self.viewbox)
                view_box.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
                view_box.sigResized.connect(self.update_views)

                self.view_boxes.append(view_box)
                self.curves.append(curve)
                self.axes.append(axis)
                axis.hide()

            if len(signals) == 1:
                self.setSignalEnable(0, 1)

            self.update_views()

            self.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_H, Qt.NoModifier))

            self.resizeEvent = self._resizeEvent

        def update_lines(self, with_dots=None, step_mode=None, force=False):
            step_mode_changed = False
            with_dots_changed = False
            if step_mode is not None and step_mode != self.step_mode:
                self.step_mode = step_mode
                step_mode_changed = True

            if with_dots is not None and with_dots != self.with_dots:
                self.with_dots = with_dots
                self.curvetype = pg.PlotDataItem if with_dots else pg.PlotCurveItem
                with_dots_changed = True

            if with_dots_changed or step_mode_changed or force:
                for i, sig in enumerate(self.signals):
                    sig.stepmode = self.step_mode
                    color = sig.color
                    if sig.stepmode:
                        if len(sig.timestamps):
                            to_append = sig.timestamps[-1]
                        else:
                            to_append = 0
                        t = np.append(sig.timestamps, to_append)
                    else:
                        t = sig.timestamps

                    if not force:
                        try:
                            curve = self.curvetype(
                                t,
                                sig.samples,
                                pen=color,
                                symbolBrush=color,
                                symbolPen=color,
                                symbol="o",
                                symbolSize=4,
                                stepMode=sig.stepmode,
                                # connect='pairs'
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
                        curve.setData(x=t, y=sig.samples, stepMode=sig.stepmode)

                    if sig.enable and self.singleton is None:
                        curve.show()
                    else:
                        curve.hide()

                if self.singleton is not None:
                    sig = self.signals[self.singleton]
                    if sig.enable:
                        color = sig.color
                        if sig.stepmode:
                            if len(sig.timestamps):
                                to_append = sig.timestamps[-1]
                            else:
                                to_append = 0
                            t = np.append(sig.timestamps, to_append)
                        else:
                            t = sig.timestamps

                        curve = self.curvetype(
                            t,
                            sig.samples,
                            pen=color,
                            symbolBrush=color,
                            symbolPen=color,
                            symbol="o",
                            symbolSize=4,
                            stepMode=sig.stepmode,
                            # connect='pairs'
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
                            viewbox.setYLink(None)
                            viewbox.setXLink(None)
                            curve.hide()

                    sig_axis = self.axes[row]
                    viewbox = self.view_boxes[row]
                    viewbox.setXLink(self.viewbox)
                    axis = self.axis

                    if sig.conversion and "text_0" in sig.conversion:
                        axis.text_conversion = sig.conversion
                    else:
                        axis.text_conversion = None

                    axis.setRange(*sig_axis.range)
                    axis.linkedView().setYRange(*sig_axis.range)

                    viewbox.setYLink(axis.linkedView())

                    if sig.stepmode:
                        if len(sig.timestamps):
                            to_append = sig.timestamps[-1]
                        else:
                            to_append = 0
                        t = np.append(sig.timestamps, to_append)
                    else:
                        t = sig.timestamps

                    if isinstance(self.curve, pg.PlotCurveItem):
                        self.curve.updateData(
                            t,
                            sig.samples,
                            pen=color,
                            symbolBrush=color,
                            symbolPen=color,
                            symbol="o",
                            symbolSize=4,
                            stepMode=sig.stepmode,
                        )
                    else:
                        self.curve.setData(
                            t,
                            sig.samples,
                            pen=color,
                            symbolBrush=color,
                            symbolPen=color,
                            symbol="o",
                            symbolSize=4,
                            stepMode=sig.stepmode,
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
                            # self.axes[i].show()
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
            self.viewbox.linkedViewChanged(self.viewbox, self.viewbox.XAxis)
            for view_box in self.view_boxes:
                if view_box.isVisible():
                    view_box.setGeometry(self.viewbox.sceneBoundingRect())
                    view_box.linkedViewChanged(self.viewbox, view_box.XAxis)

        def get_stats(self, index):
            stats = {}
            sig = self.signals[index]
            x = sig.original_timestamps
            size = len(x)

            if size:

                if sig.texts is not None:
                    stats["overall_min"] = ""
                    stats["overall_max"] = ""
                    stats["overall_start"] = sig.original_timestamps[0]
                    stats["overall_stop"] = sig.original_timestamps[-1]
                    stats["unit"] = ""
                    stats["color"] = sig.color
                    stats["name"] = sig.name

                    if self.cursor1:
                        position = self.cursor1.value()
                        stats["cursor_t"] = position

                        if x[0] <= position <= x[-1]:
                            idx = np.searchsorted(x, position)
                            text = sig.original_texts[idx]
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
                    stats["selected_delta"] = ""
                    stats["visible_min"] = ""
                    stats["visible_max"] = ""
                    stats["visible_delta"] = ""
                else:
                    stats["overall_min"] = sig.min
                    stats["overall_max"] = sig.max
                    stats["overall_start"] = sig.original_timestamps[0]
                    stats["overall_stop"] = sig.original_timestamps[-1]
                    stats["unit"] = sig.unit
                    stats["color"] = sig.color
                    stats["name"] = sig.name

                    if self.cursor1:
                        position = self.cursor1.value()
                        stats["cursor_t"] = position

                        if x[0] <= position <= x[-1]:
                            idx = np.searchsorted(x, position)
                            val = sig.original_samples[idx]
                            if sig.conversion and "text_0" in sig.conversion:
                                vals = np.array([val])
                                vals = sig.conversion.convert(vals)
                                try:
                                    vals = [s.decode("utf-8") for s in vals]
                                except:
                                    vals = [s.decode("latin-1") for s in vals]
                                val = "{}= {}".format(val, vals[0])
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
                            new_stats["visible_delta"] = (
                                cut.samples[-1] - cut.samples[0]
                            )

                        else:
                            new_stats["visible_min"] = "n.a."
                            new_stats["visible_max"] = "n.a."
                            new_stats["visible_delta"] = "n.a."

                        sig._stats["visible"] = (start, stop)
                        sig._stats["visible_stats"] = new_stats

                    stats.update(sig._stats["visible_stats"])

            else:
                stats["overall_min"] = "n.a."
                stats["overall_max"] = "n.a."
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
                    stats["selected_delta"] = "n.a."

                else:
                    stats["selected_start"] = ""
                    stats["selected_stop"] = ""
                    stats["selected_delta_t"] = ""
                    stats["selected_min"] = ""
                    stats["selected_max"] = ""
                    stats["selected_delta"] = ""

                (start, stop), _ = self.viewbox.viewRange()

                stats["visible_start"] = start
                stats["visible_stop"] = stop
                stats["visible_delta_t"] = stop - start

                stats["visible_min"] = "n.a."
                stats["visible_max"] = "n.a."
                stats["visible_delta"] = "n.a."

            return stats

        def keyPressEvent(self, event):
            key = event.key()
            modifier = event.modifiers()

            if key in self.disabled_keys:
                super(Plot, self).keyPressEvent(event)
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
                    x_range, _ = self.viewbox.viewRange()
                    for viewbox in self.view_boxes:
                        viewbox.autoRange(padding=0)
                        viewbox.disableAutoRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.setXRange(*x_range, padding=0)
                    self.viewbox.disableAutoRange()
                    if self.cursor1:
                        self.cursor_moved.emit()

                elif key == Qt.Key_G:
                    if self.plotItem.ctrl.xGridCheck.isChecked():
                        self.showGrid(x=False, y=False)
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
                    self.viewbox.setXRange(x_range[0] - step, x_range[1] + step, padding=0)

                elif key == Qt.Key_R:
                    if self.region is None:

                        self.region = pg.LinearRegionItem((0, 0))
                        self.region.setZValue(-10)
                        self.plotItem.addItem(self.region)
                        self.region.sigRegionChanged.connect(self.range_modified.emit)
                        self.region.sigRegionChangeFinished.connect(
                            self.range_modified_finished.emit
                        )
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
                                reversed(self.curves)):
                            if not signal.empty and curve.isVisible():
                                min_ = signal.min
                                max_ = signal.max
                                if min_ == max_:
                                    min_, max_ = min_ - 1, max_ + 1

                                dim = (max_ - min_) * 1.1

                                max_ = min_ + dim * count
                                min_, max_ = min_ - dim * position, max_ - dim * position

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
                    for sig, viewbox in zip(self.signals, self.view_boxes):
                        if len(sig.original_timestamps):
                            viewbox.setXRange(sig.original_timestamps[0], sig.original_timestamps[-1])
                        viewbox.autoRange(padding=0)
                        viewbox.disableAutoRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.disableAutoRange()
                    if self.cursor1:
                        self.cursor_moved.emit()

                else:
                    super(Plot, self).keyPressEvent(event)

        def xrange_changed_handle(self):
            (start, stop), _ = self.viewbox.viewRange()
            width = self.width()
            for sig in self.signals:
                dim = len(sig.original_samples)
                if dim:

                    start_t, stop_t = sig.original_timestamps[0], sig.original_timestamps[-1]
                    if start > stop_t or stop < start_t:
                        sig.samples = sig.original_samples[:0]
                        sig.timestamps = sig.original_timestamps[:0]
                        if sig.texts is not None:
                            sig.texts = sig.original_texts[:0]
                    else:
                        start_ = max(start, start_t)
                        stop_ = min(stop, stop_t)

                        visible = int((stop_ - start_) / (stop - start) * width)

                        start_ = np.searchsorted(sig.original_timestamps, start_, side="right")
                        stop_ = np.searchsorted(sig.original_timestamps, stop_, side="right")

                        if visible:
                            raster = max((stop_ - start_) // visible, 1)
                        else:
                            raster = 1
                        if raster >= 10:
                            samples = np.array_split(sig.original_samples[start_:stop_], visible)
                            max_ = np.array([np.amax(s) for s in samples])
                            min_ = np.array([np.amin(s) for s in samples])
                            samples = np.dstack((min_, max_)).ravel()
                            timestamps = np.array_split(sig.original_timestamps[start_:stop_], visible)
                            timestamps1 = np.array([s[0] for s in timestamps])
                            timestamps2 = np.array([s[1] for s in timestamps])
                            timestamps = np.dstack((timestamps1, timestamps2)).ravel()

                            sig.samples = samples
                            sig.timestamps = timestamps
                            if sig.texts is not None:
                                sig.texts = sig.original_texts[start_:stop_:raster]
                        else:
                            start_ = max(0, start_ -2)
                            stop_ += 2

                            sig.samples = sig.original_samples[start_: stop_]
                            sig.timestamps = sig.original_timestamps[start_: stop_]
                            if sig.texts is not None:
                                sig.texts = sig.original_texts[start_: stop_]

                        self.update_lines(force=True)

        def _resizeEvent(self, ev):
            self.xrange_changed_handle()
            super(Plot, self).resizeEvent(ev)

except ImportError:
    raise
    PYQTGRAPH_AVAILABLE = False

# -*- coding: utf-8 -*-
import sys
import os

PYVERSION = sys.version_info[0]
bin_ = bin
import logging
from functools import reduce, partial

import numpy as np

try:
    HERE = os.path.dirname(os.path.realpath(__file__))
except:
    HERE = os.path.abspath(os.path.dirname(sys.argv[0]))

try:
    import pyqtgraph as pg

    try:
        from PyQt5.QtGui import *
        from PyQt5.QtWidgets import *
        from PyQt5.QtCore import *
        from PyQt5 import uic
        from asammdfgui import resource_qt5 as resource_rc

        QT = 5

    except ImportError:
        from PyQt4.QtCore import *
        from PyQt4.QtGui import *
        from PyQt4 import uic
        from asammdfgui import resource_qt4 as resource_rc

        QT = 4

    from .version import __version__ as libversion

    if not hasattr(pg.InfiniteLine, "addMarker"):
        logger = logging.getLogger("asammdf")
        message = (
            "Old pyqtgraph package: Please install the latest pyqtgraph from the "
            "github develop branch\n"
            "pip install -I --no-deps "
            "https://github.com/pyqtgraph/pyqtgraph/archive/develop.zip"
        )
        logger.warning(message)

    COLORS = [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ]

    COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    class Cursor(pg.InfiniteLine):
        def __init__(self, *args, **kwargs):

            super(Cursor, self).__init__(
                *args, label="{value:.6f}s", labelOpts={"position": 0.04}, **kwargs
            )

            try:
                self.addMarker("^", 0)
                self.addMarker("v", 1)
            except:
                pass
            self.label.show()

    class FormatedAxis(pg.AxisItem):
        def __init__(self, *args, **kwargs):

            super(FormatedAxis, self).__init__(*args, **kwargs)

            self.format = "phys"
            self.text_conversion = None

        def tickStrings(self, values, scale, spacing):
            strns = []

            if self.format == "phys":
                strns = super(FormatedAxis, self).tickStrings(values, scale, spacing)
                if self.text_conversion:
                    strns = self.text_conversion.convert(np.array(values))
                    try:
                        strns = [s.decode("utf-8") for s in strns]
                    except:
                        strns = [s.decode("latin-1") for s in strns]

            elif self.format == "hex":
                for val in values:
                    val = float(val)
                    if val.is_integer():
                        val = hex(int(val))
                    else:
                        val = ""
                    strns.append(val)

            elif self.format == "bin":
                for val in values:
                    val = float(val)
                    if val.is_integer():
                        val = bin_(int(val))
                    else:
                        val = ""
                    strns.append(val)

            return strns

    class ChannelStats(QWidget):
        def __init__(self, *args, **kwargs):
            super(ChannelStats, self).__init__(*args, **kwargs)
            uic.loadUi(os.path.join(HERE, "..", "asammdfgui", "channel_stats.ui"), self)

            self.color = "#000000"
            self.fmt = "phys"
            self.name_template = '<html><head/><body><p><span style=" font-size:11pt; font-weight:600; color:{};">{}</span></p></body></html>'
            self._name = "Please select a single channel"

        def set_stats(self, stats):
            if stats:
                for name, value in stats.items():
                    try:
                        if value.dtype.kind in "ui":
                            sign = "-" if value < 0 else ""
                            value = abs(value)
                            if self.fmt == "hex":
                                value = "{}0x{:X}".format(sign, value)
                            elif self.fmt == "bin":
                                value = "{}0b{:b}".format(sign, value)
                            else:
                                value = "{}{}".format(sign, value)
                        else:
                            value = "{:.6f}".format(value)
                    except:
                        if isinstance(value, int):
                            sign = "-" if value < 0 else ""
                            value = abs(value)
                            if self.fmt == "hex":
                                value = "{}0x{:X}".format(sign, value)
                            elif self.fmt == "bin":
                                value = "{}0b{:b}".format(sign, value)
                            else:
                                value = "{}{}".format(sign, value)
                        elif isinstance(value, float):
                            value = "{:.6f}".format(value)
                        else:
                            value = value

                    if name == "unit":
                        for i in range(1, 10):
                            label = self.findChild(QLabel, "unit{}".format(i))
                            label.setText(" {}".format(value))
                    elif name == "name":
                        self._name = value
                        self.name.setText(
                            self.name_template.format(self.color, self._name)
                        )
                    elif name == "color":
                        self.color = value
                        self.name.setText(
                            self.name_template.format(self.color, self._name)
                        )
                    else:
                        label = self.findChild(QLabel, name)
                        label.setText(value)
            else:
                self.clear()

        def clear(self):
            self._name = "Please select a single channel"
            self.color = "#000000"
            self.name.setText(self.name_template.format(self.color, self._name))
            for group in (
                self.cursor_group,
                self.range_group,
                self.visible_group,
                self.overall_group,
            ):
                layout = group.layout()
                rows = layout.rowCount()
                for i in range(rows):
                    label = layout.itemAtPosition(i, 1).widget()
                    label.setText("")
                for i in range(rows // 2, rows):
                    label = layout.itemAtPosition(i, 2).widget()
                    label.setText("")

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
                        sig.texts = sig.samples
                        sig.samples = np.zeros(len(sig.samples))
                    else:
                        sig.texts = None
                sig.enable = True

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

        def update_lines(self, with_dots=None, step_mode=None):
            step_mode_changed = False
            with_dots_changed = False
            if step_mode is not None and step_mode != self.step_mode:
                self.step_mode = step_mode
                step_mode_changed = True

            if with_dots is not None and with_dots != self.with_dots:
                self.with_dots = with_dots
                self.curvetype = pg.PlotDataItem if with_dots else pg.PlotCurveItem
                with_dots_changed = True

            if with_dots_changed or step_mode_changed:
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

                    if sig.enable and self.singleton is None:
                        curve.show()
                    else:
                        curve.hide()

                if self.singleton is not None:
                    sig = self.signals[self.singleton]
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
            x = sig.timestamps
            size = len(x)

            if size:

                if sig.texts is not None:
                    stats["overall_min"] = ""
                    stats["overall_max"] = ""
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
                    stats["selected_delta"] = ""
                    stats["visible_min"] = ""
                    stats["visible_max"] = ""
                    stats["visible_delta"] = ""
                else:
                    stats["overall_min"] = sig.min
                    stats["overall_max"] = sig.max
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
                self.viewbox.autoRange(padding=0)
                self.viewbox.setXRange(*x_range, padding=0)
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
                    for i, signal in enumerate(self.signals):
                        if not signal.empty and self.curves[i].isVisible():
                            viewbox = self.view_boxes[i]
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
                for viewbox in self.view_boxes:
                    viewbox.autoRange(padding=0)
                self.viewbox.autoRange(padding=0)
                if self.cursor1:
                    self.cursor_moved.emit()

            else:
                super(Plot, self).keyPressEvent(event)

    PYQTGRAPH_AVAILABLE = True

    class StandalonePlot(QWidget):
        def __init__(self, signals, with_dots, step_mode, *args, **kwargs):
            super(StandalonePlot, self).__init__(*args, **kwargs)

            self.splitter = QSplitter(self)
            self.splitter.setOrientation(Qt.Horizontal)
            self.info = None

            self.plot = Plot(signals, with_dots, step_mode, standalone=True)
            self.splitter.addWidget(self.plot)

            self.plot.range_modified.connect(self.range_modified)
            self.plot.range_removed.connect(self.range_removed)
            self.plot.range_modified_finished.connect(self.range_modified_finished)
            self.plot.cursor_removed.connect(self.cursor_removed)
            self.plot.cursor_moved.connect(self.cursor_moved)
            self.plot.cursor_move_finished.connect(self.cursor_move_finished)
            self.plot.xrange_changed.connect(self.xrange_changed)

            vbox = QVBoxLayout()

            vbox.addWidget(self.splitter, 1)

            hbox = QHBoxLayout()

            for icon, description in (
                (":/cursor.png", "C - Cursor"),
                (":/fit.png", "F - Fit"),
                (":/grid.png", "G - Grid"),
                (":/home.png", "H - Home"),
                (":/zoom-in.png", "I - Zoom-in"),
                (":/zoom-out.png", "O - Zoom-out"),
                (":/info.png", "M - Statistics"),
                (":/range.png", "R - Range"),
                (":/right.png", "← - Move cursor left"),
                (":/left.png", "→ - Move cursor right"),
            ):
                label = QLabel("")
                label.setPixmap(QPixmap(icon).scaled(QSize(16, 16)))

                hbox.addWidget(label)
                label = QLabel(description)
                hbox.addWidget(label)
                hbox.addStretch()

            vbox.addLayout(hbox, 0)
            self.setLayout(vbox)

            icon = QIcon()
            icon.addPixmap(QPixmap(":/info.png"), QIcon.Normal, QIcon.Off)
            self.setWindowIcon(icon)

            self.show()

        def keyPressEvent(self, event):
            key = event.key()
            modifier = event.modifiers()

            if key == Qt.Key_M:
                if self.info is None:
                    self.info = ChannelStats()
                    self.splitter.addWidget(self.info)
                    stats = self.plot.get_stats(0)
                    self.info.set_stats(stats)
                else:
                    self.info.setParent(None)
                    self.info.hide()
                    self.info = None
            else:
                super(StandalonePlot, self).keyPressEvent(event)

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

                if self.plot.curve.isVisible():
                    timestamps = self.plot.curve.xData
                    samples = self.plot.curve.yData
                    index = np.argwhere(timestamps == next_pos).flatten()
                    if len(index):
                        _, (y_min, y_max) = self.plot.viewbox.viewRange()

                        sample = samples[index[0]]
                        sample = (sample - y_min) / (y_max - y_min) * (
                            hint_max - hint_min
                        ) + hint_min

                        y.append(sample)

                self.plot.viewbox.setYRange(hint_min, hint_max, padding=0)
                self.plot.cursor_hint.setData(x=[next_pos] * len(y), y=y)
                self.plot.cursor_hint.show()

            if self.info:
                stats = self.plot.get_stats(0)
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

        def range_modified(self):
            if self.info:
                stats = self.plot.get_stats(0)
                self.info.set_stats(stats)

        def xrange_changed(self):

            if self.info:
                stats = self.plot.get_stats(0)
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

        def range_removed(self):
            if self.info:
                stats = self.plot.get_stats(0)
                self.info.set_stats(stats)

        def cursor_removed(self):
            if self.info:
                stats = self.plot.get_stats(0)
                self.info.set_stats(stats)


except ImportError:
    PYQTGRAPH_AVAILABLE = True

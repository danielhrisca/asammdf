from functools import partial

from natsort import natsorted
import numpy as np
from pyqtgraph import functions as fn
from PySide6 import QtCore, QtGui, QtWidgets

from ..ui.signal_scale import Ui_ScaleDialog
from ..utils import BLUE
from .plot import PlotSignal

PLOT_HEIGTH = 600  # pixels
TEXT_WIDTH = 50  # pixels


class ScaleDialog(Ui_ScaleDialog, QtWidgets.QDialog):
    def __init__(self, signals, y_range, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.signals = signals
        self._inhibit = True

        self.signal.addItems(list(natsorted(signals)))
        self.signal.currentTextChanged.connect(self.signal_selected)

        self.setWindowTitle("Y scale configuration")

        self.scaling.setMinimum(0.000001)
        self.scaling.setMaximum(np.inf)
        self.target_min.setMinimum(-np.inf)
        self.target_min.setMaximum(np.inf)
        self.target_max.setMinimum(-np.inf)
        self.target_max.setMaximum(np.inf)
        self.offset.setMinimum(-np.inf)
        self.offset.setMaximum(np.inf)

        self.signal_selected(self.signal.currentText())

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.fit_btn.clicked.connect(self.fit)
        self.shift_up_btn.clicked.connect(partial(self.shift, step=1))
        self.shift_down_btn.clicked.connect(partial(self.shift, step=-1))
        self.fast_shift_up_btn.clicked.connect(partial(self.shift, step=10))
        self.fast_shift_down_btn.clicked.connect(partial(self.shift, step=-10))

        self.apply_btn.setAutoDefault(False)
        self.cancel_btn.setAutoDefault(False)
        self.zoom_in_btn.setAutoDefault(False)
        self.zoom_out_btn.setAutoDefault(False)
        self.fit_btn.setAutoDefault(False)
        self.shift_up_btn.setAutoDefault(False)
        self.shift_down_btn.setAutoDefault(False)
        self.fast_shift_up_btn.setAutoDefault(False)
        self.fast_shift_down_btn.setAutoDefault(False)

        self.offset.valueChanged.connect(self.draw_plot)
        self.scaling.valueChanged.connect(self.draw_plot)
        self.target_max.valueChanged.connect(self.set_target)
        self.target_min.valueChanged.connect(self.set_target)

        canvas = QtGui.QPixmap(PLOT_HEIGTH + 3 * TEXT_WIDTH, PLOT_HEIGTH)
        canvas.fill(QtCore.Qt.GlobalColor.black)
        self.plot.setPixmap(canvas)
        self.plot.setFocus()

        self._inhibit = False

        self.signal.setCurrentIndex(0)
        bottom, top = y_range

        if top == bottom:
            top += 1
            bottom -= 1
            scaling = 2
        else:
            scaling = top - bottom

        self.scaling.setValue(scaling)
        self.offset.setValue(-bottom * 100 / scaling)

    def set_target(self, *args):
        original_change = not self._inhibit

        if original_change:
            self._inhibit = True

            if self.target_max.value() < self.target_min.value():
                val = self.target_min.value()

                self.target_min.setValue(self.target_max.value())
                self.target_max.setValue(val)

            self._inhibit = False

        self.draw_plot()

    def apply(self, *args):
        self.accept()

    def cancel(self):
        self.reject()

    def zoom_in(self, *args):
        self.scaling.setValue(self.scaling.value() * 0.9)

    def zoom_out(self, *args):
        self.scaling.setValue(self.scaling.value() * 1.1)

    def fit(self, *args):
        scaling = (self.target_max.value() - self.target_min.value()) or 2
        pos = -(100 * self.target_min.value() / scaling)
        self.offset.setValue(pos)
        self.scaling.setValue(scaling)

    def draw_plot(self, *args):
        offset = self.offset.value()
        scale = self.scaling.value()

        y_bottom = -offset * scale / 100
        y_top = y_bottom + scale

        self.y_bottom.setText(f"{y_bottom:.3f}")
        self.y_top.setText(f"{y_top:.3f}")

        canvas = self.plot.pixmap()
        canvas.fill(QtCore.Qt.GlobalColor.black)

        if self.samples is None:
            self.target_max.setEnabled(True)
            self.target_min.setEnabled(True)
        else:
            self.target_max.setEnabled(False)
            self.target_min.setEnabled(False)

        if self.samples is None:
            x = np.linspace(0, 2 * np.pi * 10, PLOT_HEIGTH)
            amp = (self.target_max.value() - self.target_min.value()) / 2
            off = (self.target_max.value() + self.target_min.value()) / 2
            y = np.sin(x) * amp + off
            x = np.arange(50, 50 + PLOT_HEIGTH)
        else:
            x = self.samples.plot_timestamps
            y = self.samples.plot_samples.copy()

        ys = y_top
        y_scale = (y_top - y_bottom) / PLOT_HEIGTH

        ys = ys + y_scale

        y = (ys - y) / y_scale

        polygon = fn.create_qpolygonf(len(x))
        ndarray = fn.ndarray_from_qpolygonf(polygon)

        ndarray[:, 0] = x
        ndarray[:, 1] = y

        painter = QtGui.QPainter(canvas)
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.white)
        painter.setPen(pen)

        step = PLOT_HEIGTH // 10
        for i, x in enumerate(range(0, PLOT_HEIGTH + step, step)):
            if i == 0:
                painter.drawText(5, x + 15, f"{100 - 10 * i}%")
            elif i == 10:
                painter.drawText(5, x - 5, f"{100 - 10 * i}%")
            else:
                painter.drawText(5, x + 6, f"{100-10*i}%")

        painter.drawText(PLOT_HEIGTH + TEXT_WIDTH, 15, f"{y_top:.3f}")
        painter.drawText(
            PLOT_HEIGTH + TEXT_WIDTH,
            PLOT_HEIGTH - 5,
            f"{y_bottom:.3f}",
        )

        painter.setClipping(True)
        painter.setClipRect(QtCore.QRect(TEXT_WIDTH, 0, PLOT_HEIGTH, PLOT_HEIGTH))

        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        painter.setPen(pen)
        for i, x in enumerate(range(0, PLOT_HEIGTH + step, step)):
            painter.drawLine(0, x, PLOT_HEIGTH + 2 * TEXT_WIDTH, x)

        pen = QtGui.QPen(BLUE)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPolyline(polygon)

        pen = QtGui.QPen("#18e223")
        pen.setWidth(2)
        pen.setStyle(QtCore.Qt.PenStyle.DashDotDotLine)
        painter.setPen(pen)
        offset = PLOT_HEIGTH - self.offset.value() * PLOT_HEIGTH / 100

        p1 = QtCore.QPointF(0.0, offset)
        p2 = QtCore.QPointF(float(PLOT_HEIGTH + 2 * TEXT_WIDTH), offset)

        painter.drawLine(p1, p2)

        painter.end()

        self.plot.setPixmap(canvas)

    def shift(self, *args, step=1):
        self.offset.stepBy(step)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key.Key_I and modifiers in (
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.KeyboardModifier.ShiftModifier,
        ):
            event.accept()
            self.zoom_in()

        elif key == QtCore.Qt.Key.Key_O and modifiers in (
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.KeyboardModifier.ShiftModifier,
        ):
            event.accept()
            self.zoom_out()

        elif key == QtCore.Qt.Key.Key_F and modifiers in (
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.KeyboardModifier.ShiftModifier,
        ):
            event.accept()
            self.fit()

        else:
            super().keyPressEvent(event)

    def signal_selected(self, name):
        signal = self.signals[name].copy()
        signal.flags &= ~signal.Flags.computed
        signal.computation = {}

        signal = PlotSignal(signal)
        if not isinstance(signal.min, str):
            self.target_max.setValue(signal.max)
            self.target_min.setValue(signal.min)

            if len(signal):
                self.samples = signal
                self.samples.trim(
                    signal.timestamps[0],
                    signal.timestamps[-1],
                    PLOT_HEIGTH,
                )
                self.samples.plot_timestamps -= self.samples.plot_timestamps[0]
                x_scale = self.samples.plot_timestamps[-1] / PLOT_HEIGTH
                self.samples.plot_timestamps /= x_scale
                self.samples.plot_timestamps += TEXT_WIDTH

            else:
                self.samples = None
        else:
            self.samples = None

        if self.samples is None:
            self.target_max.setEnabled(True)
            self.target_min.setEnabled(True)
        else:
            self.target_max.setEnabled(False)
            self.target_min.setEnabled(False)

        self.draw_plot()

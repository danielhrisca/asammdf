# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
from pyqtgraph import functions as fn
from PySide6 import QtCore, QtGui, QtWidgets

from ..ui import resource_rc
from ..ui.signal_scale import Ui_ScaleDialog

PLOT_HEIGTH = 600  # pixels
TEXT_WIDTH = 50  # pixels


class ScaleDialog(Ui_ScaleDialog, QtWidgets.QDialog):
    def __init__(self, signal, y_range, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.signal = signal

        self.setWindowTitle("Y scale configuration")

        self.y_top.setMinimum(-np.inf)
        self.y_top.setMaximum(np.inf)
        self.y_bottom.setMinimum(-np.inf)
        self.y_bottom.setMaximum(np.inf)
        self.target_min.setMinimum(-np.inf)
        self.target_min.setMaximum(np.inf)
        self.target_max.setMinimum(-np.inf)
        self.target_max.setMaximum(np.inf)
        self.offset.setMinimum(-np.inf)
        self.offset.setMaximum(np.inf)

        if not isinstance(signal.min, str):
            self.target_max.setValue(signal.max)
            self.target_min.setValue(signal.min)
            self._target_max = signal.max
            self._target_min = signal.min
        else:
            self._target_max = 0
            self._target_min = 0

        self._y_top = y_range[1]
        self._y_bottom = y_range[0]

        self.y_top.setValue(y_range[1])
        self.y_bottom.setValue(y_range[0])

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

        self.y_top.valueChanged.connect(self.set_y_value)
        self.y_bottom.valueChanged.connect(self.set_y_value)
        self.offset.valueChanged.connect(self.set_offset)
        self.target_max.valueChanged.connect(self.set_target)
        self.target_min.valueChanged.connect(self.set_target)

        canvas = QtGui.QPixmap(PLOT_HEIGTH + 3 * TEXT_WIDTH, PLOT_HEIGTH)
        canvas.fill(QtCore.Qt.black)
        self.plot.setPixmap(canvas)
        self.plot.setFocus()

        self._inhibit = False

        self.draw_plot()

    def set_y_value(self, *args):

        original_change = not self._inhibit

        if original_change:
            self._inhibit = True

            if self.y_top.value() < self.y_bottom.value():
                val = self.y_bottom.value()

                self.y_bottom.setValue(self.y_top.value())
                self.y_top.setValue(val)

            elif self.y_top.value() == self.y_bottom.value():

                self.y_top.setValue(self.y_top.value() + 1)

            if self.y_bottom.value() != self.y_top.value():
                pos = (
                    100
                    * self.y_bottom.value()
                    / (self.y_bottom.value() - self.y_top.value())
                )
                self.offset.setValue(pos)
            self._inhibit = False

        self.draw_plot()

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

    def set_offset(self, *args):
        original_change = not self._inhibit

        if original_change:
            self._inhibit = True

            offset = self.offset.value()

            range = self.y_top.value() - self.y_bottom.value()
            gain = range / 100
            self.y_bottom.setValue(-gain * offset)
            self.y_top.setValue(-gain * offset + 100 * gain)

            self._inhibit = False

        self.draw_plot()

    def apply(self, *args):
        self.accept()

    def cancel(self):
        self.reject()

    def zoom_in(self, *args):
        pos = self.offset.value()
        range = self.y_top.value() - self.y_bottom.value()
        step = range / 10
        self._inhibit = True
        self.y_top.setValue(self.y_top.value() - step)
        self.y_bottom.setValue(self.y_bottom.value() + step)
        self.offset.setValue(-12345)
        self._inhibit = False
        self.offset.setValue(pos)

    def zoom_out(self, *args):
        pos = self.offset.value()
        range = self.y_top.value() - self.y_bottom.value()
        step = range / 10
        self._inhibit = True
        self.y_top.setValue(self.y_top.value() + step)
        self.y_bottom.setValue(self.y_bottom.value() - step)
        self.offset.setValue(-12345)
        self._inhibit = False
        self.offset.setValue(pos)

    def fit(self, *args):
        self._inhibit = True
        self.y_top.setValue(self.target_max.value())
        self.y_bottom.setValue(self.target_min.value())
        if self.y_bottom.value() != self.y_top.value():
            pos = (
                100
                * self.y_bottom.value()
                / (self.y_bottom.value() - self.y_top.value())
            )
            self.offset.setValue(pos)
        self._inhibit = False
        self.draw_plot()

    def draw_plot(self, *args):
        if not self._inhibit:
            canvas = self.plot.pixmap()
            canvas.fill(QtCore.Qt.black)

            x = np.linspace(0, 2 * np.pi * 10, PLOT_HEIGTH)
            amp = (self.target_max.value() - self.target_min.value()) / 2
            off = (self.target_max.value() + self.target_min.value()) / 2
            y = np.sin(x) * amp + off
            x = np.arange(50, 50 + PLOT_HEIGTH)

            ys = self.y_top.value()
            y_scale = (self.y_top.value() - self.y_bottom.value()) / PLOT_HEIGTH

            ys = ys + y_scale

            y = (ys - y) / y_scale

            polygon = fn.create_qpolygonf(PLOT_HEIGTH)
            ndarray = fn.ndarray_from_qpolygonf(polygon)

            ndarray[:, 0] = x
            ndarray[:, 1] = y

            painter = QtGui.QPainter(canvas)
            pen = QtGui.QPen(QtCore.Qt.white)
            painter.setPen(pen)

            step = PLOT_HEIGTH // 10
            for i, x in enumerate(range(0, PLOT_HEIGTH + step, step)):
                if i == 0:
                    painter.drawText(5, x + 15, f"{100 - 10 * i}%")
                elif i == 10:
                    painter.drawText(5, x - 5, f"{100 - 10 * i}%")
                else:
                    painter.drawText(5, x + 6, f"{100-10*i}%")

            painter.drawText(PLOT_HEIGTH + TEXT_WIDTH, 15, f"{self.y_top.value():.3f}")
            painter.drawText(
                PLOT_HEIGTH + TEXT_WIDTH,
                PLOT_HEIGTH - 5,
                f"{self.y_bottom.value():.3f}",
            )

            painter.setClipping(True)
            painter.setClipRect(QtCore.QRect(TEXT_WIDTH, 0, PLOT_HEIGTH, PLOT_HEIGTH))

            pen.setStyle(QtCore.Qt.DotLine)
            painter.setPen(pen)
            for i, x in enumerate(range(0, PLOT_HEIGTH + step, step)):
                painter.drawLine(0, x, PLOT_HEIGTH + 2 * TEXT_WIDTH, x)

            pen = QtGui.QPen("#61b2e2")
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawPolyline(polygon)
            painter.end()

            self.plot.setPixmap(canvas)

    def shift(self, *args, step=1):
        self.offset.stepBy(step)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key_I and modifiers == QtCore.Qt.NoModifier:
            self.zoom_in()
        elif key == QtCore.Qt.Key_O and modifiers == QtCore.Qt.NoModifier:
            self.zoom_out()
        elif key == QtCore.Qt.Key_F and modifiers == QtCore.Qt.NoModifier:
            self.fit()
        elif key == QtCore.Qt.Key_Up and modifiers == QtCore.Qt.ShiftModifier:
            self.offset.stepUp()
        elif key == QtCore.Qt.Key_Down and modifiers == QtCore.Qt.ShiftModifier:
            self.offset.stepDown()
        elif key == QtCore.Qt.Key_PageUp and modifiers == QtCore.Qt.ShiftModifier:
            self.offset.stepBy(10)
        elif key == QtCore.Qt.Key_PageDown and modifiers == QtCore.Qt.ShiftModifier:
            self.offset.stepBy(-10)
        else:
            super().keyPressEvent(event)

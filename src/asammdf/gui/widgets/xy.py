from functools import partial

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...signal import Signal
from ..ui.xy import Ui_XYDisplay


class XY(Ui_XYDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.Signal(list)

    def __init__(
        self,
        x_channel=None,
        y_channel=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.closed = False
        self.setContentsMargins(0, 0, 0, 0)

        self._settings = QtCore.QSettings()

        self.plot = pg.PlotWidget()
        self.plot_layout.addWidget(self.plot)
        self.curve = self.plot.plot(x=[], y=[], symbol="o")
        self.marker = self.plot.plot(
            x=[],
            y=[],
            symbol="o",
            symbolPen={"color": self._settings.value("cursor_color", "#ff0000"), "width": 4},
            symbolSize=12,
        )

        self.x_search_btn.clicked.connect(partial(self.search, target="x"))
        self.y_search_btn.clicked.connect(partial(self.search, target="y"))

        self.x_channel_edit.editingFinished.connect(partial(self.search, target="x", edit=True))
        self.y_channel_edit.editingFinished.connect(partial(self.search, target="y", edit=True))

        self.show()

        self._x = None
        self._y = None
        self._timebase = None
        self._timestamp = None
        self._pen = "#00ff00"
        self._requested_channel = None

        self.set_x(x_channel)
        self.set_y(y_channel)

        self.update_plot()

    def add_new_channels(self, channels):
        channel = channels[0]
        if self._requested_channel == "x":
            self.set_x(channel)

        elif self._requested_channel == "y":
            self.set_y(channel)

        self._requested_channel = None

        self.update_plot()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key.Key_C and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            event.accept()
            color = QtWidgets.QColorDialog.getColor(self._pen, parent=self)
            if color.isValid():
                self._pen = color.name()
                self.update_plot()

        elif key in (QtCore.Qt.Key.Key_S, QtCore.Qt.Key.Key_F) and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            if self._x and self._y and len(self._y):
                min_val, max_val = self._y.samples.min(), self._y.samples.max()
                if min_val == max_val:
                    delta = 1
                else:
                    delta = 0.05 * (max_val - min_val)
                self.plot.setYRange(min_val - delta, max_val + delta, padding=0)

        elif key == QtCore.Qt.Key.Key_W and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            event.accept()
            if self._x and self._y and len(self._x):
                min_val, max_val = self._x.samples.min(), self._x.samples.max()
                if min_val == max_val:
                    delta = 1
                else:
                    delta = 0.05 * (max_val - min_val)
                self.plot.setXRange(min_val - delta, max_val + delta, padding=0)

    def search(self, *args, edit=False, target=None, **kwargs):
        self._requested_channel = target
        if edit:
            if target == "x":
                channels = [self.x_channel_edit.text().strip()]
            else:
                channels = [self.y_channel_edit.text().strip()]
        else:
            channels = []

        self.add_channels_request.emit(channels)

    def set_timestamp(self, stamp):
        self._timestamp = stamp
        if stamp is None or not len(self._timebase):
            self.marker.setData(x=[], y=[])
        else:
            idx = np.searchsorted(self._timebase, stamp, side="right") - 1

            x = self._x.samples[idx : idx + 1]
            y = self._y.samples[idx : idx + 1]

            self.marker.setData(
                x=x,
                y=y,
                symbol="o",
                symbolPen={"color": self._settings.value("cursor_color", "#ff0000"), "width": 4},
                symbolSize=12,
            )

    def set_x(self, x):
        if isinstance(x, Signal):
            self._x = x
            self.x_channel_edit.setText(x.name)
            self.plot.plotItem.setLabel("bottom", x.name, x.unit)
        else:
            self._x = None
            self.x_channel_edit.setText("")
            self.plot.plotItem.setLabel("bottom", "", "")

        self.update_plot()

    def set_y(self, y):
        if isinstance(y, Signal):
            self._y = y
            self.y_channel_edit.setText(y.name)
            self.plot.plotItem.setLabel("left", y.name, y.unit)

        else:
            self._y = None
            self.y_channel_edit.setText("")
            self.plot.plotItem.setLabel("left", "", "")

        self.update_plot()

    def to_config(self):

        config = {
            "channels": [self._x.name if self._x else "", self._y.name if self._y else ""],
        }

        return config

    def update_plot(self):
        self.plot.plotItem.getAxis("left").setPen(self._pen)
        self.plot.plotItem.getAxis("left").setTextPen(self._pen)

        x, y = self._x, self._y
        if x is None or y is None:
            self.curve.setData(x=[], y=[], pen=self._pen, symbolPen=self._pen, symbolBrush=self._pen)
            self._timebase = None
        else:
            self._timebase = t = np.unique(np.concatenate([x.timestamps, y.timestamps]))
            x = x.interp(t)
            y = y.interp(t)
            self.curve.setData(
                x=x.samples, y=y.samples, pen=self._pen, symbolPen=self._pen, symbolBrush=self._pen, symbolSize=4
            )
            self.set_timestamp(self._timestamp)

from functools import partial

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ...signal import Signal
from ..ui.xy import Ui_XYDisplay


class XY(Ui_XYDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.Signal(list)
    timestamp_changed_signal = QtCore.Signal(object, float)

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

        self.plot.plotItem.scene().sigMouseClicked.connect(self.clicked)

        self.curve = self.plot.plot(x=[], y=[], symbol="o", symbolSize=4)
        self.marker = self.plot.plot(
            x=[],
            y=[],
            symbol="o",
            symbolPen={"color": self._settings.value("cursor_color", "#ff0000"), "width": 2},
            symbolBrush=QtGui.QBrush(),
            symbolSize=10,
        )

        self.x_search_btn.clicked.connect(partial(self.search, target="x"))
        self.y_search_btn.clicked.connect(partial(self.search, target="y"))

        self.x_channel_edit.editingFinished.connect(partial(self.search, target="x", edit=True))
        self.y_channel_edit.editingFinished.connect(partial(self.search, target="y", edit=True))

        self.show()

        self.timestamp.valueChanged.connect(self._timestamp_changed)
        self.timestamp_slider.valueChanged.connect(self._timestamp_slider_changed)

        self._inhibit = False

        self._x = None
        self._y = None
        self._timebase = None
        self._timestamp = None
        self._pen = {"color": "#00ff00", "dash": [16.0, 8.0, 4.0, 4.0, 4.0, 4.0, 4.0, 16.0]}
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

    def clicked(self, event):

        scene_pos = event.scenePos()
        pos = self.plot.plotItem.vb.mapSceneToView(scene_pos)
        x = pos.x()
        y = pos.y()

        delta = (self._x.samples - x) ** 2 + (self._y.samples - y) ** 2
        idx = np.argmin(delta).flatten()[0]

        x = self._x.samples[idx : idx + 1]
        y = self._y.samples[idx : idx + 1]

        self.marker.setData(
            x=x,
            y=y,
        )

        self.timestamp_slider.setValue(idx)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key.Key_C and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            event.accept()
            color = QtWidgets.QColorDialog.getColor(self._pen, parent=self)
            if color.isValid():
                self._pen["color"] = color.name()
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
        ):
            self.timestamp_slider.keyPressEvent(event)

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

    def set_timestamp(self, stamp, emit=True):
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
            )

            self._inhibit = True
            self.timestamp_slider.setValue(idx)
            self.timestamp.setValue(stamp)
            self._inhibit = False

            if emit:
                self.timestamp_changed_signal.emit(self, stamp)

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
        self.update_timebase()

    def _timestamp_changed(self, stamp):
        if not self._inhibit:
            self.set_timestamp(stamp)

    def _timestamp_slider_changed(self, idx):
        if not self._inhibit:
            if not len(self._timebase):
                return

            self.set_timestamp(self._timebase[idx])

    def to_config(self):

        config = {
            "channels": [self._x.name if self._x else "", self._y.name if self._y else ""],
        }

        return config

    def update_plot(self):
        self.plot.plotItem.getAxis("left").setPen(self._pen["color"])
        self.plot.plotItem.getAxis("left").setTextPen(self._pen["color"])

        x, y = self._x, self._y
        if x is None or y is None:
            self.curve.setData(x=[], y=[], pen=self._pen, symbolPen=self._pen, symbolBrush=self._pen)
            self._timebase = None

        else:
            self._timebase = t = np.unique(np.concatenate([x.timestamps, y.timestamps]))
            x = x.interp(t)
            y = y.interp(t)
            self.curve.setData(
                x=x.samples,
                y=y.samples,
                pen=self._pen,
                symbolPen=self._pen["color"],
                symbolBrush=self._pen["color"],
                symbolSize=4,
                antialias=False,
            )
            self.set_timestamp(self._timestamp)

    def update_timebase(self):

        if self._timebase is not None and len(self._timebase):
            count = len(self._timebase)
            min_, max_ = self._timebase[0], self._timebase[-1]
            self.timestamp_slider.setRange(0, count - 1)

        else:
            min_, max_ = 0.0, 0.0
            self.timestamp_slider.setRange(0, 0)

        self.min_t.setText(f"{min_:.9f}s")
        self.max_t.setText(f"{max_:.9f}s")
        self.timestamp.setRange(min_, max_)
        self.set_timestamp(self._timestamp, emit=False)

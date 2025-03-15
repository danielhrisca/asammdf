from functools import partial

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ...signal import Signal
from ..ui.xy import Ui_XYDisplay

ARROW = pg.arrayToQPath(np.array([0.0, -1.0, 0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0, -1.0, 0.0]), connect="all")


class XY(Ui_XYDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.Signal(list)
    timestamp_changed_signal = QtCore.Signal(object, float)

    def __init__(
        self,
        x_channel=None,
        y_channel=None,
        color="#00ff00",
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
        self._pen = color

        # self.curve = self.plot.plot(x=[], y=[], symbol="o", symbolSize=4)
        self.curve = pg.PlotCurveItem(
            size=0,
            pen=self._pen,
            antialias=False,
        )
        self.arrows = pg.ScatterPlotItem(
            size=0,
            pen=self._pen,
            brush=self._pen,
            symbolPen=self._pen,
            symbolBrush=self._pen,
            symbolSize=4,
            antialias=False,
            useCache=False,
        )
        self.plot.addItem(self.curve)
        self.plot.addItem(self.arrows)
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
        self._x_interp = None
        self._y = None
        self._y_interp = None
        self._timebase = None
        self._timestamp = None

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

        if self._x is not None and self._y is not None:
            delta = (self._x_interp.samples - x) ** 2 + (self._y_interp.samples - y) ** 2
            idx = np.argmin(delta).flatten()[0]

            x = self._x_interp.samples[idx : idx + 1]
            y = self._y_interp.samples[idx : idx + 1]

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

    def set_timestamp(self, stamp, emit=True, spinbox=False):
        if stamp is None or self._timebase is None or not len(self._timebase):
            self.marker.setData(x=[], y=[])

        else:
            idx = np.searchsorted(self._timebase, stamp, side="right") - 1

            new_stamp = self._timebase[idx]

            if spinbox:
                if new_stamp == self._timestamp and stamp > new_stamp:
                    idx += 1
                    new_stamp = self._timebase[idx]

            stamp = new_stamp

            x = self._x_interp.samples[idx : idx + 1]
            y = self._y_interp.samples[idx : idx + 1]

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

        self._timestamp = stamp

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
        self.update_timebase()

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
            self.set_timestamp(stamp, spinbox=True)

    def _timestamp_slider_changed(self, idx):
        if not self._inhibit:
            if not len(self._timebase):
                return

            self.set_timestamp(self._timebase[idx])

    def to_config(self):
        config = {
            "channels": [self._x.name if self._x else "", self._y.name if self._y else ""],
            "color": self._pen,
        }

        return config

    def update_plot(self):
        self.plot.plotItem.getAxis("left").setPen(self._pen)
        self.plot.plotItem.getAxis("left").setTextPen(self._pen)

        x, y = self._x, self._y
        if x is None or y is None:
            self.curve.clear()
            self.arrows.clear()
            self._timebase = None

        elif not len(x) or not len(y):
            self.curve.clear()
            self.arrows.clear()
            self._timebase = None

        else:
            self._timebase = t = np.unique(np.concatenate([x.timestamps, y.timestamps]))
            self._x_interp = x = x.interp(t)
            self._y_interp = y = y.interp(t)

            transform = QtGui.QTransform()

            angles = -np.arctan2(np.diff(y.samples.astype("f8")), np.diff(x.samples.astype("f8"))) * 180 / np.pi

            exit_spots = [
                {
                    "pos": (x.samples[0], y.samples[0]),
                    "symbol": ARROW,
                    "size": 6,
                    "pen": self._pen,
                    "brush": self._pen,
                }
            ]
            for angle, xpos, ypos in zip(angles.tolist(), x.samples[1:].tolist(), y.samples[1:].tolist(), strict=False):
                transform.reset()
                angle_rot = transform.rotate(angle)
                my_rotated_symbol = angle_rot.map(ARROW)

                exit_spots.append(
                    {
                        "pos": (xpos, ypos),
                        "symbol": my_rotated_symbol,
                        "size": 6,
                        "pen": self._pen,
                        "brush": self._pen,
                    }
                )

            # add the spots to the item
            self.arrows.setData(exit_spots)
            self.curve.setData(
                x=x.samples,
                y=y.samples,
                pen=self._pen,
                antialias=False,
            )

            self.set_timestamp(self._timestamp)

    def update_timebase(self):
        if self._timebase is not None and len(self._timebase):
            count = len(self._timebase)
            min_, max_ = self._timebase[0], self._timebase[-1]
            self.timestamp_slider.setRange(0, count - 1)
            if count >= 2:
                self.timestamp.setSingleStep(0.5 * np.min(np.diff(self._timebase)))

        else:
            min_, max_ = 0.0, 0.0
            self.timestamp_slider.setRange(0, 0)

        self.min_t.setText(f"{min_:.9f}s")
        self.max_t.setText(f"{max_:.9f}s")
        self.timestamp.setRange(min_, max_)
        self.set_timestamp(self._timestamp, emit=False)

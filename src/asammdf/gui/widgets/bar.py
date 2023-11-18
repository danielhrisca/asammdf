import os
import re

import numpy as np
from numpy import searchsorted
from PySide6 import QtCore, QtGui, QtWidgets

from ..dialogs.messagebox import MessageBox
from ..ui.bar import Ui_BarDisplay
from ..utils import COLORS
from .channel_bar_display import ChannelBarDisplay
from .list_item import ListItem
from .plot import PlotSignal

OPS = {
    "!=": "__ne__",
    "==": "__eq__",
    ">": "__gt__",
    ">=": "__ge__",
    "<": "__lt__",
    "<=": "__le__",
}


class Bar(Ui_BarDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.Signal(list)
    timestamp_changed_signal = QtCore.Signal(object, float)

    def __init__(self, signals, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        for sig in signals:
            sig.timestamps = np.around(sig.timestamps, 9)
        self.signals = {}
        self._min = float("inf")
        self._max = -float("inf")
        self.add_new_channels(signals)

        self.timestamp.valueChanged.connect(self._timestamp_changed)
        self.timestamp_slider.valueChanged.connect(self._timestamp_slider_changed)

        self._timestamp = self._min
        self.set_timestamp()

        self._inhibit = False

    def items_deleted(self, names):
        for name in names:
            self.signals.pop(name)
        self.build()

    def _timestamp_changed(self, stamp):
        if not self._inhibit:
            self.set_timestamp(stamp)

    def _timestamp_slider_changed(self, stamp):
        if not self._inhibit:
            factor = stamp / 99999
            stamp = (self._max - self._min) * factor + self._min
            self.set_timestamp(stamp)

    def set_timestamp(self, stamp=None):
        if stamp is None:
            stamp = self._timestamp

        if not (self._min <= stamp <= self._max):
            return

        idx_cache = {}

        for row in range(self.channels.count()):
            item = self.channels.item(row)
            widget = self.channels.itemWidget(item)
            uuid = widget.uuid
            sig = self.signals[uuid]

            if len(sig):
                if (sig.group_index, sig.origin_uuid) in idx_cache:
                    idx = idx_cache[(sig.group_index, sig.origin_uuid)]
                else:
                    idx = min(sig.size - 1, searchsorted(sig.timestamps, stamp))
                    idx_cache[(sig.group_index, sig.origin_uuid)] = idx
                value = sig.samples[idx]

                widget.set_value(value)
                widget.bar.update()

        self._inhibit = True
        if self._min != self._max:
            val = int((stamp - self._min) / (self._max - self._min) * 99999)
            self.timestamp_slider.setValue(val)
        self.timestamp.setValue(stamp)
        self._inhibit = False
        self.timestamp_changed_signal.emit(self, stamp)

    def add_new_channels(self, channels):
        for index, sig in enumerate(channels):
            sig.uuid = os.urandom(6).hex()
            sig.computed = False
            sig.computation = None
            sig.color = COLORS[index % 10]
            sig.size = len(sig)
            sig.kind = sig.samples.dtype.kind

        invalid = []

        for channel in channels:
            diff = np.diff(channel.timestamps)
            invalid_indexes = np.argwhere(diff <= 0).ravel()
            if len(invalid_indexes):
                invalid_indexes = invalid_indexes[:10] + 1
                idx = invalid_indexes[0]
                ts = channel.timestamps[idx - 1 : idx + 2]
                invalid.append(f"{channel.name} @ index {invalid_indexes[:10] - 1} with first time stamp error: {ts}")

        if invalid:
            errors = "\n".join(invalid)
            MessageBox.warning(
                self,
                "The following channels do not have monotonous increasing time stamps:",
                f"The following channels do not have monotonous increasing time stamps:\n{errors}",
            )
            self.plot._can_trim = False

        valid = []
        invalid = []
        for channel in channels:
            if len(channel):
                samples = channel.samples
                if samples.dtype.kind not in "SUV" and np.all(np.isnan(samples)):
                    invalid.append(channel.name)
                elif channel.conversion:
                    samples = channel.physical().samples
                    if samples.dtype.kind not in "SUV" and np.all(np.isnan(samples)):
                        invalid.append(channel.name)
                    else:
                        valid.append(channel)
                else:
                    valid.append(channel)
            else:
                valid.append(channel)

        if invalid:
            MessageBox.warning(
                self,
                "All NaN channels will not be plotted:",
                f"The following channels have all NaN samples and will not be plotted:\n{', '.join(invalid)}",
            )

        channels = valid

        for sig in channels:
            sig = PlotSignal(sig)
            if len(sig):
                self._min = min(self._min, sig.timestamps[0])
                self._max = max(self._max, sig.timestamps[-1])

            item = ListItem(
                (sig.group_index, sig.channel_index),
                sig.name,
                sig.computation,
                self.channels,
                sig.origin_uuid,
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, sig.name)
            tooltip = getattr(sig, "tooltip", "") or sig.comment

            it = ChannelBarDisplay(
                sig.uuid,
                0,
                (sig.min, sig.max),
                sig.max + 1,
                sig.color,
                sig.unit,
                3,
                tooltip,
                self,
            )
            it.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)

            if sig.computed:
                font = QtGui.QFont()
                font.setItalic(True)
                it.name.setFont(font)

            it.set_name(sig.name)
            it.set_color(sig.color)
            item.setSizeHint(it.sizeHint())
            self.channels.addItem(item)
            self.channels.setItemWidget(item, it)

            self.signals[sig.uuid] = sig

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if (
            key in (QtCore.Qt.Key.Key_H, QtCore.Qt.Key.Key_B, QtCore.Qt.Key.Key_P)
            and modifier == QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            if key == QtCore.Qt.Key.Key_H:
                self.format_selection.setCurrentText("hex")
            elif key == QtCore.Qt.Key.Key_B:
                self.format_selection.setCurrentText("bin")
            else:
                self.format_selection.setCurrentText("phys")
            event.accept()
        elif key == QtCore.Qt.Key.Key_Right and modifier == QtCore.Qt.KeyboardModifier.NoModifier:
            self.timestamp_slider.setValue(self.timestamp_slider.value() + 1)
        elif key == QtCore.Qt.Key.Key_Left and modifier == QtCore.Qt.KeyboardModifier.NoModifier:
            self.timestamp_slider.setValue(self.timestamp_slider.value() - 1)
        else:
            super().keyPressEvent(event)

    def to_config(self):
        channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels)
        while item := iterator.value():
            channels.append(item.text(0))
            iterator += 1

        config = {
            "format": self.format,
            "channels": channels if not self.pattern else [],
            "pattern": self.pattern,
        }

        return config

    def search_forward(self):
        if self.op.currentIndex() < 0 or not self.target.text().strip() or not self.pattern_match.text().strip():
            self.match.setText("invalid input values")
            return

        operator = self.op.currentText()

        wildcard = f"{os.urandom(6).hex()}_WILDCARD_{os.urandom(6).hex()}"
        text = self.pattern_match.text().strip()
        pattern = text.replace("*", wildcard)
        pattern = re.escape(pattern)
        pattern = pattern.replace(wildcard, ".*")

        pattern = re.compile(f"(?i){pattern}")
        matches = [name for name in self.signals if pattern.search(name)]

        if not matches:
            self.match.setText("the pattern does not match any channel name")
            return

        try:
            target = float(self.target.text().strip())
        except:
            self.match.setText("the target must a numeric value")
        else:
            if target.is_integer():
                target = int(target)

            start = self.timestamp.value()

            timestamp = None
            signal_name = ""
            for name in matches:
                sig = self.signals[name].cut(start=start)
                samples = sig.samples

                op = getattr(samples, OPS[operator])
                try:
                    idx = np.argwhere(op(target)).flatten()
                    if len(idx):
                        if len(idx) == 1 or sig.timestamps[idx[0]] != start:
                            timestamp_ = sig.timestamps[idx[0]]
                        else:
                            timestamp_ = sig.timestamps[idx[1]]

                        if timestamp is None or timestamp_ < timestamp:
                            timestamp = timestamp_
                            signal_name = name
                except:
                    continue

            if timestamp is not None:
                self.timestamp.setValue(timestamp)
                self.match.setText(f"condition found for {signal_name}")
            else:
                self.match.setText("condition not found")

    def search_backward(self):
        if self.op.currentIndex() < 0 or not self.target.text().strip() or not self.pattern_match.text().strip():
            self.match.setText("invalid input values")
            return

        operator = self.op.currentText()

        pattern = self.pattern_match.text().strip().replace("*", "_WILDCARD_")
        pattern = re.escape(pattern)
        pattern = pattern.replace("_WILDCARD_", ".*")

        pattern = re.compile(f"(?i){pattern}")
        matches = [name for name in self.signals if pattern.search(name)]

        if not matches:
            self.match.setText("the pattern does not match any channel name")
            return

        try:
            target = float(self.target.text().strip())
        except:
            self.match.setText("the target must a numeric value")
        else:
            if target.is_integer():
                target = int(target)

            stop = self.timestamp.value()

            timestamp = None
            signal_name = ""
            for name in matches:
                sig = self.signals[name].cut(stop=stop)
                samples = sig.samples[:-1]

                op = getattr(samples, OPS[operator])
                try:
                    idx = np.argwhere(op(target)).flatten()
                    if len(idx):
                        if len(idx) == 1 or sig.timestamps[idx[-1]] != stop:
                            timestamp_ = sig.timestamps[idx[-1]]
                        else:
                            timestamp_ = sig.timestamps[idx[-2]]

                        if timestamp is None or timestamp_ > timestamp:
                            timestamp = timestamp_
                            signal_name = name
                except:
                    continue

            if timestamp is not None:
                self.timestamp.setValue(timestamp)
                self.match.setText(f"condition found for {signal_name}")
            else:
                self.match.setText("condition not found")

    def set_format(self, fmt):
        self.format = fmt
        self._settings.setValue("numeric_format", fmt)
        self._update_values()

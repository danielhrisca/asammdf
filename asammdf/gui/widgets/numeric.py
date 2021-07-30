# -*- coding: utf-8 -*-
import re

from natsort import natsorted
import numpy as np
from numpy import searchsorted
from PyQt5 import QtCore, QtWidgets

from ..ui import resource_rc as resource_rc
from ..ui.numeric import Ui_NumericDisplay
from .tree_item import TreeItem

OPS = {
    "!=": "__ne__",
    "==": "__eq__",
    ">": "__gt__",
    ">=": "__ge__",
    "<": "__lt__",
    "<=": "__le__",
}


class Numeric(Ui_NumericDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.pyqtSignal(list)
    timestamp_changed_signal = QtCore.pyqtSignal(object, float)

    def __init__(self, signals, format="phys", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        for sig in signals:
            sig.timestamps = np.around(sig.timestamps, 9)
        self.signals = {}
        self._min = self._max = 0
        self.format = format
        self.add_new_channels(signals)
        self.pattern = {}

        self.timestamp.valueChanged.connect(self._timestamp_changed)
        self.timestamp_slider.valueChanged.connect(self._timestamp_slider_changed)

        self._update_values(self.timestamp.value())
        self.channels.add_channels_request.connect(self.add_channels_request)
        self.channels.items_deleted.connect(self.items_deleted)

        self._inhibit = False

        self.forward.clicked.connect(self.search_forward)
        self.backward.clicked.connect(self.search_backward)
        self.op.addItems([">", ">=", "<", "<=", "==", "!="])

        self.format_selection.currentTextChanged.connect(self.set_format)

        self._settings = QtCore.QSettings()
        integer_mode = self._settings.value("numeric_format", "phys")
        self.format_selection.setCurrentText(integer_mode)

        self.build()

    def items_deleted(self, names):
        for name in names:
            self.signals.pop(name)
        self.build()

    def build(self):
        self.channels.setSortingEnabled(False)
        self.signals = {name: self.signals[name] for name in natsorted(self.signals)}
        self.channels.clear()
        self._min = float("inf")
        self._max = -float("inf")
        items = []

        for sig in self.signals.values():
            sig.kind = sig.samples.dtype.kind
            size = len(sig)
            sig.size = size
            if size:
                if sig.kind == "f":
                    value = f"{sig.samples[0]:.3f}"
                else:
                    value = str(sig.samples[0])
                self._min = min(self._min, sig.timestamps[0])
                self._max = max(self._max, sig.timestamps[-1])
            else:
                value = "n.a."

            item = TreeItem(
                (sig.group_index, sig.channel_index),
                sig.name,
                self.channels,
                [sig.name, value, sig.unit],
                mdf_uuid=sig.mdf_uuid,
            )
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsDropEnabled)
            items.append(item)
        self.channels.addTopLevelItems(items)

        if self._min == float("inf"):
            self._min = self._max = 0

        self.timestamp.setRange(self._min, self._max)
        self.min_t.setText(f"{self._min:.3f}s")
        self.max_t.setText(f"{self._max:.3f}s")
        self._update_values()
        self.channels.setSortingEnabled(True)

    def _timestamp_changed(self, stamp):
        val = int((stamp - self._min) / (self._max - self._min) * 9999)

        if not self._inhibit:
            self._inhibit = True
            self.timestamp_slider.setValue(val)
        else:
            self._inhibit = False

        self._update_values(stamp)
        self.timestamp_changed_signal.emit(self, stamp)

    def _timestamp_slider_changed(self, stamp):
        factor = stamp / 9999
        val = (self._max - self._min) * factor + self._min

        if not self._inhibit:
            self._inhibit = True
            self.timestamp.setValue(val)
        else:
            self._inhibit = False

    def _update_values(self, stamp=None):
        if stamp is None:
            stamp = self.timestamp.value()
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels)

        idx_cache = {}

        if self.format == "bin":

            while 1:
                item = iterator.value()
                if not item:
                    break
                sig = self.signals[item.text(0)]
                if sig.size:
                    if (sig.group_index, sig.mdf_uuid) in idx_cache:
                        idx = idx_cache[(sig.group_index, sig.mdf_uuid)]
                    else:
                        idx = min(sig.size - 1, searchsorted(sig.timestamps, stamp))
                        idx_cache[(sig.group_index, sig.mdf_uuid)] = idx
                    value = sig.samples[idx]

                    if sig.kind == "f":
                        item.setText(1, f"{value:.3f}")
                    elif sig.kind in "ui":
                        item.setText(1, bin(value))
                    else:
                        item.setText(1, str(value))

                iterator += 1
        elif self.format == "hex":

            while 1:
                item = iterator.value()
                if not item:
                    break
                sig = self.signals[item.text(0)]
                if sig.size:
                    if (sig.group_index, sig.mdf_uuid) in idx_cache:
                        idx = idx_cache[(sig.group_index, sig.mdf_uuid)]
                    else:
                        idx = min(sig.size - 1, searchsorted(sig.timestamps, stamp))
                        idx_cache[(sig.group_index, sig.mdf_uuid)] = idx
                    value = sig.samples[idx]

                    if sig.kind == "f":
                        item.setText(1, f"{value:.3f}")
                    elif sig.kind in "ui":
                        item.setText(1, f"0x{value:X}")
                    else:
                        item.setText(1, str(value))

                iterator += 1
        else:
            while 1:
                item = iterator.value()
                if not item:
                    break
                sig = self.signals[item.text(0)]
                if sig.size:
                    if (sig.group_index, sig.mdf_uuid) in idx_cache:
                        idx = idx_cache[(sig.group_index, sig.mdf_uuid)]
                    else:
                        idx = min(sig.size - 1, searchsorted(sig.timestamps, stamp))
                        idx_cache[(sig.group_index, sig.mdf_uuid)] = idx
                    value = sig.samples[idx]
                    if sig.kind == "f":
                        item.setText(1, f"{value:.3f}")
                    else:
                        item.setText(1, str(value))

                iterator += 1

        header = self.channels.header()

        index = header.sortIndicatorSection()
        order = header.sortIndicatorOrder()

        self.channels.sortByColumn(index, order)

    def add_new_channels(self, channels):
        invalid = []
        for sig in channels:
            if sig:
                if np.any(np.diff(sig.timestamps) < 0):
                    invalid.append(sig.name)
                self.signals[sig.name] = sig
        if invalid:
            QtWidgets.QMessageBox.warning(
                self,
                "The following channels do not have monotonous increasing time stamps:",
                f"The following channels do not have monotonous increasing time stamps:\n{', '.join(invalid)}",
            )
        self.build()

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if (
            key in (QtCore.Qt.Key_H, QtCore.Qt.Key_B, QtCore.Qt.Key_P)
            and modifier == QtCore.Qt.ControlModifier
        ):
            if key == QtCore.Qt.Key_H:
                self.format_selection.setCurrentText("hex")
            elif key == QtCore.Qt.Key_B:
                self.format_selection.setCurrentText("bin")
            else:
                self.format_selection.setCurrentText("phys")
            event.accept()
        elif key == QtCore.Qt.Key_Right and modifier == QtCore.Qt.NoModifier:
            self.timestamp_slider.setValue(self.timestamp_slider.value() + 1)
        elif key == QtCore.Qt.Key_Left and modifier == QtCore.Qt.NoModifier:
            self.timestamp_slider.setValue(self.timestamp_slider.value() - 1)
        else:
            super().keyPressEvent(event)

    def to_config(self):

        channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels)
        while 1:
            item = iterator.value()
            if not item:
                break
            channels.append(item.text(0))
            iterator += 1

        config = {
            "format": self.format,
            "channels": channels if not self.pattern else [],
            "pattern": self.pattern,
        }

        return config

    def search_forward(self):
        if (
            self.op.currentIndex() < 0
            or not self.target.text().strip()
            or not self.pattern_match.text().strip()
        ):
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
        if (
            self.op.currentIndex() < 0
            or not self.target.text().strip()
            or not self.pattern_match.text().strip()
        ):
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
            self.match.setText(f"the target must a numeric value")
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
                self.match.setText(f"condition not found")

    def set_format(self, fmt):
        self.format = fmt
        self._settings.setValue("numeric_format", fmt)
        self._update_values()

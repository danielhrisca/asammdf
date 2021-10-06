# -*- coding: utf-8 -*-
import re

from natsort import natsorted
import numpy as np
from numpy import searchsorted
from PyQt5 import QtCore, QtWidgets

from ...mdf import MDF
from ..ui import resource_rc as resource_rc
from ..ui.numeric import Ui_NumericDisplay
from ..utils import copy_ranges
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

    def __init__(self, signals, format=None, mode=None, float_precision=None, *args, **kwargs):
        super(QtWidgets.QWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self._settings = QtCore.QSettings()

        # for sig in signals:
        #     sig.timestamps = np.around(sig.timestamps, 9)
        self.signals = {}
        self._min = self._max = 0

        if format is None:
            format = self._settings.value("numeric_format", "phys")
        self.format_selection.setCurrentText(format)
        self.format = format

        if mode is None:
            mode = self._settings.value("numeric_mode", "scaled values")
        self.mode_selection.setCurrentText(mode)
        self.mode = mode

        if float_precision is None:
            float_precision = self._settings.value("numeric_float_precision", 3)
        self.float_precision.setValue(float_precision)

        self.add_new_channels(signals)
        self.pattern = {}

        self.timestamp.valueChanged.connect(self._timestamp_changed)
        self.timestamp_slider.valueChanged.connect(self._timestamp_slider_changed)

        self.channels.add_channels_request.connect(self.add_channels_request)
        self.channels.items_deleted.connect(self.items_deleted)

        self._inhibit = False

        self.forward.clicked.connect(self.search_forward)
        self.backward.clicked.connect(self.search_backward)
        self.op.addItems([">", ">=", "<", "<=", "==", "!="])

        self.format_selection.currentTextChanged.connect(self.set_format)
        self.mode_selection.currentTextChanged.connect(self.set_mode)
        self.float_precision.valueChanged.connect(self.set_float_precision)

        self.channels.setAlternatingRowColors(False)

        self.build()

    def items_deleted(self, names):
        for uuid_, name in names:
            self.signals.pop(name)
        self.build()

    def build(self):
        self.channels.setSortingEnabled(False)
        self.signals = {name: self.signals[name] for name in natsorted(self.signals)}
        self.channels.clear()
        self._min = float("inf")
        self._max = -float("inf")
        items = []

        mode = self.mode

        float_format = f'{{:.{self.float_precision.value()}f}}'

        for sig in self.signals.values():
            if mode == "raw values":
                sig.kind = sig.raw_samples.dtype.kind
            else:
                sig.kind = sig.phys_samples.dtype.kind
            size = len(sig)
            sig.size = size
            if size:
                if sig.kind == "f":
                    value = float_format.format(sig.samples[0])
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
                computation=sig.computation,
                ranges=sig.ranges,
            )
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsDropEnabled)
            items.append(item)
        self.channels.addTopLevelItems(items)

        if self._min == float("inf"):
            self._min = self._max = 0

        self._timestamp = self._min

        self.timestamp.setRange(self._min, self._max)
        self.min_t.setText(f"{self._min:.9f}s")
        self.max_t.setText(f"{self._max:.9f}s")
        self.set_timestamp(self._min)
        self.channels.setSortingEnabled(True)

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

        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels)

        idx_cache = {}

        float_format = f'{{:.{self.float_precision.value()}f}}'

        mode = self.mode

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
                        idx = searchsorted(sig.timestamps, stamp, side="right")
                        idx -= 1
                        if idx < 0:
                            idx = 0
                        idx_cache[(sig.group_index, sig.mdf_uuid)] = idx

                    if mode == "raw values":
                        value = sig.raw_samples[idx]
                    else:
                        value = sig.phys_samples[idx]

                    if value.dtype.kind == "f":
                        item.setText(1, float_format.format(value))
                    elif value.dtype.kind in "ui":
                        item.setText(1, bin(value))
                    else:
                        item.setText(1, str(value))

                    item.check_signal_range(value)

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
                        idx = searchsorted(sig.timestamps, stamp, side="right")
                        idx -= 1
                        if idx < 0:
                            idx = 0
                        idx_cache[(sig.group_index, sig.mdf_uuid)] = idx

                    if mode == "raw values":
                        value = sig.raw_samples[idx]
                    else:
                        value = sig.phys_samples[idx]

                    if value.dtype.kind == "f":
                        item.setText(1, float_format.format(value))
                    elif value.dtype.kind in "ui":
                        item.setText(1, f"0x{value:X}")
                    else:
                        item.setText(1, str(value))

                    item.check_signal_range(value)

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
                        idx = searchsorted(sig.timestamps, stamp, side="right")
                        idx -= 1
                        if idx < 0:
                            idx = 0
                        idx_cache[(sig.group_index, sig.mdf_uuid)] = idx

                    if mode == "raw values":
                        value = sig.raw_samples[idx]
                    else:
                        value = sig.phys_samples[idx]

                    if value.dtype.kind == "f":
                        item.setText(1, float_format.format(value))
                    else:
                        item.setText(1, str(value))

                    item.check_signal_range(value)

                iterator += 1

        header = self.channels.header()

        index = header.sortIndicatorSection()
        order = header.sortIndicatorOrder()

        self.channels.sortByColumn(index, order)

        selection = self.channels.selectedItems()
        if selection:
            self.channels.setCurrentItem(selection[0], 1, QtCore.QItemSelectionModel.SelectCurrent)
        else:
            item = self.channels.itemAt(10, 10)
            if item:
                self.channels.setCurrentItem(item, 1, QtCore.QItemSelectionModel.SelectCurrent)
            self.channels.clearSelection()

        self._inhibit = True
        if self._min != self._max:
            val = int((stamp - self._min) / (self._max - self._min) * 99999)
            self.timestamp_slider.setValue(val)
        self.timestamp.setValue(stamp)
        self._inhibit = False
        self.timestamp_changed_signal.emit(self, stamp)

    def add_new_channels(self, channels, mime_data=None):
        invalid = []
        for sig in channels:
            if sig:
                if np.any(np.diff(sig.timestamps) < 0):
                    invalid.append(sig.name)

                if sig.conversion:
                    sig.phys_samples = sig.conversion.convert(sig.samples)
                    sig.raw_samples = sig.samples
                else:
                    sig.phys_samples = sig.raw_samples = sig.samples

                self.signals[sig.name] = sig

                sig.ranges = getattr(sig, "ranges", [])

        if invalid:
            errors = ', '.join(invalid)
            try:
                mdi_title = self.parent().windowTitle()
                title = f"numeric <{mdi_title}>"
            except:
                title = "numeric window"

            QtWidgets.QMessageBox.warning(
                self,
                f"Channels with corrupted time stamps added to {title}",
                f"The following channels do not have monotonous increasing time stamps:\n{errors}",
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

        elif key == QtCore.Qt.Key_S and modifier == QtCore.Qt.ControlModifier:
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select output measurement file",
                "",
                "MDF version 4 files (*.mf4)",
            )

            if file_name:
                signals = [signal for signal in self.signals if signal.enable]
                if signals:
                    with MDF() as mdf:
                        groups = {}
                        for sig in signals:
                            id_ = id(sig.timestamps)
                            group_ = groups.setdefault(id_, [])
                            group_.append(sig)

                        for signals in groups.values():
                            sigs = []
                            for signal in signals:
                                if ":" in signal.name:
                                    sig = signal.copy()
                                    sig.name = sig.name.split(":")[-1].strip()
                                    sigs.append(sig)
                                else:
                                    sigs.append(signal)
                            mdf.append(sigs, common_timebase=True)
                        mdf.save(file_name, overwrite=True)

        elif (
            key in (QtCore.Qt.Key_R, QtCore.Qt.Key_S)
            and modifier == QtCore.Qt.AltModifier
        ):
            if key == QtCore.Qt.Key_R:
                self.mode_selection.setCurrentText("raw values")
            else:
                self.mode_selection.setCurrentText("scaled values")
        else:
            super().keyPressEvent(event)

    def to_config(self):

        channels = {}
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels)
        while 1:
            item = iterator.value()
            if not item:
                break

            ranges = copy_ranges(item.ranges)

            for range_info in ranges:
                range_info['font_color'] = range_info['font_color'].color().name()
                range_info['background_color'] = range_info['background_color'].color().name()

            channels[item.name] = ranges
            iterator += 1

        pattern = self.pattern
        if pattern:
            ranges = copy_ranges(pattern["ranges"])

            for range_info in ranges:
                range_info['font_color'] = range_info['font_color'].color().name()
                range_info['background_color'] = range_info['background_color'].color().name()

            pattern["ranges"] = ranges

        config = {
            "format": self.format,
            "mode": self.mode,
            "channels": list(channels) if not self.pattern else [],
            "ranges": list(channels.values()) if not self.pattern else [],
            "pattern": pattern,
            "float_precision": self.float_precision.value(),
            "header_sections_width": [self.channels.header().sectionSize(i) for i in range(self.channels.header().count())],
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

        mode = self.mode

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
                if mode == "raw values" or sig.comversion is None:
                    samples = sig.samples
                else:
                    samples = sig.conversion.convert(sig.samples)

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

        mode = self.mode

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
                if mode == "raw values" or sig.comversion is None:
                    samples = sig.samples[:-1]
                else:
                    samples = sig.conversion.convert(sig.samples)[:-1]

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
        self.set_timestamp()

    def set_mode(self, mode):
        self.mode = mode
        self._settings.setValue("numeric_mode", mode)
        self.set_timestamp()

    def set_float_precision(self, value):
        self._settings.setValue("numeric_float_precision", value)
        self.set_timestamp()

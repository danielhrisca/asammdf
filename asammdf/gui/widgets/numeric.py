# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from natsort import natsorted
from numpy import searchsorted

from ..ui import resource_rc as resource_rc
from ..ui.numeric import Ui_NumericDisplay


class TreeItem(QtWidgets.QTreeWidgetItem):

    def __lt__(self, otherItem):
        column = self.treeWidget().sortColumn()

        if column == 1:
            val1 = self.text(column)
            try:
                val1 = float(val1)
            except:
                pass

            val2 = otherItem.text(column)
            try:
                val2 = float(val2)
            except:
                pass

            try:
                return val1 < val2
            except:
                if isinstance(val1, float):
                    return True
                else:
                    return False
        else:
            return self.text(column) < otherItem.text(column)


class Numeric(Ui_NumericDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.pyqtSignal(list)
    timestamp_changed_signal = QtCore.pyqtSignal(object, float)

    def __init__(self, signals, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.signals = {
            sig.name: sig
            for sig in signals
        }
        self._min = self._max = 0
        self.format = 'phys'

        self.timestamp.valueChanged.connect(self._timestamp_changed)
        self.timestamp_slider.valueChanged.connect(self._timestamp_slider_changed)

        self._update_values(self.timestamp.value())
        self.channels.add_channels_request.connect(self.add_channels_request)
        self.channels.items_deleted.connect(self.items_deleted)

        self.build()

    def items_deleted(self, names):
        for name in names:
            self.signals.pop(name)
        self.build()

    def build(self):
        self.channels.setSortingEnabled(False)
        self.signals = {
            name: self.signals[name]
            for name in natsorted(self.signals)
        }
        self.channels.clear()
        self._min = float('inf')
        self._max = -float('inf')
        items = []
        for sig in self.signals.values():
            sig.kind = sig.samples.dtype.kind
            size = len(sig)
            sig.size = size
            if size:
                if sig.kind == 'f':
                    value = f'{sig.samples[0]:.3f}'
                else:
                    value = str(sig.samples[0])
                self._min = min(self._min, sig.timestamps[0])
                self._max = max(self._max, sig.timestamps[-1])
            else:
                value = 'n.a.'

            item = TreeItem([sig.name, value, sig.unit])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsDropEnabled)
            items.append(
                item
            )
        self.channels.addTopLevelItems(items)

        if self._min == float('inf'):
            self._min = self._max = 0

        self.timestamp.setRange(self._min, self._max)
        self.min_t.setText(f'{self._min:.3f}s')
        self.max_t.setText(f'{self._max:.3f}s')
        self._update_values()
        self.channels.setSortingEnabled(True)

    def _timestamp_changed(self, stamp):
        val = int((stamp - self._min) / (self._max - self._min) * 9999)
        if val != self.timestamp_slider.value():
            self.timestamp_slider.setValue(val)

        self._update_values(stamp)

        self.timestamp_changed_signal.emit(self, stamp)

    def _timestamp_slider_changed(self, stamp):
        factor = stamp / 9999
        val = (self._max - self._min) * factor + self._min
        if val != self.timestamp.value():
            self.timestamp.setValue(val)

        self._update_values(val)

    def _update_values(self, stamp=None):
        if stamp is None:
            stamp = self.timestamp.value()
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels)

        idx_cache = {}

        if self.format == 'bin':

            while 1:
                item = iterator.value()
                if not item:
                    break
                sig = self.signals[item.text(0)]
                if sig.size:
                    if sig.group_index in idx_cache:
                        idx = idx_cache[sig.group_index]
                    else:
                        idx = min(sig.size-1, searchsorted(sig.timestamps, stamp))
                        idx_cache[sig.group_index] = idx
                    value = sig.samples[idx]

                    if sig.kind == 'f':
                        item.setText(1, f'{value:.3f}')
                    elif sig.kind in 'ui':
                        item.setText(1, bin(value))
                    else:
                        item.setText(1, str(value))

                iterator += 1
        elif self.format == 'hex':

            while 1:
                item = iterator.value()
                if not item:
                    break
                sig = self.signals[item.text(0)]
                if sig.size:
                    if sig.group_index in idx_cache:
                        idx = idx_cache[sig.group_index]
                    else:
                        idx = min(sig.size-1, searchsorted(sig.timestamps, stamp))
                        idx_cache[sig.group_index] = idx
                    value = sig.samples[idx]

                    if sig.kind == 'f':
                        item.setText(1, f'{value:.3f}')
                    elif sig.kind in 'ui':
                        item.setText(1, f'0x{value:X}')
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
                    if sig.group_index in idx_cache:
                        idx = idx_cache[sig.group_index]
                    else:
                        idx = min(sig.size-1, searchsorted(sig.timestamps, stamp))
                        idx_cache[sig.group_index] = idx
                    value = sig.samples[idx]
                    if sig.kind == 'f':
                        item.setText(1, f'{value:.3f}')
                    else:
                        item.setText(1, str(value))

                iterator += 1

        header = self.channels.header()

        index = header.sortIndicatorSection()
        order = header.sortIndicatorOrder()

        self.channels.sortByColumn(index, order)

    def add_new_channels(self, channels):
        for sig in channels:
            if sig:
                self.signals[sig.name] = sig
        self.build()

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if key in (QtCore.Qt.Key_H, QtCore.Qt.Key_B, QtCore.Qt.Key_P) and modifier == QtCore.Qt.ControlModifier:
            if key == QtCore.Qt.Key_H:
                self.format = 'hex'
            elif key == QtCore.Qt.Key_B:
                self.format = 'bin'
            else:
                self.format = 'phys'
            self._update_values()
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
            'format': self.format,
            'channels': channels,
        }

        return config

# -*- coding: utf-8 -*-
from pathlib import Path

HERE = Path(__file__).resolve().parent

from ..ui import resource_qt5 as resource_rc

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import uic
from natsort import natsorted
from numpy import zeros, searchsorted

from ..utils import COLORS

class Numeric(QtWidgets.QWidget):
    add_channel_request = QtCore.pyqtSignal(str)

    def __init__(self, signals, *args, **kwargs):
        super().__init__()
        self.signals = []
        self._min = self._max = 0
        uic.loadUi(HERE.joinpath("..", "ui", "numeric.ui"), self)

        self.timestamp.valueChanged.connect(self._timestamp_changed)
        self.timestamp_slider.valueChanged.connect(self._timestamp_slider_changed)

        self._update_values(self.timestamp.value())
        self.channels.add_channel_request.connect(self.add_channel_request)

        self.add_signals(signals)

    def add_signals(self, signals):
        self.signals = natsorted(signals, key=lambda x: x.name)
        self.channels.clear()
        self._min = self._max = 0
        items = []
        for i, sig in enumerate(signals):
            if sig.samples.dtype.kind == "f":
                sig.format = "phys"
                sig.plot_texts = None
            else:
                sig.format = "phys"
                if sig.samples.dtype.kind in "SV":
                    sig.plot_texts = sig.texts = sig.samples
                    sig.samples = np.zeros(len(sig.samples))
                else:
                    sig.plot_texts = None
            color = COLORS[i % 10]
            sig.color = color
            if len(sig):
                self._min = min(self._min, sig.timestamps[0])
                self._max = max(self._max, sig.timestamps[-1])
                sig.empty = False
                value = f'{sig.samples[0]:.6f}'
            else:
                sig.empty = True
                value = 'n.a.'

            items.append(
                QtWidgets.QTreeWidgetItem([sig.name, sig.unit, value])
            )
        self.channels.addTopLevelItems(items)

        self.timestamp.setRange(self._min, self._max)
        self.min_t.setText(f'{self._min:.3f}s')
        self.max_t.setText(f'{self._max:.3f}s')
        self._update_values(self.timestamp.value())

    def _timestamp_changed(self, stamp):
        val = int((stamp - self._min) / (self._max - self._min) * 9999)
        if val != self.timestamp_slider.value():
            self.timestamp_slider.setValue(val)

        self._update_values(stamp)

    def _timestamp_slider_changed(self, stamp):
        factor = stamp / 9999
        val = (self._max - self._min) * factor + self._min
        if val != self.timestamp.value():
            self.timestamp.setValue(val)

        self._update_values(val)

    def _update_values(self, stamp):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels)

        index = 0
        while 1:
            item = iterator.value()
            if not item:
                break
            sig = self.signals[index]
            if not sig.empty:
                idx = searchsorted(sig.timestamps, stamp)
                item.setText(2, f'{sig.samples[idx]:.6f}')

            index += 1
            iterator += 1

    def add_new_channel(self, sig):
        if sig:
            self.add_signals(self.signals + [sig,])

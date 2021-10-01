# -*- coding: utf-8 -*-
from functools import partial
import logging
import webbrowser

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np

from ..ui import resource_rc as resource_rc
from ..ui.fft_window import Ui_FFTWindow
from .plot import Plot, PlotSignal, Signal
import scipy.signal as scipy_signal

bin_ = bin


if not hasattr(pg.InfiniteLine, "addMarker"):
    logger = logging.getLogger("asammdf")
    message = (
        "Old pyqtgraph package: Please install the latest pyqtgraph from the "
        "github develop branch\n"
        "pip install -I --no-deps "
        "https://github.com/pyqtgraph/pyqtgraph/archive/develop.zip"
    )
    logger.warning(message)


class FFTWindow(Ui_FFTWindow, QtWidgets.QMainWindow):
    def __init__(self, signal, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self._settings = QtCore.QSettings()
        self.with_dots = self._settings.value("dots", False, type=bool)

        self.signal = signal

        self.signal_plot = Plot([], self.with_dots)
        self.fft_plot = Plot([], self.with_dots, x_axis='frequency')

        layout = self.verticalLayout
        layout.addWidget(self.signal_plot)
        layout.addWidget(self.fft_plot)

        layout.setStretch(0, 0)
        layout.setStretch(1, 1)
        layout.setStretch(2, 1)

        self.show()
        self.signal_plot.add_new_channels([signal])

        self.signal_plot.region_removed_signal.connect(self.update)
        self.signal_plot.region_moved_signal.connect(self.update)
        self.start_frequency.valueChanged.connect(self.update)
        self.end_frequency.valueChanged.connect(self.update)
        self.frequency_step.valueChanged.connect(self.update)

        self.update()

    def update(self, *args):
        if self.signal_plot.plot.region:
            start, stop = self.signal_plot.plot.region.getRegion()
            signal = self.signal.cut(start, stop)
        else:
            signal = self.signal

        start_frequency = self.start_frequency.value()
        end_frequency = self.end_frequency.value()
        frequency_step = self.frequency_step.value()
        if start_frequency > end_frequency:
            start_frequency, end_frequency = end_frequency, start_frequency
        steps = int((end_frequency - start_frequency) / frequency_step)

        self.fft_plot.clear()

        if len(self.signal) and steps:
            f = np.linspace(start_frequency, end_frequency, steps)
            pgram = scipy_signal.lombscargle(signal.timestamps, signal.samples, f * 2 * np.pi, normalize=True)

            signal = Signal(
                samples=pgram,
                timestamps=f,
                name=f'{self.signal.name}_FFT'
            )
            signal.color = self.signal.color
            signal.computed = False
            signal.computation = {}

            signal = PlotSignal(signal)

            self.fft_plot.add_new_channels([signal])


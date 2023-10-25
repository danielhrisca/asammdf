import numpy as np
from PySide6 import QtCore, QtWidgets
import scipy.signal as scipy_signal

from ..ui.fft_window import Ui_FFTWindow
from .plot import Plot, PlotSignal, Signal


class FFTWindow(Ui_FFTWindow, QtWidgets.QMainWindow):
    def __init__(self, signal, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self._settings = QtCore.QSettings()
        self.with_dots = self._settings.value("dots", False, type=bool)

        self.signal = signal

        self.signal_plot = Plot([], self.with_dots)
        self.fft_plot = Plot([], self.with_dots, x_axis="frequency")

        layout = self.layout
        layout.addWidget(self.signal_plot)
        layout.addWidget(self.fft_plot)

        self.show()
        self.signal_plot.add_new_channels([signal])

        self.signal_plot.region_removed_signal.connect(self.update)
        self.signal_plot.region_moved_signal.connect(self.update)
        self.start_frequency.valueChanged.connect(self.update)
        self.end_frequency.valueChanged.connect(self.update)
        self.frequency_step.valueChanged.connect(self.update)

        self.setWindowTitle(f"{self.signal.name} FFT using Lomb-Scargle periodogram")

        self.update(initial=True)

    def update(self, *args, initial=False):
        xrange, yrange = self.fft_plot.plot.viewbox.viewRange()

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

            signal = Signal(samples=pgram, timestamps=f, name=f"{self.signal.name}_FFT")
            signal.color = self.signal.color
            signal.computed = False
            signal.computation = {}

            signal = PlotSignal(signal)

            self.fft_plot.add_new_channels([signal])

        if not initial:
            self.fft_plot.plot.viewbox.setXRange(*xrange, padding=0)
            self.fft_plot.plot.viewbox.setYRange(*yrange, padding=0)

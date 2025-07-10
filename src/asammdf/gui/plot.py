import logging

from ..blocks.utils import plausible_timestamps
from ..signal import Signal

try:
    from PySide6 import QtWidgets

    from .widgets.plot_standalone import PlotWindow

    QT = True
except ImportError:
    QT = False


logger = logging.getLogger("asammdf")


def plot(signals: Signal | list[Signal], title: str = "", validate: bool = True) -> None:
    """Create a stand-alone plot using the input signal or signals.

    Arguments
    ---------
    signals : iterable | Signal
        Signals to plot.
    title : str, default ''
        Window title.
    validate : bool, default True
        Consider the invalidation bits.
    """

    if QT:
        app = QtWidgets.QApplication([])
        app.setOrganizationName("py-asammdf")
        app.setOrganizationDomain("py-asammdf")
        app.setApplicationName("py-asammdf")

        if validate:
            if isinstance(signals, (tuple, list)):
                signals = [signal.validate() for signal in signals]
            else:
                signals = [signals.validate()]

        for signal in signals:
            all_ok, idx = plausible_timestamps(signal.timestamps, -1e6, 1e9)
            if not all_ok:
                signal.samples = signal.samples[idx]
                signal.timestamps = signal.timestamps[idx]

        main = PlotWindow({sig.name: sig for sig in signals})
        if title.strip():
            main.setWindowTitle(title.strip())
        else:
            if isinstance(signals, (tuple, list)):
                main.setWindowTitle(", ".join(sig.name for sig in signals))
            else:
                main.setWindowTitle(signals.name)

        app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

        app.exec_()

    else:
        raise Exception("Signal plotting requires pyqtgraph or matplotlib")

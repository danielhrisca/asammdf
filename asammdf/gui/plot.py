# -*- coding: utf-8 -*-
import logging

try:
    from PyQt5 import QtWidgets
    from .widgets.plot_standalone import PlotWindow

    QT = True
except ImportError:
    QT = False


logger = logging.getLogger("asammdf")


def plot(signals, title="", validate=True):
    """ create a stand-alone plot using the input signal or signals

    Arguments
    ---------
    signals : iterable | Signal

    title (""): str
        window title

    validate (True): bool
        consider the invalidation bits

    """

    if QT:
        app = QtWidgets.QApplication([])
        app.setOrganizationName("py-asammdf")
        app.setOrganizationDomain("py-asammdf")
        app.setApplicationName("py-asammdf")

        if validate:
            if isinstance(signals, (tuple, list)):
                signals = [
                    signal.validate()
                    for signal in signals
                ]
            else:
                signals = [signals.validate()]
        main = PlotWindow(signals)
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
        logging.warning("Signal plotting requires pyqtgraph or matplotlib")
        raise Exception("Signal plotting requires pyqtgraph or matplotlib")

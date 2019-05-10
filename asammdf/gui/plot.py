# -*- coding: utf-8 -*-
import logging

try:
    from PyQt5 import QtWidgets
    from .widgets.plot_standalone import PlotWindow
    QT = True
except ImportError:
    QT = False


logger = logging.getLogger("asammdf")


def plot(signals):
    """ create a stand-alone plot using the input signal or signals """

    if QT:
        app = QtWidgets.QApplication([])
        app.setOrganizationName("py-asammdf")
        app.setOrganizationDomain("py-asammdf")
        app.setApplicationName("py-asammdf")
        main = PlotWindow(signals)

        app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

        app.exec_()

    else:
        logging.warning("Signal plotting requires pyqtgraph or matplotlib")
        raise Exception("Signal plotting requires pyqtgraph or matplotlib")

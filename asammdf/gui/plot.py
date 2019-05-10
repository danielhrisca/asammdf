# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets

from .widgets.plot_standalone import PlotWindow


def plot(signals):
    """ create a stand-alone plot using the input signal or signals """
    app = QtWidgets.QApplication([])
    app.setOrganizationName("py-asammdf")
    app.setOrganizationDomain("py-asammdf")
    app.setApplicationName("py-asammdf")
    main = PlotWindow(signals)

    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    app.exec_()

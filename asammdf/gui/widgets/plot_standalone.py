# -*- coding: utf-8 -*-
import logging

import numpy as np

from ..ui import resource_rc as resource_rc


bin_ = bin

try:
    import pyqtgraph as pg

    from PyQt5 import QtGui
    from PyQt5 import QtWidgets
    from PyQt5 import QtCore

    from .channel_stats import ChannelStats
    from .plot import Plot

    if not hasattr(pg.InfiniteLine, "addMarker"):
        logger = logging.getLogger("asammdf")
        message = (
            "Old pyqtgraph package: Please install the latest pyqtgraph from the "
            "github develop branch\n"
            "pip install -I --no-deps "
            "https://github.com/pyqtgraph/pyqtgraph/archive/develop.zip"
        )
        logger.warning(message)

    class StandalonePlot(QtWidgets.QWidget):
        def __init__(self, signals, with_dots, step_mode, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.splitter = QtWidgets.QSplitter(self)
            self.splitter.setOrientation(QtCore.Qt.Horizontal)
            self.info = None

            self.plot = Plot(signals, with_dots)

            vbox = QtWidgets.QVBoxLayout()

            vbox.addWidget(self.plot, 1)

            hbox = QtWidgets.QHBoxLayout()

            for icon, description in (
                (":/cursor.png", "C - Cursor"),
                (":/fit.png", "F - Fit"),
                (":/grid.png", "G - Grid"),
                (":/home.png", "H - Home"),
                (":/zoom-in.png", "I - Zoom-in"),
                (":/zoom-out.png", "O - Zoom-out"),
                (":/info.png", "M - Statistics"),
                (":/range.png", "R - Range"),
                (":/right.png", "← - Move cursor left"),
                (":/left.png", "→ - Move cursor right"),
            ):
                label = QtWidgets.QLabel("")
                label.setPixmap(QtGui.QPixmap(icon).scaled(QtCore.QSize(16, 16)))

                hbox.addWidget(label)
                label = QtWidgets.QLabel(description)
                hbox.addWidget(label)
                hbox.addStretch()

            vbox.addLayout(hbox, 0)
            self.setLayout(vbox)

            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.setWindowIcon(icon)

            self.show()


except ImportError:
    PYQTGRAPH_AVAILABLE = False

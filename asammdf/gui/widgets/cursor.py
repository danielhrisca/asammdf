# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui
import pyqtgraph as pg


class Cursor(pg.InfiniteLine):
    def __init__(self, cursor_unit="s", *args, **kwargs):

        super().__init__(
            *args,
            label=f"{{value:.6f}}{cursor_unit}",
            labelOpts={"position": 0.04},
            **kwargs,
        )

        self.addMarker("^", 0)
        self.addMarker("v", 1)

        self._settings = QtCore.QSettings()
        if self._settings.value("plot_background") == "White":
            self.label.setColor(QtGui.QColor(0, 59, 126))
        else:
            self.label.setColor(QtGui.QColor("#ffffff"))

        self.label.show()

    def set_value(self, value):
        self.setPos(value)

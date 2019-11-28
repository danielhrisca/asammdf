# -*- coding: utf-8 -*-
import logging
from functools import partial
import webbrowser
import sys

import numpy as np

from ..ui import resource_rc as resource_rc


bin_ = bin


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


class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, signals, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.menubar = QtWidgets.QMenuBar()
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)

        self._settings = QtCore.QSettings()
        self._light_palette = self.palette()

        menu = QtWidgets.QMenu("Settings", self.menubar)
        self.menubar.addMenu(menu)

        # search mode menu
        plot_background_option = QtWidgets.QActionGroup(self)

        for option in ("Black", "White"):

            action = QtWidgets.QAction(option, menu)
            action.setCheckable(True)
            plot_background_option.addAction(action)
            action.triggered.connect(partial(self.set_plot_background, option))

            if option == self._settings.value("plot_background", "Black"):
                action.setChecked(True)
                action.triggered.emit()

        submenu = QtWidgets.QMenu("Plot background", self.menubar)
        submenu.addActions(plot_background_option.actions())
        menu.addMenu(submenu)

        # search mode menu
        theme_option = QtWidgets.QActionGroup(self)

        for option in ("Dark", "Light"):

            action = QtWidgets.QAction(option, menu)
            action.setCheckable(True)
            theme_option.addAction(action)
            action.triggered.connect(partial(self.set_theme, option))

            if option == self._settings.value("theme", "Light"):
                action.setChecked(True)
                action.triggered.emit()

        submenu = QtWidgets.QMenu("Theme", self.menubar)
        submenu.addActions(theme_option.actions())
        menu.addMenu(submenu)

        # plot option menu
        plot_actions = QtWidgets.QActionGroup(self)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/fit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action = QtWidgets.QAction(icon, f"{'Fit trace': <20}\tF", menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_F))
        action.setShortcut(QtCore.Qt.Key_F)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/grid.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action = QtWidgets.QAction(icon, "{: <20}\tG".format("Grid"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_G))
        action.setShortcut(QtCore.Qt.Key_G)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/home.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action = QtWidgets.QAction(icon, "{: <20}\tH".format("Home"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_H))
        action.setShortcut(QtCore.Qt.Key_H)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/list2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        action = QtWidgets.QAction(icon, "{: <20}\tS".format("Stack"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_S))
        action.setShortcut(QtCore.Qt.Key_S)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/zoom-in.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        action = QtWidgets.QAction(icon, "{: <20}\tI".format("Zoom in"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_I))
        action.setShortcut(QtCore.Qt.Key_I)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/zoom-out.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        action = QtWidgets.QAction(icon, "{: <20}\tO".format("Zoom out"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_O))
        action.setShortcut(QtCore.Qt.Key_O)
        plot_actions.addAction(action)

        action = QtWidgets.QAction("{: <20}\t.".format("Toggle dots"), menu)
        action.triggered.connect(partial(self.toggle_dots, key=QtCore.Qt.Key_O))
        action.setShortcut(QtCore.Qt.Key_Period)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/plus.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action = QtWidgets.QAction(
            icon, "{: <20}\tIns".format("Insert computation"), menu
        )
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_Insert))
        action.setShortcut(QtCore.Qt.Key_Insert)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action = QtWidgets.QAction(
            icon, "{: <20}\tCtrl+S".format("Save active subplot channels"), menu
        )
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key_S,
                modifier=QtCore.Qt.ControlModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        plot_actions.addAction(action)

        # values display

        display_format_actions = QtWidgets.QActionGroup(self)

        action = QtWidgets.QAction("{: <20}\tCtrl+H".format("Hex"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key_H,
                modifier=QtCore.Qt.ControlModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        display_format_actions.addAction(action)

        action = QtWidgets.QAction("{: <20}\tCtrl+B".format("Bin"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key_B,
                modifier=QtCore.Qt.ControlModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        display_format_actions.addAction(action)

        action = QtWidgets.QAction("{: <20}\tCtrl+P".format("Physical"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key_P,
                modifier=QtCore.Qt.ControlModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Ctrl+P"))
        display_format_actions.addAction(action)

        # info

        info = QtWidgets.QActionGroup(self)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action = QtWidgets.QAction(icon, "{: <20}\tM".format("Statistics"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_M))
        action.setShortcut(QtGui.QKeySequence("M"))
        info.addAction(action)

        # cursors
        cursors_actions = QtWidgets.QActionGroup(self)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/cursor.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        action = QtWidgets.QAction(icon, "{: <20}\tC".format("Cursor"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_C))
        action.setShortcut(QtCore.Qt.Key_C)
        cursors_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/right.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        action = QtWidgets.QAction(icon, "{: <20}\t←".format("Move cursor left"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_Left))
        action.setShortcut(QtCore.Qt.Key_Left)
        cursors_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/left.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        action = QtWidgets.QAction(icon, "{: <20}\t→".format("Move cursor right"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_Right))
        action.setShortcut(QtCore.Qt.Key_Right)
        cursors_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/range.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        action = QtWidgets.QAction(icon, "{: <20}\tR".format("Range"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key_R))
        action.setShortcut(QtCore.Qt.Key_R)
        cursors_actions.addAction(action)

        self.plot_menu = QtWidgets.QMenu("Plot", self.menubar)
        self.plot_menu.addActions(plot_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(cursors_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(display_format_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(info.actions())
        self.menubar.addMenu(self.plot_menu)

        menu = self.menubar.addMenu("Help")
        open_group = QtWidgets.QActionGroup(self)
        action = QtWidgets.QAction("Online documentation", menu)
        action.triggered.connect(self.help)
        open_group.addAction(action)
        menu.addActions(open_group.actions())

        self.with_dots = self._settings.value("dots", False, type=bool)

        if not isinstance(signals, (list, tuple)):
            signals = [
                signals,
            ]

        self.plot = Plot(signals, self.with_dots)

        self.setCentralWidget(self.plot)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        self.show()

    def plot_action(self, key, modifier=QtCore.Qt.NoModifier):
        event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, modifier)
        self.plot.keyPressEvent(event)

    def toggle_dots(self, key):
        self.with_dots = not self.with_dots
        self._settings.setValue("dots", self.with_dots)

        self.plot.plot.update_lines(with_dots=self.with_dots)

    def set_plot_background(self, option):
        self._settings.setValue("plot_background", option)
        if option == "Black":
            pg.setConfigOption("background", "k")
            pg.setConfigOption("foreground", "w")
        else:
            pg.setConfigOption("background", "w")
            pg.setConfigOption("foreground", "k")

    def set_theme(self, option):
        self._settings.setValue("theme", option)
        app = QtWidgets.QApplication.instance()
        if option == "Light":
            app.setPalette(self._light_palette)
        else:

            palette = QtGui.QPalette()
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
            brush = QtGui.QBrush(QtGui.QColor(82, 82, 82))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
            brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
            brush = QtGui.QBrush(QtGui.QColor(36, 36, 36))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)

            brush = QtGui.QBrush(QtGui.QColor(100, 100, 100))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
            brush = QtGui.QBrush(QtGui.QColor(82, 82, 82))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
            brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
            brush = QtGui.QBrush(QtGui.QColor(36, 36, 36))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(
                QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush
            )
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)

            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
            brush = QtGui.QBrush(QtGui.QColor(82, 82, 82))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
            brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
            brush = QtGui.QBrush(QtGui.QColor(36, 36, 36))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(
                QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush
            )
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
            app.setPalette(palette)

    def help(self, event):
        webbrowser.open_new(r"http://asammdf.readthedocs.io/en/master/gui.html")

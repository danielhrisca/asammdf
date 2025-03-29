from functools import partial
import logging
import webbrowser

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .plot import Plot, PlotSignal

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


class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, signals, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.menubar = QtWidgets.QMenuBar()
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)

        self._settings = QtCore.QSettings()
        self.with_dots = self._settings.value("dots", False, type=bool)

        if isinstance(signals, (list, tuple)):
            signals = {sig.name: PlotSignal(sig) for sig in signals}

        self.plot = Plot({}, self.with_dots)

        self._light_palette = self.palette()

        menu = QtWidgets.QMenu("Settings", self.menubar)
        self.menubar.addMenu(menu)

        # search mode menu
        plot_background_option = QtGui.QActionGroup(self)

        for option in ("Black", "White"):
            action = QtGui.QAction(option, menu)
            action.setCheckable(True)
            plot_background_option.addAction(action)
            action.triggered.connect(partial(self.set_plot_background, option))

            if option == self._settings.value("plot_background", "Black"):
                action.setChecked(True)
                action.triggered.emit()

        submenu = QtWidgets.QMenu("Plot background", self.menubar)
        submenu.addActions(plot_background_option.actions())
        menu.addMenu(submenu)

        # plot X axis display mode
        plot_xaxis_option = QtGui.QActionGroup(self)

        for option in ("seconds", "time", "date"):
            action = QtGui.QAction(option, menu)
            action.setCheckable(True)
            plot_xaxis_option.addAction(action)
            action.triggered.connect(partial(self.set_plot_xaxis, option))

            if option == self._settings.value("plot_xaxis", "seconds"):
                action.setChecked(True)
                action.triggered.emit()

        submenu = QtWidgets.QMenu("Plot X axis", self.menubar)
        submenu.addActions(plot_xaxis_option.actions())
        menu.addMenu(submenu)

        # search mode menu
        theme_option = QtGui.QActionGroup(self)

        for option in ("Dark", "Light"):
            action = QtGui.QAction(option, menu)
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
        plot_actions = QtGui.QActionGroup(self)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/fit.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, f"{'Fit trace': <20}\tF", menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_F))
        action.setShortcut(QtCore.Qt.Key.Key_F)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/grid.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tG".format("Grid"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_G))
        action.setShortcut(QtCore.Qt.Key.Key_G)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/home.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tH".format("Home"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_H))
        action.setShortcut(QtCore.Qt.Key.Key_H)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/list2.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tS".format("Stack"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_S))
        action.setShortcut(QtCore.Qt.Key.Key_S)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/zoom-in.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tI".format("Zoom in"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_I))
        action.setShortcut(QtCore.Qt.Key.Key_I)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/zoom-out.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tO".format("Zoom out"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_O))
        action.setShortcut(QtCore.Qt.Key.Key_O)
        plot_actions.addAction(action)

        action = QtGui.QAction("{: <20}\t.".format("Toggle dots"), menu)
        action.triggered.connect(partial(self.toggle_dots, key=QtCore.Qt.Key.Key_O))
        action.setShortcut(QtCore.Qt.Key.Key_Period)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/plus.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tIns".format("Insert computation"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_Insert))
        action.setShortcut(QtCore.Qt.Key.Key_Insert)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/save.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tCtrl+S".format("Save active subplot channels"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key.Key_S,
                modifier=QtCore.Qt.KeyboardModifier.ControlModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        plot_actions.addAction(action)

        # values display

        display_format_actions = QtGui.QActionGroup(self)

        action = QtGui.QAction("{: <20}\tCtrl+H".format("Hex"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key.Key_H,
                modifier=QtCore.Qt.KeyboardModifier.ControlModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        display_format_actions.addAction(action)

        action = QtGui.QAction("{: <20}\tCtrl+B".format("Bin"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key.Key_B,
                modifier=QtCore.Qt.KeyboardModifier.ControlModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        display_format_actions.addAction(action)

        action = QtGui.QAction("{: <20}\tCtrl+P".format("Physical"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key.Key_P,
                modifier=QtCore.Qt.KeyboardModifier.ControlModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Ctrl+P"))
        display_format_actions.addAction(action)

        # scaled display

        samples_format_actions = QtGui.QActionGroup(self)

        action = QtGui.QAction("{: <20}\tAlt+R".format("Raw samples"), menu)
        action.triggered.connect(
            partial(self.plot_action, key=QtCore.Qt.Key.Key_R, modifier=QtCore.Qt.KeyboardModifier.AltModifier)
        )
        action.setShortcut(QtGui.QKeySequence("Alt+R"))
        samples_format_actions.addAction(action)

        action = QtGui.QAction("{: <20}\tAlt+S".format("Scaled samples"), menu)
        action.triggered.connect(
            partial(self.plot_action, key=QtCore.Qt.Key.Key_S, modifier=QtCore.Qt.KeyboardModifier.AltModifier)
        )
        action.setShortcut(QtGui.QKeySequence("Alt+S"))
        samples_format_actions.addAction(action)

        # info

        info = QtGui.QActionGroup(self)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tM".format("Statistics"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_M))
        action.setShortcut(QtGui.QKeySequence("M"))
        info.addAction(action)

        # cursors
        cursors_actions = QtGui.QActionGroup(self)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/cursor.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tC".format("Cursor"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_C))
        action.setShortcut(QtCore.Qt.Key.Key_C)
        cursors_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/right.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\t←".format("Move cursor left"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_Left))
        action.setShortcut(QtCore.Qt.Key.Key_Left)
        cursors_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/left.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\t→".format("Move cursor right"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_Right))
        action.setShortcut(QtCore.Qt.Key.Key_Right)
        cursors_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/range.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tR".format("Range"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_R))
        action.setShortcut(QtCore.Qt.Key.Key_R)
        cursors_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/lock_range.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tY".format("Lock/unlock range"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_Y))
        action.setShortcut(QtCore.Qt.Key.Key_Y)
        cursors_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/comments.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tCtrl+I".format("Insert cursor comment"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key.Key_I,
                modifier=QtCore.Qt.KeyboardModifier.ControlModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Ctrl+I"))
        cursors_actions.addAction(action)

        icon = QtGui.QIcon()
        action = QtGui.QAction("{: <20}\tAlt+I".format("Toggle trigger texts"), menu)
        action.triggered.connect(
            partial(self.plot_action, key=QtCore.Qt.Key.Key_I, modifier=QtCore.Qt.KeyboardModifier.AltModifier)
        )
        action.setShortcut(QtGui.QKeySequence("Alt+I"))
        cursors_actions.addAction(action)

        self.plot_menu = QtWidgets.QMenu("Plot", self.menubar)
        self.plot_menu.addActions(plot_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(cursors_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(display_format_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(samples_format_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(info.actions())
        self.menubar.addMenu(self.plot_menu)

        menu = self.menubar.addMenu("Help")
        open_group = QtGui.QActionGroup(self)
        action = QtGui.QAction("Online documentation", menu)
        action.triggered.connect(self.help)
        open_group.addAction(action)
        menu.addActions(open_group.actions())

        self.setCentralWidget(self.plot)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.setWindowIcon(icon)

        self.show()
        self.plot.add_new_channels(signals)

    def plot_action(self, key, modifier=QtCore.Qt.KeyboardModifier.NoModifier):
        event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, modifier)
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

    def set_plot_xaxis(self, option):
        self._settings.setValue("plot_xaxis", option)
        if option == "seconds":
            fmt = "phys"
        elif option == "time":
            fmt = "time"
        elif option == "date":
            fmt = "time"

        plot = self.plot

        plot.plot.x_axis.format = fmt
        plot.plot.x_axis.updateAutoSIPrefix()
        if plot.plot.cursor1 is not None:
            plot.cursor_moved()
        if plot.plot.region is not None:
            plot.range_modified()

    def set_theme(self, option):
        self._settings.setValue("theme", option)
        app = QtWidgets.QApplication.instance()
        if option == "Light":
            app.setPalette(self._light_palette)
        else:
            palette = QtGui.QPalette()
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.WindowText, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Button, brush)
            brush = QtGui.QBrush(QtGui.QColor(82, 82, 82))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Light, brush)
            brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Midlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Dark, brush)
            brush = QtGui.QBrush(QtGui.QColor(36, 36, 36))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Mid, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Text, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.BrightText, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.ButtonText, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Base, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Window, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Shadow, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.AlternateBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.ToolTipBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.ToolTipText, brush)

            brush = QtGui.QBrush(QtGui.QColor(100, 100, 100))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Highlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.WindowText, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Button, brush)
            brush = QtGui.QBrush(QtGui.QColor(82, 82, 82))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Light, brush)
            brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Midlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Dark, brush)
            brush = QtGui.QBrush(QtGui.QColor(36, 36, 36))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Mid, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Text, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.BrightText, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.ButtonText, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Base, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Window, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Shadow, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.AlternateBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.ToolTipBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.ToolTipText, brush)

            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.WindowText, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Button, brush)
            brush = QtGui.QBrush(QtGui.QColor(82, 82, 82))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Light, brush)
            brush = QtGui.QBrush(QtGui.QColor(68, 68, 68))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Midlight, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Dark, brush)
            brush = QtGui.QBrush(QtGui.QColor(36, 36, 36))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Mid, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.BrightText, brush)
            brush = QtGui.QBrush(QtGui.QColor(27, 27, 27))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Base, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Window, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Shadow, brush)
            brush = QtGui.QBrush(QtGui.QColor(55, 55, 55))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.AlternateBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ToolTipBase, brush)
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ToolTipText, brush)
            app.setPalette(palette)

    def help(self, event):
        webbrowser.open_new(r"http://asammdf.readthedocs.io/en/master/gui.html")

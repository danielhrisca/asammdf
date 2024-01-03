from functools import partial
import gc
import os
from pathlib import Path
import platform
import sys
from textwrap import wrap
import webbrowser

from natsort import natsorted
import pyqtgraph as pg
from PySide6 import __version__ as pyside6_version
from PySide6 import QtCore, QtGui, QtWidgets

from ...version import __version__ as libversion
from ..dialogs.bus_database_manager import BusDatabaseManagerDialog
from ..dialogs.dependencies_dlg import DependenciesDlg
from ..dialogs.functions_manager import FunctionsManagerDialog
from ..dialogs.messagebox import MessageBox
from ..dialogs.multi_search import MultiSearch
from ..ui.main_window import Ui_PyMDFMainWindow
from ..utils import draw_color_icon
from .batch import BatchWidget
from .file import FileWidget
from .mdi_area import MdiAreaWidget, WithMDIArea
from .plot import Plot


class MainWindow(WithMDIArea, Ui_PyMDFMainWindow, QtWidgets.QMainWindow):
    def __init__(self, files=None, *args, **kwargs):
        super(Ui_PyMDFMainWindow, self).__init__(*args, **kwargs)
        WithMDIArea.__init__(self, comparison=True)
        self.setupUi(self)
        self._settings = QtCore.QSettings()
        self._settings.setValue("current_theme", self._settings.value("theme", "Light"))
        self._light_palette = self.palette()

        self.ignore_value2text_conversions = self._settings.value("ignore_value2text_conversions", False, type=bool)

        self.display_cg_name = self._settings.value("display_cg_name", False, type=bool)

        self.integer_interpolation = int(self._settings.value("integer_interpolation", "2 - hybrid interpolation")[0])

        self.float_interpolation = int(self._settings.value("float_interpolation", "1 - linear interpolation")[0])

        self.batch = BatchWidget(
            self.ignore_value2text_conversions,
            self.integer_interpolation,
            self.float_interpolation,
        )
        self.stackedWidget.addWidget(self.batch)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        widget.setLayout(layout)

        multi_search = QtWidgets.QPushButton("Search")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/search.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        multi_search.setIcon(icon)
        multi_search.clicked.connect(self.comparison_search)

        multi_info = QtWidgets.QPushButton("Measurements information")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        multi_info.setIcon(icon)
        multi_info.clicked.connect(self.comparison_info)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(multi_search)
        hbox.addWidget(multi_info)
        hbox.addStretch()

        self.mdi_area = MdiAreaWidget(self)
        self.mdi_area.add_window_request.connect(self.add_window)
        self.mdi_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.mdi_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        layout.addLayout(hbox)
        layout.addWidget(self.mdi_area)

        self.stackedWidget.addWidget(widget)
        self.stackedWidget.setCurrentIndex(0)

        self.progress = None

        self.files.tabCloseRequested.connect(self.close_file)
        self.stackedWidget.currentChanged.connect(self.mode_changed)

        menu = self.menubar.addMenu("File")
        open_group = QtGui.QActionGroup(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/open.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        action = QtGui.QAction(icon, "Open", menu)
        action.triggered.connect(self.open)
        action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        open_group.addAction(action)

        action = QtGui.QAction(icon, "Open folder", menu)
        action.triggered.connect(self.open_folder)
        open_group.addAction(action)

        menu.addActions(open_group.actions())

        menu.addSeparator()

        open_group = QtGui.QActionGroup(self)
        action = QtGui.QAction(icon, "Open configuration", menu)
        action.triggered.connect(self.open_configuration)
        open_group.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/save.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "Save configuration", menu)
        action.triggered.connect(self.save_configuration)
        open_group.addAction(action)

        menu.addActions(open_group.actions())

        # mode_actions
        mode_actions = QtGui.QActionGroup(self)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/file.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}".format("Single files"), menu)
        action.triggered.connect(partial(self.stackedWidget.setCurrentIndex, 0))
        mode_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/list.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}".format("Batch processing"), menu)
        action.triggered.connect(partial(self.stackedWidget.setCurrentIndex, 1))
        mode_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/compare.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}".format("Comparison"), menu)
        action.triggered.connect(partial(self.stackedWidget.setCurrentIndex, 2))
        mode_actions.addAction(action)

        menu = QtWidgets.QMenu("Mode", self.menubar)
        menu.addActions(mode_actions.actions())
        self.menubar.addMenu(menu)

        # managers
        actions = QtGui.QActionGroup(self)

        action = QtGui.QAction("{: <20}\tF6".format("Functions manager"), menu)
        action.triggered.connect(self.functions_manager)
        action.setShortcut("F6")
        actions.addAction(action)

        action = QtGui.QAction("{: <20}".format("Bus database manager"), menu)
        action.triggered.connect(self.bus_database_manager)
        actions.addAction(action)

        menu = QtWidgets.QMenu("Managers", self.menubar)
        menu.addActions(actions.actions())
        self.menubar.addMenu(menu)

        # settings

        menu = QtWidgets.QMenu("Settings", self.menubar)
        self.menubar.addMenu(menu)

        # sub plots
        subplot_action = QtGui.QAction("Sub-windows", menu)
        subplot_action.setCheckable(True)

        state = self._settings.value("subplots", True, type=bool)
        subplot_action.toggled.connect(self.set_subplot_option)
        subplot_action.triggered.connect(self.set_subplot_option)
        subplot_action.setChecked(state)
        menu.addAction(subplot_action)

        # Link sub-windows X-axis
        subplot_action = QtGui.QAction("Link sub-windows X-axis", menu)
        subplot_action.setCheckable(True)
        state = self._settings.value("subplots_link", True, type=bool)
        subplot_action.toggled.connect(self.set_subplot_link_option)
        subplot_action.setChecked(state)
        menu.addAction(subplot_action)

        # Ignore value2text conversions
        subplot_action = QtGui.QAction("Ignore value2text conversions", menu)
        subplot_action.setCheckable(True)
        subplot_action.toggled.connect(self.set_ignore_value2text_conversions_option)
        subplot_action.setChecked(self.ignore_value2text_conversions)
        menu.addAction(subplot_action)

        # Show Channel Group Name
        subplot_action = QtGui.QAction("Display Channel Group Name", menu)
        subplot_action.setCheckable(True)
        subplot_action.toggled.connect(self.set_display_cg_name_option)
        subplot_action.setChecked(self.display_cg_name)
        menu.addAction(subplot_action)

        # plot background
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

        # theme menu
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

        # step line connect menu
        step_option = QtGui.QActionGroup(self)

        for option in ("line", "left", "right"):
            icon = QtGui.QIcon()
            icon.addPixmap(
                QtGui.QPixmap(f":/{option}_interconnect.png"),
                QtGui.QIcon.Mode.Normal,
                QtGui.QIcon.State.Off,
            )
            action = QtGui.QAction(icon, option, menu)
            action.setCheckable(True)
            step_option.addAction(action)
            action.triggered.connect(partial(self.set_line_interconnect, option))

            if option == self._settings.value("line_interconnect", "line"):
                action.setChecked(True)
                action.triggered.emit()

        submenu = QtWidgets.QMenu("Step mode", self.menubar)
        submenu.addActions(step_option.actions())
        menu.addMenu(submenu)

        # integer interpolation menu
        theme_option = QtGui.QActionGroup(self)

        for option, tooltip in zip(
            (
                "0 - repeat previous sample",
                "1 - linear interpolation",
                "2 - hybrid interpolation",
            ),
            (
                "",
                "",
                "channels with integer data type (raw values) that have a conversion that outputs float "
                "values will use linear interpolation, otherwise the previous sample is used",
            ),
        ):
            action = QtGui.QAction(option, menu)
            action.setCheckable(True)
            if tooltip:
                action.setToolTip(tooltip)
            theme_option.addAction(action)
            action.triggered.connect(partial(self.set_integer_interpolation, option))

            if option == self._settings.value("integer_interpolation", "2 - hybrid interpolation"):
                action.setChecked(True)
                action.triggered.emit()

        submenu = QtWidgets.QMenu("Integer interpolation", self.menubar)
        submenu.addActions(theme_option.actions())
        submenu.setToolTipsVisible(True)
        menu.addMenu(submenu)

        # float interpolation menu
        theme_option = QtGui.QActionGroup(self)

        for option in ("0 - repeat previous sample", "1 - linear interpolation"):
            action = QtGui.QAction(option, menu)
            action.setCheckable(True)
            theme_option.addAction(action)
            action.triggered.connect(partial(self.set_float_interpolation, option))

            if option == self._settings.value("float_interpolation", "1 - linear interpolation"):
                action.setChecked(True)
                action.triggered.emit()

        submenu = QtWidgets.QMenu("Float interpolation", self.menubar)
        submenu.addActions(theme_option.actions())
        submenu.setToolTipsVisible(True)
        menu.addMenu(submenu)

        submenu = QtWidgets.QMenu("Cursor", self.menubar)

        action = QtGui.QAction("Color")
        action.triggered.connect(partial(self.edit_cursor_options, action=action))
        color = self._settings.value("cursor_color", "white")
        icon = draw_color_icon(color)
        action.setIcon(icon)
        submenu.addAction(action)

        action = QtWidgets.QWidgetAction(submenu)
        action.setText("Line width")
        combo = QtWidgets.QComboBox()
        combo.addItems([f"{size}pixels width" for size in range(1, 5)])
        combo.currentIndexChanged.connect(partial(self.edit_cursor_options, action=action))
        action.setDefaultWidget(combo)

        submenu.addAction(action)

        action = QtGui.QAction("Show circle")
        action.setCheckable(True)
        action.toggled.connect(partial(self.edit_cursor_options, action=action))
        action.setChecked(self._settings.value("show_cursor_circle", False, type=bool))
        submenu.addAction(action)

        action = QtGui.QAction("Show horizontal line")
        action.setCheckable(True)
        action.toggled.connect(partial(self.edit_cursor_options, action=action))
        action.setChecked(self._settings.value("show_cursor_horizontal_line", False, type=bool))
        submenu.addAction(action)

        menu.addMenu(submenu)

        self.edit_cursor_options()

        # plot option menu
        plot_actions = QtGui.QActionGroup(self)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/fit.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        fullscreen = QtGui.QAction(icon, f"{'Fullscreen': <20}\tF11", menu)
        fullscreen.triggered.connect(self.toggle_fullscreen)
        fullscreen.setShortcut(QtCore.Qt.Key.Key_F11)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/fit.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, f"{'Fit all': <20}\tF", menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_F))
        action.setShortcut(QtCore.Qt.Key.Key_F)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/fit.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, f"{'Fit selected': <20}\tShift+F", menu)
        action.triggered.connect(
            partial(self.plot_action, key=QtCore.Qt.Key.Key_F, modifier=QtCore.Qt.KeyboardModifier.ShiftModifier)
        )
        action.setShortcut(QtGui.QKeySequence("Shift+F"))
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/grid.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tG".format("Grid"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_G))
        action.setShortcut(QtCore.Qt.Key.Key_G)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/axis.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tH".format("Honeywell"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_H))
        action.setShortcut(QtCore.Qt.Key.Key_H)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/home.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tW".format("Home"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_W))
        action.setShortcut(QtCore.Qt.Key.Key_W)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/list2.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tS".format("Stack all"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_S))
        action.setShortcut(QtCore.Qt.Key.Key_S)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/list2.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tShift+S".format("Stack selected"), menu)
        action.triggered.connect(
            partial(self.plot_action, key=QtCore.Qt.Key.Key_S, modifier=QtCore.Qt.KeyboardModifier.ShiftModifier)
        )
        action.setShortcut(QtGui.QKeySequence("Shift+S"))
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

        action = QtGui.QAction("{: <20}\tX".format("Zoom to range"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_X))
        action.setShortcut(QtCore.Qt.Key.Key_X)
        plot_actions.addAction(action)

        action = QtGui.QAction("{: <20}\t.".format("Toggle dots"), menu)
        action.triggered.connect(partial(self.toggle_dots, key=QtCore.Qt.Key.Key_Period))
        action.setShortcut(QtCore.Qt.Key.Key_Period)
        plot_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/focus.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\t2".format("Focused mode"), menu)
        action.triggered.connect(partial(self.plot_action, key=QtCore.Qt.Key.Key_2))
        action.setShortcut(QtCore.Qt.Key.Key_2)
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

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/save.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tCtrl+Shift+S".format("Save all subplot channels"), menu)
        action.triggered.connect(self.save_all_subplots)
        action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        plot_actions.addAction(action)

        # channel shifting

        channel_shift_actions = QtGui.QActionGroup(self)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/shift_left.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tShift+←".format("Shift channels left"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key.Key_Left,
                modifier=QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Shift+Left"))
        channel_shift_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/shift_right.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tShift+→".format("Shift channels right"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key.Key_Right,
                modifier=QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Shift+Right"))
        channel_shift_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/shift_up.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tShift+↑".format("Shift channels up"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key.Key_Up,
                modifier=QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Shift+Up"))
        channel_shift_actions.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/shift_down.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tShift+↓".format("Shift channels down"), menu)
        action.triggered.connect(
            partial(
                self.plot_action,
                key=QtCore.Qt.Key.Key_Down,
                modifier=QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
        )
        action.setShortcut(QtGui.QKeySequence("Shift+Down"))
        channel_shift_actions.addAction(action)

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

        # sub_plots

        subs = QtGui.QActionGroup(self)

        action = QtGui.QAction("{: <20}\tShift+C".format("Cascade sub-windows"), menu)
        action.triggered.connect(partial(self.show_sub_windows, mode="cascade"))
        action.setShortcut(QtGui.QKeySequence("Shift+C"))
        subs.addAction(action)

        action = QtGui.QAction("{: <20}\tShift+T".format("Tile sub-windows in a grid"), menu)
        action.triggered.connect(partial(self.show_sub_windows, mode="tile"))
        action.setShortcut(QtGui.QKeySequence("Shift+T"))
        subs.addAction(action)

        action = QtGui.QAction("{: <20}\tShift+V".format("Tile sub-windows vertically"), menu)
        action.triggered.connect(partial(self.show_sub_windows, mode="tile vertically"))
        action.setShortcut(QtGui.QKeySequence("Shift+V"))
        subs.addAction(action)

        action = QtGui.QAction("{: <20}\tShift+H".format("Tile sub-windows horizontally"), menu)
        action.triggered.connect(partial(self.show_sub_windows, mode="tile horizontally"))
        action.setShortcut(QtGui.QKeySequence("Shift+H"))
        subs.addAction(action)

        action = QtGui.QAction("{: <20}\tShift+Alt+F".format("Toggle sub-windows frames"), menu)
        action.triggered.connect(self.toggle_frames)
        action.setShortcut(QtGui.QKeySequence("Shift+Alt+F"))
        subs.addAction(action)

        action = QtGui.QAction("{: <20}\tShift+L".format("Toggle channel list"), menu)
        action.triggered.connect(self.toggle_channels_list)
        action.setShortcut(QtGui.QKeySequence("Shift+L"))
        subs.addAction(action)

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
        icon.addPixmap(QtGui.QPixmap(":/bookmark.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tCtrl+I".format("Insert bookmark"), menu)
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
        icon.addPixmap(QtGui.QPixmap(":/bookmark.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "{: <20}\tAlt+I".format("Toggle bookmarks"), menu)
        action.triggered.connect(
            partial(self.plot_action, key=QtCore.Qt.Key.Key_I, modifier=QtCore.Qt.KeyboardModifier.AltModifier)
        )
        action.setShortcut(QtGui.QKeySequence("Alt+I"))
        cursors_actions.addAction(action)

        self.plot_menu = QtWidgets.QMenu("Plot", self.menubar)
        self.plot_menu.addAction(fullscreen)
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(plot_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(channel_shift_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(cursors_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(display_format_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(samples_format_actions.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(subs.actions())
        self.plot_menu.addSeparator()
        self.plot_menu.addActions(info.actions())
        self.menubar.addMenu(self.plot_menu)

        menu = self.menubar.addMenu("Help")

        open_group = QtGui.QActionGroup(self)
        action = QtGui.QAction("Dependencies", menu)
        action.triggered.connect(partial(DependenciesDlg.show_dependencies, "asammdf"))
        open_group.addAction(action)
        action = QtGui.QAction("Online documentation", menu)
        action.triggered.connect(self.help)
        action.setShortcut(QtGui.QKeySequence("F1"))
        open_group.addAction(action)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        action = QtGui.QAction(icon, "About asammdf-gui", menu)
        action.triggered.connect(self.show_about)
        open_group.addAction(action)
        open_group.addAction(action)

        menu.addActions(open_group.actions())

        self.with_dots = self._settings.value("dots", False, type=bool)
        self.setWindowTitle(f"asammdf {libversion} [PID={os.getpid()}] - Single files")

        self.set_subplot_option(self._settings.value("subplots", "Disabled"))
        self.set_subplot_link_option(self._settings.value("subplots_link", "Disabled"))
        self.hide_missing_channels = False
        self.hide_disabled_channels = False

        self.allways_accept_dots = False

        if files:
            for name in files:
                self._open_file(name)

        self.setAcceptDrops(True)

        self.show()
        self.fullscreen = None

    def sizeHint(self):
        return QtCore.QSize(1, 1)

    def help(self, event):
        webbrowser.open_new(r"http://asammdf.readthedocs.io/en/master/gui.html")

    def save_all_subplots(self, key):
        if self.stackedWidget.currentIndex() == 0:
            widget = self.files.currentWidget()
        elif self.stackedWidget.currentIndex() == 2:
            widget = self
        widget = self.files.currentWidget()
        widget.save_all_subplots()

    def plot_action(self, key, modifier=QtCore.Qt.KeyboardModifier.NoModifier):
        event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, modifier)

        if self.stackedWidget.currentIndex() == 0:
            widget = self.files.currentWidget()
            if widget and widget.get_current_widget():
                widget.get_current_widget().keyPressEvent(event)
        elif self.stackedWidget.currentIndex() == 2:
            widget = self
            if widget and widget.get_current_widget():
                widget.get_current_widget().keyPressEvent(event)

        event.accept()

    def toggle_dots(self, key):
        file_widget = self.files.currentWidget()

        if file_widget:
            widget = file_widget.get_current_widget()
            if widget and isinstance(widget, Plot):
                new_setting_has_dots = not file_widget.with_dots

                current_plot = widget
                count = len(current_plot.plot.signals)

                self.with_dots = new_setting_has_dots
                self._settings.setValue("dots", self.with_dots)
                file_widget.set_line_style(with_dots=new_setting_has_dots)
                self.set_line_style(with_dots=self.with_dots)
        else:
            widget = self.get_current_widget()
            if widget and isinstance(widget, Plot):
                self.with_dots = not self.with_dots
                self._settings.setValue("dots", self.with_dots)
                self.set_line_style(with_dots=self.with_dots)

    def show_sub_windows(self, mode):
        if self.stackedWidget.currentIndex() == 0:
            widget = self.files.currentWidget()
            if widget:
                if mode == "tile":
                    widget.mdi_area.tileSubWindows()
                elif mode == "cascade":
                    widget.mdi_area.cascadeSubWindows()
                elif mode == "tile vertically":
                    widget.mdi_area.tile_vertically()
                elif mode == "tile horizontally":
                    widget.mdi_area.tile_horizontally()

        else:
            widget = self
            if widget:
                if mode == "tile":
                    widget.mdi_area.tileSubWindows()
                elif mode == "cascade":
                    widget.mdi_area.cascadeSubWindows()
                elif mode == "tile vertically":
                    widget.mdi_area.tile_vertically()
                elif mode == "tile horizontally":
                    widget.mdi_area.tile_horizontally()

    def edit_cursor_options(self, checked=None, action=None):
        if action:
            if action.text() == "Color":
                color = self._settings.value("cursor_color", "white")
                color = QtWidgets.QColorDialog.getColor(color)
                if not color.isValid():
                    return

                self._settings.setValue("cursor_color", color.name())

                icon = draw_color_icon(color)
                action.setIcon(icon)

            elif action.text() == "Show circle":
                self._settings.setValue("show_cursor_circle", action.isChecked())

            elif action.text() == "Show horizontal line":
                self._settings.setValue("show_cursor_horizontal_line", action.isChecked())

            elif action.text() == "Line width":
                self._settings.setValue("cursor_line_width", action.defaultWidget().currentIndex() + 1)

        cursor_circle = self._settings.value("show_cursor_circle", False, type=bool)
        cursor_horizontal_line = self._settings.value("show_cursor_horizontal_line", False, type=bool)
        cursor_line_width = self._settings.value("cursor_line_width", 1, type=int)
        cursor_color = self._settings.value("cursor_color", "#e69138")

        for i in range(self.files.count()):
            file = self.files.widget(i)
            file.set_cursor_options(cursor_circle, cursor_horizontal_line, cursor_line_width, cursor_color)

    def set_subplot_option(self, state):
        if isinstance(state, str):
            state = True if state == "true" else False
        self.set_subplots(state)
        self._settings.setValue("subplots", state)

        count = self.files.count()

        for i in range(count):
            self.files.widget(i).set_subplots(state)

    def set_plot_background(self, option):
        self._settings.setValue("plot_background", option)
        if option == "Black":
            pg.setConfigOption("background", "k")
            pg.setConfigOption("foreground", "w")
        else:
            pg.setConfigOption("background", "w")
            pg.setConfigOption("foreground", "k")

    def set_integer_interpolation(self, option):
        self._settings.setValue("integer_interpolation", option)

        option = int(option[0])
        self.integer_interpolation = option

        count = self.files.count()

        for i in range(count):
            self.files.widget(i).mdf.configure(integer_interpolation=option)

        self.batch.integer_interpolation = option

    def set_float_interpolation(self, option):
        self._settings.setValue("float_interpolation", option)

        option = int(option[0])
        self.float_interpolation = option

        count = self.files.count()

        for i in range(count):
            self.files.widget(i).mdf.configure(float_interpolation=option)

        self.batch.float_interpolation = option

    def set_plot_xaxis(self, option):
        self._settings.setValue("plot_xaxis", option)
        if option == "seconds":
            fmt = "phys"
        elif option == "time":
            fmt = "time"
        elif option == "date":
            fmt = "date"

        if self.stackedWidget.currentIndex() == 0:
            widget = self.files.currentWidget()
        elif self.stackedWidget.currentIndex() == 2:
            widget = self
        else:
            widget = None
        if widget:
            plot = widget.get_current_widget()
            if plot and isinstance(plot, Plot):
                widget.get_current_widget().plot.x_axis.format = fmt
                widget.get_current_widget().plot.x_axis.updateAutoSIPrefix()
                if plot.plot.cursor1 is not None:
                    plot.cursor_moved()
                if plot.plot.region is not None:
                    plot.range_modified(plot.plot.region)

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
            brush = QtGui.QBrush(QtGui.QColor(100, 100, 100))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.PlaceholderText, brush)
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
            brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
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
            brush = QtGui.QBrush(QtGui.QColor(100, 100, 100))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.Highlight, brush)
            app.setPalette(palette)

    def set_line_interconnect(self, option):
        self._settings.setValue("line_interconnect", option)

        self.line_interconnect = option

        count = self.files.count()

        for i in range(count):
            self.files.widget(i).set_line_interconnect(option)

    def set_subplot_link_option(self, state):
        if isinstance(state, str):
            state = True if state == "true" else False
        self.set_subplots_link(state)
        self._settings.setValue("subplots_link", state)
        count = self.files.count()

        for i in range(count):
            self.files.widget(i).set_subplots_link(self.subplots_link)

    def set_ignore_value2text_conversions_option(self, state):
        if isinstance(state, str):
            state = True if state == "true" else False
        self.ignore_value2text_conversions = state
        self._settings.setValue("ignore_value2text_conversions", state)
        count = self.files.count()

        for i in range(count):
            self.files.widget(i).ignore_value2text_conversions = state
        self.batch.ignore_value2text_conversions = state

    def set_display_cg_name_option(self, state):
        if isinstance(state, str):
            state = True if state == "true" else False
        self.display_cg_name = state
        self._settings.setValue("display_cg_name", state)
        count = self.files.count()

        for i in range(count):
            self.files.widget(i).display_cg_name = state
            if self.files.widget(i).isVisible():
                self.files.widget(i).update_all_channel_trees()

        self.batch.display_cg_name = state

    def update_progress(self, current_index, max_index):
        self.progress = current_index, max_index

    def open_batch_files(self, event):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select measurement file",
            "",
            "CSV (*.csv);;MDF v3 (*.dat *.mdf);;MDF v4(*.mf4);;DL3/ERG files (*.dl3 *.erg);;All files (*.csv *.dat *.mdf *.mf4 *.dl3 *.erg)",
            "All files (*.csv *.dat *.mdf *.mf4 *.dl3 *.erg)",
        )

        if file_names:
            self.batch.files_list.addItems(natsorted(file_names))
            count = self.batch.files_list.count()

            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/file.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

            for row in range(count):
                self.batch.files_list.item(row).setIcon(icon)

    def open(self, event):
        if self.stackedWidget.currentIndex() in (0, 2):
            self.open_file(event)
            self.stackedWidget.setCurrentIndex(0)
        else:
            self.open_batch_files(event)

    def _open_file(self, file_name):
        file_name = Path(file_name)
        index = self.files.count()

        try:
            widget = FileWidget(
                file_name,
                self.with_dots,
                self.subplots,
                self.subplots_link,
                self.ignore_value2text_conversions,
                self.display_cg_name,
                self.line_interconnect,
                1,
                None,
                None,
                self,
            )

        except:
            raise
        else:
            widget.mdf.configure(integer_interpolation=self.integer_interpolation)
            self.files.addTab(widget, file_name.name)
            self.files.setTabToolTip(index, str(file_name))
            self.files.setCurrentIndex(index)
            widget.open_new_file.connect(self._open_file)
            widget.full_screen_toggled.connect(self.toggle_fullscreen)

            self.edit_cursor_options()

    def open_file(self, event):
        system = platform.system().lower()
        if system == "linux":
            # see issue #567
            # file extension is case sensitive on linux
            file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                "Select measurement file",
                self._settings.value("last_opened_path", "", str),
                "CSV (*.csv);;MDF v3 (*.dat *.mdf);;MDF v4(*.mf4 *.mf4z);;DL3/ERG files (*.dl3 *.erg);;All files (*.csv *.dat *.mdf *.mf4 *.mf4z *.dl3 *.erg)",
                "All files (*.csv *.dat *.mdf *.mf4 *.mf4z *.dl3 *.erg)",
                options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
            )
        else:
            file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                "Select measurement file",
                self._settings.value("last_opened_path", "", str),
                "CSV (*.csv);;MDF v3 (*.dat *.mdf);;MDF v4(*.mf4 *.mf4z);;DL3/ERG files (*.dl3 *.erg);;All files (*.csv *.dat *.mdf *.mf4 *.mf4z *.dl3 *.erg)",
                "All files (*.csv *.dat *.mdf *.mf4 *.mf4z *.dl3 *.erg)",
            )

        if file_names:
            self._settings.setValue("last_opened_path", file_names[0])
            gc.collect()

        for file_name in natsorted(file_names):
            self._open_file(file_name)

    def open_folder(self, event):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select folder",
            "",
            QtWidgets.QFileDialog.Option.ShowDirsOnly | QtWidgets.QFileDialog.Option.DontResolveSymlinks,
        )
        if not folder:
            return

        gc.collect()

        if self.stackedWidget.currentIndex() == 0:
            for root, dirs, files in os.walk(folder):
                for file in natsorted(files):
                    if file.lower().endswith((".csv", ".erg", ".dl3", ".dat", ".mdf", ".mf4", ".mf4z")):
                        self._open_file(os.path.join(root, file))
        else:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/file.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

            self.batch._ignore = True

            for root, dirs, files in os.walk(folder):
                for file in natsorted(files):
                    if file.lower().endswith((".csv", ".erg", ".dl3", ".dat", ".mdf", ".mf4", ".mf4z")):
                        item = QtWidgets.QListWidgetItem(icon, os.path.join(root, file))
                        self.batch.files_list.addItem(item)

            self.batch._ignore = False
            self.batch.update_channel_tree()

    def close_file(self, index):
        widget = self.files.widget(index)
        if widget:
            widget.close()
            widget.setParent(None)
            widget.deleteLater()

        if self.files.count():
            self.files.setCurrentIndex(0)

    def closeEvent(self, event):
        count = self.files.count()
        for i in range(count):
            self.files.widget(i).close()
        if self.fullscreen:
            widget, index = self.fullscreen
            widget.close()
            widget.deleteLater()
        event.accept()

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        try:
            if self.stackedWidget.currentIndex() == 0:
                for path in e.mimeData().text().splitlines():
                    path = Path(path.replace(r"file:///", ""))
                    if path.suffix.lower() in (
                        ".csv",
                        ".zip",
                        ".erg",
                        ".dat",
                        ".mdf",
                        ".mf4",
                        ".mf4z",
                    ):
                        self._open_file(path)
            else:
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(":/file.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

                for path in e.mimeData().text().splitlines():
                    path = Path(path.replace(r"file:///", ""))
                    if path.suffix.lower() in (
                        ".csv",
                        ".zip",
                        ".erg",
                        ".dat",
                        ".mdf",
                        ".mf4",
                        ".mf4z",
                    ):
                        row = self.batch.files_list.count()
                        self.batch.files_list.addItem(str(path))
                        self.batch.files_list.item(row).setIcon(icon)
        except:
            pass

    def mode_changed(self, index):
        if index == 0:
            self.plot_menu.setEnabled(True)
            self.setWindowTitle(f"asammdf {libversion} [PID={os.getpid()}] - Single files")
        elif index == 1:
            self.plot_menu.setEnabled(False)
            self.setWindowTitle(f"asammdf {libversion} [PID={os.getpid()}] - Batch processing")
        elif index == 2:
            self.plot_menu.setEnabled(True)
            self.setWindowTitle(f"asammdf {libversion} [PID={os.getpid()}] - Comparison")

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if key == QtCore.Qt.Key.Key_F and modifier == QtCore.Qt.KeyboardModifier.ControlModifier:
            if self.files.count() and self.stackedWidget.currentIndex() == 0:
                self.files.currentWidget().keyPressEvent(event)
            elif self.files.count() and self.stackedWidget.currentIndex() == 2:
                event.accept()
                count = self.files.count()
                channels_dbs = [self.files.widget(i).mdf.channels_db for i in range(count)]
                measurements = [str(self.files.widget(i).mdf.name) for i in range(count)]

                dlg = MultiSearch(channels_dbs, measurements, parent=self)
                dlg.setModal(True)
                dlg.exec_()
                result = dlg.result
                if result:
                    ret, ok = QtWidgets.QInputDialog.getItem(
                        None,
                        "Select window type",
                        "Type:",
                        ["Plot", "Numeric", "Tabular"],
                        0,
                        False,
                    )
                    if ok:
                        names = []
                        for file_index, entry in result:
                            group, ch_index = entry
                            mdf = self.files.widget(file_index).mdf
                            uuid = self.files.widget(file_index).uuid
                            name = mdf.groups[group].channels[ch_index].name
                            names.append(
                                {
                                    "name": name,
                                    "origin_uuid": uuid,
                                    "type": "channel",
                                    "ranges": [],
                                    "group_index": group,
                                    "channel_index": ch_index,
                                    "uuid": os.urandom(6),
                                    "computed": False,
                                    "computation": {},
                                    "precision": 3,
                                    "common_axis": False,
                                    "individual_axis": False,
                                }
                            )
                        self.add_window((ret, names))

        elif key == QtCore.Qt.Key.Key_F11:
            self.toggle_fullscreen()
            event.accept()

        elif key in (QtCore.Qt.Key.Key_F2, QtCore.Qt.Key.Key_F3, QtCore.Qt.Key.Key_F4):
            if self.files.count() and self.stackedWidget.currentIndex() == 0:
                if key == QtCore.Qt.Key.Key_F2:
                    window_type = "Plot"
                elif key == QtCore.Qt.Key.Key_F3:
                    window_type = "Numeric"
                elif key == QtCore.Qt.Key.Key_F4:
                    window_type = "Tabular"
                self.files.currentWidget()._create_window(None, window_type)
            event.accept()

        else:
            super().keyPressEvent(event)

    def comparison_search(self, event):
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_F, QtCore.Qt.KeyboardModifier.ControlModifier
        )
        self.keyPressEvent(event)

    def comparison_info(self, event):
        count = self.files.count()
        measurements = [str(self.files.widget(i).mdf.name) for i in range(count)]

        info = []
        for i, name in enumerate(measurements, 1):
            info.extend(wrap(f"{i:> 2}: {name}", 120))

        MessageBox.information(self, "Measurement files used for comparison", "\n".join(info))

    def toggle_fullscreen(self):
        if self.files.count() > 0 or self.fullscreen is not None:
            if self.fullscreen is None:
                index = self.files.currentIndex()
                widget = self.files.widget(index)
                if widget:
                    widget.setParent(None)
                    widget.showFullScreen()
                    widget.autofit_sub_plots()
                    self.fullscreen = widget, index
            else:
                widget, index = self.fullscreen
                file_name = str(Path(widget.mdf.name).name)
                self.files.insertTab(index, widget, file_name)
                self.files.setTabToolTip(index, str(widget.mdf.name))
                self.files.setCurrentIndex(index)
                self.fullscreen = None
                self.activateWindow()

                widget.autofit_sub_plots()

                self.with_dots = widget.with_dots
                self._settings.setValue("dots", self.with_dots)

                count = self.files.count()

                for i in range(count):
                    self.files.widget(i).set_line_style(with_dots=self.with_dots)

    def toggle_frames(self, event=None):
        count = self.files.count()

        for i in range(count):
            self.files.widget(i).toggle_frames()

    def toggle_channels_list(self, event=None):
        if self.stackedWidget.currentIndex() == 0:
            widget = self.files.currentWidget()
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_L, QtCore.Qt.KeyboardModifier.ShiftModifier
            )
            if widget:
                widget.keyPressEvent(event)

    def open_configuration(self, event=None):
        if self.stackedWidget.currentIndex() == 0:
            widget = self.files.currentWidget()
            if widget:
                widget.load_channel_list()

    def save_configuration(self, event=None):
        if self.stackedWidget.currentIndex() == 0:
            widget = self.files.currentWidget()
            if widget:
                widget.save_channel_list()

    def functions_manager(self):
        if self.stackedWidget.currentIndex() == 0:
            file = self.files.currentWidget()
            if file:
                channels = {}
                mdf = file.mdf
                for name, entries in mdf.channels_db.items():
                    gp_index, ch_index = entries[0]
                    comment = mdf.groups[gp_index].channels[ch_index].comment

                    channels[name] = comment

                dlg = FunctionsManagerDialog(file.functions, channels, parent=self)
                dlg.setModal(True)
                dlg.exec_()

                if dlg.pressed_button == "apply":
                    original_definitions = dlg.original_definitions
                    modified_definitions = dlg.modified_definitions

                    file.update_functions(original_definitions, modified_definitions)

    def bus_database_manager(self):
        dlg = BusDatabaseManagerDialog(parent=self)
        dlg.setModal(True)
        dlg.exec_()

        if dlg.pressed_button == "apply":
            dlg.store()

    def show_about(self):
        bits = "x86" if sys.maxsize < 2**32 else "x64"
        cpython = ".".join(str(e) for e in sys.version_info[:3])
        cpython = f"{cpython} {bits}"

        MessageBox.about(
            self,
            "About asammdf-gui",
            f"""Graphical user interface for the asammdf package

* * *

Build information:

*   version {libversion}
*   PySide6 {pyside6_version}
*   CPython {cpython}

  

Copyright © 2018-2023 Daniel Hrisca""",
            markdown=True,
        )

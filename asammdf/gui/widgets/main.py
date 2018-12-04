# -*- coding: utf-8 -*-
import os
from functools import partial

try:
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5 import uic
    from ..ui import resource_qt5 as resource_rc

    QT = 5

except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    from PyQt4 import uic
    from ..ui import resource_qt4 as resource_rc

    QT = 4

from ...mdf import MDF, SUPPORTED_VERSIONS
from ...version import __version__ as libversion
from ..utils import TERMINATED, run_thread_with_progress, setup_progress
from ..dialogs.advanced_search import AdvancedSearch
from .list import ListWidget
from .file import FileWidget

HERE = os.path.dirname(os.path.realpath(__file__))


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):

        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "..", "ui", "main_window.ui"), self)

        self.progress = None

        self.files.tabCloseRequested.connect(self.close_file)

        self.concatenate.toggled.connect(self.function_select)
        self.cs_btn.clicked.connect(self.cs_clicked)
        self.cs_format.insertItems(0, SUPPORTED_VERSIONS)
        self.cs_compression.insertItems(
            0, ("no compression", "deflate", "transposed deflate")
        )
        self.cs_split_size.setValue(10)

        self.files_list = ListWidget(self)
        self.files_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.files_layout.addWidget(self.files_list, 0, 0, 1, 2)
        self.files_list.itemDoubleClicked.connect(self.delete_item)

        self.statusbar.addPermanentWidget(QLabel("asammdf {}".format(libversion)))

        menu = self.menubar.addMenu("File")
        open_group = QActionGroup(self)
        icon = QIcon()
        icon.addPixmap(QPixmap(":/open.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "Open", menu)
        action.triggered.connect(self.open)
        open_group.addAction(action)
        menu.addActions(open_group.actions())

        menu = QMenu("Settings", self.menubar)
        self.menubar.addMenu(menu)

        # memory option menu
        memory_option = QActionGroup(self)

        for option in ("full", "low", "minimum"):

            action = QAction(option, menu)
            action.setCheckable(True)
            memory_option.addAction(action)
            action.triggered.connect(partial(self.set_memory_option, option))

            if option == "minimum":
                action.setChecked(True)

        submenu = QMenu("Memory", self.menubar)
        submenu.addActions(memory_option.actions())
        menu.addMenu(submenu)

        # graph option menu
        memory_option = QActionGroup(self)

        for option in ("Simple", "With dots"):

            action = QAction(option, menu)
            action.setCheckable(True)
            memory_option.addAction(action)
            action.triggered.connect(partial(self.set_with_dots, option))

            if option == "Simple":
                action.setChecked(True)

        submenu = QMenu("Plot lines", self.menubar)
        submenu.addActions(memory_option.actions())
        menu.addMenu(submenu)

        # integer stepmode menu
        memory_option = QActionGroup(self)

        for option in ("Step mode", "Direct connect mode"):

            action = QAction(option, menu)
            action.setCheckable(True)
            memory_option.addAction(action)
            action.triggered.connect(partial(self.set_step_mode, option))

            if option == "Step mode":
                action.setChecked(True)

        submenu = QMenu("Integer line style", self.menubar)
        submenu.addActions(memory_option.actions())
        menu.addMenu(submenu)

        # search mode menu
        search_option = QActionGroup(self)

        for option in ("Match start", "Match contains"):

            action = QAction(option, menu)
            action.setCheckable(True)
            search_option.addAction(action)
            action.triggered.connect(partial(self.set_search_option, option))

            if option == "Match start":
                action.setChecked(True)

        submenu = QMenu("Search", self.menubar)
        submenu.addActions(search_option.actions())
        menu.addMenu(submenu)

        # plot option menu
        plot_actions = QActionGroup(self)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/fit.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tF".format("Fit trace"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_F))
        action.setShortcut(Qt.Key_F)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/grid.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tG".format("Grid"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_G))
        action.setShortcut(Qt.Key_G)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/home.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tH".format("Home"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_H))
        action.setShortcut(Qt.Key_H)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/list2.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tS".format("Stack"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_S))
        action.setShortcut(Qt.Key_S)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/zoom-in.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tI".format("Zoom in"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_I))
        action.setShortcut(Qt.Key_I)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/zoom-out.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tO".format("Zoom out"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_O))
        action.setShortcut(Qt.Key_O)
        plot_actions.addAction(action)

        # values display

        display_format_actions = QActionGroup(self)

        action = QAction("{: <20}\tCtrl+H".format("Hex"), menu)
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_H, modifier=Qt.ControlModifier)
        )
        action.setShortcut(QKeySequence("Ctrl+H"))
        display_format_actions.addAction(action)

        action = QAction("{: <20}\tCtrl+B".format("Bin"), menu)
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_B, modifier=Qt.ControlModifier)
        )
        action.setShortcut(QKeySequence("Ctrl+B"))
        display_format_actions.addAction(action)

        action = QAction("{: <20}\tCtrl+P".format("Physical"), menu)
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_P, modifier=Qt.ControlModifier)
        )
        action.setShortcut(QKeySequence("Ctrl+P"))
        display_format_actions.addAction(action)

        # info

        info = QActionGroup(self)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/info.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tM".format("Statistics"), menu)
        action.triggered.connect(partial(self.file_action, key=Qt.Key_M))
        action.setShortcut(QKeySequence("M"))
        info.addAction(action)

        # cursors
        cursors_actions = QActionGroup(self)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/cursor.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tC".format("Cursor"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_C))
        action.setShortcut(Qt.Key_C)
        cursors_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/right.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\t←".format("Move cursor left"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_Left))
        action.setShortcut(Qt.Key_Left)
        cursors_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/left.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\t→".format("Move cursor right"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_Right))
        action.setShortcut(Qt.Key_Right)
        cursors_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/range.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tR".format("Range"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_R))
        action.setShortcut(Qt.Key_R)
        cursors_actions.addAction(action)

        menu = QMenu("Plot", self.menubar)
        menu.addActions(plot_actions.actions())
        menu.addSeparator()
        menu.addActions(cursors_actions.actions())
        menu.addSeparator()
        menu.addActions(display_format_actions.actions())
        menu.addSeparator()
        menu.addActions(info.actions())
        self.menubar.addMenu(menu)

        menu = self.menubar.addMenu("Help")
        open_group = QActionGroup(self)
        action = QAction("Online documentation", menu)
        action.triggered.connect(self.help)
        open_group.addAction(action)
        menu.addActions(open_group.actions())

        self.memory = "minimum"
        self.match = "Match start"
        self.with_dots = False
        self.step_mode = True
        self.toolBox.setCurrentIndex(0)

        self.show()

    def help(self, event):
        os.system(r'start "" http://asammdf.readthedocs.io/en/development/gui.html')

    def file_action(self, key, modifier=Qt.NoModifier):
        event = QKeyEvent(QEvent.KeyPress, key, modifier)
        widget = self.files.currentWidget()
        if widget and widget.plot:
            widget.keyPressEvent(event)

    def plot_action(self, key, modifier=Qt.NoModifier):
        event = QKeyEvent(QEvent.KeyPress, key, modifier)
        widget = self.files.currentWidget()
        if widget and widget.plot:
            widget.plot.keyPressEvent(event)
            widget.keyPressEvent(event)

    def set_memory_option(self, option):
        self.memory = option

    def set_with_dots(self, option):
        self.with_dots = True if option == "With dots" else False

        count = self.files.count()

        for i in range(count):
            self.files.widget(i).set_line_style(with_dots=self.with_dots)

    def set_step_mode(self, option):
        self.step_mode = True if option == "Step mode" else False

        count = self.files.count()

        for i in range(count):
            self.files.widget(i).set_line_style(step_mode=self.step_mode)

    def set_search_option(self, option):
        self.match = option
        count = self.files.count()
        for i in range(count):
            self.files.widget(i).search_field.set_search_option(option)
            self.files.widget(i).filter_field.set_search_option(option)

    def update_progress(self, current_index, max_index):
        self.progress = current_index, max_index

    def delete_item(self, item):
        index = self.files_list.row(item)
        self.files_list.takeItem(index)

    def function_select(self, val):
        if self.concatenate.isChecked():
            self.cs_btn.setText("Concatenate")
        else:
            self.cs_btn.setText("Stack")

    def cs_clicked(self, event):
        if self.concatenate.isChecked():
            func = MDF.concatenate
            operation = "Concatenating"
        else:
            func = MDF.stack
            operation = "Stacking"

        version = self.cs_format.currentText()

        sync = self.sync.checkState() == Qt.Checked

        memory = self.memory

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.cs_split.checkState() == Qt.Checked
        if split:
            split_size = int(self.cs_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.cs_compression.currentIndex()

        count = self.files_list.count()

        files = [self.files_list.item(row).text() for row in range(count)]

        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select output measurement file", "", filter
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select output measurement file", "", filter
            )
            file_name = str(file_name)

        if file_name:

            progress = setup_progress(
                parent=self,
                title="{} measurements".format(operation),
                message="{} files and saving to {} format".format(operation, version),
                icon_name="stack",
            )

            target = func
            kwargs = {
                "files": files,
                "outversion": version,
                "memory": memory,
                "callback": self.update_progress,
                "sync": sync,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=50,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # save it
            progress.setLabelText('Saving output file "{}"'.format(file_name))

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=50,
                offset=50,
                progress=progress,
            )

            progress.cancel()

    def open_multiple_files(self, event):
        if QT > 4:
            file_names, _ = QFileDialog.getOpenFileNames(
                self, "Select measurement file", "", "MDF files (*.dat *.mdf *.mf4)"
            )
        else:
            file_names = QFileDialog.getOpenFileNames(
                self, "Select measurement file", "", "MDF files (*.dat *.mdf *.mf4)"
            )
            file_names = [str(file_name) for file_name in file_names]

        if file_names:
            self.files_list.addItems(file_names)
            count = self.files_list.count()

            icon = QIcon()
            icon.addPixmap(QPixmap(":/file.png"), QIcon.Normal, QIcon.Off)

            for row in range(count):
                self.files_list.item(row).setIcon(icon)

    def open(self, event):
        if self.toolBox.currentIndex() == 0:
            self.open_file(event)
        else:
            self.open_multiple_files(event)

    def open_file(self, event):
        if QT > 4:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select measurement file",
                "",
                "MDF/DL3/ERG files (*.dat *.mdf *.mf4 *.dl3 *.erg)",
            )
        else:
            file_name = QFileDialog.getOpenFileName(
                self,
                "Select measurement file",
                "",
                "MDF/DL3/ERG files (*.dat *.mdf *.mf4 *.dl3 *.erg)",
            )
            file_name = str(file_name)

        if file_name:
            file_name = str(file_name)
            index = self.files.count()

            try:
                widget = FileWidget(
                    file_name, self.memory, self.step_mode, self.with_dots, self
                )
                widget.search_field.set_search_option(self.match)
                widget.filter_field.set_search_option(self.match)
            except:
                raise
            else:
                self.files.addTab(widget, os.path.basename(file_name))
                self.files.setTabToolTip(index, file_name)
                self.files.setCurrentIndex(index)

    def close_file(self, index):
        widget = self.files.widget(index)
        if widget:
            widget.close()
            widget.setParent(None)

        if self.files.count():
            self.files.setCurrentIndex(0)

    def closeEvent(self, event):
        count = self.files.count()
        for i in range(count):
            self.files.widget(i).close()
        event.accept()

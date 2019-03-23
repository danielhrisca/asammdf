# -*- coding: utf-8 -*-
from functools import partial
from pathlib import Path
import webbrowser

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

from ..ui import resource_qt5 as resource_rc

from ...mdf import MDF, SUPPORTED_VERSIONS
from ...version import __version__ as libversion
from ..utils import TERMINATED, run_thread_with_progress, setup_progress
from .list import ListWidget
from .file import FileWidget

HERE = Path(__file__).resolve().parent


class MainWindow(QMainWindow):
    def __init__(self, files=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        uic.loadUi(HERE.joinpath("..", "ui", "main_window.ui"), self)

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

        menu = self.menubar.addMenu("File")
        open_group = QActionGroup(self)
        icon = QIcon()
        icon.addPixmap(QPixmap(":/open.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "Open", menu)
        action.triggered.connect(self.open)
        open_group.addAction(action)
        menu.addActions(open_group.actions())

        # mode_actions
        mode_actions = QActionGroup(self)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/file.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}".format("Single files"), menu)
        action.triggered.connect(partial(self.stackedWidget.setCurrentIndex, 0))
        mode_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/list.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}".format("Multiple files"), menu)
        action.triggered.connect(partial(self.stackedWidget.setCurrentIndex, 1))
        mode_actions.addAction(action)

        menu = QMenu("Mode", self.menubar)
        menu.addActions(mode_actions.actions())
        self.menubar.addMenu(menu)

        menu = QMenu("Settings", self.menubar)
        self.menubar.addMenu(menu)

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
        action = QAction(icon, f"{'Fit trace': <20}\tF", menu)
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

        action = QAction("{: <20}\t.".format("Toggle dots"), menu)
        action.triggered.connect(partial(self.toggle_dots, key=Qt.Key_O))
        action.setShortcut(Qt.Key_Period)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/plus.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tIns".format("Insert computation"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_Insert))
        action.setShortcut(Qt.Key_Insert)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/save.png"), QIcon.Normal, QIcon.Off)
        action = QAction(icon, "{: <20}\tCtrl+S".format("Save plot channels"), menu)
        action.triggered.connect(partial(self.plot_action, key=Qt.Key_S, modifier=Qt.ControlModifier))
        action.setShortcut(QKeySequence("Ctrl+S"))
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

        self.match = "Match start"
        self.with_dots = False
        self.step_mode = True
        self.stackedWidget.setCurrentIndex(0)
        self.setWindowTitle(f'asammdf {libversion}')

        if files:
            for name in files:
                self._open_file(name)

        self.setAcceptDrops(True)

        self.show()

    def help(self, event):
        webbrowser.open_new(r'http://asammdf.readthedocs.io/en/development/gui.html')

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

    def toggle_dots(self, key):
        self.with_dots = not self.with_dots

        count = self.files.count()

        for i in range(count):
            self.files.widget(i).set_line_style(with_dots=self.with_dots)

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
        add_samples_origin = self.add_samples_origin.checkState() == Qt.Checked

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

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:

            progress = setup_progress(
                parent=self,
                title=f"{operation} measurements",
                message=f"{operation} files and saving to {version} format",
                icon_name="stack",
            )

            target = func
            kwargs = {
                "files": files,
                "outversion": version,
                "callback": self.update_progress,
                "sync": sync,
                "add_samples_origin": add_samples_origin,
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
            progress.setLabelText(f'Saving output file "{file_name}"')

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
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Select measurement file",
            "",
            "MDF v3 (*.dat *.mdf);;MDF v4(*.mf4);;DL3/ERG files (*.dl3 *.erg);;All files (*.dat *.mdf *.mf4 *.dl3 *.erg)",
            "All files (*.dat *.mdf *.mf4 *.dl3 *.erg)",
        )

        if file_names:
            self.files_list.addItems(file_names)
            count = self.files_list.count()

            icon = QIcon()
            icon.addPixmap(QPixmap(":/file.png"), QIcon.Normal, QIcon.Off)

            for row in range(count):
                self.files_list.item(row).setIcon(icon)

    def open(self, event):
        if self.stackedWidget.currentIndex() == 0:
            self.open_file(event)
        else:
            self.open_multiple_files(event)


    def _open_file(self, file_name):
        file_name = Path(file_name)
        index = self.files.count()

        try:
            widget = FileWidget(file_name, self.with_dots, self)
            widget.search_field.set_search_option(self.match)
            widget.filter_field.set_search_option(self.match)
        except:
            raise
        else:
            self.files.addTab(widget, file_name.name)
            self.files.setTabToolTip(index, str(file_name))
            self.files.setCurrentIndex(index)
            widget.file_scrambled.connect(self.open_scrambled_file)

    def open_file(self, event):
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Select measurement file",
            "",
            "MDF v3 (*.dat *.mdf);;MDF v4(*.mf4);;DL3/ERG files (*.dl3 *.erg);;All files (*.dat *.mdf *.mf4 *.dl3 *.erg)",
            "All files (*.dat *.mdf *.mf4 *.dl3 *.erg)",
        )

        for file_name in file_names:
            self._open_file(file_name)

    def open_scrambled_file(self, name):
        filename = Path(name)
        index = self.files.count()

        try:
            widget = FileWidget(name, self.with_dots, self)
            widget.search_field.set_search_option(self.match)
            widget.filter_field.set_search_option(self.match)
        except:
            raise
        else:
            self.files.addTab(widget, filename.name)
            self.files.setTabToolTip(index, str(filename))
            self.files.setCurrentIndex(index)
            widget.file_scrambled.connect(self.open_scrambled_file)

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

    def dragEnterEvent(self, e):

        e.accept()

    def dropEvent(self, e):
        try:
            for path in e.mimeData().text().splitlines():
                path = Path(path.replace(r'file:///', ''))
                if path.suffix.lower() in ('.dat', '.mdf', '.mf4'):
                    self._open_file(path)
        except:
            pass

# -*- coding: utf-8 -*-
import os
import sys

PYVERSION = sys.version_info[0]
bin_ = bin
import traceback
import re
import logging

from copy import deepcopy
from datetime import datetime
from functools import reduce, partial
from io import StringIO
from threading import Thread
from time import sleep

import numpy as np

try:
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5 import uic
    from asammdfgui import resource_qt5 as resource_rc

    QT = 5

except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    from PyQt4 import uic
    from asammdfgui import resource_qt4 as resource_rc

    QT = 4
try:
    HERE = os.path.dirname(os.path.realpath(__file__))
except:
    HERE = os.path.abspath(os.path.dirname(sys.argv[0]))
# import asammdfgui
# HERE = asammdfgui.__path__[0]
# print(*list(os.listdir(asammdfgui.__path__[0])), sep='\n')


class AdvancedSearch(QDialog):
    def __init__(self, channels_db, *args, **kwargs):

        super(AdvancedSearch, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "search_dialog.ui"), self)

        self.result = set()
        self.channels_db = channels_db

        self.apply_btn.clicked.connect(self._apply)
        self.apply_all_btn.clicked.connect(self._apply_all)
        self.cancel_btn.clicked.connect(self._cancel)

        self.search_box.textChanged.connect(self.search_text_changed)
        self.match_kind.currentTextChanged.connect(self.search_box.textChanged.emit)

        self.setWindowTitle("Search & select channels")

    def search_text_changed(self, text):
        if len(text) >= 2:
            if self.match_kind.currentText() == "Wildcard":
                pattern = text.replace("*", "_WILDCARD_")
                pattern = re.escape(pattern)
                pattern = pattern.replace("_WILDCARD_", ".*")
            else:
                pattern = text

            try:
                pattern = re.compile("(?i){}".format(pattern))
                matches = [name for name in self.channels_db if pattern.match(name)]
                self.matches.clear()
                self.matches.addItems(matches)
                if matches:
                    self.status.setText("")
                else:
                    self.status.setText("No match found")
            except Exception as err:
                self.status.setText(str(err))
                self.matches.clear()

    def _apply(self, event):
        self.result = set()
        for item in self.matches.selectedItems():
            for entry in self.channels_db[item.text()]:
                self.result.add(entry)
        self.close()

    def _apply_all(self, event):
        count = self.matches.count()
        self.result = set()
        for i in range(count):
            for entry in self.channels_db[self.matches.item(i).text()]:
                self.result.add(entry)
        self.close()

    def _cancel(self, event):
        self.result = set()
        self.close()


from asammdf import MDF, MDF2, MDF3, MDF4, SUPPORTED_VERSIONS
from asammdf import __version__ as libversion
from asammdf.plot import Plot, ChannelStats

import pyqtgraph as pg


__version__ = "0.1.0"
TERMINATED = object()


def excepthook(exc_type, exc_value, tracebackobj):
    """
    Global function to catch unhandled exceptions.

    Parameters
    ----------
    exc_type : str
        exception type
    exc_value : int
        exception value
    tracebackobj : traceback
        traceback object
    """
    separator = "-" * 80
    notice = "The following error was triggered:"

    now = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")

    info = StringIO()
    traceback.print_tb(tracebackobj, None, info)
    info.seek(0)
    info = info.read()

    errmsg = "{}\t \n{}".format(exc_type, exc_value)
    sections = [now, separator, errmsg, separator, info]
    msg = "\n".join(sections)

    print(msg)

    QMessageBox.warning(None, notice, msg)


sys.excepthook = excepthook


def run_thread_with_progress(widget, target, kwargs, factor, offset, progress):
    termination_request = False

    thr = WorkerThread(target=target, kwargs=kwargs)

    thr.start()

    while widget.progress is None:
        sleep(0.1)

    while thr.is_alive():
        QApplication.processEvents()
        termination_request = progress.wasCanceled()
        if termination_request:
            MDF._terminate = True
            MDF2._terminate = True
            MDF3._terminate = True
            MDF4._terminate = True
        else:
            progress.setValue(
                int(widget.progress[0] / widget.progress[1] * factor) + offset
            )
        sleep(0.1)

    if termination_request:
        MDF._terminate = False
        MDF2._terminate = False
        MDF3._terminate = False
        MDF4._terminate = False

    progress.setValue(factor + offset)

    if thr.error:
        widget.progress = None
        progress.cancel()
        raise Exception(thr.error)

    widget.progress = None

    if termination_request:
        return TERMINATED
    else:
        return thr.output


def setup_progress(parent, title, message, icon_name):
    progress = QProgressDialog(message, "", 0, 100, parent)

    progress.setWindowModality(Qt.ApplicationModal)
    progress.setCancelButton(None)
    progress.setAutoClose(True)
    progress.setWindowTitle(title)
    icon = QIcon()
    icon.addPixmap(QPixmap(":/{}.png".format(icon_name)), QIcon.Normal, QIcon.Off)
    progress.setWindowIcon(icon)
    progress.show()

    return progress


class WorkerThread(Thread):
    def __init__(self, *args, **kargs):
        super(WorkerThread, self).__init__(*args, **kargs)
        self.output = None
        self.error = ""

    def run(self):
        if PYVERSION < 3:
            try:
                self.output = self._Thread__target(
                    *self._Thread__args, **self._Thread__kwargs
                )
            except Exception as err:
                self.error = err
        else:
            try:
                self.output = self._target(*self._args, **self._kwargs)
            except Exception as err:
                self.error = err


class TreeItem(QTreeWidgetItem):
    def __init__(self, entry, *args, **kwargs):

        super(TreeItem, self).__init__(*args, **kwargs)

        self.entry = entry


class TreeWidget(QTreeWidget):
    def __init__(self, *args, **kwargs):

        super(TreeWidget, self).__init__(*args, **kwargs)

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            elif len(selected_items) == 1:
                item = selected_items[0]
                checked = item.checkState(0)
                if checked == Qt.Checked:
                    item.setCheckState(0, Qt.Unchecked)
                else:
                    item.setCheckState(0, Qt.Checked)
            else:
                if any(item.checkState(0) == Qt.Unchecked for item in selected_items):
                    checked = Qt.Checked
                else:
                    checked = Qt.Unchecked
                for item in selected_items:
                    item.setCheckState(0, checked)
        else:
            super(TreeWidget, self).keyPressEvent(event)


class ListWidget(QListWidget):

    itemsDeleted = pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super(ListWidget, self).__init__(*args, **kwargs)

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.setAlternatingRowColors(True)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Delete:
            selected_items = self.selectedItems()
            deleted = []
            for item in selected_items:
                row = self.row(item)
                deleted.append(row)
                self.takeItem(row)
            if deleted:
                self.itemsDeleted.emit(deleted)
        elif key == Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            try:
                states = [
                    self.itemWidget(item).display.checkState()
                    for item in selected_items
                ]

                if any(state == Qt.Unchecked for state in states):
                    state = Qt.Checked
                else:
                    state = Qt.Unchecked
                for item in selected_items:
                    wid = self.itemWidget(item)
                    wid.display.setCheckState(state)
            except:
                pass
        else:
            super(ListWidget, self).keyPressEvent(event)


class ChannelInfoWidget(QWidget):
    def __init__(self, channel, *args, **kwargs):
        super(ChannelInfoWidget, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "channel_info_widget.ui"), self)

        self.channel_label.setText(channel.metadata())

        if channel.conversion:
            self.conversion_label.setText(channel.conversion.metadata())

        if channel.source:
            self.source_label.setText(channel.source.metadata())


class RangeEditor(QDialog):
    def __init__(self, unit="", ranges=None, *args, **kwargs):
        super(RangeEditor, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "range_editor_dialog.ui"), self)

        self.unit = unit
        self.result = {}
        self.pressed_button = None

        if ranges:
            for i, (range, color) in enumerate(ranges.items()):
                self.cell_pressed(i, 0, range, color)

        self.table.cellPressed.connect(self.cell_pressed)
        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)
        self.reset_btn.clicked.connect(self.reset)

        self.setWindowTitle("Edit channel range colors")

    def cell_pressed(self, row, column, range=(0, 0), color="#000000"):

        for col in (0, 1):
            box = QDoubleSpinBox(self.table)
            box.setSuffix(" {}".format(self.unit))
            box.setRange(-10 ** 10, 10 ** 10)
            box.setDecimals(6)
            box.setValue(range[col])

            self.table.setCellWidget(row, col, box)

        button = QPushButton("", self.table)
        button.setStyleSheet("background-color: {};".format(color))
        self.table.setCellWidget(row, 2, button)
        button.clicked.connect(partial(self.select_color, button=button))

        button = QPushButton("Delete", self.table)
        self.table.setCellWidget(row, 3, button)
        button.clicked.connect(partial(self.delete_row, row=row))

    def delete_row(self, event, row):
        for column in range(4):
            self.table.setCellWidget(row, column, None)

    def select_color(self, event, button):
        color = button.palette().button().color()
        color = QColorDialog.getColor(color).name()
        button.setStyleSheet("background-color: {};".format(color))

    def apply(self, event):
        for row in range(100):
            try:
                start = self.table.cellWidget(row, 0).value()
                stop = self.table.cellWidget(row, 1).value()
                button = self.table.cellWidget(row, 2)
                color = button.palette().button().color().name()
            except:
                continue
            else:
                self.result[(start, stop)] = color
        self.pressed_button = "apply"
        self.close()

    def reset(self, event):
        for row in range(100):
            for column in range(4):
                self.table.setCellWidget(row, column, None)
        self.result = {}

    def cancel(self, event):
        self.result = {}
        self.pressed_button = "cancel"
        self.close()


class ChannelInfoDialog(QDialog):
    def __init__(self, channel, *args, **kwargs):
        super(QDialog, self).__init__(*args, **kwargs)

        self.setWindowFlags(Qt.Window)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle(channel.name)

        layout.addWidget(ChannelInfoWidget(channel, self))

        self.setStyleSheet('font: 8pt "Consolas";}')

        icon = QIcon()
        icon.addPixmap(QPixmap(":/info.png"), QIcon.Normal, QIcon.Off)

        self.setWindowIcon(icon)
        self.setGeometry(240, 60, 1200, 600)

        screen = QApplication.desktop().screenGeometry()
        self.move((screen.width() - 1200) // 2, (screen.height() - 600) // 2)


class TabularValuesDialog(QDialog):
    def __init__(self, signals, ranges, *args, **kwargs):
        super(QDialog, self).__init__(*args, **kwargs)

        self.setWindowFlags(Qt.Window)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle("Tabular values")

        self.table = QTableWidget(self)

        self.header = []
        for sig in signals:
            self.header.append("t [s]")
            self.header.append("{} ({})".format(sig.name, sig.unit))

        self.table.setColumnCount(2 * len(signals))
        self.table.setRowCount(max(len(sig) for sig in signals))
        self.table.setHorizontalHeaderLabels(self.header)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setMinimumSectionSize(QHeaderView.Stretch)
        self.table.horizontalHeader().setToolTip("")
        self.table.horizontalHeader().setMinimumSectionSize(100)
        self.table.horizontalHeader().sectionClicked.connect(self.show_name)
        self.table.horizontalHeader().entered.connect(self.hover)
        self.table.cellEntered.connect(self.hover)

        for i, sig in enumerate(signals):
            range_ = ranges[i]
            size = len(sig)
            for j in range(size):
                # self.table.setCellWidget(
                #     j,
                #     2*i,
                #     QLabel(str(sig.timestamps[j]), self.table),
                # )
                #
                # value = sig.samples[j]
                #
                # label = QLabel(str(sig.samples[j]), self.table)
                #
                # for (start, stop), color in range_.items():
                #     if start <= value < stop:
                #         label.setStyleSheet(
                #             "background-color: {};".format(color))
                #         break
                # else:
                #     label.setStyleSheet("background-color: transparent;")
                #
                # self.table.setCellWidget(
                #     j,
                #     2 * i + 1,
                #     label,
                # )

                self.table.setItem(j, 2 * i, QTableWidgetItem(str(sig.timestamps[j])))

                self.table.setItem(j, 2 * i + 1, QTableWidgetItem(str(sig.samples[j])))

        layout.addWidget(self.table)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/info.png"), QIcon.Normal, QIcon.Off)

        self.setWindowIcon(icon)
        self.setGeometry(240, 60, 1200, 600)

        screen = QApplication.desktop().screenGeometry()
        self.move((screen.width() - 1200) // 2, (screen.height() - 600) // 2)

    def hover(self, row, column):
        print("hover", row, column)

    def show_name(self, index):
        name = self.header[index // 2]
        widget = self.table.horizontalHeader()
        QToolTip.showText(widget.mapToGlobal(QPoint(0, 0)), name)


class ChannelDisplay(QWidget):

    color_changed = pyqtSignal(int, str)
    enable_changed = pyqtSignal(int, int)

    def __init__(self, index, unit="", *args, **kwargs):
        super(ChannelDisplay, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "channel_display_widget.ui"), self)

        self.color = "#ff0000"
        self._value_prefix = ""
        self._value = ""
        self._name = ""
        self.fmt = "{}"
        self.index = index
        self.ranges = {}
        self.unit = unit

        self.color_btn.clicked.connect(self.select_color)
        self.display.stateChanged.connect(self.display_changed)

    def mouseDoubleClickEvent(self, event):
        dlg = RangeEditor(self.unit, self.ranges)
        dlg.exec_()
        if dlg.pressed_button == "apply":
            self.ranges = dlg.result

    def display_changed(self, state):
        state = self.display.checkState()
        self.enable_changed.emit(self.index, state)

    def select_color(self):
        color = QColorDialog.getColor(QColor(self.color)).name()
        self.setColor(color)

        self.color_changed.emit(self.index, color)

    def setFmt(self, fmt):
        if fmt == "hex":
            self.fmt = "0x{:X}"
        elif fmt == "bin":
            self.fmt = "0b{:b}"
        elif fmt == "phys":
            self.fmt = "{}"
        else:
            self.fmt = fmt

    def setColor(self, color):
        self.color = color
        self.setName(self._name)
        self.setValue(self._value)
        self.color_btn.setStyleSheet("background-color: {};".format(color))

    def setName(self, text=""):
        self._name = text
        self.name.setText(
            '<html><head/><body><p><span style=" color:{};">{}</span></p></body></html>'.format(
                self.color, self._name
            )
        )

    def setPrefix(self, text=""):
        self._value_prefix = text

    def setValue(self, value):
        self._value = value
        if self.ranges and value not in ("", "n.a."):
            for (start, stop), color in self.ranges.items():
                if start <= value < stop:
                    self.setStyleSheet("background-color: {};".format(color))
                    break
            else:
                self.setStyleSheet("background-color: transparent;")
        else:
            self.setStyleSheet("background-color: transparent;")
        template = '<html><head/><body><p><span style=" color:{{}};">{{}}{}</span></p></body></html>'
        if value not in ("", "n.a."):
            template = template.format(self.fmt)
        else:
            template = template.format("{}")
        self.value.setText(template.format(self.color, self._value_prefix, value))


class SearchWidget(QWidget):

    selectionChanged = pyqtSignal()

    def __init__(self, channels_db, *args, **kwargs):
        super(SearchWidget, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "search_widget.ui"), self)
        self.channels_db = channels_db

        self.matches = 0
        self.current_index = 1
        self.entries = []

        completer = QCompleter(sorted(self.channels_db, key=lambda x: x.lower()), self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setModelSorting(QCompleter.CaseInsensitivelySortedModel)
        if QT == 5:
            completer.setFilterMode(Qt.MatchContains)
        self.search.setCompleter(completer)

        self.search.textChanged.connect(self.display_results)

        self.up_btn.clicked.connect(self.up)
        self.down_btn.clicked.connect(self.down)

    def down(self, event):
        if self.matches:
            self.current_index += 1
            if self.current_index >= self.matches:
                self.current_index = 0
            self.label.setText("{} of {}".format(self.current_index + 1, self.matches))
            self.selectionChanged.emit()

    def up(self, event):
        if self.matches:
            self.current_index -= 1
            if self.current_index < 0:
                self.current_index = self.matches - 1
            self.label.setText("{} of {}".format(self.current_index + 1, self.matches))
            self.selectionChanged.emit()

    def set_search_option(self, option):
        if QT == 5:
            if option == "Match start":
                self.search.completer().setFilterMode(Qt.MatchStartsWith)
            elif option == "Match contains":
                self.search.completer().setFilterMode(Qt.MatchContains)

    def display_results(self, text):
        channel_name = text.strip()
        if channel_name in self.channels_db:
            self.entries = self.channels_db[channel_name]
            self.matches = len(self.entries)
            self.label.setText("1 of {}".format(self.matches))
            self.current_index = 0
            self.selectionChanged.emit()

        else:
            self.label.setText("No match")
            self.matches = 0
            self.current_index = 0
            self.entries = []


class FileWidget(QWidget):
    def __init__(self, file_name, memory, step_mode, with_dots, *args, **kwargs):
        super(FileWidget, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "file_widget.ui"), self)

        self.plot = None

        self.file_name = file_name
        self.progress = None
        self.mdf = None
        self.memory = memory
        self.info = None
        self.info_index = None
        self.step_mode = step_mode
        self.with_dots = with_dots

        progress = QProgressDialog(
            'Opening "{}"'.format(self.file_name), "", 0, 100, self.parent()
        )

        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setAutoClose(True)
        progress.setWindowTitle("Opening measurement")
        icon = QIcon()
        icon.addPixmap(QPixmap(":/open.png"), QIcon.Normal, QIcon.Off)
        progress.setWindowIcon(icon)
        progress.show()

        if file_name.lower().endswith("erg"):
            progress.setLabelText("Converting from erg to mdf")
            try:
                from mfile import ERG

                self.mdf = ERG(file_name).export_mdf()
            except Exception as err:
                print(err)
                return
        else:

            if file_name.lower().endswith("dl3"):
                progress.setLabelText("Converting from dl3 to mdf")
                try:
                    import win32com.client

                    index = 0
                    while True:
                        mdf_name = "{}.{}.mdf".format(file_name, index)
                        if os.path.exists(mdf_name):
                            index += 1
                        else:
                            break

                    datalyser = win32com.client.Dispatch("Datalyser3.Datalyser3_COM")
                    try:
                        datalyser.DCOM_set_datalyser_visibility(False)
                    except:
                        pass
                    ret = datalyser.DCOM_convert_file_mdf_dl3(file_name, mdf_name, 0)
                    datalyser.DCOM_TerminateDAS()
                    file_name = mdf_name
                except Exception as err:
                    print(err)
                    return

            target = MDF
            kwargs = {
                "name": file_name,
                "memory": memory,
                "callback": self.update_progress,
            }

            self.mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=0,
                progress=progress,
            )

            if self.mdf is TERMINATED:
                return

        progress.setLabelText("Loading graphical elements")

        progress.setValue(35)

        self.filter_field = SearchWidget(deepcopy(self.mdf.channels_db), self)

        progress.setValue(37)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)

        channel_and_search = QWidget(splitter)

        self.channels_tree = TreeWidget(channel_and_search)
        self.search_field = SearchWidget(
            deepcopy(self.mdf.channels_db), channel_and_search
        )
        self.filter_tree = TreeWidget()

        self.search_field.selectionChanged.connect(
            partial(
                self.new_search_result,
                tree=self.channels_tree,
                search=self.search_field,
            )
        )
        self.filter_field.selectionChanged.connect(
            partial(
                self.new_search_result, tree=self.filter_tree, search=self.filter_field
            )
        )

        vbox = QVBoxLayout(channel_and_search)
        self.advanced_search_btn = QPushButton("", channel_and_search)
        icon = QIcon()
        icon.addPixmap(QPixmap(":/search.png"), QIcon.Normal, QIcon.Off)
        self.advanced_search_btn.setIcon(icon)
        self.advanced_search_btn.setToolTip("Advanced search and select channels")
        self.advanced_search_btn.clicked.connect(self.search)
        vbox.addWidget(self.search_field)

        vbox.addWidget(self.channels_tree, 1)
        channel_and_search.setLayout(vbox)

        hbox = QHBoxLayout(channel_and_search)

        self.clear_channels_btn = QPushButton("", channel_and_search)
        self.clear_channels_btn.setToolTip("Reset selection")
        icon = QIcon()
        icon.addPixmap(QPixmap(":/erase.png"), QIcon.Normal, QIcon.Off)
        self.clear_channels_btn.setIcon(icon)
        self.clear_channels_btn.setObjectName("clear_channels_btn")

        self.load_channel_list_btn = QPushButton("", channel_and_search)
        self.load_channel_list_btn.setToolTip("Load channel selection list")
        icon1 = QIcon()
        icon1.addPixmap(QPixmap(":/open.png"), QIcon.Normal, QIcon.Off)
        self.load_channel_list_btn.setIcon(icon1)
        self.load_channel_list_btn.setObjectName("load_channel_list_btn")

        self.save_channel_list_btn = QPushButton("", channel_and_search)
        self.save_channel_list_btn.setToolTip("Save channel selection list")
        icon2 = QIcon()
        icon2.addPixmap(QPixmap(":/save.png"), QIcon.Normal, QIcon.Off)
        self.save_channel_list_btn.setIcon(icon2)
        self.save_channel_list_btn.setObjectName("save_channel_list_btn")

        self.select_all_btn = QPushButton("", channel_and_search)
        self.select_all_btn.setToolTip("Select all channels")
        icon1 = QIcon()
        icon1.addPixmap(QPixmap(":/checkmark.png"), QIcon.Normal, QIcon.Off)
        self.select_all_btn.setIcon(icon1)

        hbox.addWidget(self.load_channel_list_btn)
        hbox.addWidget(self.save_channel_list_btn)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        hbox.addWidget(line)
        hbox.addWidget(self.select_all_btn)
        hbox.addWidget(self.clear_channels_btn)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        hbox.addWidget(line)
        hbox.addWidget(self.advanced_search_btn)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        hbox.addWidget(line)
        self.plot_btn = QPushButton("", channel_and_search)
        self.plot_btn.setToolTip("Plot selected channels")
        icon3 = QIcon()
        icon3.addPixmap(QPixmap(":/graph.png"), QIcon.Normal, QIcon.Off)
        self.plot_btn.setIcon(icon3)
        self.plot_btn.setObjectName("plot_btn")
        hbox.addWidget(self.plot_btn)
        hbox.addSpacerItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        vbox.addLayout(hbox)

        selection_list = QWidget(splitter)
        self.channel_selection = ListWidget(selection_list)
        self.channel_selection.setAlternatingRowColors(False)

        vbox = QVBoxLayout(selection_list)

        hbox = QHBoxLayout(selection_list)
        hbox.addWidget(QLabel("Selected channels"))
        self.cursor_info = QLabel("")
        self.cursor_info.setTextFormat(Qt.RichText)
        self.cursor_info.setAlignment(
            Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter
        )
        hbox.addWidget(self.cursor_info)

        vbox.addLayout(hbox)

        vbox.addWidget(self.channel_selection)

        self.filter_layout.addWidget(self.filter_field, 0, 0, 1, 1)

        self.channels_tree.itemDoubleClicked.connect(self.show_channel_info)
        self.filter_tree.itemDoubleClicked.connect(self.show_channel_info)

        self.channels_layout.insertWidget(0, splitter)
        self.filter_layout.addWidget(self.filter_tree, 1, 0, 8, 1)

        groups_nr = len(self.mdf.groups)

        self.channels_tree.setHeaderLabel("Channels")
        self.channels_tree.setToolTip(
            "Double click channel to see extended information"
        )
        self.filter_tree.setHeaderLabel("Channels")
        self.filter_tree.setToolTip("Double click channel to see extended information")

        for i, group in enumerate(self.mdf.groups):
            channel_group = QTreeWidgetItem()
            filter_channel_group = QTreeWidgetItem()
            channel_group.setText(0, "Channel group {}".format(i))
            filter_channel_group.setText(0, "Channel group {}".format(i))
            channel_group.setFlags(
                channel_group.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable
            )
            filter_channel_group.setFlags(
                channel_group.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable
            )

            self.channels_tree.addTopLevelItem(channel_group)
            self.filter_tree.addTopLevelItem(filter_channel_group)

            for j, ch in enumerate(group["channels"]):

                name = self.mdf.get_channel_name(i, j)
                channel = TreeItem((i, j), channel_group)
                channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                channel.setText(0, name)
                channel.setCheckState(0, Qt.Unchecked)

                channel = TreeItem((i, j), filter_channel_group)
                channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                channel.setText(0, name)
                channel.setCheckState(0, Qt.Unchecked)

            if self.mdf.version >= "4.00":
                for j, ch in enumerate(group["logging_channels"], 1):
                    name = ch.name

                    channel = TreeItem((i, -j), channel_group)
                    channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                    channel.setText(0, name)
                    channel.setCheckState(0, Qt.Unchecked)

                    channel = TreeItem((i, -j), filter_channel_group)
                    channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                    channel.setText(0, name)
                    channel.setCheckState(0, Qt.Unchecked)

            progress.setValue(37 + int(53 * (i + 1) / groups_nr))
            QApplication.processEvents()

        progress.setValue(90)

        self.resample_format.insertItems(0, SUPPORTED_VERSIONS)
        index = self.resample_format.findText(self.mdf.version)
        if index >= 0:
            self.resample_format.setCurrentIndex(index)
        self.resample_compression.insertItems(
            0, ("no compression", "deflate", "transposed deflate")
        )
        self.resample_split_size.setValue(10)
        self.resample_btn.clicked.connect(self.resample)

        self.filter_format.insertItems(0, SUPPORTED_VERSIONS)
        index = self.filter_format.findText(self.mdf.version)
        if index >= 0:
            self.filter_format.setCurrentIndex(index)
        self.filter_compression.insertItems(
            0, ("no compression", "deflate", "transposed deflate")
        )
        self.filter_split_size.setValue(10)
        self.filter_btn.clicked.connect(self.filter)

        self.convert_format.insertItems(0, SUPPORTED_VERSIONS)
        self.convert_compression.insertItems(
            0, ("no compression", "deflate", "transposed deflate")
        )
        self.convert_split_size.setValue(10)
        self.convert_btn.clicked.connect(self.convert)

        self.cut_format.insertItems(0, SUPPORTED_VERSIONS)
        index = self.cut_format.findText(self.mdf.version)
        if index >= 0:
            self.cut_format.setCurrentIndex(index)
        self.cut_compression.insertItems(
            0, ("no compression", "deflate", "transposed deflate")
        )
        self.cut_split_size.setValue(10)
        self.cut_btn.clicked.connect(self.cut)

        self.cut_interval.setText("Unknown measurement interval")

        progress.setValue(99)

        self.empty_channels.insertItems(0, ("zeros", "skip"))
        self.mat_format.insertItems(0, ("4", "5", "7.3"))
        self.oned_as.insertItems(0, ("row", "column"))
        self.export_type.insertItems(0, ("csv", "excel", "hdf5", "mat", "parquet"))
        self.export_btn.clicked.connect(self.export)

        # self.channels_tree.itemChanged.connect(self.select)
        self.plot_btn.clicked.connect(self.plot_pyqtgraph)
        self.clear_filter_btn.clicked.connect(self.clear_filter)
        self.clear_channels_btn.clicked.connect(self.clear_channels)

        self.aspects.setCurrentIndex(0)

        progress.setValue(100)

        self.load_channel_list_btn.clicked.connect(self.load_channel_list)
        self.save_channel_list_btn.clicked.connect(self.save_channel_list)
        self.load_filter_list_btn.clicked.connect(self.load_filter_list)
        self.save_filter_list_btn.clicked.connect(self.save_filter_list)

        self.channel_selection.itemsDeleted.connect(self.channel_selection_reduced)
        self.channel_selection.itemSelectionChanged.connect(
            self.channel_selection_modified
        )

    def set_line_style(self, with_dots=None, step_mode=None):
        if (with_dots, step_mode) != (None, None):
            if step_mode is not None:
                self.step_mode = step_mode

            if with_dots is not None:
                self.with_dots = with_dots

            if self.plot:
                self.plot.update_lines(step_mode=step_mode, with_dots=with_dots)

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if key == Qt.Key_M:

            if self.info is None:

                self.info = ChannelStats(parent=self.splitter)
                if self.info_index is None:
                    self.info.clear()
                else:
                    stats = self.plot.get_stats(self.info_index)
                    self.info.set_stats(stats)
            else:
                self.info.setParent(None)
                self.info.hide()
                self.info = None

        elif modifier == Qt.ControlModifier and key in (Qt.Key_B, Qt.Key_H, Qt.Key_P):
            if key == Qt.Key_B:
                fmt = "bin"
            elif key == Qt.Key_H:
                fmt = "hex"
            else:
                fmt = "phys"
            if self.info and self.info_index is not None:
                self.info.fmt = fmt
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

        elif modifier == Qt.ControlModifier and key == Qt.Key_T:
            selected_items = self.channel_selection.selectedItems()
            rows = self.channel_selection.count()

            indexes = [
                i
                for i in range(rows)
                if self.channel_selection.item(i) in selected_items
            ]

            ranges = [
                self.channel_selection.itemWidget(item).ranges
                for item in selected_items
            ]

            signals = [self.plot.signals[i] for i in indexes]

            dlg = TabularValuesDialog(signals, ranges, self)
            dlg.setModal(True)
            dlg.exec_()

        else:
            super(FileWidget, self).keyPressEvent(event)

    def search(self):
        dlg = AdvancedSearch(self.mdf.channels_db, self)
        dlg.setModal(True)
        dlg.exec_()
        result = dlg.result
        if result:
            iterator = QTreeWidgetItemIterator(self.channels_tree)

            dg_cntr = -1
            ch_cntr = 0

            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    dg_cntr += 1
                    ch_cntr = 0
                    continue

                if (dg_cntr, ch_cntr) in result:
                    item.setCheckState(0, Qt.Checked)

                iterator += 1
                ch_cntr += 1

    def channel_selection_reduced(self, deleted):

        for i in sorted(deleted, reverse=True):
            item = self.plot.curves.pop(i)
            item.hide()
            item.setParent(None)

            item = self.plot.axes.pop(i)
            item.hide()
            item.setParent(None)

            item = self.plot.view_boxes.pop(i)
            item.hide()
            item.setParent(None)

            self.plot.signals.pop(i)

        rows = self.channel_selection.count()

        for i in range(rows):
            item = self.channel_selection.item(i)
            wid = self.channel_selection.itemWidget(item)
            wid.index = i

    def channel_selection_modified(self):
        selected_items = self.channel_selection.selectedItems()
        count = len([sig for sig in self.plot.signals if sig.enable])
        rows = self.channel_selection.count()

        for i in range(rows):
            item = self.channel_selection.item(i)
            if count > 1 and item in selected_items:
                if self.plot.signals[i].enable and not self.plot.axes[i].isVisible():
                    self.plot.axes[i].show()
                if self.info:
                    self.info.clear()
            else:
                if self.plot.axes[i].isVisible():
                    self.plot.axes[i].hide()

        if len(selected_items) == 1:
            self.info_index = self.channel_selection.row(selected_items[0])
        else:
            self.info_index = None

        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def save_channel_list(self):
        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select output channel list file", "", "TXT files (*.txt)"
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select output channel list file", "", "TXT files (*.txt)"
            )
            file_name = str(file_name)
        if file_name:
            with open(file_name, "w") as output:
                iterator = QTreeWidgetItemIterator(self.channels_tree)

                signals = []
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue

                    if item.checkState(0) == Qt.Checked:
                        signals.append(item.text(0))

                    iterator += 1

                output.write("\n".join(signals))

    def load_channel_list(self):
        if QT > 4:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Select channel list file", "", "TXT files (*.txt)"
            )
        else:
            file_name = QFileDialog.getOpenFileName(
                self, "Select channel list file", "", "TXT files (*.txt)"
            )
            file_name = str(file_name)

        if file_name:
            with open(file_name, "r") as infile:
                channels = [line.strip() for line in infile.readlines()]
                channels = [name for name in channels if name]

            iterator = QTreeWidgetItemIterator(self.channels_tree)

            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    continue

                channel_name = item.text(0)
                if channel_name in channels:
                    item.setCheckState(0, Qt.Checked)
                    channels.pop(channels.index(channel_name))
                else:
                    item.setCheckState(0, Qt.Unchecked)

                iterator += 1

    def save_filter_list(self):
        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select output filter list file", "", "TXT files (*.txt)"
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select output filter list file", "", "TXT files (*.txt)"
            )
            file_name = str(file_name)

        if file_name:
            with open(file_name, "w") as output:
                iterator = QTreeWidgetItemIterator(self.filter_tree)

                signals = []
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue

                    if item.checkState(0) == Qt.Checked:
                        signals.append(item.text(0))

                    iterator += 1

                output.write("\n".join(signals))

    def load_filter_list(self):
        if QT > 4:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Select filter list file", "", "TXT files (*.txt)"
            )
        else:
            file_name = QFileDialog.getOpenFileName(
                self, "Select filter list file", "", "TXT files (*.txt)"
            )
            file_name = str(file_name)

        if file_name:
            with open(file_name, "r") as infile:
                channels = [line.strip() for line in infile.readlines()]
                channels = [name for name in channels if name]

            iterator = QTreeWidgetItemIterator(self.filter_tree)

            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    continue

                channel_name = item.text(0)
                if channel_name in channels:
                    item.setCheckState(0, Qt.Checked)
                    channels.pop(channels.index(channel_name))
                else:
                    item.setCheckState(0, Qt.Unchecked)

                iterator += 1

    def cursor_move_finished(self):
        x = self.plot.timebase

        if x is not None and len(x):
            dim = len(x)
            position = self.plot.cursor1.value()

            right = np.searchsorted(x, position, side="right")
            if right == 0:
                next_pos = x[0]
            elif right == dim:
                next_pos = x[-1]
            else:
                if position - x[right - 1] < x[right] - position:
                    next_pos = x[right - 1]
                else:
                    next_pos = x[right]
            self.plot.cursor1.setPos(next_pos)

        self.plot.cursor_hint.setData(x=[], y=[])

    def cursor_moved(self):
        position = self.plot.cursor1.value()

        x = self.plot.timebase

        if x is not None and len(x):
            dim = len(x)
            position = self.plot.cursor1.value()

            right = np.searchsorted(x, position, side="right")
            if right == 0:
                next_pos = x[0]
            elif right == dim:
                next_pos = x[-1]
            else:
                if position - x[right - 1] < x[right] - position:
                    next_pos = x[right - 1]
                else:
                    next_pos = x[right]

            y = []

            _, (hint_min, hint_max) = self.plot.viewbox.viewRange()

            for viewbox, sig, curve in zip(
                self.plot.view_boxes, self.plot.signals, self.plot.curves
            ):
                if curve.isVisible():
                    index = np.argwhere(sig.timestamps == next_pos).flatten()
                    if len(index):
                        _, (y_min, y_max) = viewbox.viewRange()

                        sample = sig.samples[index[0]]
                        sample = (sample - y_min) / (y_max - y_min) * (
                            hint_max - hint_min
                        ) + hint_min

                        y.append(sample)

            if self.plot.curve.isVisible():
                timestamps = self.plot.curve.xData
                samples = self.plot.curve.yData
                if len(samples):
                    index = np.argwhere(timestamps == next_pos).flatten()
                    if len(index):
                        _, (y_min, y_max) = self.plot.viewbox.viewRange()

                        sample = samples[index[0]]
                        sample = (sample - y_min) / (y_max - y_min) * (
                            hint_max - hint_min
                        ) + hint_min

                        y.append(sample)

            self.plot.viewbox.setYRange(hint_min, hint_max, padding=0)
            self.plot.cursor_hint.setData(x=[next_pos] * len(y), y=y)
            self.plot.cursor_hint.show()

        if not self.plot.region:
            self.cursor_info.setText("t = {:.6f}s".format(position))
            for i, signal in enumerate(self.plot.signals):
                cut_sig = signal.cut(position, position)
                if signal.texts is None or len(cut_sig) == 0:
                    samples = cut_sig.samples
                    if signal.conversion and "text_0" in signal.conversion:
                        samples = signal.conversion.convert(samples)
                        try:
                            samples = [s.decode("utf-8") for s in samples]
                        except:
                            samples = [s.decode("latin-1") for s in samples]
                else:
                    t = np.argwhere(signal.timestamps == cut_sig.timestamps).flatten()
                    try:
                        samples = [e.decode("utf-8") for e in signal.texts[t]]
                    except:
                        samples = [e.decode("latin-1") for e in signal.texts[t]]

                item = self.channel_selection.item(i)
                item = self.channel_selection.itemWidget(item)

                item.setPrefix("= ")
                item.setFmt(signal.format)

                if len(samples):
                    item.setValue(samples[0])
                else:
                    item.setValue("n.a.")

        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def cursor_removed(self):
        for i, signal in enumerate(self.plot.signals):
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            if not self.plot.region:
                self.cursor_info.setText("")
                item.setPrefix("")
                item.setValue("")
        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def range_modified(self):
        start, stop = self.plot.region.getRegion()
        self.cut_start.setValue(start)
        self.cut_stop.setValue(stop)

        self.cursor_info.setText(
            (
                "< html > < head / > < body >"
                "< p >t1 = {:.6f}s< / p > "
                "< p >t2 = {:.6f}s< / p > "
                "< p >Δt = {:.6f}s< / p > "
                "< / body > < / html >"
            ).format(start, stop, stop - start)
        )

        for i, signal in enumerate(self.plot.signals):
            samples = signal.cut(start, stop).samples
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            item.setPrefix("Δ = ")
            item.setFmt(signal.format)

            if len(samples):
                if samples.dtype.kind in "ui":
                    delta = np.int64(np.float64(samples[-1]) - np.float64(samples[0]))
                else:
                    delta = samples[-1] - samples[0]

                item.setValue(delta)

            else:
                item.setValue("n.a.")

        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def xrange_changed(self):

        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def range_modified_finished(self):
        start, stop = self.plot.region.getRegion()

        if self.plot.timebase is not None and len(self.plot.timebase):
            timebase = self.plot.timebase
            dim = len(timebase)

            right = np.searchsorted(timebase, start, side="right")
            if right == 0:
                next_pos = timebase[0]
            elif right == dim:
                next_pos = timebase[-1]
            else:
                if start - timebase[right - 1] < timebase[right] - start:
                    next_pos = timebase[right - 1]
                else:
                    next_pos = timebase[right]
            start = next_pos

            right = np.searchsorted(timebase, stop, side="right")
            if right == 0:
                next_pos = timebase[0]
            elif right == dim:
                next_pos = timebase[-1]
            else:
                if stop - timebase[right - 1] < timebase[right] - stop:
                    next_pos = timebase[right - 1]
                else:
                    next_pos = timebase[right]
            stop = next_pos

            self.plot.region.setRegion((start, stop))

    def range_removed(self):
        for i, signal in enumerate(self.plot.signals):
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            item.setPrefix("")
            item.setValue("")
            self.cursor_info.setText("")
        if self.plot.cursor1:
            self.plot.cursor_moved.emit()
        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def compute_cut_hints(self):
        # TODO : use master channel physical min and max values
        times = []
        groups_nr = len(self.mdf.groups)
        for i in range(groups_nr):
            master = self.mdf.get_master(i)
            if len(master):
                times.append(master[0])
                times.append(master[-1])
            QApplication.processEvents()

        if len(times):
            time_range = min(times), max(times)

            self.cut_start.setRange(*time_range)
            self.cut_stop.setRange(*time_range)

            self.cut_interval.setText(
                "Cut interval ({:.6f}s - {:.6f}s)".format(*time_range)
            )
        else:
            self.cut_start.setRange(0, 0)
            self.cut_stop.setRange(0, 0)

            self.cut_interval.setText("Empty measurement")

    def update_progress(self, current_index, max_index):
        self.progress = current_index, max_index

    def show_channel_info(self, item, column):
        if item and item.parent():
            group, index = item.entry

            channel = self.mdf.get_channel_metadata(group=group, index=index)

            msg = ChannelInfoDialog(channel, self)
            msg.show()

    def clear_filter(self):
        iterator = QTreeWidgetItemIterator(self.filter_tree)

        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, Qt.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def clear_channels(self):
        iterator = QTreeWidgetItemIterator(self.channels_tree)

        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, Qt.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def new_search_result(self, tree, search):
        group_index, channel_index = search.entries[search.current_index]

        grp = self.mdf.groups[group_index]
        channel_count = len(grp["channels"])

        iterator = QTreeWidgetItemIterator(tree)

        group = -1
        index = 0
        while iterator.value():
            item = iterator.value()
            if item.parent() is None:
                iterator += 1
                group += 1
                index = 0
                continue

            if group == group_index:

                if (
                    channel_index >= 0
                    and index == channel_index
                    or channel_index < 0
                    and index == -channel_index - 1 + channel_count
                ):
                    tree.scrollToItem(item, QAbstractItemView.PositionAtTop)
                    item.setSelected(True)

            index += 1
            iterator += 1

    def close(self):
        mdf_name = self.mdf.name
        self.mdf.close()
        if self.file_name.lower().endswith("dl3"):
            os.remove(mdf_name)

    def convert(self, event):
        version = self.convert_format.currentText()

        memory = self.memory

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.convert_split.checkState() == Qt.Checked
        if split:
            split_size = int(self.convert_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.convert_compression.currentIndex()

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
                title="Converting measurement",
                message='Converting "{}" from {} to {} '.format(
                    self.file_name, self.mdf.version, version
                ),
                icon_name="convert",
            )

            # convert self.mdf
            target = self.mdf.convert
            kwargs = {"version": version, "memory": memory}

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

            # then save it
            progress.setLabelText('Saving converted file "{}"'.format(file_name))

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

    def resample(self, event):
        version = self.resample_format.currentText()
        raster = self.raster.value()
        memory = self.memory

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.resample_split.checkState() == Qt.Checked
        if split:
            split_size = int(self.resample_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.resample_compression.currentIndex()

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
                title="Resampling measurement",
                message='Resampling "{}" to {}s raster '.format(self.file_name, raster),
                icon_name="resample",
            )

            # resample self.mdf
            target = self.mdf.resample
            kwargs = {"raster": raster, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            # convert mdf
            progress.setLabelText(
                "Converting from {} to {}".format(mdf.version, version)
            )

            target = mdf.convert
            kwargs = {"to": version, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=33,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText('Saving resampled file "{}"'.format(file_name))

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=34,
                offset=66,
                progress=progress,
            )

    def cut(self, event):
        version = self.cut_format.currentText()
        start = self.cut_start.value()
        stop = self.cut_stop.value()
        memory = self.memory
        if self.whence.checkState() == Qt.Checked:
            whence = 1
        else:
            whence = 0

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.cut_split.checkState() == Qt.Checked
        if split:
            split_size = int(self.cut_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.cut_compression.currentIndex()

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
                title="Cutting measurement",
                message='Cutting "{}" from {}s to {}s'.format(
                    self.file_name, start, stop
                ),
                icon_name="cut",
            )

            # cut self.mdf
            target = self.mdf.cut
            kwargs = {"start": start, "stop": stop, "whence": whence}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            # convert mdf
            progress.setLabelText(
                "Converting from {} to {}".format(mdf.version, version)
            )

            target = mdf.convert
            kwargs = {"to": version, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=33,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText('Saving cut file "{}"'.format(file_name))

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=34,
                offset=66,
                progress=progress,
            )

    def export(self, event):
        export_type = self.export_type.currentText()

        single_time_base = self.single_time_base.checkState() == Qt.Checked
        time_from_zero = self.time_from_zero.checkState() == Qt.Checked
        use_display_names = self.use_display_names.checkState() == Qt.Checked
        empty_channels = self.empty_channels.currentText()
        mat_format = self.mat_format.currentText()
        raster = self.export_raster.value()
        oned_as = self.oned_as.currentText()

        filters = {
            "csv": "CSV files (*.csv)",
            "excel": "Excel files (*.xlsx)",
            "hdf5": "HDF5 files (*.hdf)",
            "mat": "Matlab MAT files (*.mat)",
            "parquet": "Apache Parquet files (*.parquet)",
        }

        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select export file", "", filters[export_type]
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select export file", "", filters[export_type]
            )
            file_name = str(file_name)

        if file_name:
            thr = Thread(
                target=self.mdf.export,
                kwargs={
                    "fmt": export_type,
                    "filename": file_name,
                    "single_time_base": single_time_base,
                    "use_display_names": use_display_names,
                    "time_from_zero": time_from_zero,
                    "empty_channels": empty_channels,
                    "format": mat_format,
                    "raster": raster,
                    "oned_as": oned_as,
                },
            )

            progress = QProgressDialog(
                "Exporting to {} ...".format(export_type), "Abort export", 0, 100
            )
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.setAutoClose(True)
            progress.setWindowTitle("Running export")
            icon = QIcon()
            icon.addPixmap(QPixmap(":/export.png"), QIcon.Normal, QIcon.Off)
            progress.setWindowIcon(icon)

            thr.start()

            cntr = 0

            while thr.is_alive():
                cntr += 1
                progress.setValue(cntr % 98)
                sleep(0.1)

            progress.cancel()

    def plot_pyqtgraph(self, event):
        try:
            iter(event)
            signals = event
        except:

            iterator = QTreeWidgetItemIterator(self.channels_tree)

            group = -1
            index = 0
            signals = []
            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    group += 1
                    index = 0
                    continue

                if item.checkState(0) == Qt.Checked:
                    group, index = item.entry
                    signals.append((None, group, index))

                index += 1
                iterator += 1

            signals = self.mdf.select(signals)

            signals = [
                sig
                for sig in signals
                if not sig.samples.dtype.names and len(sig.samples.shape) <= 1
            ]

        count = self.channel_selection.count()
        for i in range(count):
            self.channel_selection.takeItem(0)

        if self.info:
            self.info.setParent(None)
            self.info = None

        self.plot = Plot(signals, self.with_dots, self.step_mode, self)
        self.plot.range_modified.connect(self.range_modified)
        self.plot.range_removed.connect(self.range_removed)
        self.plot.range_modified_finished.connect(self.range_modified_finished)
        self.plot.cursor_removed.connect(self.cursor_removed)
        self.plot.cursor_moved.connect(self.cursor_moved)
        self.plot.cursor_move_finished.connect(self.cursor_move_finished)
        self.plot.xrange_changed.connect(self.xrange_changed)
        self.plot.show()

        for i, sig in enumerate(self.plot.signals):
            if sig.empty:
                name = "{} [has no samples]".format(sig.name)
            else:
                name = "{} ({})".format(sig.name, sig.unit)
            is_float = sig.samples.dtype.kind == "f"
            item = QListWidgetItem(self.channel_selection)
            it = ChannelDisplay(i, sig.unit, self)
            it.setAttribute(Qt.WA_StyledBackground)

            it.setName(name)
            it.setValue("")
            it.setColor(sig.color)
            item.setSizeHint(it.sizeHint())
            self.channel_selection.addItem(item)
            self.channel_selection.setItemWidget(item, it)

            it.color_changed.connect(self.plot.setColor)
            it.enable_changed.connect(self.plot.setSignalEnable)

        if self.splitter.count() > 1:
            old_plot = self.splitter.widget(1)
            old_plot.setParent(None)
            old_plot.hide()
        self.splitter.addWidget(self.plot)

        width = sum(self.splitter.sizes())

        self.splitter.setSizes((0.2 * width, 0.8 * width))
        QApplication.processEvents()

        self.plot.update_lines(force=True)

    def filter(self, event):
        iterator = QTreeWidgetItemIterator(self.filter_tree)
        memory = self.memory

        group = -1
        index = 0
        channels = []
        while iterator.value():
            item = iterator.value()
            if item.parent() is None:
                iterator += 1
                group += 1
                index = 0
                continue

            if item.checkState(0) == Qt.Checked:
                channels.append((None, group, index))

            index += 1
            iterator += 1

        version = self.filter_format.itemText(self.filter_format.currentIndex())

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.filter_split.checkState() == Qt.Checked
        if split:
            split_size = int(self.filter_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.filter_compression.currentIndex()

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
                title="Filtering measurement",
                message='Filtering selected channels from "{}"'.format(self.file_name),
                icon_name="filter",
            )

            # filtering self.mdf
            target = self.mdf.filter
            kwargs = {"channels": channels, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            # convert mdf
            progress.setLabelText(
                "Converting from {} to {}".format(mdf.version, version)
            )

            target = mdf.convert
            kwargs = {"to": version, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=33,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText('Saving filtered file "{}"'.format(file_name))

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=34,
                offset=66,
                progress=progress,
            )


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):

        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "main_window.ui"), self)

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


def main():

    app = QApplication(sys.argv)
    main = MainWindow()
    app.exec_()


if __name__ == "__main__":
    main()

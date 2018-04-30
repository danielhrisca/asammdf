import os
import sys
import traceback

from copy import deepcopy
from datetime import datetime
from functools import reduce, partial
from io import StringIO
from threading import Thread
from time import sleep


import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from asammdf import MDF, MDF2, MDF3, MDF4, SUPPORTED_VERSIONS
from asammdf import __version__ as libversion

import asammdfgui.main_window as main_window
import asammdfgui.file_widget as file_widget
import asammdfgui.search_widget as search_widget
import asammdfgui.channel_info_widget as channel_info_widget

from pyqtgraph import PlotWidget, AxisItem, ViewBox
import pyqtgraph as pg


__version__ = '0.1.0'
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
    separator = '-' * 80
    notice = 'The following error was triggered:'

    now = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")

    info = StringIO()
    traceback.print_tb(tracebackobj, None, info)
    info.seek(0)
    info = info.read()

    errmsg = '{}\t \n{}'.format(exc_type, exc_value)
    sections = [now, separator, errmsg, separator, info]
    msg = '\n'.join(sections)

    print(msg)

    QMessageBox.warning(
        None,
        notice,
        msg,
    )


sys.excepthook = excepthook


def run_thread_with_progress(
        widget,
        target,
        kwargs,
        factor,
        offset,
        progress):
    termination_request = False

    thr = WorkerThread(
        target=target,
        kwargs=kwargs,
    )

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
    progress = QProgressDialog(
        message,
        "",
        0,
        100,
        parent,
    )

    progress.setWindowModality(Qt.ApplicationModal)
    progress.setCancelButton(None)
    progress.setAutoClose(True)
    progress.setWindowTitle(title)
    icon = QIcon()
    icon.addPixmap(
        QPixmap(":/{}.png".format(icon_name)),
        QIcon.Normal,
        QIcon.Off,
    )
    progress.setWindowIcon(icon)
    progress.show()

    return progress


class WorkerThread(Thread):
    def __init__(self, output=None, *args, **kargs):
        super(WorkerThread, self).__init__(*args, **kargs)
        self.output = None
        self.error = ''

    def run(self):
        try:
            self.output = self._target(*self._args, **self._kwargs)
        except Exception as err:
            self.error = err


class TreeItem(QTreeWidgetItem):

    def __init__(self, entry, *args, **kwargs):

        super(TreeItem, self).__init__(*args, **kwargs)

        self.entry = entry


class Plot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):

        super(Plot, self).__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F:
            print('fit now')
        else:
            super(Plot, self).keyPressEvent(event)


class TreeWidget(QTreeWidget):

    def __init__(self, *args, **kwargs):

        super(TreeWidget, self).__init__(*args, **kwargs)

        self.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )

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

    def __init__(self, *args, **kwargs):

        super(ListWidget, self).__init__(*args, **kwargs)

        self.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )

        self.setAlternatingRowColors(True)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Delete:
            selected_items = self.selectedItems()
            for item in selected_items:
                row = self.row(item)
                self.takeItem(row)
        else:
            super(ListWidget, self).keyPressEvent(event)


class ChannelInfoWidget(QWidget, channel_info_widget.Ui_ChannelInfo):
    def __init__(self, channel, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.channel_label.setText(
            channel.metadata()
        )

        if channel.conversion:
            self.conversion_label.setText(
                channel.conversion.metadata()
            )

        if channel.source:
            self.source_label.setText(
                channel.source.metadata()
            )


class ChannelInfoDialog(QDialog):
    def __init__(self, channel, *args, **kwargs):
        super(QDialog, self).__init__(*args, **kwargs)

        self.setWindowFlags(Qt.Window)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle(channel.name)

        layout.addWidget(ChannelInfoWidget(channel, self))

        self.setStyleSheet("font: 8pt \"Consolas\";}")

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/info.png"),
            QIcon.Normal,
            QIcon.Off,
        )

        self.setWindowIcon(icon)
        self.setGeometry(
            240,
            60,
            1200,
            600,
        )

        screen = QApplication.desktop().screenGeometry()
        self.move(
            (screen.width() - 1200) // 2,
            (screen.height() - 600) // 2,
        )


class SearchWidget(QWidget, search_widget.Ui_SearchWidget):

    selectionChanged = pyqtSignal()

    def __init__(self, channels_db, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.channels_db = channels_db

        self.matches = 0
        self.current_index = 1
        self.entries = []

        completer = QCompleter(
            sorted(self.channels_db, key=lambda x: x.lower()),
            self,
        )
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setModelSorting(QCompleter.CaseInsensitivelySortedModel)
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
            self.label.setText('{} of {}'.format(self.current_index + 1, self.matches))
            self.selectionChanged.emit()

    def up(self, event):
        if self.matches:
            self.current_index -= 1
            if self.current_index < 0:
                self.current_index = self.matches - 1
            self.label.setText('{} of {}'.format(self.current_index + 1, self.matches))
            self.selectionChanged.emit()

    def set_search_option(self, option):
        if option == 'Match start':
            self.search.completer().setFilterMode(Qt.MatchStartsWith)
        elif option == 'Match contains':
            self.search.completer().setFilterMode(Qt.MatchContains)

    def display_results(self, text):
        channel_name = text.strip()
        if channel_name in self.channels_db:
            self.entries = self.channels_db[channel_name]
            self.matches = len(self.entries)
            self.label.setText('1 of {}'.format(self.matches))
            self.current_index = 0
            self.selectionChanged.emit()

        else:
            self.label.setText('No match')
            self.matches = 0
            self.current_index = 0
            self.entries = []


class FileWidget(QWidget, file_widget.Ui_file_widget):
    def __init__(self, file_name, memory, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.file_name = file_name
        self.progress = None
        self.mdf = None
        self.memory = memory

        progress = QProgressDialog(
            'Opening "{}"'.format(self.file_name),
            "",
            0,
            100,
            self.parent(),
        )

        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setAutoClose(True)
        progress.setWindowTitle('Opening measurement')
        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/open.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        progress.setWindowIcon(icon)
        progress.show()

        target = MDF
        kwargs = {
            'name': file_name,
            'memory': memory,
            'callback': self.update_progress,
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

        progress.setLabelText('Loading graphical elements')

        progress.setValue(35)

        self.filter_field = SearchWidget(
            deepcopy(self.mdf.channels_db),
            self,
        )

        progress.setValue(37)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)

        channel_and_search = QWidget(splitter)

        self.channels_tree = TreeWidget(channel_and_search)
        self.search_field = SearchWidget(
            deepcopy(self.mdf.channels_db),
            channel_and_search,
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
                self.new_search_result,
                tree=self.filter_tree,
                search=self.filter_field,
            )
        )

        selection_list = QWidget(splitter)
        self.channel_selection = ListWidget(selection_list)
        self.channel_selection.itemSelectionChanged.connect(
            self.update_graph
        )

        vbox = QVBoxLayout(selection_list)
        vbox.addWidget(QLabel('Selected channels'))
        vbox.addWidget(self.channel_selection)

        vbox = QVBoxLayout(channel_and_search)
        vbox.addWidget(self.search_field)
        vbox.addWidget(self.channels_tree)

        self.filter_layout.addWidget(self.filter_field, 0, 0, 1, 1)

        self.channels_tree.itemDoubleClicked.connect(self.show_channel_info)
        self.filter_tree.itemDoubleClicked.connect(self.show_channel_info)

        self.channels_layout.insertWidget(0, splitter)
        self.filter_layout.addWidget(self.filter_tree, 1, 0, 8, 1)

        groups_nr = len(self.mdf.groups)

        self.channels_tree.setHeaderLabel('Channels')
        self.filter_tree.setHeaderLabel('Channels')

        for i, group in enumerate(self.mdf.groups):
            channel_group = QTreeWidgetItem()
            filter_channel_group = QTreeWidgetItem()
            channel_group.setText(
                0,
                "Channel group {}".format(i)
            )
            filter_channel_group.setText(
                0,
                "Channel group {}".format(i)
            )
            channel_group.setFlags(
                channel_group.flags()
                | Qt.ItemIsTristate
                | Qt.ItemIsUserCheckable
            )
            filter_channel_group.setFlags(
                channel_group.flags()
                | Qt.ItemIsTristate
                | Qt.ItemIsUserCheckable
            )

            self.channels_tree.addTopLevelItem(channel_group)
            self.filter_tree.addTopLevelItem(filter_channel_group)

            for j, ch in enumerate(group['channels']):

                name = self.mdf.get_channel_name(group=i, index=j)
                channel = TreeItem((i, j), channel_group)
                channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                channel.setText(0, name)
                channel.setCheckState(0, Qt.Unchecked)

                channel = TreeItem((i, j), filter_channel_group)
                channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                channel.setText(0, name)
                channel.setCheckState(0, Qt.Unchecked)

            if self.mdf.version >= '4.00':
                for j, ch in enumerate(group['logging_channels'], 1):
                    name = ch.name

                    channel = TreeItem((i, -j), channel_group)
                    channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                    channel.setText(0, name)
                    channel.setCheckState(0, Qt.Unchecked)

                    channel = TreeItem((i, -j), filter_channel_group)
                    channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                    channel.setText(0, name)
                    channel.setCheckState(0, Qt.Unchecked)

            progress.setValue(37 + int(53 * (i+1) / groups_nr))
            QApplication.processEvents()

        progress.setValue(90)

        self.resample_format.insertItems(
            0,
            SUPPORTED_VERSIONS,
        )
        index = self.resample_format.findText(self.mdf.version)
        if index >= 0:
            self.resample_format.setCurrentIndex(index)
        self.resample_compression.insertItems(
            0,
            (
                'no compression',
                'deflate',
                'transposed deflate',
            ),
        )
        self.resample_split_size.setValue(10)
        self.resample_btn.clicked.connect(self.resample)

        self.filter_format.insertItems(
            0,
            SUPPORTED_VERSIONS,
        )
        index = self.filter_format.findText(self.mdf.version)
        if index >= 0:
            self.filter_format.setCurrentIndex(index)
        self.filter_compression.insertItems(
            0,
            (
                'no compression',
                'deflate',
                'transposed deflate',
            ),
        )
        self.filter_split_size.setValue(10)
        self.filter_btn.clicked.connect(self.filter)

        self.convert_format.insertItems(
            0,
            SUPPORTED_VERSIONS,
        )
        self.convert_compression.insertItems(
            0,
            (
                'no compression',
                'deflate',
                'transposed deflate',
            ),
        )
        self.convert_split_size.setValue(10)
        self.convert_btn.clicked.connect(self.convert)

        self.cut_format.insertItems(
            0,
            SUPPORTED_VERSIONS,
        )
        index = self.cut_format.findText(self.mdf.version)
        if index >= 0:
            self.cut_format.setCurrentIndex(index)
        self.cut_compression.insertItems(
            0,
            (
                'no compression',
                'deflate',
                'transposed deflate',
            ),
        )
        self.cut_split_size.setValue(10)
        self.cut_btn.clicked.connect(self.cut)

        self.cut_interval.setText(
            'Unknown measurement interval'
        )

        progress.setValue(99)

        self.empty_channels.insertItems(
            0,
            (
                'zeros',
                'skip',
            ),
        )
        self.mat_format.insertItems(
            0,
            (
                '4',
                '5',
                '7.3',
            ),
        )
        self.export_type.insertItems(
            0,
            (
                'csv',
                'excel',
                'hdf5',
                'mat',
            ),
        )
        self.export_btn.clicked.connect(self.export)

        # self.channels_tree.itemChanged.connect(self.select)
        self.plot_btn.clicked.connect(self.plot_pyqtgraph)
        self.clear_filter_btn.clicked.connect(self.clear_filter)
        self.clear_channels_btn.clicked.connect(self.clear_channels)

        self.aspects.setCurrentIndex(0)

        progress.setValue(100)

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
                'Cut interval ({:.6f}s - {:.6f}s)'.format(
                    *time_range
                )
            )
        else:
            self.cut_start.setRange(0, 0)
            self.cut_stop.setRange(0, 0)

            self.cut_interval.setText('Empty measurement')

    def update_progress(self, current_index, max_index):
        self.progress = current_index, max_index

    def show_channel_info(self, item, column):
        if item and item.parent():
            group, index = item.entry

            channel = self.mdf.get_channel_metadata(
                group=group,
                index=index,
            )

            msg = ChannelInfoDialog(channel, self)
            msg.show()

    def clear_filter(self):
        iterator = QTreeWidgetItemIterator(
            self.filter_tree,
        )

        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, Qt.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def clear_channels(self):
        iterator = QTreeWidgetItemIterator(
            self.channels_tree,
        )

        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, Qt.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def new_search_result(self, tree, search):
        group_index, channel_index = search.entries[search.current_index]

        grp = self.mdf.groups[group_index]
        channel_count = len(grp['channels'])

        iterator = QTreeWidgetItemIterator(
            tree,
        )

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

                if channel_index >= 0 and index == channel_index \
                        or channel_index < 0 and index == -channel_index -1 + channel_count:
                    tree.scrollToItem(
                        item,
                        QAbstractItemView.PositionAtTop,
                    )
                    item.setSelected(True)

            index += 1
            iterator += 1

    def close(self):
        self.mdf.close()

    def convert(self, event):
        version = self.convert_format.currentText()

        memory = self.memory

        if version < '4.00':
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

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            '',
            filter,
        )

        if file_name:

            progress = setup_progress(
                parent=self,
                title='Converting measurement',
                message='Converting "{}" from {} to {} '.format(
                    self.file_name,
                    self.mdf.version,
                    version,
                ),
                icon_name='convert',
            )

            # convert self.mdf
            target = self.mdf.convert
            kwargs = {
                'to': version,
                'memory': memory,
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

            # then save it
            progress.setLabelText(
                'Saving resampled file "{}"'.format(
                    file_name,
                )
            )

            target = mdf.save
            kwargs = {
                'dst': file_name,
                'compression': compression,
                'overwrite': True,
            }

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

        if version < '4.00':
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

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            '',
            filter,
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title='Resampling measurement',
                message='Resampling "{}" to {}s raster '.format(
                    self.file_name,
                    raster,
                ),
                icon_name='resample',
            )

            # resample self.mdf
            target = self.mdf.resample
            kwargs = {
                'raster': raster,
                'memory': memory,
            }

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
                'Converting from {} to {}'.format(
                    mdf.version,
                    version,
                )
            )

            target = mdf.convert
            kwargs = {
                'to': version,
                'memory': memory,
            }

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
            progress.setLabelText(
                'Saving resampled file "{}"'.format(
                    file_name,
                )
            )

            target = mdf.save
            kwargs = {
                'dst': file_name,
                'compression': compression,
                'overwrite': True,
            }

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

        if version < '4.00':
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

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            '',
            filter,
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title='Cutting measurement',
                message='Cutting "{}" from {}s to {}s'.format(
                    self.file_name,
                    start,
                    stop,
                ),
                icon_name='cut',
            )

            # cut self.mdf
            target = self.mdf.cut
            kwargs = {
                'start': start,
                'stop': stop,
            }

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
                'Converting from {} to {}'.format(
                    mdf.version,
                    version,
                )
            )

            target = mdf.convert
            kwargs = {
                'to': version,
                'memory': memory,
            }

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
            progress.setLabelText(
                'Saving cut file "{}"'.format(
                    file_name,
                )
            )

            target = mdf.save
            kwargs = {
                'dst': file_name,
                'compression': compression,
                'overwrite': True,
            }

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

        filters = {
            'csv': "CSV files (*.csv)",
            'excel': "Excel files (*.xlsx)",
            'hdf5': "HDF5 files (*.hdf)",
            'mat': "Matlab MAT files (*.mat)",
        }

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select export file",
            '',
            filters[export_type],
        )

        if file_name:
            thr = Thread(
                target=self.mdf.export,
                kwargs={
                    'fmt': export_type,
                    'filename': file_name,
                    'single_time_base': single_time_base,
                    'use_display_names': use_display_names,
                    'time_from_zero': time_from_zero,
                    'empty_channels': empty_channels,
                    'format': mat_format,
                    'raster': raster,
                },
            )

            progress = QProgressDialog(
                "Exporting to {} ...".format(export_type),
                "Abort export",
                0,
                100)
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.setAutoClose(True)
            progress.setWindowTitle('Running export')
            icon = QIcon()
            icon.addPixmap(
                QPixmap(":/export.png"),
                QIcon.Normal,
                QIcon.Off,
            )
            progress.setWindowIcon(icon)

            thr.start()

            cntr = 0

            while thr.is_alive():
                cntr += 1
                progress.setValue(cntr % 98)
                sleep(0.1)

            progress.cancel()

    def plot(self, event):

        # QtWidgets.QMainWindow.__init__(self)
        # self.widget = QtWidgets.QWidget()
        # self.setCentralWidget(self.widget)
        # self.widget.setLayout(QtWidgets.QVBoxLayout())
        # self.widget.layout().setContentsMargins(0, 0, 0, 0)
        # self.widget.layout().setSpacing(0)
        #
        # self.fig = fig
        # self.canvas = FigureCanvas(self.fig)
        # self.canvas.draw()
        # self.scroll = QtWidgets.QScrollArea(self.widget)
        # self.scroll.setWidget(self.canvas)
        #
        # self.nav = NavigationToolbar(self.canvas, self.widget)
        # self.widget.layout().addWidget(self.nav)
        # self.widget.layout().addWidget(self.scroll)
        #
        # # self.channels_tree.itemChanged.connect(self.select)
        # self.plot_btn.clicked.connect(self.plot)
        #
        #
        #
        # self.figure = Figure()
        #
        # # this is the Canvas Widget that displays the `figure`
        # # it takes the `figure` instance as a parameter to __init__
        # self.canvas = FigureCanvas(self.figure)
        #
        # # this is the Navigation widget
        # # it takes the Canvas widget and a parent
        # self.toolbar = NavigationToolbar(self.canvas, self)
        #
        # self.ax = self.figure.add_subplot(111)
        #
        # # set the layout
        # layout = QVBoxLayout()
        # layout.addWidget(self.toolbar)
        # layout.addWidget(self.canvas)
        # self.channels_layout.addLayout(layout)
        # self.channels_layout.setStretch(0, 0)
        # self.channels_layout.setStretch(1, 1)
        #
        # # refresh canvas
        # self.canvas.draw()

        item = self.channels_grid.itemAtPosition(2, 1)
        if item:
            item.widget().setParent(None)

        iterator = QTreeWidgetItemIterator(
            self.channels_tree,
        )

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
                signals.append((group, index))

            index += 1
            iterator += 1

        rows = len(signals)

        # fig, axes = plt.subplots(ncols=1, nrows=rows, figsize=(5, 3 * rows))
        # for ax, (group, index) in zip(axes.flatten(), signals):
        #     sig = self.mdf.get(group=group, index=index)
        #     ax.plot(sig.timestamps, sig.samples, '.-')
        #     ax.set_title(sig.name)
        #     ax.set_ylabel(sig.unit)
        #
        # canvas = FigureCanvas(fig)
        # canvas.draw()
        #
        # self.plot_scroll.setWidget(canvas)
        #
        # nav = NavigationToolbar(canvas, self)
        # self.channels_grid.addWidget(nav, 1, 1)

        figure = Figure(figsize=(6, 30))


        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__

        for i, (group, index) in enumerate(signals):
            sig = self.mdf.get(group=group, index=index)
            ax = figure.add_subplot(rows, 1, i+1)
            ax.plot(sig.timestamps, sig.samples, '.-')
            ax.set_title(sig.name)
            ax.set_ylabel(sig.unit)

        # refresh canvas
        canvas = FigureCanvas(figure)
        canvas.draw()

        self.scroll_layout.addWidget(canvas)

        nav = NavigationToolbar(canvas, self)
        self.channels_grid.addWidget(nav, 2, 1)

    def update_selection_list(self):
        iterator = QTreeWidgetItemIterator(
            self.channels_tree,
        )

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
                signals.append(item.text(0))

            index += 1
            iterator += 1

        count = self.channel_selection.count()
        for i in range(count):
            self.channel_selection.takeItem(0)

        self.channel_selection.addItems(signals)

    def update_graph(self):
        signals_nr = self.channel_selection.count()
        selected_items = [
            self.channel_selection.row(item)
            for item in self.channel_selection.selectedItems()
        ]

        pw = self.splitter.widget(1)

        scene = pw.plotItem.scene()
        layout = pw.plotItem.layout
        parent_vb = pw.plotItem.vb

        plot_cntr = 1

        for i in range(signals_nr):
            item = layout.itemAt(2, i+2)
            if i in selected_items:
                item.show()
            else:
                item.hide()

            if len(selected_items) == 1:
                parent_vb.setYLink(item.linkedView())

            else:
                parent_vb.setYLink(None)



        for item in scene.items():
            if isinstance(item, pg.PlotDataItem):
                if signals_nr - plot_cntr in selected_items:
                    item.show()

                else:
                    item.hide()
                plot_cntr += 1

        pw.plotItem.showGrid(x=True, y=True)

    def plot_pyqtgraph(self, event):
        self.update_selection_list()

        iterator = QTreeWidgetItemIterator(
            self.channels_tree,
        )

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
                signals.append(item.entry)

            index += 1
            iterator += 1

        rows = len(signals)

        pw = Plot()
        pw.showGrid(x = True, y = True, alpha = 0.3)

        plot_item = pw.plotItem
        # plot_item.hideAxis('left')
        # plot_item.showGrid(True, True, 0.1)
        layout = plot_item.layout
        scene = plot_item.scene()
        vb = plot_item.vb


        # ## create a new ViewBox, link the right axis to its coordinate system
        # p2 = pg.ViewBox()
        # p1.showAxis('right')
        # p1.scene().addItem(p2)
        # p1.getAxis('right').linkToView(p2)
        # p2.setXLink(p1)
        # p1.getAxis('right').setLabel('axis2', color='#0000ff')
        #
        # ## create third ViewBox.
        # ## this time we need to create a new axis as well.
        # p3 = pg.ViewBox()
        # ax3 = pg.AxisItem('right')
        # p1.layout.addItem(ax3, 2, 3)
        # p1.scene().addItem(p3)
        # ax3.linkToView(p3)
        # p3.setXLink(p1)
        # ax3.setZValue(-10000)
        # ax3.setLabel('axis 3', color='#ff0000')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        parent_vb = vb

        view_boxes = []

        # slot: update view when resized
        def updateViews():
            for view_box in view_boxes:
                view_box.setGeometry(vb.sceneBoundingRect())
                view_box.linkedViewChanged(vb, view_box.XAxis)
            pw.showGrid(x=True, y=True, alpha=0.3)

        vb.sigResized.connect(updateViews)

        for i, (group, index) in enumerate(signals):
            sig = self.mdf.get(group=group, index=index)

            axis = pg.AxisItem("left")

            view_box = pg.ViewBox()

            axis.linkToView(view_box)
            axis.setLabel(sig.name, sig.unit, color=colors[i%10])

            layout.addItem(axis, 2, i+2)

            scene.addItem(view_box)

            conditions = [
                not sig.samples.dtype.names,
                sig.samples.dtype.kind not in 'SV',
                len(sig.samples.shape) <= 1,
            ]

            if all(conditions):
                color = colors[i%10]
                curve = pg.PlotDataItem(
                    sig.timestamps,
                    sig.samples,
                    pen=color,
                    symbolBrush=color,
                    symbolPen='w',
                    symbol='o',
                    symbolSize=2,
                )

                view_box.addItem(
                    curve
                )

            view_box.setXLink(parent_vb)
            view_box.enableAutoRange(
                axis=pg.ViewBox.XYAxes,
                enable=True,
            )

            view_boxes.append(view_box)
            parent_vb = view_box

        updateViews()

        pw.show()

        if self.splitter.count() == 1:
            self.splitter.addWidget(pw)
        else:
            self.splitter.replaceWidget(1, pw)

        width = sum(self.splitter.sizes())

        self.splitter.setSizes(
            (0.2 * width, 0.8*width)
        )

        for i in range(layout.columnCount()):
            item = layout.itemAt(2, i)
            print(item)

    def filter(self, event):
        iterator = QTreeWidgetItemIterator(
            self.filter_tree,
        )
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

        version = self.filter_format.itemText(
            self.filter_format.currentIndex())

        if version < '4.00':
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

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            '',
            filter,
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title='Filtering measurement',
                message='Filtering selected channels from "{}"'.format(
                    self.file_name,
                ),
                icon_name='filter',
            )

            # filtering self.mdf
            target = self.mdf.filter
            kwargs = {
                'channels': channels,
                'memory': memory,
            }

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
                'Converting from {} to {}'.format(
                    mdf.version,
                    version,
                )
            )

            target = mdf.convert
            kwargs = {
                'to': version,
                'memory': memory,
            }

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
            progress.setLabelText(
                'Saving filtered file "{}"'.format(
                    file_name,
                )
            )

            target = mdf.save
            kwargs = {
                'dst': file_name,
                'compression': compression,
                'overwrite': True,
            }

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=34,
                offset=66,
                progress=progress,
            )


class MainWindow(QMainWindow, main_window.Ui_PyMDFMainWindow):
    def __init__(self, parent=None):

        super().__init__(parent)

        self.progress = None

        self.last_folder = ''
        self.setupUi(self)

        self.open_file_btn.clicked.connect(self.open_file)
        self.open_multiple_files_btn.clicked.connect(self.open_multiple_files)
        self.files.tabCloseRequested.connect(self.close_file)

        self.concatenate.toggled.connect(self.function_select)
        self.cs_btn.clicked.connect(self.cs_clicked)
        self.cs_format.insertItems(
            0,
            SUPPORTED_VERSIONS,
        )
        self.cs_compression.insertItems(
            0,
            (
                'no compression',
                'deflate',
                'transposed deflate',
            ),
        )
        self.cs_split_size.setValue(10)

        self.files_list = ListWidget(self)
        self.files_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.files_layout.addWidget(self.files_list, 1, 0, 1, 2)
        self.files_list.itemDoubleClicked.connect(self.delete_item)

        self.statusbar.addPermanentWidget(
            QLabel(
                'asammdfgui {} with asammdf {}'.format(
                    __version__,
                    libversion,
                )
            )
        )

        # memory option menu
        memory_option = QActionGroup(self)

        for option in (
                'full',
                'low',
                'minimum'):

            action = QAction(option)
            action.setCheckable(True)
            memory_option.addAction(action)
            action.triggered.connect(
                partial(self.set_memory_option, option)
            )

            if option == 'minimum':
                action.setChecked(True)

        menu = QMenu('Memory', self)
        menu.addActions(memory_option.actions())
        self.menubar.addMenu(menu)

        # search mode menu
        search_option = QActionGroup(self)

        for option in (
                'Match start',
                'Match contains'):

            action = QAction(option)
            action.setCheckable(True)
            search_option.addAction(action)
            action.triggered.connect(
                partial(self.set_search_option, option)
            )

            if option == 'Match start':
                action.setChecked(True)

        menu = QMenu('Search', self)
        menu.addActions(search_option.actions())
        self.menubar.addMenu(menu)

        self.memory = 'minimum'
        self.match = 'Match start'
        self.toolBox.setCurrentIndex(0)

        self.show()

    def set_memory_option(self, option):
        self.memory = option

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
            self.cs_btn.setText('Concatenate')
        else:
            self.cs_btn.setText('Stack')

    def cs_clicked(self, event):
        if self.concatenate.isChecked():
            func = MDF.concatenate
            operation = 'Concatenating'
        else:
            func = MDF.stack
            operation = 'Stacking'

        version = self.cs_format.currentText()

        memory = self.memory

        if version < '4.00':
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

        files = [
            self.files_list.item(row).text()
            for row in range(count)
        ]

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            '',
            filter,
        )

        if file_name:

            progress = setup_progress(
                parent=self,
                title='{} measurements'.format(operation),
                message='{} files and saving to {} format'.format(
                    operation,
                    version,
                ),
                icon_name='stack',
            )

            target = func
            kwargs = {
                'files': files,
                'outversion': version,
                'memory': memory,
                'callback': self.update_progress,
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
            progress.setLabelText(
                'Saving output file "{}"'.format(
                    file_name,
                )
            )

            target = mdf.save
            kwargs = {
                'dst': file_name,
                'compression': compression,
                'overwrite': True,
            }

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
            self.last_folder,
            "MDF files (*.dat *.mdf *.mf4)",
        )
        if file_names:
            self.files_list.addItems(file_names)
            count = self.files_list.count()

            icon = QIcon()
            icon.addPixmap(
                QPixmap(":/file.png"),
                QIcon.Normal,
                QIcon.Off,
            )

            for row in range(count):
                self.files_list.item(row).setIcon(icon)

    def open_file(self, event):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select measurement file",
            self.last_folder,
            "MDF files (*.dat *.mdf *.mf4)",
        )
        if file_name:
            index = self.files.count()
            try:
                widget = FileWidget(file_name, self.memory)
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

        self.files.removeTab(index)
        if self.files.count():
            self.files.setCurrentIndex(0)

    def closeEvent(self, event):
        count = self.files.count()
        for i in range(count):
            self.files.tabCloseRequested.emit(0)
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F:
            print('fit main now')
        else:
            super(MainWindow, self).keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    main = MainWindow()
    app.exec_()


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
from datetime import datetime
from functools import partial, reduce
import json
from pathlib import Path
import os

import psutil
from natsort import natsorted
import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from ..ui import resource_rc as resource_rc
from ..ui.file_widget import Ui_file_widget

from ...mdf import MDF, SUPPORTED_VERSIONS
from ...signal import Signal
from ...blocks.utils import MdfException, extract_cncomment_xml, csv_bytearray2hex
from ..utils import TERMINATED, run_thread_with_progress, setup_progress
from .plot import Plot
from .numeric import Numeric
from .tabular import Tabular
from .search import SearchWidget
from .tree import TreeWidget
from .tree_item import TreeItem
from .mdi_area import MdiAreaWidget

from ..dialogs.advanced_search import AdvancedSearch
from ..dialogs.channel_info import ChannelInfoDialog
from ..dialogs.channel_group_info import ChannelGroupInfoDialog


class FileWidget(Ui_file_widget, QtWidgets.QWidget):

    open_new_file = QtCore.pyqtSignal(str)

    def __init__(
        self,
        file_name,
        with_dots,
        subplots=False,
        subplots_link=False,
        ignore_value2text_conversions=False,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self._settings = QtCore.QSettings()

        file_name = Path(file_name)
        self.subplots = subplots
        self.subplots_link = subplots_link
        self.ignore_value2text_conversions = ignore_value2text_conversions
        self._viewbox = pg.ViewBox()
        self._viewbox.setXRange(0, 10)

        self.file_name = file_name
        self.progress = None
        self.mdf = None
        self.info_index = None
        self.with_dots = with_dots

        self._window_counter = 1

        progress = QtWidgets.QProgressDialog(
            f'Opening "{self.file_name}"', "", 0, 100, self.parent()
        )

        progress.setWindowModality(QtCore.Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setAutoClose(True)
        progress.setWindowTitle("Opening measurement")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        progress.setWindowIcon(icon)
        progress.show()

        if file_name.suffix.lower() == ".erg":
            progress.setLabelText("Converting from erg to mdf")
            try:
                from mfile import ERG

                self.mdf = ERG(file_name).export_mdf()
            except Exception as err:
                print(err)
                return
        else:

            if file_name.suffix.lower() == ".dl3":
                progress.setLabelText("Converting from dl3 to mdf")
                datalyser_active = any(
                    proc.name() == 'Datalyser3.exe'
                    for proc in psutil.process_iter()
                )
                try:
                    import win32com.client

                    index = 0
                    while True:
                        mdf_name = file_name.with_suffix(f".{index}.mdf")
                        if mdf_name.exists():
                            index += 1
                        else:
                            break


                    datalyser = win32com.client.Dispatch("Datalyser3.Datalyser3_COM")
                    if not datalyser_active:
                        try:
                            datalyser.DCOM_set_datalyser_visibility(False)
                        except:
                            pass
                    datalyser.DCOM_convert_file_mdf_dl3(file_name, str(mdf_name), 0)
                    if not datalyser_active:
                        datalyser.DCOM_TerminateDAS()
                    file_name = mdf_name
                except Exception as err:
                    print(err)
                    return

            target = MDF
            kwargs = {"name": file_name, "callback": self.update_progress}

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

        channels_db_items = sorted(self.mdf.channels_db, key = lambda x: x.lower())

        progress.setLabelText("Loading graphical elements")

        progress.setValue(35)

        self.filter_field = SearchWidget(channels_db_items, self.mdf.channels_db, self)

        progress.setValue(37)

        splitter = QtWidgets.QSplitter(self)
        splitter.setOrientation(QtCore.Qt.Vertical)

        channel_and_search = QtWidgets.QWidget(splitter)

        self.channel_view = QtWidgets.QComboBox()
        self.channel_view.addItems(['Natural sort', 'Internal file structure'])
        self.channel_view.setCurrentIndex(0)
        self.channel_view.currentIndexChanged.connect(self._update_channel_tree)

        self.channels_tree = TreeWidget(channel_and_search)
        self.channels_tree.setDragEnabled(True)
        self.search_field = SearchWidget(
            channels_db_items, self.mdf.channels_db, channel_and_search
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

        vbox = QtWidgets.QVBoxLayout(channel_and_search)
        vbox.setSpacing(2)
        self.advanced_search_btn = QtWidgets.QPushButton("", channel_and_search)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/search.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.advanced_search_btn.setIcon(icon)
        self.advanced_search_btn.setToolTip("Advanced search and select channels")
        self.advanced_search_btn.clicked.connect(self.search)
        vbox.addWidget(self.search_field)
        vbox.addWidget(self.channel_view)

        vbox.addWidget(self.channels_tree, 1)

        hbox = QtWidgets.QHBoxLayout()

        self.clear_channels_btn = QtWidgets.QPushButton("", channel_and_search)
        self.clear_channels_btn.setToolTip("Reset selection")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/erase.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.clear_channels_btn.setIcon(icon)
        self.clear_channels_btn.setObjectName("clear_channels_btn")

        self.load_channel_list_btn = QtWidgets.QPushButton("", channel_and_search)
        self.load_channel_list_btn.setToolTip("Load channel selection list")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.load_channel_list_btn.setIcon(icon1)
        self.load_channel_list_btn.setObjectName("load_channel_list_btn")

        self.save_channel_list_btn = QtWidgets.QPushButton("", channel_and_search)
        self.save_channel_list_btn.setToolTip("Save channel selection list")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_channel_list_btn.setIcon(icon2)
        self.save_channel_list_btn.setObjectName("save_channel_list_btn")

        self.select_all_btn = QtWidgets.QPushButton("", channel_and_search)
        self.select_all_btn.setToolTip("Select all channels")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/checkmark.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.select_all_btn.setIcon(icon1)

        hbox.addWidget(self.load_channel_list_btn)
        hbox.addWidget(self.save_channel_list_btn)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        hbox.addWidget(line)
        hbox.addWidget(self.select_all_btn)
        hbox.addWidget(self.clear_channels_btn)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        hbox.addWidget(line)
        hbox.addWidget(self.advanced_search_btn)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        hbox.addWidget(line)
        self.create_window_btn = QtWidgets.QPushButton("", channel_and_search)
        self.create_window_btn.setToolTip("Create window using the selected channels")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/graph.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.create_window_btn.setIcon(icon3)
        self.create_window_btn.setObjectName("create_window_btn")
        hbox.addWidget(self.create_window_btn)

        hbox.addSpacerItem(
            QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        )
        vbox.addLayout(hbox)

        self.mdi_area = MdiAreaWidget()
        self.mdi_area.add_window_request.connect(self.add_window)
        self.mdi_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.mdi_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.splitter.addWidget(self.mdi_area)

        self.filter_layout.addWidget(self.filter_field, 0, 0, 1, 1)

        self.channels_tree.itemDoubleClicked.connect(self.show_info)
        self.filter_tree.itemDoubleClicked.connect(self.show_info)

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

            filter_channel_group = QtWidgets.QTreeWidgetItem()
            filter_channel_group.setText(0, f"Channel group {i}")
            filter_channel_group.setFlags(
                filter_channel_group.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable
            )

            self.filter_tree.addTopLevelItem(filter_channel_group)

            filter_children = []

            for j, ch in enumerate(group.channels):
                entry = i, j

                name = ch.name

                channel = TreeItem(entry, name)
                channel.setText(0, name)
                filter_children.append(channel)

            if self.mdf.version >= "4.00":
                for j, ch in enumerate(group.logging_channels, 1):
                    name = ch.name
                    entry = i, -j

                    channel = TreeItem(entry, name)
                    channel.setText(0, name)
                    filter_children.append(channel)

            filter_channel_group.addChildren(filter_children)

            del filter_children

            progress.setValue(37 + int(53 * (i + 1) / groups_nr / 2))

        self.channel_view.setCurrentText(
            self._settings.value('channels_view', 'Internal file structure')
        )

        self.raster_channel.addItems(channels_db_items)

        self.raster_type_channel.toggled.connect(self.set_raster_type)

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

        self.extract_can_format.insertItems(0, SUPPORTED_VERSIONS)
        index = self.extract_can_format.findText(self.mdf.version)
        if index >= 0:
            self.extract_can_format.setCurrentIndex(index)
        self.extract_can_compression.insertItems(
            0, ("no compression", "deflate", "transposed deflate")
        )
        self.extract_can_btn.clicked.connect(self.extract_can_logging)
        self.extract_can_csv_btn.clicked.connect(self.extract_can_csv_logging)
        self.load_can_database_btn.clicked.connect(self.load_can_database)

        if self.mdf.version >= '4.00':
            if any(group.CAN_logging for group in self.mdf.groups):
                self.aspects.setTabEnabled(7, True)
            else:
                self.aspects.setTabEnabled(7, False)

        progress.setValue(99)

        self.empty_channels.insertItems(0, ("skip", "zeros"))
        self.empty_channels_can.insertItems(0, ("skip", "zeros"))
        self.mat_format.insertItems(0, ("4", "5", "7.3"))
        self.oned_as.insertItems(0, ("row", "column"))
        self.export_type.insertItems(0, ("csv", "excel", "hdf5", "mat", "parquet"))
        self.export_btn.clicked.connect(self.export)
        self.export_type.currentTextChanged.connect(self.export_changed)
        self.export_type.setCurrentIndex(-1)

        # info tab
        file_stats = os.stat(self.mdf.name)
        file_info = QtWidgets.QTreeWidgetItem()
        file_info.setText(0, 'File information')

        self.info.addTopLevelItem(file_info)

        children = []

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, 'Path')
        item.setText(1, str(self.mdf.name))
        children.append(item)

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, 'Size')
        item.setText(1, f'{file_stats.st_size / 1024 / 1024:.1f} MB')
        children.append(item)

        date_ = datetime.fromtimestamp(file_stats.st_ctime)
        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, 'Created')
        item.setText(1, date_.strftime('%d-%b-%Y %H-%M-%S'))
        children.append(item)

        date_ = datetime.fromtimestamp(file_stats.st_mtime)
        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, 'Last modified')
        item.setText(1, date_.strftime('%d-%b-%Y %H:%M:%S'))
        children.append(item)

        file_info.addChildren(children)

        mdf_info = QtWidgets.QTreeWidgetItem()
        mdf_info.setText(0, 'MDF information')

        self.info.addTopLevelItem(mdf_info)

        children = []

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, 'Version')
        item.setText(1, self.mdf.version)
        children.append(item)

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, 'Program identification')
        item.setText(1, self.mdf.identification.program_identification.decode('ascii').strip(' \r\n\t\0'))
        children.append(item)

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, 'Measurement start time')
        item.setText(1, self.mdf.header.start_time.strftime('%d-%b-%Y %H:%M:%S + %fus UTC'))
        children.append(item)

        channel_groups = QtWidgets.QTreeWidgetItem()
        channel_groups.setText(0, 'Channel groups')
        channel_groups.setText(1, str(len(self.mdf.groups)))
        children.append(channel_groups)

        channel_groups_children = []
        for i, group in enumerate(self.mdf.groups):
            channel_group = group.channel_group
            if hasattr(channel_group, 'comment'):
                comment = channel_group.comment
            else:
                comment = ''
            if comment:
                name = f'Channel group {i} ({comment})'
            else:
                name = f'Channel group {i}'

            cycles = channel_group.cycles_nr
            if self.mdf.version < '4.00':
                size = channel_group.samples_byte_nr * cycles
            else:
                if channel_group.flags & 0x1:
                    size = channel_group.samples_byte_nr + (channel_group.invalidation_bytes_nr << 32)
                else:
                    size = (channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr) * cycles

            channel_group = QtWidgets.QTreeWidgetItem()
            channel_group.setText(0, name)

            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, 'Channels')
            item.setText(1, f'{len(group.channels)}')
            channel_group.addChild(item)

            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, 'Cycles')
            item.setText(1, str(cycles))
            channel_group.addChild(item)

            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, 'Raw size')
            item.setText(1, f'{size / 1024 / 1024:.1f} MB')
            channel_group.addChild(item)

            channel_groups_children.append(channel_group)

        channel_groups.addChildren(channel_groups_children)

        channels = QtWidgets.QTreeWidgetItem()
        channels.setText(0, 'Channels')
        channels.setText(1, str(sum(len(entry) for entry in self.mdf.channels_db.values())))
        children.append(channels)

        mdf_info.addChildren(children)

        self.info.expandAll()

        # self.channels_tree.itemChanged.connect(self.select)
        self.create_window_btn.clicked.connect(self._create_window)

        self.clear_filter_btn.clicked.connect(self.clear_filter)
        self.clear_channels_btn.clicked.connect(self.clear_channels)

        self.aspects.setCurrentIndex(0)

        progress.setValue(100)

        self.load_channel_list_btn.clicked.connect(self.load_channel_list)
        self.save_channel_list_btn.clicked.connect(self.save_channel_list)
        self.load_filter_list_btn.clicked.connect(self.load_filter_list)
        self.save_filter_list_btn.clicked.connect(self.save_filter_list)

        self.scramble_btn.clicked.connect(self.scramble)
        self.setAcceptDrops(True)

        self._cursor_source = None

    def set_raster_type(self, event):
        if self.raster_type_channel.isChecked():
            self.raster_channel.setEnabled(True)
            self.raster.setEnabled(False)
            self.raster.setValue(0)
        else:
            self.raster_channel.setEnabled(False)
            self.raster_channel.setCurrentIndex(0)
            self.raster.setEnabled(True)

    def _update_channel_tree(self, index=None):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)
        signals = set()

        if self.channel_view.currentIndex() == 0:
            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    continue

                if item.checkState(0) == QtCore.Qt.Checked:
                    signals.add(item.entry)

                iterator += 1
        else:
            while iterator.value():
                item = iterator.value()

                if item.checkState(0) == QtCore.Qt.Checked:
                    signals.add(item.entry)

                iterator += 1

        self.channels_tree.clear()

        if self.channel_view.currentIndex() == 0:
            items = []
            for i, group in enumerate(self.mdf.groups):
                for j, ch in enumerate(group.channels):
                    entry = i, j

                    channel = TreeItem(entry, ch.name)
                    channel.setText(0, ch.name)
                    if entry in signals:
                        channel.setCheckState(0, QtCore.Qt.Checked)
                    else:
                        channel.setCheckState(0, QtCore.Qt.Unchecked)
                    items.append(channel)

                if self.mdf.version >= "4.00":
                    for j, ch in enumerate(group.logging_channels, 1):
                        entry = i, -j

                        channel = TreeItem(entry, ch.name)
                        channel.setText(0, ch.name)
                        if entry in signals:
                            channel.setCheckState(0, QtCore.Qt.Checked)
                        else:
                            channel.setCheckState(0, QtCore.Qt.Unchecked)
                        items.append(channel)
            if len(items) < 30000:
                items = natsorted(items, key=lambda x: x.name)
            else:
                items.sort(key=lambda x: x.name)
            self.channels_tree.addTopLevelItems(items)
        else:
            for i, group in enumerate(self.mdf.groups):
                entry = i, None
                channel_group = TreeItem(entry)
                comment = group.channel_group.comment
                comment = extract_cncomment_xml(comment)

                if comment:
                    channel_group.setText(0, f"Channel group {i} ({comment})")
                else:
                    channel_group.setText(0, f"Channel group {i}")
                channel_group.setFlags(
                    channel_group.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable
                )

                self.channels_tree.addTopLevelItem(channel_group)

                group_children = []

                for j, ch in enumerate(group.channels):
                    entry = i, j

                    channel = TreeItem(entry, ch.name)
                    channel.setText(0, ch.name)
                    if entry in signals:
                        channel.setCheckState(0, QtCore.Qt.Checked)
                    else:
                        channel.setCheckState(0, QtCore.Qt.Unchecked)
                    group_children.append(channel)

                if self.mdf.version >= "4.00":
                    for j, ch in enumerate(group.logging_channels, 1):
                        name = ch.name
                        entry = i, -j

                        channel = TreeItem(entry, name)
                        channel.setText(0, name)
                        if entry in signals:
                            channel.setCheckState(0, QtCore.Qt.Checked)
                        else:
                            channel.setCheckState(0, QtCore.Qt.Unchecked)
                        group_children.append(channel)

                channel_group.addChildren(group_children)

                del group_children

        self._settings.setValue('channels_view', self.channel_view.currentText())

    def export_changed(self, name):
        if name == 'parquet':
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(['GZIP', 'SNAPPY'])
            self.export_compression.setCurrentIndex(-1)
        elif name == 'hdf5':
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(["gzip", "lzf", "szip"])
            self.export_compression.setCurrentIndex(-1)
        elif name == 'mat':
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(["enabled", "disabled"])
            self.export_compression.setCurrentIndex(-1)
        else:
            self.export_compression.clear()
            self.export_compression.setEnabled(False)

    def set_line_style(self, with_dots=None):
        if with_dots is not None:

            self.with_dots = with_dots

            current_plot = self.get_current_plot()
            if current_plot:
                current_plot.plot.update_lines(with_dots=with_dots)

    def set_subplots_link(self, subplots_link):
        self.subplots_link = subplots_link
        viewbox = None
        if subplots_link:
            for i, mdi in enumerate(self.mdi_area.subWindowList()):
                widget = mdi.widget()
                if isinstance(widget, Plot):
                    if viewbox is None:
                        viewbox = widget.plot.viewbox
                    else:
                        widget.plot.viewbox.setXLink(viewbox)
                    widget.cursor_moved_signal.connect(self.set_cursor)
                    widget.cursor_removed_signal.connect(self.remove_cursor)
                elif isinstance(widget, Numeric):
                    widget.timestamp_changed_signal.connect(self.set_cursor)
        else:
            for mdi in self.mdi_area.subWindowList():
                widget = mdi.widget()
                if isinstance(widget, Plot):
                    widget.plot.viewbox.setXLink(None)
                    try:
                        widget.cursor_moved_signal.disconnect(self.set_cursor)
                    except:
                        pass
                    try:
                        widget.cursor_removed_signal.disconnect(self.remove_cursor)
                    except:
                        pass
                elif isinstance(widget, Numeric):
                    try:
                        widget.timestamp_changed_signal.disconnect(self.set_cursor)
                    except:
                        pass

    def set_cursor(self, widget, pos):
        if self._cursor_source is None:
            self._cursor_source = widget
            for mdi in self.mdi_area.subWindowList():
                wid = mdi.widget()
                if isinstance(wid, Plot) and wid is not widget:
                    if wid.plot.cursor1 is None:
                        event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_C, QtCore.Qt.NoModifier)
                        wid.plot.keyPressEvent(event)
                    wid.plot.cursor1.setPos(pos)
                elif isinstance(wid, Numeric) and wid is not widget:
                    wid.timestamp.setValue(pos)
            self._cursor_source = None

    def remove_cursor(self, widget):
        if self._cursor_source is None:
            self._cursor_source = widget
            for mdi in self.mdi_area.subWindowList():
                plt = mdi.widget()
                if isinstance(plt, Plot) and plt is not widget:
                    plt.cursor_removed()
            self._cursor_source = None

    def save_all_subplots(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output measurement file", "",
            "MDF version 4 files (*.mf4)",
        )

        if file_name:
            with MDF() as mdf:
                for mdi in self.mdi_area.subWindowList():
                    plt = mdi.widget()

                    mdf.append(plt.plot.signals)
                mdf.save(file_name, overwrite=True)

    def search(self):
        dlg = AdvancedSearch(
            self.mdf.channels_db,
            show_add_window=True,
            parent=self,
        )
        dlg.setModal(True)
        dlg.exec_()
        result = dlg.result
        if result:
            names = set()
            if self.channel_view.currentIndex() == 1:
                iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)

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
                        item.setCheckState(0, QtCore.Qt.Checked)
                        names.add(item.text(0))

                    iterator += 1
                    ch_cntr += 1
            else:
                iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)
                while iterator.value():
                    item = iterator.value()

                    if item.entry in result:
                        item.setCheckState(0, QtCore.Qt.Checked)
                        names.add(item.text(0))

                    iterator += 1

            if dlg.add_window_request:
                options = ["New plot window", "New numeric window", "New tabular window"] + [
                    mdi.windowTitle()
                    for mdi in self.mdi_area.subWindowList()
                ]
                ret, ok = QtWidgets.QInputDialog.getItem(
                    None,
                    "Select window type",
                    "Type:",
                    options,
                    0,
                    False,
                )
                if ok:
                    index = options.index(ret)
                    if index == 0:
                        self.add_window(['Plot', sorted(names)])
                    elif index == 1:
                        self.add_window(['Numeric', sorted(names)])
                    elif index == 2:
                        self.add_window(['Tabular', sorted(names)])
                    else:
                        widgets = [
                            mdi.widget()
                            for mdi in self.mdi_area.subWindowList()
                        ]
                        widget = widgets[index-3]
                        self.add_new_channels(names, widget)

    def save_channel_list(self, event=None, file_name=None):

        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Select output channel list file", "", "TXT files (*.txt)"
            )

        if file_name:

            config = {}
            with open(file_name, "w") as output:
                iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)

                signals = []
                if self.channel_view.currentIndex() == 1:
                    while iterator.value():
                        item = iterator.value()
                        if item.parent() is None:
                            iterator += 1
                            continue

                        if item.checkState(0) == QtCore.Qt.Checked:
                            signals.append(item.text(0))

                        iterator += 1
                else:
                    while iterator.value():
                        item = iterator.value()

                        if item.checkState(0) == QtCore.Qt.Checked:
                            signals.append(item.text(0))

                        iterator += 1

                config['selected_channels'] = signals

                windows = []
                for window in self.mdi_area.subWindowList():
                    wid = window.widget()
                    window_config = {
                        'title': window.windowTitle(),
                        'configuration': wid.to_config(),
                    }
                    if isinstance(wid, Numeric):
                        window_config['type'] = 'Numeric'
                    elif isinstance(wid, Plot):
                        window_config['type'] = 'Plot'
                    else:
                        window_config['type'] = 'Tabular'
                    windows.append(window_config)

                config['windows'] = windows

                output.write(json.dumps(config, indent=4, sort_keys=True))

    def load_channel_list(self, event=None, file_name=None):
        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select channel list file", "", "TXT files (*.txt)"
            )

        if file_name:
            with open(file_name, "r") as infile:
                info = json.load(infile)

            channels = info['selected_channels']

            iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)

            if self.channel_view.currentIndex() == 1:
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue

                    channel_name = item.text(0)
                    if channel_name in channels:
                        item.setCheckState(0, QtCore.Qt.Checked)
                        channels.pop(channels.index(channel_name))
                    else:
                        item.setCheckState(0, QtCore.Qt.Unchecked)

                    iterator += 1
            else:
                while iterator.value():
                    item = iterator.value()

                    channel_name = item.text(0)
                    if channel_name in channels:
                        item.setCheckState(0, QtCore.Qt.Checked)
                        channels.pop(channels.index(channel_name))
                    else:
                        item.setCheckState(0, QtCore.Qt.Unchecked)

                    iterator += 1

            for window in info['windows']:
                self.load_window(window)

    def save_filter_list(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output filter list file", "", "TXT files (*.txt)"
        )

        if file_name:
            with open(file_name, "w") as output:
                iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

                signals = []
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue

                    if item.checkState(0) == QtCore.Qt.Checked:
                        signals.append(item.text(0))

                    iterator += 1

                output.write("\n".join(signals))

    def load_filter_list(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select filter list file", "", "TXT files (*.txt)"
        )

        if file_name:
            with open(file_name, "r") as infile:
                channels = [line.strip() for line in infile.readlines()]
                channels = [name for name in channels if name]

            iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    continue

                channel_name = item.text(0)
                if channel_name in channels:
                    item.setCheckState(0, QtCore.Qt.Checked)
                    channels.pop(channels.index(channel_name))
                else:
                    item.setCheckState(0, QtCore.Qt.Unchecked)

                iterator += 1

    def compute_cut_hints(self):
        t_min = []
        t_max = []
        for i, group in enumerate(self.mdf.groups):
            cycles_nr = group.channel_group.cycles_nr
            if cycles_nr:
                master_min = self.mdf.get_master(
                    i,
                    record_offset=0,
                    record_count=1,
                )
                if len(master_min):
                    t_min.append(master_min[0])
                self.mdf._master_channel_cache.clear()
                master_max = self.mdf.get_master(
                    i,
                    record_offset=cycles_nr-1,
                    record_count=1,
                )
                if len(master_max):
                    t_max.append(master_max[0])
                self.mdf._master_channel_cache.clear()

        if t_min:
            time_range = t_min, t_max

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

    def show_info(self, item, column):
        group, index = item.entry
        if index is None:
            channel_group = self.mdf.groups[group].channel_group

            msg = ChannelGroupInfoDialog(channel_group, group, self)
            msg.show()
        else:

            channel = self.mdf.get_channel_metadata(group=group, index=index)

            msg = ChannelInfoDialog(channel, self)
            msg.show()

    def clear_filter(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, QtCore.Qt.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def clear_channels(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)

        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, QtCore.Qt.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def get_current_plot(self):
        mdi = self.mdi_area.activeSubWindow()
        if mdi is not None:
            widget = mdi.widget()
            if isinstance(widget, Plot):
                return widget
            else:
                return None
        else:
            return None

    def new_search_result(self, tree, search):
        group_index, channel_index = search.entries[search.current_index]

        iterator = QtWidgets.QTreeWidgetItemIterator(tree)

        if self.channel_view.currentIndex() == 1:

            while iterator.value():
                item = iterator.value()
                if item.parent() is not None:
                    if item.entry == (group_index, channel_index):

                        tree.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtTop)
                        item.setSelected(True)
                        tree.setFocus()
                iterator += 1
        else:
            while iterator.value():
                item = iterator.value()
                if item.entry == (group_index, channel_index):

                    tree.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtTop)
                    item.setSelected(True)
                    tree.setFocus()
                iterator += 1

    def close(self):
        mdf_name = self.mdf.name
        self.mdf.close()
        if self.file_name.suffix.lower() == ".dl3":
            mdf_name.unlink()

    def convert(self, event):
        version = self.convert_format.currentText()

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = '.mdf'
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = '.mf4'

        split = self.convert_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.convert_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.convert_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", f'{filter};;All files (*.*)', filter
        )

        if file_name:
            file_name = Path(file_name).with_suffix(suffix)

            progress = setup_progress(
                parent=self,
                title="Converting measurement",
                message=f'Converting "{self.file_name}" from {self.mdf.version} to {version}',
                icon_name="convert",
            )

            # convert self.mdf
            target = self.mdf.convert
            kwargs = {"version": version}

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
            progress.setLabelText(f'Saving converted file "{file_name}"')

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

        self.progress = None
        progress.cancel()

    def resample(self, event):
        version = self.resample_format.currentText()

        if self.raster_type_channel.isChecked():
            raster = self.raster_channel.currentText()
        else:
            raster = self.raster.value()

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = '.mdf'
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = '.mf4'

        split = self.resample_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.resample_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.resample_compression.currentIndex()
        time_from_zero = self.resample_time_from_zero.checkState() == QtCore.Qt.Checked

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", f'{filter};;All files (*.*)', filter
        )

        if file_name:
            file_name = Path(file_name).with_suffix(suffix)

            progress = setup_progress(
                parent=self,
                title="Resampling measurement",
                message=f'Resampling "{self.file_name}" to {raster}s raster ',
                icon_name="resample",
            )

            # resample self.mdf
            target = self.mdf.resample
            kwargs = {
                "raster": raster,
                "version": version,
                "time_from_zero": time_from_zero,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=66,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText(f'Saving resampled file "{file_name}"')

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

        self.progress = None
        progress.cancel()

    def cut(self, event):
        version = self.cut_format.currentText()
        start = self.cut_start.value()
        stop = self.cut_stop.value()
        time_from_zero = self.cut_time_from_zero.checkState() == QtCore.Qt.Checked

        if self.whence.checkState() == QtCore.Qt.Checked:
            whence = 1
        else:
            whence = 0

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = '.mdf'
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = '.mf4'

        split = self.cut_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.cut_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.cut_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", f'{filter};;All files (*.*)', filter
        )

        if file_name:
            file_name = Path(file_name).with_suffix(suffix)
            progress = setup_progress(
                parent=self,
                title="Cutting measurement",
                message=f'Cutting "{self.file_name}" from {start}s to {stop}s',
                icon_name="cut",
            )

            # cut self.mdf
            target = self.mdf.cut
            kwargs = {
                "start": start,
                "stop": stop,
                "whence": whence,
                "version": version,
                "time_from_zero": time_from_zero,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=66,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText(f'Saving cut file "{file_name}"')

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

        self.progress = None
        progress.cancel()

    def export(self, event):
        export_type = self.export_type.currentText()

        single_time_base = self.single_time_base.checkState() == QtCore.Qt.Checked
        time_from_zero = self.time_from_zero.checkState() == QtCore.Qt.Checked
        use_display_names = self.use_display_names.checkState() == QtCore.Qt.Checked
        empty_channels = self.empty_channels.currentText()
        mat_format = self.mat_format.currentText()
        raster = self.export_raster.value()
        oned_as = self.oned_as.currentText()
        reduce_memory_usage = self.reduce_memory_usage.checkState() == QtCore.Qt.Checked
        compression = self.export_compression.currentText()
        time_as_date = self.time_as_date.checkState() == QtCore.Qt.Checked

        filters = {
            "csv": "CSV files (*.csv)",
            "excel": "Excel files (*.xlsx)",
            "hdf5": "HDF5 files (*.hdf)",
            "mat": "Matlab MAT files (*.mat)",
            "parquet": "Apache Parquet files (*.parquet)",
        }

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select export file", "", f"{filters[export_type]};;All files (*.*)", filters[export_type]
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title="Export measurement",
                message=f'Exporting "{self.file_name}" to {export_type}',
                icon_name="export",
            )

            # cut self.mdf
            target = self.mdf.export
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
                "reduce_memory_usage": reduce_memory_usage,
                "compression": compression,
                "time_as_date": time_as_date,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=100,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            self.progress = None
            progress.cancel()

    def add_window(self, args):
        window_type, names = args
        if names and isinstance(names[0], str):
            signals_ = [
                (None, *self.mdf.whereis(name)[0])
                for name in names
                if name in self.mdf
            ]
        else:
            signals_ = names

        if window_type == 'Tabular':
            signals = self.mdf.to_dataframe(
                channels=signals_,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
            )
        else:

            signals = self.mdf.select(
                signals_,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
            )

            for sig, sig_ in zip(signals, signals_):
                sig.group_index = sig_[1]

            signals = [
                sig
                for sig in signals
                if not sig.samples.dtype.names
            ]

            for signal in signals:
                if len(signal.samples.shape) > 1:
                    signal.samples = csv_bytearray2hex(pd.Series(list(signal.samples))).astype(bytes)

            signals = natsorted(signals, key=lambda x: x.name)

        if window_type == 'Numeric':
            numeric = Numeric(signals)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(numeric)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(numeric)

                if len(self.mdi_area.subWindowList()) == 1:
                    w.showMaximized()
                else:
                    w.show()
                    self.mdi_area.tileSubWindows()

            menu = w.systemMenu()

            def set_title(mdi):
                name, ok = QtWidgets.QInputDialog.getText(
                    None,
                    'Set sub-plot title',
                    'Title:',
                )
                if ok and name:
                    mdi.setWindowTitle(name)

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)
            w.setSystemMenu(menu)

            w.setWindowTitle(f'Numeric {self._window_counter}')
            self._window_counter += 1

            numeric.add_channels_request.connect(partial(self.add_new_channels, widget=numeric))
            if self.subplots_link:
                numeric.timestamp_changed_signal.connect(self.set_cursor)

        elif window_type == 'Plot':
            plot = Plot([], False)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(plot)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(plot)

                if len(self.mdi_area.subWindowList()) == 1:
                    w.showMaximized()
                else:
                    w.show()
                    self.mdi_area.tileSubWindows()

            plot.hide()

            plot.add_new_channels(signals)
            plot.plot.update_lines(with_dots=self.with_dots)

            plot.show()

            menu = w.systemMenu()

            def set_title(mdi):
                name, ok = QtWidgets.QInputDialog.getText(
                    None,
                    'Set sub-plot title',
                    'Title:',
                )
                if ok and name:
                    mdi.setWindowTitle(name)

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)
            w.setSystemMenu(menu)

            w.setWindowTitle(f'Plot {self._window_counter}')
            self._window_counter += 1

            if self.subplots_link:

                for i, mdi in enumerate(self.mdi_area.subWindowList()):
                    viewbox = mdi.widget().plot.viewbox
                    plot.plot.viewbox.setXLink(viewbox)
                    break

            plot.add_channels_request.connect(partial(self.add_new_channels, widget=plot))

            self.set_subplots_link(self.subplots_link)

        elif window_type == 'Tabular':
            numeric = Tabular(signals, start=self.mdf.header.start_time.timestamp())

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(numeric)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(numeric)

                if len(self.mdi_area.subWindowList()) == 1:
                    w.showMaximized()
                else:
                    w.show()
                    self.mdi_area.tileSubWindows()

            menu = w.systemMenu()

            def set_title(mdi):
                name, ok = QtWidgets.QInputDialog.getText(
                    None,
                    'Set sub-plot title',
                    'Title:',
                )
                if ok and name:
                    mdi.setWindowTitle(name)

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)
            w.setSystemMenu(menu)

            w.setWindowTitle(f'Tabular {self._window_counter}')
            self._window_counter += 1

    def load_window(self, window_info):

        if window_info['type'] == 'Numeric':
            fmt = window_info['configuration']['format']

            signals_ = [
                (None, *self.mdf.whereis(name)[0])
                for name in window_info['configuration']['channels']
                if name in self.mdf
            ]
            signals = self.mdf.select(
                signals_,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
            )

            for sig, sig_ in zip(signals, signals_):
                sig.group_index = sig_[1]

            signals = [
                sig
                for sig in signals
                if not sig.samples.dtype.names and len(sig.samples.shape) <= 1
            ]

            signals = natsorted(signals, key=lambda x: x.name)

            numeric = Numeric(signals)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(numeric)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(numeric)

                if len(self.mdi_area.subWindowList()) == 1:
                    w.showMaximized()
                else:
                    w.show()
                    self.mdi_area.tileSubWindows()

            if window_info['title']:
                w.setWindowTitle(window_info['title'])
            else:
                w.setWindowTitle(f'Numeric {self._window_counter}')
                self._window_counter += 1

            numeric.format = fmt
            numeric._update_values()

            menu = w.systemMenu()

            def set_title(mdi):
                name, ok = QtWidgets.QInputDialog.getText(
                    None,
                    'Set sub-plot title',
                    'Title:',
                )
                if ok and name:
                    mdi.setWindowTitle(name)

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)
            w.setSystemMenu(menu)

            numeric.add_channels_request.connect(partial(self.add_new_channels, widget=numeric))

        elif window_info['type'] == 'Plot':
            measured_signals = [
                (None, *self.mdf.whereis(channel['name'])[0])
                for channel in window_info['configuration']['channels']
                if not channel['computed'] and channel['name']  in self.mdf
            ]
            measured_signals = {
                sig.name: sig
                for sig in self.mdf.select(
                    measured_signals,
                    ignore_value2text_conversions=self.ignore_value2text_conversions,
                )
            }

            for signal in measured_signals.values():
                signal.computed = False
                signal.computation = {}

            if measured_signals:
                all_timebase = reduce(
                    np.union1d, (sig.timestamps for sig in measured_signals.values())
                )
            else:
                all_timebase = []

            computed_signals_descriptions = [
                channel
                for channel in window_info['configuration']['channels']
                if channel['computed']
            ]

            computed_signals = {}

            def compute(description, measured_signals, all_timebase):
                type_ = description['type']

                if type_ == 'arithmetic':
                    op = description['op']

                    operand1 = description['operand1']
                    if isinstance(operand1, dict):
                        operand1 = compute(operand1, measured_signals, all_timebase)
                    elif isinstance(operand1, str):
                        operand1 = measured_signals[operand1]

                    operand2 = description['operand2']
                    if isinstance(operand2, dict):
                        operand2 = compute(operand2, measured_signals, all_timebase)
                    elif isinstance(operand2, str):
                        operand2 = measured_signals[operand2]

                    result = eval(f'operand1 {op} operand2')
                    if not hasattr(result, 'name'):
                        result = Signal(
                            name='_',
                            samples=np.ones(len(all_timebase))*result,
                            timestamps=all_timebase,
                        )

                else:
                    function = description['name']
                    args = description['args']

                    channel = description['channel']

                    if isinstance(channel, dict):
                        channel = compute(channel, measured_signals, all_timebase)
                    else:
                        channel = measured_signals[channel]

                    func = getattr(np, function)

                    if function in [
                        'arccos',
                        'arcsin',
                        'arctan',
                        'cos',
                        'deg2rad',
                        'degrees',
                        'rad2deg',
                        'radians',
                        'sin',
                        'tan',
                        'floor',
                        'rint',
                        'fix',
                        'trunc',
                        'cumprod',
                        'cumsum',
                        'diff',
                        'exp',
                        'log10',
                        'log',
                        'log2',
                        'absolute',
                        'cbrt',
                        'sqrt',
                        'square',
                        'gradient',
                    ]:

                        samples = func(channel.samples)
                        if function == 'diff':
                            timestamps = channel.timestamps[1:]
                        else:
                            timestamps = channel.timestamps

                    elif function == 'around':
                        samples = func(channel.samples, *args)
                        timestamps = channel.timestamps
                    elif function == 'clip':
                        samples = func(channel.samples, *args)
                        timestamps = channel.timestamps

                    result = Signal(
                        samples=samples,
                        timestamps=timestamps,
                        name='_',
                    )

                return result

            for channel in computed_signals_descriptions:
                computation = channel['computation']

                try:

                    signal = compute(computation, measured_signals, all_timebase)
                    signal.color = channel['color']
                    signal.computed = True
                    signal.computation = channel['computation']
                    signal.name = channel['name']
                    signal.unit = channel['unit']

                    computed_signals[signal.name] = signal
                except:
                    raise
                    pass

            signals = list(measured_signals.values()) + list(computed_signals.values())

            plot = Plot([], self.with_dots)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(plot)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(plot)

                if len(self.mdi_area.subWindowList()) == 1:
                    w.showMaximized()
                else:
                    w.show()
                    self.mdi_area.tileSubWindows()

            plot.hide()

            plot.add_new_channels(signals)

            plot.show()

            menu = w.systemMenu()

            def set_title(mdi):
                name, ok = QtWidgets.QInputDialog.getText(
                    None,
                    'Set sub-plot title',
                    'Title:',
                )
                if ok and name:
                    mdi.setWindowTitle(name)

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)
            w.setSystemMenu(menu)

            if window_info['title']:
                w.setWindowTitle(window_info['title'])
            else:
                w.setWindowTitle(f'Plot {self._window_counter}')
                self._window_counter += 1

            plot.add_channels_request.connect(partial(self.add_new_channels, widget=plot))

            descriptions = {
                channel['name']: channel
                for channel in window_info['configuration']['channels']
            }

            count = plot.channel_selection.count()

            for i in range(count):
                wid = plot.channel_selection.itemWidget(plot.channel_selection.item(i))
                name = wid._name

                description = descriptions[name]

                wid.setFmt(description['fmt'])
                wid.set_precision(description['precision'])
                wid.setColor(description['color'])
                wid.color_changed.emit(wid.index, description['color'])
                wid.ranges = {
                    (range['start'], range['stop']): range['color']
                    for range in description['ranges']
                }
                wid.ylink.setCheckState(
                    QtCore.Qt.Checked
                    if description['common_axis']
                    else QtCore.Qt.Unchecked
                )
                wid.display.setCheckState(
                    QtCore.Qt.Checked
                    if description['enabled']
                    else QtCore.Qt.Unchecked
                )

            self.set_subplots_link(self.subplots_link)

        elif window_info['type'] == 'Tabular':

            signals_ = [
                (None, *self.mdf.whereis(name)[0])
                for name in window_info['configuration']['channels']
                if name in self.mdf
            ]
            signals = self.mdf.to_dataframe(
                channels=signals_,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
            )

            tabular = Tabular(signals, start=self.mdf.header.start_time.timestamp())

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(tabular)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(tabular)

                if len(self.mdi_area.subWindowList()) == 1:
                    w.showMaximized()
                else:
                    w.show()
                    self.mdi_area.tileSubWindows()

            if window_info['title']:
                w.setWindowTitle(window_info['title'])
            else:
                w.setWindowTitle(f'Tabular {self._window_counter}')
                self._window_counter += 1

            filter_count = 0
            available_columns = [signals.index.name,] + list(signals.columns)
            for filter_info in window_info['configuration']['filters']:
                if filter_info['column'] in available_columns:
                    tabular.add_filter()
                    filter = tabular.filters.itemWidget(tabular.filters.item(filter_count))
                    filter.enabled.setCheckState(
                        QtCore.Qt.Checked
                        if filter_info['enabled']
                        else QtCore.Qt.Unchecked
                    )
                    filter.relation.setCurrentText(filter_info['relation'])
                    filter.column.setCurrentText(filter_info['column'])
                    filter.op.setCurrentText(filter_info['op'])
                    filter.target.setText(str(filter_info['target']))
                    filter.validate_target()

                    filter_count += 1

            if filter_count and window_info['configuration']['filtered']:
                tabular.apply_filters()

            tabular.time_as_date.setCheckState(
                QtCore.Qt.Checked
                if window_info['configuration']['time_as_date']
                else QtCore.Qt.Unchecked
            )

            tabular.sort.setCheckState(
                QtCore.Qt.Checked
                if window_info['configuration']['sorted']
                else QtCore.Qt.Unchecked
            )

            menu = w.systemMenu()

            def set_title(mdi):
                name, ok = QtWidgets.QInputDialog.getText(
                    None,
                    'Set sub-plot title',
                    'Title:',
                )
                if ok and name:
                    mdi.setWindowTitle(name)

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)
            w.setSystemMenu(menu)

    def _create_window(self, event):

        ret, ok = QtWidgets.QInputDialog.getItem(
            None,
            "Select window type",
            "Type:",
            ["Plot", "Numeric", "Tabular"],
            0,
            False,
        )
        if ok:

            try:
                iter(event)
                signals = event
            except:

                iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)

                signals = []

                if self.channel_view.currentIndex() == 1:
                    while iterator.value():
                        item = iterator.value()
                        if item.parent() is None:
                            iterator += 1
                            continue

                        if item.checkState(0) == QtCore.Qt.Checked:
                            group, index = item.entry
                            ch = self.mdf.groups[group].channels[index]
                            if not ch.component_addr:
                                signals.append((None, group, index))

                        iterator += 1
                else:
                    while iterator.value():
                        item = iterator.value()

                        if item.checkState(0) == QtCore.Qt.Checked:
                            group, index = item.entry
                            ch = self.mdf.groups[group].channels[index]
                            if not ch.component_addr:
                                signals.append((None, group, index))

                        iterator += 1

            self.add_window((ret, signals))

    def filter(self, event):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

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

            if item.checkState(0) == QtCore.Qt.Checked:
                channels.append((None, group, index))

            index += 1
            iterator += 1

        version = self.filter_format.itemText(self.filter_format.currentIndex())

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = '.mdf'
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = '.mf4'

        split = self.filter_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.filter_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.filter_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", f'{filter};;All files (*.*)', filter
        )

        if file_name:
            file_name = Path(file_name).with_suffix(suffix)
            progress = setup_progress(
                parent=self,
                title="Filtering measurement",
                message=f'Filtering selected channels from "{self.file_name}"',
                icon_name="filter",
            )

            # filtering self.mdf
            target = self.mdf.filter
            kwargs = {"channels": channels, "version": version}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=66,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText(f'Saving filtered file "{file_name}"')

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

        self.progress = None
        progress.cancel()

    def scramble(self, event):

        progress = setup_progress(
            parent=self,
            title="Scrambling measurement",
            message=f'Scrambling "{self.file_name}"',
            icon_name="scramble",
        )

        # scrambling self.mdf
        target = MDF.scramble
        kwargs = {"name": self.file_name, "callback": self.update_progress}

        mdf = run_thread_with_progress(
            self,
            target=target,
            kwargs=kwargs,
            factor=100,
            offset=0,
            progress=progress,
        )

        if mdf is TERMINATED:
            progress.cancel()
            return

        self.progress = None
        progress.cancel()

        path = Path(self.file_name)

        self.open_new_file.emit(str(path.with_suffix(f".scrambled{path.suffix}")))

    def add_new_channels(self, names, widget):
        try:
            sigs = self.mdf.select(names)

            for sig in sigs:
                group, index = self.mdf.whereis(sig.name)[0]
                sig.group_index = group

            sigs = [
                sig
                for sig in sigs
                if not sig.samples.dtype.names and len(sig.samples.shape) <= 1
            ]
            widget.add_new_channels(sigs)
        except MdfException:
            pass

    def extract_can_logging(self, event):
        version = self.extract_can_format.currentText()
        count = self.can_database_list.count()

        dbc_files = []
        for i in range(count):
            item = self.can_database_list.item(i)
            dbc_files.append(item.text())

        compression = self.extract_can_compression.currentIndex()
        ignore_invalid_signals = self.ignore_invalid_signals_mdf.checkState() == QtCore.Qt.Checked

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = '.mdf'
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = '.mf4'


        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", f'{filter};;All files (*.*)', filter
        )

        if file_name:

            file_name = Path(file_name).with_suffix(suffix)

            progress = setup_progress(
                parent=self,
                title="Extract CAN logging",
                message=f'Extracting CAN signals from "{self.file_name}"',
                icon_name="down",
            )

            # convert self.mdf
            target = self.mdf.extract_can_logging
            kwargs = {
                "dbc_files": dbc_files,
                "version": version,
                "ignore_invalid_signals": ignore_invalid_signals,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=70,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            # then save it
            progress.setLabelText(f'Saving file to "{file_name}"')

            target = mdf.save
            kwargs = {
                "dst": file_name,
                "compression": compression,
                "overwrite": True,
            }

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=30,
                offset=70,
                progress=progress,
            )

            self.progress = None
            progress.cancel()

            self.open_new_file.emit(str(file_name))

    def extract_can_csv_logging(self, event):
        version = self.extract_can_format.currentText()
        count = self.can_database_list.count()

        dbc_files = []
        for i in range(count):
            item = self.can_database_list.item(i)
            dbc_files.append(item.text())

        ignore_invalid_signals = self.ignore_invalid_signals_csv.checkState() == QtCore.Qt.Checked
        single_time_base = self.single_time_base_can.checkState() == QtCore.Qt.Checked
        time_from_zero = self.time_from_zero_can.checkState() == QtCore.Qt.Checked
        empty_channels = self.empty_channels_can.currentText()
        raster = self.export_raster.value()
        time_as_date = self.can_time_as_date.checkState() == QtCore.Qt.Checked

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output CSV file", "", "CSV (*.csv);;All files (*.*)", "CSV (*.csv)"
        )

        if file_name:

            progress = setup_progress(
                parent=self,
                title="Extract CAN logging to CSV",
                message=f'Extracting CAN signals from "{self.file_name}"',
                icon_name="csv",
            )

            # convert self.mdf
            target = self.mdf.extract_can_logging
            kwargs = {
                "dbc_files": dbc_files,
                "version": version,
                "ignore_invalid_signals": ignore_invalid_signals,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=70,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            # then save it
            progress.setLabelText(f'Saving file to "{file_name}"')

            target = mdf.export
            kwargs = {
                "fmt": "csv",
                "filename": file_name,
                "single_time_base": single_time_base,
                "time_from_zero": time_from_zero,
                "empty_channels": empty_channels,
                "raster": raster,
                "time_as_date": time_as_date,
                "ignore_value2text_conversions": self.ignore_value2text_conversions,
            }

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=30,
                offset=70,
                progress=progress,
            )

            self.progress = None
            progress.cancel()

    def load_can_database(self, event):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select CAN database file",
            "",
            "ARXML or DBC (*.dbc *.axml)",
            "ARXML or DBC (*.dbc *.axml)",
        )

        for file_name in file_names:
            self.can_database_list.addItems(file_names)

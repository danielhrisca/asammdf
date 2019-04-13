# -*- coding: utf-8 -*-
from functools import partial
from threading import Thread
from time import sleep
from pathlib import Path

import psutil
from natsort import natsorted
import numpy as np

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import uic

from ..ui import resource_qt5 as resource_rc

from ...mdf import MDF, SUPPORTED_VERSIONS
from ...blocks.utils import MdfException
from ..utils import TERMINATED, run_thread_with_progress, setup_progress
from .plot import Plot
from .numeric import Numeric
from .search import SearchWidget
from .tree import TreeWidget
from .tree_item import TreeItem
from .mdi_area import MdiAreaWidget

from ..dialogs.advanced_search import AdvancedSearch
from ..dialogs.channel_info import ChannelInfoDialog

HERE = Path(__file__).resolve().parent


class FileWidget(QtWidgets.QWidget):

    file_scrambled = QtCore.pyqtSignal(str)

    def __init__(self, file_name, with_dots, subplots=False, subplots_link=False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        uic.loadUi(HERE.joinpath("..", "ui", "file_widget.ui"), self)

        file_name = Path(file_name)
        self.subplots = subplots
        self.subplots_link = subplots_link

        self.file_name = file_name
        self.progress = None
        self.mdf = None
        self.info = None
        self.info_index = None
        self.with_dots = with_dots

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

        progress.setLabelText("Loading graphical elements")

        progress.setValue(35)

        self.filter_field = SearchWidget(self.mdf.channels_db, self)

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
            self.mdf.channels_db, channel_and_search
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
        self.plot_btn = QtWidgets.QPushButton("", channel_and_search)
        self.plot_btn.setToolTip("Plot selected channels")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/graph.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.plot_btn.setIcon(icon3)
        self.plot_btn.setObjectName("plot_btn")
        hbox.addWidget(self.plot_btn)

        hbox.addWidget(line)
        self.numeric_btn = QtWidgets.QPushButton("", channel_and_search)
        self.numeric_btn.setToolTip("Numeric display selected channels")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/numeric.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.numeric_btn.setIcon(icon3)
        self.numeric_btn.setObjectName("numeric_btn")
        hbox.addWidget(self.numeric_btn)

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

        flags = None

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

                name = self.mdf.get_channel_name(i, j)

                channel = TreeItem(entry)
                if flags is None:
                    flags = channel.flags() | QtCore.Qt.ItemIsUserCheckable
                channel.setFlags(flags)
                channel.setText(0, name)
                channel.setCheckState(0, QtCore.Qt.Unchecked)
                filter_children.append(channel)

            if self.mdf.version >= "4.00":
                for j, ch in enumerate(group.logging_channels, 1):
                    name = ch.name
                    entry = i, -j

                    channel = TreeItem(entry)
                    channel.setFlags(flags)
                    channel.setText(0, name)
                    channel.setCheckState(0, QtCore.Qt.Unchecked)
                    filter_children.append(channel)

            filter_channel_group.addChildren(filter_children)

            del filter_children

            progress.setValue(37 + int(53 * (i + 1) / groups_nr / 2))

        self._update_channel_tree()

        self.raster_channel.addItems(
            natsorted(self.mdf.channels_db)
        )

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

        progress.setValue(99)

        self.empty_channels.insertItems(0, ("zeros", "skip"))
        self.mat_format.insertItems(0, ("4", "5", "7.3"))
        self.oned_as.insertItems(0, ("row", "column"))
        self.export_type.insertItems(0, ("csv", "excel", "hdf5", "mat", "parquet"))
        self.export_btn.clicked.connect(self.export)
        self.export_type.currentTextChanged.connect(self.export_changed)
        self.export_type.setCurrentIndex(-1)

        # self.channels_tree.itemChanged.connect(self.select)
        self.plot_btn.clicked.connect(self.plot_pyqtgraph)
        self.numeric_btn.clicked.connect(self.numeric_pyqtgraph)

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

        flags = None
        if self.channel_view.currentIndex() == 0:
            items = []
            for i, group in enumerate(self.mdf.groups):
                for j, ch in enumerate(group.channels):
                    entry = i, j

                    channel = TreeItem(entry, ch.name)
                    if flags is None:
                        flags = channel.flags() | QtCore.Qt.ItemIsUserCheckable
                    channel.setFlags(flags)
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
                        channel.setFlags(flags)
                        channel.setText(0, ch.name)
                        if entry in signals:
                            channel.setCheckState(0, QtCore.Qt.Checked)
                        else:
                            channel.setCheckState(0, QtCore.Qt.Unchecked)
                        items.append(channel)
            items = natsorted(items, key=lambda x: x.name)
            self.channels_tree.addTopLevelItems(items)
        else:
            for i, group in enumerate(self.mdf.groups):
                channel_group = QtWidgets.QTreeWidgetItem()
                channel_group.setText(0, f"Channel group {i}")
                channel_group.setFlags(
                    channel_group.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable
                )

                self.channels_tree.addTopLevelItem(channel_group)

                group_children = []

                for j, ch in enumerate(group.channels):
                    entry = i, j

                    name = self.mdf.get_channel_name(i, j)
                    channel = TreeItem(entry)
                    if flags is None:
                        flags = channel.flags() | QtCore.Qt.ItemIsUserCheckable
                    channel.setFlags(flags)
                    channel.setText(0, name)
                    if entry in signals:
                        channel.setCheckState(0, QtCore.Qt.Checked)
                    else:
                        channel.setCheckState(0, QtCore.Qt.Unchecked)
                    group_children.append(channel)

                if self.mdf.version >= "4.00":
                    for j, ch in enumerate(group.logging_channels, 1):
                        name = ch.name
                        entry = i, -j

                        channel = TreeItem(entry)
                        channel.setFlags(flags)
                        channel.setText(0, name)
                        if entry in signals:
                            channel.setCheckState(0, QtCore.Qt.Checked)
                        else:
                            channel.setCheckState(0, QtCore.Qt.Unchecked)
                        group_children.append(channel)

                channel_group.addChildren(group_children)

                del group_children

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
        if subplots_link:
            viewbox = None
            for mdi in self.mdi_area.subWindowList():
                plt = mdi.widget()
                if viewbox is None:
                    viewbox = plt.plot.viewbox
                else:
                    plt.plot.viewbox.setXLink(viewbox)
        else:
            for mdi in self.mdi_area.subWindowList():
                plt = mdi.widget()
                plt.plot.viewbox.setXLink(None)

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
        dlg = AdvancedSearch(self.mdf.channels_db, parent=self)
        dlg.setModal(True)
        dlg.exec_()
        result = dlg.result
        if result:
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

                    iterator += 1
                    ch_cntr += 1
            else:
                iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)
                while iterator.value():
                    item = iterator.value()

                    if item.entry in result:
                        item.setCheckState(0, QtCore.Qt.Checked)

                    iterator += 1

    def save_channel_list(self, event=None, file_name=None):

        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Select output channel list file", "", "TXT files (*.txt)"
            )

        if file_name:
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

                output.write("\n".join(signals))

    def load_channel_list(self, event=None, file_name=None):
        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select channel list file", "", "TXT files (*.txt)"
            )

        if file_name:
            with open(file_name, "r") as infile:
                channels = [line.strip() for line in infile.readlines()]
                channels = [name for name in channels if name]

            print('load', channels)

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
        # TODO : use master channel physical min and max values
        times = []
        groups_nr = len(self.mdf.groups)
        for i in range(groups_nr):
            master = self.mdf.get_master(i)
            if len(master):
                times.append(master[0])
                times.append(master[-1])
            QtWidgets.QApplication.processEvents()

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
        if self.channel_view.currentIndex() == 1:
            if item and item.parent():
                group, index = item.entry

                channel = self.mdf.get_channel_metadata(group=group, index=index)

                msg = ChannelInfoDialog(channel, self)
                msg.show()
        else:
            if item:
                group, index = item.entry

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
            return mdi.widget()
        else:
            return None

    def new_search_result(self, tree, search):
        group_index, channel_index = search.entries[search.current_index]

        grp = self.mdf.groups[group_index]
        channel_count = len(grp.channels)

        iterator = QtWidgets.QTreeWidgetItemIterator(tree)

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
                    tree.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtTop)
                    item.setSelected(True)

            index += 1
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
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.convert_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.convert_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.convert_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:

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

    def resample(self, event):
        version = self.resample_format.currentText()

        if self.raster_type_channel.isChecked():
            raster = self.raster_channel.currentText()
        else:
            raster = self.raster.value()

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.resample_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.resample_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.resample_compression.currentIndex()
        time_from_zero = self.resample_time_from_zero.checkState() == QtCore.Qt.Checked

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:
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
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.cut_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.cut_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.cut_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title="Cutting measurement",
                message='Cutting "{self.file_name}" from {start}s to {stop}s',
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

        filters = {
            "csv": "CSV files (*.csv)",
            "excel": "Excel files (*.xlsx)",
            "hdf5": "HDF5 files (*.hdf)",
            "mat": "Matlab MAT files (*.mat)",
            "parquet": "Apache Parquet files (*.parquet)",
        }

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select export file", "", filters[export_type]
        )

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
                    "reduce_memory_usage": reduce_memory_usage,
                    "compression": compression,
                },
            )

            progress = QtWidgets.QProgressDialog(
                f"Exporting to {export_type} ...", "Abort export", 0, 100
            )
            progress.setWindowModality(QtCore.Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.setAutoClose(True)
            progress.setWindowTitle("Running export")
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/export.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            progress.setWindowIcon(icon)

            thr.start()

            cntr = 0

            while thr.is_alive():
                cntr += 1
                progress.setValue(cntr % 98)
                sleep(0.1)

            progress.cancel()

    def add_window(self, args):
        window_type, names = args
        signals_ = [
            (None, *self.mdf.whereis(name)[0])
            for name in names
            if name in self.mdf
        ]
        signals = self.mdf.select(signals_)

        for sig, sig_ in zip(signals, signals_):
            sig.group_index = sig_[1]

        signals = [
            sig
            for sig in signals
            if not sig.samples.dtype.names and len(sig.samples.shape) <= 1
        ]

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

            def set_tile(mdi):
                name, ok = QtWidgets.QInputDialog.getText(
                    None,
                    'Set sub-plot title',
                    'Title:',
                )
                if ok and name:
                    mdi.setWindowTitle(name)

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_tile, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)
            w.setSystemMenu(menu)

            numeric.add_channels_request.connect(partial(self.add_new_channels, widget=numeric))
        else:
            plot = Plot(signals, self.with_dots)
            plot.plot.update_lines(force=True)

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

            menu = w.systemMenu()

            def set_tile(mdi):
                name, ok = QtWidgets.QInputDialog.getText(
                    None,
                    'Set sub-plot title',
                    'Title:',
                )
                if ok and name:
                    mdi.setWindowTitle(name)

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_tile, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)
            w.setSystemMenu(menu)

            plot.add_channels_request.connect(partial(self.add_new_channels, widget=plot))

    def plot_pyqtgraph(self, event):
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

            signals = self.mdf.select(signals)

            signals = [
                sig
                for sig in signals
                if not sig.samples.dtype.names and len(sig.samples.shape) <= 1
            ]

            signals = natsorted(signals, key=lambda x: x.name)

        plot = Plot(signals, self.with_dots)
        plot.plot.update_lines(force=True)

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

        menu = w.systemMenu()

        def set_tile(mdi):
            name, ok = QtWidgets.QInputDialog.getText(
                None,
                'Set sub-plot title',
                'Title:',
            )
            if ok and name:
                mdi.setWindowTitle(name)

        action = QtWidgets.QAction("Set title", menu)
        action.triggered.connect(partial(set_tile, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)
        w.setSystemMenu(menu)

        plot.add_channels_request.connect(partial(self.add_new_channels, widget=plot))

    def numeric_pyqtgraph(self, event):
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

            signals_ = self.mdf.select(signals)

            for sig, s_ in zip(signals_, signals):
                sig.group_index = s_[1]

            signals = signals_

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

        menu = w.systemMenu()

        def set_tile(mdi):
            name, ok = QtWidgets.QInputDialog.getText(
                None,
                'Set sub-plot title',
                'Title:',
            )
            if ok and name:
                mdi.setWindowTitle(name)

        action = QtWidgets.QAction("Set title", menu)
        action.triggered.connect(partial(set_tile, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)
        w.setSystemMenu(menu)

        numeric.add_channels_request.connect(partial(self.add_new_channels, widget=numeric))

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
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.filter_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.filter_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.filter_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:
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

        self.file_scrambled.emit(str(Path(self.file_name).with_suffix(".scrambled.mf4")))

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

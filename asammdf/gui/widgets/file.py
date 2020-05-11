# -*- coding: utf-8 -*-
from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
from tempfile import gettempdir
from time import perf_counter
from traceback import format_exc
from uuid import uuid4

from natsort import natsorted
import numpy as np
import pandas as pd
import psutil
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ...blocks.utils import csv_bytearray2hex, extract_cncomment_xml, MdfException
from ...blocks.v4_constants import FLAG_CG_BUS_EVENT
from ...mdf import MDF, SUPPORTED_VERSIONS
from ..dialogs.advanced_search import AdvancedSearch
from ..dialogs.channel_group_info import ChannelGroupInfoDialog
from ..dialogs.channel_info import ChannelInfoDialog
from ..ui import resource_rc as resource_rc
from ..ui.file_widget import Ui_file_widget
from ..utils import (
    add_children,
    compute_signal,
    get_required_signals,
    HelperChannel,
    load_dsp,
    run_thread_with_progress,
    setup_progress,
    TERMINATED,
)
from .mdi_area import MdiAreaWidget, WithMDIArea
from .numeric import Numeric
from .plot import Plot
from .tree import TreeWidget
from .tree_item import TreeItem


class FileWidget(WithMDIArea, Ui_file_widget, QtWidgets.QWidget):

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

        super(Ui_file_widget, self).__init__(*args, **kwargs)
        WithMDIArea.__init__(self)
        self.setupUi(self)
        self._settings = QtCore.QSettings()
        self.uuid = uuid4()

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

        self._show_filter_tree = False

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

        if file_name.suffix.lower() in (".erg", ".bsig"):

            extension = file_name.suffix.lower().strip(".")
            progress.setLabelText(f"Converting from {extension} to mdf")

            from mfile import ERG, BSIG

            if file_name.suffix.lower() == ".erg":
                cls = ERG
            else:
                cls = BSIG

            out_file = Path(gettempdir()) / file_name.name

            mdf_path = (
                cls(file_name).export_mdf().save(out_file.with_suffix(".tmp.mf4"))
            )
            self.mdf = MDF(mdf_path)

        elif file_name.suffix.lower() == ".zip":
            progress.setLabelText("Opening zipped MF4 file")
            from mfile import ZIP

            self.mdf = ZIP(file_name)

        else:

            if file_name.suffix.lower() == ".dl3":
                progress.setLabelText("Converting from dl3 to mdf")
                datalyser_active = any(
                    proc.name() == "Datalyser3.exe" for proc in psutil.process_iter()
                )

                out_file = Path(gettempdir()) / file_name.name

                import win32com.client

                index = 0
                while True:
                    mdf_name = out_file.with_suffix(f".{index}.mdf")
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

        channels_db_items = sorted(self.mdf.channels_db, key=lambda x: x.lower())
        self.channels_db_items = channels_db_items

        progress.setLabelText("Loading graphical elements")

        progress.setValue(37)

        splitter = QtWidgets.QSplitter(self)
        splitter.setOrientation(QtCore.Qt.Vertical)

        channel_and_search = QtWidgets.QWidget(splitter)

        self.channel_view = QtWidgets.QComboBox()
        self.channel_view.addItems(["Natural sort", "Internal file structure"])
        self.channel_view.setCurrentIndex(0)
        self.channel_view.currentIndexChanged.connect(self._update_channel_tree)

        self.channels_tree = TreeWidget(self)
        self.channels_tree.setDragEnabled(True)

        self.filter_tree = TreeWidget(self)

        vbox = QtWidgets.QVBoxLayout(channel_and_search)
        vbox.setSpacing(2)
        self.advanced_search_btn = QtWidgets.QPushButton("", channel_and_search)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/search.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.advanced_search_btn.setIcon(icon)
        self.advanced_search_btn.setToolTip("Advanced search and select channels")
        self.advanced_search_btn.clicked.connect(self.search)
        vbox.addWidget(self.channel_view)

        vbox.addWidget(self.channels_tree, 1)

        hbox = QtWidgets.QHBoxLayout()

        self.clear_channels_btn = QtWidgets.QPushButton("", channel_and_search)
        self.clear_channels_btn.setToolTip("Reset selection")
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/erase.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.clear_channels_btn.setIcon(icon)
        self.clear_channels_btn.setObjectName("clear_channels_btn")

        self.load_channel_list_btn = QtWidgets.QPushButton("", channel_and_search)
        self.load_channel_list_btn.setToolTip("Load channel selection list")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(
            QtGui.QPixmap(":/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.load_channel_list_btn.setIcon(icon1)
        self.load_channel_list_btn.setObjectName("load_channel_list_btn")

        self.save_channel_list_btn = QtWidgets.QPushButton("", channel_and_search)
        self.save_channel_list_btn.setToolTip("Save channel selection list")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(
            QtGui.QPixmap(":/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.save_channel_list_btn.setIcon(icon2)
        self.save_channel_list_btn.setObjectName("save_channel_list_btn")

        self.select_all_btn = QtWidgets.QPushButton("", channel_and_search)
        self.select_all_btn.setToolTip("Select all channels")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(
            QtGui.QPixmap(":/checkmark.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
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
        icon3.addPixmap(
            QtGui.QPixmap(":/graph.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.create_window_btn.setIcon(icon3)
        self.create_window_btn.setObjectName("create_window_btn")
        hbox.addWidget(self.create_window_btn)

        hbox.addSpacerItem(
            QtWidgets.QSpacerItem(
                40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
            )
        )
        vbox.addLayout(hbox)

        self.mdi_area = MdiAreaWidget()
        self.mdi_area.add_window_request.connect(self.add_window)
        self.mdi_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.mdi_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.splitter.addWidget(self.mdi_area)

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

        self.channel_view.setCurrentIndex(-1)
        self.channel_view.setCurrentText(
            self._settings.value("channels_view", "Internal file structure")
        )

        progress.setValue(70)

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

        if self.mdf.version >= "4.00":
            if any(
                group.channel_group.flags & FLAG_CG_BUS_EVENT
                for group in self.mdf.groups
            ):
                self.aspects.setTabEnabled(7, True)
            else:
                self.aspects.setTabEnabled(7, False)
        else:
            self.aspects.setTabEnabled(7, False)

        progress.setValue(99)

        self.empty_channels.insertItems(0, ("skip", "zeros"))
        self.empty_channels_can.insertItems(0, ("skip", "zeros"))
        self.mat_format.insertItems(0, ("4", "5", "7.3"))
        self.oned_as.insertItems(0, ("row", "column"))
        self.export_type.insertItems(0, ("csv", "hdf5", "mat", "parquet"))
        self.export_btn.clicked.connect(self.export)
        self.export_type.currentTextChanged.connect(self.export_changed)
        self.export_type.setCurrentIndex(-1)

        # info tab
        file_stats = os.stat(self.mdf.name)
        file_info = QtWidgets.QTreeWidgetItem()
        file_info.setText(0, "File information")

        self.info.addTopLevelItem(file_info)

        children = []

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, "Path")
        item.setText(1, str(self.mdf.name))
        children.append(item)

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, "Size")
        item.setText(1, f"{file_stats.st_size / 1024 / 1024:.1f} MB")
        children.append(item)

        date_ = datetime.fromtimestamp(file_stats.st_ctime)
        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, "Created")
        item.setText(1, date_.strftime("%d-%b-%Y %H-%M-%S"))
        children.append(item)

        date_ = datetime.fromtimestamp(file_stats.st_mtime)
        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, "Last modified")
        item.setText(1, date_.strftime("%d-%b-%Y %H:%M:%S"))
        children.append(item)

        file_info.addChildren(children)

        mdf_info = QtWidgets.QTreeWidgetItem()
        mdf_info.setText(0, "MDF information")

        self.info.addTopLevelItem(mdf_info)

        children = []

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, "Version")
        item.setText(1, self.mdf.version)
        children.append(item)

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, "Program identification")
        item.setText(
            1,
            self.mdf.identification.program_identification.decode("ascii").strip(
                " \r\n\t\0"
            ),
        )
        children.append(item)

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, "Measurement start time")
        item.setText(
            1, self.mdf.header.start_time.strftime("%d-%b-%Y %H:%M:%S + %fus UTC")
        )
        children.append(item)

        item = QtWidgets.QTreeWidgetItem()
        item.setText(0, "Measurement comment")
        item.setText(1, self.mdf.header.comment)
        item.setTextAlignment(0, QtCore.Qt.AlignTop)
        children.append(item)

        channel_groups = QtWidgets.QTreeWidgetItem()
        channel_groups.setText(0, "Channel groups")
        channel_groups.setText(1, str(len(self.mdf.groups)))
        children.append(channel_groups)

        channel_groups_children = []
        for i, group in enumerate(self.mdf.groups):
            channel_group = group.channel_group
            if hasattr(channel_group, "comment"):
                comment = channel_group.comment
            else:
                comment = ""
            if comment:
                name = f"Channel group {i} ({comment})"
            else:
                name = f"Channel group {i}"

            cycles = channel_group.cycles_nr
            if self.mdf.version < "4.00":
                size = channel_group.samples_byte_nr * cycles
            else:
                if channel_group.flags & 0x1:
                    size = channel_group.samples_byte_nr + (
                        channel_group.invalidation_bytes_nr << 32
                    )
                else:
                    size = (
                        channel_group.samples_byte_nr
                        + channel_group.invalidation_bytes_nr
                    ) * cycles

            channel_group = QtWidgets.QTreeWidgetItem()
            channel_group.setText(0, name)

            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, "Channels")
            item.setText(1, f"{len(group.channels)}")
            channel_group.addChild(item)

            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, "Cycles")
            item.setText(1, str(cycles))
            channel_group.addChild(item)

            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, "Raw size")
            item.setText(1, f"{size / 1024 / 1024:.1f} MB")
            channel_group.addChild(item)

            channel_groups_children.append(channel_group)

        channel_groups.addChildren(channel_groups_children)

        channels = QtWidgets.QTreeWidgetItem()
        channels.setText(0, "Channels")
        channels.setText(
            1, str(sum(len(entry) for entry in self.mdf.channels_db.values()))
        )
        children.append(channels)

        mdf_info.addChildren(children)

        self.info.expandAll()

        self.info.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )

        # self.channels_tree.itemChanged.connect(self.select)
        self.create_window_btn.clicked.connect(self._create_window)

        self.clear_filter_btn.clicked.connect(self.clear_filter)
        self.clear_channels_btn.clicked.connect(self.clear_channels)
        self.select_all_btn.clicked.connect(self.select_all_channels)

        self.aspects.setCurrentIndex(0)

        self.aspects.currentChanged.connect(self.aspect_changed)

        progress.setValue(100)
        progress.deleteLater()

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
        if self.channel_view.currentIndex() == -1:
            return

        if self._show_filter_tree:
            widgets = (self.channels_tree, self.filter_tree)
        else:
            widgets = (self.channels_tree,)
        for widget in widgets:
            iterator = QtWidgets.QTreeWidgetItemIterator(widget)
            signals = set()

            if self.channel_view.currentIndex() == 0:
                while iterator.value():
                    item = iterator.value()

                    if item.entry[1] != 0xFFFFFFFFFFFFFFFF:
                        if item.checkState(0) == QtCore.Qt.Checked:
                            signals.add(item.entry)

                    iterator += 1
            else:
                while iterator.value():
                    item = iterator.value()

                    if item.checkState(0) == QtCore.Qt.Checked:
                        signals.add(item.entry)

                    iterator += 1

            widget.clear()

            if self.channel_view.currentIndex() == 0:
                items = []
                for i, group in enumerate(self.mdf.groups):
                    for j, ch in enumerate(group.channels):
                        entry = i, j

                        channel = TreeItem(entry, ch.name, mdf_uuid=self.uuid)
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
                widget.addTopLevelItems(items)
            else:
                for i, group in enumerate(self.mdf.groups):
                    entry = i, 0xFFFFFFFFFFFFFFFF
                    channel_group = TreeItem(entry, mdf_uuid=self.uuid)
                    comment = group.channel_group.comment
                    comment = extract_cncomment_xml(comment)

                    if comment:
                        channel_group.setText(0, f"Channel group {i} ({comment})")
                    else:
                        channel_group.setText(0, f"Channel group {i}")
                    channel_group.setFlags(
                        channel_group.flags()
                        | QtCore.Qt.ItemIsTristate
                        | QtCore.Qt.ItemIsUserCheckable
                    )

                    widget.addTopLevelItem(channel_group)

                    channels = [
                        HelperChannel(name=ch.name, entry=(i, j))
                        for j, ch in enumerate(group.channels)
                    ]

                    add_children(
                        channel_group,
                        channels,
                        group.channel_dependencies,
                        signals,
                        entries=None,
                        mdf_uuid=self.uuid,
                    )

        self._settings.setValue("channels_view", self.channel_view.currentText())

    def export_changed(self, name):
        if name == "parquet":
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(["GZIP", "SNAPPY"])
            self.export_compression.setCurrentIndex(-1)
        elif name == "hdf5":
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(["gzip", "lzf", "szip"])
            self.export_compression.setCurrentIndex(-1)
        elif name == "mat":
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(["enabled", "disabled"])
            self.export_compression.setCurrentIndex(-1)
        else:
            self.export_compression.clear()
            self.export_compression.setEnabled(False)

    def search(self):
        if self.aspects.tabText(self.aspects.currentIndex()) == "Channels":
            show_add_window = True
            widget = self.channels_tree
        else:
            show_add_window = False
            widget = self.filter_tree
        dlg = AdvancedSearch(
            self.mdf.channels_db, show_add_window=show_add_window, parent=self,
        )
        dlg.setModal(True)
        dlg.exec_()
        result = dlg.result
        if result:
            names = set()
            if self.channel_view.currentIndex() == 1:
                iterator = QtWidgets.QTreeWidgetItemIterator(widget)

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
                iterator = QtWidgets.QTreeWidgetItemIterator(widget)
                while iterator.value():
                    item = iterator.value()

                    if item.entry in result:
                        item.setCheckState(0, QtCore.Qt.Checked)
                        names.add(item.text(0))

                    iterator += 1

            if dlg.add_window_request:
                options = [
                    "New plot window",
                    "New numeric window",
                    "New tabular window",
                ] + [mdi.windowTitle() for mdi in self.mdi_area.subWindowList()]
                ret, ok = QtWidgets.QInputDialog.getItem(
                    None, "Select window type", "Type:", options, 0, False,
                )
                if ok:
                    index = options.index(ret)
                    signals = [
                        (None, *self.mdf.whereis(name)[0], self.uuid) for name in names
                    ]

                    if index == 0:
                        self.add_window(["Plot", signals])
                    elif index == 1:
                        self.add_window(["Numeric", signals])
                    elif index == 2:
                        self.add_window(["Tabular", signals])
                    else:
                        widgets = [
                            mdi.widget() for mdi in self.mdi_area.subWindowList()
                        ]
                        widget = widgets[index - 3]

                        self.add_new_channels(signals, widget)

    def to_config(self):
        config = {}

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

        config["selected_channels"] = signals

        windows = []
        for window in self.mdi_area.subWindowList():
            wid = window.widget()
            window_config = {
                "title": window.windowTitle(),
                "configuration": wid.to_config(),
            }
            if isinstance(wid, Numeric):
                window_config["type"] = "Numeric"
            elif isinstance(wid, Plot):
                window_config["type"] = "Plot"
            else:
                window_config["type"] = "Tabular"
            windows.append(window_config)

        config["windows"] = windows

        return config

    def save_channel_list(self, event=None, file_name=None):

        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Select output channel list file", "", "TXT files (*.txt)"
            )

        if file_name:
            Path(file_name).write_text(
                json.dumps(self.to_config(), indent=4, sort_keys=True)
            )

    def load_channel_list(self, event=None, file_name=None):
        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select channel list file",
                "",
                "Config file (*.cfg);;TXT files (*.txt);;Display files (*.dsp);;All file types (*.cfg *.dsp *.txt)",
                "All file types (*.cfg *.dsp *.txt)",
            )

        if file_name:
            if not isinstance(file_name, dict):
                file_name = Path(file_name)
                if file_name.suffix.lower() == ".dsp":
                    info = load_dsp(file_name)
                    channels = info.get("display", [])

                else:
                    with open(file_name, "r") as infile:
                        info = json.load(infile)
                    channels = info.get("selected_channels", [])

            else:
                info = file_name
                channels = info.get("selected_channels", [])

            if channels:

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

            for window in info.get("windows", []):
                self.load_window(window)

    def save_filter_list(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output filter list file", "", "TXT files (*.txt)"
        )

        if file_name:
            with open(file_name, "w") as output:
                iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

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

    def load_filter_list(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select filter list file", "", "TXT files (*.txt)"
        )

        if file_name:
            self.aspects.setCurrentIndex(4)
            with open(file_name, "r") as infile:
                channels = [line.strip() for line in infile.readlines()]
                channels = [name for name in channels if name]

            iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

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

    def compute_cut_hints(self):
        t_min = []
        t_max = []
        for i, group in enumerate(self.mdf.groups):
            cycles_nr = group.channel_group.cycles_nr
            if cycles_nr:
                master_min = self.mdf.get_master(i, record_offset=0, record_count=1,)
                if len(master_min):
                    t_min.append(master_min[0])
                self.mdf._master_channel_cache.clear()
                master_max = self.mdf.get_master(
                    i, record_offset=cycles_nr - 1, record_count=1,
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
        group_index, index = item.entry
        if index == 0xFFFFFFFFFFFFFFFF:
            group = self.mdf.groups[group_index]

            msg = ChannelGroupInfoDialog(self.mdf, group, group_index, self)
            msg.show()
        else:
            channel = self.mdf.get_channel_metadata(group=group_index, index=index)

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

        if self.channel_view.currentIndex() == 1:
            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    item.setExpanded(False)
                else:
                    item.setCheckState(0, QtCore.Qt.Unchecked)
                iterator += 1
        else:
            while iterator.value():
                item = iterator.value()
                item.setCheckState(0, QtCore.Qt.Unchecked)
                iterator += 1

    def select_all_channels(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)

        if self.channel_view.currentIndex() == 1:
            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    item.setExpanded(False)
                else:
                    item.setCheckState(0, QtCore.Qt.Checked)
                iterator += 1
        else:
            while iterator.value():
                item = iterator.value()
                item.setCheckState(0, QtCore.Qt.Checked)
                iterator += 1

    def close(self):
        mdf_name = self.mdf.name
        self.mdf.close()
        if self.file_name.suffix.lower() in (".dl3", ".erg"):
            mdf_name.unlink()
        self.channels_tree.clear()
        self.filter_tree.clear()

        self.mdf = None

    def convert(self, event):
        version = self.convert_format.currentText()

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = ".mdf"
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = ".mf4"

        split = self.convert_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.convert_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.convert_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            "",
            f"{filter};;All files (*.*)",
            filter,
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
            suffix = ".mdf"
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = ".mf4"

        split = self.resample_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.resample_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.resample_compression.currentIndex()
        time_from_zero = self.resample_time_from_zero.checkState() == QtCore.Qt.Checked

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            "",
            f"{filter};;All files (*.*)",
            filter,
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
            suffix = ".mdf"
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = ".mf4"

        split = self.cut_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.cut_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.cut_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            "",
            f"{filter};;All files (*.*)",
            filter,
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
            "hdf5": "HDF5 files (*.hdf)",
            "mat": "Matlab MAT files (*.mat)",
            "parquet": "Apache Parquet files (*.parquet)",
        }

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select export file",
            "",
            f"{filters[export_type]};;All files (*.*)",
            filters[export_type],
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title="Export measurement",
                message=f'Exporting "{self.file_name}" to {export_type}',
                icon_name="export",
            )

            if export_type == "mat":
                if compression:
                    compression = True if compression == "enabled" else False
                else:
                    compression = False

            # cut self.mdf
            target = self.mdf.export
            kwargs = {
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

    def _create_window(self, event):

        ret, ok = QtWidgets.QInputDialog.getItem(
            self,
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
                                signals.append((None, group, index, self.uuid))

                        iterator += 1
                else:
                    while iterator.value():
                        item = iterator.value()

                        if item.checkState(0) == QtCore.Qt.Checked:
                            group, index = item.entry
                            ch = self.mdf.groups[group].channels[index]
                            if not ch.component_addr:
                                signals.append((None, group, index, self.uuid))

                        iterator += 1

            self.add_window((ret, signals))

    def filter(self, event):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

        channels = []

        if self.channel_view.currentIndex() == 1:
            while iterator.value():
                item = iterator.value()

                if item.checkState(0) == QtCore.Qt.Checked:
                    group, index = item.entry

                    if index != 0xFFFFFFFFFFFFFFFF:
                        channels.append((None, group, index))

                iterator += 1
        else:
            while iterator.value():
                item = iterator.value()

                if item.checkState(0) == QtCore.Qt.Checked:
                    group, index = item.entry
                    channels.append((None, group, index))

                iterator += 1

        version = self.filter_format.itemText(self.filter_format.currentIndex())

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = ".mdf"
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = ".mf4"

        split = self.filter_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.filter_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.filter_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            "",
            f"{filter};;All files (*.*)",
            filter,
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
            self, target=target, kwargs=kwargs, factor=100, offset=0, progress=progress,
        )

        if mdf is TERMINATED:
            progress.cancel()
            return

        self.progress = None
        progress.cancel()

        path = Path(self.file_name)

        self.open_new_file.emit(str(path.with_suffix(f".scrambled{path.suffix}")))

    def extract_can_logging(self, event):
        version = self.extract_can_format.currentText()
        count = self.can_database_list.count()

        self.output_info_can.setPlainText("")

        dbc_files = []
        for i in range(count):
            item = self.can_database_list.item(i)
            dbc_files.append(item.text())

        compression = self.extract_can_compression.currentIndex()
        ignore_invalid_signals = (
            self.ignore_invalid_signals_mdf.checkState() == QtCore.Qt.Checked
        )

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = ".mdf"
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = ".mf4"

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            "",
            f"{filter};;All files (*.*)",
            filter,
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

            call_info = dict(self.mdf.last_call_info)

            found_id_count = sum(len(e) for e in call_info["found_ids"].values())

            message = [
                "Summary:",
                f'- {found_id_count} of {len(call_info["total_unique_ids"])} IDs in the MDF4 file were matched in the DBC and converted',
            ]
            if call_info["unknown_id_count"]:
                message.append(
                    f'- {call_info["unknown_id_count"]} unknown IDs in the MDF4 file'
                )
            else:
                message.append(f"- no unknown IDs inf the MDF4 file")

            message += [
                "",
                "Detailed information:",
                "",
                "The following CAN IDs were in the MDF log file and matched in the DBC:",
            ]
            for dbc_name, found_ids in call_info["found_ids"].items():
                for msg_id, msg_name in sorted(found_ids):
                    message.append(f"- 0x{msg_id:X} --> {msg_name} in <{dbc_name}>")

            message += [
                "",
                "The following CAN IDs were in the MDF log file, but not matched in the DBC:",
            ]
            for msg_id in sorted(call_info["unknown_ids"]):
                message.append(f"- 0x{msg_id:X}")

            self.output_info_can.setPlainText("\n".join(message))

            self.open_new_file.emit(str(file_name))

    def extract_can_csv_logging(self, event):
        version = self.extract_can_format.currentText()
        count = self.can_database_list.count()

        self.output_info_can.setPlainText("")

        dbc_files = []
        for i in range(count):
            item = self.can_database_list.item(i)
            dbc_files.append(item.text())

        ignore_invalid_signals = (
            self.ignore_invalid_signals_csv.checkState() == QtCore.Qt.Checked
        )
        single_time_base = self.single_time_base_can.checkState() == QtCore.Qt.Checked
        time_from_zero = self.time_from_zero_can.checkState() == QtCore.Qt.Checked
        empty_channels = self.empty_channels_can.currentText()
        raster = self.export_raster.value()
        time_as_date = self.can_time_as_date.checkState() == QtCore.Qt.Checked

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output CSV file",
            "",
            "CSV (*.csv);;All files (*.*)",
            "CSV (*.csv)",
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

            call_info = dict(self.mdf.last_call_info)

            found_id_count = sum(len(e) for e in call_info["found_ids"].values())

            message = [
                "Summary:",
                f'- {found_id_count} of {len(call_info["total_unique_ids"])} IDs in the MDF4 file were matched in the DBC and converted',
            ]
            if call_info["unknown_id_count"]:
                message.append(
                    f'- {call_info["unknown_id_count"]} unknown IDs in the MDF4 file'
                )
            else:
                message.append(f"- no unknown IDs inf the MDF4 file")

            message += [
                "",
                "Detailed information:",
                "",
                "The following CAN IDs were in the MDF log file and matched in the DBC:",
            ]
            for dbc_name, found_ids in call_info["found_ids"].items():
                for msg_id, msg_name in sorted(found_ids):
                    message.append(f"- 0x{msg_id:X} --> {msg_name} in <{dbc_name}>")

            message += [
                "",
                "The following CAN IDs were in the MDF log file, but not matched in the DBC:",
            ]
            for msg_id in sorted(call_info["unknown_ids"]):
                message.append(f"- 0x{msg_id:X}")

            self.output_info_can.setPlainText("\n".join(message))

    def load_can_database(self, event):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select CAN database file",
            "",
            "ARXML or DBC (*.dbc *.axml)",
            "ARXML or DBC (*.dbc *.axml)",
        )

        if file_names:
            self.can_database_list.addItems(file_names)

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if key == QtCore.Qt.Key_F and modifier == QtCore.Qt.ControlModifier:
            self.search()
        else:
            super().keyPressEvent(event)

    def aspect_changed(self, index):
        if (
            self.aspects.tabText(self.aspects.currentIndex()) == "Resample"
            and not self.raster_channel.count()
        ):
            self.raster_channel.setSizeAdjustPolicy(
                QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
            )
            self.raster_channel.addItems(self.channels_db_items)
            self.raster_channel.setMinimumWidth(100)
        elif (
            self.aspects.tabText(self.aspects.currentIndex()) == "Filter"
            and not self._show_filter_tree
        ):
            self._show_filter_tree = True

            widget = self.filter_tree

            if self.channel_view.currentIndex() == 0:
                items = []
                for i, group in enumerate(self.mdf.groups):
                    for j, ch in enumerate(group.channels):
                        entry = i, j

                        channel = TreeItem(entry, ch.name)
                        channel.setText(0, ch.name)
                        channel.setCheckState(0, QtCore.Qt.Unchecked)
                        items.append(channel)

                if len(items) < 30000:
                    items = natsorted(items, key=lambda x: x.name)
                else:
                    items.sort(key=lambda x: x.name)
                widget.addTopLevelItems(items)
            else:
                for i, group in enumerate(self.mdf.groups):
                    entry = i, 0xFFFFFFFFFFFFFFFF
                    channel_group = TreeItem(entry)
                    comment = group.channel_group.comment
                    comment = extract_cncomment_xml(comment)

                    if comment:
                        channel_group.setText(0, f"Channel group {i} ({comment})")
                    else:
                        channel_group.setText(0, f"Channel group {i}")
                    channel_group.setFlags(
                        channel_group.flags()
                        | QtCore.Qt.ItemIsTristate
                        | QtCore.Qt.ItemIsUserCheckable
                    )

                    widget.addTopLevelItem(channel_group)

                    channels = [
                        HelperChannel(name=ch.name, entry=(i, j))
                        for j, ch in enumerate(group.channels)
                    ]

                    add_children(
                        channel_group,
                        channels,
                        group.channel_dependencies,
                        set(),
                        entries=None,
                    )

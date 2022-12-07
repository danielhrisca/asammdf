# -*- coding: utf-8 -*-
from datetime import datetime, timezone
from functools import partial
from hashlib import sha1
import json
import os
from pathlib import Path
import re
from tempfile import gettempdir
from traceback import format_exc

from natsort import natsorted
import pandas as pd
import psutil
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ...blocks.utils import extract_xml_comment
from ...blocks.v4_constants import (
    BUS_TYPE_CAN,
    BUS_TYPE_ETHERNET,
    BUS_TYPE_FLEXRAY,
    BUS_TYPE_LIN,
    BUS_TYPE_USB,
    FLAG_AT_TO_STRING,
    FLAG_CG_BUS_EVENT,
)
from ...mdf import MDF, SUPPORTED_VERSIONS
from ..dialogs.advanced_search import AdvancedSearch
from ..dialogs.channel_group_info import ChannelGroupInfoDialog
from ..dialogs.channel_info import ChannelInfoDialog
from ..dialogs.gps_dialog import GPSDialog
from ..dialogs.window_selection_dialog import WindowSelectionDialog
from ..ui import resource_rc
from ..ui.file_widget import Ui_file_widget
from ..utils import (
    flatten_dsp,
    HelperChannel,
    load_dsp,
    load_lab,
    run_thread_with_progress,
    setup_progress,
    TERMINATED,
)
from .attachment import Attachment
from .can_bus_trace import CANBusTrace
from .database_item import DatabaseItem
from .flexray_bus_trace import FlexRayBusTrace
from .gps import GPS
from .lin_bus_trace import LINBusTrace
from .mdi_area import get_functions, MdiAreaWidget, WithMDIArea
from .numeric import Numeric
from .plot import Plot
from .tabular import Tabular
from .tree import add_children
from .tree_item import TreeItem


def _process_dict(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = _process_dict(v)
        if k == "mdf_uuid":
            k = "origin_uuid"
        new_d[k] = v

    return new_d


FRIENDLY_ATRRIBUTES = {
    "author": "Author",
    "subject": "Subject",
    "host": "Host",
    "department": "Department",
    "pr_project": "Project",
    "project": "Project Name",
    "pr_location": "Location",
    "pr_surface": "Surface",
    "pr_manover": "Maneuver",
    "pr_manufacturer": "Manufacturere",
    "pr_platform": "Platform",
    "pr_vehicle": "Vehicle",
    "pr_weight": "Vehicle Weight",
    "pr_tire": "Tire Make and Size",
    "pr_transmission": "Transmission",
    "pr_transm_mode": "Transmission Mode",
    "pr_specification": "Specification",
    "pr_test_report": "Test report",
}


class FileWidget(WithMDIArea, Ui_file_widget, QtWidgets.QWidget):

    open_new_file = QtCore.Signal(str)
    full_screen_toggled = QtCore.Signal()
    display_file_modified = QtCore.Signal(str)

    def __init__(
        self,
        file_name,
        with_dots,
        subplots=False,
        subplots_link=False,
        ignore_value2text_conversions=False,
        display_cg_name=False,
        line_interconnect="line",
        line_width=1,
        password=None,
        hide_missing_channels=False,
        hide_disabled_channels=False,
        *args,
        **kwargs,
    ):

        self.default_folder = kwargs.get("default_folder", "")
        if "default_folder" in kwargs:
            kwargs.pop("default_folder")

        self.loaded_display_file = Path(""), b""

        self.line_width = line_width

        super(Ui_file_widget, self).__init__(*args, **kwargs)
        WithMDIArea.__init__(self)
        self.setupUi(self)
        self._settings = QtCore.QSettings()
        self.uuid = os.urandom(6).hex()

        self.hide_missing_channels = hide_missing_channels
        self.hide_disabled_channels = hide_disabled_channels
        self.display_cg_name = display_cg_name

        file_name = Path(file_name)
        self.subplots = subplots
        self.subplots_link = subplots_link
        self.ignore_value2text_conversions = ignore_value2text_conversions

        self.file_name = file_name
        self.progress = None
        self.mdf = None
        self.info_index = None
        self.with_dots = with_dots

        self._show_filter_tree = False
        self.line_interconnect = line_interconnect

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
        progress.setMinimumWidth(600)
        progress.show()

        try:
            if file_name.suffix.lower() in (".erg", ".bsig", ".dl3"):

                extension = file_name.suffix.lower().strip(".")
                progress.setLabelText(f"Converting from {extension} to mdf")

                try:
                    from mfile import BSIG, DL3, ERG
                except ImportError:
                    from cmerg import BSIG, ERG

                if file_name.suffix.lower() == ".erg":
                    cls = ERG
                elif file_name.suffix.lower() == ".bsig":
                    cls = BSIG
                else:
                    cls = DL3

                out_file = Path(gettempdir()) / file_name.name
                meas_file = cls(file_name)

                mdf_path = meas_file.export_mdf().save(out_file.with_suffix(".tmp.mf4"))
                meas_file.close()
                self.mdf = MDF(mdf_path)
                self.mdf.original_name = file_name
                self.mdf.uuid = self.uuid

            elif file_name.suffix.lower() == ".csv":
                try:
                    with open(file_name) as csv:
                        names = [n.strip() for n in csv.readline().split(",")]
                        units = [n.strip() for n in csv.readline().split(",")]

                        try:
                            float(units[0])
                        except:
                            units = {name: unit for name, unit in zip(names, units)}
                        else:
                            csv.seek(0)
                            csv.readline()
                            units = None

                        df = pd.read_csv(csv, header=None, names=names)
                        df.set_index(df[names[0]], inplace=True)
                        self.mdf = MDF()
                        self.mdf.append(df, units=units)
                        self.mdf.uuid = self.uuid
                        self.mdf.original_name = file_name
                except:
                    progress.cancel()
                    print(format_exc())
                    raise Exception(
                        "Could not load CSV. The first line must contain the channel names. The seconds line "
                        "can optionally contain the channel units. The first column must be the time"
                    )

            else:

                original_name = file_name

                target = MDF
                kwargs = {
                    "name": file_name,
                    "callback": self.update_progress,
                    "password": password,
                    "use_display_names": True,
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

                self.mdf.original_name = original_name
                self.mdf.uuid = self.uuid

            self.mdf.configure(raise_on_multiple_occurrences=False)

            channels_db_items = sorted(self.mdf.channels_db, key=lambda x: x.lower())
            self.channels_db_items = channels_db_items

            progress.setLabelText("Loading graphical elements")

            progress.setValue(37)

            self.channels_tree.setDragEnabled(True)

            self.mdi_area = MdiAreaWidget()

            self.mdi_area.add_window_request.connect(self.add_window)
            self.mdi_area.open_file_request.connect(self.open_new_file.emit)
            self.mdi_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            self.mdi_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            self.splitter.addWidget(self.mdi_area)

            self.channels_tree.itemDoubleClicked.connect(self.show_info)
            self.filter_tree.itemDoubleClicked.connect(self.show_info)

            self.channel_view.setCurrentIndex(-1)
            self.filter_view.setCurrentIndex(-1)

            self.filter_view.setCurrentText(
                self._settings.value("filter_view", "Internal file structure")
            )

            self.channel_view.currentIndexChanged.connect(
                partial(self._update_channel_tree, widget=self.channels_tree)
            )
            self.filter_view.currentIndexChanged.connect(
                partial(self._update_channel_tree, widget=self.filter_tree)
            )
            self.channel_view.currentTextChanged.connect(
                partial(self._update_channel_tree, widget=self.channels_tree)
            )
            self.filter_view.currentTextChanged.connect(
                partial(self._update_channel_tree, widget=self.filter_tree)
            )

            self.channel_view.setCurrentText(
                self._settings.value("channels_view", "Internal file structure")
            )

            progress.setValue(70)

            self.raster_type_channel.toggled.connect(self.set_raster_type)

            progress.setValue(90)

            self.output_options.setCurrentIndex(0)

            self.mdf_version.insertItems(0, SUPPORTED_VERSIONS)
            self.mdf_version.setCurrentText("4.10")
            self.mdf_compression.insertItems(
                0, ("no compression", "deflate", "transposed deflate")
            )
            self.mdf_compression.setCurrentText("transposed deflate")
            self.mdf_split_size.setValue(4)

            self.extract_bus_format.insertItems(0, SUPPORTED_VERSIONS)
            self.extract_bus_format.setCurrentText("4.10")
            index = self.extract_bus_format.findText(self.mdf.version)
            if index >= 0:
                self.extract_bus_format.setCurrentIndex(index)
            self.extract_bus_compression.insertItems(
                0, ("no compression", "deflate", "transposed deflate")
            )
            self.extract_bus_compression.setCurrentText("transposed deflate")
            self.extract_bus_btn.clicked.connect(self.extract_bus_logging)
            self.extract_bus_csv_btn.clicked.connect(self.extract_bus_csv_logging)
            self.load_can_database_btn.clicked.connect(self.load_can_database)
            self.load_lin_database_btn.clicked.connect(self.load_lin_database)

            progress.setValue(99)

            self.empty_channels.insertItems(0, ("skip", "zeros"))
            self.empty_channels_bus.insertItems(0, ("skip", "zeros"))
            self.empty_channels_mat.insertItems(0, ("skip", "zeros"))
            self.empty_channels_csv.insertItems(0, ("skip", "zeros"))
            try:
                import scipy

                self.mat_format.insertItems(0, ("4", "5", "7.3"))
            except:
                self.mat_format.insertItems(0, ("7.3",))
            self.oned_as.insertItems(0, ("row", "column"))

            self.output_format.currentTextChanged.connect(self.output_format_changed)

            # self.channels_tree.itemChanged.connect(self.select)
            self.create_window_btn.clicked.connect(self._create_window)

            self.clear_filter_btn.clicked.connect(self.clear_filter)
            self.clear_channels_btn.clicked.connect(self.clear_channels)
            self.select_all_btn.clicked.connect(self.select_all_channels)

            self.aspects.setCurrentIndex(0)

            self.aspects.currentChanged.connect(self.aspect_changed)

        except:

            progress.setValue(100)
            progress.deleteLater()
            raise

        else:
            progress.setValue(100)
            progress.deleteLater()

        self.load_channel_list_btn.clicked.connect(self.load_channel_list)
        self.save_channel_list_btn.clicked.connect(self.save_channel_list)
        self.load_filter_list_btn.clicked.connect(self.load_filter_list)
        self.save_filter_list_btn.clicked.connect(self.save_filter_list)
        self.advanced_search_btn.clicked.connect(self.search)
        self.advanced_serch_filter_btn.clicked.connect(self.search)
        self.raster_search_btn.clicked.connect(self.raster_search)

        self.filter_tree.itemChanged.connect(self.filter_changed)
        self._selected_filter = set()
        self._filter_timer = QtCore.QTimer()
        self._filter_timer.setSingleShot(True)
        self._filter_timer.timeout.connect(self.update_selected_filter_channels)

        self.scramble_btn.clicked.connect(self.scramble)
        self.setAcceptDrops(True)

        self.apply_btn.clicked.connect(self.apply_processing)

        if self.mdf.version >= "4.00" and self.mdf.attachments:
            for i, attachment in enumerate(self.mdf.attachments, 1):
                att = Attachment(i - 1, self.mdf)
                att.number.setText(f"{i}.")

                fields = []

                field = QtWidgets.QTreeWidgetItem()
                field.setText(0, "ATBLOCK address")
                field.setText(1, f"0x{attachment.address:X}")
                fields.append(field)

                field = QtWidgets.QTreeWidgetItem()
                field.setText(0, "File name")
                field.setText(1, str(attachment.file_name))
                fields.append(field)

                field = QtWidgets.QTreeWidgetItem()
                field.setText(0, "MIME type")
                field.setText(1, attachment.mime)
                fields.append(field)

                field = QtWidgets.QTreeWidgetItem()
                field.setText(0, "Comment")
                field.setText(1, attachment.comment)
                fields.append(field)

                field = QtWidgets.QTreeWidgetItem()
                field.setText(0, "Flags")
                if attachment.flags:
                    flags = []
                    for flag, string in FLAG_AT_TO_STRING.items():
                        if attachment.flags & flag:
                            flags.append(string)
                    text = f'{attachment.flags} [0x{attachment.flags:X}= {", ".join(flags)}]'
                else:
                    text = "0"
                field.setText(1, text)
                fields.append(field)

                field = QtWidgets.QTreeWidgetItem()
                field.setText(0, "MD5 sum")
                field.setText(1, attachment.md5_sum.hex().upper())
                fields.append(field)

                size = attachment.original_size
                if size <= 1 << 10:
                    text = f"{size} B"
                elif size <= 1 << 20:
                    text = f"{size/1024:.1f} KB"
                elif size <= 1 << 30:
                    text = f"{size/1024/1024:.1f} MB"
                else:
                    text = f"{size/1024/1024/1024:.1f} GB"

                field = QtWidgets.QTreeWidgetItem()
                field.setText(0, "Size")
                field.setText(1, text)
                fields.append(field)

                att.fields.addTopLevelItems(fields)

                item = QtWidgets.QListWidgetItem()
                item.setSizeHint(att.sizeHint())
                self.attachments.addItem(item)
                self.attachments.setItemWidget(item, att)
        else:
            self.aspects.removeTab(4)

        if self.mdf.version >= "4.00":
            if not any(
                group.channel_group.flags & FLAG_CG_BUS_EVENT
                for group in self.mdf.groups
            ):
                self.aspects.removeTab(2)
        else:
            self.aspects.removeTab(2)

        self._splitter_sizes = None

    def sizeHint(self):
        return QtCore.QSize(1, 1)

    def set_raster_type(self, event):
        if self.raster_type_channel.isChecked():
            self.raster_channel.setEnabled(True)
            self.raster.setEnabled(False)
            self.raster.setValue(0)
        else:
            self.raster_channel.setEnabled(False)
            self.raster_channel.setCurrentIndex(0)
            self.raster.setEnabled(True)

    def update_all_channel_trees(self):
        widgetList = [self.channels_tree, self.filter_tree]
        for widget in widgetList:
            self._update_channel_tree(widget=widget)

    def _update_channel_tree(self, index=None, widget=None):

        if widget is None:
            widget = self.channels_tree
        if widget is self.channels_tree and self.channel_view.currentIndex() == -1:
            return
        elif widget is self.filter_tree and (self.filter_view.currentIndex() == -1):
            return

        view = self.channel_view if widget is self.channels_tree else self.filter_view

        iterator = QtWidgets.QTreeWidgetItemIterator(widget)
        signals = set()

        if widget.mode == "Internal file structure":

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
        widget.mode = view.currentText()

        if widget.mode == "Natural sort":
            items = []
            for i, group in enumerate(self.mdf.groups):
                for j, ch in enumerate(group.channels):
                    entry = i, j

                    channel = TreeItem(entry, ch.name, origin_uuid=self.uuid)
                    channel.setToolTip(0, f"{ch.name} @ group {i}, index {j}")
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

        elif widget.mode == "Internal file structure":

            items = []

            for i, group in enumerate(self.mdf.groups):
                entry = i, 0xFFFFFFFFFFFFFFFF

                channel_group = TreeItem(entry, origin_uuid=self.uuid)

                comment = extract_xml_comment(group.channel_group.comment)

                if self.mdf.version >= "4.00" and group.channel_group.acq_source:
                    source = group.channel_group.acq_source
                    if source.bus_type == BUS_TYPE_CAN:
                        ico = ":/bus_can.png"
                    elif source.bus_type == BUS_TYPE_LIN:
                        ico = ":/bus_lin.png"
                    elif source.bus_type == BUS_TYPE_ETHERNET:
                        ico = ":/bus_eth.png"
                    elif source.bus_type == BUS_TYPE_USB:
                        ico = ":/bus_usb.png"
                    elif source.bus_type == BUS_TYPE_FLEXRAY:
                        ico = ":/bus_flx.png"
                    else:
                        ico = None

                    if ico is not None:

                        icon = QtGui.QIcon()
                        icon.addPixmap(
                            QtGui.QPixmap(ico), QtGui.QIcon.Normal, QtGui.QIcon.Off
                        )

                        channel_group.setIcon(0, icon)

                if self.display_cg_name:
                    if group.channel_group.acq_name:
                        base_name = f"CG {i} {group.channel_group.acq_name}"
                    else:
                        base_name = f"CG {i}"
                    if comment and group.channel_group.acq_name != comment:
                        name = base_name + f" ({comment})"
                    else:
                        name = base_name

                else:
                    base_name = f"Channel group {i}"
                    if comment:
                        name = base_name + f" ({comment})"
                    else:
                        name = base_name

                channel_group.setText(0, name)
                channel_group.setFlags(
                    channel_group.flags()
                    | QtCore.Qt.ItemIsAutoTristate
                    | QtCore.Qt.ItemIsUserCheckable
                )

                # widget.addTopLevelItems(i, channel_group)
                items.append(channel_group)

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
                    origin_uuid=self.uuid,
                    version=self.mdf.version,
                )

            widget.addTopLevelItems(items)

        else:
            items = []
            for entry in signals:
                gp_index, ch_index = entry
                ch = self.mdf.groups[gp_index].channels[ch_index]
                channel = TreeItem(entry, ch.name, origin_uuid=self.uuid)
                channel.setToolTip(0, f"{ch.name} @ group {gp_index}, index {ch_index}")
                channel.setText(0, ch.name)
                channel.setCheckState(0, QtCore.Qt.Checked)
                items.append(channel)

            if len(items) < 30000:
                items = natsorted(items, key=lambda x: x.name)
            else:
                items.sort(key=lambda x: x.name)
            widget.addTopLevelItems(items)

        setting = "channels_view" if widget is self.channels_tree else "filter_view"
        self._settings.setValue(setting, view.currentText())

    def output_format_changed(self, name):
        if name == "MDF":
            self.output_options.setCurrentIndex(0)
        elif name == "MAT":
            self.output_options.setCurrentIndex(2)

            self.export_compression_mat.clear()
            self.export_compression_mat.addItems(["enabled", "disabled"])
            self.export_compression_mat.setCurrentIndex(0)
        elif name == "CSV":
            self.output_options.setCurrentIndex(3)

        else:
            self.output_options.setCurrentIndex(1)
            if name == "Parquet":
                self.export_compression.setEnabled(True)
                self.export_compression.clear()
                self.export_compression.addItems(["GZIP", "SNAPPY"])
                self.export_compression.setCurrentIndex(0)
            elif name == "HDF5":
                self.export_compression.setEnabled(True)
                self.export_compression.clear()
                self.export_compression.addItems(["gzip", "lzf", "szip"])
                self.export_compression.setCurrentIndex(0)
            elif name:
                self.export_compression.clear()
                self.export_compression.setEnabled(False)

    def search(self, event=None):
        toggle_frames = False
        if self.aspects.tabText(self.aspects.currentIndex()) == "Channels":
            show_add_window = True
            show_apply = True
            apply_text = "Check channels"
            widget = self.channels_tree
            view = self.channel_view

            if self._frameless_windows:
                toggle_frames = True
                self.toggle_frames()
        else:
            show_add_window = False
            show_apply = True
            apply_text = "Check channels"
            widget = self.filter_tree
            view = self.filter_view

        dlg = AdvancedSearch(
            self.mdf,
            show_add_window=show_add_window,
            show_apply=show_apply,
            apply_text=apply_text,
            parent=self,
        )
        dlg.setModal(True)
        dlg.exec_()
        result, pattern_window = dlg.result, dlg.pattern_window

        if result:
            if pattern_window:
                options = [
                    "New pattern based plot window",
                    "New pattern based numeric window",
                    "New pattern based tabular window",
                ]

                dialog = WindowSelectionDialog(options=options, parent=self)
                dialog.setModal(True)
                dialog.exec_()

                if dialog.result():
                    window_type = dialog.selected_type()

                    if window_type == "New pattern based plot window":
                        self.load_window(
                            {
                                "type": "Plot",
                                "title": result["pattern"],
                                "configuration": {"channels": [], "pattern": result},
                            }
                        )
                    elif window_type == "New pattern based numeric window":
                        self.load_window(
                            {
                                "type": "Numeric",
                                "title": result["pattern"],
                                "configuration": {
                                    "channels": [],
                                    "pattern": result,
                                    "format": "phys",
                                },
                            }
                        )
                    elif window_type == "New pattern based tabular window":
                        self.load_window(
                            {
                                "type": "Tabular",
                                "title": result["pattern"],
                                "configuration": {
                                    "channels": [],
                                    "pattern": result,
                                    "filters": [],
                                    "time_as_date": False,
                                    "sorted": False,
                                    "filtered": False,
                                },
                            }
                        )

            else:

                names = set()
                if view.currentText() == "Internal file structure":
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

                        entry = (dg_cntr, ch_cntr)

                        if entry in result:
                            item.setCheckState(0, QtCore.Qt.Checked)
                            names.add((result[entry], dg_cntr, ch_cntr))

                        iterator += 1
                        ch_cntr += 1
                elif view.currentText() == "Selected channels only":
                    iterator = QtWidgets.QTreeWidgetItemIterator(widget)

                    signals = set()
                    while iterator.value():
                        item = iterator.value()

                        if item.checkState(0) == QtCore.Qt.Checked:
                            signals.add((item.text(0), *item.entry))

                        iterator += 1

                    names = set((_name, *entry) for entry, _name in result.items())

                    signals = signals | names

                    widget.clear()

                    items = []
                    for name, gp_index, ch_index in signals:
                        entry = gp_index, ch_index
                        ch = self.mdf.groups[gp_index].channels[ch_index]
                        channel = TreeItem(entry, ch.name, origin_uuid=self.uuid)
                        channel.setText(0, ch.name)
                        channel.setCheckState(0, QtCore.Qt.Checked)
                        items.append(channel)

                    if len(items) < 30000:
                        items = natsorted(items, key=lambda x: x.name)
                    else:
                        items.sort(key=lambda x: x.name)
                    widget.addTopLevelItems(items)

                else:
                    iterator = QtWidgets.QTreeWidgetItemIterator(widget)
                    while iterator.value():
                        item = iterator.value()

                        if item.entry in result:
                            item.setCheckState(0, QtCore.Qt.Checked)
                            names.add((result[item.entry], *item.entry))

                        iterator += 1

                if dlg.add_window_request:
                    options = [
                        "New plot window",
                        "New numeric window",
                        "New tabular window",
                    ] + [
                        mdi.windowTitle()
                        for mdi in self.mdi_area.subWindowList()
                        if not isinstance(mdi.widget(), CANBusTrace)
                    ]

                    dialog = WindowSelectionDialog(options=options, parent=self)
                    dialog.setModal(True)
                    dialog.exec_()

                    if dialog.result():
                        window_type = dialog.selected_type()

                        signals = natsorted(
                            [
                                {
                                    "name": name,
                                    "group_index": dg_cntr,
                                    "channel_index": ch_cntr,
                                    "origin_uuid": self.uuid,
                                    "type": "channel",
                                    "ranges": [],
                                    "uuid": os.urandom(6).hex(),
                                    "enabled": True,
                                }
                                for name, dg_cntr, ch_cntr in names
                            ],
                            key=lambda x: (
                                x["name"],
                                x["group_index"],
                                x["channel_index"],
                            ),
                        )

                        if window_type == "New plot window":
                            self.add_window(["Plot", signals])
                        elif window_type == "New numeric window":
                            self.add_window(["Numeric", signals])
                        elif window_type == "New tabular window":
                            self.add_window(["Tabular", signals])
                        else:
                            for mdi in self.mdi_area.subWindowList():
                                if mdi.windowTitle() == window_type:
                                    self.add_new_channels(signals, mdi.widget())
                                    break

        if toggle_frames:
            self.toggle_frames()

    def to_config(self):
        config = {}

        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)

        signals = []
        if self.channel_view.currentText() == "Internal file structure":
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
            geometry = window.geometry()
            window_config = {
                "title": window.windowTitle(),
                "configuration": wid.to_config(),
                "geometry": [
                    geometry.x(),
                    geometry.y(),
                    geometry.width(),
                    geometry.height(),
                ],
                "maximized": window.isMaximized(),
                "minimized": window.isMinimized(),
            }
            if isinstance(wid, Numeric):
                window_config["type"] = "Numeric"
            elif isinstance(wid, Plot):
                window_config["type"] = "Plot"
                if wid.closed:
                    continue
                del window_config["configuration"]["x_range"]
            elif isinstance(wid, Tabular):
                window_config["type"] = "Tabular"
            elif isinstance(wid, GPS):
                window_config["type"] = "GPS"
            elif isinstance(wid, CANBusTrace):
                window_config["type"] = "CAN Bus Trace"
            elif isinstance(wid, FlexRayBusTrace):
                window_config["type"] = "FlexRay Bus Trace"
            elif isinstance(wid, LINBusTrace):
                window_config["type"] = "LIN Bus Trace"
            else:
                continue

            windows.append(window_config)

        config["windows"] = windows
        config["functions"] = self.functions

        return config

    def save_channel_list(self, event=None, file_name=None):

        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select output display file",
                self.default_folder,
                "Display files (*.dspf)",
            )

        if file_name:
            file_name = Path(file_name)
            file_name.write_text(json.dumps(self.to_config(), indent=2))

            loaded_display_file, hash_sum = self.loaded_display_file
            if (
                file_name.samefile(loaded_display_file)
                and file_name.suffix.lower() == ".dspf"
            ):
                worker = sha1()
                worker.update(loaded_display_file.read_bytes())
                self.loaded_display_file = loaded_display_file, worker.hexdigest()

    def load_channel_list(self, event=None, file_name=None):
        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select channel list file",
                self.default_folder,
                "Config file (*.cfg);;TXT files (*.txt);;Display files (*.dsp *.dspf);;CANape Lab file (*.lab);;All file types (*.cfg *.dsp *.dspf *.lab *.txt)",
                "All file types (*.cfg *.dsp *.dspf *.lab *.txt)",
            )

        if file_name:
            if not isinstance(file_name, dict):
                file_name = Path(file_name)

                extension = file_name.suffix.lower()
                if extension == ".dsp":
                    palette = self.palette()
                    info = load_dsp(file_name, palette.color(palette.Base).name())
                    if info.get("has_virtual_channels", False):
                        message = (
                            "The DSP file contains virtual channels that are not supported.\n"
                            'For tracking purpose, the virtual channels will appear as regular (no computation) channels inside the group "Datalyser Virtual channels"'
                        )
                        c_functions = info.get("c_functions", [])

                        msg = QtWidgets.QMessageBox(
                            QtWidgets.QMessageBox.Information,
                            "DSP loading warning",
                            message,
                            parent=self,
                        )
                        if c_functions:
                            msg.setInformativeText(
                                'The user defined C function will NOT be available. Press "Show details" for the complete list'
                            )
                            msg.setDetailedText("\n".join(c_functions))

                        msg.exec()
                    channels = info.get("display", [])

                elif extension == ".lab":
                    info = load_lab(file_name)
                    if info:
                        section, ok = QtWidgets.QInputDialog.getItem(
                            None,
                            "Select section",
                            "Available sections:",
                            list(info),
                            0,
                            False,
                        )
                        if ok:
                            channels = info[section]
                        else:
                            return

                elif extension in (".cfg", ".txt", ".dspf"):
                    with open(file_name, "r") as infile:
                        info = json.load(infile)
                    channels = info.get("selected_channels", [])
                else:
                    return

                worker = sha1()
                worker.update(file_name.read_bytes())
                self.loaded_display_file = file_name, worker.hexdigest()

            else:
                extension = None
                info = file_name
                channels = info.get("selected_channels", [])
                self.loaded_display_file = Path(info.get("display_file_name", "")), b""

                self.functions.update(info.get("functions", {}))

            if channels:
                iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)

                if self.channel_view.currentText() == "Internal file structure":
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

            if extension in (".dspf", ".dsp"):

                new_functions = {}

                if "functions" in info:
                    for name, definition in info["functions"].items():
                        if name in self.functions:
                            if self.functions[name] != definition:
                                new_functions[os.urandom(6).hex()] = {
                                    "name": name,
                                    "definition": definition,
                                }
                        else:
                            new_functions[os.urandom(6).hex()] = {
                                "name": name,
                                "definition": definition,
                            }

                else:
                    for window in info["windows"]:
                        if window["type"] == "Plot":
                            for name, definition in get_functions(
                                window["configuration"]["channels"]
                            ).items():
                                if name in self.functions:
                                    if self.functions[name] != definition:
                                        new_functions[os.urandom(6).hex()] = {
                                            "name": name,
                                            "definition": definition,
                                        }
                                else:
                                    new_functions[os.urandom(6).hex()] = {
                                        "name": name,
                                        "definition": definition,
                                    }

                if new_functions:
                    self.update_functions({}, new_functions)

            self.clear_windows()

            windows = info.get("windows", [])
            if windows:
                count = len(windows)

                progress = setup_progress(
                    parent=self,
                    title=f"Loading display windows",
                    message=f"",
                    icon_name="window",
                )
                progress.setRange(0, count - 1)
                progress.resize(500, progress.height())

                try:
                    for i, window in enumerate(windows, 1):
                        window = _process_dict(window)
                        window_type = window["type"]
                        window_title = window["title"]
                        progress.setLabelText(
                            f"Loading {window_type} window <{window_title}>"
                        )
                        QtWidgets.QApplication.processEvents()
                        self.load_window(window)
                        progress.setValue(i)
                except:
                    print(format_exc())
                finally:
                    progress.cancel()

            self.display_file_modified.emit(Path(self.loaded_display_file[0]).name)

    def save_filter_list(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output filter list file",
            self.default_folder,
            "CANape Lab file (*.lab);;TXT files (*.txt);;All file types (*.lab *.txt)",
            "CANape Lab file (*.lab)",
        )

        if file_name:
            file_name = Path(file_name)

            iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

            signals = []
            if self.filter_view.currentText() == "Internal file structure":
                while True:
                    item = iterator.value()
                    if item is None:
                        break

                    iterator += 1

                    if item.parent() is None:
                        continue

                    if item.checkState(0) == QtCore.Qt.Checked:
                        signals.append(item.text(0))
            else:
                while True:
                    item = iterator.value()
                    if item is None:
                        break

                    iterator += 1

                    if item.checkState(0) == QtCore.Qt.Checked:
                        signals.append(item.text(0))

            suffix = file_name.suffix.lower()
            if suffix == ".lab":
                section_name, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Provide .lab file ASAP section name",
                    "Section name:",
                )
                if not ok:
                    section_name = "Selected channels"

            with open(file_name, "w") as output:
                if suffix == ".lab":
                    output.write(f"[{section_name}]\n")
                output.write("\n".join(natsorted(signals)))

    def load_filter_list(self, event=None, file_name=None):
        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select channel list file",
                self.default_folder,
                "Config file (*.cfg);;TXT files (*.txt);;Display files (*.dsp *.dspf);;CANape Lab file (*.lab);;All file types (*.cfg *.dsp *.dspf *.lab *.txt)",
                "CANape Lab file (*.lab)",
            )

        if file_name:
            if not isinstance(file_name, dict):
                file_name = Path(file_name)

                extension = file_name.suffix.lower()
                if extension == ".dsp":
                    channels = load_dsp(file_name, flat=True)

                elif extension == ".dspf":
                    with open(file_name, "r") as infile:
                        info = json.load(infile)

                    channels = []
                    for window in info["windows"]:

                        if window["type"] == "Plot":
                            channels.extend(
                                flatten_dsp(window["configuration"]["channels"])
                            )
                        elif window["type"] == "Numeric":
                            channels.extend(
                                [
                                    item["name"]
                                    for item in window["configuration"]["channels"]
                                ]
                            )
                        elif window["type"] == "Tabular":
                            channels.extend(window["configuration"]["channels"])

                elif extension == ".lab":
                    info = load_lab(file_name)
                    if info:
                        if len(info) > 1:
                            section, ok = QtWidgets.QInputDialog.getItem(
                                None,
                                "Please select the ASAP section name",
                                "Available sections:",
                                list(info),
                                0,
                                False,
                            )
                            if ok:
                                channels = info[section]
                            else:
                                return
                        else:
                            channels = list(info.values())[0]

                elif extension == ".cfg":
                    with open(file_name, "r") as infile:
                        info = json.load(infile)
                    channels = info.get("selected_channels", [])
                elif extension == ".txt":
                    try:
                        with open(file_name, "r") as infile:
                            info = json.load(infile)
                        channels = info.get("selected_channels", [])
                    except:
                        with open(file_name, "r") as infile:
                            channels = [line.strip() for line in infile.readlines()]
                            channels = [name for name in channels if name]

            else:
                info = file_name
                channels = info.get("selected_channels", [])

            if channels:

                iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

                if self.filter_view.currentText() == "Internal file structure":
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
                elif self.filter_view.currentText() == "Natural sort":
                    while iterator.value():
                        item = iterator.value()

                        channel_name = item.text(0)
                        if channel_name in channels:
                            item.setCheckState(0, QtCore.Qt.Checked)
                            channels.pop(channels.index(channel_name))
                        else:
                            item.setCheckState(0, QtCore.Qt.Unchecked)

                        iterator += 1

                else:
                    items = []
                    self.filter_tree.clear()

                    for i, gp in enumerate(self.mdf.groups):
                        for j, ch in enumerate(gp.channels):
                            if ch.name in channels:
                                entry = i, j
                                channel = TreeItem(
                                    entry, ch.name, origin_uuid=self.uuid
                                )
                                channel.setText(0, ch.name)
                                channel.setCheckState(0, QtCore.Qt.Checked)
                                items.append(channel)

                                channels.pop(channels.index(ch.name))

                    if len(items) < 30000:
                        items = natsorted(items, key=lambda x: x.name)
                    else:
                        items.sort(key=lambda x: x.name)
                    self.filter_tree.addTopLevelItems(items)

    def compute_cut_hints(self):
        t_min = []
        t_max = []
        for i, group in enumerate(self.mdf.groups):
            cycles_nr = group.channel_group.cycles_nr
            if cycles_nr:
                master_min = self.mdf.get_master(i, record_offset=0, record_count=1)
                if len(master_min):
                    t_min.append(master_min[0])
                self.mdf._master_channel_cache.clear()
                master_max = self.mdf.get_master(
                    i, record_offset=cycles_nr - 1, record_count=1
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

        self.clear_windows()

        self.mdf = None

    def _create_window(self, event=None, window_type=None):

        if window_type is None:
            dialog = WindowSelectionDialog(
                options=(
                    "Plot",
                    "Numeric",
                    "Tabular",
                    "CAN Bus Trace",
                    "FlexRay Bus Trace",
                    "LIN Bus Trace",
                    "GPS",
                ),
                parent=self,
            )
            dialog.setModal(True)
            dialog.exec_()

            if dialog.result():
                window_type = dialog.selected_type()
            else:
                window_type = None

        if window_type is None:
            return
        elif window_type in ("CAN Bus Trace", "FlexRay Bus Trace", "LIN Bus Trace"):
            signals = []
        elif window_type == "GPS":

            target = "(latitude|gps_y)"
            sig = re.compile(target, re.IGNORECASE)

            latitude = ""

            for name in self.mdf.channels_db:
                if sig.fullmatch(name):
                    latitude = name
                    break
            else:
                for name in self.mdf.channels_db:
                    if sig.search(name):
                        latitude = name
                        break

            target = "(longitude|gps_x)"
            sig = re.compile(target, re.IGNORECASE)

            longitude = ""

            for name in self.mdf.channels_db:
                if sig.fullmatch(name):
                    longitude = name
                    break
            else:
                for name in self.mdf.channels_db:
                    if sig.search(name):
                        longitude = name
                        break

            dlg = GPSDialog(
                self.mdf,
                latitude=latitude,
                longitude=longitude,
                parent=self,
            )
            dlg.setModal(True)
            dlg.exec_()

            if dlg.valid:
                latitude = dlg.latitude.text().strip()
                longitude = dlg.longitude.text().strip()

                signals = [
                    (name, *self.mdf.whereis(name)[0], self.uuid, "channel")
                    for name in [latitude, longitude]
                    if name in self.mdf
                ]
                if len(signals) != 2:
                    return
            else:
                return
        else:

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
                                signals.append(
                                    {
                                        "name": ch.name,
                                        "group_index": group,
                                        "channel_index": index,
                                        "origin_uuid": self.uuid,
                                        "type": "channel",
                                        "ranges": [],
                                        "uuid": os.urandom(6).hex(),
                                    }
                                )

                        iterator += 1
                else:
                    while iterator.value():
                        item = iterator.value()

                        if item.checkState(0) == QtCore.Qt.Checked:
                            group, index = item.entry
                            ch = self.mdf.groups[group].channels[index]
                            if not ch.component_addr:
                                signals.append(
                                    {
                                        "name": ch.name,
                                        "group_index": group,
                                        "channel_index": index,
                                        "origin_uuid": self.uuid,
                                        "type": "channel",
                                        "ranges": [],
                                        "uuid": os.urandom(6).hex(),
                                    }
                                )

                        iterator += 1

        self.add_window((window_type, signals))

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
            self, target=target, kwargs=kwargs, factor=100, offset=0, progress=progress
        )

        if mdf is TERMINATED:
            progress.cancel()
            return

        self.progress = None
        progress.cancel()

        path = Path(self.file_name)

        self.open_new_file.emit(str(path.with_suffix(f".scrambled{path.suffix}")))

    def extract_bus_logging(self, event):
        version = self.extract_bus_format.currentText()

        self.output_info_bus.setPlainText("")

        database_files = {}

        count1 = self.can_database_list.count()
        if count1:
            database_files["CAN"] = []
            for i in range(count1):
                item = self.can_database_list.item(i)
                widget = self.can_database_list.itemWidget(item)
                database_files["CAN"].append(
                    (widget.database.text(), widget.bus.currentIndex())
                )

        count2 = self.lin_database_list.count()
        if count2:
            database_files["LIN"] = []
            for i in range(count2):
                item = self.lin_database_list.item(i)
                widget = self.lin_database_list.itemWidget(item)
                database_files["LIN"].append(
                    (widget.database.text(), widget.bus.currentIndex())
                )

        compression = self.extract_bus_compression.currentIndex()

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = ".mdf"
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = ".mf4"

        if not (count1 + count2):
            return

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
                title="Extract Bus logging",
                message=f'Extracting Bus signals from "{self.file_name}"',
                icon_name="down",
            )

            # convert self.mdf
            target = self.mdf.extract_bus_logging
            kwargs = {
                "database_files": database_files,
                "version": version,
                "prefix": self.prefix.text().strip(),
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

            bus_call_info = dict(self.mdf.last_call_info)

            message = []

            for bus, call_info in bus_call_info.items():

                found_id_count = sum(len(e) for e in call_info["found_ids"].values())

                message += [
                    f"{bus} bus summary:",
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
                    f"The following {bus} IDs were in the MDF log file and matched in the DBC:",
                ]
                for dbc_name, found_ids in call_info["found_ids"].items():
                    for msg_id, msg_name in sorted(found_ids):
                        try:
                            message.append(
                                f"- 0x{msg_id:X} --> {msg_name} in <{dbc_name}>"
                            )
                        except:
                            pgn, sa = msg_id
                            message.append(
                                f"- PGN=0x{pgn:X} SA=0x{sa:X} --> {msg_name} in <{dbc_name}>"
                            )

                message += [
                    "",
                    f"The following {bus} IDs were in the MDF log file, but not matched in the DBC:",
                ]

                unknown_standard_can = sorted(
                    [e for e in call_info["unknown_ids"] if isinstance(e, int)]
                )
                unknown_j1939 = sorted(
                    [e for e in call_info["unknown_ids"] if not isinstance(e, int)]
                )
                for msg_id in unknown_standard_can:
                    message.append(f"- 0x{msg_id:X}")

                for pgn, sa in unknown_j1939:
                    message.append(f"- PGN=0x{pgn:X} SA=0x{sa:X}")

                message.append("\n\n")

            self.output_info_bus.setPlainText("\n".join(message))

            self.open_new_file.emit(str(file_name))

    def extract_bus_csv_logging(self, event):
        version = self.extract_bus_format.currentText()

        self.output_info_bus.setPlainText("")

        database_files = {}

        count1 = self.can_database_list.count()
        if count1:
            database_files["CAN"] = []
            for i in range(count1):
                item = self.can_database_list.item(i)
                widget = self.can_database_list.itemWidget(item)
                database_files["CAN"].append(
                    (widget.database.text(), widget.bus.currentIndex())
                )

        count2 = self.lin_database_list.count()
        if count2:
            database_files["LIN"] = []
            for i in range(count2):
                item = self.lin_database_list.item(i)
                widget = self.lin_database_list.itemWidget(item)
                database_files["LIN"].append(
                    (widget.database.text(), widget.bus.currentIndex())
                )

        if not (count1 + count2):
            return

        single_time_base = self.single_time_base_bus.checkState() == QtCore.Qt.Checked
        time_from_zero = self.time_from_zero_bus.checkState() == QtCore.Qt.Checked
        empty_channels = self.empty_channels_bus.currentText()
        raster = self.export_raster_bus.value()
        time_as_date = self.bus_time_as_date.checkState() == QtCore.Qt.Checked
        delimiter = self.delimiter_bus.text() or ","
        doublequote = self.doublequote_bus.checkState() == QtCore.Qt.Checked
        escapechar = self.escapechar_bus.text() or None
        lineterminator = (
            self.lineterminator_bus.text().replace("\\r", "\r").replace("\\n", "\n")
        )
        quotechar = self.quotechar_bus.text() or '"'
        quoting = self.quoting_bus.currentText()
        add_units = self.add_units_bus.checkState() == QtCore.Qt.Checked

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
                title="Extract Bus logging to CSV",
                message=f'Extracting Bus signals from "{self.file_name}"',
                icon_name="csv",
            )

            # convert self.mdf
            target = self.mdf.extract_bus_logging
            kwargs = {
                "database_files": database_files,
                "version": version,
                "prefix": self.prefix.text().strip(),
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

            mdf.configure(
                integer_interpolation=self.mdf._integer_interpolation,
                float_interpolation=self.mdf._float_interpolation,
            )

            target = mdf.export
            kwargs = {
                "fmt": "csv",
                "filename": file_name,
                "single_time_base": single_time_base,
                "time_from_zero": time_from_zero,
                "empty_channels": empty_channels,
                "raster": raster or None,
                "time_as_date": time_as_date,
                "ignore_value2text_conversions": self.ignore_value2text_conversions,
                "delimiter": delimiter,
                "doublequote": doublequote,
                "escapechar": escapechar,
                "lineterminator": lineterminator,
                "quotechar": quotechar,
                "quoting": quoting,
                "add_units": add_units,
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

            bus_call_info = dict(self.mdf.last_call_info)

            message = []

            for bus, call_info in bus_call_info.items():

                found_id_count = sum(len(e) for e in call_info["found_ids"].values())

                message += [
                    f"{bus} bus summary:",
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
                    f"The following {bus} IDs were in the MDF log file and matched in the DBC:",
                ]
                for dbc_name, found_ids in call_info["found_ids"].items():
                    for msg_id, msg_name in sorted(found_ids):
                        message.append(f"- 0x{msg_id:X} --> {msg_name} in <{dbc_name}>")

                message += [
                    "",
                    f"The following {bus} IDs were in the MDF log file, but not matched in the DBC:",
                ]
                for msg_id in sorted(call_info["unknown_ids"]):
                    message.append(f"- 0x{msg_id:X}")
                message.append("\n\n")

            self.output_info_bus.setPlainText("\n".join(message))

    def load_can_database(self, event):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select CAN database file",
            "",
            "ARXML or DBC (*.dbc *.arxml)",
            "ARXML or DBC (*.dbc *.arxml)",
        )

        if file_names:
            for database in file_names:
                item = QtWidgets.QListWidgetItem()
                widget = DatabaseItem(database, bus_type="CAN")

                self.can_database_list.addItem(item)
                self.can_database_list.setItemWidget(item, widget)
                item.setSizeHint(widget.sizeHint())

    def load_lin_database(self, event):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select LIN database file",
            "",
            "ARXML or DBC database (*.dbc *.arxml);;LDF database (*.ldf);;All supported formats (*.dbc *.arxml *ldf)",
            "All supported formats (*.dbc *.arxml *ldf)",
        )

        if file_names:
            for database in file_names:
                item = QtWidgets.QListWidgetItem()
                widget = DatabaseItem(database, bus_type="LIN")

                self.lin_database_list.addItem(item)
                self.lin_database_list.setItemWidget(item, widget)
                item.setSizeHint(widget.sizeHint())

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if key == QtCore.Qt.Key_F and modifier == QtCore.Qt.ControlModifier:
            self.search()

        elif key == QtCore.Qt.Key_F11:
            self.full_screen_toggled.emit()

        elif (
            key in (QtCore.Qt.Key_V, QtCore.Qt.Key_H, QtCore.Qt.Key_C, QtCore.Qt.Key_T)
            and modifier == QtCore.Qt.ShiftModifier
        ):
            if key == QtCore.Qt.Key_V:
                mode = "tile vertically"
            elif key == QtCore.Qt.Key_H:
                mode = "tile horizontally"
            elif key == QtCore.Qt.Key_C:
                mode = "cascade"
            elif key == QtCore.Qt.Key_T:
                mode = "tile"

            if mode == "tile":
                self.mdi_area.tileSubWindows()
            elif mode == "cascade":
                self.mdi_area.cascadeSubWindows()
            elif mode == "tile vertically":
                self.mdi_area.tile_vertically()
            elif mode == "tile horizontally":
                self.mdi_area.tile_horizontally()

        elif key == QtCore.Qt.Key_F and modifier == (
            QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier
        ):
            self.toggle_frames()

        elif key == QtCore.Qt.Key_L and modifier == QtCore.Qt.ShiftModifier:
            if self.channel_view.isVisible():

                self._splitter_sizes = self.splitter.sizes()

                self.channel_view.hide()
                self.channels_tree.hide()

                self.splitter.setSizes([0, max(sum(self._splitter_sizes), 2)])
                self.splitter.setStretchFactor(0, 0)
                self.splitter.setStretchFactor(1, 1)
                self.splitter.handle(0).setEnabled(False)
                self.splitter.handle(1).setEnabled(False)

            else:
                self.channel_view.show()
                self.channels_tree.show()

                self.splitter.setStretchFactor(0, 0)
                self.splitter.setStretchFactor(1, 1)

                self.splitter.setSizes(self._splitter_sizes)
                self.splitter.handle(0).setEnabled(True)
                self.splitter.handle(1).setEnabled(True)

        elif key == QtCore.Qt.Key_Period and modifier == QtCore.Qt.NoModifier:
            self.set_line_style()

        else:
            widget = self.get_current_widget()
            if widget:
                widget.keyPressEvent(event)
            else:
                super().keyPressEvent(event)

    def aspect_changed(self, index):

        current_index = self.aspects.currentIndex()
        count = self.aspects.count()
        for i in range(count):
            widget = self.aspects.widget(i)
            if i == current_index:
                widget.show()
            else:
                widget.hide()

        if self.aspects.tabText(current_index) == "Modify && Export":

            if not self.raster_channel.count():
                self.raster_channel.setSizeAdjustPolicy(
                    QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
                )
                self.raster_channel.addItems(self.channels_db_items)
                self.raster_channel.setMinimumWidth(100)

            if not self._show_filter_tree:
                self._show_filter_tree = True

                widget = self.filter_tree

                widget.clear()

                if self.filter_view.currentText() == "Natural sort":
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

                elif self.filter_view.currentText() == "Internal file structure":
                    for i, group in enumerate(self.mdf.groups):
                        entry = i, 0xFFFFFFFFFFFFFFFF
                        channel_group = TreeItem(entry)
                        comment = extract_xml_comment(group.channel_group.comment)

                        if self.display_cg_name:
                            base_name = f"CG {i} {group.channel_group.acq_name}"
                        else:
                            base_name = f"Channel group {i}"

                        if comment:
                            name = base_name + f" ({comment})"
                        else:
                            name = base_name

                        channel_group.setText(0, name)

                        channel_group.setFlags(
                            channel_group.flags()
                            | QtCore.Qt.ItemIsAutoTristate
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
                            version=self.mdf.version,
                        )

            for w in self.mdi_area.subWindowList():
                widget = w.widget()
                if isinstance(widget, Plot):
                    if widget.plot.region is not None:
                        start, stop = widget.plot.region.getRegion()
                        self.cut_start.setValue(start)
                        self.cut_stop.setValue(stop)
                        break

        elif self.aspects.tabText(current_index) == "Info":
            self.info.clear()
            # self.mdf.reload_header()
            # info tab
            try:
                file_stats = os.stat(self.mdf.name)
            except:
                file_stats = None
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
            if file_stats is not None:
                item.setText(1, f"{file_stats.st_size / 1024 / 1024:.1f} MB")
            else:
                try:
                    item.setText(1, f"{self.mdf.file_limit / 1024 / 1024:.1f} MB")
                except:
                    item.setText(1, f"Unknown size")
            children.append(item)

            if file_stats is not None:
                date_ = datetime.fromtimestamp(file_stats.st_ctime)
            else:
                date_ = datetime.now(timezone.utc)
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, "Created")
            item.setText(1, date_.strftime("%d-%b-%Y %H-%M-%S"))
            children.append(item)

            if file_stats is not None:
                date_ = datetime.fromtimestamp(file_stats.st_mtime)
            else:
                date_ = datetime.now(timezone.utc)
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
            item.setText(1, self.mdf.header.start_time_string())
            children.append(item)

            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, "Measurement comment")
            item.setText(1, self.mdf.header.description)
            item.setTextAlignment(0, QtCore.Qt.AlignTop)
            children.append(item)

            mesaurement_attributes = QtWidgets.QTreeWidgetItem()
            mesaurement_attributes.setText(0, "Measurement attributes")
            children.append(mesaurement_attributes)

            for name, value in self.mdf.header._common_properties.items():
                if isinstance(value, dict):
                    tree = QtWidgets.QTreeWidgetItem()
                    tree.setText(0, name)
                    tree.setTextAlignment(0, QtCore.Qt.AlignTop)

                    for subname, subvalue in value.items():
                        item = QtWidgets.QTreeWidgetItem()
                        item.setText(0, subname)
                        item.setText(1, str(subvalue).strip())
                        item.setTextAlignment(0, QtCore.Qt.AlignTop)

                        tree.addChild(item)

                    mesaurement_attributes.addChild(tree)

                else:
                    item = QtWidgets.QTreeWidgetItem()
                    item.setText(0, FRIENDLY_ATRRIBUTES.get(name, name))
                    item.setText(1, str(value).strip())
                    item.setTextAlignment(0, QtCore.Qt.AlignTop)
                    mesaurement_attributes.addChild(item)

            channel_groups = QtWidgets.QTreeWidgetItem()
            channel_groups.setText(0, "Channel groups")
            channel_groups.setText(1, str(len(self.mdf.groups)))
            children.append(channel_groups)

            channel_groups_children = []
            for i, group in enumerate(self.mdf.groups):
                channel_group = group.channel_group
                if hasattr(channel_group, "comment"):
                    comment = extract_xml_comment(channel_group.comment)
                else:
                    comment = ""

                if self.display_cg_name:
                    base_name = f"CG {i} {channel_group.acq_name}"
                else:
                    base_name = f"Channel group {i}"
                if comment:
                    name = base_name + f" ({comment})"
                else:
                    name = base_name

                cycles = channel_group.cycles_nr

                channel_group_item = QtWidgets.QTreeWidgetItem()
                channel_group_item.setText(0, name)

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

                    if group.channel_group.acq_source:
                        source = group.channel_group.acq_source
                        if source.bus_type == BUS_TYPE_CAN:
                            ico = ":/bus_can.png"
                        elif source.bus_type == BUS_TYPE_LIN:
                            ico = ":/bus_lin.png"
                        elif source.bus_type == BUS_TYPE_ETHERNET:
                            ico = ":/bus_eth.png"
                        elif source.bus_type == BUS_TYPE_USB:
                            ico = ":/bus_usb.png"
                        elif source.bus_type == BUS_TYPE_FLEXRAY:
                            ico = ":/bus_flx.png"
                        else:
                            ico = None

                        if ico is not None:
                            icon = QtGui.QIcon()
                            icon.addPixmap(
                                QtGui.QPixmap(ico), QtGui.QIcon.Normal, QtGui.QIcon.Off
                            )
                            channel_group_item.setIcon(0, icon)

                item = QtWidgets.QTreeWidgetItem()
                item.setText(0, "Channels")
                item.setText(1, f"{len(group.channels)}")
                channel_group_item.addChild(item)

                item = QtWidgets.QTreeWidgetItem()
                item.setText(0, "Cycles")
                item.setText(1, str(cycles))
                if cycles:
                    item.setForeground(1, QtGui.QBrush(QtCore.Qt.darkGreen))
                channel_group_item.addChild(item)

                if size <= 1 << 10:
                    text = f"{size} B"
                elif size <= 1 << 20:
                    text = f"{size / 1024:.1f} KB"
                elif size <= 1 << 30:
                    text = f"{size / 1024 / 1024:.1f} MB"
                else:
                    text = f"{size / 1024 / 1024 / 1024:.1f} GB"

                item = QtWidgets.QTreeWidgetItem()
                item.setText(0, "Raw size")
                item.setText(1, text)
                if cycles:
                    item.setForeground(1, QtGui.QBrush(QtCore.Qt.darkGreen))
                channel_group_item.addChild(item)

                channel_groups_children.append(channel_group_item)

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

    def toggle_frames(self, event=None):
        self._frameless_windows = not self._frameless_windows

        for w in self.mdi_area.subWindowList():
            if self._frameless_windows:
                w.setWindowFlags(w.windowFlags() | QtCore.Qt.FramelessWindowHint)
            else:
                w.setWindowFlags(w.windowFlags() & (~QtCore.Qt.FramelessWindowHint))

    def autofit_sub_plots(self):
        geometries = []
        for window in self.mdi_area.subWindowList():
            geometry = window.geometry()
            geometries.append(geometry)

        if len(set((g.width(), g.x()) for g in geometries)) == 1:
            self.mdi_area.tile_vertically()
        elif len(set((g.height(), g.y()) for g in geometries)) == 1:
            self.mdi_area.tile_horizontally()
        else:
            self.mdi_area.tileSubWindows()

    def _current_options(self):
        options = {
            "needs_cut": self.cut_group.isChecked(),
            "cut_start": self.cut_start.value(),
            "cut_stop": self.cut_stop.value(),
            "cut_time_from_zero": self.cut_time_from_zero.checkState()
            == QtCore.Qt.Checked,
            "whence": int(self.whence.checkState() == QtCore.Qt.Checked),
            "needs_resample": self.resample_group.isChecked(),
            "raster_type_step": self.raster_type_step.isChecked(),
            "raster_type_channel": self.raster_type_channel.isChecked(),
            "raster": self.raster.value(),
            "raster_channel": self.raster_channel.currentText(),
            "resample_time_from_zero": self.resample_time_from_zero.checkState()
            == QtCore.Qt.Checked,
            "output_format": self.output_format.currentText(),
        }

        output_format = self.output_format.currentText()

        if output_format == "MDF":

            new = {
                "mdf_version": self.mdf_version.currentText(),
                "mdf_compression": self.mdf_compression.currentIndex(),
                "mdf_split": self.mdf_split.checkState() == QtCore.Qt.Checked,
                "mdf_split_size": self.mdf_split_size.value() * 1024 * 1024,
            }

        elif output_format == "MAT":

            new = {
                "single_time_base": self.single_time_base_mat.checkState()
                == QtCore.Qt.Checked,
                "time_from_zero": self.time_from_zero_mat.checkState()
                == QtCore.Qt.Checked,
                "time_as_date": self.time_as_date_mat.checkState() == QtCore.Qt.Checked,
                "use_display_names": self.use_display_names_mat.checkState()
                == QtCore.Qt.Checked,
                "reduce_memory_usage": self.reduce_memory_usage_mat.checkState()
                == QtCore.Qt.Checked,
                "compression": self.export_compression_mat.currentText() == "enabled",
                "empty_channels": self.empty_channels_mat.currentText(),
                "mat_format": self.mat_format.currentText(),
                "oned_as": self.oned_as.currentText(),
                "raw": self.raw_mat.checkState() == QtCore.Qt.Checked,
            }

        elif output_format == "CSV":

            new = {
                "single_time_base": self.single_time_base_csv.checkState()
                == QtCore.Qt.Checked,
                "time_from_zero": self.time_from_zero_csv.checkState()
                == QtCore.Qt.Checked,
                "time_as_date": self.time_as_date_csv.checkState() == QtCore.Qt.Checked,
                "use_display_names": self.use_display_names_csv.checkState()
                == QtCore.Qt.Checked,
                "reduce_memory_usage": False,
                "compression": False,
                "empty_channels": self.empty_channels_csv.currentText(),
                "raw": self.raw_csv.checkState() == QtCore.Qt.Checked,
                "delimiter": self.delimiter.text() or ",",
                "doublequote": self.doublequote.checkState() == QtCore.Qt.Checked,
                "escapechar": self.escapechar.text() or None,
                "lineterminator": self.lineterminator.text()
                .replace("\\r", "\r")
                .replace("\\n", "\n"),
                "quotechar": self.quotechar.text() or '"',
                "quoting": self.quoting.currentText(),
                "mat_format": None,
                "oned_as": None,
            }

        else:

            new = {
                "single_time_base": self.single_time_base.checkState()
                == QtCore.Qt.Checked,
                "time_from_zero": self.time_from_zero.checkState() == QtCore.Qt.Checked,
                "time_as_date": self.time_as_date.checkState() == QtCore.Qt.Checked,
                "use_display_names": self.use_display_names.checkState()
                == QtCore.Qt.Checked,
                "reduce_memory_usage": self.reduce_memory_usage.checkState()
                == QtCore.Qt.Checked,
                "compression": self.export_compression.currentText(),
                "empty_channels": self.empty_channels.currentText(),
                "mat_format": None,
                "oned_as": None,
                "raw": self.raw.checkState() == QtCore.Qt.Checked,
            }

        options.update(new)

        class Options:
            def __init__(self, opts):
                self._opts = opts
                for k, v in opts.items():
                    setattr(self, k, v)

        return Options(options)

    def _get_filtered_channels(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

        channels = []
        count = 0
        total = 0

        if self.filter_view.currentText() == "Internal file structure":
            while iterator.value():
                item = iterator.value()

                group, index = item.entry
                if index != 0xFFFFFFFFFFFFFFFF:
                    total += 1

                if item.checkState(0) == QtCore.Qt.Checked:

                    if index != 0xFFFFFFFFFFFFFFFF:
                        channels.append((None, group, index))
                        count += 1

                iterator += 1
        else:
            while iterator.value():
                item = iterator.value()

                if item.checkState(0) == QtCore.Qt.Checked:
                    group, index = item.entry
                    channels.append((None, group, index))
                    count += 1

                total += 1

                iterator += 1

        if not channels:
            return False, channels
        else:
            if total == count:
                return False, channels
            else:
                return True, channels

    def apply_processing(self, event):

        steps = 1
        if self.cut_group.isChecked():
            steps += 1
        if self.resample_group.isChecked():
            steps += 1
        needs_filter, channels = self._get_filtered_channels()
        if needs_filter:
            steps += 1

        opts = self._current_options()

        output_format = opts.output_format

        if output_format == "MDF":
            version = opts.mdf_version

            if version < "4.00":
                filter = "MDF version 3 files (*.dat *.mdf)"
                default = filter
            else:
                filter = (
                    "MDF version 4 files (*.mf4);;Zipped MDF version 4 files (*.mf4z)"
                )
                if Path(self.mdf.original_name).suffix.lower() == ".mf4z":
                    default = "Zipped MDF version 4 files (*.mf4z)"
                else:
                    default = "MDF version 4 files (*.mf4)"

            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select output measurement file",
                "",
                f"{filter};;All files (*.*)",
                default,
            )

        else:
            filters = {
                "CSV": "CSV files (*.csv)",
                "HDF5": "HDF5 files (*.hdf)",
                "MAT": "Matlab MAT files (*.mat)",
                "Parquet": "Apache Parquet files (*.parquet)",
            }

            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select export file",
                "",
                f"{filters[output_format]};;All files (*.*)",
                filters[output_format],
            )

        if file_name:
            if output_format == "HDF5":
                try:
                    from h5py import File as HDF5
                except ImportError:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Export to HDF5 unavailale",
                        "h5py package not found; export to HDF5 is unavailable",
                    )
                    return

            elif output_format == "MAT":
                if opts.mat_format == "7.3":
                    try:
                        from hdf5storage import savemat
                    except ImportError:
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Export to mat v7.3 unavailale",
                            "hdf5storage package not found; export to mat 7.3 is unavailable",
                        )
                        return
                else:
                    try:
                        from scipy.io import savemat
                    except ImportError:
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Export to mat v4 and v5 unavailale",
                            "scipy package not found; export to mat is unavailable",
                        )
                        return

            elif output_format == "Parquet":
                try:
                    from fastparquet import write as write_parquet
                except ImportError:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Export to parquet unavailale",
                        "fastparquet package not found; export to parquet is unavailable",
                    )
                    return
        else:
            return

        split_size = opts.mdf_split_size if output_format == "MDF" else 0
        self.mdf.configure(read_fragment_size=split_size)

        mdf = None
        progress = None
        integer_interpolation = self.mdf._integer_interpolation
        float_interpolation = self.mdf._float_interpolation

        if needs_filter:

            progress = setup_progress(
                parent=self,
                title="Filtering measurement",
                message=f'Filtering selected channels from "{self.file_name}"',
                icon_name="filter",
            )

            # filtering self.mdf
            target = self.mdf.filter
            kwargs = {
                "channels": channels,
                "version": opts.mdf_version if output_format == "MDF" else "4.10",
            }

            result = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=99,
                offset=0,
                progress=progress,
            )

            if result is TERMINATED:
                progress.cancel()
                return
            else:
                mdf = result

            mdf.configure(
                read_fragment_size=split_size,
                write_fragment_size=split_size,
                integer_interpolation=integer_interpolation,
                float_interpolation=float_interpolation,
            )

        if opts.needs_cut:

            if progress is None:
                progress = setup_progress(
                    parent=self,
                    title="Cutting measurement",
                    message=f"Cutting from {opts.cut_start}s to {opts.cut_stop}s",
                    icon_name="cut",
                )
            else:
                icon = QtGui.QIcon()
                icon.addPixmap(
                    QtGui.QPixmap(":/cut.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                )
                progress.setWindowIcon(icon)
                progress.setWindowTitle("Cutting measurement")
                progress.setLabelText(
                    f"Cutting from {opts.cut_start}s to {opts.cut_stop}s"
                )

            # cut self.mdf
            target = self.mdf.cut if mdf is None else mdf.cut
            kwargs = {
                "start": opts.cut_start,
                "stop": opts.cut_stop,
                "whence": opts.whence,
                "version": opts.mdf_version if output_format == "MDF" else "4.10",
                "time_from_zero": opts.cut_time_from_zero,
            }

            result = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=99,
                offset=0,
                progress=progress,
            )

            if result is TERMINATED:
                progress.cancel()
                return
            else:
                if mdf is None:
                    mdf = result
                else:
                    mdf.close()
                    mdf = result

            mdf.configure(
                read_fragment_size=split_size,
                write_fragment_size=split_size,
                integer_interpolation=integer_interpolation,
                float_interpolation=float_interpolation,
            )

        if opts.needs_resample:

            if opts.raster_type_channel:
                raster = opts.raster_channel
                message = f'Resampling using channel "{raster}"'
            else:
                raster = opts.raster
                message = f"Resampling to {raster}s raster"

            if progress is None:
                progress = setup_progress(
                    parent=self,
                    title="Resampling measurement",
                    message=message,
                    icon_name="resample",
                )
            else:
                icon = QtGui.QIcon()
                icon.addPixmap(
                    QtGui.QPixmap(":/resample.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                )
                progress.setWindowIcon(icon)
                progress.setWindowTitle("Resampling measurement")
                progress.setLabelText(message)

            # resample self.mdf
            target = self.mdf.resample if mdf is None else mdf.resample
            kwargs = {
                "raster": raster,
                "version": opts.mdf_version if output_format == "MDF" else "4.10",
                "time_from_zero": opts.resample_time_from_zero,
            }

            result = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=99,
                offset=0,
                progress=progress,
            )

            if result is TERMINATED:
                progress.cancel()
                return
            else:
                if mdf is None:
                    mdf = result
                else:
                    mdf.close()
                    mdf = result

            mdf.configure(
                read_fragment_size=split_size,
                write_fragment_size=split_size,
                integer_interpolation=integer_interpolation,
                float_interpolation=float_interpolation,
            )

        if output_format == "MDF":
            if mdf is None:
                if progress is None:
                    progress = setup_progress(
                        parent=self,
                        title="Converting measurement",
                        message=f'Converting "{self.file_name}" from {self.mdf.version} to {version}',
                        icon_name="convert",
                    )
                else:
                    icon = QtGui.QIcon()
                    icon.addPixmap(
                        QtGui.QPixmap(":/convert.png"),
                        QtGui.QIcon.Normal,
                        QtGui.QIcon.Off,
                    )
                    progress.setWindowIcon(icon)
                    progress.setWindowTitle("Converting measurement")
                    progress.setLabelText(
                        f'Converting "{self.file_name}" from {self.mdf.version} to {version}'
                    )

                # convert self.mdf
                target = self.mdf.convert
                kwargs = {"version": version}

                result = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=99,
                    offset=0,
                    progress=progress,
                )

                if result is TERMINATED:
                    progress.cancel()
                    return
                else:
                    mdf = result

            mdf.configure(
                read_fragment_size=split_size,
                write_fragment_size=split_size,
                integer_interpolation=integer_interpolation,
                float_interpolation=float_interpolation,
            )

            # then save it
            progress.setLabelText(f'Saving output file "{file_name}"')

            handle_overwrite = Path(file_name) == self.mdf.name

            if handle_overwrite:
                dspf = self.to_config()

                _password = self.mdf._password
                self.mdf.close()

                windows = list(self.mdi_area.subWindowList())
                for window in windows:
                    widget = window.widget()
                    self.mdi_area.removeSubWindow(window)
                    widget.setParent(None)
                    widget.close()
                    window.close()

            target = mdf.save
            kwargs = {
                "dst": file_name,
                "compression": opts.mdf_compression,
                "overwrite": True,
            }

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=99,
                offset=0,
                progress=progress,
            )

            self.progress = None
            progress.cancel()

            if handle_overwrite:

                original_name = file_name

                target = MDF
                kwargs = {
                    "name": file_name,
                    "callback": self.update_progress,
                    "password": _password,
                    "use_display_names": True,
                }

                self.mdf = MDF(**kwargs)

                self.mdf.original_name = original_name
                self.mdf.uuid = self.uuid

                self.aspects.setCurrentIndex(0)
                self.load_channel_list(file_name=dspf)

                self.aspects.setCurrentIndex(1)

        else:
            if progress is None:
                progress = setup_progress(
                    parent=self,
                    title="Export measurement",
                    message=f"Exporting to {output_format}",
                    icon_name="export",
                )
            else:
                icon = QtGui.QIcon()
                icon.addPixmap(
                    QtGui.QPixmap(":/export.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                )
                progress.setWindowIcon(icon)
                progress.setWindowTitle("Export measurement")
                progress.setLabelText(
                    f"Exporting to {output_format} (be patient this might take a while)"
                )

            delimiter = self.delimiter.text() or ","
            doublequote = self.doublequote.checkState() == QtCore.Qt.Checked
            escapechar = self.escapechar.text() or None
            lineterminator = (
                self.lineterminator.text().replace("\\r", "\r").replace("\\n", "\n")
            )
            quotechar = self.quotechar.text() or '"'
            quoting = self.quoting.currentText()
            add_units = self.add_units.checkState() == QtCore.Qt.Checked

            target = self.mdf.export if mdf is None else mdf.export
            kwargs = {
                "fmt": opts.output_format.lower(),
                "filename": file_name,
                "single_time_base": opts.single_time_base,
                "use_display_names": opts.use_display_names,
                "time_from_zero": opts.time_from_zero,
                "empty_channels": opts.empty_channels,
                "format": opts.mat_format,
                "raster": None,
                "oned_as": opts.oned_as,
                "reduce_memory_usage": opts.reduce_memory_usage,
                "compression": opts.compression,
                "time_as_date": opts.time_as_date,
                "ignore_value2text_conversions": self.ignore_value2text_conversions,
                "raw": opts.raw,
                "delimiter": delimiter,
                "doublequote": doublequote,
                "escapechar": escapechar,
                "lineterminator": lineterminator,
                "quotechar": quotechar,
                "quoting": quoting,
                "add_units": add_units,
            }

            result = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=99,
                offset=0,
                progress=progress,
            )

            self.progress = None
            progress.cancel()

    def raster_search(self, event):
        dlg = AdvancedSearch(
            self.mdf,
            show_add_window=False,
            show_apply=True,
            apply_text="Set raster channel",
            show_pattern=False,
            return_names=True,
            parent=self,
        )
        dlg.setModal(True)
        dlg.exec_()
        result = dlg.result
        if result:
            name = list(result)[0]
            self.raster_channel.setCurrentText(name)

    def filter_changed(self, item, column):
        name = item.text(0)
        if item.checkState(0) == QtCore.Qt.Checked:
            self._selected_filter.add(name)
        else:
            if name in self._selected_filter:
                self._selected_filter.remove(name)
        self._filter_timer.start(10)

    def update_selected_filter_channels(self):
        self.selected_filter_channels.clear()
        self.selected_filter_channels.addItems(sorted(self._selected_filter))

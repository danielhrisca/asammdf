from datetime import datetime, timezone
from functools import partial
from hashlib import sha1
import json
import os
from pathlib import Path
import re
from tempfile import gettempdir
from traceback import format_exc
from zipfile import ZIP_DEFLATED, ZipFile

from natsort import natsorted
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

import asammdf.mdf as mdf_module

from ... import tool
from ...blocks.utils import (
    extract_encryption_information,
    extract_xml_comment,
    load_channel_names_from_file,
    load_dsp,
    load_lab,
    TERMINATED,
)
from ...blocks.v4_blocks import AttachmentBlock, FileHistory, HeaderBlock
from ...blocks.v4_blocks import TextBlock as TextV4
from ...blocks.v4_constants import (
    BUS_TYPE_CAN,
    BUS_TYPE_ETHERNET,
    BUS_TYPE_FLEXRAY,
    BUS_TYPE_LIN,
    BUS_TYPE_USB,
    FLAG_AT_TO_STRING,
    FLAG_CG_BUS_EVENT,
)
from ..dialogs.advanced_search import AdvancedSearch
from ..dialogs.channel_group_info import ChannelGroupInfoDialog
from ..dialogs.channel_info import ChannelInfoDialog
from ..dialogs.error_dialog import ErrorDialog
from ..dialogs.gps_dialog import GPSDialog
from ..dialogs.messagebox import MessageBox
from ..dialogs.window_selection_dialog import WindowSelectionDialog
from ..ui.file_widget import Ui_file_widget
from ..utils import GREEN, HelperChannel, run_thread_with_progress, setup_progress
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
from .tree_item import MinimalTreeItem


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
    "pr_display_file": "Default display file",
}


class Delegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        self.editor = QtWidgets.QPlainTextEdit(parent)
        self.editor.setReadOnly(True)
        return self.editor

    def setEditorData(self, editor, index):
        if editor:
            editor.setPlainText(index.data())

    def setModelData(self, editor, model, index):
        return


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
        password=None,
        hide_missing_channels=False,
        hide_disabled_channels=False,
        *args,
        **kwargs,
    ):
        self.default_folder = kwargs.pop("default_folder", "")
        display_file = kwargs.pop("display_file", "")
        database_file = kwargs.pop("database_file", None)
        show_progress = kwargs.pop("show_progress", True)
        process_bus_logging = kwargs.pop("process_bus_logging", True)

        self._progress = None

        self.loaded_display_file = Path(""), b""

        super(Ui_file_widget, self).__init__(*args, **kwargs)
        WithMDIArea.__init__(self, comparison=False)
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

        self.unknown_windows = []

        self._show_filter_tree = False
        self.line_interconnect = line_interconnect
        if show_progress:
            progress = QtWidgets.QProgressDialog(f'Opening "{self.file_name}"', "", 0, 100, self.parent())

            progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
            progress.setCancelButton(None)
            progress.setAutoClose(True)
            progress.setWindowTitle("Opening measurement")
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/open.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            progress.setWindowIcon(icon)
            progress.setMinimumWidth(600)
            progress.show()
        else:
            progress = None

        try:
            if file_name.suffix.lower() in (".asc", ".blf", ".erg", ".bsig", ".dl3", ".tdms"):
                extension = file_name.suffix.lower().strip(".")
                if progress:
                    progress.setLabelText(f"Converting from {extension} to mdf")

                try:
                    from mfile import ASC, BLF, BSIG, DL3, ERG, TDMS
                except ImportError:
                    from cmerg import BSIG, ERG

                if file_name.suffix.lower() == ".erg":
                    cls = ERG
                elif file_name.suffix.lower() == ".bsig":
                    cls = BSIG
                elif file_name.suffix.lower() == ".tdms":
                    cls = TDMS
                elif file_name.suffix.lower() == ".asc":
                    cls = ASC
                elif file_name.suffix.lower() == ".blf":
                    cls = BLF
                else:
                    cls = DL3

                out_file = Path(gettempdir()) / file_name.name
                if file_name.suffix.lower() in (".asc", ".blf"):
                    meas_file = cls(file_name, database=database_file)
                else:
                    meas_file = cls(file_name)

                mdf_path = meas_file.export_mdf().save(out_file.with_suffix(".tmp.mf4"))
                meas_file.close()
                self.mdf = mdf_module.MDF(mdf_path, process_bus_logging=process_bus_logging)
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
                            units = dict(zip(names, units))
                        else:
                            csv.seek(0)
                            csv.readline()
                            units = None

                        df = pd.read_csv(csv, header=None, names=names)
                        df.set_index(df[names[0]], inplace=True)
                        self.mdf = mdf_module.MDF()
                        self.mdf.append(df, units=units)
                        self.mdf.uuid = self.uuid
                        self.mdf.original_name = file_name
                except Exception as exc:
                    if progress:
                        progress.cancel()
                    print(format_exc())
                    raise Exception(
                        "Could not load CSV. The first line must contain the channel names. The seconds line "
                        "can optionally contain the channel units. The first column must be the time"
                    ) from exc

            else:
                original_name = file_name

                target = mdf_module.MDF
                kwargs = {
                    "name": file_name,
                    "callback": self.update_progress,
                    "password": password,
                    "use_display_names": True,
                    "process_bus_logging": process_bus_logging,
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

            if progress:
                progress.setLabelText("Loading graphical elements")
            QtWidgets.QApplication.processEvents()

            if progress:
                progress.setValue(37)

            self.channels_tree.setDragEnabled(True)

            self.mdi_area = MdiAreaWidget()

            self.mdi_area.add_window_request.connect(self.add_window)
            self.mdi_area.open_file_request.connect(self.open_new_file.emit)
            self.mdi_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.mdi_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.splitter.addWidget(self.mdi_area)

            self.channels_tree.itemDoubleClicked.connect(self.show_info)
            self.filter_tree.itemDoubleClicked.connect(self.show_info)

            self.channel_view.setCurrentIndex(-1)
            self.filter_view.setCurrentIndex(-1)

            self.filter_view.setCurrentText(self._settings.value("filter_view", "Internal file structure"))

            self.channel_view.currentIndexChanged.connect(partial(self._update_channel_tree, widget=self.channels_tree))
            self.filter_view.currentIndexChanged.connect(partial(self._update_channel_tree, widget=self.filter_tree))

            self.channel_view.setCurrentText(self._settings.value("channels_view", "Internal file structure"))

            if progress:
                progress.setValue(70)
            QtWidgets.QApplication.processEvents()

            self.raster_type_channel.toggled.connect(self.set_raster_type)
            if progress:
                progress.setValue(90)
            QtWidgets.QApplication.processEvents()

            self.output_options.setCurrentIndex(0)

            self.mdf_version.insertItems(0, mdf_module.SUPPORTED_VERSIONS)
            self.mdf_version.setCurrentText("4.10")
            self.mdf_compression.insertItems(0, ("no compression", "deflate", "transposed deflate"))
            self.mdf_compression.setCurrentText("transposed deflate")
            self.mdf_split_size.setValue(4)

            self.extract_bus_format.insertItems(0, mdf_module.SUPPORTED_VERSIONS)
            self.extract_bus_format.setCurrentText("4.10")
            index = self.extract_bus_format.findText(self.mdf.version)
            if index >= 0:
                self.extract_bus_format.setCurrentIndex(index)
            self.extract_bus_compression.insertItems(0, ("no compression", "deflate", "transposed deflate"))
            self.extract_bus_compression.setCurrentText("transposed deflate")
            self.extract_bus_btn.clicked.connect(self.extract_bus_logging)
            self.extract_bus_csv_btn.clicked.connect(self.extract_bus_csv_logging)
            self.load_can_database_btn.clicked.connect(self.load_can_database)
            self.load_lin_database_btn.clicked.connect(self.load_lin_database)

            if progress:
                progress.setValue(99)

            self.empty_channels.insertItems(0, ("skip", "zeros"))
            self.empty_channels_bus.insertItems(0, ("skip", "zeros"))
            self.empty_channels_mat.insertItems(0, ("skip", "zeros"))
            self.empty_channels_csv.insertItems(0, ("skip", "zeros"))
            try:
                import scipy  # noqa: F401

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
            if progress:
                progress.setValue(100)
                progress.deleteLater()
            raise

        else:
            if progress:
                progress.setValue(100)
                progress.deleteLater()

        self.load_channel_list_btn.clicked.connect(partial(self.load_channel_list, manually=True))
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

        hide_embedded_btn = True

        if self.mdf.version >= "4.00" and self.mdf.attachments:
            for i, attachment in enumerate(self.mdf.attachments, 1):
                if attachment.file_name == "user_embedded_display.dspf" and attachment.mime == r"application/x-dspf":
                    hide_embedded_btn = False

                att = Attachment(i - 1, self)
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
            self.aspects.setTabVisible(4, False)

        if hide_embedded_btn:
            self.load_embedded_channel_list_btn.setDisabled(True)
        self.load_embedded_channel_list_btn.clicked.connect(self.load_embedded_display_file)
        self.save_embedded_channel_list_btn.clicked.connect(self.embed_display_file)

        if self.mdf.version >= "4.00":
            if self.mdf.original_name.suffix.lower() not in (".mf4", ".mf4z"):
                self.save_embedded_channel_list_btn.setEnabled(False)
        else:
            self.load_embedded_channel_list_btn.setEnabled(False)
            self.save_embedded_channel_list_btn.setEnabled(False)

        if self.mdf.version >= "4.00":
            if not any(group.channel_group.flags & FLAG_CG_BUS_EVENT for group in self.mdf.groups):
                self.aspects.setTabVisible(2, False)
        else:
            self.aspects.setTabVisible(2, False)

        databases = {}

        can_databases = self._settings.value("can_databases", [])
        buses = can_databases[::2]
        dbs = can_databases[1::2]

        databases["CAN"] = list(zip(buses, dbs))

        lin_databases = self._settings.value("lin_databases", [])
        buses = lin_databases[::2]
        dbs = lin_databases[1::2]

        databases["LIN"] = list(zip(buses, dbs))

        for bus, database in databases["CAN"]:
            item = QtWidgets.QListWidgetItem()
            widget = DatabaseItem(database, bus_type="CAN")
            widget.bus.setCurrentText(bus)

            self.can_database_list.addItem(item)
            self.can_database_list.setItemWidget(item, widget)
            item.setSizeHint(widget.sizeHint())

        for bus, database in databases["LIN"]:
            item = QtWidgets.QListWidgetItem()
            widget = DatabaseItem(database, bus_type="LIN")
            widget.bus.setCurrentText(bus)

            self.lin_database_list.addItem(item)
            self.lin_database_list.setItemWidget(item, widget)
            item.setSizeHint(widget.sizeHint())

        self._splitter_sizes = None

        if display_file:
            self.load_channel_list(file_name=display_file)
        else:
            default_display_file = self.mdf.header._common_properties.get("pr_display_file", "")

            if default_display_file:
                default_display_file = Path(default_display_file)
                if default_display_file.exists():
                    self.load_channel_list(file_name=default_display_file)
                else:
                    default_display_file = Path(self.mdf.original_name).parent / default_display_file.name
                    if default_display_file.exists():
                        self.load_channel_list(file_name=default_display_file)

        self.restore_export_setttings()
        self.connect_export_updates()

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

    def _update_channel_tree(self, index=None, widget=None, force=False):
        if widget is None:
            widget = self.channels_tree
        if not force:
            if widget is self.channels_tree and self.channel_view.currentIndex() == -1:
                return
            elif widget is self.filter_tree and (self.filter_view.currentIndex() == -1):
                return

        view = self.channel_view if widget is self.channels_tree else self.filter_view

        iterator = QtWidgets.QTreeWidgetItemIterator(widget)
        signals = set()

        if widget.mode == "Internal file structure":
            while item := iterator.value():

                if item.entry[1] != 0xFFFFFFFFFFFFFFFF:
                    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                        signals.add(item.entry)

                iterator += 1
        else:
            while item := iterator.value():

                if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                    signals.add(item.entry)

                iterator += 1

        widget.collapseAll()
        widget.clear()
        widget.mode = view.currentText()

        if widget.mode == "Natural sort":
            items = []
            for i, group in enumerate(self.mdf.groups):
                for j, ch in enumerate(group.channels):
                    entry = i, j

                    channel = MinimalTreeItem(entry, ch.name, strings=[ch.name], origin_uuid=self.uuid)
                    channel.setToolTip(0, f"{ch.name} @ group {i}, index {j}")

                    if entry in signals:
                        channel.setCheckState(0, QtCore.Qt.CheckState.Checked)
                    else:
                        channel.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

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

                channel_group = MinimalTreeItem(entry, origin_uuid=self.uuid)

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
                        icon.addPixmap(QtGui.QPixmap(ico), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

                        channel_group.setIcon(0, icon)

                if self.display_cg_name:
                    acq_name = getattr(group.channel_group, "acq_name", "")
                    if acq_name:
                        base_name = f"CG {i} {acq_name}"
                    else:
                        base_name = f"CG {i}"
                    if comment and acq_name != comment:
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
                    | QtCore.Qt.ItemFlag.ItemIsAutoTristate
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                )

                if group.channel_group.cycles_nr:
                    channel_group.setForeground(0, QtGui.QBrush(QtGui.QColor(GREEN)))
                items.append(channel_group)

                channels = [HelperChannel(name=ch.name, entry=(i, j)) for j, ch in enumerate(group.channels)]

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
                channel = MinimalTreeItem(entry, ch.name, strings=[ch.name], origin_uuid=self.uuid)
                channel.setToolTip(0, f"{ch.name} @ group {gp_index}, index {ch_index}")
                channel.setCheckState(0, QtCore.Qt.CheckState.Checked)
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
        elif name == "ASC":
            self.output_options.setCurrentIndex(4)

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
                            },
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

                    while item := iterator.value():
                        if item.parent() is None:
                            iterator += 1
                            dg_cntr += 1
                            ch_cntr = 0
                            continue

                        entry = (dg_cntr, ch_cntr)

                        if entry in result:
                            item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                            names.add((result[entry], dg_cntr, ch_cntr))

                        iterator += 1
                        ch_cntr += 1

                elif view.currentText() == "Selected channels only":
                    iterator = QtWidgets.QTreeWidgetItemIterator(widget)

                    signals = set()
                    while item := iterator.value():

                        if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                            signals.add((item.text(0), *item.entry))

                        iterator += 1

                    names = {(_name, *entry) for entry, _name in result.items()}

                    signals = signals | names

                    widget.clear()
                    self._selected_filter = {e[0] for e in signals}

                    items = []
                    for name, gp_index, ch_index in signals:
                        entry = gp_index, ch_index
                        ch = self.mdf.groups[gp_index].channels[ch_index]
                        channel = MinimalTreeItem(entry, ch.name, strings=[ch.name], origin_uuid=self.uuid)
                        channel.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        items.append(channel)

                    if len(items) < 30000:
                        items = natsorted(items, key=lambda x: x.name)
                    else:
                        items.sort(key=lambda x: x.name)
                    widget.addTopLevelItems(items)

                    self.update_selected_filter_channels()

                else:
                    iterator = QtWidgets.QTreeWidgetItemIterator(widget)
                    while item := iterator.value():

                        if item.entry in result:
                            item.setCheckState(0, QtCore.Qt.CheckState.Checked)
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
                        if not isinstance(
                            mdi.widget(),
                            (CANBusTrace, LINBusTrace, FlexRayBusTrace, GPSDialog),
                        )
                    ]

                    if active_window := self.mdi_area.activeSubWindow():
                        default = active_window.windowTitle()
                    else:
                        default = None

                    dialog = WindowSelectionDialog(options=options, default=default, parent=self)
                    dialog.setModal(True)
                    dialog.exec_()

                    if dialog.result():
                        window_type = dialog.selected_type()
                        disable_new_channels = dialog.disable_new_channels()

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
                                    "enabled": not disable_new_channels,
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
            while item := iterator.value():
                if item.parent() is None:
                    iterator += 1
                    continue

                if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                    signals.append(item.text(0))

                iterator += 1
        else:
            while item := iterator.value():

                if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                    signals.append(item.text(0))

                iterator += 1

        config["selected_channels"] = signals

        windows = list(self.unknown_windows)
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

        current_window = self.mdi_area.currentSubWindow()

        config["windows"] = windows
        config["active_window"] = current_window.windowTitle() if current_window else ""
        config["functions"] = self.functions

        return config

    def save_channel_list(self, event=None, file_name=None):
        if file_name is None:
            if self.loaded_display_file[0].is_file():
                dir = str(self.loaded_display_file[0])
            else:
                dir = self.default_folder

            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select output display file",
                dir,
                "Display files (*.dspf)",
            )

        if file_name:
            file_name = Path(file_name)
            file_name.write_text(json.dumps(self.to_config(), indent=2))

            worker = sha1()
            worker.update(file_name.read_bytes())
            self.loaded_display_file = file_name, worker.hexdigest()

            self.display_file_modified.emit(Path(self.loaded_display_file[0]).name)

    def load_channel_list(self, event=None, file_name=None, manually=False):
        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select channel list file",
                self.default_folder,
                "Config file (*.cfg);;TXT files (*.txt);;Display files (*.dsp *.dspf);;CANape Lab file (*.lab);;All file types (*.cfg *.dsp *.dspf *.lab *.txt)",
                "All file types (*.cfg *.dsp *.dspf *.lab *.txt)",
            )

            if not file_name or Path(file_name).suffix.lower() not in (
                ".cfg",
                ".dsp",
                ".dspf",
                ".lab",
                ".txt",
            ):
                return

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

                    msg = MessageBox(
                        MessageBox.Information,
                        "DSP loading warning",
                        message,
                        parent=self,
                        defaultButton=MessageBox.Ok,
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
                if not info:
                    return
                section, ok = QtWidgets.QInputDialog.getItem(
                    self,
                    "Select section",
                    "Available sections:",
                    list(info),
                    0,
                    False,
                )
                if not ok:
                    return

                channels = [name.split(";")[0] for name in info[section]]

            elif extension in (".cfg", ".txt"):
                with open(file_name) as infile:
                    info = json.load(infile)
                channels = info.get("selected_channels", [])

            elif extension == ".dspf":
                with open(file_name) as infile:
                    info = json.load(infile)
                channels = info.get("selected_channels", [])

                original_file_name = Path(self.mdf.original_name)

                if (
                    original_file_name.suffix.lower()
                    in (
                        ".mf4",
                        ".mf4z",
                    )
                    and not self.mdf.header._common_properties.get("pr_display_file", "")
                    and manually
                ):
                    result = MessageBox.question(
                        self,
                        "Set default display file?",
                        "Would you like to use this display file as the default display file for this measurement file?",
                    )

                    if result == MessageBox.Yes:
                        display_file_name = str(Path(file_name).resolve())

                        _password = self.mdf._password

                        uuid = self.mdf.uuid

                        header = self.mdf.header

                        self.mdf.close()

                        windows = list(self.mdi_area.subWindowList())
                        for window in windows:
                            widget = window.widget()

                            self.mdi_area.removeSubWindow(window)
                            widget.setParent(None)
                            widget.close()
                            widget.deleteLater()
                            window.close()

                        suffix = original_file_name.suffix.lower()
                        if suffix == ".mf4z":
                            with ZipFile(original_file_name, allowZip64=True) as archive:
                                files = archive.namelist()
                                if len(files) != 1:
                                    return
                                fname = files[0]
                                if Path(fname).suffix.lower() != ".mf4":
                                    return

                                tmpdir = gettempdir()
                                mdf_file_name = archive.extract(fname, tmpdir)
                                mdf_file_name = Path(tmpdir) / mdf_file_name
                        else:
                            mdf_file_name = original_file_name

                        with open(mdf_file_name, "r+b") as mdf:
                            try:
                                header._common_properties["pr_display_file"] = display_file_name
                                comment = TextV4(meta=True, text=header.comment)

                                mdf.seek(0, 2)
                                address = mdf.tell()
                                align = address % 8
                                if align:
                                    mdf.write(b"\0" * (8 - align))
                                    address += 8 - align

                                mdf.write(bytes(comment))

                                header.comment_addr = address

                                mdf.seek(header.address)

                                mdf.write(bytes(header))

                            except:
                                print(format_exc())
                                return

                        if suffix == ".mf4z":
                            zipped_mf4 = ZipFile(original_file_name, "w", compression=ZIP_DEFLATED)
                            zipped_mf4.write(
                                str(mdf_file_name),
                                original_file_name.with_suffix(".mf4").name,
                                compresslevel=1,
                            )
                            zipped_mf4.close()
                            mdf_file_name.unlink()

                        self.mdf = mdf_module.MDF(
                            name=original_file_name,
                            callback=self.update_progress,
                            password=_password,
                            use_display_names=True,
                        )

                        self.mdf.original_name = original_file_name
                        self.mdf.uuid = uuid

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
                while item := iterator.value():
                    if item.parent() is None:
                        iterator += 1
                        continue

                    channel_name = item.text(0)
                    if channel_name in channels:
                        item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        channels.pop(channels.index(channel_name))
                    else:
                        item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

                    iterator += 1
            else:
                while item := iterator.value():

                    channel_name = item.text(0)
                    if channel_name in channels:
                        item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        channels.pop(channels.index(channel_name))
                    else:
                        item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

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
                        for name, definition in get_functions(window["configuration"]["channels"]).items():
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
        errors = {}
        if windows:
            count = len(windows)

            progress = setup_progress(
                parent=self,
                title="Loading display windows",
                message="",
                icon_name="window",
            )
            progress.setRange(0, count - 1)
            progress.resize(500, progress.height())

            for i, window in enumerate(windows, 1):
                window = _process_dict(window)
                window_type = window["type"]
                window_title = window["title"]
                progress.setLabelText(f"Loading {window_type} window <{window_title}>")
                QtWidgets.QApplication.processEvents()
                try:
                    self.load_window(window)
                    progress.setValue(i)
                except:
                    print(format_exc())
                    errors[window_title] = format_exc()

            progress.cancel()

            active_window = info.get("active_window", "")
            for window in self.mdi_area.subWindowList():
                if window.windowTitle() == active_window:
                    self.mdi_area.setActiveSubWindow(window)
                    break

        self.display_file_modified.emit(Path(self.loaded_display_file[0]).name)

        if errors:
            ErrorDialog(
                title="Errors while loading display file",
                message=f"There were errors while loading the following windows : {list(errors)}",
                trace="\n\n".join(list(errors.values())),
                parent=self,
            ).exec()

    def save_filter_list(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output filter list file",
            self.default_folder,
            "CANape Lab file (*.lab)",
            "CANape Lab file (*.lab)",
        )

        if file_name:
            file_name = Path(file_name)

            iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

            signals = []
            if self.filter_view.currentText() == "Internal file structure":
                while item := iterator.value():
                    iterator += 1

                    if item.parent() is None:
                        continue

                    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                        signals.append(item.text(0))
            else:
                while item := iterator.value():
                    iterator += 1

                    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
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
                    output.write(
                        """[SETTINGS]
Version;V1.1
MultiRasterSeparator;&

"""
                    )
                    output.write(f"[{section_name}]\n")
                output.write("\n".join(natsorted(signals)))

    def load_filter_list(self, event=None, file_name=None):
        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select channel list file",
                self.default_folder,
                "Config file (*.cfg);;Display files (*.dsp *.dspf);;CANape Lab file (*.lab);;All file types (*.cfg *.dsp *.dspf *.lab)",
                "All file types (*.cfg *.dsp *.dspf *.lab)",
            )

            if file_name is None or Path(file_name).suffix.lower() not in (
                ".cfg",
                ".dsp",
                ".dspf",
                ".lab",
                ".txt",
            ):
                return

        if not isinstance(file_name, dict):
            file_name = Path(file_name)

            if file_name.suffix.lower() == ".lab":
                info = load_lab(file_name)
                if info:
                    if len(info) > 1:
                        lab_section, ok = QtWidgets.QInputDialog.getItem(
                            self,
                            "Please select the ASAP section name",
                            "Available sections:",
                            list(info),
                            0,
                            False,
                        )
                        if ok:
                            channels = info[lab_section]
                        else:
                            return
                    else:
                        channels = list(info.values())[0]

                    channels = [name.split(";")[0] for name in channels]

            else:
                channels = load_channel_names_from_file(file_name)

        else:
            info = file_name
            channels = info.get("selected_channels", [])

        if channels:
            iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

            if self.filter_view.currentText() == "Internal file structure":
                while item := iterator.value():
                    iterator += 1

                    if item.parent() is None:
                        continue

                    channel_name = item.text(0)
                    if channel_name in channels:
                        item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        channels.pop(channels.index(channel_name))
                    else:
                        item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

            elif self.filter_view.currentText() == "Natural sort":
                while item := iterator.value():
                    channel_name = item.text(0)
                    if channel_name in channels:
                        item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        channels.pop(channels.index(channel_name))
                    else:
                        item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

                    iterator += 1

            else:
                items = []
                self.filter_tree.clear()

                self._selected_filter = set(channels)

                for i, gp in enumerate(self.mdf.groups):
                    for j, ch in enumerate(gp.channels):
                        if ch.name in channels:
                            entry = i, j
                            channel = MinimalTreeItem(entry, ch.name, strings=[ch.name], origin_uuid=self.uuid)
                            channel.setCheckState(0, QtCore.Qt.CheckState.Checked)
                            items.append(channel)
                            channels.pop(channels.index(ch.name))

                if len(items) < 30000:
                    items = natsorted(items, key=lambda x: x.name)
                else:
                    items.sort(key=lambda x: x.name)
                self.filter_tree.addTopLevelItems(items)

                self.update_selected_filter_channels()

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
                master_max = self.mdf.get_master(i, record_offset=cycles_nr - 1, record_count=1)
                if len(master_max):
                    t_max.append(master_max[0])
                self.mdf._master_channel_cache.clear()

        if t_min:
            time_range = t_min, t_max

            self.cut_start.setRange(*time_range)
            self.cut_stop.setRange(*time_range)

            self.cut_interval.setText("Cut interval ({:.6f}s - {:.6f}s)".format(*time_range))
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

        while item := iterator.value():
            item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def clear_channels(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)

        if self.channel_view.currentIndex() == 1:
            while item := iterator.value():
                if item.parent() is None:
                    item.setExpanded(False)
                else:
                    item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                iterator += 1
        else:
            while item := iterator.value():
                item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                iterator += 1

    def select_all_channels(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels_tree)

        if self.channel_view.currentIndex() == 1:
            while item := iterator.value():
                if item.parent() is None:
                    item.setExpanded(False)
                else:
                    item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                iterator += 1
        else:
            while item := iterator.value():
                item.setCheckState(0, QtCore.Qt.CheckState.Checked)
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

            disable_new_channels = dialog.disable_new_channels()

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
                    while item := iterator.value():
                        if item.parent() is None:
                            iterator += 1
                            continue

                        if item.checkState(0) == QtCore.Qt.CheckState.Checked:
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
                                        "enabled": not disable_new_channels,
                                    }
                                )

                        iterator += 1
                else:
                    while item := iterator.value():

                        if item.checkState(0) == QtCore.Qt.CheckState.Checked:
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
                                        "enabled": not disable_new_channels,
                                    }
                                )

                        iterator += 1

        self.add_window((window_type, signals))

    def scramble_thread(self, progress):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/scramble.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit("Scrambling measurement")
        progress.signals.setLabelText.emit(f'Scrambling "{self.file_name}"')

        # scrambling self.mdf
        mdf_module.MDF.scramble(name=self.file_name, progress=progress)

    def scramble_finished(self):
        if self._progress.error is None and self._progress.result is not TERMINATED:
            path = Path(self.file_name)
            self.open_new_file.emit(str(path.with_suffix(f".scrambled{path.suffix}")))

        self._progress = None

    def scramble(self, event):
        self._progress = setup_progress(parent=self)
        self._progress.qfinished.connect(self.scramble_finished)

        self._progress.run_thread_with_progress(
            target=self.scramble_thread,
            args=(),
            kwargs={},
        )

    def extract_bus_logging_finished(self):
        if self._progress.error is None and self._progress.result is not TERMINATED:
            file_name, message = self._progress.result

            self.output_info_bus.setPlainText("\n".join(message))
            self.open_new_file.emit(str(file_name))

        self._progress = None

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
                database_files["CAN"].append((widget.database.text(), widget.bus.currentIndex()))

        count2 = self.lin_database_list.count()
        if count2:
            database_files["LIN"] = []
            for i in range(count2):
                item = self.lin_database_list.item(i)
                widget = self.lin_database_list.itemWidget(item)
                database_files["LIN"].append((widget.database.text(), widget.bus.currentIndex()))

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
            "Save as measurement file",
            "",
            f"{filter};;All files (*.*)",
            filter,
        )

        if not file_name:
            return

        self._progress = setup_progress(parent=self)
        self._progress.qfinished.connect(self.extract_bus_logging_finished)

        self._progress.run_thread_with_progress(
            target=self.extract_bus_logging_thread,
            args=(file_name, suffix, database_files, version, compression),
            kwargs={},
        )

    def extract_bus_logging_thread(self, file_name, suffix, database_files, version, compression, progress):
        file_name = Path(file_name).with_suffix(suffix)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/down.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit("Extract Bus logging")
        progress.signals.setLabelText.emit(f'Extracting Bus signals from "{self.file_name}"')

        # convert self.mdf
        result = self.mdf.extract_bus_logging(
            database_files=database_files,
            version=version,
            prefix=self.prefix.text().strip(),
            progress=progress,
        )

        if result is TERMINATED:
            return
        else:
            mdf = result

        # then save it
        progress.signals.setLabelText.emit(f'Saving file to "{file_name}"')

        result = mdf.save(
            dst=file_name,
            compression=compression,
            overwrite=True,
            progress=progress,
        )

        if result is TERMINATED:
            return

        bus_call_info = dict(self.mdf.last_call_info)

        message = []

        for bus, call_info in bus_call_info.items():
            found_id_count = sum(len(e) for e in call_info["found_ids"].values())

            message += [
                f"{bus} bus summary:",
                f'- {found_id_count} of {len(call_info["total_unique_ids"])} IDs in the MDF4 file were matched in the DBC and converted',
            ]
            if call_info["unknown_id_count"]:
                message.append(f'- {call_info["unknown_id_count"]} unknown IDs in the MDF4 file')
            else:
                message.append("- no unknown IDs inf the MDF4 file")

            message += [
                "",
                "Detailed information:",
                "",
                f"The following {bus} IDs were in the MDF log file and matched in the DBC:",
            ]
            for dbc_name, found_ids in call_info["found_ids"].items():
                for msg_id_info, msg_name in sorted(found_ids):
                    if msg_id_info[2]:
                        pgn, sa = msg_id_info[:2]
                        message.append(f"- PGN=0x{pgn:X} SA=0x{sa:X} --> {msg_name} in <{dbc_name}>")
                    else:
                        msg_id, extended = msg_id_info[:2]
                        message.append(f"- 0x{msg_id:X} {extended=} --> {msg_name} in <{dbc_name}>")

            message += [
                "",
                f"The following {bus} IDs were in the MDF log file, but not matched in the DBC:",
            ]

            unknown_standard_can = sorted([e for e in call_info["unknown_ids"] if isinstance(e, int)])
            unknown_j1939 = sorted([e for e in call_info["unknown_ids"] if not isinstance(e, int)])
            for msg_id in unknown_standard_can:
                message.append(f"- 0x{msg_id:X}")

            for pgn, sa in unknown_j1939:
                message.append(f"- PGN=0x{pgn:X} SA=0x{sa:X}")

            message.append("\n\n")

        return file_name, message

    def extract_bus_csv_logging_finished(self):
        if self._progress.error is None and self._progress.result is not TERMINATED:
            message = self._progress.result

            self.output_info_bus.setPlainText("\n".join(message))

        self._progress = None

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
                database_files["CAN"].append((widget.database.text(), widget.bus.currentIndex()))

        count2 = self.lin_database_list.count()
        if count2:
            database_files["LIN"] = []
            for i in range(count2):
                item = self.lin_database_list.item(i)
                widget = self.lin_database_list.itemWidget(item)
                database_files["LIN"].append((widget.database.text(), widget.bus.currentIndex()))

        if not (count1 + count2):
            return

        single_time_base = self.single_time_base_bus.checkState() == QtCore.Qt.CheckState.Checked
        time_from_zero = self.time_from_zero_bus.checkState() == QtCore.Qt.CheckState.Checked
        empty_channels = self.empty_channels_bus.currentText()
        raster = self.export_raster_bus.value()
        time_as_date = self.bus_time_as_date.checkState() == QtCore.Qt.CheckState.Checked
        delimiter = self.delimiter_bus.text() or ","
        doublequote = self.doublequote_bus.checkState() == QtCore.Qt.CheckState.Checked
        escapechar = self.escapechar_bus.text() or None
        lineterminator = self.lineterminator_bus.text().replace("\\r", "\r").replace("\\n", "\n")
        quotechar = self.quotechar_bus.text() or '"'
        quoting = self.quoting_bus.currentText()
        add_units = self.add_units_bus.checkState() == QtCore.Qt.CheckState.Checked

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output CSV file",
            "",
            "CSV (*.csv);;All files (*.*)",
            "CSV (*.csv)",
        )

        if not file_name:
            return

        self._progress = setup_progress(parent=self)
        self._progress.qfinished.connect(self.extract_bus_csv_logging_finished)

        self._progress.run_thread_with_progress(
            target=self.extract_bus_csv_logging_thread,
            args=(
                file_name,
                database_files,
                version,
                single_time_base,
                time_from_zero,
                empty_channels,
                raster,
                time_as_date,
                delimiter,
                doublequote,
                escapechar,
                lineterminator,
                quotechar,
                quoting,
                add_units,
            ),
            kwargs={},
        )

    def extract_bus_csv_logging_thread(
        self,
        file_name,
        database_files,
        version,
        single_time_base,
        time_from_zero,
        empty_channels,
        raster,
        time_as_date,
        delimiter,
        doublequote,
        escapechar,
        lineterminator,
        quotechar,
        quoting,
        add_units,
        progress,
    ):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/csv.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit("Extract Bus logging to CSV")
        progress.signals.setLabelText.emit(f'Extracting Bus signals from "{self.file_name}"')

        # convert self.mdf
        result = self.mdf.extract_bus_logging(
            database_files=database_files,
            version=version,
            prefix=self.prefix.text().strip(),
            progress=progress,
        )

        if result is TERMINATED:
            return
        else:
            mdf = result

        # then save it
        progress.signals.setLabelText.emit(f'Saving file to "{file_name}"')

        mdf.configure(
            integer_interpolation=self.mdf._integer_interpolation,
            float_interpolation=self.mdf._float_interpolation,
        )

        result = mdf.export(
            fmt="csv",
            filename=file_name,
            single_time_base=single_time_base,
            time_from_zero=time_from_zero,
            empty_channels=empty_channels,
            raster=raster or None,
            time_as_date=time_as_date,
            ignore_value2text_conversions=self.ignore_value2text_conversions,
            delimiter=delimiter,
            doublequote=doublequote,
            escapechar=escapechar,
            lineterminator=lineterminator,
            quotechar=quotechar,
            quoting=quoting,
            add_units=add_units,
            progress=progress,
        )

        if result is TERMINATED:
            return

        bus_call_info = dict(self.mdf.last_call_info)

        message = []

        for bus, call_info in bus_call_info.items():
            found_id_count = sum(len(e) for e in call_info["found_ids"].values())

            message += [
                f"{bus} bus summary:",
                f'- {found_id_count} of {len(call_info["total_unique_ids"])} IDs in the MDF4 file were matched in the DBC and converted',
            ]
            if call_info["unknown_id_count"]:
                message.append(f'- {call_info["unknown_id_count"]} unknown IDs in the MDF4 file')
            else:
                message.append("- no unknown IDs inf the MDF4 file")

            message += [
                "",
                "Detailed information:",
                "",
                f"The following {bus} IDs were in the MDF log file and matched in the DBC:",
            ]
            for dbc_name, found_ids in call_info["found_ids"].items():
                for msg_id, msg_name in sorted(found_ids):
                    try:
                        message.append(f"- 0x{msg_id:X} --> {msg_name} in <{dbc_name}>")
                    except:
                        pgn, sa = msg_id
                        message.append(f"- PGN=0x{pgn:X} SA=0x{sa:X} --> {msg_name} in <{dbc_name}>")

            message += [
                "",
                f"The following {bus} IDs were in the MDF log file, but not matched in the DBC:",
            ]
            for msg_id in sorted(call_info["unknown_ids"]):
                message.append(f"- 0x{msg_id:X}")
            message.append("\n\n")

        return message

    def load_can_database(self, event):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select CAN database file",
            "",
            "ARXML or DBC (*.dbc *.arxml)",
            "ARXML or DBC (*.dbc *.arxml)",
        )

        if file_names:
            file_names = [name for name in file_names if Path(name).suffix.lower() in (".arxml", ".dbc")]

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
            file_names = [name for name in file_names if Path(name).suffix.lower() in (".arxml", ".dbc", ".ldf")]

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

        if key == QtCore.Qt.Key.Key_F and modifier == QtCore.Qt.KeyboardModifier.ControlModifier:
            self.search()
            event.accept()

        elif key == QtCore.Qt.Key.Key_F11:
            self.full_screen_toggled.emit()
            event.accept()

        elif (
            key in (QtCore.Qt.Key.Key_V, QtCore.Qt.Key.Key_H, QtCore.Qt.Key.Key_C, QtCore.Qt.Key.Key_T)
            and modifier == QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            if key == QtCore.Qt.Key.Key_V:
                mode = "tile vertically"
            elif key == QtCore.Qt.Key.Key_H:
                mode = "tile horizontally"
            elif key == QtCore.Qt.Key.Key_C:
                mode = "cascade"
            elif key == QtCore.Qt.Key.Key_T:
                mode = "tile"

            if mode == "tile":
                self.mdi_area.tileSubWindows()
            elif mode == "cascade":
                self.mdi_area.cascadeSubWindows()
            elif mode == "tile vertically":
                self.mdi_area.tile_vertically()
            elif mode == "tile horizontally":
                self.mdi_area.tile_horizontally()

            event.accept()

        elif key == QtCore.Qt.Key.Key_F and modifier == (
            QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.AltModifier
        ):
            self.toggle_frames()
            event.accept()

        elif key == QtCore.Qt.Key.Key_L and modifier == QtCore.Qt.KeyboardModifier.ShiftModifier:
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
            event.accept()

        elif key == QtCore.Qt.Key.Key_Period and modifier == QtCore.Qt.KeyboardModifier.NoModifier:
            self.set_line_style()
            event.accept()

        else:
            widget = self.get_current_widget()
            if widget:
                if isinstance(widget, Plot):
                    widget.plot.viewbox.keyPressEvent(event)
                else:
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
                    QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
                )
                self.raster_channel.addItems(sorted(self.mdf.channels_db, key=lambda x: x.lower()))
                self.raster_channel.setMinimumWidth(100)

            if not self._show_filter_tree:
                self._show_filter_tree = True

                widget = self.filter_tree
                widget.clear()

                self._update_channel_tree(current_index, widget, force=True)

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
            self.info.setItemDelegate(Delegate(self.info))

            # info tab
            try:
                file_stats = os.stat(self.mdf.original_name)
            except:
                file_stats = None
            file_info = QtWidgets.QTreeWidgetItem()
            file_info.setText(0, "File information")

            self.info.addTopLevelItem(file_info)

            children = []

            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, "Path")
            item.setText(1, str(self.mdf.original_name))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            children.append(item)

            item = QtWidgets.QTreeWidgetItem()
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setText(0, "Size")
            if file_stats is not None:
                item.setText(1, f"{file_stats.st_size / 1024 / 1024:.1f} MB")
            else:
                try:
                    item.setText(1, f"{self.mdf.file_limit / 1024 / 1024:.1f} MB")
                except:
                    item.setText(1, "Unknown size")
            children.append(item)

            if file_stats is not None:
                date_ = datetime.fromtimestamp(file_stats.st_ctime)
            else:
                date_ = datetime.now(timezone.utc)
            item = QtWidgets.QTreeWidgetItem()
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setText(0, "Created")
            item.setText(1, date_.strftime("%d-%b-%Y %H:%M:%S"))
            children.append(item)

            if file_stats is not None:
                date_ = datetime.fromtimestamp(file_stats.st_mtime)
            else:
                date_ = datetime.now(timezone.utc)
            item = QtWidgets.QTreeWidgetItem()
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setText(0, "Last modified")
            item.setText(1, date_.strftime("%d-%b-%Y %H:%M:%S"))
            children.append(item)

            file_info.addChildren(children)

            mdf_info = QtWidgets.QTreeWidgetItem()
            mdf_info.setText(0, "MDF information")

            self.info.addTopLevelItem(mdf_info)

            children = []

            item = QtWidgets.QTreeWidgetItem()
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setText(0, "Version")
            item.setText(1, self.mdf.version)
            children.append(item)

            item = QtWidgets.QTreeWidgetItem()
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setText(0, "Program identification")
            item.setText(
                1,
                self.mdf.identification.program_identification.decode("ascii").strip(" \r\n\t\0"),
            )
            children.append(item)

            item = QtWidgets.QTreeWidgetItem()
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setText(0, "Measurement start time")
            item.setText(1, self.mdf.header.start_time_string())
            children.append(item)

            item = QtWidgets.QTreeWidgetItem()
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setText(0, "Measurement comment")
            item.setText(1, self.mdf.header.description)
            item.setTextAlignment(0, QtCore.Qt.AlignmentFlag.AlignTop)
            children.append(item)

            mesaurement_attributes = QtWidgets.QTreeWidgetItem()
            mesaurement_attributes.setText(0, "Measurement attributes")
            children.append(mesaurement_attributes)

            for name, value in self.mdf.header._common_properties.items():
                if isinstance(value, dict):
                    tree = QtWidgets.QTreeWidgetItem()
                    tree.setFlags(tree.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    tree.setText(0, name)
                    tree.setTextAlignment(0, QtCore.Qt.AlignmentFlag.AlignTop)

                    for subname, subvalue in value.items():
                        item = QtWidgets.QTreeWidgetItem()
                        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                        item.setText(0, subname)
                        item.setText(1, str(subvalue).strip())
                        item.setTextAlignment(0, QtCore.Qt.AlignmentFlag.AlignTop)

                        tree.addChild(item)

                    mesaurement_attributes.addChild(tree)

                else:
                    item = QtWidgets.QTreeWidgetItem()
                    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    item.setText(0, FRIENDLY_ATRRIBUTES.get(name, name))
                    item.setText(1, str(value).strip())
                    item.setTextAlignment(0, QtCore.Qt.AlignmentFlag.AlignTop)
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

                acq_name = getattr(group.channel_group, "acq_name", "")
                if acq_name:
                    base_name = f"CG {i} {acq_name}"
                else:
                    base_name = f"CG {i}"
                if comment and acq_name != comment:
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
                        size = channel_group.samples_byte_nr + (channel_group.invalidation_bytes_nr << 32)
                    else:
                        size = (channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr) * cycles

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
                            icon.addPixmap(QtGui.QPixmap(ico), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
                            channel_group_item.setIcon(0, icon)

                item = QtWidgets.QTreeWidgetItem()
                item.setText(0, "Channels")
                item.setText(1, f"{len(group.channels)}")
                channel_group_item.addChild(item)

                item = QtWidgets.QTreeWidgetItem()
                item.setText(0, "Cycles")
                item.setText(1, str(cycles))
                if cycles:
                    item.setForeground(1, QtGui.QBrush(QtGui.QColor(GREEN)))
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
                    item.setForeground(1, QtGui.QBrush(QtGui.QColor(GREEN)))
                channel_group_item.addChild(item)

                channel_groups_children.append(channel_group_item)

            channel_groups.addChildren(channel_groups_children)

            channels = QtWidgets.QTreeWidgetItem()
            channels.setText(0, "Channels")
            channels.setText(1, str(sum(len(entry) for entry in self.mdf.channels_db.values())))
            children.append(channels)

            mdf_info.addChildren(children)

            self.info.expandAll()

            self.info.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

    def toggle_frames(self, event=None):
        self._frameless_windows = not self._frameless_windows

        for window in self.mdi_area.subWindowList():
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = True

        for w in self.mdi_area.subWindowList():
            if self._frameless_windows:
                w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)
            else:
                w.setWindowFlags(w.windowFlags() & (~QtCore.Qt.WindowType.FramelessWindowHint))

        for window in self.mdi_area.subWindowList():
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = True

    def autofit_sub_plots(self):
        geometries = []
        for window in self.mdi_area.subWindowList():
            geometry = window.geometry()
            geometries.append(geometry)

        if len({(g.width(), g.x()) for g in geometries}) == 1:
            self.mdi_area.tile_vertically()
        elif len({(g.height(), g.y()) for g in geometries}) == 1:
            self.mdi_area.tile_horizontally()
        else:
            self.mdi_area.tileSubWindows()

    def _current_options(self):
        options = {
            "needs_cut": self.cut_group.isChecked(),
            "cut_start": self.cut_start.value(),
            "cut_stop": self.cut_stop.value(),
            "cut_time_from_zero": self.cut_time_from_zero.checkState() == QtCore.Qt.CheckState.Checked,
            "whence": int(self.whence.checkState() == QtCore.Qt.CheckState.Checked),
            "needs_resample": self.resample_group.isChecked(),
            "raster_type_step": self.raster_type_step.isChecked(),
            "raster_type_channel": self.raster_type_channel.isChecked(),
            "raster": self.raster.value(),
            "raster_channel": self.raster_channel.currentText(),
            "resample_time_from_zero": self.resample_time_from_zero.checkState() == QtCore.Qt.CheckState.Checked,
            "output_format": self.output_format.currentText(),
        }

        output_format = self.output_format.currentText()

        if output_format == "MDF":
            new = {
                "mdf_version": self.mdf_version.currentText(),
                "mdf_compression": self.mdf_compression.currentIndex(),
                "mdf_split": self.mdf_split.checkState() == QtCore.Qt.CheckState.Checked,
                "mdf_split_size": self.mdf_split_size.value() * 1024 * 1024,
            }

        elif output_format == "MAT":
            new = {
                "single_time_base": self.single_time_base_mat.checkState() == QtCore.Qt.CheckState.Checked,
                "time_from_zero": self.time_from_zero_mat.checkState() == QtCore.Qt.CheckState.Checked,
                "time_as_date": self.time_as_date_mat.checkState() == QtCore.Qt.CheckState.Checked,
                "use_display_names": self.use_display_names_mat.checkState() == QtCore.Qt.CheckState.Checked,
                "reduce_memory_usage": self.reduce_memory_usage_mat.checkState() == QtCore.Qt.CheckState.Checked,
                "compression": self.export_compression_mat.currentText() == "enabled",
                "empty_channels": self.empty_channels_mat.currentText(),
                "mat_format": self.mat_format.currentText(),
                "oned_as": self.oned_as.currentText(),
                "raw": self.raw_mat.checkState() == QtCore.Qt.CheckState.Checked,
            }

        elif output_format == "CSV":
            new = {
                "single_time_base": self.single_time_base_csv.checkState() == QtCore.Qt.CheckState.Checked,
                "time_from_zero": self.time_from_zero_csv.checkState() == QtCore.Qt.CheckState.Checked,
                "time_as_date": self.time_as_date_csv.checkState() == QtCore.Qt.CheckState.Checked,
                "use_display_names": self.use_display_names_csv.checkState() == QtCore.Qt.CheckState.Checked,
                "reduce_memory_usage": False,
                "compression": False,
                "empty_channels": self.empty_channels_csv.currentText(),
                "raw": self.raw_csv.checkState() == QtCore.Qt.CheckState.Checked,
                "delimiter": self.delimiter.text() or ",",
                "doublequote": self.doublequote.checkState() == QtCore.Qt.CheckState.Checked,
                "escapechar": self.escapechar.text() or None,
                "lineterminator": self.lineterminator.text().replace("\\r", "\r").replace("\\n", "\n"),
                "quotechar": self.quotechar.text() or '"',
                "quoting": self.quoting.currentText(),
                "mat_format": None,
                "oned_as": None,
            }

        else:
            new = {
                "single_time_base": self.single_time_base.checkState() == QtCore.Qt.CheckState.Checked,
                "time_from_zero": self.time_from_zero.checkState() == QtCore.Qt.CheckState.Checked,
                "time_as_date": self.time_as_date.checkState() == QtCore.Qt.CheckState.Checked,
                "use_display_names": self.use_display_names.checkState() == QtCore.Qt.CheckState.Checked,
                "reduce_memory_usage": self.reduce_memory_usage.checkState() == QtCore.Qt.CheckState.Checked,
                "compression": self.export_compression.currentText(),
                "empty_channels": self.empty_channels.currentText(),
                "mat_format": None,
                "oned_as": None,
                "raw": self.raw.checkState() == QtCore.Qt.CheckState.Checked,
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

        if self.filter_view.currentText() == "Internal file structure":
            while item := iterator.value():

                group, index = item.entry

                if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                    if index != 0xFFFFFFFFFFFFFFFF:
                        channels.append((None, group, index))

                iterator += 1
        else:
            while item := iterator.value():

                if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                    group, index = item.entry
                    channels.append((None, group, index))

                iterator += 1

        needs_filter = self.selected_filter_channels.count() > 0

        return needs_filter, channels

    def apply_processing(self, event):
        needs_filter, channels = self._get_filtered_channels()

        opts = self._current_options()

        output_format = opts.output_format

        if output_format == "HDF5":
            try:
                from h5py import File as HDF5  # noqa: F401
            except ImportError:
                MessageBox.critical(
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
                    MessageBox.critical(
                        self,
                        "Export to mat v7.3 unavailale",
                        "hdf5storage package not found; export to mat 7.3 is unavailable",
                    )
                    return
            else:
                try:
                    from scipy.io import savemat  # noqa: F401
                except ImportError:
                    MessageBox.critical(
                        self,
                        "Export to mat v4 and v5 unavailale",
                        "scipy package not found; export to mat is unavailable",
                    )
                    return

        elif output_format == "Parquet":
            try:
                from fastparquet import write as write_parquet  # noqa: F401
            except ImportError:
                MessageBox.critical(
                    self,
                    "Export to parquet unavailale",
                    "fastparquet package not found; export to parquet is unavailable",
                )
                return

        if output_format == "MDF":
            version = opts.mdf_version

            if version < "4.00":
                filter = "MDF version 3 files (*.dat *.mdf)"
                default = filter
            else:
                filter = "MDF version 4 files (*.mf4);;Zipped MDF version 4 files (*.mf4z)"
                if Path(self.mdf.original_name).suffix.lower() == ".mf4z":
                    default = "Zipped MDF version 4 files (*.mf4z)"
                else:
                    default = "MDF version 4 files (*.mf4)"

            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save as measurement file",
                "",
                f"{filter};;All files (*.*)",
                default,
            )

        else:
            filters = {
                "ASC": "Vector ascii files (*.asc)",
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

            version = ""

        if not file_name:
            return

        self._progress = setup_progress(parent=self)
        self._progress.qfinished.connect(self.apply_processing_finished)

        self._progress.run_thread_with_progress(
            target=self.apply_processing_thread,
            args=(),
            kwargs={
                "file_name": file_name,
                "opts": opts,
                "version": version,
                "needs_filter": needs_filter,
                "channels": channels,
            },
        )

    def apply_processing_finished(self):
        self._progress = None

    def apply_processing_thread(self, file_name, opts, version, needs_filter, channels, progress=None):
        output_format = opts.output_format

        split_size = opts.mdf_split_size if output_format == "MDF" else 0
        self.mdf.configure(read_fragment_size=split_size)

        mdf = None
        integer_interpolation = self.mdf._integer_interpolation
        float_interpolation = self.mdf._float_interpolation

        if needs_filter:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/filter.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            progress.signals.setWindowIcon.emit(icon)
            progress.signals.setWindowTitle.emit("Filtering measurement")
            progress.signals.setLabelText.emit(f'Filtering selected channels from "{self.file_name}"')

            # filtering self.mdf
            result = self.mdf.filter(
                channels=channels,
                version=opts.mdf_version if output_format == "MDF" else "4.10",
                progress=progress,
            )

            if result is TERMINATED:
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
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/cut.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            progress.signals.setWindowIcon.emit(icon)
            progress.signals.setWindowTitle.emit("Cutting measurement")
            progress.signals.setLabelText.emit(f"Cutting from {opts.cut_start}s to {opts.cut_stop}s")

            # cut self.mdf
            target = self.mdf.cut if mdf is None else mdf.cut
            result = target(
                start=opts.cut_start,
                stop=opts.cut_stop,
                whence=opts.whence,
                version=opts.mdf_version if output_format == "MDF" else "4.10",
                time_from_zero=opts.cut_time_from_zero,
                progress=progress,
            )

            if result is TERMINATED:
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

            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/resample.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            progress.signals.setWindowIcon.emit(icon)
            progress.signals.setWindowTitle.emit("Resampling measurement")
            progress.signals.setLabelText.emit(message)

            # resample self.mdf
            target = self.mdf.resample if mdf is None else mdf.resample

            result = target(
                raster=raster,
                version=opts.mdf_version if output_format == "MDF" else "4.10",
                time_from_zero=opts.resample_time_from_zero,
                progress=progress,
            )

            if result is TERMINATED:
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
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(":/convert.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit("Converting measurement")
                progress.signals.setLabelText.emit(
                    f'Converting "{self.file_name}" from {self.mdf.version} to {version}'
                )

                # convert self.mdf
                result = self.mdf.convert(
                    version=version,
                    progress=progress,
                )

                if result is TERMINATED:
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
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/save.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            progress.signals.setWindowIcon.emit(icon)
            progress.signals.setWindowTitle.emit("Saving measurement")
            progress.signals.setLabelText.emit(f'Saving output file "{file_name}"')

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
                    widget.deleteLater()
                    window.close()

            result = mdf.save(
                dst=file_name,
                compression=opts.mdf_compression,
                overwrite=True,
                progress=progress,
            )

            if result is TERMINATED:
                return

            if handle_overwrite:
                original_name = file_name

                self.mdf = mdf_module.MDF(
                    name=file_name,
                    password=_password,
                    use_display_names=True,
                )

                self.mdf.original_name = original_name
                self.mdf.uuid = self.uuid

                self.aspects.setCurrentIndex(0)

                # TO DO: may crash when modifying the GUI from this thread
                self.load_channel_list(file_name=dspf)

                self.aspects.setCurrentIndex(1)

        else:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/export.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            progress.signals.setWindowIcon.emit(icon)
            progress.signals.setWindowTitle.emit("Export measurement")
            progress.signals.setLabelText.emit(f"Exporting to {output_format} (be patient this might take a while)")

            delimiter = self.delimiter.text() or ","
            doublequote = self.doublequote.checkState() == QtCore.Qt.CheckState.Checked
            escapechar = self.escapechar.text() or None
            lineterminator = self.lineterminator.text().replace("\\r", "\r").replace("\\n", "\n")
            quotechar = self.quotechar.text() or '"'
            quoting = self.quoting.currentText()
            add_units = self.add_units.checkState() == QtCore.Qt.CheckState.Checked

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
                "progress": progress,
            }

            target(**kwargs)

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

    def filter_changed(self, item, column=0):
        name = item.text(0)
        if self.filter_view.currentText() == "Internal file structure":
            if item.checkState(0) == QtCore.Qt.CheckState.Checked and item.parent() is not None:
                self._selected_filter.add(name)
            else:
                if name in self._selected_filter:
                    self._selected_filter.remove(name)

        elif self.filter_view.currentText() == "Natural sort":
            if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                self._selected_filter.add(name)
            else:
                if name in self._selected_filter:
                    self._selected_filter.remove(name)

        else:
            if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                self._selected_filter.add(name)
            else:
                if name in self._selected_filter:
                    self._selected_filter.remove(name)
            self._update_channel_tree(widget=self.filter_tree)

        self._filter_timer.start(10)

    def update_selected_filter_channels(self):
        self.selected_filter_channels.clear()
        self.selected_filter_channels.addItems(sorted(self._selected_filter))

    def embed_display_file(self, event=None):
        if not self.save_embedded_channel_list_btn.isVisible() or not self.save_embedded_channel_list_btn.isEnabled():
            return

        original_file_name = Path(self.mdf.original_name)

        if original_file_name.suffix.lower() not in (".mf4", ".mf4z"):
            return MessageBox.warning(
                self,
                "Wrong file type",
                "The display file can only be embedded in .mf4 or .mf4z files" f"\n{original_file_name}",
            )

        _password = self.mdf._password

        uuid = self.mdf.uuid

        creator_index = len(self.mdf.file_history)
        current_display = self.to_config()
        data = json.dumps(current_display, indent=2).encode("utf-8", errors="replace")

        self.mdf.close()

        windows = list(self.mdi_area.subWindowList())
        for window in windows:
            widget = window.widget()

            self.mdi_area.removeSubWindow(window)
            widget.setParent(None)
            widget.close()
            widget.deleteLater()
            window.close()

        suffix = original_file_name.suffix.lower()
        if suffix == ".mf4z":
            with ZipFile(original_file_name, allowZip64=True) as archive:
                files = archive.namelist()
                if len(files) != 1:
                    return
                fname = files[0]
                if Path(fname).suffix.lower() != ".mf4":
                    return

                tmpdir = gettempdir()
                file_name = archive.extract(fname, tmpdir)
                file_name = Path(tmpdir) / file_name
        else:
            file_name = original_file_name

        with open(file_name, "r+b") as mdf:
            try:
                embedded_file_name = "user_embedded_display.dspf"
                mime = r"application/x-dspf"

                header = HeaderBlock(stream=mdf, address=64)

                at_addr = header.first_attachment_addr
                parent = header
                while at_addr:
                    at_block = AttachmentBlock(stream=mdf, address=at_addr)

                    if at_block.file_name == embedded_file_name and at_block.mime == mime:
                        new_at_block = AttachmentBlock(
                            data=data,
                            file_name=embedded_file_name,
                            comment="user embedded display file",
                            compression=True,
                            mime=mime,
                            embedded=True,
                            password=_password,
                            creator_index=at_block.creator_index,
                        )
                        new_at_block.next_at_addr = at_block.next_at_addr

                        blocks = []

                        mdf.seek(0, 2)
                        file_end = mdf.tell()
                        if (file_end - (at_block.address + at_block.block_len)) <= 7:
                            address = at_block.address
                        else:
                            address = file_end

                        mdf.seek(address)

                        align = address % 8
                        if align:
                            mdf.write(b"\0" * (8 - align))
                            address += 8 - align

                        address = new_at_block.to_blocks(address, blocks, {})
                        for block in blocks:
                            mdf.write(bytes(block))

                        if parent is header:
                            header.first_attachment_addr = new_at_block.address
                        else:
                            parent.next_at_addr = new_at_block.address

                        mdf.seek(parent.address)
                        mdf.write(bytes(parent))

                        mdf.truncate(new_at_block.address + new_at_block.block_len)

                        break

                    at_addr = at_block.next_at_addr
                    parent = at_block

                else:
                    at_block = AttachmentBlock(
                        data=data,
                        file_name=embedded_file_name,
                        comment="user embedded display file",
                        compression=True,
                        mime=mime,
                        embedded=True,
                        password=_password,
                    )

                    fh_block = FileHistory()
                    fh_block.comment = f"""<FHcomment>
    <TX>Added new embedded attachment from {file_name}</TX>
    <tool_id>{tool.__tool__}</tool_id>
    <tool_vendor>{tool.__vendor__}</tool_vendor>
    <tool_version>{tool.__version__}</tool_version>
</FHcomment>"""

                    at_block["creator_index"] = creator_index

                    blocks = []

                    mdf.seek(0, 2)
                    address = mdf.tell()
                    align = address % 8
                    if align:
                        mdf.write(b"\0" * (8 - align))
                        address += 8 - align

                    address = fh_block.to_blocks(address, blocks, {})
                    address = at_block.to_blocks(address, blocks, {})

                    for block in blocks:
                        mdf.write(bytes(block))

                    if header.first_attachment_addr:
                        at_addr = header.first_attachment_addr
                        while at_addr:
                            last_at = AttachmentBlock(stream=mdf, address=at_addr)
                            at_addr = last_at.next_at_addr

                        last_at.next_at_addr = at_block.address
                        mdf.seek(last_at.address)
                        mdf.write(bytes(last_at))
                    else:
                        header.first_attachment_addr = at_block.address

                    if header.file_history_addr:
                        fh_addr = header.file_history_addr
                        while fh_addr:
                            last_fh = FileHistory(stream=mdf, address=fh_addr)
                            fh_addr = last_fh.next_fh_addr

                        last_fh.next_fh_addr = fh_block.address
                        mdf.seek(last_fh.address)
                        mdf.write(bytes(last_fh))
                    else:
                        header.file_history_addr = fh_block.address

                    mdf.seek(header.address)

                    mdf.write(bytes(header))

            except:
                print(format_exc())
                return

        if suffix == ".mf4z":
            zipped_mf4 = ZipFile(original_file_name, "w", compression=ZIP_DEFLATED)
            zipped_mf4.write(
                str(file_name),
                original_file_name.with_suffix(".mf4").name,
                compresslevel=1,
            )
            zipped_mf4.close()
            file_name.unlink()

        self.mdf = mdf_module.MDF(
            name=original_file_name,
            callback=self.update_progress,
            password=_password,
            use_display_names=True,
        )

        self.mdf.original_name = original_file_name
        self.mdf.uuid = uuid

        self.attachments.clear()

        self.save_embedded_channel_list_btn.setEnabled(True)
        self.load_embedded_channel_list_btn.setEnabled(True)

        current_display["display_file_name"] = self.loaded_display_file[0]
        self.load_channel_list(file_name=current_display)

        if self.mdf.attachments:
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
                    text = f"{size / 1024:.1f} KB"
                elif size <= 1 << 30:
                    text = f"{size / 1024 / 1024:.1f} MB"
                else:
                    text = f"{size / 1024 / 1024 / 1024:.1f} GB"

                field = QtWidgets.QTreeWidgetItem()
                field.setText(0, "Size")
                field.setText(1, text)
                fields.append(field)

                att.fields.addTopLevelItems(fields)

                item = QtWidgets.QListWidgetItem()
                item.setSizeHint(att.sizeHint())
                self.attachments.addItem(item)
                self.attachments.setItemWidget(item, att)

            self.aspects.setTabVisible(4, True)

    def load_embedded_display_file(self, event=None):
        if not self.load_embedded_channel_list_btn.isVisible() or not self.load_embedded_channel_list_btn.isEnabled():
            return

        for index, attachment in enumerate(self.mdf.attachments):
            if attachment.file_name == "user_embedded_display.dspf" and attachment.mime == r"application/x-dspf":
                encryption_info = extract_encryption_information(attachment.comment)
                password = None
                if encryption_info.get("encrypted", False) and self.mdf._password is None:
                    text, ok = QtWidgets.QInputDialog.getText(
                        self,
                        "Attachment password",
                        "The attachment is encrypted. Please provide the password:",
                        QtWidgets.QLineEdit.EchoMode.Password,
                    )
                    if ok and text:
                        password = text

                data, file_path, md5_sum = self.mdf.extract_attachment(index, password=password)

                dsp = json.loads(data.decode("utf-8", errors="replace"))
                dsp["display_file_name"] = "user_embedded_display.dspf"

                self.load_channel_list(file_name=dsp)

    def connect_export_updates(self):
        self.output_format.currentTextChanged.connect(self.store_export_setttings)

        self.mdf_version.currentTextChanged.connect(self.store_export_setttings)
        self.mdf_compression.currentTextChanged.connect(self.store_export_setttings)
        self.mdf_split.stateChanged.connect(self.store_export_setttings)
        self.mdf_split_size.valueChanged.connect(self.store_export_setttings)

        self.single_time_base.stateChanged.connect(self.store_export_setttings)
        self.time_from_zero.stateChanged.connect(self.store_export_setttings)
        self.time_as_date.stateChanged.connect(self.store_export_setttings)
        self.raw.stateChanged.connect(self.store_export_setttings)
        self.use_display_names.stateChanged.connect(self.store_export_setttings)
        self.reduce_memory_usage.stateChanged.connect(self.store_export_setttings)
        self.export_compression.currentTextChanged.connect(self.store_export_setttings)
        self.empty_channels.currentTextChanged.connect(self.store_export_setttings)

        self.single_time_base_csv.stateChanged.connect(self.store_export_setttings)
        self.time_from_zero_csv.stateChanged.connect(self.store_export_setttings)
        self.time_as_date_csv.stateChanged.connect(self.store_export_setttings)
        self.raw_csv.stateChanged.connect(self.store_export_setttings)
        self.add_units.stateChanged.connect(self.store_export_setttings)
        self.doublequote.stateChanged.connect(self.store_export_setttings)
        self.use_display_names_csv.stateChanged.connect(self.store_export_setttings)
        self.empty_channels_csv.currentTextChanged.connect(self.store_export_setttings)
        self.delimiter.editingFinished.connect(self.store_export_setttings)
        self.escapechar.editingFinished.connect(self.store_export_setttings)
        self.lineterminator.editingFinished.connect(self.store_export_setttings)
        self.quotechar.editingFinished.connect(self.store_export_setttings)
        self.quoting.currentTextChanged.connect(self.store_export_setttings)

        self.single_time_base_mat.stateChanged.connect(self.store_export_setttings)
        self.time_from_zero_mat.stateChanged.connect(self.store_export_setttings)
        self.time_as_date_mat.stateChanged.connect(self.store_export_setttings)
        self.raw_mat.stateChanged.connect(self.store_export_setttings)
        self.use_display_names_mat.stateChanged.connect(self.store_export_setttings)
        self.reduce_memory_usage_mat.stateChanged.connect(self.store_export_setttings)
        self.empty_channels_mat.currentTextChanged.connect(self.store_export_setttings)
        self.export_compression_mat.currentTextChanged.connect(self.store_export_setttings)
        self.mat_format.currentTextChanged.connect(self.store_export_setttings)
        self.oned_as.currentTextChanged.connect(self.store_export_setttings)

    def restore_export_setttings(self):
        self.output_format.setCurrentText(self._settings.value("export", "MDF"))

        self.mdf_version.setCurrentText(self._settings.setValue("export/MDF/version", "4.10"))
        self.mdf_compression.setCurrentText(self._settings.value("export/MDF/compression", "transposed deflate"))
        self.mdf_split.setChecked(self._settings.value("export/MDF/split_data_blocks", True, type=bool))
        self.mdf_split_size.setValue(self._settings.value("export/MDF/split_size", 4, type=int))

        self.single_time_base.setChecked(self._settings.value("export/HDF5/single_time_base", False, type=bool))
        self.time_from_zero.setChecked(self._settings.value("export/HDF5/time_from_zero", False, type=bool))
        self.time_as_date.setChecked(self._settings.value("export/HDF5/time_as_date", False, type=bool))
        self.raw.setChecked(self._settings.value("export/HDF5/raw", False, type=bool))
        self.use_display_names.setChecked(self._settings.value("export/HDF5/use_display_names", False, type=bool))
        self.reduce_memory_usage.setChecked(self._settings.value("export/HDF5/reduce_memory_usage", False, type=bool))
        self.export_compression.setCurrentText(self._settings.value("export/HDF5/export_compression", "gzip"))
        self.empty_channels.setCurrentText(self._settings.value("export/HDF5/empty_channels", "skip"))

        self.single_time_base_csv.setChecked(self._settings.value("export/CSV/single_time_base_csv", False, type=bool))
        self.time_from_zero_csv.setChecked(self._settings.value("export/CSV/time_from_zero_csv", False, type=bool))
        self.time_as_date_csv.setChecked(self._settings.value("export/CSV/time_as_date_csv", False, type=bool))
        self.raw_csv.setChecked(self._settings.value("export/CSV/raw_csv", False, type=bool))
        self.add_units.setChecked(self._settings.value("export/CSV/add_units", False, type=bool))
        self.doublequote.setChecked(self._settings.value("export/CSV/doublequote", False, type=bool))
        self.use_display_names_csv.setChecked(
            self._settings.value("export/CSV/use_display_names_csv", False, type=bool)
        )
        self.empty_channels_csv.setCurrentText(self._settings.value("export/CSV/empty_channels_csv", "skip"))
        self.delimiter.setText(self._settings.value("export/CSV/delimiter", ","))
        self.escapechar.setText(self._settings.value("export/CSV/escapechar", ""))
        self.lineterminator.setText(self._settings.value("export/CSV/lineterminator", r"\r\n"))
        self.quotechar.setText(self._settings.value("export/CSV/quotechar", '"'))
        self.quoting.setCurrentText(self._settings.value("export/CSV/quoting", "MINIMAL"))

        self.single_time_base_mat.setChecked(self._settings.value("export/MAT/single_time_base_mat", False, type=bool))
        self.time_from_zero_mat.setChecked(self._settings.value("export/MAT/time_from_zero_mat", False, type=bool))
        self.time_as_date_mat.setChecked(self._settings.value("export/MAT/time_as_date_mat", False, type=bool))
        self.raw_mat.setChecked(self._settings.value("export/MAT/raw_mat", False, type=bool))
        self.use_display_names_mat.setChecked(
            self._settings.value("export/MAT/use_display_names_mat", False, type=bool)
        )
        self.reduce_memory_usage_mat.setChecked(
            self._settings.value("export/MAT/reduce_memory_usage_mat", False, type=bool)
        )
        self.empty_channels_mat.setCurrentText(self._settings.value("export/MAT/empty_channels_mat", "skip"))
        self.export_compression_mat.setCurrentText(self._settings.value("export/MAT/export_compression_mat", "enabled"))
        self.mat_format.setCurrentText(self._settings.value("export/MAT/mat_format", "4"))
        self.oned_as.setCurrentText(self._settings.value("export/MAT/oned_as", "row"))

    def store_export_setttings(self, *args):
        self._settings.setValue("export", self.output_format.currentText())

        self._settings.setValue("export/MDF/version", self.mdf_version.currentText())
        self._settings.setValue("export/MDF/compression", self.mdf_compression.currentText())
        self._settings.setValue("export/MDF/split_data_blocks", self.mdf_split.isChecked())
        self._settings.setValue("export/MDF/split_size", self.mdf_split_size.value())

        self._settings.setValue("export/HDF5/single_time_base", self.single_time_base.isChecked())
        self._settings.setValue("export/HDF5/time_from_zero", self.time_from_zero.isChecked())
        self._settings.setValue("export/HDF5/time_as_date", self.time_as_date.isChecked())
        self._settings.setValue("export/HDF5/raw", self.raw.isChecked())
        self._settings.setValue("export/HDF5/use_display_names", self.use_display_names.isChecked())
        self._settings.setValue("export/HDF5/reduce_memory_usage", self.reduce_memory_usage.isChecked())
        self._settings.setValue("export/HDF5/export_compression", self.export_compression.currentText())
        self._settings.setValue("export/HDF5/empty_channels", self.empty_channels.currentText())

        self._settings.setValue("export/CSV/single_time_base_csv", self.single_time_base_csv.isChecked())
        self._settings.setValue("export/CSV/time_from_zero_csv", self.time_from_zero_csv.isChecked())
        self._settings.setValue("export/CSV/time_as_date_csv", self.time_as_date_csv.isChecked())
        self._settings.setValue("export/CSV/raw_csv", self.raw_csv.isChecked())
        self._settings.setValue("export/CSV/add_units", self.add_units.isChecked())
        self._settings.setValue("export/CSV/doublequote", self.doublequote.isChecked())
        self._settings.setValue("export/CSV/use_display_names_csv", self.use_display_names_csv.isChecked())
        self._settings.setValue("export/CSV/empty_channels_csv", self.empty_channels_csv.currentText())
        self._settings.setValue("export/CSV/delimiter", self.delimiter.text())
        self._settings.setValue("export/CSV/escapechar", self.escapechar.text())
        self._settings.setValue("export/CSV/lineterminator", self.lineterminator.text())
        self._settings.setValue("export/CSV/quotechar", self.quotechar.text())
        self._settings.setValue("export/CSV/quoting", self.quoting.currentText())

        self._settings.setValue("export/MAT/single_time_base_mat", self.single_time_base_mat.isChecked())
        self._settings.setValue("export/MAT/time_from_zero_mat", self.time_from_zero_mat.isChecked())
        self._settings.setValue("export/MAT/time_as_date_mat", self.time_as_date_mat.isChecked())
        self._settings.setValue("export/MAT/raw_mat", self.raw_mat.isChecked())
        self._settings.setValue("export/MAT/use_display_names_mat", self.use_display_names_mat.isChecked())
        self._settings.setValue("export/MAT/reduce_memory_usage_mat", self.reduce_memory_usage_mat.isChecked())
        self._settings.setValue("export/MAT/empty_channels_mat", self.empty_channels_mat.currentText())
        self._settings.setValue("export/MAT/export_compression_mat", self.export_compression_mat.currentText())
        self._settings.setValue("export/MAT/mat_format", self.mat_format.currentText())
        self._settings.setValue("export/MAT/oned_as", self.oned_as.currentText())

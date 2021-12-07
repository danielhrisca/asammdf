# -*- coding: utf-8 -*-
from copy import deepcopy
from functools import partial
import itertools
import json
import os
import re
import sys
from traceback import format_exc

from natsort import natsorted
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from ...blocks import v4_constants as v4c
from ...blocks.conversion_utils import from_dict
from ...blocks.utils import (
    csv_bytearray2hex,
    extract_cncomment_xml,
    load_can_database,
    MdfException,
)
from ...mdf import MDF
from ...signal import Signal
from ..dialogs.channel_info import ChannelInfoDialog
from ..dialogs.window_selection_dialog import WindowSelectionDialog
from ..utils import compute_signal, copy_ranges, extract_mime_names
from .bar import Bar
from .can_bus_trace import CANBusTrace
from .channel_display import ChannelDisplay
from .flexray_bus_trace import FlexRayBusTrace
from .gps import GPS
from .lin_bus_trace import LINBusTrace
from .numeric import Numeric
from .plot import Plot
from .tabular import Tabular
from .tree import ChannelsGroupTreeItem

COMPONENT = re.compile(r"\[(?P<index>\d+)\]$")
SIG_RE = re.compile(r"\{\{(?!\}\})(?P<name>.*?)\}\}")
NOT_FOUND = 2 ** 32 - 1


def build_mime_from_config(channels, mdf=None, uuid=None, default_index=-1):
    mime = []
    for channel in channels:
        if channel.get("type", "channel") == "group":
            if channel.get("pattern", None) is None:
                mime.append(
                    (
                        channel["name"],
                        None,
                        build_mime_from_config(
                            channel["channels"], mdf, uuid, default_index
                        ),
                        None,
                        "group",
                        channel.get("ranges", []),
                    )
                )
            else:
                mime.append(
                    (
                        channel["name"],
                        channel["pattern"],
                        [],
                        None,
                        "group",
                        channel.get("ranges", []),
                    )
                )
        else:
            occurrences = mdf.whereis(channel["name"]) if mdf else None
            if occurrences:
                group_index, channel_index = occurrences[0]
            else:
                group_index, channel_index = default_index, default_index
            mime.append(
                (
                    channel["name"],
                    group_index,
                    channel_index,
                    uuid or channel["mdf_uuid"],
                    "channel",
                    channel.get("ranges", []),
                )
            )
    return mime


def extract_signals_using_pattern(
    mdf, pattern_info, ignore_value2text_conversions, uuid
):
    pattern = pattern_info["pattern"]
    match_type = pattern_info["match_type"]
    filter_value = pattern_info["filter_value"]
    filter_type = pattern_info["filter_type"]
    raw = pattern_info["raw"]

    if match_type == "Wildcard":
        pattern = pattern.replace("*", "_WILDCARD_")
        pattern = re.escape(pattern)
        pattern = pattern.replace("_WILDCARD_", ".*")

    try:
        pattern = re.compile(f"(?i){pattern}")
        matches = {
            name: entries[0]
            for name, entries in mdf.channels_db.items()
            if pattern.search(name)
        }
    except:
        print(format_exc())
        signals = []
    else:

        psignals = mdf.select(
            list(matches),
            ignore_value2text_conversions=ignore_value2text_conversions,
            copy_master=False,
            validate=True,
            raw=True,
        )

        if filter_type == "Unspecified":
            keep = psignals
        else:

            keep = []
            for i, (name, entry) in enumerate(matches.items()):
                sig = psignals[i]
                sig.mdf_uuid = uuid
                sig.group_index, sig.channel_index = entry

                size = len(sig)
                if not size:
                    continue

                target = np.ones(size) * filter_value

                if not raw:
                    samples = sig.physical().samples
                else:
                    samples = sig.samples

                if filter_type == "Contains":
                    try:
                        if np.any(np.isclose(samples, target)):
                            keep.append(sig)
                    except:
                        continue
                elif filter_type == "Do not contain":
                    try:
                        if not np.allclose(samples, target):
                            keep.append(sig)
                    except:
                        continue
                else:
                    try:
                        if np.allclose(samples, target):
                            keep.append(sig)
                    except:
                        continue
        signals = keep

    return signals


def generate_window_title(mdi, window_name="", title=""):
    used_names = {
        window.windowTitle()
        for window in mdi.mdiArea().subWindowList()
        if window is not mdi
    }

    if not title or title in used_names:
        window_name = title or window_name or "Subwindow"

        i = 0
        while True:
            name = f"{window_name} {i}"
            if name in used_names:
                i += 1
            else:
                break
    else:
        name = title

    return name


def get_descriptions(channels):
    descriptions = {}
    for channel in channels:
        if channel.get("type", "channel") == "group":
            new_descriptions = get_descriptions(channel["channels"])
            descriptions.update(new_descriptions)
        else:
            descriptions[channel["name"]] = channel

    return descriptions


def get_flatten_entries_from_mime(data, default_index=None):
    entries = []

    for (name, group_index, channel_index, mdf_uuid, type_, ranges) in data:
        if type_ == "channel":
            if default_index is not None:
                entries.append(
                    (name, default_index, default_index, mdf_uuid, "channel", ranges)
                )
            else:
                entries.append(
                    (name, group_index, channel_index, mdf_uuid, "channel", ranges)
                )
        else:
            entries.extend(get_flatten_entries_from_mime(channel_index, default_index))
    return entries


def get_pattern_groups(data):
    groups = []
    for (name, pattern, channels, mdf_uuid, type_, ranges) in data:
        if type_ == "group":
            if pattern is not None:
                groups.append(name, pattern, channels, mdf_uuid, type_, ranges)
            else:
                groups.extend(get_pattern_groups(channels))
    return groups


def get_required_from_computed(channel):
    names = []
    if "computed" in channel:
        if channel["computed"]:
            computation = channel["computation"]
            if computation["type"] == "arithmetic":
                for op in (
                    computation["operand1"],
                    computation["operand2"],
                ):
                    if isinstance(op, str):
                        names.append(op)
                    elif isinstance(op, (int, float)):
                        pass
                    else:
                        names.extend(get_required_from_computed(op))
            elif computation["type"] == "function":
                op = computation["channel"]
                if isinstance(op, str):
                    names.append(op)
                else:
                    names.extend(get_required_from_computed(op))
            elif computation["type"] == "expression":
                expression_string = computation["expression"]
                names.extend(
                    [
                        match.group("name")
                        for match in SIG_RE.finditer(expression_string)
                    ]
                )
        else:
            names.append(channel["name"])
    else:
        if channel["type"] == "arithmetic":
            for op in (channel["operand1"], channel["operand2"]):
                if isinstance(op, str):
                    names.append(op)
                elif isinstance(op, (int, float)):
                    pass
                else:
                    names.extend(get_required_from_computed(op))
        else:
            op = channel["channel"]
            if isinstance(op, str):
                names.append(op)
            else:
                names.extend(get_required_from_computed(op))
    return names


def get_required_from_config(channels, mdf):
    required, found, not_found, computed = set(), {}, set(), []
    for channel in channels:
        if channel.get("type", "channel") == "group":
            (
                new_required,
                new_found,
                new_not_found,
                new_computed,
            ) = get_required_from_config(channel["channels"], mdf)
            required |= new_required
            found.update(new_found)
            not_found |= new_not_found
            computed.extend(new_computed)
        else:
            if channel.get("computed", False):
                computed.append(channel)
            name = channel["name"]
            required.add(name)

            if name in mdf:
                found[name] = channel
            elif not channel.get("computed", False):
                not_found.add(name)

    return required, found, not_found, computed


def set_title(mdi):
    name, ok = QtWidgets.QInputDialog.getText(
        None,
        "Set sub-plot title",
        "Title:",
    )
    if ok and name:
        mdi.setWindowTitle(generate_window_title(mdi, title=name))
        mdi.titleModified.emit()


def parse_matrix_component(name):
    indexes = []
    while True:
        match = COMPONENT.search(name)
        if match:
            name = name[: match.start()]
            indexes.insert(0, int(match.group("index")))
        else:
            break

    return name, tuple(indexes)


class MdiSubWindow(QtWidgets.QMdiSubWindow):
    sigClosed = QtCore.pyqtSignal()
    titleModified = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def closeEvent(self, event):
        super().closeEvent(event)
        self.sigClosed.emit()


class MdiAreaWidget(QtWidgets.QMdiArea):

    add_window_request = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setAcceptDrops(True)
        self.placeholder_text = "Drag and drop channels, or select channels and press the <Create window> button, to create new windows"
        self.show()

    def dragEnterEvent(self, e):
        e.accept()
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        if e.source() is self:
            super().dropEvent(e)
        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):

                def count(data):
                    s = 0
                    for (
                        name,
                        group_index,
                        channel_index,
                        mdf_uuid,
                        type_,
                        ranges,
                    ) in data:
                        if type_ == "channel":
                            s += 1
                        else:
                            s += count(channel_index)
                    return s

                names = extract_mime_names(data)

                dialog = WindowSelectionDialog(parent=self)
                dialog.setModal(True)
                dialog.exec_()

                if dialog.result():
                    window_type = dialog.selected_type()

                    if window_type == "Plot" and count(names) > 200:
                        ret = QtWidgets.QMessageBox.question(
                            self,
                            "Continue plotting large number of channels?",
                            "For optimal performance it is advised not plot more than 200 channels. "
                            f"You are attempting to plot {count(names)} channels.\n"
                            "Do you wish to continue?",
                        )

                        if ret != QtWidgets.QMessageBox.Yes:
                            return

                    self.add_window_request.emit([window_type, names])

    def tile_vertically(self):
        sub_windows = self.subWindowList()

        position = QtCore.QPoint(0, 0)

        width = self.width()
        height = self.height()
        ratio = height // len(sub_windows)

        for window in sub_windows:
            if window.isMinimized() or window.isMaximized():
                window.showNormal()
            rect = QtCore.QRect(0, 0, width, ratio)

            window.setGeometry(rect)
            window.move(position)
            position.setY(position.y() + ratio)

    def tile_horizontally(self):
        sub_windows = self.subWindowList()

        position = QtCore.QPoint(0, 0)

        width = self.width()
        height = self.height()
        ratio = width // len(sub_windows)

        for window in sub_windows:
            if window.isMinimized() or window.isMaximized():
                window.showNormal()
            rect = QtCore.QRect(0, 0, ratio, height)

            window.setGeometry(rect)
            window.move(position)
            position.setX(position.x() + ratio)

    def paintEvent(self, event):
        super().paintEvent(event)
        sub_windows = self.subWindowList()
        if not sub_windows and self.placeholder_text:
            painter = QtGui.QPainter(self.viewport())
            painter.save()
            col = self.palette().placeholderText().color()
            painter.setPen(col)
            fm = self.fontMetrics()
            elided_text = fm.elidedText(
                self.placeholder_text, QtCore.Qt.ElideRight, self.viewport().width()
            )
            painter.drawText(self.viewport().rect(), QtCore.Qt.AlignCenter, elided_text)
            painter.restore()


class WithMDIArea:

    windows_modified = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        self._cursor_source = None
        self._region_source = None
        self._splitter_source = None
        self._window_counter = 0
        self._frameless_windows = False

    def add_pattern_group(self, plot, group):

        signals = extract_signals_using_pattern(
            self.mdf,
            group.pattern,
            self.ignore_value2text_conversions,
            self.uuid,
        )

        signals = [
            sig
            for sig in signals
            if sig.samples.dtype.kind not in "SU"
            and not sig.samples.dtype.names
            and not len(sig.samples.shape) > 1
        ]

        group.count = len(signals)
        if signals:
            plot.add_new_channels(
                signals,
                mime_data=None,
                destination=group,
            )

    def add_new_channels(self, names, widget, mime_data=None):
        if isinstance(widget, Plot):
            ignore_value2text_conversions = False
            current_count = len(widget.plot.signals)
            count = len(names)
            if current_count + count > 200:
                ret = QtWidgets.QMessageBox.question(
                    self,
                    "Continue plotting large number of channels?",
                    "For optimal performance it is advised not plot more than 200 channels. "
                    f"You are attempting to add {count} new channels to a plot that already "
                    f"contains {current_count} channels.\n"
                    "Do you wish to continue?",
                )

                if ret != QtWidgets.QMessageBox.Yes:
                    return
        else:
            ignore_value2text_conversions = self.ignore_value2text_conversions

        try:
            names = list(names)
            if names and isinstance(names[0], str):
                signals_ = [
                    (name, *self.mdf.whereis(name)[0], self.uuid, "channel", [])
                    for name in names
                    if name in self.mdf
                ]

                uuids = {self.uuid}

                mime_data = signals_

            else:
                mime_data = names

                entries = get_flatten_entries_from_mime(names)
                signals_ = [name for name in entries if tuple(name[1:3]) != (-1, -1)]

                computed = [name[0] for name in entries if tuple(name[1:3]) == (-1, -1)]

                uuids = set(entry[3] for entry in entries)

            # print(computed)
            # print(names)
            # print(signals_)

            if isinstance(widget, Tabular):
                dfs = []

                for uuid in uuids:
                    uuids_signals = [
                        entry[:3] for entry in signals_ if entry[3] == uuid and entry
                    ]

                    file_info = self.file_by_uuid(uuid)
                    if not file_info:
                        continue

                    file_index, file = file_info

                    selected_signals = file.mdf.to_dataframe(
                        channels=uuids_signals,
                        ignore_value2text_conversions=self.ignore_value2text_conversions,
                        time_from_zero=False,
                    )

                    dfs.append(selected_signals)

                signals = pd.concat(dfs, axis=1)

                for name in signals.columns:
                    if name.endswith(
                        (
                            "CAN_DataFrame.ID",
                            "FLX_Frame.ID",
                            "FlexRay_DataFrame.ID",
                            "LIN_Frame.ID",
                            "MOST_DataFrame.ID",
                            "ETH_Frame.ID",
                        )
                    ):
                        signals[name] = signals[name].astype("<u4") & 0x1FFFFFFF

                widget.add_new_channels(signals)

            else:

                signals = []

                for uuid in uuids:
                    uuids_signals = [
                        entry[:3] for entry in signals_ if entry[3] == uuid
                    ]

                    file_info = self.file_by_uuid(uuid)
                    if not file_info:
                        continue

                    file_index, file = file_info

                    selected_signals = file.mdf.select(
                        uuids_signals,
                        ignore_value2text_conversions=ignore_value2text_conversions,
                        copy_master=False,
                        validate=True,
                        raw=True,
                    )

                    for sig, sig_ in zip(selected_signals, uuids_signals):
                        sig.group_index = sig_[1]
                        sig.channel_index = sig_[2]
                        sig.computed = False
                        sig.computation = {}
                        sig.mdf_uuid = uuid
                        sig.name = sig_[0]

                        if not hasattr(self, "mdf"):
                            # MainWindow => comparison plots

                            sig.tooltip = f"{sig.name}\n@ {file.file_name}"
                            sig.name = f"{file_index+1}: {sig.name}"

                    signals.extend(selected_signals)

                if isinstance(widget, Plot):
                    signals = [
                        sig
                        for sig in signals
                        if sig.samples.dtype.kind not in "SU"
                        and not sig.samples.dtype.names
                        and not len(sig.samples.shape) > 1
                    ]

                for signal in signals:
                    if len(signal.samples.shape) > 1:

                        signal.samples = csv_bytearray2hex(
                            pd.Series(list(signal.samples))
                        )

                    if signal.name.endswith("CAN_DataFrame.ID"):
                        signal.samples = signal.samples.astype("<u4") & 0x1FFFFFFF

                signals = sigs = natsorted(signals, key=lambda x: x.name)

                if isinstance(widget, Plot) and computed:
                    measured_signals = {sig.name: sig for sig in sigs}
                    if measured_signals:
                        all_timebase = np.unique(
                            np.concatenate(
                                [sig.timestamps for sig in measured_signals.values()]
                            )
                        )
                    else:
                        all_timebase = []

                    required_channels = []
                    for ch in computed:
                        required_channels.extend(get_required_from_computed(ch))

                    required_channels = set(required_channels)
                    required_channels = [
                        (None, *self.mdf.whereis(channel)[0])
                        for channel in required_channels
                        if channel not in list(measured_signals) and channel in self.mdf
                    ]
                    required_channels = {
                        sig.name: sig
                        for sig in self.mdf.select(
                            required_channels,
                            ignore_value2text_conversions=self.ignore_value2text_conversions,
                            copy_master=False,
                        )
                    }

                    required_channels.update(measured_signals)

                    computed_signals = {}

                    for channel in computed:

                        computation = channel["computation"]

                        try:

                            signal = compute_signal(
                                computation, required_channels, all_timebase
                            )
                            signal.color = channel["color"]
                            signal.computed = True
                            signal.computation = channel["computation"]
                            signal.name = channel["name"]
                            signal.unit = channel["unit"]
                            signal.group_index = -1
                            signal.channel_index = -1
                            signal.mdf_uuid = self.uuid

                            if "conversion" in channel:
                                signal.conversion = from_dict(channel["conversion"])
                                signal.name = channel["user_defined_name"]

                            computed_signals[signal.name] = signal
                        except:
                            pass
                    signals.extend(computed_signals.values())

                widget.add_new_channels(sigs, mime_data=mime_data)

        except MdfException:
            print(format_exc())

    def _add_can_bus_trace_window(self, ranges=None):
        items = []
        groups_count = len(self.mdf.groups)

        for index in range(groups_count):
            group = self.mdf.groups[index]
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                source = group.channel_group.acq_source

                names = [ch.name for ch in group.channels]

                if source and source.bus_type == v4c.BUS_TYPE_CAN:
                    if "CAN_DataFrame" in names:
                        data = self.mdf.get("CAN_DataFrame", index, raw=True)
                        items.append(data)

                    elif "CAN_RemoteFrame" in names:
                        data = self.mdf.get("CAN_RemoteFrame", index, raw=True)
                        items.append(data)

                    elif "CAN_ErrorFrame" in names:
                        data = self.mdf.get("CAN_ErrorFrame", index, raw=True)
                        items.append(data)

        if not len(items):
            return

        df_index = np.sort(np.concatenate([item.timestamps for item in items]))
        count = len(df_index)

        columns = {
            "timestamps": df_index,
            "Bus": np.full(count, "Unknown", dtype="O"),
            "ID": np.full(count, 0xFFFFFFFF, dtype="u4"),
            "Name": np.full(count, "", dtype="O"),
            "Event Type": np.full(count, "CAN Frame", dtype="O"),
            "Details": np.full(count, "", dtype="O"),
            "DLC": np.zeros(count, dtype="u1"),
            "Data Length": np.zeros(count, dtype="u1"),
            "Data Bytes": np.full(count, "", dtype="O"),
        }

        count = len(items)

        for string in v4c.CAN_ERROR_TYPES.values():
            sys.intern(string)

        for _ in range(count):
            item = items.pop()

            frame_map = None
            if item.attachment and item.attachment[0]:
                dbc = load_can_database(item.attachment[1], item.attachment[0])
                if dbc:
                    frame_map = {frame.arbitration_id.id: frame.name for frame in dbc}

                    for name in frame_map.values():
                        sys.intern(name)

            if item.name == "CAN_DataFrame":

                index = np.searchsorted(df_index, item.timestamps)

                vals = item["CAN_DataFrame.BusChannel"].astype("u1")

                vals = [f"CAN {chn}" for chn in vals.tolist()]
                columns["Bus"][index] = vals

                vals = item["CAN_DataFrame.ID"].astype("u4") & 0x1FFFFFFF
                columns["ID"][index] = vals
                if frame_map:
                    columns["Name"][index] = [frame_map[_id] for _id in vals]

                columns["DLC"][index] = item["CAN_DataFrame.DLC"].astype("u1")
                data_length = item["CAN_DataFrame.DataLength"].astype("u2").tolist()
                columns["Data Length"][index] = data_length

                vals = csv_bytearray2hex(
                    pd.Series(list(item["CAN_DataFrame.DataBytes"])),
                    data_length,
                )
                columns["Data Bytes"][index] = vals

                vals = None
                data_length = None

            elif item.name == "CAN_RemoteFrame":

                index = np.searchsorted(df_index, item.timestamps)

                vals = item["CAN_RemoteFrame.BusChannel"].astype("u1")
                vals = [f"CAN {chn}" for chn in vals.tolist()]
                columns["Bus"][index] = vals

                vals = item["CAN_RemoteFrame.ID"].astype("u4") & 0x1FFFFFFF
                columns["ID"][index] = vals
                if frame_map:
                    columns["Name"][index] = [frame_map[_id] for _id in vals]

                columns["DLC"][index] = item["CAN_RemoteFrame.DLC"].astype("u1")
                data_length = item["CAN_RemoteFrame.DataLength"].astype("u2").tolist()
                columns["Data Length"][index] = data_length
                columns["Event Type"][index] = "Remote Frame"

                vals = None
                data_length = None

            elif item.name == "CAN_ErrorFrame":

                index = np.searchsorted(df_index, item.timestamps)

                names = set(item.samples.dtype.names)

                if "CAN_ErrorFrame.BusChannel" in names:
                    vals = item["CAN_ErrorFrame.BusChannel"].astype("u1")
                    vals = [f"CAN {chn}" for chn in vals.tolist()]
                    columns["Bus"][index] = vals

                if "CAN_ErrorFrame.ID" in names:
                    vals = item["CAN_ErrorFrame.ID"].astype("u4") & 0x1FFFFFFF
                    columns["ID"][index] = vals
                    if frame_map:
                        columns["Name"][index] = [frame_map[_id] for _id in vals]

                if "CAN_ErrorFrame.DLC" in names:
                    columns["DLC"][index] = item["CAN_ErrorFrame.DLC"].astype("u1")

                if "CAN_ErrorFrame.DataLength" in names:
                    columns["Data Length"][index] = (
                        item["CAN_ErrorFrame.DataLength"].astype("u2").tolist()
                    )

                columns["Event Type"][index] = "Error Frame"

                if "CAN_ErrorFrame.ErrorType" in names:
                    vals = item["CAN_ErrorFrame.ErrorType"].astype("u1").tolist()
                    vals = [v4c.CAN_ERROR_TYPES.get(err, "Other error") for err in vals]

                    columns["Details"][index] = vals

        signals = pd.DataFrame(columns)

        trace = CANBusTrace(
            signals, start=self.mdf.header.start_time.timestamp(), ranges=ranges
        )

        sub = MdiSubWindow(parent=self)
        sub.setWidget(trace)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/bus_can.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        sub.setWindowIcon(icon)

        if not self.subplots:
            for mdi in self.mdi_area.subWindowList():
                mdi.close()
            w = self.mdi_area.addSubWindow(sub)

            w.showMaximized()
        else:
            w = self.mdi_area.addSubWindow(sub)

            if len(self.mdi_area.subWindowList()) == 1:
                w.showMaximized()
            else:
                w.show()
                self.mdi_area.tileSubWindows()

        menu = w.systemMenu()
        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.FramelessWindowHint)

        w.layout().setSpacing(1)

        def set_title(mdi):
            name, ok = QtWidgets.QInputDialog.getText(
                None, "Set sub-plot title", "Title:"
            )
            if ok and name:
                mdi.setWindowTitle(name)

        action = QtWidgets.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        w.setWindowTitle(f"CAN Bus Trace {self._window_counter}")
        self._window_counter += 1

        if self.subplots_link:
            trace.timestamp_changed_signal.connect(self.set_cursor)

        self.windows_modified.emit()
        trace.tree.auto_size_header()

        return trace

    def _add_flexray_bus_trace_window(self, ranges=None):
        items = []
        groups_count = len(self.mdf.groups)

        for index in range(groups_count):
            group = self.mdf.groups[index]
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                source = group.channel_group.acq_source

                names = [ch.name for ch in group.channels]

                if source and source.bus_type == v4c.BUS_TYPE_FLEXRAY:
                    if "FLX_Frame" in names:
                        data = self.mdf.get("FLX_Frame", index, raw=True)
                        items.append(data)

                    elif "FLX_NullFrame" in names:
                        data = self.mdf.get("FLX_NullFrame", index, raw=True)
                        items.append(data)

                    elif "FLX_StartCycle" in names:
                        data = self.mdf.get("FLX_StartCycle", index, raw=True)
                        items.append(data)

                    elif "FLX_Status" in names:
                        data = self.mdf.get("FLX_Status", index, raw=True)
                        items.append(data)

        if not len(items):
            return

        df_index = np.sort(np.concatenate([item.timestamps for item in items]))
        count = len(df_index)

        columns = {
            "timestamps": df_index,
            "Bus": np.full(count, "Unknown", dtype="O"),
            "ID": np.full(count, 0xFFFF, dtype="u2"),
            "Cycle": np.full(count, 0xFF, dtype="u1"),
            "Name": np.full(count, "", dtype="O"),
            "Event Type": np.full(count, "FlexRay Frame", dtype="O"),
            "Details": np.full(count, "", dtype="O"),
            "Data Length": np.zeros(count, dtype="u1"),
            "Data Bytes": np.full(count, "", dtype="O"),
            "Header CRC": np.full(count, 0xFFFF, dtype="u2"),
        }

        count = len(items)

        # TO DO: add flexray error types
        # for string in v4c.CAN_ERROR_TYPES.values():
        #     sys.intern(string)

        for _ in range(count):
            item = items.pop()

            frame_map = None

            # TO DO : add flexray fibex support
            # if item.attachment and item.attachment[0]:
            #     dbc = load_can_database(item.attachment[1], item.attachment[0])
            #     if dbc:
            #         frame_map = {
            #             frame.arbitration_id.id: frame.name for frame in dbc
            #         }
            #
            #         for name in frame_map.values():
            #             sys.intern(name)

            if item.name == "FLX_Frame":

                index = np.searchsorted(df_index, item.timestamps)

                vals = item["FLX_Frame.FlxChannel"].astype("u1")

                vals = [f"FlexRay {chn}" for chn in vals.tolist()]
                columns["Bus"][index] = vals

                vals = item["FLX_Frame.ID"].astype("u2")
                columns["ID"][index] = vals
                if frame_map:
                    columns["Name"][index] = [frame_map[_id] for _id in vals]

                vals = item["FLX_Frame.Cycle"].astype("u1")
                columns["Cycle"][index] = vals

                data_length = item["FLX_Frame.PayloadLength"].astype("u1").tolist()
                columns["Data Length"][index] = data_length

                vals = csv_bytearray2hex(
                    pd.Series(list(item["FLX_Frame.DataBytes"])),
                    data_length,
                )
                columns["Data Bytes"][index] = vals

                vals = item["FLX_Frame.HeaderCRC"].astype("u2")
                columns["Header CRC"][index] = vals

                vals = None
                data_length = None

            elif item.name == "FLX_NullFrame":

                index = np.searchsorted(df_index, item.timestamps)

                vals = item["FLX_NullFrame.FlxChannel"].astype("u1")
                vals = [f"FlexRay {chn}" for chn in vals.tolist()]
                columns["Bus"][index] = vals

                vals = item["FLX_NullFrame.ID"].astype("u2")
                columns["ID"][index] = vals
                if frame_map:
                    columns["Name"][index] = [frame_map[_id] for _id in vals]

                vals = item["FLX_NullFrame.Cycle"].astype("u1")
                columns["Cycle"][index] = vals

                columns["Event Type"][index] = "FlexRay NullFrame"

                vals = item["FLX_NullFrame.HeaderCRC"].astype("u2")
                columns["Header CRC"][index] = vals

                vals = None
                data_length = None

            elif item.name == "FLX_StartCycle":

                index = np.searchsorted(df_index, item.timestamps)

                vals = item["FLX_StartCycle.cycleCount"].astype("u1")
                columns["Cycle"][index] = vals

                columns["Event Type"][index] = "FlexRay StartCycle"

                vals = None
                data_length = None

            elif item.name == "FLX_Status":

                index = np.searchsorted(df_index, item.timestamps)

                vals = item["FLX_Status.StatusType"].astype("u1")
                columns["Details"][index] = vals.astype("U").astype("O")

                columns["Event Type"][index] = "FlexRay Status"

                vals = None
                data_length = None

        signals = pd.DataFrame(columns)

        trace = FlexRayBusTrace(
            signals, start=self.mdf.header.start_time.timestamp(), ranges=ranges
        )

        sub = MdiSubWindow(parent=self)
        sub.setWidget(trace)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/bus_flx.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        sub.setWindowIcon(icon)

        if not self.subplots:
            for mdi in self.mdi_area.subWindowList():
                mdi.close()
            w = self.mdi_area.addSubWindow(sub)

            w.showMaximized()
        else:
            w = self.mdi_area.addSubWindow(sub)

            if len(self.mdi_area.subWindowList()) == 1:
                w.showMaximized()
            else:
                w.show()
                self.mdi_area.tileSubWindows()

        menu = w.systemMenu()
        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.FramelessWindowHint)

        w.layout().setSpacing(1)

        def set_title(mdi):
            name, ok = QtWidgets.QInputDialog.getText(
                None, "Set sub-plot title", "Title:"
            )
            if ok and name:
                mdi.setWindowTitle(name)

        action = QtWidgets.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        w.setWindowTitle(f"FlexRay Bus Trace {self._window_counter}")
        self._window_counter += 1

        if self.subplots_link:
            trace.timestamp_changed_signal.connect(self.set_cursor)

        self.windows_modified.emit()
        trace.tree.auto_size_header()

        return trace

    def _add_lin_bus_trace_window(self, ranges=None):
        items = []
        groups_count = len(self.mdf.groups)

        for index in range(groups_count):
            group = self.mdf.groups[index]
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                source = group.channel_group.acq_source

                names = [ch.name for ch in group.channels]

                if source and source.bus_type == v4c.BUS_TYPE_LIN:
                    if "LIN_Frame" in names:
                        data = self.mdf.get("LIN_Frame", index, raw=True)
                        items.append(data)

                    elif "LIN_SyncError" in names:
                        data = self.mdf.get("LIN_SyncError", index, raw=True)
                        items.append(data)

                    elif "LIN_TransmissionError" in names:
                        data = self.mdf.get("LIN_TransmissionError", index, raw=True)
                        items.append(data)

                    elif "LIN_ChecksumError" in names:
                        data = self.mdf.get("LIN_ChecksumError", index, raw=True)
                        items.append(data)

                    elif "LIN_ReceiveError" in names:
                        data = self.mdf.get("LIN_ReceiveError", index, raw=True)
                        items.append(data)

        if not len(items):
            return

        df_index = np.sort(np.concatenate([item.timestamps for item in items]))
        count = len(df_index)

        columns = {
            "timestamps": df_index,
            "Bus": np.full(count, "Unknown", dtype="O"),
            "ID": np.full(count, 0xFFFFFFFF, dtype="u4"),
            "Name": np.full(count, "", dtype="O"),
            "Event Type": np.full(count, "LIN Frame", dtype="O"),
            "Details": np.full(count, "", dtype="O"),
            "Received Byte Count": np.zeros(count, dtype="u1"),
            "Data Length": np.zeros(count, dtype="u1"),
            "Data Bytes": np.full(count, "", dtype="O"),
        }

        count = len(items)

        for _ in range(count):
            item = items.pop()

            frame_map = None
            if item.attachment and item.attachment[0]:
                dbc = load_can_database(item.attachment[1], item.attachment[0])
                if dbc:
                    frame_map = {frame.arbitration_id.id: frame.name for frame in dbc}

                    for name in frame_map.values():
                        sys.intern(name)

            if item.name == "LIN_Frame":

                index = np.searchsorted(df_index, item.timestamps)

                vals = item["LIN_Frame.BusChannel"].astype("u1")
                vals = [f"LIN {chn}" for chn in vals.tolist()]
                columns["Bus"][index] = vals

                vals = item["LIN_Frame.ID"].astype("u1") & 0x3F
                columns["ID"][index] = vals
                if frame_map:
                    columns["Name"][index] = [frame_map[_id] for _id in vals]

                columns["Received Byte Count"][index] = item[
                    "LIN_Frame.ReceivedDataByteCount"
                ].astype("u1")
                data_length = item["LIN_Frame.DataLength"].astype("u1").tolist()
                columns["Data Length"][index] = data_length

                vals = csv_bytearray2hex(
                    pd.Series(list(item["LIN_Frame.DataBytes"])),
                    data_length,
                )
                columns["Data Bytes"][index] = vals

                vals = None
                data_length = None

            elif item.name == "LIN_SyncError":

                index = np.searchsorted(df_index, item.timestamps)
                names = set(item.samples.dtype.names)

                if "LIN_SyncError.BusChannel" in names:
                    vals = item["LIN_SyncError.BusChannel"].astype("u1")
                    vals = [f"LIN {chn}" for chn in vals.tolist()]
                    columns["Bus"][index] = vals

                if "LIN_SyncError.BaudRate" in names:
                    vals = item["LIN_SyncError.BaudRate"]
                    unique = np.unique(vals).tolist()
                    for val in unique:
                        sys.intern((f"Baudrate {val}"))
                    vals = [f"Baudrate {val}" for val in vals.tolist()]
                    columns["Details"][index] = vals

                columns["Event Type"][index] = "Sync Error Frame"

                vals = None
                data_length = None

            elif item.name == "LIN_TransmissionError":

                index = np.searchsorted(df_index, item.timestamps)

                names = set(item.samples.dtype.names)

                if "LIN_TransmissionError.BusChannel" in names:
                    vals = item["LIN_TransmissionError.BusChannel"].astype("u1")
                    vals = [f"LIN {chn}" for chn in vals.tolist()]
                    columns["Bus"][index] = vals

                if "LIN_TransmissionError.BaudRate" in names:
                    vals = item["LIN_TransmissionError.BaudRate"]
                    unique = np.unique(vals).tolist()
                    for val in unique:
                        sys.intern((f"Baudrate {val}"))
                    vals = [f"Baudrate {val}" for val in vals.tolist()]
                    columns["Details"][index] = vals

                vals = item["LIN_TransmissionError.ID"].astype("u1") & 0x3F
                columns["ID"][index] = vals
                if frame_map:
                    columns["Name"][index] = [frame_map[_id] for _id in vals]

                columns["Event Type"][index] = "Transmission Error Frame"

                vals = None

            elif item.name == "LIN_ReceiveError":

                index = np.searchsorted(df_index, item.timestamps)

                names = set(item.samples.dtype.names)

                if "LIN_ReceiveError.BusChannel" in names:
                    vals = item["LIN_ReceiveError.BusChannel"].astype("u1")
                    vals = [f"LIN {chn}" for chn in vals.tolist()]
                    columns["Bus"][index] = vals

                if "LIN_ReceiveError.BaudRate" in names:
                    vals = item["LIN_ReceiveError.BaudRate"]
                    unique = np.unique(vals).tolist()
                    for val in unique:
                        sys.intern((f"Baudrate {val}"))
                    vals = [f"Baudrate {val}" for val in vals.tolist()]
                    columns["Details"][index] = vals

                if "LIN_ReceiveError.ID" in names:
                    vals = item["LIN_ReceiveError.ID"].astype("u1") & 0x3F
                    columns["ID"][index] = vals
                    if frame_map:
                        columns["Name"][index] = [frame_map[_id] for _id in vals]

                columns["Event Type"][index] = "Receive Error Frame"

                vals = None

            elif item.name == "LIN_ChecksumError":

                index = np.searchsorted(df_index, item.timestamps)

                names = set(item.samples.dtype.names)

                if "LIN_ChecksumError.BusChannel" in names:
                    vals = item["LIN_ChecksumError.BusChannel"].astype("u1")
                    vals = [f"LIN {chn}" for chn in vals.tolist()]
                    columns["Bus"][index] = vals

                if "LIN_ChecksumError.Checksum" in names:
                    vals = item["LIN_ChecksumError.Checksum"]
                    unique = np.unique(vals).tolist()
                    for val in unique:
                        sys.intern((f"Baudrate {val}"))
                    vals = [f"Checksum 0x{val:02X}" for val in vals.tolist()]
                    columns["Details"][index] = vals

                if "LIN_ChecksumError.ID" in names:
                    vals = item["LIN_ChecksumError.ID"].astype("u1") & 0x3F
                    columns["ID"][index] = vals
                    if frame_map:
                        columns["Name"][index] = [frame_map[_id] for _id in vals]

                columns["Event Type"][index] = "Checksum Error Frame"

                vals = None

        signals = pd.DataFrame(columns)

        trace = LINBusTrace(
            signals, start=self.mdf.header.start_time.timestamp(), range=ranges
        )

        sub = MdiSubWindow(parent=self)
        sub.setWidget(trace)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/bus_lin.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        sub.setWindowIcon(icon)

        if not self.subplots:
            for mdi in self.mdi_area.subWindowList():
                mdi.close()
            w = self.mdi_area.addSubWindow(sub)

            w.showMaximized()
        else:
            w = self.mdi_area.addSubWindow(sub)

            if len(self.mdi_area.subWindowList()) == 1:
                w.showMaximized()
            else:
                w.show()
                self.mdi_area.tileSubWindows()

        menu = w.systemMenu()
        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.FramelessWindowHint)

        w.layout().setSpacing(1)

        def set_title(mdi):
            name, ok = QtWidgets.QInputDialog.getText(
                None, "Set sub-plot title", "Title:"
            )
            if ok and name:
                mdi.setWindowTitle(name)

        action = QtWidgets.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        w.setWindowTitle(f"LIN Bus Trace {self._window_counter}")
        self._window_counter += 1

        if self.subplots_link:
            trace.timestamp_changed_signal.connect(self.set_cursor)

        self.windows_modified.emit()
        trace.tree.auto_size_header()

        return trace

    def _add_gps_window(self, signals):

        signals = [sig[:3] for sig in signals]
        latitude_channel, longitude_channel = self.mdf.select(signals)

        gps = GPS(latitude_channel, longitude_channel)
        sub = MdiSubWindow(parent=self)
        sub.setWidget(gps)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/globe.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        sub.setWindowIcon(icon)

        w = self.mdi_area.addSubWindow(sub)

        if len(self.mdi_area.subWindowList()) == 1:
            w.showMaximized()
        else:
            w.show()
            self.mdi_area.tileSubWindows()

        menu = w.systemMenu()
        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.FramelessWindowHint)

        w.layout().setSpacing(1)

        def set_title(mdi):
            name, ok = QtWidgets.QInputDialog.getText(
                None, "Set sub-plot title", "Title:"
            )
            if ok and name:
                mdi.setWindowTitle(name)

        action = QtWidgets.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        w.setWindowTitle(f"GPS {self._window_counter}")
        self._window_counter += 1

        if self.subplots_link:
            gps.timestamp_changed_signal.connect(self.set_cursor)

        self.windows_modified.emit()

    def add_window(self, args):
        window_type, names = args

        if window_type == "CAN Bus Trace":
            return self._add_can_bus_trace_window()
        elif window_type == "FlexRay Bus Trace":
            return self._add_flexray_bus_trace_window()
        elif window_type == "LIN Bus Trace":
            return self._add_lin_bus_trace_window()
        elif window_type == "GPS":
            return self._add_gps_window(names)

        if names and isinstance(names[0], str):
            signals_ = [
                (name, *self.mdf.whereis(name)[0], self.uuid, "channel")
                for name in names
                if name in self.mdf
            ]
            computed = []
        else:
            flatten_entries = get_flatten_entries_from_mime(names)
            signals_ = [
                entry for entry in flatten_entries if tuple(entry[1:3]) != (-1, -1)
            ]

            computed = [
                entry[0] for entry in flatten_entries if tuple(entry[1:3]) == (-1, -1)
            ]

        if window_type == "Tabular":
            uuids = set(entry[3] for entry in signals_)

            dfs = []
            start = []

            for uuid in uuids:
                uuids_signals = [entry[:3] for entry in signals_ if entry[3] == uuid]

                file_info = self.file_by_uuid(uuid)
                if not file_info:
                    continue

                file_index, file = file_info
                start.append(file.mdf.header.start_time.timestamp())

                uuids_signals = [
                    entry
                    for entry in uuids_signals
                    if entry[2] != file.mdf.masters_db.get(entry[1], None)
                ]

                df = file.mdf.to_dataframe(
                    channels=uuids_signals,
                    ignore_value2text_conversions=self.ignore_value2text_conversions,
                    time_from_zero=False,
                    empty_channels="zeros",
                )

                if not hasattr(self, "mdf"):
                    # MainWindow => comparison plots
                    columns = {name: f"{file_index+1}: {name}" for name in df.columns}
                    df.rename(columns=columns, inplace=True)

                dfs.append(df)

            if not dfs:
                return

            signals = pd.concat(dfs, axis=1)
            start = min(start)

            for name in signals.columns:
                if name.endswith(
                    (
                        "CAN_DataFrame.ID",
                        "FLX_Frame.ID",
                        "FlexRay_DataFrame.ID",
                        "LIN_Frame.ID",
                        "MOST_DataFrame.ID",
                        "ETH_Frame.ID",
                    )
                ):
                    signals[name] = signals[name].astype("<u4") & 0x1FFFFFFF

        else:

            uuids = set(entry[3] for entry in signals_)

            signals = []

            for uuid in uuids:
                uuids_signals = [entry[:3] for entry in signals_ if entry[3] == uuid]

                file_info = self.file_by_uuid(uuid)
                if not file_info:
                    continue

                file_index, file = file_info

                selected_signals = file.mdf.select(
                    uuids_signals,
                    ignore_value2text_conversions=self.ignore_value2text_conversions,
                    copy_master=False,
                    validate=True,
                    raw=True,
                )

                for sig, sig_ in zip(selected_signals, uuids_signals):
                    sig.group_index = sig_[1]
                    sig.channel_index = sig_[2]
                    sig.computed = False
                    sig.computation = {}
                    sig.mdf_uuid = uuid
                    sig.name = sig_[0] or sig.name

                    if not hasattr(self, "mdf"):
                        # MainWindow => comparison plots

                        sig.tooltip = f"{sig.name}\n@ {file.file_name}"
                        sig.name = f"{file_index+1}: {sig.name}"

                signals.extend(selected_signals)

            if window_type == "Plot":
                nd = [
                    sig
                    for sig in signals
                    if sig.samples.dtype.kind not in "SU"
                    and (sig.samples.dtype.names or len(sig.samples.shape) > 1)
                ]

                signals = [
                    sig
                    for sig in signals
                    if sig.samples.dtype.kind not in "SU"
                    and not sig.samples.dtype.names
                    and not len(sig.samples.shape) > 1
                ]

                for sig in nd:
                    if sig.samples.dtype.names is None:
                        shape = sig.samples.shape[1:]

                        matrix_dims = [list(range(dim)) for dim in shape]

                        matrix_name = sig.name

                        for indexes in itertools.product(*matrix_dims):
                            indexes_string = "".join(
                                f"[{_index}]" for _index in indexes
                            )

                            samples = sig.samples
                            for idx in indexes:
                                samples = samples[:, idx]
                            sig_name = f"{matrix_name}{indexes_string}"

                            new_sig = sig.copy()
                            new_sig.name = sig_name
                            new_sig.samples = samples
                            new_sig.group_index = sig.group_index
                            new_sig.channel_index = sig.channel_index
                            new_sig.computed = False
                            new_sig.computation = {}
                            new_sig.mdf_uuid = sig.mdf_uuid

                            signals.append(new_sig)
                    else:
                        name = sig.samples.dtype.names[0]
                        if name == sig.name:
                            array_samples = sig.samples[name]

                            shape = array_samples.shape[1:]

                            matrix_dims = [list(range(dim)) for dim in shape]

                            matrix_name = sig.name

                            for indexes in itertools.product(*matrix_dims):
                                indexes_string = "".join(
                                    f"[{_index}]" for _index in indexes
                                )

                                samples = array_samples
                                for idx in indexes:
                                    samples = samples[:, idx]
                                sig_name = f"{matrix_name}{indexes_string}"

                                new_sig = sig.copy()
                                new_sig.name = sig_name
                                new_sig.samples = samples
                                new_sig.group_index = sig.group_index
                                new_sig.channel_index = sig.channel_index
                                new_sig.computed = False
                                new_sig.computation = {}
                                new_sig.mdf_uuid = sig.mdf_uuid

                                signals.append(new_sig)

            elif window_type == "Numeric":
                for (
                    name,
                    pattern_info,
                    channels,
                    mdf_uuid,
                    type_,
                    ranges,
                ) in get_pattern_groups(names):

                    file_info = self.file_by_uuid(mdf_uuid)
                    if not file_info:
                        continue

                    file_index, file = file_info

                    signals.extend(
                        extract_signals_using_pattern(
                            file.mdf,
                            pattern_info,
                            file.ignore_value2text_conversions,
                            file.uuid,
                        )
                    )

            for signal in signals:
                if len(signal.samples.shape) > 1:
                    if signal.name.endswith(".DataBytes"):
                        length_name = signal.name.replace(".DataBytes", ".DataLength")
                        for s in signals:
                            if s.name == length_name:
                                length = s.samples
                                break
                        else:
                            if length_name in self.mdf:
                                length = self.mdf.get(length_name, samples_only=True)[0]
                            else:
                                length = None
                    else:
                        length = None
                    signal.samples = csv_bytearray2hex(
                        pd.Series(list(signal.samples)), length
                    )

                if signal.name.endswith("CAN_DataFrame.ID"):
                    signal.samples = signal.samples.astype("<u4") & 0x1FFFFFFF

            signals = natsorted(signals, key=lambda x: x.name)

        if window_type == "Numeric":
            numeric = Numeric([], parent=self, mode="offline")

            numeric.show()
            numeric.hide()

            sub = MdiSubWindow(parent=self)
            sub.setWidget(numeric)
            sub.sigClosed.connect(self.window_closed_handler)
            sub.titleModified.connect(self.window_closed_handler)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(sub)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(sub)

                if len(self.mdi_area.subWindowList()) == 1:
                    w.showMaximized()
                else:
                    w.show()
                    self.mdi_area.tileSubWindows()

            if self._frameless_windows:
                w.setWindowFlags(w.windowFlags() | QtCore.Qt.FramelessWindowHint)

            w.layout().setSpacing(1)

            menu = w.systemMenu()

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)

            w.setWindowTitle(generate_window_title(w, window_type))

            numeric.add_channels_request.connect(
                partial(self.add_new_channels, widget=numeric)
            )
            if self.subplots_link:
                numeric.timestamp_changed_signal.connect(self.set_cursor)

            numeric.add_new_channels(signals)
            numeric.show()

        elif window_type == "Bar":
            bar = Bar(signals)

            sub = MdiSubWindow(parent=self)
            sub.setWidget(bar)
            sub.sigClosed.connect(self.window_closed_handler)
            sub.titleModified.connect(self.window_closed_handler)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(sub)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(sub)

                if len(self.mdi_area.subWindowList()) == 1:
                    w.showMaximized()
                else:
                    w.show()
                    self.mdi_area.tileSubWindows()

            if self._frameless_windows:
                w.setWindowFlags(w.windowFlags() | QtCore.Qt.FramelessWindowHint)

            w.layout().setSpacing(1)

            menu = w.systemMenu()

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)

            w.setWindowTitle(generate_window_title(w, window_type))

            bar.add_channels_request.connect(partial(self.add_new_channels, widget=bar))
            if self.subplots_link:
                bar.timestamp_changed_signal.connect(self.set_cursor)

        elif window_type == "Plot":
            mime_data = names
            if hasattr(self, "mdf"):
                events = []
                origin = self.mdf.start_time

                if self.mdf.version >= "4.00":
                    mdf_events = list(self.mdf.events)

                    for pos, event in enumerate(mdf_events):
                        event_info = {}
                        event_info["value"] = event.value
                        event_info["type"] = v4c.EVENT_TYPE_TO_STRING[event.event_type]
                        description = event.name
                        if event.comment:
                            try:
                                comment = extract_cncomment_xml(event.comment)
                            except:
                                comment = event.comment
                            description += f" ({comment})"
                        event_info["description"] = description
                        event_info["index"] = pos

                        if event.range_type == v4c.EVENT_RANGE_TYPE_POINT:
                            events.append(event_info)
                        elif event.range_type == v4c.EVENT_RANGE_TYPE_BEGINNING:
                            events.append([event_info])
                        else:
                            if event.parent is not None:
                                parent = events[event.parent]
                                parent.append(event_info)
                            events.append(None)
                    events = [ev for ev in events if ev is not None]
                else:
                    for gp in self.mdf.groups:
                        if not gp.trigger:
                            continue

                        for i in range(gp.trigger.trigger_events_nr):
                            event = {
                                "value": gp.trigger[f"trigger_{i}_time"],
                                "index": i,
                                "description": gp.trigger.comment,
                                "type": v4c.EVENT_TYPE_TO_STRING[
                                    v4c.EVENT_TYPE_TRIGGER
                                ],
                            }
                            events.append(event)
            else:
                events = []
                origin = self.files.widget(0).mdf.start_time

            if hasattr(self, "mdf"):
                mdf = self.mdf
            else:
                mdf = None
            plot = Plot(
                [],
                events=events,
                with_dots=self.with_dots,
                line_interconnect=self.line_interconnect,
                origin=origin,
                mdf=mdf,
                parent=self,
                hide_missing_channels=self.hide_missing_channels,
                hide_disabled_channels=self.hide_disabled_channels,
            )
            plot.pattern_group_added.connect(self.add_pattern_group)
            plot.pattern = {}

            sub = MdiSubWindow(parent=self)
            sub.setWidget(plot)
            sub.sigClosed.connect(self.window_closed_handler)
            sub.titleModified.connect(self.window_closed_handler)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(sub)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(sub)

                if len(self.mdi_area.subWindowList()) == 1:
                    w.showMaximized()
                else:
                    w.show()
                    self.mdi_area.tileSubWindows()

            if self._frameless_windows:
                w.setWindowFlags(w.windowFlags() | QtCore.Qt.FramelessWindowHint)

            w.layout().setSpacing(1)

            plot.show()
            plot.hide()

            menu = w.systemMenu()

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)

            w.setWindowTitle(generate_window_title(w, window_type))

            if self.subplots_link:

                for i, mdi in enumerate(self.mdi_area.subWindowList()):
                    try:
                        viewbox = mdi.widget().plot.viewbox
                        if plot.plot.viewbox is not viewbox:
                            plot.plot.viewbox.setXLink(viewbox)
                        break
                    except:
                        continue

            plot.add_channels_request.connect(
                partial(self.add_new_channels, widget=plot)
            )

            plot.show_properties.connect(self._show_info)
            # TO DO plot.channel_selection.setCurrentRow(0)

            plot.add_new_channels(signals, mime_data)

            if computed:
                measured_signals = {sig.name: sig for sig in signals}
                if measured_signals:
                    all_timebase = np.unique(
                        np.concatenate(
                            [sig.timestamps for sig in measured_signals.values()]
                        )
                    )
                else:
                    all_timebase = []

                required_channels = []
                for ch in computed:
                    required_channels.extend(get_required_from_computed(ch))

                required_channels = set(required_channels)
                required_channels = [
                    (None, *self.mdf.whereis(channel)[0])
                    for channel in required_channels
                    if channel not in list(measured_signals) and channel in self.mdf
                ]
                required_channels = {
                    sig.name: sig
                    for sig in self.mdf.select(
                        required_channels,
                        ignore_value2text_conversions=self.ignore_value2text_conversions,
                        copy_master=False,
                    )
                }

                required_channels.update(measured_signals)

                computed_signals = {}

                for channel in computed:
                    computation = channel["computation"]

                    try:

                        signal = compute_signal(
                            computation, required_channels, all_timebase
                        )
                        signal.color = channel["color"]
                        signal.computed = True
                        signal.computation = channel["computation"]
                        signal.name = channel["name"]
                        signal.unit = channel["unit"]
                        signal.group_index = -1
                        signal.channel_index = -1

                        computed_signals[signal.name] = signal
                    except:
                        pass
                signals = list(computed_signals.values())
                plot.add_new_channels(signals)

            plot.show()
            self.set_subplots_link(self.subplots_link)

            iterator = QtWidgets.QTreeWidgetItemIterator(plot.channel_selection)
            while iterator.value():
                item = iterator.value()
                iterator += 1

                if isinstance(item, ChannelsGroupTreeItem):
                    if item.pattern:
                        plot.pattern_group_added.emit(plot, item)

        elif window_type == "Tabular":
            tabular = Tabular(signals, start=start, parent=self)

            sub = MdiSubWindow(parent=self)
            sub.setWidget(tabular)
            sub.sigClosed.connect(self.window_closed_handler)
            sub.titleModified.connect(self.window_closed_handler)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(sub)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(sub)

                if len(self.mdi_area.subWindowList()) == 1:
                    w.showMaximized()
                else:
                    w.show()
                    self.mdi_area.tileSubWindows()

            menu = w.systemMenu()
            if self._frameless_windows:
                w.setWindowFlags(w.windowFlags() | QtCore.Qt.FramelessWindowHint)

            w.layout().setSpacing(1)

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)

            w.setWindowTitle(generate_window_title(w, window_type))

            if self.subplots_link:
                tabular.timestamp_changed_signal.connect(self.set_cursor)

            tabular.add_channels_request.connect(
                partial(self.add_new_channels, widget=tabular)
            )

            tabular.tree.auto_size_header()

        self.windows_modified.emit()

    def get_current_widget(self):
        mdi = self.mdi_area.activeSubWindow()
        if mdi is not None:
            widget = mdi.widget()

            return widget
        else:
            return None

    def load_window(self, window_info):

        uuid = self.uuid
        geometry = window_info.get("geometry", None)

        if window_info["type"] == "Numeric":
            # patterns
            pattern_info = window_info["configuration"].get("pattern", {})
            if pattern_info:
                required = set()
                found_signals = []
                fmt = "phys"

                signals = extract_signals_using_pattern(
                    self.mdf,
                    pattern_info,
                    self.ignore_value2text_conversions,
                    self.uuid,
                )

                for sig in signals:
                    sig.mdf_uuid = uuid
                    sig.computation = None

                try:
                    ranges = [
                        {
                            "font_color": range["color"],
                            "background_color": range["color"],
                            "op1": "<=",
                            "op2": "<=",
                            "value1": float(range["start"]),
                            "value2": float(range["stop"]),
                        }
                        for range in pattern_info["ranges"]
                    ]
                except KeyError:
                    ranges = pattern_info["ranges"]

                for range in ranges:
                    range["font_color"] = QtGui.QBrush(
                        QtGui.QColor(range["font_color"])
                    )
                    range["background_color"] = QtGui.QBrush(
                        QtGui.QColor(range["background_color"])
                    )

                pattern_info["ranges"] = ranges

            else:

                required = set(window_info["configuration"]["channels"])

                signals_ = [
                    (name, *self.mdf.whereis(name)[0])
                    for name in window_info["configuration"]["channels"]
                    if name in self.mdf
                ]

                if window_info["configuration"].get("ranges", []):
                    ranges = [
                        range
                        for name, range in zip(
                            window_info["configuration"]["channels"],
                            window_info["configuration"]["ranges"],
                        )
                        if name in self.mdf
                    ]

                    for channel_ranges in ranges:
                        for range in channel_ranges:
                            range["font_color"] = QtGui.QBrush(
                                QtGui.QColor(range["font_color"])
                            )
                            range["background_color"] = QtGui.QBrush(
                                QtGui.QColor(range["background_color"])
                            )
                else:
                    ranges = [
                        []
                        for name in window_info["configuration"]["channels"]
                        if name in self.mdf
                    ]

                if not signals_:
                    return

                signals = self.mdf.select(
                    signals_,
                    ignore_value2text_conversions=self.ignore_value2text_conversions,
                    copy_master=False,
                    validate=True,
                    raw=True,
                )

                for sig, sig_, channel_ranges in zip(signals, signals_, ranges):
                    sig.group_index = sig_[1]
                    sig.mdf_uuid = uuid
                    sig.computation = None
                    sig.ranges = channel_ranges

                signals = [
                    sig
                    for sig in signals
                    if not sig.samples.dtype.names and len(sig.samples.shape) <= 1
                ]

                signals = natsorted(signals, key=lambda x: x.name)

                found = set(sig.name for sig in signals)
                not_found = [
                    Signal([], [], name=name) for name in sorted(required - found)
                ]
                uuid = os.urandom(6).hex()
                for sig in not_found:
                    sig.mdf_uuid = uuid
                    sig.group_index = 0

                signals.extend(not_found)

            numeric = Numeric(
                [],
                format=window_info["configuration"]["format"],
                float_precision=window_info["configuration"].get("float_precision", 3),
                parent=self,
                mode="offline",
            )
            numeric.pattern = pattern_info

            sub = MdiSubWindow(parent=self)
            sub.setWidget(numeric)
            sub.sigClosed.connect(self.window_closed_handler)
            sub.titleModified.connect(self.window_closed_handler)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(sub)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(sub)
                w.show()

                if geometry:
                    w.setGeometry(*geometry)
                else:
                    self.mdi_area.tileSubWindows()

            w.setWindowTitle(
                generate_window_title(w, window_info["type"], window_info["title"])
            )

            numeric.add_new_channels(signals)

            menu = w.systemMenu()

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)

            numeric.add_channels_request.connect(
                partial(self.add_new_channels, widget=numeric)
            )

            if self.subplots_link:
                numeric.timestamp_changed_signal.connect(self.set_cursor)

            sections_width = window_info["configuration"].get(
                "header_sections_width", []
            )
            if sections_width:
                sections_width = reversed(
                    [(i, width) for i, width in enumerate(sections_width)]
                )
                for column_index, width in sections_width:
                    numeric.channels.columnHeader.setColumnWidth(column_index, width)
                    numeric.channels.dataView.setColumnWidth(
                        column_index,
                        numeric.channels.columnHeader.columnWidth(column_index),
                    )

        elif window_info["type"] == "GPS":
            signals_ = [
                (None, *self.mdf.whereis(name)[0])
                for name in (
                    window_info["configuration"]["latitude_channel"],
                    window_info["configuration"]["longitude_channel"],
                )
                if name in self.mdf
            ]

            if len(signals_) != 2:
                return

            latitude, longitude = self.mdf.select(
                signals_,
                copy_master=False,
                validate=True,
                raw=False,
            )

            gps = GPS(latitude, longitude, window_info["configuration"]["zoom"])

            sub = MdiSubWindow(parent=self)
            sub.setWidget(gps)
            sub.sigClosed.connect(self.window_closed_handler)
            sub.titleModified.connect(self.window_closed_handler)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(sub)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(sub)
                w.show()

                if geometry:
                    w.setGeometry(*geometry)
                else:
                    self.mdi_area.tileSubWindows()

            w.setWindowTitle(
                generate_window_title(w, window_info["type"], window_info["title"])
            )

            gps._update_values()

            menu = w.systemMenu()

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)

            if self.subplots_link:
                gps.timestamp_changed_signal.connect(self.set_cursor)

        elif window_info["type"] == "Plot":
            # patterns
            pattern_info = window_info["configuration"].get("pattern", {})
            if pattern_info:
                required = set()
                found_signals = []

                signals = extract_signals_using_pattern(
                    self.mdf,
                    pattern_info,
                    self.ignore_value2text_conversions,
                    self.uuid,
                )

                mime_data = None

                found_descriptions = {}

            else:

                (
                    required,
                    found_signals,
                    not_found,
                    computed_signals_descriptions,
                ) = get_required_from_config(
                    window_info["configuration"]["channels"], self.mdf
                )
                mime_data = build_mime_from_config(
                    window_info["configuration"]["channels"], self.mdf, self.uuid
                )

                measured_signals_ = [
                    (name, *self.mdf.whereis(name)[0]) for name in found_signals
                ]

                found_descriptions = found_signals
                found_signals = list(found_signals.values())

                measured_signals = {
                    sig_info[0]: sig
                    for sig, sig_info in zip(
                        self.mdf.select(
                            measured_signals_,
                            ignore_value2text_conversions=self.ignore_value2text_conversions,
                            copy_master=False,
                            validate=True,
                            raw=True,
                        ),
                        measured_signals_,
                    )
                }

                for signal, entry_info, channel in zip(
                    measured_signals.values(), measured_signals_, found_signals
                ):
                    signal.computed = False
                    signal.computation = {}
                    signal.color = channel["color"]
                    signal.group_index = entry_info[1]
                    signal.channel_index = entry_info[2]
                    signal.mdf_uuid = uuid
                    signal.name = entry_info[0]

                matrix_components = []
                for name in not_found:
                    name, indexes = parse_matrix_component(name)
                    if indexes and name in self.mdf:
                        matrix_components.append((name, indexes))

                matrix_signals = {
                    str(matrix_element): sig
                    for sig, matrix_element in zip(
                        self.mdf.select(
                            [el[0] for el in matrix_components],
                            ignore_value2text_conversions=self.ignore_value2text_conversions,
                            copy_master=False,
                        ),
                        matrix_components,
                    )
                }

                new_matrix_signals = {}
                for signal_mat, (_n, indexes) in zip(
                    matrix_signals.values(), matrix_components
                ):
                    signal = deepcopy(signal_mat)
                    signal.computed = False
                    signal.computation = {}
                    signal.group_index, signal.channel_index = self.mdf.whereis(
                        signal.name
                    )[0]

                    indexes_string = "".join(f"[{_index}]" for _index in indexes)

                    samples = signal.samples
                    if samples.dtype.names:
                        samples = samples[signal.name]

                    for idx in indexes:
                        samples = samples[:, idx]
                    sig_name = f"{signal.name}{indexes_string}"
                    signal.name = sig_name
                    signal.samples = samples

                    new_matrix_signals[signal.name] = signal

                measured_signals.update(
                    {name: sig for name, sig in new_matrix_signals.items()}
                )

                if measured_signals:
                    all_timebase = np.unique(
                        np.concatenate(
                            [sig.timestamps for sig in measured_signals.values()]
                        )
                    )
                else:
                    all_timebase = []

                required_channels = []
                for ch in computed_signals_descriptions:
                    required_channels.extend(get_required_from_computed(ch))

                required_channels = set(required_channels)
                required_channels = [
                    (None, *self.mdf.whereis(channel)[0])
                    for channel in required_channels
                    if channel not in list(measured_signals) and channel in self.mdf
                ]
                required_channels = {
                    sig.name: sig
                    for sig in self.mdf.select(
                        required_channels,
                        ignore_value2text_conversions=self.ignore_value2text_conversions,
                        copy_master=False,
                    )
                }

                required_channels.update(measured_signals)

                computed_signals = {}

                for channel in computed_signals_descriptions:
                    computation = channel["computation"]

                    try:

                        signal = compute_signal(
                            computation, required_channels, all_timebase
                        )
                        signal.color = channel["color"]
                        signal.computed = True
                        signal.computation = channel["computation"]
                        signal.name = channel["name"]
                        signal.unit = channel["unit"]
                        signal.group_index = -1
                        signal.channel_index = -1
                        signal.mdf_uuid = uuid

                        if "conversion" in channel:
                            signal.conversion = from_dict(channel["conversion"])
                            signal.name = channel["user_defined_name"]

                        computed_signals[signal.name] = signal
                    except:
                        pass

                signals = list(measured_signals.values()) + list(
                    computed_signals.values()
                )

            signals = [
                sig
                for sig in signals
                if sig.samples.dtype.kind not in "SU"
                and not sig.samples.dtype.names
                and not len(sig.samples.shape) > 1
            ]

            if hasattr(self, "mdf"):
                events = []
                origin = self.mdf.start_time

                if self.mdf.version >= "4.00":
                    mdf_events = list(self.mdf.events)

                    for pos, event in enumerate(mdf_events):
                        event_info = {}
                        event_info["value"] = event.value
                        event_info["type"] = v4c.EVENT_TYPE_TO_STRING[event.event_type]
                        description = event.name
                        if event.comment:
                            try:
                                comment = extract_cncomment_xml(event.comment)
                            except:
                                comment = event.comment
                            description += f" ({comment})"
                        event_info["description"] = description
                        event_info["index"] = pos

                        if event.range_type == v4c.EVENT_RANGE_TYPE_POINT:
                            events.append(event_info)
                        elif event.range_type == v4c.EVENT_RANGE_TYPE_BEGINNING:
                            events.append([event_info])
                        else:
                            parent = events[event.parent]
                            parent.append(event_info)
                            events.append(None)
                    events = [ev for ev in events if ev is not None]
                else:
                    for gp in self.mdf.groups:
                        if not gp.trigger:
                            continue

                        for i in range(gp.trigger.trigger_events_nr):
                            event = {
                                "value": gp.trigger[f"trigger_{i}_time"],
                                "index": i,
                                "description": gp.trigger.comment,
                                "type": v4c.EVENT_TYPE_TO_STRING[
                                    v4c.EVENT_TYPE_TRIGGER
                                ],
                            }
                            events.append(event)
            else:
                events = []
                origin = self.files.widget(0).mdf.start_time

            found = set(sig.name for sig in signals)
            not_found = [Signal([], [], name=name) for name in sorted(required - found)]
            uuid = os.urandom(6).hex()
            for sig in not_found:
                sig.mdf_uuid = uuid
                sig.group_index = NOT_FOUND

            signals.extend(not_found)

            if hasattr(self, "mdf"):
                mdf = self.mdf
            else:
                mdf = None
            plot = Plot(
                [],
                with_dots=self.with_dots,
                line_interconnect=self.line_interconnect,
                events=events,
                origin=origin,
                mdf=mdf,
                parent=self,
                hide_missing_channels=self.hide_missing_channels,
                hide_disabled_channels=self.hide_disabled_channels,
            )
            plot.pattern_group_added.connect(self.add_pattern_group)
            plot.pattern = pattern_info

            plot.show()
            plot.hide()

            sub = MdiSubWindow(parent=self)
            sub.setWidget(plot)
            sub.sigClosed.connect(self.window_closed_handler)
            sub.titleModified.connect(self.window_closed_handler)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(sub)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(sub)

                w.show()

                if geometry:
                    w.setGeometry(*geometry)
                else:
                    self.mdi_area.tileSubWindows()

            plot.hide()

            menu = w.systemMenu()

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)

            w.setWindowTitle(
                generate_window_title(w, window_info["type"], window_info["title"])
            )

            plot.add_new_channels(signals, mime_data)

            iterator = QtWidgets.QTreeWidgetItemIterator(plot.channel_selection)
            while iterator.value():
                item = iterator.value()
                iterator += 1

                widget = plot.channel_selection.itemWidget(item, 1)

                if isinstance(widget, ChannelDisplay):
                    sig, index = plot.plot.signal_by_uuid(widget.uuid)
                    if sig.group_index == NOT_FOUND:
                        widget.does_not_exist()

            needs_update = False
            for channel, sig in zip(found_signals, plot.plot.signals):
                if "mode" in channel:
                    sig.mode = channel["mode"]
                    needs_update = True
            if needs_update:
                plot.plot.update_lines()

            plot.show()

            plot.add_channels_request.connect(
                partial(self.add_new_channels, widget=plot)
            )

            descriptions = get_descriptions(window_info["configuration"]["channels"])

            iterator = QtWidgets.QTreeWidgetItemIterator(plot.channel_selection)
            while iterator.value():
                item = iterator.value()
                iterator += 1
                wid = plot.channel_selection.itemWidget(item, 1)
                if isinstance(wid, ChannelDisplay):
                    name = wid._name

                    description = descriptions.get(name, None)
                    if description is not None:

                        _, _idx = plot.plot.signal_by_uuid(wid.uuid)

                        if "y_range" in description:
                            plot.plot.view_boxes[_idx].setYRange(
                                *description["y_range"], padding=0
                            )

                        wid.set_fmt(description["fmt"])
                        wid.set_precision(description["precision"])
                        try:
                            wid.set_ranges(
                                [
                                    {
                                        "font_color": range["color"],
                                        "background_color": range["color"],
                                        "op1": "<=",
                                        "op2": "<=",
                                        "value1": float(range["start"]),
                                        "value2": float(range["stop"]),
                                    }
                                    for range in description["ranges"]
                                ]
                            )
                        except KeyError:
                            wid.set_ranges(description["ranges"])

                        for range in wid.ranges:
                            range["font_color"] = QtGui.QColor(range["font_color"])
                            range["background_color"] = QtGui.QColor(
                                range["background_color"]
                            )

                        wid.ylink.setCheckState(
                            QtCore.Qt.Checked
                            if description["common_axis"]
                            else QtCore.Qt.Unchecked
                        )
                        wid.individual_axis.setCheckState(
                            QtCore.Qt.Checked
                            if description.get("individual_axis", False)
                            else QtCore.Qt.Unchecked
                        )
                        width = description.get("individual_axis_width", 0)
                        if width:
                            plot.plot.axes[_idx].setWidth(width)
                        item.setCheckState(
                            0,
                            QtCore.Qt.Checked
                            if description["enabled"]
                            else QtCore.Qt.Unchecked,
                        )

                    elif pattern_info:
                        wid.set_ranges(pattern_info["ranges"])

            self.set_subplots_link(self.subplots_link)

            if "x_range" in window_info["configuration"]:
                plot.plot.viewbox.setXRange(
                    *window_info["configuration"]["x_range"], padding=0
                )

            if "splitter" in window_info["configuration"]:
                plot.splitter.setSizes(window_info["configuration"]["splitter"])

            if "y_axis_width" in window_info["configuration"]:
                plot.plot.y_axis.setWidth(window_info["configuration"]["y_axis_width"])

            if "grid" in window_info["configuration"]:
                x_grid, y_grid = window_info["configuration"]["grid"]
                plot.plot.plotItem.ctrl.xGridCheck.setChecked(x_grid)
                plot.plot.plotItem.ctrl.yGridCheck.setChecked(y_grid)

            plot.splitter.setContentsMargins(1, 1, 1, 1)
            plot.setContentsMargins(1, 1, 1, 1)

            if "cursor_precision" in window_info["configuration"]:
                plot.cursor_info.set_precision(
                    window_info["configuration"]["cursor_precision"]
                )

            iterator = QtWidgets.QTreeWidgetItemIterator(plot.channel_selection)
            while iterator.value():
                item = iterator.value()
                iterator += 1

                if isinstance(item, ChannelsGroupTreeItem):
                    if item.pattern:
                        plot.pattern_group_added.emit(plot, item)

            plot.channel_selection.refresh()

        elif window_info["type"] == "Tabular":
            # patterns
            pattern_info = window_info["configuration"].get("pattern", {})
            if pattern_info:
                required = set()
                found_signals = []

                signals_ = extract_signals_using_pattern(
                    self.mdf,
                    pattern_info,
                    self.ignore_value2text_conversions,
                    self.uuid,
                )

                try:
                    ranges = [
                        {
                            "font_color": range["color"],
                            "background_color": range["color"],
                            "op1": "<=",
                            "op2": "<=",
                            "value1": float(range["start"]),
                            "value2": float(range["stop"]),
                        }
                        for range in pattern_info["ranges"]
                    ]
                except KeyError:
                    ranges = pattern_info["ranges"]

                for range_info in ranges:
                    range_info["font_color"] = QtGui.QBrush(
                        QtGui.QColor(range_info["font_color"])
                    )
                    range_info["background_color"] = QtGui.QBrush(
                        QtGui.QColor(range_info["background_color"])
                    )

                ranges = {sig.name: copy_ranges(ranges) for sig in signals_}

                signals_ = [
                    (sig.name, sig.group_index, sig.channel_index) for sig in signals_
                ]

                pattern_info["ranges"] = ranges

            else:
                required = set(window_info["configuration"]["channels"])

                signals_ = [
                    (name, *self.mdf.whereis(name)[0])
                    for name in window_info["configuration"]["channels"]
                    if name in self.mdf
                ]

                ranges = window_info["configuration"].get("ranges", {})
                for channel_ranges in ranges.values():
                    for range_info in channel_ranges:
                        range_info["font_color"] = QtGui.QBrush(
                            QtGui.QColor(range_info["font_color"])
                        )
                        range_info["background_color"] = QtGui.QBrush(
                            QtGui.QColor(range_info["background_color"])
                        )

                if not signals_:
                    return

            signals = self.mdf.to_dataframe(
                channels=signals_,
                time_from_zero=False,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
            )

            found = set(signals.columns)
            dim = len(signals.index)

            for name in sorted(required - found):
                vals = np.empty(dim)
                vals.fill(np.NaN)
                signals[name] = pd.Series(vals, index=signals.index)

            tabular = Tabular(
                signals,
                ranges=ranges,
                start=self.mdf.header.start_time.timestamp(),
                parent=self,
            )
            tabular.pattern = pattern_info

            sub = MdiSubWindow(parent=self)
            sub.setWidget(tabular)
            sub.sigClosed.connect(self.window_closed_handler)
            sub.titleModified.connect(self.window_closed_handler)

            if not self.subplots:
                for mdi in self.mdi_area.subWindowList():
                    mdi.close()
                w = self.mdi_area.addSubWindow(sub)

                w.showMaximized()
            else:
                w = self.mdi_area.addSubWindow(sub)

                w.show()

                if geometry:
                    w.setGeometry(*geometry)
                else:
                    self.mdi_area.tileSubWindows()

            w.setWindowTitle(
                generate_window_title(w, window_info["type"], window_info["title"])
            )

            filter_count = 0
            available_columns = [signals.index.name] + list(signals.columns)
            for filter_info in window_info["configuration"]["filters"]:
                if filter_info["column"] in available_columns:
                    tabular.add_filter()
                    filter = tabular.filters.itemWidget(
                        tabular.filters.item(filter_count)
                    )
                    filter.enabled.setCheckState(
                        QtCore.Qt.Checked
                        if filter_info["enabled"]
                        else QtCore.Qt.Unchecked
                    )
                    filter.relation.setCurrentText(filter_info["relation"])
                    filter.column.setCurrentText(filter_info["column"])
                    filter.op.setCurrentText(filter_info["op"])
                    filter.target.setText(str(filter_info["target"]).strip('"'))
                    filter.validate_target()

                    filter_count += 1

            if filter_count and window_info["configuration"]["filtered"]:
                tabular.apply_filters()

            tabular.time_as_date.setCheckState(
                QtCore.Qt.Checked
                if window_info["configuration"]["time_as_date"]
                else QtCore.Qt.Unchecked
            )
            tabular.add_channels_request.connect(
                partial(self.add_new_channels, widget=tabular)
            )

            menu = w.systemMenu()

            action = QtWidgets.QAction("Set title", menu)
            action.triggered.connect(partial(set_title, w))
            before = menu.actions()[0]
            menu.insertAction(before, action)

            if self.subplots_link:
                tabular.timestamp_changed_signal.connect(self.set_cursor)

            sections_width = window_info["configuration"].get(
                "header_sections_width", []
            )
            if sections_width:
                for i, width in enumerate(sections_width):

                    tabular.tree.columnHeader.setColumnWidth(i, width)
                    tabular.tree.dataView.setColumnWidth(i, width)

                tabular.tree.dataView.updateGeometry()
                tabular.tree.columnHeader.updateGeometry()

        elif window_info["type"] in (
            "CAN Bus Trace",
            "FlexRay Bus Trace",
            "LIN Bus Trace",
        ):

            ranges = window_info["configuration"].get("ranges", {})
            for channel_ranges in ranges.values():
                for range_info in channel_ranges:
                    range_info["font_color"] = QtGui.QBrush(
                        QtGui.QColor(range_info["font_color"])
                    )
                    range_info["background_color"] = QtGui.QBrush(
                        QtGui.QColor(range_info["background_color"])
                    )

            if window_info["type"] == "CAN Bus Trace":
                widget = self._add_can_bus_trace_window(ranges)
            elif window_info["type"] == "FlexRay Bus Trace":
                widget = self._add_can_bus_trace_window(ranges)
            elif window_info["type"] == "LIN Bus Trace":
                widget = self._add_lin_bus_trace_window(ranges)

            sections_width = window_info["configuration"].get(
                "header_sections_width", []
            )
            if sections_width:
                for i, width in enumerate(sections_width):
                    widget.tree.header().resizeSection(i, width)

            scroll = widget.tree.horizontalScrollBar()
            if scroll:
                scroll.setValue(scroll.minimum())

            return

        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.FramelessWindowHint)

        if pattern_info:
            icon = QtGui.QIcon()
            icon.addPixmap(
                QtGui.QPixmap(":/filter.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
            )
            w.setWindowIcon(icon)

        w.layout().setSpacing(1)

        self.windows_modified.emit()

    def set_line_style(self, with_dots=None):
        if with_dots is None:
            with_dots = not self.with_dots

        current_plot = self.get_current_widget()
        if current_plot and isinstance(current_plot, Plot):
            self.with_dots = with_dots
            current_plot.with_dots = with_dots
            current_plot.plot.with_dots = with_dots
            current_plot.plot.update_lines()

    def set_line_interconnect(self, line_interconnect):

        if line_interconnect == "line":
            line_interconnect = ""

        self.line_interconnect = line_interconnect
        for i, mdi in enumerate(self.mdi_area.subWindowList()):
            widget = mdi.widget()
            if isinstance(widget, Plot):
                widget.line_interconnect = line_interconnect
                widget.plot.line_interconnect = line_interconnect
                widget.plot.update_lines()

    def set_subplots(self, option):
        self.subplots = option

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
                    widget.region_removed_signal.connect(self.remove_region)
                    widget.region_moved_signal.connect(self.set_region)
                    widget.splitter_moved.connect(self.set_splitter)
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
                    try:
                        widget.region_removed_signal.disconnect(self.remove_region)
                    except:
                        pass
                    try:
                        widget.region_modified_signal.disconnect(self.set_region)
                    except:
                        pass
                    try:
                        widget.splitter_moved.disconnect(self.set_splitter)
                    except:
                        pass
                elif isinstance(widget, Numeric):
                    try:
                        widget.timestamp_changed_signal.disconnect(self.set_cursor)
                    except:
                        pass

    def set_cursor(self, widget, pos):
        if not self.subplots_link:
            return

        if self._cursor_source is None:
            self._cursor_source = widget
            for mdi in self.mdi_area.subWindowList():
                wid = mdi.widget()
                if wid is not widget:
                    wid.set_timestamp(pos)

            self._cursor_source = None

    def set_region(self, widget, region):
        if not self.subplots_link:
            return

        if self._region_source is None:
            self._region_source = widget
            for mdi in self.mdi_area.subWindowList():
                wid = mdi.widget()
                if isinstance(wid, Plot) and wid is not widget:
                    if wid.plot.region is None:
                        event = QtGui.QKeyEvent(
                            QtCore.QEvent.KeyPress,
                            QtCore.Qt.Key_R,
                            QtCore.Qt.NoModifier,
                        )
                        wid.plot.keyPressEvent(event)
                    wid.plot.region.setRegion(region)
            self._region_source = None

    def set_splitter(self, widget, selection_width):
        if not self.subplots_link:
            return

        if self._splitter_source is None:
            self._splitter_source = widget
            for mdi in self.mdi_area.subWindowList():
                wid = mdi.widget()
                if isinstance(wid, Plot) and wid is not widget:
                    if selection_width is not None:
                        total_size = sum(wid.splitter.sizes())
                        if total_size > selection_width:
                            wid.splitter.setSizes(
                                [selection_width, total_size - selection_width]
                            )

            self._splitter_source = None

    def remove_cursor(self, widget):
        if not self.subplots_link:
            return

        if self._cursor_source is None:
            self._cursor_source = widget
            for mdi in self.mdi_area.subWindowList():
                plt = mdi.widget()
                if isinstance(plt, Plot) and plt is not widget:
                    plt.cursor_removed()
            self._cursor_source = None

    def remove_region(self, widget):
        if not self.subplots_link:
            return

        if self._region_source is None:
            self._region_source = widget
            for mdi in self.mdi_area.subWindowList():
                plt = mdi.widget()
                if isinstance(plt, Plot) and plt is not widget:
                    if plt.plot.region is not None:
                        event = QtGui.QKeyEvent(
                            QtCore.QEvent.KeyPress,
                            QtCore.Qt.Key_R,
                            QtCore.Qt.NoModifier,
                        )
                        plt.plot.keyPressEvent(event)
            self._region_source = None

    def save_all_subplots(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", "MDF version 4 files (*.mf4)"
        )

        if file_name:
            with MDF() as mdf:
                for mdi in self.mdi_area.subWindowList():
                    widget = mdi.widget()

                    if isinstance(widget, Plot):
                        mdf.append(widget.plot.signals)
                    elif isinstance(widget, Numeric):
                        mdf.append(list(widget.signals.values()))
                    elif isinstance(widget, Tabular):
                        mdf.append(widget.df)
                mdf.save(file_name, overwrite=True)

    def file_by_uuid(self, uuid):
        try:
            for file_index in range(self.files.count()):
                if self.files.widget(file_index).uuid == uuid:
                    return file_index, self.files.widget(file_index)
            return None
        except:
            if self.uuid == uuid:
                return 0, self
            else:
                return None

    def _show_info(self, lst):
        group_index, index, uuid = lst
        file_info = self.file_by_uuid(uuid)
        if file_info:
            _, file = file_info
            channel = file.mdf.get_channel_metadata(group=group_index, index=index)

            msg = ChannelInfoDialog(channel, self)
            msg.show()

    def window_closed_handler(self):
        self.windows_modified.emit()

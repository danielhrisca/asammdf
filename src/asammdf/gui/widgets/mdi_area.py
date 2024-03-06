from copy import deepcopy
from functools import partial
import inspect
import itertools
import json
import os
from pathlib import Path
from random import randint
import re
import sys
from tempfile import gettempdir
from traceback import format_exc
from zipfile import ZIP_DEFLATED, ZipFile

from natsort import natsorted
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

import asammdf.mdf as mdf_module

from ...blocks import v4_constants as v4c
from ...blocks.conversion_utils import from_dict
from ...blocks.utils import (
    csv_bytearray2hex,
    extract_mime_names,
    extract_xml_comment,
    load_can_database,
    MdfException,
)
from ...blocks.v4_blocks import EventBlock, HeaderBlock
from ...signal import Signal
from ..dialogs.channel_info import ChannelInfoDialog
from ..dialogs.messagebox import MessageBox
from ..dialogs.window_selection_dialog import WindowSelectionDialog
from ..utils import (
    computation_to_python_function,
    compute_signal,
    copy_ranges,
    replace_computation_dependency,
)
from .can_bus_trace import CANBusTrace
from .flexray_bus_trace import FlexRayBusTrace
from .gps import GPS
from .lin_bus_trace import LINBusTrace
from .numeric import Numeric
from .plot import Plot
from .tabular import Tabular

COMPONENT = re.compile(r"\[(?P<index>\d+)\]$")
SIG_RE = re.compile(r"\{\{(?!\}\})(?P<name>.*?)\}\}")
NOT_FOUND = 0xFFFFFFFF


def rename_origin_uuid(items):
    for item in items:
        if item.get("type", "channel") == "channel":
            if "mdf_uuid" in item:
                item["origin_uuid"] = item["mdf_uuid"]
                del item["mdf_uuid"]
            else:
                return
        else:
            rename_origin_uuid(item["channels"])


def get_origin_uuid(item):
    # if item.get("type", "channel") == "group":
    #     for subitem in item["channels"]:
    #         if subitem.get("type", "channel") == "channel":
    #             return subitem["origin_uuid"]
    #
    #     for subitem in item["channels"]:
    #         if subitem.get("type", "channel") == "group":
    #             uuid = get_origin_uuid(subitem)
    #             if uuid is not None:
    #                 return uuid
    #
    #     return None
    #
    # else:
    #     return item["origin_uuid"]
    return item["origin_uuid"]


def build_mime_from_config(
    items,
    mdf=None,
    computed_origin_uuid=None,
    default_index=NOT_FOUND,
    top=True,
    has_flags=None,
):
    if computed_origin_uuid is None:
        computed_origin_uuid = os.urandom(6).hex()
    if top:
        rename_origin_uuid(items)

    descriptions = {}
    found = {}
    not_found = {}
    computed = {}
    mime = []
    for item in items:
        uuid = os.urandom(6).hex()
        item["uuid"] = uuid

        if item.get("type", "channel") == "group":
            if item.get("pattern", None) is None:
                (
                    new_mine,
                    new_descriptions,
                    new_found,
                    new_not_found,
                    new_computed,
                ) = build_mime_from_config(
                    item["channels"],
                    mdf,
                    computed_origin_uuid,
                    default_index,
                    top=False,
                    has_flags=has_flags,
                )
                descriptions.update(new_descriptions)
                found.update(new_found)
                not_found.update(new_not_found)
                computed.update(new_computed)

                item["channels"] = new_mine

                mime.append(item)
            else:
                mime.append(item)
        else:
            descriptions[uuid] = item

            if has_flags is None:
                has_flags = "flags" in item

            if has_flags:
                # item["flags"] = Signal.Flags(item["flags"])
                item_is_computed = item["flags"] & Signal.Flags.computed

            else:
                item_is_computed = item.get("computed", False)
                flags = Signal.Flags.no_flags

                if "comment" in item:
                    flags |= Signal.Flags.user_defined_comment

                if "conversion" in item:
                    flags |= Signal.Flags.user_defined_conversion

                if "user_defined_name" in item:
                    flags |= Signal.Flags.user_defined_name

                if item_is_computed:
                    flags |= Signal.Flags.computed

                item["flags"] = flags

            if item_is_computed:
                group_index, channel_index = -1, -1
                computed[uuid] = item
                item["computation"] = computation_to_python_function(item["computation"])
                item["computation"].pop("definition", None)
                item["origin_uuid"] = computed_origin_uuid

            else:
                occurrences = mdf.whereis(item["name"]) if mdf else None
                if occurrences:
                    group_index, channel_index = occurrences[0]
                    found[uuid] = item["name"], group_index, channel_index

                else:
                    group_index, channel_index = default_index, default_index
                    not_found[item["name"]] = uuid

            item["group_index"] = group_index
            item["channel_index"] = channel_index
            mime.append(item)

    return mime, descriptions, found, not_found, computed


def extract_signals_using_pattern(
    mdf, channels_db, pattern_info, ignore_value2text_conversions, uuid=None, as_names=False
):
    if not mdf and not channels_db:
        if as_names:
            return set()
        else:
            return {}

    elif not channels_db:
        channels_db = mdf.channels_db

    pattern = pattern_info["pattern"]
    match_type = pattern_info["match_type"]
    case_sensitive = pattern_info.get("case_sensitive", False)
    filter_value = pattern_info["filter_value"]
    filter_type = pattern_info["filter_type"]
    raw = pattern_info["raw"]
    integer_format = pattern_info.get("integer_format", "phys")

    if match_type == "Wildcard":
        wild = f"__{os.urandom(3).hex()}WILDCARD{os.urandom(3).hex()}__"
        pattern = pattern.replace("*", wild)
        pattern = re.escape(pattern)
        pattern = pattern.replace(wild, ".*")

    try:
        if case_sensitive:
            pattern = re.compile(pattern)
        else:
            pattern = re.compile(f"(?i){pattern}")

        matches = {}

        for name, entries in channels_db.items():
            if pattern.fullmatch(name):
                for entry in entries:
                    if entry in matches:
                        continue
                    matches[entry] = name

        matches = natsorted((name, *entry) for entry, name in matches.items())
    except:
        print(format_exc())
        signals = []
    else:
        if (as_names and filter_type == "Unspecified") or not mdf:
            return {match[0] for match in matches}

        psignals = mdf.select(
            matches,
            ignore_value2text_conversions=ignore_value2text_conversions,
            copy_master=False,
            validate=True,
            raw=True,
        )

        if filter_type == "Unspecified":
            keep = psignals
        else:
            keep = []
            for i, (name, group_index, channel_index) in enumerate(matches):
                sig = psignals[i]
                sig.origin_uuid = uuid
                sig.group_index, sig.channel_index = group_index, channel_index

                size = len(sig)
                if not size:
                    continue

                target = np.full(size, filter_value)

                if not raw:
                    samples = sig.physical(copy=False).samples
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

    output_signals = {}
    for sig in signals:
        uuid = os.urandom(6).hex()
        sig.uuid = uuid
        sig.format = integer_format
        sig.ranges = []
        output_signals[uuid] = sig

    if as_names:
        return {sig.name for sig in signals}
    else:
        return output_signals


def generate_window_title(mdi, window_name="", title=""):
    used_names = {window.windowTitle() for window in mdi.mdiArea().subWindowList() if window is not mdi}

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


def get_descriptions_by_name(channels):
    descriptions = {}
    for channel in channels:
        if channel.get("type", "channel") == "group":
            new_descriptions = get_descriptions_by_name(channel["channels"])
            descriptions.update(new_descriptions)
        else:
            descriptions[channel["name"]] = channel

    return descriptions


def get_flatten_entries_from_mime(data, default_index=None):
    entries = []

    for item in data:
        if item.get("type", "channel") == "channel":
            new_item = dict(item)

            if default_index is not None:
                new_item["group_index"] = default_index
                new_item["channel_index"] = default_index

            entries.append(new_item)

        else:
            entries.extend(get_flatten_entries_from_mime(item["channels"], default_index))
    return entries


def get_functions(data):
    functions = {}

    for item in data:
        if item.get("type", "channel") == "group":
            functions.update(get_functions(item["channels"]))
        else:
            if item.get("computed", False):
                computation = item["computation"] = computation_to_python_function(item["computation"])

                functions[computation["function"]] = computation["definition"]

    return functions


def get_pattern_groups(data):
    groups = []
    for item in data:
        if item.get("type", "channel") == "group":
            if item["pattern"] is not None:
                groups.append(item)
            else:
                groups.extend(get_pattern_groups(item["channels"]))
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
                names.extend([match.group("name") for match in SIG_RE.finditer(expression_string)])
            elif computation["type"] == "python_function":
                for alternative_names in computation["args"].values():
                    for name in alternative_names:
                        if name:
                            names.append(name)

                triggering = computation.get("triggering", "triggering_on_all")

                if triggering == "triggering_on_channel":
                    triggering_channel = computation["triggering_value"]
                    if triggering_channel:
                        names.append(triggering_channel)

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

        elif channel["type"] == "expression":
            expression_string = channel["expression"]
            names.extend([match.group("name") for match in SIG_RE.finditer(expression_string)])

        elif channel["type"] == "function":
            op = channel["channel"]
            if isinstance(op, str):
                names.append(op)
            else:
                names.extend(get_required_from_computed(op))

        elif channel["type"] == "python_function":
            for alternative_names in channel["args"].values():
                for name in alternative_names:
                    if name:
                        names.append(name)

    return names


def substitude_mime_uuids(mime, uuid=None, force=False):
    if not mime:
        return mime

    new_mime = []

    for item in mime:
        if item.get("type", "channel") == "channel":
            if force or item["origin_uuid"] is None:
                item["origin_uuid"] = uuid
            new_mime.append(item)
        else:
            item["channels"] = substitude_mime_uuids(item["channels"], uuid, force=force)
            if force or item["origin_uuid"] is None:
                item["origin_uuid"] = uuid
            new_mime.append(item)
    return new_mime


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


def load_comparison_display_file(file_name, uuids):
    with open(file_name) as infile:
        info = json.load(infile)
    windows = info.get("windows", [])
    plot_windows = []
    for window in windows:
        if window["type"] != "Plot":
            continue

        window["configuration"]["channels"] = get_comparison_mime(window["configuration"]["channels"], uuids)

        plot_windows.append(window)


def get_comparison_mime(data, uuids):
    entries = []

    for item in data:
        if item.get("type", "channel") == "channel":
            for uuid in uuids:
                new_item = dict(item)
                new_item["origin_uuid"] = uuid
                entries.append(new_item)

        else:
            new_item = dict(item)
            new_item["channels"] = get_comparison_mime(item["channels"], uuids)
            entries.append(new_item)

    return entries


class MdiSubWindow(QtWidgets.QMdiSubWindow):
    sigClosed = QtCore.Signal(object)
    titleModified = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

    def closeEvent(self, event):
        if isinstance(self.widget(), Plot):
            self.widget().close()
        super().closeEvent(event)
        self.sigClosed.emit(self)


class MdiAreaWidget(QtWidgets.QMdiArea):
    add_window_request = QtCore.Signal(list)
    open_file_request = QtCore.Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setAcceptDrops(True)
        self.placeholder_text = (
            "Drag and drop channels, or select channels and press the <Create window> button, to create new windows"
        )
        self.show()

    def cascadeSubWindows(self):
        sub_windows = self.subWindowList()
        if not sub_windows:
            return

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = True

        super().cascadeSubWindows()

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = False

    def dragEnterEvent(self, e):
        e.accept()
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        if e.source() is self:
            super().dropEvent(e)
        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                dialog = WindowSelectionDialog(parent=self)
                dialog.setModal(True)
                dialog.exec_()

                if dialog.result():
                    window_type = dialog.selected_type()
                    disable_new_channels = dialog.disable_new_channels()
                    names = extract_mime_names(data, disable_new_channels=disable_new_channels)

                    self.add_window_request.emit([window_type, names])
            else:
                try:
                    for path in e.mimeData().text().splitlines():
                        path = Path(path.replace(r"file:///", ""))
                        if path.suffix.lower() in (
                            ".csv",
                            ".zip",
                            ".erg",
                            ".dat",
                            ".mdf",
                            ".mf4",
                            ".mf4z",
                        ):
                            self.open_file_request.emit(str(path))
                except:
                    print(format_exc())

    def tile_horizontally(self):
        sub_windows = self.subWindowList()
        if not sub_windows:
            return

        position = QtCore.QPoint(0, 0)

        width = self.width()
        height = self.height()
        ratio = height // len(sub_windows)

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = True

        for window in sub_windows:
            if window.isMinimized() or window.isMaximized():
                window.showNormal()
            rect = QtCore.QRect(0, 0, width, ratio)

            window.setGeometry(rect)
            window.move(position)
            position.setY(position.y() + ratio)

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = False

    def tile_vertically(self):
        sub_windows = self.subWindowList()
        if not sub_windows:
            return

        position = QtCore.QPoint(0, 0)

        width = self.width()
        height = self.height()
        ratio = width // len(sub_windows)

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = True

        for window in sub_windows:
            if window.isMinimized() or window.isMaximized():
                window.showNormal()
            rect = QtCore.QRect(0, 0, ratio, height)

            window.setGeometry(rect)
            window.move(position)
            position.setX(position.x() + ratio)

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = False

    def tileSubWindows(self):
        sub_windows = self.subWindowList()
        if not sub_windows:
            return

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = True

        super().tileSubWindows()

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = False

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
                self.placeholder_text, QtCore.Qt.TextElideMode.ElideRight, self.viewport().width()
            )
            painter.drawText(self.viewport().rect(), QtCore.Qt.AlignmentFlag.AlignCenter, elided_text)
            painter.restore()


class WithMDIArea:
    windows_modified = QtCore.Signal()
    load_plot_x_range = False

    def __init__(self, comparison=False, *args, **kwargs):
        self.comparison = comparison
        self._cursor_source = None
        self._region_source = None
        self._splitter_source = None
        self._window_counter = 0
        self._frameless_windows = False

        self.cursor_circle = True
        self.cursor_horizontal_line = True
        self.cursor_line_width = 1
        self.cursor_color = "#e69138"

        self.functions = {}

    def add_pattern_group(self, plot, group):
        signals = extract_signals_using_pattern(
            mdf=self.mdf,
            channels_db=None,
            pattern_info=group.pattern,
            ignore_value2text_conversions=self.ignore_value2text_conversions,
            uuid=self.uuid,
        )

        signals = {
            sig_uuid: sig
            for sig_uuid, sig in signals.items()
            if sig.samples.dtype.kind not in "SU" and not sig.samples.dtype.names and not len(sig.samples.shape) > 1
        }

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
        else:
            ignore_value2text_conversions = self.ignore_value2text_conversions

        try:
            names = list(names)
            if names and isinstance(names[0], str):
                signals_ = [
                    {
                        "name": name,
                        "group_index": self.mdf.whereis(name)[0][0],
                        "channel_index": self.mdf.whereis(name)[0][1],
                        "origin_uuid": self.uuid,
                        "type": "channel",
                        "ranges": [],
                        "uuid": os.urandom(6).hex(),
                    }
                    for name in names
                    if name in self.mdf
                ]

                uuids = {self.uuid}

                mime_data = signals_
                computed = []

            else:
                mime_data = names

                try:
                    mime_data = substitude_mime_uuids(mime_data, self.uuid)
                except:
                    pass

                entries = get_flatten_entries_from_mime(mime_data)

                uuids = {entry["origin_uuid"] for entry in entries}
                for uuid in uuids:
                    if self.file_by_uuid(uuid):
                        break
                else:
                    mime_data = substitude_mime_uuids(mime_data, uuid=self.uuid, force=True)
                    entries = get_flatten_entries_from_mime(mime_data)

                signals_ = [entry for entry in entries if (entry["group_index"], entry["channel_index"]) != (-1, -1)]

                computed = [entry for entry in entries if (entry["group_index"], entry["channel_index"]) == (-1, -1)]

                uuids = {entry["origin_uuid"] for entry in entries}

            if isinstance(widget, Tabular):
                dfs = []

                for uuid in uuids:
                    uuids_signals = [
                        (entry["name"], entry["group_index"], entry["channel_index"])
                        for entry in signals_
                        if entry["origin_uuid"] == uuid
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
                    if name.endswith(".ID"):
                        signals[name] = signals[name].astype("<u4") & 0x1FFFFFFF

                widget.add_new_channels(signals)

            elif isinstance(widget, Numeric):
                signals = []

                for uuid in uuids:
                    uuids_signals = [
                        (entry["name"], entry["group_index"], entry["channel_index"])
                        for entry in signals_
                        if entry["origin_uuid"] == uuid
                    ]

                    uuids_signals_uuid = [entry for entry in signals_ if entry["origin_uuid"] == uuid]

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

                    for sig, sig_, sig_uuid in zip(selected_signals, uuids_signals, uuids_signals_uuid):
                        sig.group_index = sig_[1]
                        sig.channel_index = sig_[2]
                        sig.flags &= ~sig.Flags.computed
                        sig.computation = {}
                        sig.origin_uuid = uuid
                        sig.name = sig_[0]
                        sig.uuid = sig_uuid["uuid"]
                        sig.ranges = sig_uuid["ranges"]

                        if not hasattr(self, "mdf"):
                            # MainWindow => comparison plots

                            sig.tooltip = f"{sig.name}\n@ {file.file_name}"
                            sig.name = f"{file_index+1}: {sig.name}"

                    signals.extend(selected_signals)

                for signal in signals:
                    if len(signal.samples.shape) > 1:
                        signal.samples = csv_bytearray2hex(pd.Series(list(signal.samples)))

                    if signal.name.endswith(".ID"):
                        signal.samples = signal.samples.astype("<u4") & 0x1FFFFFFF

                signals = natsorted(signals, key=lambda x: x.name)

                widget.add_new_channels(signals, mime_data=mime_data)

            elif isinstance(widget, Plot):
                signals = {}

                not_found = []

                for uuid in uuids:
                    uuids_entries = [entry for entry in signals_ if entry["origin_uuid"] == uuid]

                    uuids_signals = []

                    file_info = self.file_by_uuid(uuid)
                    if not file_info:
                        continue

                    file_index, file = file_info

                    for entry in uuids_entries:
                        if entry["name"] in file.mdf:
                            entries = file.mdf.whereis(entry["name"])

                            if (
                                entry["group_index"],
                                entry["channel_index"],
                            ) not in entries:
                                entry["group_index"], entry["channel_index"] = entries[0]
                            uuids_signals.append(entry)
                        else:
                            not_found.append(entry)

                    selected_signals = file.mdf.select(
                        [
                            (
                                entry["name"],
                                entry["group_index"],
                                entry["channel_index"],
                            )
                            for entry in uuids_signals
                        ],
                        ignore_value2text_conversions=ignore_value2text_conversions,
                        copy_master=False,
                        validate=True,
                        raw=True,
                    )

                    nd = []

                    for sig, sig_ in zip(selected_signals, uuids_signals):
                        sig.group_index = sig_["group_index"]
                        sig.channel_index = sig_["channel_index"]
                        sig.flags &= ~sig.Flags.computed
                        sig.computation = {}
                        sig.origin_uuid = uuid
                        sig.name = sig_["name"]
                        sig.color = sig_.get("color", None)
                        sig.uuid = sig_["uuid"]

                        if not hasattr(self, "mdf"):
                            # MainWindow => comparison plots

                            sig.tooltip = f"{sig.name}\n@ {file.file_name}"
                            sig.name = f"{file_index+1}: {sig.name}"

                        if sig.samples.dtype.kind not in "SU" and (
                            sig.samples.dtype.names or len(sig.samples.shape) > 1
                        ):
                            nd.append(sig)
                        else:
                            signals[sig.uuid] = sig

                    for sig in nd:
                        if sig.samples.dtype.names is None:
                            shape = sig.samples.shape[1:]

                            matrix_dims = [list(range(dim)) for dim in shape]

                            matrix_name = sig.name

                            for indexes in itertools.product(*matrix_dims):
                                indexes_string = "".join(f"[{_index}]" for _index in indexes)

                                samples = sig.samples
                                for idx in indexes:
                                    samples = samples[:, idx]
                                sig_name = f"{matrix_name}{indexes_string}"

                                new_sig = sig.copy()
                                new_sig.name = sig_name
                                new_sig.samples = samples
                                new_sig.group_index = sig.group_index
                                new_sig.channel_index = sig.channel_index
                                new_sig.flags &= ~sig.Flags.computed
                                new_sig.computation = {}
                                new_sig.origin_uuid = sig.origin_uuid
                                new_sig.uuid = os.urandom(6).hex()
                                new_sig.enable = getattr(sig, "enable", True)

                                signals[new_sig.uuid] = new_sig
                        else:
                            name = sig.samples.dtype.names[0]
                            if name == sig.name:
                                array_samples = sig.samples[name]

                                shape = array_samples.shape[1:]

                                matrix_dims = [list(range(dim)) for dim in shape]

                                matrix_name = sig.name

                                for indexes in itertools.product(*matrix_dims):
                                    indexes_string = "".join(f"[{_index}]" for _index in indexes)

                                    samples = array_samples
                                    for idx in indexes:
                                        samples = samples[:, idx]
                                    sig_name = f"{matrix_name}{indexes_string}"

                                    new_sig = sig.copy()
                                    new_sig.name = sig_name
                                    new_sig.samples = samples
                                    new_sig.group_index = sig.group_index
                                    new_sig.channel_index = sig.channel_index
                                    new_sig.flags &= ~sig.Flags.computed
                                    new_sig.computation = {}
                                    new_sig.origin_uuid = sig.origin_uuid
                                    new_sig.uuid = os.urandom(6).hex()
                                    new_sig.enable = getattr(sig, "enable", True)

                                    signals[new_sig.uuid] = new_sig

                signals = {
                    key: sig
                    for key, sig in signals.items()
                    if sig.samples.dtype.kind not in "SU"
                    and not sig.samples.dtype.names
                    and not len(sig.samples.shape) > 1
                }

                for signal in signals.values():
                    if len(signal.samples.shape) > 1:
                        signal.samples = csv_bytearray2hex(pd.Series(list(signal.samples)))

                    if signal.name.endswith(".ID"):
                        signal.samples = signal.samples.astype("<u4") & 0x1FFFFFFF

                sigs = signals

                if computed:
                    required_channels = []
                    for ch in computed:
                        required_channels.extend(get_required_from_computed(ch))

                    required_channels = set(required_channels)

                    measured_signals = {sig.name: sig for sig in sigs.values()}

                    required_channels = [
                        (channel, *file.mdf.whereis(channel)[0])
                        for channel in required_channels
                        if channel not in measured_signals and channel in file.mdf
                    ]
                    required_channels = {
                        sig.name: sig
                        for sig in file.mdf.select(
                            required_channels,
                            ignore_value2text_conversions=file.ignore_value2text_conversions,
                            copy_master=False,
                        )
                    }

                    required_channels.update(measured_signals)

                    if required_channels:
                        all_timebase = np.unique(
                            np.concatenate(
                                list(
                                    {id(sig.timestamps): sig.timestamps for sig in required_channels.values()}.values()
                                )
                            )
                        )
                    else:
                        all_timebase = []

                    computed_signals = {}

                    for channel in computed:
                        computation = channel["computation"]

                        signal = compute_signal(
                            computation,
                            required_channels,
                            all_timebase,
                            self.functions,
                        )
                        signal.name = channel["name"]
                        signal.unit = channel["unit"]
                        signal.color = channel["color"]
                        signal.flags |= signal.Flags.computed
                        signal.computation = channel["computation"]
                        signal.group_index = -1
                        signal.channel_index = -1
                        signal.origin_uuid = file.uuid
                        signal.comment = channel["computation"].get("channel_comment", "")
                        signal.uuid = channel.get("uuid", os.urandom(6).hex())

                        if channel["flags"] & Signal.Flags.user_defined_conversion:
                            signal.conversion = from_dict(channel["conversion"])
                            signal.flags |= signal.Flags.user_defined_conversion

                        if channel["flags"] & Signal.Flags.user_defined_name:
                            sig.original_name = channel["name"]
                            sig.name = channel.get("user_defined_name", "") or ""

                            signal.flags |= signal.Flags.user_defined_name

                        computed_signals[signal.uuid] = signal
                    signals.update(computed_signals)

                not_found_uuid = os.urandom(6).hex()

                for entry in not_found:
                    sig = Signal([], [], name=entry["name"])
                    sig.uuid = entry["uuid"]

                    sig.origin_uuid = not_found_uuid
                    sig.group_index = NOT_FOUND
                    sig.channel_index = NOT_FOUND
                    sig.color = entry.get("color", None)

                    if entry["flags"] & Signal.Flags.user_defined_conversion:
                        sig.conversion = from_dict(entry["conversion"])
                        sig.flags |= Signal.Flags.user_defined_conversion

                    if entry["flags"] & Signal.Flags.user_defined_name:
                        sig.original_name = entry["name"]
                        sig.name = entry.get("user_defined_name", "") or ""
                        sig.flags |= Signal.Flags.user_defined_name

                    signals[sig.uuid] = sig

                if widget.channel_selection.selectedItems():
                    item = widget.channel_selection.selectedItems()[0]
                    if item.type() == item.Channel:
                        item_below = widget.channel_selection.itemBelow(item)
                        if item_below is None or item_below.parent() != item.parent():
                            destination = item.parent()
                        else:
                            destination = item_below
                    elif item.type() == item.Group:
                        destination = item
                    else:
                        destination = None

                else:
                    destination = None
                widget.add_new_channels(signals, mime_data=mime_data, destination=destination)

        except MdfException:
            print(format_exc())

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
        elif window_type == "Plot":
            return self._add_plot_window(names)
        elif window_type == "Numeric":
            return self._add_numeric_window(names)
        elif window_type == "Tabular":
            return self._add_tabular_window(names)

    def _add_can_bus_trace_window(self, ranges=None):
        if self.mdf.version < "4.00":
            return

        groups_count = len(self.mdf.groups)

        dfs = []

        for index in range(groups_count):
            group = self.mdf.groups[index]
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                source = group.channel_group.acq_source

                names = [ch.name for ch in group.channels]

                if source and source.bus_type == v4c.BUS_TYPE_CAN:
                    if "CAN_DataFrame" in names:
                        data = self.mdf.get("CAN_DataFrame", index)  # , raw=True)

                    elif "CAN_RemoteFrame" in names:
                        data = self.mdf.get("CAN_RemoteFrame", index, raw=True)

                    elif "CAN_ErrorFrame" in names:
                        data = self.mdf.get("CAN_ErrorFrame", index, raw=True)

                    else:
                        continue

                    df_index = data.timestamps
                    count = len(df_index)

                    columns = {
                        "timestamps": df_index,
                        "Bus": np.full(count, "Unknown", dtype="O"),
                        "ID": np.full(count, 0xFFFFFFFF, dtype="u4"),
                        "IDE": np.zeros(count, dtype="u1"),
                        "Direction": np.full(count, "", dtype="O"),
                        "Name": np.full(count, "", dtype="O"),
                        "Event Type": np.full(count, "CAN Frame", dtype="O"),
                        "Details": np.full(count, "", dtype="O"),
                        "ESI": np.full(count, "", dtype="O"),
                        "EDL": np.full(count, "Standard CAN", dtype="O"),
                        "BRS": np.full(count, "", dtype="O"),
                        "DLC": np.zeros(count, dtype="u1"),
                        "Data Length": np.zeros(count, dtype="u1"),
                        "Data Bytes": np.full(count, "", dtype="O"),
                    }

                    for string in v4c.CAN_ERROR_TYPES.values():
                        sys.intern(string)

                    frame_map = None
                    if data.attachment and data.attachment[0]:
                        dbc = load_can_database(data.attachment[1], data.attachment[0])
                        if dbc:
                            frame_map = {frame.arbitration_id.id: frame.name for frame in dbc}

                            for name in frame_map.values():
                                sys.intern(name)

                    if data.name == "CAN_DataFrame":
                        vals = data["CAN_DataFrame.BusChannel"].astype("u1")

                        vals = [f"CAN {chn}" for chn in vals.tolist()]
                        columns["Bus"] = vals

                        vals = data["CAN_DataFrame.ID"].astype("u4") & 0x1FFFFFFF
                        columns["ID"] = vals
                        if frame_map:
                            columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        if "CAN_DataFrame.IDE" in names:
                            columns["IDE"] = data["CAN_DataFrame.IDE"].astype("u1")

                        columns["DLC"] = data["CAN_DataFrame.DLC"].astype("u1")
                        data_length = data["CAN_DataFrame.DataLength"].astype("u1")
                        columns["Data Length"] = data_length

                        vals = csv_bytearray2hex(
                            pd.Series(list(data["CAN_DataFrame.DataBytes"])),
                            data_length.tolist(),
                        )
                        columns["Data Bytes"] = vals

                        if "CAN_DataFrame.Dir" in names:
                            if data["CAN_DataFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [v.decode("utf-8") for v in data["CAN_DataFrame.Dir"].tolist()]
                            else:
                                columns["Direction"] = [
                                    "TX" if dir else "RX" for dir in data["CAN_DataFrame.Dir"].astype("u1").tolist()
                                ]

                        if "CAN_DataFrame.ESI" in names:
                            columns["ESI"] = [
                                "Error" if dir else "No error"
                                for dir in data["CAN_DataFrame.ESI"].astype("u1").tolist()
                            ]

                        if "CAN_DataFrame.EDL" in names:
                            columns["EDL"] = [
                                "CAN FD" if dir else "Standard CAN"
                                for dir in data["CAN_DataFrame.EDL"].astype("u1").tolist()
                            ]

                        if "CAN_DataFrame.BRS" in names:
                            columns["BRS"] = [str(dir) for dir in data["CAN_DataFrame.BRS"].astype("u1").tolist()]

                        vals = None
                        data_length = None

                    elif data.name == "CAN_RemoteFrame":
                        vals = data["CAN_RemoteFrame.BusChannel"].astype("u1")
                        vals = [f"CAN {chn}" for chn in vals.tolist()]
                        columns["Bus"] = vals

                        vals = data["CAN_RemoteFrame.ID"].astype("u4") & 0x1FFFFFFF
                        columns["ID"] = vals
                        if frame_map:
                            columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        if "CAN_RemoteFrame.IDE" in names:
                            columns["IDE"] = data["CAN_RemoteFrame.IDE"].astype("u1")

                        columns["DLC"] = data["CAN_RemoteFrame.DLC"].astype("u1")
                        data_length = data["CAN_RemoteFrame.DataLength"].astype("u1")
                        columns["Data Length"] = data_length
                        columns["Event Type"] = "Remote Frame"

                        if "CAN_RemoteFrame.Dir" in names:
                            if data["CAN_RemoteFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [v.decode("utf-8") for v in data["CAN_RemoteFrame.Dir"].tolist()]
                            else:
                                columns["Direction"] = [
                                    "TX" if dir else "RX" for dir in data["CAN_RemoteFrame.Dir"].astype("u1").tolist()
                                ]

                        vals = None
                        data_length = None

                    elif data.name == "CAN_ErrorFrame":
                        names = set(data.samples.dtype.names)

                        if "CAN_ErrorFrame.BusChannel" in names:
                            vals = data["CAN_ErrorFrame.BusChannel"].astype("u1")
                            vals = [f"CAN {chn}" for chn in vals.tolist()]
                            columns["Bus"] = vals

                        if "CAN_ErrorFrame.ID" in names:
                            vals = data["CAN_ErrorFrame.ID"].astype("u4") & 0x1FFFFFFF
                            columns["ID"] = vals
                            if frame_map:
                                columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        if "CAN_ErrorFrame.IDE" in names:
                            columns["IDE"] = data["CAN_ErrorFrame.IDE"].astype("u1")

                        if "CAN_ErrorFrame.DLC" in names:
                            columns["DLC"] = data["CAN_ErrorFrame.DLC"].astype("u1")

                        if "CAN_ErrorFrame.DataLength" in names:
                            columns["Data Length"] = data["CAN_ErrorFrame.DataLength"].astype("u1")

                        columns["Event Type"] = "Error Frame"

                        if "CAN_ErrorFrame.ErrorType" in names:
                            vals = data["CAN_ErrorFrame.ErrorType"].astype("u1").tolist()
                            vals = [v4c.CAN_ERROR_TYPES.get(err, "Other error") for err in vals]

                            columns["Details"] = vals

                        if "CAN_ErrorFrame.Dir" in names:
                            if data["CAN_ErrorFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [v.decode("utf-8") for v in data["CAN_ErrorFrame.Dir"].tolist()]
                            else:
                                columns["Direction"] = [
                                    "TX" if dir else "RX" for dir in data["CAN_ErrorFrame.Dir"].astype("u1").tolist()
                                ]

                    df = pd.DataFrame(columns, index=df_index)
                    dfs.append(df)

        if not dfs:
            return
        else:
            signals = pd.concat(dfs).sort_index()

            index = pd.Index(range(len(signals)))
            signals.set_index(index, inplace=True)

        del dfs

        trace = CANBusTrace(signals, start=self.mdf.header.start_time, ranges=ranges)

        sub = MdiSubWindow(parent=self)
        sub.setWidget(trace)
        trace.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/bus_can.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
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
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        w.layout().setSpacing(1)

        def set_title(mdi):
            name, ok = QtWidgets.QInputDialog.getText(self, "Set sub-plot title", "Title:")
            if ok and name:
                mdi.setWindowTitle(name)

        action = QtGui.QAction("Set title", menu)
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
        if self.mdf.version < "4.00":
            return

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
                        items.append((data, names))

                    elif "FLX_NullFrame" in names:
                        data = self.mdf.get("FLX_NullFrame", index, raw=True)
                        items.append((data, names))

                    elif "FLX_StartCycle" in names:
                        data = self.mdf.get("FLX_StartCycle", index, raw=True)
                        items.append((data, names))

                    elif "FLX_Status" in names:
                        data = self.mdf.get("FLX_Status", index, raw=True)
                        items.append((data, names))

        if not len(items):
            return

        df_index = np.sort(np.concatenate([item.timestamps for (item, names) in items]))
        count = len(df_index)

        columns = {
            "timestamps": df_index,
            "Bus": np.full(count, "Unknown", dtype="O"),
            "ID": np.full(count, 0xFFFF, dtype="u2"),
            "Direction": np.full(count, "", dtype="O"),
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
            data, names = items.pop()

            frame_map = {}

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

            if data.name == "FLX_Frame":
                index = np.searchsorted(df_index, data.timestamps)

                vals = data["FLX_Frame.FlxChannel"].astype("u1")

                vals = [f"FlexRay {chn}" for chn in vals.tolist()]
                columns["Bus"][index] = vals

                vals = data["FLX_Frame.ID"].astype("u2")
                columns["ID"][index] = vals
                if frame_map:
                    columns["Name"][index] = [frame_map.get(_id, "") for _id in vals.tolist()]

                vals = data["FLX_Frame.Cycle"].astype("u1")
                columns["Cycle"][index] = vals

                data_length = data["FLX_Frame.DataLength"].astype("u1")
                columns["Data Length"][index] = data_length

                vals = csv_bytearray2hex(
                    pd.Series(list(data["FLX_Frame.DataBytes"])),
                    data_length,
                )
                columns["Data Bytes"][index] = vals

                vals = data["FLX_Frame.HeaderCRC"].astype("u2")
                columns["Header CRC"][index] = vals

                if "FLX_Frame.Dir" in names:
                    if data["FLX_Frame.Dir"].dtype.kind == "S":
                        columns["Direction"][index] = [v.decode("utf-8") for v in data["FLX_Frame.Dir"].tolist()]
                    else:
                        columns["Direction"][index] = [
                            "TX" if dir else "RX" for dir in data["FLX_Frame.Dir"].astype("u1").tolist()
                        ]

                vals = None
                data_length = None

            elif data.name == "FLX_NullFrame":
                index = np.searchsorted(df_index, data.timestamps)

                vals = data["FLX_NullFrame.FlxChannel"].astype("u1")
                vals = [f"FlexRay {chn}" for chn in vals.tolist()]
                columns["Bus"][index] = vals

                vals = data["FLX_NullFrame.ID"].astype("u2")
                columns["ID"][index] = vals
                if frame_map:
                    columns["Name"][index] = [frame_map.get(_id, "") for _id in vals.tolist()]

                vals = data["FLX_NullFrame.Cycle"].astype("u1")
                columns["Cycle"][index] = vals

                columns["Event Type"][index] = "FlexRay NullFrame"

                vals = data["FLX_NullFrame.HeaderCRC"].astype("u2")
                columns["Header CRC"][index] = vals

                if "FLX_NullFrame.Dir" in names:
                    if data["FLX_NullFrame.Dir"].dtype.kind == "S":
                        columns["Direction"][index] = [v.decode("utf-8") for v in data["FLX_NullFrame.Dir"].tolist()]
                    else:
                        columns["Direction"][index] = [
                            "TX" if dir else "RX" for dir in data["FLX_NullFrame.Dir"].astype("u1").tolist()
                        ]

                vals = None
                data_length = None

            elif data.name == "FLX_StartCycle":
                index = np.searchsorted(df_index, data.timestamps)

                vals = data["FLX_StartCycle.Cycle"].astype("u1")
                columns["Cycle"][index] = vals

                columns["Event Type"][index] = "FlexRay StartCycle"

                vals = None
                data_length = None

            elif data.name == "FLX_Status":
                index = np.searchsorted(df_index, data.timestamps)

                vals = data["FLX_Status.StatusType"].astype("u1")
                columns["Details"][index] = vals.astype("U").astype("O")

                columns["Event Type"][index] = "FlexRay Status"

                vals = None
                data_length = None

        signals = pd.DataFrame(columns)

        trace = FlexRayBusTrace(signals, start=self.mdf.header.start_time, ranges=ranges)

        sub = MdiSubWindow(parent=self)
        sub.setWidget(trace)
        trace.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/bus_flx.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
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
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        w.layout().setSpacing(1)

        def set_title(mdi):
            name, ok = QtWidgets.QInputDialog.getText(self, "Set sub-plot title", "Title:")
            if ok and name:
                mdi.setWindowTitle(name)

        action = QtGui.QAction("Set title", menu)
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

    def _add_gps_window(self, signals):
        signals = [sig[:3] for sig in signals]
        latitude_channel, longitude_channel = self.mdf.select(signals, validate=True)

        gps = GPS(latitude_channel, longitude_channel)
        sub = MdiSubWindow(parent=self)
        sub.setWidget(gps)
        gps.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/globe.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        sub.setWindowIcon(icon)

        w = self.mdi_area.addSubWindow(sub)

        if len(self.mdi_area.subWindowList()) == 1:
            w.showMaximized()
        else:
            w.show()
            self.mdi_area.tileSubWindows()

        menu = w.systemMenu()
        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        w.layout().setSpacing(1)

        def set_title(mdi):
            name, ok = QtWidgets.QInputDialog.getText(self, "Set sub-plot title", "Title:")
            if ok and name:
                mdi.setWindowTitle(name)

        action = QtGui.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        w.setWindowTitle(f"GPS {self._window_counter}")
        self._window_counter += 1

        if self.subplots_link:
            gps.timestamp_changed_signal.connect(self.set_cursor)

        self.windows_modified.emit()

    def _add_lin_bus_trace_window(self, ranges=None):
        if self.mdf.version < "4.00":
            return

        dfs = []
        groups_count = len(self.mdf.groups)

        for index in range(groups_count):
            group = self.mdf.groups[index]
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                source = group.channel_group.acq_source

                names = [ch.name for ch in group.channels]

                if source and source.bus_type == v4c.BUS_TYPE_LIN:
                    if "LIN_Frame" in names:
                        data = self.mdf.get("LIN_Frame", index, raw=True)

                    elif "LIN_SyncError" in names:
                        data = self.mdf.get("LIN_SyncError", index, raw=True)

                    elif "LIN_TransmissionError" in names:
                        data = self.mdf.get("LIN_TransmissionError", index, raw=True)

                    elif "LIN_ChecksumError" in names:
                        data = self.mdf.get("LIN_ChecksumError", index, raw=True)

                    elif "LIN_ReceiveError" in names:
                        data = self.mdf.get("LIN_ReceiveError", index, raw=True)

                    df_index = data.timestamps
                    count = len(df_index)

                    columns = {
                        "timestamps": df_index,
                        "Bus": np.full(count, "Unknown", dtype="O"),
                        "ID": np.full(count, 0xFFFFFFFF, dtype="u4"),
                        "Direction": np.full(count, "", dtype="O"),
                        "Name": np.full(count, "", dtype="O"),
                        "Event Type": np.full(count, "LIN Frame", dtype="O"),
                        "Details": np.full(count, "", dtype="O"),
                        "Received Byte Count": np.zeros(count, dtype="u1"),
                        "Data Length": np.zeros(count, dtype="u1"),
                        "Data Bytes": np.full(count, "", dtype="O"),
                    }

                    frame_map = None
                    if data.attachment and data.attachment[0]:
                        dbc = load_can_database(data.attachment[1], data.attachment[0])
                        if dbc:
                            frame_map = {frame.arbitration_id.id: frame.name for frame in dbc}

                            for name in frame_map.values():
                                sys.intern(name)

                    if data.name == "LIN_Frame":
                        vals = data["LIN_Frame.BusChannel"].astype("u1")
                        vals = [f"LIN {chn}" for chn in vals.tolist()]
                        columns["Bus"] = vals

                        vals = data["LIN_Frame.ID"].astype("u1") & 0x3F
                        columns["ID"] = vals
                        if frame_map:
                            columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        columns["Received Byte Count"] = data["LIN_Frame.ReceivedDataByteCount"].astype("u1")
                        data_length = data["LIN_Frame.DataLength"].astype("u1").tolist()
                        columns["Data Length"] = data_length

                        vals = csv_bytearray2hex(
                            pd.Series(list(data["LIN_Frame.DataBytes"])),
                            data_length,
                        )
                        columns["Data Bytes"] = vals

                        if "LIN_Frame.Dir" in names:
                            if data["LIN_Frame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [v.decode("utf-8") for v in data["LIN_Frame.Dir"].tolist()]
                            else:
                                columns["Direction"] = [
                                    "TX" if dir else "RX" for dir in data["LIN_Frame.Dir"].astype("u1").tolist()
                                ]

                        vals = None
                        data_length = None

                    elif data.name == "LIN_SyncError":
                        names = set(data.samples.dtype.names)

                        if "LIN_SyncError.BusChannel" in names:
                            vals = data["LIN_SyncError.BusChannel"].astype("u1")
                            vals = [f"LIN {chn}" for chn in vals.tolist()]
                            columns["Bus"] = vals

                        if "LIN_SyncError.BaudRate" in names:
                            vals = data["LIN_SyncError.BaudRate"]
                            unique = np.unique(vals).tolist()
                            for val in unique:
                                sys.intern(f"Baudrate {val}")
                            vals = [f"Baudrate {val}" for val in vals.tolist()]
                            columns["Details"] = vals

                        columns["Event Type"] = "Sync Error Frame"

                        vals = None
                        data_length = None

                    elif data.name == "LIN_TransmissionError":
                        names = set(data.samples.dtype.names)

                        if "LIN_TransmissionError.BusChannel" in names:
                            vals = data["LIN_TransmissionError.BusChannel"].astype("u1")
                            vals = [f"LIN {chn}" for chn in vals.tolist()]
                            columns["Bus"] = vals

                        if "LIN_TransmissionError.BaudRate" in names:
                            vals = data["LIN_TransmissionError.BaudRate"]
                            unique = np.unique(vals).tolist()
                            for val in unique:
                                sys.intern(f"Baudrate {val}")
                            vals = [f"Baudrate {val}" for val in vals.tolist()]
                            columns["Details"] = vals

                        vals = data["LIN_TransmissionError.ID"].astype("u1") & 0x3F
                        columns["ID"] = vals
                        if frame_map:
                            columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        columns["Event Type"] = "Transmission Error Frame"
                        columns["Direction"] = ["TX"] * count

                        vals = None

                    elif data.name == "LIN_ReceiveError":
                        names = set(data.samples.dtype.names)

                        if "LIN_ReceiveError.BusChannel" in names:
                            vals = data["LIN_ReceiveError.BusChannel"].astype("u1")
                            vals = [f"LIN {chn}" for chn in vals.tolist()]
                            columns["Bus"] = vals

                        if "LIN_ReceiveError.BaudRate" in names:
                            vals = data["LIN_ReceiveError.BaudRate"]
                            unique = np.unique(vals).tolist()
                            for val in unique:
                                sys.intern(f"Baudrate {val}")
                            vals = [f"Baudrate {val}" for val in vals.tolist()]
                            columns["Details"] = vals

                        if "LIN_ReceiveError.ID" in names:
                            vals = data["LIN_ReceiveError.ID"].astype("u1") & 0x3F
                            columns["ID"] = vals
                            if frame_map:
                                columns["Name"] = [frame_map[_id] for _id in vals]

                        columns["Event Type"] = "Receive Error Frame"

                        columns["Direction"] = ["RX"] * count

                        vals = None

                    elif data.name == "LIN_ChecksumError":
                        names = set(data.samples.dtype.names)

                        if "LIN_ChecksumError.BusChannel" in names:
                            vals = data["LIN_ChecksumError.BusChannel"].astype("u1")
                            vals = [f"LIN {chn}" for chn in vals.tolist()]
                            columns["Bus"] = vals

                        if "LIN_ChecksumError.Checksum" in names:
                            vals = data["LIN_ChecksumError.Checksum"]
                            unique = np.unique(vals).tolist()
                            for val in unique:
                                sys.intern(f"Baudrate {val}")
                            vals = [f"Checksum 0x{val:02X}" for val in vals.tolist()]
                            columns["Details"] = vals

                        if "LIN_ChecksumError.ID" in names:
                            vals = data["LIN_ChecksumError.ID"].astype("u1") & 0x3F
                            columns["ID"] = vals
                            if frame_map:
                                columns["Name"] = [frame_map[_id] for _id in vals]

                        if "LIN_ChecksumError.DataBytes" in names:
                            data_length = data["LIN_ChecksumError.DataLength"].astype("u1").tolist()
                            columns["Data Length"] = data_length

                            vals = csv_bytearray2hex(
                                pd.Series(list(data["LIN_ChecksumError.DataBytes"])),
                                data_length,
                            )
                            columns["Data Bytes"] = vals

                        columns["Event Type"] = "Checksum Error Frame"

                        if "LIN_ChecksumError.Dir" in names:
                            columns["Direction"] = [
                                "TX" if dir else "RX" for dir in data["LIN_ChecksumError.Dir"].astype("u1").tolist()
                            ]

                        vals = None

                    dfs.append(pd.DataFrame(columns, index=df_index))

        if not dfs:
            return
        else:
            signals = pd.concat(dfs).sort_index()
            index = pd.Index(range(len(signals)))
            signals.set_index(index, inplace=True)

        del dfs

        trace = LINBusTrace(signals, start=self.mdf.header.start_time, range=ranges)

        sub = MdiSubWindow(parent=self)
        sub.setWidget(trace)
        trace.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/bus_lin.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
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
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        w.layout().setSpacing(1)

        def set_title(mdi):
            name, ok = QtWidgets.QInputDialog.getText(self, "Set sub-plot title", "Title:")
            if ok and name:
                mdi.setWindowTitle(name)

        action = QtGui.QAction("Set title", menu)
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

    def _add_numeric_window(self, names):
        if names and isinstance(names[0], str):
            signals_ = [
                (
                    name,
                    *self.mdf.whereis(name)[0],
                    self.uuid,
                    "channel",
                    [],
                    os.urandom(6).hex(),
                )
                for name in names
                if name in self.mdf
            ]

        else:
            flatten_entries = get_flatten_entries_from_mime(names)

            uuids = {entry["origin_uuid"] for entry in flatten_entries}

            for uuid in uuids:
                if self.file_by_uuid(uuid):
                    break
            else:
                names = substitude_mime_uuids(names, uuid=self.uuid, force=True)
                flatten_entries = get_flatten_entries_from_mime(names)

            signals_ = [
                entry for entry in flatten_entries if (entry["group_index"], entry["channel_index"]) != (-1, -1)
            ]

        signals_ = natsorted(signals_)

        uuids = {entry["origin_uuid"] for entry in signals_}

        signals = []

        for uuid in uuids:
            uuids_signals = [
                (entry["name"], entry["group_index"], entry["channel_index"])
                for entry in signals_
                if entry["origin_uuid"] == uuid and entry["group_index"] != NOT_FOUND
            ]

            not_found = [
                (entry["name"], entry["group_index"], entry["channel_index"])
                for entry in signals_
                if entry["origin_uuid"] == uuid and entry["group_index"] == NOT_FOUND
            ]

            uuids_signals_objs = [
                entry for entry in signals_ if entry["origin_uuid"] == uuid and entry["group_index"] != NOT_FOUND
            ]

            not_found_objs = [
                entry for entry in signals_ if entry["origin_uuid"] == uuid and entry["group_index"] == NOT_FOUND
            ]

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

            for sig, sig_, sig_obj in zip(selected_signals, uuids_signals, uuids_signals_objs):
                sig.group_index = sig_[1]
                sig.channel_index = sig_[2]
                sig.flags &= ~sig.Flags.computed
                sig.computation = {}
                sig.origin_uuid = uuid
                sig.name = sig_[0] or sig.name
                sig.ranges = sig_obj["ranges"]
                sig.uuid = sig_obj["uuid"]

                if not hasattr(self, "mdf"):
                    # MainWindow => comparison plots

                    sig.tooltip = f"{sig.name}\n@ {file.file_name}"
                    sig.name = f"{file_index+1}: {sig.name}"

            signals.extend(selected_signals)

            for pattern_group in get_pattern_groups(names):
                file_info = self.file_by_uuid(uuid)
                if not file_info:
                    continue

                file_index, file = file_info

                signals.extend(
                    extract_signals_using_pattern(
                        mdf=file.mdf,
                        channels_db=None,
                        pattern_info=pattern_group["pattern"],
                        ignore_value2text_conversions=file.ignore_value2text_conversions,
                        uuid=file.uuid,
                    ).values()
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
                    signal.samples = csv_bytearray2hex(pd.Series(list(signal.samples)), length)

                if signal.name.endswith(".ID"):
                    signal.samples = signal.samples.astype("<u4") & 0x1FFFFFFF

            not_found = [Signal([], [], name=name) for name, gp_index, ch_index in not_found]
            uuid = os.urandom(6).hex()
            for sig, sig_obj in zip(not_found, not_found_objs):
                sig.origin_uuid = uuid
                sig.group_index = NOT_FOUND
                sig.channel_index = randint(0, NOT_FOUND)
                sig.exists = False

                ranges = sig_obj["ranges"]
                for range in ranges:
                    range["font_color"] = QtGui.QBrush(QtGui.QColor(range["font_color"]))
                    range["background_color"] = QtGui.QBrush(QtGui.QColor(range["background_color"]))
                sig.ranges = ranges
                sig.format = sig_obj["format"]

            signals.extend(not_found)

            signals = natsorted(signals, key=lambda x: x.name)

        numeric = Numeric([], parent=self, mode="offline")

        numeric.show()
        numeric.hide()

        sub = MdiSubWindow(parent=self)
        sub.setWidget(numeric)
        numeric.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
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
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        w.layout().setSpacing(1)

        menu = w.systemMenu()

        action = QtGui.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        w.setWindowTitle(generate_window_title(w, "Numeric"))

        numeric.add_channels_request.connect(partial(self.add_new_channels, widget=numeric))
        if self.subplots_link:
            numeric.timestamp_changed_signal.connect(self.set_cursor)

        numeric.add_new_channels(signals)
        numeric.show()

        self.windows_modified.emit()

    def _add_plot_window(self, signals, disable_new_channels=False):
        if signals and isinstance(signals[0], str):
            mime_data = [
                {
                    "name": name,
                    "group_index": self.mdf.whereis(name)[0][0],
                    "channel_index": self.mdf.whereis(name)[0][0],
                    "origin_uuid": self.uuid,
                    "type": "channel",
                    "ranges": [],
                    "uuid": os.urandom(6).hex(),
                    "enabled": not disable_new_channels,
                }
                for name in signals
                if name in self.mdf
            ]
        else:
            mime_data = signals

        flatten_entries = get_flatten_entries_from_mime(mime_data)
        uuids = {entry["origin_uuid"] for entry in flatten_entries}

        for uuid in uuids:
            if self.file_by_uuid(uuid):
                break
        else:
            mime_data = substitude_mime_uuids(mime_data, uuid=self.uuid, force=True)
            flatten_entries = get_flatten_entries_from_mime(mime_data)

        for entry in flatten_entries:
            entry["enabled"] = not disable_new_channels

        signals_ = {
            entry["uuid"]: entry
            for entry in flatten_entries
            if (entry["group_index"], entry["channel_index"]) not in ((-1, -1), (NOT_FOUND, NOT_FOUND))
        }

        not_found = {
            entry["uuid"]: entry
            for entry in flatten_entries
            if (entry["group_index"], entry["channel_index"]) == (NOT_FOUND, NOT_FOUND)
        }

        computed = {
            entry["uuid"]: entry
            for entry in flatten_entries
            if (entry["group_index"], entry["channel_index"]) == (-1, -1)
        }

        uuids = {entry["origin_uuid"] for entry in signals_.values()}

        signals = {}

        for uuid in uuids:
            uuids_signals = {key: entry for key, entry in signals_.items() if entry["origin_uuid"] == uuid}

            file_info = self.file_by_uuid(uuid)
            if not file_info:
                continue

            file_index, file = file_info

            selected_signals = file.mdf.select(
                [(entry["name"], entry["group_index"], entry["channel_index"]) for entry in uuids_signals.values()],
                ignore_value2text_conversions=self.ignore_value2text_conversions,
                copy_master=False,
                validate=True,
                raw=True,
            )

            for sig, (sig_uuid, sig_) in zip(selected_signals, uuids_signals.items()):
                sig.group_index = sig_["group_index"]
                sig.channel_index = sig_["channel_index"]
                sig.flags &= ~sig.Flags.computed
                sig.computation = {}
                sig.origin_uuid = uuid
                sig.name = sig_["name"] or sig.name
                sig.uuid = sig_uuid
                if "color" in sig_:
                    sig.color = sig_["color"]

                sig.ranges = sig_["ranges"]
                sig.enable = sig_["enabled"]

                if not hasattr(self, "mdf"):
                    # MainWindow => comparison plots

                    sig.tooltip = f"{sig.name}\n@ {file.file_name}"
                    sig.name = f"{file_index+1}: {sig.name}"

                signals[sig_uuid] = sig

            nd = {
                key: sig
                for key, sig in signals.items()
                if sig.samples.dtype.kind not in "SU" and (sig.samples.dtype.names or len(sig.samples.shape) > 1)
            }

            signals = {
                key: sig
                for key, sig in signals.items()
                if sig.samples.dtype.kind not in "SU" and not sig.samples.dtype.names and not len(sig.samples.shape) > 1
            }

            for sig in nd.values():
                if sig.samples.dtype.names is None:
                    shape = sig.samples.shape[1:]

                    matrix_dims = [list(range(dim)) for dim in shape]

                    matrix_name = sig.name

                    for indexes in itertools.product(*matrix_dims):
                        indexes_string = "".join(f"[{_index}]" for _index in indexes)

                        samples = sig.samples
                        for idx in indexes:
                            samples = samples[:, idx]
                        sig_name = f"{matrix_name}{indexes_string}"

                        new_sig = sig.copy()
                        new_sig.name = sig_name
                        new_sig.samples = samples
                        new_sig.group_index = sig.group_index
                        new_sig.channel_index = sig.channel_index
                        new_sig.flags &= ~sig.Flags.computed
                        new_sig.computation = {}
                        new_sig.origin_uuid = sig.origin_uuid
                        new_sig.uuid = os.urandom(6).hex()
                        new_sig.enable = getattr(sig, "enable", True)

                        signals[new_sig.uuid] = new_sig
                else:
                    name = sig.samples.dtype.names[0]
                    if name == sig.name:
                        array_samples = sig.samples[name]

                        shape = array_samples.shape[1:]

                        matrix_dims = [list(range(dim)) for dim in shape]

                        matrix_name = sig.name

                        for indexes in itertools.product(*matrix_dims):
                            indexes_string = "".join(f"[{_index}]" for _index in indexes)

                            samples = array_samples
                            for idx in indexes:
                                samples = samples[:, idx]
                            sig_name = f"{matrix_name}{indexes_string}"

                            new_sig = sig.copy()
                            new_sig.name = sig_name
                            new_sig.samples = samples
                            new_sig.group_index = sig.group_index
                            new_sig.channel_index = sig.channel_index
                            new_sig.flags &= ~sig.Flags.computed
                            new_sig.computation = {}
                            new_sig.origin_uuid = sig.origin_uuid
                            new_sig.uuid = os.urandom(6).hex()
                            new_sig.enable = getattr(sig, "enable", True)

                            signals[new_sig.uuid] = new_sig

            for signal in signals.values():
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
                    signal.samples = csv_bytearray2hex(pd.Series(list(signal.samples)), length.astype("u2"))

                if signal.name.endswith(".ID"):
                    signal.samples = signal.samples.astype("<u4") & 0x1FFFFFFF

        for uuid, sig_ in not_found.items():
            sig = Signal([], [], name=sig_["name"])
            sig.uuid = uuid
            sig.flags &= ~sig.Flags.computed
            sig.computation = {}
            sig.origin_uuid = sig_.get("origin_uuid", self.uuid)
            sig.origin_uuid = self.uuid
            sig.group_index = NOT_FOUND
            sig.channel_index = NOT_FOUND
            sig.enable = sig_["enabled"]
            if "color" in sig_:
                sig.color = sig_["color"]

            if sig_["flags"] & Signal.Flags.user_defined_conversion:
                sig.conversion = from_dict(sig_["conversion"])
                sig.flags |= Signal.Flags.user_defined_conversion

            if sig_["flags"] & Signal.Flags.user_defined_name:
                sig.original_name = sig_.name
                sig.name = sig_.get("user_defined_name", "") or ""
                sig.flags |= Signal.Flags.user_defined_name

            signals[uuid] = sig

        if computed:
            measured_signals = {sig.uuid: sig for sig in signals.values()}
            if measured_signals:
                all_timebase = np.unique(
                    np.concatenate(
                        list({id(sig.timestamps): sig.timestamps for sig in measured_signals.values()}.values())
                    )
                )
            else:
                all_timebase = []

            required_channels = []

            for ch in computed.values():
                required_channels.extend(get_required_from_computed(ch))

            required_channels = set(required_channels)
            required_channels_list = [
                (channel, *self.mdf.whereis(channel)[0]) for channel in required_channels if channel in self.mdf
            ]

            required_channels = {}

            for sig in self.mdf.select(
                required_channels_list,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
                copy_master=False,
            ):
                required_channels[sig.name] = sig

            required_channels.update(measured_signals)

            computed_signals = {}

            for channel in computed.values():
                computation = channel["computation"]

                signal = compute_signal(
                    computation,
                    required_channels,
                    all_timebase,
                    self.functions,
                )
                signal.name = channel["name"]
                signal.unit = channel["unit"]
                signal.color = channel["color"]
                signal.enable = channel["enabled"]
                signal.flags |= signal.Flags.computed
                signal.computation = channel["computation"]
                signal.group_index = -1
                signal.channel_index = -1
                signal.origin_uuid = self.uuid
                signal.comment = channel["computation"].get("channel_comment", "")
                signal.uuid = channel.get("uuid", os.urandom(6).hex())

                if channel["flags"] & Signal.Flags.user_defined_conversion:
                    signal.conversion = from_dict(channel["conversion"])
                    signal.flags |= signal.Flags.user_defined_conversion

                if channel["flags"] & Signal.Flags.user_defined_name:
                    signal.original_name = channel.name
                    signal.name = channel.get("user_defined_name", "") or ""
                    signal.flags |= signal.Flags.user_defined_name

                computed_signals[signal.uuid] = signal

            signals.update(computed_signals)

        if hasattr(self, "mdf"):
            events = []
            origin = self.mdf.start_time

            if self.mdf.version >= "4.00":
                mdf_events = list(self.mdf.events)

                for pos, event in enumerate(mdf_events):
                    event_info = {}
                    event_info["value"] = event.value
                    event_info["type"] = v4c.EVENT_TYPE_TO_STRING[event.event_type]

                    if event.name:
                        description = event.name
                    else:
                        description = ""

                    if event.comment:
                        try:
                            comment = extract_xml_comment(event.comment)
                        except:
                            comment = event.comment
                        if description:
                            description = f"{description} ({comment})"
                        else:
                            description = comment
                    event_info["description"] = description
                    event_info["index"] = pos
                    event_info["tool"] = event.tool

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
                            "type": v4c.EVENT_TYPE_TO_STRING[v4c.EVENT_TYPE_TRIGGER],
                        }
                        events.append(event)
        else:
            events = []
            if isinstance(self.files, QtWidgets.QMdiArea):
                origin = self.files.subWindowList()[0].widget().mdf.start_time
            else:
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
            show_cursor_circle=self.cursor_circle,
            show_cursor_horizontal_line=self.cursor_horizontal_line,
            cursor_line_width=self.cursor_line_width,
            cursor_color=self.cursor_color,
            owner=self,
        )
        plot.pattern_group_added.connect(self.add_pattern_group)
        plot.verify_bookmarks.connect(self.verify_bookmarks)
        plot.pattern = {}

        sub = MdiSubWindow(parent=self)
        sub.setWidget(plot)
        plot.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
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
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        w.layout().setSpacing(1)

        plot.show()

        menu = w.systemMenu()

        action = QtGui.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        w.setWindowTitle(generate_window_title(w, "Plot"))

        if self.subplots_link:
            for i, mdi in enumerate(self.mdi_area.subWindowList()):
                try:
                    viewbox = mdi.widget().plot.viewbox
                    if plot.plot.viewbox is not viewbox:
                        plot.plot.viewbox.setXLink(viewbox)
                    break
                except:
                    continue

        plot.add_channels_request.connect(partial(self.add_new_channels, widget=plot))
        plot.edit_channel_request.connect(partial(self.edit_channel, widget=plot))

        plot.show_properties.connect(self._show_info)

        plot.add_new_channels(signals, mime_data)
        self.set_subplots_link(self.subplots_link)

        iterator = QtWidgets.QTreeWidgetItemIterator(plot.channel_selection)
        while item := iterator.value():
            iterator += 1

            if item.type() == item.Group:
                if item.pattern:
                    plot.pattern_group_added.emit(plot, item)

        if len(plot.plot.all_timebase):
            start = plot.plot.all_timebase[0]
            stop = plot.plot.all_timebase[-1]

            if start == stop:
                padding = 1
            else:
                padding = (stop - start) * 0.05

            # plot.plot.viewbox.setXRange(start - padding, stop + padding, padding=0)

        self.windows_modified.emit()

        return w, plot

    def _add_tabular_window(self, names):
        if names and isinstance(names[0], str):
            signals_ = [
                (
                    name,
                    *self.mdf.whereis(name)[0],
                    self.uuid,
                    "channel",
                    [],
                    os.urandom(6).hex(),
                )
                for name in names
                if name in self.mdf
            ]
        else:
            flatten_entries = get_flatten_entries_from_mime(names)

            uuids = {entry["origin_uuid"] for entry in flatten_entries}

            for uuid in uuids:
                if self.file_by_uuid(uuid):
                    break
            else:
                names = substitude_mime_uuids(names, uuid=self.uuid, force=True)
                flatten_entries = get_flatten_entries_from_mime(names)

            signals_ = [
                entry
                for entry in flatten_entries
                if (entry["group_index"], entry["channel_index"]) != (NOT_FOUND, NOT_FOUND)
            ]

        signals_ = natsorted(signals_)

        uuids = {entry["origin_uuid"] for entry in signals_}

        dfs = []
        ranges = {}
        start = []

        for uuid in uuids:
            uuids_signals = [
                (entry["name"], entry["group_index"], entry["channel_index"])
                for entry in signals_
                if entry["origin_uuid"] == uuid
            ]

            file_info = self.file_by_uuid(uuid)
            if not file_info:
                continue

            file_index, file = file_info

            if not hasattr(self, "mdf"):
                # MainWindow => comparison plots

                ranges.update(
                    {
                        f"{file_index+1}: {entry['name']}": entry["ranges"]
                        for entry in signals_
                        if entry["origin_uuid"] == uuid
                    }
                )
            else:
                ranges.update({entry["name"]: entry["ranges"] for entry in signals_ if entry["origin_uuid"] == uuid})

            start.append(file.mdf.header.start_time)

            for pattern_group in get_pattern_groups(names):
                uuids_signals.extend(
                    [
                        (sig.name, sig.group_index, sig.channel_index)
                        for sig in extract_signals_using_pattern(
                            mdf=file.mdf,
                            channels_db=None,
                            pattern_info=pattern_group["pattern"],
                            ignore_value2text_conversions=file.ignore_value2text_conversions,
                            uuid=file.uuid,
                        ).values()
                    ]
                )

            uuids_signals = [entry for entry in uuids_signals if entry[2] != file.mdf.masters_db.get(entry[1], None)]

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
                ".ID",
            ):
                signals[name] = signals[name].astype("<u4") & 0x1FFFFFFF

        tabular = Tabular(signals, start=start, parent=self, ranges=ranges)

        sub = MdiSubWindow(parent=self)
        sub.setWidget(tabular)
        tabular.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
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
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        w.layout().setSpacing(1)

        action = QtGui.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        w.setWindowTitle(generate_window_title(w, "Tabular"))

        if self.subplots_link:
            tabular.timestamp_changed_signal.connect(self.set_cursor)

        tabular.add_channels_request.connect(partial(self.add_new_channels, widget=tabular))

        tabular.tree.auto_size_header()

        self.windows_modified.emit()

    def clear_windows(self):
        for window in self.mdi_area.subWindowList():
            widget = window.widget()
            self.mdi_area.removeSubWindow(window)
            widget.setParent(None)
            window.close()
            widget.deleteLater()
            widget.close()

    def delete_functions(self, deleted_functions):
        deleted = set()
        for info in deleted_functions:
            del self.functions[info["name"]]
            deleted.add(info["name"])

        for mdi in self.mdi_area.subWindowList():
            wid = mdi.widget()
            if isinstance(wid, Plot):
                iterator = QtWidgets.QTreeWidgetItemIterator(wid.channel_selection)
                while item := iterator.value():
                    if item.type() == item.Channel:
                        if item.signal.flags & item.signal.Flags.computed:
                            if item.signal.computation["function"] in deleted:
                                self.edit_channel(wid.channel_item_to_config(item), item, wid)

                    iterator += 1

    def edit_channel(self, channel, item, widget):
        required_channels = set(get_required_from_computed(channel))

        required_channels = [
            (channel, *self.mdf.whereis(channel)[0]) for channel in required_channels if channel in self.mdf
        ]
        required_channels = {
            sig.name: sig
            for sig in self.mdf.select(
                required_channels,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
                copy_master=False,
            )
        }

        if required_channels:
            all_timebase = np.unique(
                np.concatenate(
                    list({id(sig.timestamps): sig.timestamps for sig in required_channels.values()}.values())
                )
            )
        else:
            all_timebase = []

        computation = channel["computation"]

        signal = compute_signal(
            computation,
            required_channels,
            all_timebase,
            self.functions,
        )
        signal.name = channel["name"]
        signal.unit = channel["unit"]
        signal.color = channel["color"]
        signal.flags |= signal.Flags.computed
        signal.computation = channel["computation"]
        signal.group_index = -1
        signal.channel_index = -1
        signal.origin_uuid = self.uuid
        signal.comment = channel["computation"].get("channel_comment", "")
        signal.uuid = channel.get("uuid", os.urandom(6).hex())

        if channel["flags"] & Signal.Flags.user_defined_conversion:
            signal.conversion = from_dict(channel["conversion"])
            signal.flags |= signal.Flags.user_defined_conversion

        if channel["flags"] & Signal.Flags.user_defined_name:
            signal.original_name = channel.name
            signal.name = channel.get("user_defined_name", "") or ""
            signal.flags |= signal.Flags.user_defined_name

        old_name = item.name
        new_name = signal.name
        uuid = item.uuid

        item.signal.samples = item.signal.raw_samples = item.signal.phys_samples = signal.samples
        item.signal.timestamps = signal.timestamps
        item.signal.trim(force=True)
        item.signal.computation = signal.computation
        item.signal._compute_basic_stats()

        item.setToolTip(item.NameColumn, f"{signal.name}\n{signal.comment}")

        if old_name != new_name:
            item.name = new_name

        if item.unit != signal.unit:
            item.unit = signal.unit
        widget.cursor_moved()
        widget.range_modified()

        widget.plot.update()

        for channel in widget.plot.signals:
            if channel.uuid == uuid:
                continue

            if channel.flags & channel.Flags.computed:
                required_channels = set(get_required_from_computed(channel.computation))
                if old_name in required_channels:
                    item = widget.item_by_uuid

                    computed_channel = widget.plot.channel_item_to_config(item)
                    computed_channel["computation"] = replace_computation_dependency(
                        computed_channel["computation"], old_name, new_name
                    )

                    widget.edit_channel_request.emit(computed_channel, item)

    def get_current_widget(self):
        mdi = self.mdi_area.currentSubWindow()
        if mdi is not None:
            widget = mdi.widget()

            return widget
        else:
            return None

    def load_window(self, window_info):
        functions = {
            "Numeric": self._load_numeric_window,
            "Plot": self._load_plot_window,
            "GPS": self._load_gps_window,
            "Tabular": self._load_tabular_window,
            "CAN Bus Trace": self._load_can_bus_trace_window,
            "FlexRay Bus Trace": self._load_flexray_bus_trace_window,
            "LIN Bus Trace": self._load_lin_bus_trace_window,
        }

        if window_info["type"] not in functions:
            self.unknown_windows.append(window_info)

        else:
            load_window_function = functions[window_info["type"]]

            w, pattern_info = load_window_function(window_info)

            if w:
                if self._frameless_windows:
                    w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

                if pattern_info:
                    icon = QtGui.QIcon()
                    icon.addPixmap(
                        QtGui.QPixmap(":/filter.png"),
                        QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off,
                    )
                    w.setWindowIcon(icon)

                w.layout().setSpacing(1)

                self.windows_modified.emit()

    def _load_numeric_window(self, window_info):
        uuid = self.uuid
        geometry = window_info.get("geometry", None)

        # patterns
        pattern_info = window_info["configuration"].get("pattern", {})
        if pattern_info:
            signals = extract_signals_using_pattern(
                mdf=self.mdf,
                channels_db=None,
                pattern_info=pattern_info,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
                uuid=self.uuid,
            )

            signals = list(signals.values())

            for sig in signals:
                sig.origin_uuid = uuid
                sig.computation = None
                sig.ranges = []

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
                range["font_color"] = QtGui.QBrush(QtGui.QColor(range["font_color"]))
                range["background_color"] = QtGui.QBrush(QtGui.QColor(range["background_color"]))

            pattern_info["ranges"] = ranges

        else:
            required = window_info["configuration"]["channels"]

            found = [elem for elem in required if elem["name"] in self.mdf]

            signals_ = [(elem["name"], *self.mdf.whereis(elem["name"])[0]) for elem in found]

            if not signals_:
                return None, False

            signals = self.mdf.select(
                signals_,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
                copy_master=False,
                validate=True,
                raw=True,
            )

            for sig, sig_, description in zip(signals, signals_, found):
                sig.group_index = sig_[1]
                sig.channel_index = sig_[2]
                sig.origin_uuid = uuid
                sig.computation = None
                ranges = description["ranges"]
                for range in ranges:
                    range["font_color"] = QtGui.QBrush(QtGui.QColor(range["font_color"]))
                    range["background_color"] = QtGui.QBrush(QtGui.QColor(range["background_color"]))
                sig.ranges = ranges
                sig.format = description["format"]

            signals = [sig for sig in signals if not sig.samples.dtype.names and len(sig.samples.shape) <= 1]

            signals = natsorted(signals, key=lambda x: x.name)

            found = {sig.name for sig in signals}
            required = {description["name"] for description in required}
            not_found = [Signal([], [], name=name) for name in sorted(required - found)]
            uuid = os.urandom(6).hex()
            for sig in not_found:
                sig.origin_uuid = uuid
                sig.group_index = 0
                sig.ranges = []

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
        numeric.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
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

        if window_info.get("maximized", False):
            w.showMaximized()
        elif window_info.get("minimized", False):
            w.showMinimized()

        w.setWindowTitle(generate_window_title(w, window_info["type"], window_info["title"]))

        numeric.add_new_channels(signals)

        menu = w.systemMenu()

        action = QtGui.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        numeric.add_channels_request.connect(partial(self.add_new_channels, widget=numeric))

        if self.subplots_link:
            numeric.timestamp_changed_signal.connect(self.set_cursor)

        sections_width = window_info["configuration"].get("header_sections_width", [])
        if sections_width:
            sections_width = reversed(list(enumerate(sections_width)))
            for column_index, width in sections_width:
                numeric.channels.columnHeader.setColumnWidth(column_index, width)
                numeric.channels.dataView.setColumnWidth(
                    column_index,
                    numeric.channels.columnHeader.columnWidth(column_index),
                )

        font_size = window_info["configuration"].get("font_size", numeric.font().pointSize())
        numeric.set_font_size(font_size)

        return w, pattern_info

    def _load_gps_window(self, window_info):
        uuid = self.uuid
        geometry = window_info.get("geometry", None)

        signals_ = [
            (None, *self.mdf.whereis(name)[0])
            for name in (
                window_info["configuration"]["latitude_channel"],
                window_info["configuration"]["longitude_channel"],
            )
            if name in self.mdf
        ]

        if len(signals_) != 2:
            return None, False

        latitude, longitude = self.mdf.select(
            signals_,
            copy_master=False,
            validate=True,
            raw=False,
        )

        gps = GPS(latitude, longitude, zoom=window_info["configuration"]["zoom"])

        sub = MdiSubWindow(parent=self)
        sub.setWidget(gps)
        gps.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
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

        w.setWindowTitle(generate_window_title(w, window_info["type"], window_info["title"]))

        if window_info.get("maximized", False):
            w.showMaximized()
        elif window_info.get("minimized", False):
            w.showMinimized()

        menu = w.systemMenu()

        action = QtGui.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        if self.subplots_link:
            gps.timestamp_changed_signal.connect(self.set_cursor)

        return w, False

    def _load_plot_window(self, window_info):
        geometry = window_info.get("geometry", None)

        # patterns
        pattern_info = window_info["configuration"].get("pattern", {})
        if pattern_info:
            plot_signals = extract_signals_using_pattern(
                mdf=self.mdf,
                channels_db=None,
                pattern_info=pattern_info,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
                uuid=self.uuid,
            )

            mime_data = None
            descriptions = {}

        else:
            (
                mime_data,
                descriptions,
                found,
                not_found,
                computed,
            ) = build_mime_from_config(window_info["configuration"]["channels"], self.mdf, self.uuid)

            plot_signals = {}
            measured_signals = {}

            for (sig_uuid, entry), signal in zip(
                found.items(),
                self.mdf.select(
                    list(found.values()),
                    ignore_value2text_conversions=self.ignore_value2text_conversions,
                    copy_master=False,
                    validate=True,
                    raw=True,
                ),
            ):
                description = descriptions[sig_uuid]

                signal.flags &= ~signal.Flags.computed
                signal.computation = {}
                signal.color = description["color"]
                signal.group_index = entry[1]
                signal.channel_index = entry[2]
                signal.origin_uuid = self.uuid
                signal.name = entry[0]
                signal.mode = description.get("mode", "phys")
                signal.uuid = sig_uuid

                measured_signals[signal.name] = signal
                plot_signals[sig_uuid] = signal

            matrix_components = []
            for nf_name in not_found:
                name, indexes = parse_matrix_component(nf_name)
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
            for signal_mat, (_n, indexes) in zip(matrix_signals.values(), matrix_components):
                indexes_string = "".join(f"[{_index}]" for _index in indexes)
                sig_name = f"{signal_mat.name}{indexes_string}"

                if sig_name in not_found:
                    signal = deepcopy(signal_mat)
                    samples = signal.samples
                    if samples.dtype.names:
                        samples = samples[signal.name]

                    if len(samples.shape) <= len(indexes):
                        # samples does not have enough dimensions
                        continue

                    for idx in indexes:
                        samples = samples[:, idx]

                    signal.samples = samples

                    sig_uuid = not_found[sig_name]

                    description = descriptions[sig_uuid]

                    signal.color = description["color"]
                    signal.flags &= ~signal.Flags.computed
                    signal.computation = {}
                    signal.origin_uuid = self.uuid
                    signal.name = sig_name
                    signal.mode = description.get("mode", "phys")
                    signal.uuid = sig_uuid

                    measured_signals[signal.name] = signal

                    plot_signals[sig_uuid] = signal

            measured_signals.update(new_matrix_signals)

            if measured_signals:
                all_timebase = np.unique(
                    np.concatenate(
                        list({id(sig.timestamps): sig.timestamps for sig in measured_signals.values()}.values())
                    )
                )
            else:
                all_timebase = []

            required_channels = []
            for ch in computed.values():
                required_channels.extend(get_required_from_computed(ch))

            required_channels = set(required_channels)

            required_channels = [
                (channel, *self.mdf.whereis(channel)[0])
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

            for sig_uuid, channel in computed.items():
                computation = channel["computation"]

                signal = compute_signal(
                    computation,
                    required_channels,
                    all_timebase,
                    self.functions,
                )
                signal.color = channel["color"]
                signal.flags |= signal.Flags.computed
                signal.computation = channel["computation"]
                signal.name = channel["name"]
                signal.unit = channel["unit"]
                signal.group_index = -1
                signal.channel_index = -1
                signal.origin_uuid = self.uuid
                signal.uuid = sig_uuid

                if channel["flags"] & Signal.Flags.user_defined_conversion:
                    signal.conversion = from_dict(channel["conversion"])
                    signal.flags |= signal.Flags.user_defined_conversion

                if channel["flags"] & Signal.Flags.user_defined_name:
                    signal.original_name = channel["name"]
                    signal.name = channel.get("user_defined_name", "") or ""
                    signal.flags |= signal.Flags.user_defined_name

                plot_signals[sig_uuid] = signal

        signals = {
            sig_uuid: sig
            for sig_uuid, sig in plot_signals.items()
            if sig.samples.dtype.kind not in "SU" and not sig.samples.dtype.names and not len(sig.samples.shape) > 1
        }

        for uuid in descriptions:
            if uuid not in signals:
                description = descriptions[uuid]

                sig = Signal([], [], name=description["name"])
                sig.uuid = uuid

                sig.origin_uuid = self.uuid
                sig.group_index = NOT_FOUND
                sig.channel_index = NOT_FOUND
                sig.color = description["color"]

                if description["flags"] & Signal.Flags.user_defined_conversion:
                    sig.conversion = from_dict(description["conversion"])
                    sig.flags |= Signal.Flags.user_defined_conversion

                if description["flags"] & Signal.Flags.user_defined_name:
                    sig.original_name = sig.name
                    sig.name = description.get("user_defined_name", "") or ""
                    sig.flags |= Signal.Flags.user_defined_name

                if description["flags"] & Signal.Flags.user_defined_unit:
                    sig.unit = description.get("user_defined_unit", "") or ""
                    sig.flags |= Signal.Flags.user_defined_unit

                signals[uuid] = sig

        if hasattr(self, "mdf"):
            events = []
            origin = self.mdf.start_time

            if self.mdf.version >= "4.00":
                mdf_events = list(self.mdf.events)

                for pos, event in enumerate(mdf_events):
                    event_info = {}
                    event_info["value"] = event.value
                    event_info["type"] = v4c.EVENT_TYPE_TO_STRING[event.event_type]
                    event_info["tool"] = event.tool
                    if event.name:
                        description = event.name
                    else:
                        description = ""

                    if event.comment:
                        try:
                            comment = extract_xml_comment(event.comment)
                        except:
                            comment = event.comment
                        if description:
                            description = f"{description} ({comment})"
                        else:
                            description = comment

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
                            "type": v4c.EVENT_TYPE_TO_STRING[v4c.EVENT_TYPE_TRIGGER],
                        }
                        events.append(event)
        else:
            events = []
            if isinstance(self.files, QtWidgets.QMdiArea):
                origin = self.files.subWindowList()[0].widget().mdf.start_time
            else:
                origin = self.files.widget(0).mdf.start_time

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
            show_cursor_circle=self.cursor_circle,
            show_cursor_horizontal_line=self.cursor_horizontal_line,
            cursor_line_width=self.cursor_line_width,
            cursor_color=self.cursor_color,
            owner=self,
        )

        plot.plot._can_compute_all_timebase = False

        plot.pattern_group_added.connect(self.add_pattern_group)
        plot.verify_bookmarks.connect(self.verify_bookmarks)
        plot.pattern = pattern_info

        plot.plot._can_paint_global = False

        plot.show()

        sub = MdiSubWindow(parent=self)
        sub.setWidget(plot)
        plot.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
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

        if window_info.get("maximized", False):
            w.showMaximized()
        elif window_info.get("minimized", False):
            w.showMinimized()

        menu = w.systemMenu()

        action = QtGui.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        w.setWindowTitle(generate_window_title(w, window_info["type"], window_info["title"]))

        if "x_range" in window_info["configuration"] and WithMDIArea.load_plot_x_range:
            x_range = window_info["configuration"]["x_range"]
            if isinstance(x_range, float):
                x_range = 0, x_range

            plot.plot.viewbox.setXRange(*x_range, padding=0)
            plot.plot.initial_x_range = "shift"

        if "splitter" in window_info["configuration"]:
            plot.splitter.setSizes(window_info["configuration"]["splitter"])

        if "y_axis_width" in window_info["configuration"]:
            plot.plot.y_axis.setWidth(window_info["configuration"]["y_axis_width"])

        if "grid" in window_info["configuration"]:
            x_grid, y_grid = window_info["configuration"]["grid"]
            plot.plot.plotItem.ctrl.xGridCheck.setChecked(x_grid)
            plot.plot.plotItem.ctrl.yGridCheck.setChecked(y_grid)

        if "font_size" in window_info["configuration"]:
            plot.set_font_size(window_info["configuration"]["font_size"])

        plot.splitter.setContentsMargins(1, 1, 1, 1)
        plot.setContentsMargins(1, 1, 1, 1)

        # plot.hide()
        plot.show_properties.connect(self._show_info)

        plot.add_new_channels(signals, mime_data)

        # plot.show()

        plot.add_channels_request.connect(partial(self.add_new_channels, widget=plot))
        plot.edit_channel_request.connect(partial(self.edit_channel, widget=plot))

        self.set_subplots_link(self.subplots_link)

        if "cursor_precision" in window_info["configuration"]:
            plot.cursor_info.set_precision(window_info["configuration"]["cursor_precision"])

        iterator = QtWidgets.QTreeWidgetItemIterator(plot.channel_selection)
        while item := iterator.value():
            iterator += 1

            if item.type() == item.Group:
                if item.pattern:
                    state = item.checkState(item.NameColumn)
                    plot.pattern_group_added.emit(plot, item)
                    item.setCheckState(item.NameColumn, state)

        if "common_axis_y_range" in window_info["configuration"]:
            plot.plot.common_axis_y_range = tuple(window_info["configuration"]["common_axis_y_range"])

        if "channels_header" in window_info["configuration"]:
            width, sizes = window_info["configuration"]["channels_header"]
            current_width = sum(plot.splitter.sizes())
            plot.splitter.setSizes([width, max(current_width - width, 50)])
            for i, size in enumerate(sizes):
                plot.channel_selection.setColumnWidth(i, size)

        plot.set_locked(locked=window_info["configuration"].get("locked", False))
        plot.hide_axes(hide=window_info["configuration"].get("hide_axes", False))
        plot.hide_selected_channel_value(
            hide=window_info["configuration"].get("hide_selected_channel_value_panel", True)
        )
        plot.toggle_bookmarks(hide=window_info["configuration"].get("hide_bookmarks", False))
        plot.toggle_focused_mode(focused=window_info["configuration"].get("focused_mode", False))
        plot.toggle_region_values_display_mode(mode=window_info["configuration"].get("delta_mode", "value"))

        plot_graphics = plot.plot

        plot_graphics._can_paint_global = True
        plot_graphics._can_compute_all_timebase = True

        plot_graphics._compute_all_timebase()

        if len(plot_graphics.all_timebase) and plot_graphics.cursor1 is not None:
            plot_graphics.cursor1.set_value(plot_graphics.all_timebase[0])

        plot_graphics.viewbox._matrixNeedsUpdate = True
        plot_graphics.viewbox.updateMatrix()

        plot.update()
        plot.channel_selection.refresh()
        plot.set_initial_zoom()

        return w, pattern_info

    def _load_tabular_window(self, window_info):
        uuid = self.uuid
        geometry = window_info.get("geometry", None)

        # patterns
        pattern_info = window_info["configuration"].get("pattern", {})
        if pattern_info:
            required = set()
            found_signals = []

            signals_ = extract_signals_using_pattern(
                mdf=self.mdf,
                channels_db=None,
                pattern_info=pattern_info,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
                uuid=self.uuid,
            ).values()

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
                range_info["font_color"] = QtGui.QBrush(QtGui.QColor(range_info["font_color"]))
                range_info["background_color"] = QtGui.QBrush(QtGui.QColor(range_info["background_color"]))

            ranges = {sig.name: copy_ranges(ranges) for sig in signals_}

            signals_ = [(sig.name, sig.group_index, sig.channel_index) for sig in signals_]

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
                    range_info["font_color"] = QtGui.QBrush(QtGui.QColor(range_info["font_color"]))
                    range_info["background_color"] = QtGui.QBrush(QtGui.QColor(range_info["background_color"]))

            if not signals_:
                return None, False

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
            start=self.mdf.header.start_time,
            parent=self,
        )
        tabular.pattern = pattern_info

        sub = MdiSubWindow(parent=self)
        sub.setWidget(tabular)
        tabular.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
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

        if window_info.get("maximized", False):
            w.showMaximized()
        elif window_info.get("minimized", False):
            w.showMinimized()

        w.setWindowTitle(generate_window_title(w, window_info["type"], window_info["title"]))

        mode = tabular.format_selection.currentText()

        filter_count = 0
        available_columns = [signals.index.name, *signals.columns]
        for filter_info in window_info["configuration"]["filters"]:
            if filter_info["column"] in available_columns:
                tabular.add_filter()
                filter = tabular.filters.itemWidget(tabular.filters.item(filter_count))
                filter.enabled.setCheckState(
                    QtCore.Qt.CheckState.Checked if filter_info["enabled"] else QtCore.Qt.CheckState.Unchecked
                )
                filter.relation.setCurrentText(filter_info["relation"])
                filter.column.setCurrentText(filter_info["column"])
                filter.op.setCurrentText(filter_info["op"])
                if mode == "phys":
                    filter.target.setText(str(filter_info["target"]).strip('"'))
                elif mode == "hex":
                    filter.target.setText(hex(filter_info["target"]).strip('"'))
                elif mode == "bin":
                    filter.target.setText(bin(filter_info["target"]).strip('"'))
                filter.validate_target()

                filter_count += 1

        if filter_count and window_info["configuration"]["filtered"]:
            tabular.apply_filters()

        tabular.time_as_date.setCheckState(
            QtCore.Qt.CheckState.Checked
            if window_info["configuration"]["time_as_date"]
            else QtCore.Qt.CheckState.Unchecked
        )
        tabular.add_channels_request.connect(partial(self.add_new_channels, widget=tabular))

        menu = w.systemMenu()

        action = QtGui.QAction("Set title", menu)
        action.triggered.connect(partial(set_title, w))
        before = menu.actions()[0]
        menu.insertAction(before, action)

        if self.subplots_link:
            tabular.timestamp_changed_signal.connect(self.set_cursor)

        sections_width = window_info["configuration"].get("header_sections_width", [])
        if sections_width:
            for i, width in enumerate(sections_width):
                tabular.tree.columnHeader.setColumnWidth(i, width)
                tabular.tree.dataView.setColumnWidth(i, width)

            tabular.tree.dataView.updateGeometry()
            tabular.tree.columnHeader.updateGeometry()

        return w, pattern_info

    def _load_can_bus_trace_window(self, window_info):
        if self.mdf.version < "4.00":
            return None, False

        ranges = window_info["configuration"].get("ranges", {})
        for channel_ranges in ranges.values():
            for range_info in channel_ranges:
                range_info["font_color"] = QtGui.QBrush(QtGui.QColor(range_info["font_color"]))
                range_info["background_color"] = QtGui.QBrush(QtGui.QColor(range_info["background_color"]))

        widget = self._add_can_bus_trace_window(ranges)

        sections_width = window_info["configuration"].get("header_sections_width", [])
        if sections_width:
            for i, width in enumerate(sections_width):
                widget.tree.columnHeader.setColumnWidth(i, width)
                widget.tree.dataView.setColumnWidth(i, width)

            widget.tree.dataView.updateGeometry()
            widget.tree.columnHeader.updateGeometry()

        return None, False

    def _load_flexray_bus_trace_window(self, window_info):
        if self.mdf.version < "4.00":
            return None, False

        ranges = window_info["configuration"].get("ranges", {})
        for channel_ranges in ranges.values():
            for range_info in channel_ranges:
                range_info["font_color"] = QtGui.QBrush(QtGui.QColor(range_info["font_color"]))
                range_info["background_color"] = QtGui.QBrush(QtGui.QColor(range_info["background_color"]))

        widget = self._add_flexray_bus_trace_window(ranges)

        sections_width = window_info["configuration"].get("header_sections_width", [])
        if sections_width:
            for i, width in enumerate(sections_width):
                widget.tree.columnHeader.setColumnWidth(i, width)
                widget.tree.dataView.setColumnWidth(i, width)

            widget.tree.dataView.updateGeometry()
            widget.tree.columnHeader.updateGeometry()

        return None, False

    def _load_lin_bus_trace_window(self, window_info):
        if self.mdf.version < "4.00":
            return None, False

        ranges = window_info["configuration"].get("ranges", {})
        for channel_ranges in ranges.values():
            for range_info in channel_ranges:
                range_info["font_color"] = QtGui.QBrush(QtGui.QColor(range_info["font_color"]))
                range_info["background_color"] = QtGui.QBrush(QtGui.QColor(range_info["background_color"]))

        widget = self._add_lin_bus_trace_window(ranges)

        sections_width = window_info["configuration"].get("header_sections_width", [])
        if sections_width:
            for i, width in enumerate(sections_width):
                widget.tree.columnHeader.setColumnWidth(i, width)
                widget.tree.dataView.setColumnWidth(i, width)

            widget.tree.dataView.updateGeometry()
            widget.tree.columnHeader.updateGeometry()

        return None, False

    def set_line_style(self, with_dots=None):
        if with_dots is None:
            with_dots = not self.with_dots

        current_plot = self.get_current_widget()
        if current_plot and isinstance(current_plot, Plot):
            self.with_dots = with_dots
            current_plot.with_dots = with_dots
            current_plot.plot.set_dots(with_dots)

    def set_line_interconnect(self, line_interconnect):
        if line_interconnect == "line":
            line_interconnect = ""

        self.line_interconnect = line_interconnect
        for i, mdi in enumerate(self.mdi_area.subWindowList()):
            widget = mdi.widget()
            if isinstance(widget, Plot):
                widget.line_interconnect = line_interconnect
                widget.plot.set_line_interconnect(line_interconnect)

    def set_subplots(self, option):
        self.subplots = option

    def set_subplots_link(self, subplots_link):
        self.subplots_link = subplots_link
        if subplots_link:
            for i, mdi in enumerate(self.mdi_area.subWindowList()):
                widget = mdi.widget()
                if isinstance(widget, Plot):
                    widget.x_range_changed_signal.connect(self.set_x_range)
                    widget.cursor_moved_signal.connect(self.set_cursor)
                    widget.region_removed_signal.connect(self.remove_region)
                    widget.region_moved_signal.connect(self.set_region)
                    widget.splitter_moved.connect(self.set_splitter)
                elif isinstance(widget, Numeric):
                    widget.timestamp_changed_signal.connect(self.set_cursor)
        else:
            for mdi in self.mdi_area.subWindowList():
                widget = mdi.widget()
                if isinstance(widget, Plot):
                    try:
                        widget.cursor_moved_signal.disconnect(self.set_cursor)
                    except:
                        pass
                    try:
                        widget.x_range_changed_signal.disconnect(self.set_x_range)
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

        active_window = self.mdi_area.currentSubWindow()
        if active_window is None:
            return

        active_widget = active_window.widget()

        if widget is not active_widget:
            return

        for mdi in self.mdi_area.subWindowList():
            wid = mdi.widget()
            if wid is not widget:
                wid.set_timestamp(pos)

    def set_x_range(self, widget, x_range):
        if not self.subplots_link:
            return

        if not isinstance(x_range, (tuple, list)):
            return

        if not len(x_range) == 2:
            return

        if np.any(np.isnan(x_range)) or not np.all(np.isfinite(x_range)):
            return

        for mdi in self.mdi_area.subWindowList():
            wid = mdi.widget()
            if wid is not widget and isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = True
                wid.plot.viewbox.setXRange(*x_range, padding=0, update=True)
                wid._inhibit_x_range_changed_signal = False

    def set_region(self, widget, region):
        if not self.subplots_link:
            return

        active_window = self.mdi_area.currentSubWindow()
        if active_window is None:
            return

        active_widget = active_window.widget()

        if widget is not active_widget:
            return

        for mdi in self.mdi_area.subWindowList():
            wid = mdi.widget()
            if isinstance(wid, Plot) and wid is not widget:
                if wid.plot.region is None:
                    event = QtGui.QKeyEvent(
                        QtCore.QEvent.Type.KeyPress,
                        QtCore.Qt.Key.Key_R,
                        QtCore.Qt.KeyboardModifier.NoModifier,
                    )
                    wid.plot.keyPressEvent(event)
                wid.plot.region.setRegion(region)

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
                            wid.splitter.setSizes([selection_width, total_size - selection_width])

            self._splitter_source = None

    def update_functions(self, original_definitions, modified_definitions):
        # new definitions
        new_functions = [info for uuid, info in modified_definitions.items() if uuid not in original_definitions]

        for info in new_functions:
            self.functions[info["name"]] = info["definition"]

        new = {info["name"] for info in new_functions}

        # changed definitions
        translation = {}
        changed = set()

        changed_functions = [
            (info, original_definitions[uuid])
            for uuid, info in modified_definitions.items()
            if uuid in original_definitions and info != original_definitions[uuid]
        ]

        for new_info, old_info in changed_functions:
            translation[old_info["name"]] = new_info["name"]
            del self.functions[old_info["name"]]
            self.functions[new_info["name"]] = new_info["definition"]
            changed.add(old_info["name"])

        # deleted definitions

        deleted = set()

        deleted_functions = [info for uuid, info in original_definitions.items() if uuid not in modified_definitions]

        for info in deleted_functions:
            del self.functions[info["name"]]
            deleted.add(info["name"])

        # apply changes

        for mdi in self.mdi_area.subWindowList():
            wid = mdi.widget()
            if isinstance(wid, Plot):
                iterator = QtWidgets.QTreeWidgetItemIterator(wid.channel_selection)
                while item := iterator.value():
                    if item.type() == item.Channel:
                        if item.signal.flags & item.signal.Flags.computed:
                            function = item.signal.computation["function"]

                            if function in changed:
                                try:
                                    item.signal.computation["function"] = translation[
                                        item.signal.computation["function"]
                                    ]

                                    func_name = item.signal.computation["function"]
                                    definition = self.functions[func_name]
                                    _globals = {}
                                    exec(definition.replace("\t", "    "), _globals)
                                    func = _globals[func_name]

                                    parameters = list(inspect.signature(func).parameters)[:-1]
                                    args = {name: [] for name in parameters}
                                    for arg_name, alternatives in zip(
                                        parameters,
                                        item.signal.computation["args"].values(),
                                    ):
                                        args[arg_name] = alternatives

                                    item.signal.computation["args"] = args
                                except:
                                    print(format_exc())

                            self.edit_channel(wid.channel_item_to_config(item), item, wid)

                    iterator += 1

        return bool(new or changed or deleted)

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
                            QtCore.QEvent.Type.KeyPress,
                            QtCore.Qt.Key.Key_R,
                            QtCore.Qt.KeyboardModifier.NoModifier,
                        )
                        plt.plot.keyPressEvent(event)
            self._region_source = None

    def save_all_subplots(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save as measurement file", "", "MDF version 4 files (*.mf4)"
        )

        if file_name:
            with mdf_module.MDF() as mdf:
                for mdi in self.mdi_area.subWindowList():
                    widget = mdi.widget()

                    if isinstance(widget, Plot):
                        mdf.append(widget.plot.signals)
                    elif isinstance(widget, Numeric):
                        mdf.append([s.signal for s in widget.channels.backend.signals])
                    elif isinstance(widget, Tabular):
                        mdf.append(widget.tree.pgdf.df_unfiltered)
                mdf.save(file_name, compression=2, overwrite=True)

    def file_by_uuid(self, uuid):
        try:
            if isinstance(self.files, QtWidgets.QMdiArea):
                for file_index, file_window in enumerate(self.files.subWindowList()):
                    if file_window.widget().uuid == uuid:
                        return file_index, file_window.widget()
                return None
            else:
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
        group_index, index, sig = lst
        uuid = sig.origin_uuid
        file_info = self.file_by_uuid(uuid)
        if file_info:
            _, file = file_info
            try:
                channel = file.mdf.get_channel_metadata(group=group_index, index=index)
                msg = ChannelInfoDialog(channel, self)
                msg.show()
            except MdfException:
                MessageBox.warning(
                    self,
                    "Missing channel",
                    f"The channel {sig.name} does not exit in the current measurement file.",
                )

    def verify_bookmarks(self, bookmarks, plot):
        if not hasattr(self, "mdf"):
            return

        original_file_name = Path(self.mdf.original_name)

        if original_file_name.suffix.lower() not in (".mf4", ".mf4z"):
            return

        last_bookmark_index = None

        for i, bookmark in enumerate(bookmarks):
            if bookmark.editable:
                if last_bookmark_index is None:
                    last_bookmark_index = i - 1

                if bookmark.edited or bookmark.deleted:
                    break

        else:
            return

        result = MessageBox.question(
            self,
            "Save measurement bookmarks?",
            "You have modified bookmarks.\n\n" "Do you want to save the changes in the measurement file?\n" "",
        )

        if result == MessageBox.No:
            return

        _password = self.mdf._password

        uuid = self.mdf.uuid
        dspf = self.to_config()

        self.mdf.close()

        windows = list(self.mdi_area.subWindowList())
        for window in windows:
            widget = window.widget()
            if widget is plot:
                continue

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
                if Path(fname).suffix.lower() not in (".mdf", ".dat", ".mf4"):
                    return

                tmpdir = gettempdir()
                file_name = archive.extract(fname, tmpdir)
                file_name = Path(tmpdir) / file_name
        else:
            file_name = original_file_name

        with open(file_name, "r+b") as mdf:
            try:
                mdf.seek(0, 2)
                address = mdf.tell()

                blocks = []
                events = []

                alignment = address % 8
                if alignment:
                    offset = 8 - alignment
                    blocks.append(b"\0" * offset)
                    address += offset

                for i, bookmark in enumerate(bookmarks[last_bookmark_index + 1 :]):
                    if not bookmark.deleted:
                        event = EventBlock(
                            cause=v4c.EVENT_CAUSE_USER,
                            range_type=v4c.EVENT_RANGE_TYPE_POINT,
                            sync_type=v4c.EVENT_SYNC_TYPE_S,
                            event_type=v4c.EVENT_TYPE_MARKER,
                            flags=v4c.FLAG_EV_POST_PROCESSING,
                        )
                        event.value = bookmark.value()
                        event.comment = bookmark.xml_comment()

                        events.append(event)
                        address = event.to_blocks(address, blocks)

                for i in range(len(events) - 1):
                    events[i].next_ev_addr = events[i + 1].address

                for block in blocks:
                    mdf.write(bytes(block))

                header = HeaderBlock(stream=mdf, address=0x40)
                if last_bookmark_index >= 0:
                    address = header.first_event_addr
                    for i in range(last_bookmark_index + 1):
                        event = EventBlock(stream=mdf, address=address)
                        address = event.next_ev_addr

                    if events:
                        event.next_ev_addr = events[0].address
                    else:
                        event.next_ev_addr = 0
                    mdf.seek(event.address)
                    mdf.write(bytes(event))

                else:
                    if events:
                        header.first_event_addr = events[0].address
                        mdf.seek(header.address)
                        mdf.write(bytes(header))
                    else:
                        header.first_event_addr = 0
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

        self.aspects.setCurrentIndex(0)
        self.load_channel_list(file_name=dspf)

    def window_closed_handler(self, obj=None):
        self.windows_modified.emit()

    def set_cursor_options(self, cursor_circle, cursor_horizontal_line, cursor_line_width, cursor_color):
        cursor_color = QtGui.QColor(cursor_color)
        self.cursor_circle = cursor_circle
        self.cursor_horizontal_line = cursor_horizontal_line
        self.cursor_line_width = cursor_line_width
        self.cursor_color = cursor_color

        for i, mdi in enumerate(self.mdi_area.subWindowList()):
            widget = mdi.widget()
            if isinstance(widget, Plot):
                widget.plot.cursor1.show_circle = cursor_circle
                widget.plot.cursor1.show_horizontal_line = cursor_horizontal_line
                widget.plot.cursor1.line_width = cursor_line_width
                widget.plot.cursor1.color = cursor_color
                if widget.plot.region is not None:
                    for cursor in widget.plot.region.lines:
                        cursor.show_circle = cursor_circle
                        cursor.show_horizontal_line = cursor_horizontal_line
                        cursor.line_width = cursor_line_width
                        cursor.color = cursor_color
                widget.plot.update()

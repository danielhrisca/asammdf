from copy import deepcopy
import datetime
from functools import partial
import inspect
import itertools
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
from pyqtgraph import functions as fn
from PySide6 import QtCore, QtGui, QtWidgets

import asammdf.mdf as mdf_module

from ...blocks import v4_constants as v4c
from ...blocks.conversion_utils import from_dict
from ...blocks.utils import csv_bytearray2hex, extract_xml_comment, load_can_database, MdfException, UniqueDB
from ...blocks.v4_blocks import EventBlock, HeaderBlock
from ...signal import Signal
from .. import utils
from ..dialogs.advanced_search import AdvancedSearch
from ..dialogs.channel_info import ChannelInfoDialog
from ..dialogs.messagebox import MessageBox
from ..dialogs.window_selection_dialog import WindowSelectionDialog
from ..serde import extract_mime_names
from ..utils import (
    computation_to_python_function,
    compute_signal,
    copy_ranges,
    generate_python_function_globals,
    replace_computation_dependency,
)
from .can_bus_trace import CANBusTrace
from .flexray_bus_trace import FlexRayBusTrace
from .gps import GPS
from .lin_bus_trace import LINBusTrace
from .numeric import Numeric
from .plot import Plot
from .tabular import Tabular
from .xy import XY

COMPONENT = re.compile(r"\[(?P<index>\d+)\]$")
SIG_RE = re.compile(r"\{\{(?!\}\})(?P<name>.*?)\}\}")
NOT_FOUND = 0xFFFFFFFF

SNAP_PIXELS_DISTANCE = 20
CASCADE_PIXELS_DISTANCE = SNAP_PIXELS_DISTANCE + 10


def deepcopy_cfg_item(item):
    if item.get("type", "channel") == "group":
        ranges = [
            {
                "background_color": fn.mkColor(range_item["background_color"]),
                "font_color": fn.mkColor(range_item["font_color"]),
                "op1": range_item["op1"],
                "op2": range_item["op2"],
                "value1": range_item["value1"],
                "value2": range_item["value2"],
            }
            for range_item in item["ranges"]
        ]

        new_item = {
            "type": "group",
            "name": item["name"],
            "enabled": item["enabled"],
            "pattern": None,
            "ranges": ranges,
            "origin_uuid": item["origin_uuid"],
            "expanded": item["expanded"],
            "disabled": item["disabled"],
            "channels": deepcopy_cfg_item(item["channels"]),
        }

        if pattern := item["pattern"]:
            ranges = [
                {
                    "background_color": fn.mkColor(range_item["background_color"]),
                    "font_color": fn.mkColor(range_item["font_color"]),
                    "op1": range_item["op1"],
                    "op2": range_item["op2"],
                    "value1": range_item["value1"],
                    "value2": range_item["value2"],
                }
                for range_item in pattern["ranges"]
            ]
            new_item["pattern"] = {
                "pattern": pattern["pattern"],
                "match_type": pattern["match_type"],
                "case_sensitive": pattern["case_sensitive"],
                "filter_type": pattern["filter_type"],
                "filter_value": pattern["filter_value"],
                "raw": pattern["raw"],
                "ranges": ranges,
                "name": pattern["name"],
                "integer_format": pattern["integer_format"],
                "y_range": deepcopy(pattern["pattern"]),
            }

    else:
        ranges = [
            {
                "background_color": fn.mkColor(range_item["background_color"]),
                "font_color": fn.mkColor(range_item["font_color"]),
                "op1": range_item["op1"],
                "op2": range_item["op2"],
                "value1": range_item["value1"],
                "value2": range_item["value2"],
            }
            for range_item in item["ranges"]
        ]
        new_item = {
            "type": "channel",
            "name": item["name"],
            "unit": item.get("unit", ""),
            "flags": item.get("flags", 0),
            "enabled": item.get("enabled", True),
            "individual_axis": item.get("individual_axis", False),
            "common_axis": item.get("common_axis", False),
            "color": fn.mkColor(item["color"]),
            "computed": item.get("computed", False),
            "ranges": ranges,
            "precision": item.get("precision", 3),
            "fmt": item.get("fmt", "{}"),
            "format": item.get("format", "phys"),
            "mode": item.get("mode", "phys"),
            "y_range": deepcopy(item.get("y_range", [])),
            "origin_uuid": item["origin_uuid"],
        }

        for key in ("computation", "conversion", "user_defined_name", "individual_axis_width", "user_defined_unit"):
            if key in item:
                new_item[key] = deepcopy(item[key])

    return new_item


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
    mdfs=None,
    computed_origin_uuid=None,
    default_index=NOT_FOUND,
    top=True,
    has_flags=None,
):
    if mdfs is None:
        mdfs = [None]
    elif not isinstance(mdfs, (tuple, list)):
        mdfs = [mdfs]

    if top:
        rename_origin_uuid(items)
        if not mdfs:
            computed_origin_uuid = os.urandom(6).hex()

    descriptions = {}
    found = {}
    not_found = {}
    computed = {}
    mime = []
    for cfg_item in items:
        if cfg_item.get("type", "channel") == "group":
            uuid = os.urandom(6).hex()
            cfg_item["uuid"] = uuid

            if cfg_item.get("pattern", None) is None:
                (
                    new_mine,
                    new_descriptions,
                    new_found,
                    new_not_found,
                    new_computed,
                ) = build_mime_from_config(
                    cfg_item["channels"],
                    mdfs,
                    computed_origin_uuid,
                    default_index,
                    top=False,
                    has_flags=has_flags,
                )
                descriptions.update(new_descriptions)
                found.update(new_found)
                not_found.update(new_not_found)
                computed.update(new_computed)

                cfg_item["channels"] = new_mine

                mime.append(cfg_item)
            else:
                mime.append(cfg_item)
        else:
            for mdf in mdfs:
                if mdf is None:
                    origin_uuid = computed_origin_uuid
                else:
                    origin_uuid = mdf.uuid

                uuid = os.urandom(6).hex()
                item = deepcopy_cfg_item(cfg_item)
                item["uuid"] = uuid
                item["origin_uuid"] = origin_uuid

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
                    item["origin_uuid"] = origin_uuid

                else:
                    occurrences = mdf.whereis(item["name"]) if mdf else None
                    if occurrences:
                        group_index, channel_index = occurrences[0]
                        found[uuid] = item

                    else:
                        group_index, channel_index = default_index, default_index
                        not_found[uuid] = item

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

    origin_uuid = getattr(mdf, "uuid", os.urandom(6).hex())
    if mdf is not None:
        origin_mdf = mdf.original_name.name

    pattern = pattern_info["pattern"]
    match_type = pattern_info["match_type"]
    case_sensitive = pattern_info.get("case_sensitive", False)
    filter_value = pattern_info["filter_value"]
    filter_type = pattern_info["filter_type"]
    raw = pattern_info["raw"]
    integer_format = pattern_info.get("integer_format", "phys")
    pattern_ranges = pattern_info["ranges"]

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

        matches = set()

        for name, entries in channels_db.items():
            if pattern.fullmatch(name):
                for entry in entries:
                    if entry in matches:
                        continue
                    matches.add((name, *entry))

        matches = natsorted(matches)
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
        sig.ranges = copy_ranges(pattern_ranges)
        output_signals[uuid] = sig
        sig.origin_uuid = origin_uuid
        if mdf is not None:
            sig.origin_mdf = origin_mdf
        sig.enable = True

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


class MdiAreaMixin:
    def addSubWindow(self, window):
        geometry = window.geometry()
        geometry.setSize(QtCore.QSize(400, 400))
        window.setGeometry(geometry)
        window.resized.connect(self.window_resized)
        window.moved.connect(self.window_moved)
        return super().addSubWindow(window)

    def clear_windows(self):
        for window in self.subWindowList():
            widget = window.widget()
            self.removeSubWindow(window)
            widget.setParent(None)
            window.close()
            widget.deleteLater()
            widget.close()

    def window_moved(self, window, new_position, old_position):
        snap = False

        window_geometry = window.geometry()
        area_geometry = self.geometry()
        sub_windows = [sub.geometry() for sub in self.subWindowList() if sub is not window]

        # left edge snapping
        snap_candidates = [
            0,
        ]

        for sub in sub_windows:
            snap_candidates.append(sub.x())
            snap_candidates.append(sub.x() + sub.width())

        for x in snap_candidates:
            if abs(new_position.x() - x) <= SNAP_PIXELS_DISTANCE:
                new_position.setX(x)
                snap = True
                break
        else:
            # right edge snapping
            snap_candidates = [
                area_geometry.width(),
            ]

            for sub in sub_windows:
                snap_candidates.append(sub.x())
                snap_candidates.append(sub.x() + sub.width())

            for x in snap_candidates:
                if abs(new_position.x() + window_geometry.width() - x) <= SNAP_PIXELS_DISTANCE:
                    new_position.setX(x - window_geometry.width())
                    snap = True
                    break

        # top edge snapping
        snap_candidates = [
            0,
        ]

        for sub in sub_windows:
            snap_candidates.append(sub.y())
            snap_candidates.append(sub.y() + sub.height())

        for y in snap_candidates:
            if abs(new_position.y() - y) <= SNAP_PIXELS_DISTANCE:
                new_position.setY(y)
                snap = True
                break
        else:
            # bottom edge snapping
            snap_candidates = [
                area_geometry.height(),
            ]

            for sub in sub_windows:
                if sub is not window:
                    snap_candidates.append(sub.y())
                    snap_candidates.append(sub.y() + sub.height())

            for y in snap_candidates:
                if abs(new_position.y() + window_geometry.height() - y) <= SNAP_PIXELS_DISTANCE:
                    new_position.setY(y - window_geometry.height())
                    snap = True
                    break

        if snap:
            window.blockSignals(True)
            window_geometry.moveTo(new_position)
            window.setGeometry(window_geometry)
            window.blockSignals(False)

            window.previous_position = old_position

    def window_resized(self, window, new_size, old_size):
        snap = False

        window_geometry = new_geometry = window.geometry()
        new_position = window_geometry.topLeft()
        area_geometry = self.geometry()
        sub_windows = [sub for sub in self.subWindowList() if sub is not window]

        for sub in sub_windows:
            sub.blockSignals(True)

        previous_position = window.previous_position
        previous_geometry = QtCore.QRect(0, 0, 0, 0)
        previous_geometry.moveTo(previous_position)
        previous_geometry.setSize(old_size)

        # right edge of other windows growing and snapping
        snap_candidates = [
            area_geometry.width(),
        ]

        for sub in sub_windows:
            if sub is not window:
                snap_candidates.append(sub.geometry().x() + sub.geometry().width())

        for x in snap_candidates:
            if abs(new_position.x() + window_geometry.width() - x) <= SNAP_PIXELS_DISTANCE:
                window_geometry.setWidth(abs(x - new_position.x()))
                snap = True
                break

        else:
            # left edge of other windows growing and snapping
            snap_candidates = []

            for sub in sub_windows:
                if sub is not window:
                    snap_candidates.append(sub.geometry().x())

            for x in snap_candidates:
                if abs(new_position.x() + window_geometry.width() - x) <= SNAP_PIXELS_DISTANCE:
                    window_geometry.setWidth(abs(x - new_position.x()))
                    snap = True
                    break

        # bottom edge of other windows snapping
        snap_candidates = [
            area_geometry.height(),
        ]

        for sub in sub_windows:
            if sub is not window:
                snap_candidates.append(sub.geometry().y() + sub.geometry().height())

        for y in snap_candidates:
            if abs(new_position.y() + window_geometry.height() - y) <= SNAP_PIXELS_DISTANCE:
                window_geometry.setHeight(abs(y - new_position.y()))
                snap = True
                break
        else:
            # top edge of other windows snapping
            snap_candidates = []

            for sub in sub_windows:
                if sub is not window:
                    snap_candidates.append(sub.geometry().y())

            for y in snap_candidates:
                if abs(new_position.y() + window_geometry.height() - y) <= SNAP_PIXELS_DISTANCE:
                    window_geometry.setHeight(abs(y - new_position.y()))
                    snap = True
                    break

        if snap:
            window.blockSignals(True)
            window.setGeometry(window_geometry)
            window.blockSignals(False)

            new_size = window_geometry.size()
            new_geometry = window_geometry

        # manage edge driven resize of other windows

        x_delta = new_size.width() - old_size.width()
        y_delta = new_size.height() - old_size.height()

        previous_x = None
        previous_y = None

        # top left corner unchanged
        if new_geometry.topLeft() == previous_geometry.topLeft():
            if x_delta:
                # right edge was dragged
                previous_x = previous_geometry.x() + old_size.width()

            if y_delta:
                # bottom edge was dragged
                previous_y = previous_geometry.y() + old_size.height()

        # top right corner unchanged
        elif new_geometry.topRight() == previous_geometry.topRight():
            if x_delta:
                # left edge was dragged
                previous_x = previous_geometry.x()
                x_delta = -x_delta
            if y_delta:
                # bottom edge was dragged
                previous_y = previous_geometry.y() + old_size.height()

        # bottom left corner unchanged
        elif new_geometry.bottomLeft() == previous_geometry.bottomLeft():
            if x_delta:
                # right edge was dragged
                previous_x = previous_geometry.x() + old_size.width()

            if y_delta:
                # top edge was dragged
                previous_y = previous_geometry.y()
                y_delta = -y_delta

        # bottom right corner unchanged
        elif new_geometry.bottomRight() == previous_geometry.bottomRight():
            if x_delta:
                # left edge was dragged
                previous_x = previous_geometry.x()
                x_delta = -x_delta

            if y_delta:
                # top edge was dragged
                previous_y = previous_geometry.y()
                y_delta = -y_delta

        if previous_x is not None:
            for sub in sub_windows:
                geometry = sub.geometry()
                if abs(geometry.x() - previous_x) <= SNAP_PIXELS_DISTANCE:
                    geometry.setX(geometry.x() + x_delta)
                    sub.setGeometry(geometry)
                elif abs(geometry.x() + geometry.width() - previous_x) <= SNAP_PIXELS_DISTANCE:
                    geometry.setWidth(geometry.width() + x_delta)
                    sub.setGeometry(geometry)

        if previous_y is not None:
            for sub in sub_windows:
                geometry = sub.geometry()
                if abs(geometry.y() - previous_y) <= SNAP_PIXELS_DISTANCE:
                    geometry.setY(geometry.y() + y_delta)
                    sub.setGeometry(geometry)
                elif abs(geometry.y() + geometry.height() - previous_y) <= SNAP_PIXELS_DISTANCE:
                    geometry.setHeight(geometry.height() + y_delta)
                    sub.setGeometry(geometry)

        for sub in sub_windows:
            sub.blockSignals(False)


class MdiSubWindow(QtWidgets.QMdiSubWindow):
    sigClosed = QtCore.Signal(object)
    titleModified = QtCore.Signal()
    resized = QtCore.Signal(object, object, object)
    moved = QtCore.Signal(object, object, object)
    pattern_modified = QtCore.Signal(object, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = self.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setOption(MdiSubWindow.RubberBandResize)
        self.setOption(MdiSubWindow.RubberBandMove)

        self.previous_position = QtCore.QPoint(0, 0)

        menu = self.systemMenu()
        before = menu.actions()[0]

        action = QtGui.QAction("Set title", menu)
        action.triggered.connect(self.set_title)
        menu.insertAction(before, action)

        action = QtGui.QAction("Edit window pattern", menu)
        action.triggered.connect(self.edit_window_pattern)
        menu.insertAction(before, action)

    def closeEvent(self, event):
        if isinstance(self.widget(), Plot):
            self.widget().close()
        super().closeEvent(event)
        self.sigClosed.emit(self)

    def edit_window_pattern(self):
        widget = self.widget()
        if not (pattern := getattr(widget, "pattern", None)):
            MessageBox.information(
                self,
                "Cannot edit window pattern",
                f"{self.windowTitle()} is not a pattern based window",
            )
        else:
            if hasattr(widget.owner, "mdf"):
                mdf = widget.owner.mdf
                channels_db = None
            else:
                mdf = None
                channels_db = widget.owner.channels_db

            dlg = AdvancedSearch(
                mdf=mdf,
                show_add_window=False,
                show_apply=True,
                show_search=False,
                window_title="Show pattern based group",
                parent=self,
                pattern=pattern,
                channels_db=channels_db,
            )
            dlg.setModal(True)
            dlg.exec_()

            if new_pattern := dlg.result:
                del dlg
                self.pattern_modified.emit(new_pattern, id(self))

    def moveEvent(self, event):
        old_position = event.oldPos()
        new_position = event.pos()

        # old_position == new_position => restore from minimized or maximized

        super().moveEvent(event)

        if not (self.isMinimized() or self.isMaximized()):
            self.moved.emit(self, new_position, old_position)

    def resizeEvent(self, event):
        old_size = event.oldSize()
        new_size = event.size()

        super().resizeEvent(event)

        if old_size.isValid() and not (self.isMinimized() or self.isMaximized()):
            self.resized.emit(self, new_size, old_size)

    def set_title(self):
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Set sub-plot title",
            "Title:",
            text=self.windowTitle(),
        )
        if ok and name:
            self.setWindowTitle(generate_window_title(self, title=name))
            self.titleModified.emit()


class MdiAreaWidget(MdiAreaMixin, QtWidgets.QMdiArea):
    add_window_request = QtCore.Signal(list)
    create_window_request = QtCore.Signal()
    open_files_request = QtCore.Signal(object)
    search_request = QtCore.Signal()

    def __init__(self, comparison=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setAcceptDrops(True)
        self.comparison = comparison

        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)

        self.show()

    def cascadeSubWindows(self):
        sub_windows = self.subWindowList()
        if not sub_windows:
            return

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = True
            window.blockSignals(True)

        position = QtCore.QPoint(0, 0)
        rect = QtCore.QRect(0, 0, 400, 300)

        for i, window in enumerate(sub_windows, 1):
            rect.moveTo(position)
            window.setGeometry(rect)

            position.setX(position.x() + CASCADE_PIXELS_DISTANCE)
            position.setY(position.y() + CASCADE_PIXELS_DISTANCE)
            if i % 5 == 0:
                position.setY(0)

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = False
            window.blockSignals(False)

    def dragEnterEvent(self, e):
        e.accept()
        super().dragEnterEvent(e)

    def dropEvent(self, event):
        if event.source() is self:
            super().dropEvent(event)
        else:
            data = event.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                dialog = WindowSelectionDialog(
                    options=("Plot", "Numeric") if self.comparison else ("Plot", "Numeric", "Tabular"), parent=self
                )
                dialog.setModal(True)
                dialog.exec_()

                if dialog.result():
                    window_type = dialog.selected_type()
                    disable_new_channels = dialog.disable_new_channels()
                    names = extract_mime_names(data, disable_new_channels=disable_new_channels)

                    self.add_window_request.emit([window_type, names])
                event.accept()
            else:
                try:
                    files = []
                    for url in event.mimeData().urls():
                        if sys.platform == "win32":
                            path = Path(url.path().strip("/"))
                        else:
                            path = Path(url.path())

                        if (
                            path.suffix.lower()
                            in utils.SUPPORTED_FILE_EXTENSIONS | utils.SUPPORTED_BUS_DATABASE_EXTENSIONS
                        ):
                            files.append(str(path))

                    if files:
                        self.open_files_request.emit(files)
                        event.accept()
                    else:
                        event.ignore()
                except:
                    print(format_exc())
                    event.ignore()

    def open_menu(self, position=None):
        viewport = self.viewport()
        if not self.childAt(position) is viewport:
            return

        self.context_menu = menu = QtWidgets.QMenu()
        menu.addAction(f"{len(self.subWindowList())} existing windows")
        menu.addSeparator()
        action = QtGui.QAction(QtGui.QIcon(":/search.png"), "Search", menu)
        action.setShortcut(QtGui.QKeySequence("Ctrl+F"))
        menu.addAction(action)
        menu.addSeparator()
        menu.addAction(QtGui.QIcon(":/plus.png"), "Add new window")

        menu.addSeparator()
        action = QtGui.QAction("Cascade sub-windows", menu)
        action.setShortcut(QtGui.QKeySequence("Shift+C"))
        menu.addAction(action)

        action = QtGui.QAction("Tile sub-windows in a grid", menu)
        action.setShortcut(QtGui.QKeySequence("Shift+T"))
        menu.addAction(action)

        action = QtGui.QAction("Tile sub-windows vertically", menu)
        action.setShortcut(QtGui.QKeySequence("Shift+V"))
        menu.addAction(action)

        action = QtGui.QAction("Tile sub-windows horizontally", menu)
        action.setShortcut(QtGui.QKeySequence("Shift+H"))
        menu.addAction(action)

        action = menu.exec(viewport.mapToGlobal(position))

        if action is None:
            return

        action_text = action.text()
        match action_text:
            case "Search":
                self.search_request.emit()
            case "Add new window":
                self.create_window_request.emit()
            case "Cascade sub-windows":
                self.cascadeSubWindows()
            case "Tile sub-windows in a grid":
                self.tileSubWindows()
            case "Tile sub-windows vertically":
                self.tile_vertically()
            case "Tile sub-windows horizontally":
                self.tile_horizontally()
            case _:
                pass

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
            window.blockSignals(True)

        for window in sub_windows:
            if window.isMinimized() or window.isMaximized():
                window.showNormal()
            rect = QtCore.QRect(0, 0, width, ratio)
            rect.moveTo(position)

            window.setGeometry(rect)
            position.setY(position.y() + ratio)

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = False
            window.blockSignals(False)

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
            window.blockSignals(True)

        for window in sub_windows:
            if window.isMinimized() or window.isMaximized():
                window.showNormal()
            rect = QtCore.QRect(0, 0, ratio, height)
            rect.moveTo(position)
            window.setGeometry(rect)
            position.setX(position.x() + ratio)

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = False
            window.blockSignals(False)

    def tileSubWindows(self):
        sub_windows = self.subWindowList()
        if not sub_windows:
            return

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = True
            window.blockSignals(True)

        super().tileSubWindows()

        for window in sub_windows:
            wid = window.widget()
            if isinstance(wid, Plot):
                wid._inhibit_x_range_changed_signal = False
            window.blockSignals(False)


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

        self._busy = False

        self.functions = {}
        self.global_variables = ""

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
        elif isinstance(widget, XY):
            if names:
                name = names[0]
            else:
                dlg = AdvancedSearch(
                    self.mdf,
                    show_add_window=False,
                    show_apply=True,
                    show_pattern=False,
                    apply_text="Apply",
                    parent=self,
                )
                dlg.setModal(True)
                dlg.exec_()
                result, pattern_window = dlg.result, dlg.pattern_window

                if result:
                    name = list(result.values())[0]
                else:
                    name = ""

            entries = self.mdf.whereis(name)
            if entries:
                entry = entries[0]
                channels = self.mdf.select(
                    [(name, *entry)],
                    raw=False,
                    ignore_value2text_conversions=True,
                    validate=True,
                )
                widget.add_new_channels(channels)

            else:
                widget.add_new_channels([None])

            return

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
                        use_interpolation=QtCore.QSettings().value("tabular_interpolation", True, type=bool),
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

                    for sig, sig_, sig_uuid in zip(selected_signals, uuids_signals, uuids_signals_uuid, strict=False):
                        sig.group_index = sig_[1]
                        sig.channel_index = sig_[2]
                        sig.flags &= ~sig.Flags.computed
                        sig.computation = {}
                        sig.origin_uuid = uuid
                        sig.name = sig_[0]
                        sig.uuid = sig_uuid["uuid"]
                        sig.ranges = sig_uuid["ranges"]
                        sig.color = fn.mkColor(sig_uuid.get("color", "#505050"))

                        if self.comparison:
                            sig.tooltip = f"{sig.name}\n@ {file.mdf.orignial_name}"
                            sig.name = f"{file_index + 1}: {sig.name}"

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

                    for sig, sig_ in zip(selected_signals, uuids_signals, strict=False):
                        sig.group_index = sig_["group_index"]
                        sig.channel_index = sig_["channel_index"]
                        sig.flags &= ~sig.Flags.computed
                        sig.computation = {}
                        sig.origin_uuid = uuid
                        sig.name = sig_["name"]
                        sig.color = sig_.get("color", None)
                        sig.uuid = sig_["uuid"]

                        if self.comparison:
                            sig.tooltip = f"{sig.name}\n@ {file.mdf.original_name}"
                            sig.name = f"{file_index + 1}: {sig.name}"

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
                            raw=True,
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
                            self.global_variables,
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

        if self.comparison:
            if window_type == "Plot":
                return self._add_plot_window(names)
            elif window_type == "Numeric":
                return self._add_numeric_window(names)
        else:
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
            elif window_type == "XY":
                return self._add_xy_window(names)

    def _add_can_bus_trace_window(self, ranges=None):
        dfs = []

        if self.mdf.version >= "4.00":
            groups_count = len(self.mdf.groups)

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
                                    columns["Direction"] = [
                                        v.decode("utf-8") for v in data["CAN_DataFrame.Dir"].tolist()
                                    ]
                                else:
                                    columns["Direction"] = [
                                        "Tx" if dir else "Rx" for dir in data["CAN_DataFrame.Dir"].astype("u1").tolist()
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
                                    columns["Direction"] = [
                                        v.decode("utf-8") for v in data["CAN_RemoteFrame.Dir"].tolist()
                                    ]
                                else:
                                    columns["Direction"] = [
                                        "Tx" if dir else "Rx"
                                        for dir in data["CAN_RemoteFrame.Dir"].astype("u1").tolist()
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
                                    columns["Direction"] = [
                                        v.decode("utf-8") for v in data["CAN_ErrorFrame.Dir"].tolist()
                                    ]
                                else:
                                    columns["Direction"] = [
                                        "Tx" if dir else "Rx"
                                        for dir in data["CAN_ErrorFrame.Dir"].astype("u1").tolist()
                                    ]

                        df = pd.DataFrame(columns, index=df_index)
                        dfs.append(df)

        if not dfs:
            df_index = []
            count = 0

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
            signals = pd.DataFrame(columns, index=df_index)

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

        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        w.setWindowTitle(f"CAN Bus Trace {self._window_counter}")
        self._window_counter += 1

        if self.subplots_link:
            trace.timestamp_changed_signal.connect(self.set_cursor)

        self.windows_modified.emit()
        trace.tree.auto_size_header()

        return trace

    def _add_flexray_bus_trace_window(self, ranges=None):
        items = []
        if self.mdf.version >= "4.00":
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

        if items:
            df_index = np.sort(np.concatenate([item.timestamps for (item, names) in items]))
            count = len(df_index)

            columns = {
                "timestamps": df_index,
                "Bus": np.full(count, "", dtype="O"),
                "Channel": np.full(count, "", dtype="O"),
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

                    vals = data["FLX_Frame.BusChannel"].astype("u1")
                    vals = [f"FlexRay {chn}" for chn in vals.tolist()]
                    columns["Bus"][index] = vals

                    if data["FLX_Frame.FlxChannel"].dtype.kind == "S":
                        columns["Channel"][index] = [v.decode("utf-8") for v in data["FLX_Frame.FlxChannel"].tolist()]
                    else:
                        columns["Channel"][index] = [
                            "B" if chn else "A" for chn in data["FLX_Frame.FlxChannel"].astype("u1").tolist()
                        ]

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
                                "Tx" if dir else "Rx" for dir in data["FLX_Frame.Dir"].astype("u1").tolist()
                            ]

                    vals = None
                    data_length = None

                elif data.name == "FLX_NullFrame":
                    index = np.searchsorted(df_index, data.timestamps)

                    vals = data["FLX_NullFrame.BusChannel"].astype("u1")
                    vals = [f"FlexRay {chn}" for chn in vals.tolist()]
                    columns["Bus"][index] = vals

                    if data["FLX_NullFrame.FlxChannel"].dtype.kind == "S":
                        columns["Channel"][index] = [
                            v.decode("utf-8") for v in data["FLX_NullFrame.FlxChannel"].tolist()
                        ]
                    else:
                        columns["Channel"][index] = [
                            "B" if chn else "A" for chn in data["FLX_NullFrame.FlxChannel"].astype("u1").tolist()
                        ]

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
                            columns["Direction"][index] = [
                                v.decode("utf-8") for v in data["FLX_NullFrame.Dir"].tolist()
                            ]
                        else:
                            columns["Direction"][index] = [
                                "Tx" if dir else "Rx" for dir in data["FLX_NullFrame.Dir"].astype("u1").tolist()
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
        else:
            df_index = []
            count = 0

            columns = {
                "timestamps": df_index,
                "Bus": np.full(count, "", dtype="O"),
                "Channel": np.full(count, "", dtype="O"),
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

        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

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

        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        w.setWindowTitle(f"GPS {self._window_counter}")
        self._window_counter += 1

        if self.subplots_link:
            gps.timestamp_changed_signal.connect(self.set_cursor)

        self.windows_modified.emit()

    def _add_lin_bus_trace_window(self, ranges=None):
        dfs = []
        if self.mdf.version >= "4.00":
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
                                        "Tx" if dir else "Rx" for dir in data["LIN_Frame.Dir"].astype("u1").tolist()
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
                            columns["Direction"] = ["Tx"] * count

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

                            columns["Direction"] = ["Rx"] * count

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
                                    "Tx" if dir else "Rx" for dir in data["LIN_ChecksumError.Dir"].astype("u1").tolist()
                                ]

                            vals = None

                        dfs.append(pd.DataFrame(columns, index=df_index))

        if not dfs:
            df_index = []
            count = 0

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

            signals = pd.DataFrame(columns, index=df_index)

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

        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

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

            if not self.comparison:
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
            origin_mdf = file.mdf.original_name.name

            selected_signals = file.mdf.select(
                uuids_signals,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
                copy_master=False,
                validate=True,
                raw=True,
            )

            for sig, sig_, sig_obj in zip(selected_signals, uuids_signals, uuids_signals_objs, strict=False):
                sig.group_index = sig_[1]
                sig.channel_index = sig_[2]
                sig.flags &= ~sig.Flags.computed
                sig.computation = {}
                sig.origin_uuid = uuid
                sig.origin_mdf = origin_mdf
                sig.name = sig_[0] or sig.name
                sig.ranges = sig_obj["ranges"]
                sig.uuid = sig_obj["uuid"]
                if "color" in sig_obj:
                    sig.color = fn.mkColor(sig_obj["color"])

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
            for sig, sig_obj in zip(not_found, not_found_objs, strict=False):
                sig.origin_uuid = uuid
                sig.origin_mdf = origin_mdf
                sig.group_index = NOT_FOUND
                sig.channel_index = randint(0, NOT_FOUND)
                sig.exists = False
                sig.ranges = copy_ranges(sig_obj["ranges"])
                sig.format = sig_obj["format"]

            signals.extend(not_found)

            signals = natsorted(signals, key=lambda x: x.name)

        numeric = Numeric([], parent=self, mode="offline", owner=self)

        numeric.show()
        numeric.hide()

        sub = MdiSubWindow(parent=self)
        sub.setWidget(numeric)
        numeric.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)
        sub.pattern_modified.connect(self.window_pattern_modified)

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

        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

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

        if not self.comparison:
            mime_data = substitude_mime_uuids(mime_data, uuid=self.uuid, force=True)
            flatten_entries = get_flatten_entries_from_mime(mime_data)

        # TO DO : is this necessary here?
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

            origin_mdf = file.mdf.original_name.name

            for sig, (sig_uuid, sig_) in zip(selected_signals, uuids_signals.items(), strict=False):
                sig.group_index = sig_["group_index"]
                sig.channel_index = sig_["channel_index"]
                sig.flags &= ~sig.Flags.computed
                sig.computation = {}
                sig.origin_uuid = uuid
                sig.origin_mdf = origin_mdf
                sig.name = sig_["name"] or sig.name
                sig.uuid = sig_uuid
                if "color" in sig_:
                    sig.color = sig_["color"]

                sig.ranges = sig_["ranges"]
                sig.enable = sig_["enabled"]

                if self.comparison:
                    sig.tooltip = f"{sig.name}\n@ {file.mdf.original_name}"

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
                        new_sig.origin_mdf = origin_mdf
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
                            new_sig.origin_mdf = origin_mdf
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

        # TO DO:
        # fix for comparison mode where self.mdf does not exist
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
                    self.global_variables,
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

        if not self.comparison:
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
            mdf = self.mdf

        else:
            events = []
            origin = next(self.iter_files()).mdf.start_time
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
        sub.pattern_modified.connect(self.window_pattern_modified)

        if not self.subplots:
            self.mdi_area.clear_windows()
            w = self.mdi_area.addSubWindow(sub)

            w.showMaximized()
        else:
            w = self.mdi_area.addSubWindow(sub)

            if len(self.mdi_area.subWindowList()) == 1:
                w.showMaximized()
            else:
                w.show()

        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        plot.show()

        w.setWindowTitle(generate_window_title(w, "Plot"))

        plot.add_channels_request.connect(partial(self.add_new_channels, widget=plot))
        plot.edit_channel_request.connect(partial(self.edit_channel, widget=plot))

        plot.show_overlapping_alias.connect(self._show_overlapping_alias)
        plot.show_properties.connect(self._show_info)

        plot.add_new_channels(signals, mime_data)
        if self.subplots_link:
            plot.x_range_changed_signal.connect(self.set_x_range)
            plot.cursor_moved_signal.connect(self.set_cursor)
            plot.region_removed_signal.connect(self.remove_region)
            plot.region_moved_signal.connect(self.set_region)
            plot.splitter_moved.connect(self.set_splitter)

            for i, mdi in enumerate(self.mdi_area.subWindowList()):
                widget = mdi.widget()
                if isinstance(widget, Plot):
                    plot.plot.viewbox.setXRange(*widget.plot.viewbox.viewRange()[0], padding=0, update=True)
                    break

        iterator = QtWidgets.QTreeWidgetItemIterator(plot.channel_selection)
        while item := iterator.value():
            iterator += 1

            if item.type() == item.Group:
                if item.pattern:
                    plot.pattern_group_added.emit(plot, item)

        if self.comparison:
            plot.channel_selection.setColumnHidden(plot.channel_selection.OriginColumn, False)

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

        signals_ = natsorted(signals_, key=lambda x: (x["group_index"], x["name"], x["channel_index"]))

        uuids = {entry["origin_uuid"] for entry in signals_}

        dfs = []
        ranges = {}
        start = []

        for uuid in uuids:
            unique_names = UniqueDB()

            uuids_signals = [
                (entry["name"], entry["group_index"], entry["channel_index"])
                for entry in signals_
                if entry["origin_uuid"] == uuid
            ]

            file_info = self.file_by_uuid(uuid)
            if not file_info:
                continue

            file_index, file = file_info

            if self.comparison:
                for entry in signals_:
                    if entry["origin_uuid"] != uuid:
                        continue

                    name = unique_names.get_unique_name(entry["name"])

                    ranges[f"{file_index + 1}: {name}"] = entry["ranges"]
            else:
                for entry in signals_:
                    if entry["origin_uuid"] != uuid:
                        continue
                    name = unique_names.get_unique_name(entry["name"])
                    ranges[name] = entry["ranges"]

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
                use_interpolation=QtCore.QSettings().value("tabular_interpolation", True, type=bool),
            )

            if self.comparison:
                columns = {name: f"{file_index + 1}: {name}" for name in df.columns}
                df.rename(columns=columns, inplace=True)

            dfs.append(df)

        if not dfs:
            signals = pd.DataFrame()
            try:
                start = self.mdf.header.start_time
            except:
                start = datetime.datetime.now()
        else:
            signals = pd.concat(dfs, axis=1)

            start = min(start)

        for name in signals.columns:
            if name.endswith(
                ".ID",
            ):
                signals[name] = signals[name].astype("<u4") & 0x1FFFFFFF

        tabular = Tabular(signals, start=start, parent=self, ranges=ranges, owner=self)

        sub = MdiSubWindow(parent=self)
        sub.setWidget(tabular)
        tabular.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)
        sub.pattern_modified.connect(self.window_pattern_modified)

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

        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        w.setWindowTitle(generate_window_title(w, "Tabular"))

        if self.subplots_link:
            tabular.timestamp_changed_signal.connect(self.set_cursor)

        tabular.add_channels_request.connect(partial(self.add_new_channels, widget=tabular))

        tabular.tree.auto_size_header()

        self.windows_modified.emit()

    def _add_xy_window(self, signals):
        signals = [sig[:3] for sig in signals]
        if len(signals) == 2:
            x_channel, y_channel = self.mdf.select(signals, validate=True)
        else:
            x_channel, y_channel = None, None

        xy = XY(x_channel, y_channel)
        sub = MdiSubWindow(parent=self)
        sub.setWidget(xy)
        xy.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)

        w = self.mdi_area.addSubWindow(sub)

        if len(self.mdi_area.subWindowList()) == 1:
            w.showMaximized()
        else:
            w.show()

        if self._frameless_windows:
            w.setWindowFlags(w.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint)

        xy.add_channels_request.connect(partial(self.add_new_channels, widget=xy))

        w.setWindowTitle(f"XY {self._window_counter}")
        self._window_counter += 1

        if self.subplots_link:
            xy.timestamp_changed_signal.connect(self.set_cursor)

        self.windows_modified.emit()

    def clear_windows(self):
        self.mdi_area.clear_windows()

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
            self.global_variables,
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

        if item.signal.flags & Signal.Flags.user_defined_name:
            signal.original_name = channel.name
            signal.name = channel.get("user_defined_name", "") or ""
            signal.flags |= signal.Flags.user_defined_name

        old_name = item.name
        new_name = signal.name
        uuid = item.uuid

        item.signal.samples = item.signal.raw_samples = item.signal.phys_samples = signal.samples
        item.signal.timestamps = signal.timestamps

        if item.signal.flags & Signal.Flags.user_defined_conversion:
            item.set_conversion(item.signal.conversion)
            signal.flags |= signal.Flags.user_defined_conversion

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

        for _channel in widget.plot.signals:
            if _channel.uuid == uuid:
                continue

            if _channel.flags & _channel.Flags.computed:
                required_channels = set(get_required_from_computed(_channel.computation))
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
        if self.comparison:
            functions = {
                "Numeric": self._load_numeric_window,
                "Plot": self._load_plot_window,
            }
        else:
            functions = {
                "Numeric": self._load_numeric_window,
                "Plot": self._load_plot_window,
                "GPS": self._load_gps_window,
                "Tabular": self._load_tabular_window,
                "CAN Bus Trace": self._load_can_bus_trace_window,
                "FlexRay Bus Trace": self._load_flexray_bus_trace_window,
                "LIN Bus Trace": self._load_lin_bus_trace_window,
                "XY": self._load_xy_window,
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

                self.windows_modified.emit()

    def _load_numeric_window(self, window_info):
        geometry = window_info.get("geometry", None)

        # patterns
        pattern_info = window_info["configuration"].get("pattern", {})
        if pattern_info:
            if self.comparison:
                signals = {}
                for file_index, file in enumerate(self.iter_files()):
                    new_plot_signals = extract_signals_using_pattern(
                        mdf=file.mdf,
                        channels_db=None,
                        pattern_info=pattern_info,
                        ignore_value2text_conversions=self.ignore_value2text_conversions,
                        uuid=file.uuid,
                    )
                    signals |= new_plot_signals

            else:
                signals = extract_signals_using_pattern(
                    mdf=self.mdf,
                    channels_db=None,
                    pattern_info=pattern_info,
                    ignore_value2text_conversions=self.ignore_value2text_conversions,
                    uuid=self.uuid,
                )

            signals = list(signals.values())

            for sig in signals:
                sig.computation = None
                sig.ranges = copy_ranges(pattern_info["ranges"])

        else:
            if self.comparison:
                mdfs = [file.mdf for file in self.iter_files()]
            else:
                mdfs = [self.mdf]

            required = window_info["configuration"]["channels"]

            signals = []

            for mdf in mdfs:
                origin_uuid = mdf.uuid
                origin_mdf = mdf.original_name.name

                mdf_required = [elem for elem in required if elem["origin_uuid"] == origin_uuid]

                found = [elem for elem in required if elem["name"] in mdf]

                signals_ = [(elem["name"], *mdf.whereis(elem["name"])[0]) for elem in found]

                if signals_:
                    mdf_signals = mdf.select(
                        signals_,
                        ignore_value2text_conversions=self.ignore_value2text_conversions,
                        copy_master=False,
                        validate=True,
                        raw=True,
                    )
                else:
                    mdf_signals = []

                for sig, sig_, description in zip(mdf_signals, signals_, found, strict=False):
                    sig.group_index = sig_[1]
                    sig.channel_index = sig_[2]
                    sig.origin_uuid = origin_uuid
                    sig.origin_mdf = origin_mdf
                    sig.computation = None
                    sig.ranges = copy_ranges(description["ranges"])
                    sig.format = description["format"]
                    sig.color = fn.mkColor(description.get("color", "#505050"))
                    sig.uuid = os.urandom(6).hex()

                mdf_signals = [
                    sig for sig in mdf_signals if not sig.samples.dtype.names and len(sig.samples.shape) <= 1
                ]

                mdf_signals = natsorted(mdf_signals, key=lambda x: x.name)

                mdf_found = {sig.name for sig in mdf_signals}
                mdf_required = {description["name"] for description in mdf_required}
                not_found = [Signal([], [], name=name) for name in sorted(mdf_required - mdf_found)]
                for sig in not_found:
                    sig.uuid = os.urandom(6).hex()
                    sig.origin_uuid = origin_uuid
                    sig.origin_mdf = origin_mdf
                    sig.group_index = 0
                    sig.ranges = []

                mdf_signals.extend(not_found)

                signals.extend(mdf_signals)

            signals.sort(key=lambda x: (x.name, x.origin_uuid))

        numeric = Numeric(
            [],
            format=window_info["configuration"]["format"],
            float_precision=window_info["configuration"].get("float_precision", 3),
            parent=self,
            mode="offline",
            owner=self,
        )
        numeric.pattern = pattern_info

        sub = MdiSubWindow(parent=self)
        sub.setWidget(numeric)
        numeric.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)
        sub.pattern_modified.connect(self.window_pattern_modified)

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

        numeric.add_channels_request.connect(partial(self.add_new_channels, widget=numeric))

        if self.subplots_link:
            numeric.timestamp_changed_signal.connect(self.set_cursor)

        sections_width = window_info["configuration"].get("header_sections_width", [])
        if sections_width:
            numeric.channels.columnHeader.columns_width = dict(enumerate(sections_width))
            sections_width = reversed(list(enumerate(sections_width)))
            for column_index, width in sections_width:
                numeric.channels.columnHeader.setColumnWidth(column_index, width)
                numeric.channels.dataView.setColumnWidth(
                    column_index,
                    numeric.channels.columnHeader.columnWidth(column_index),
                )

        font_size = window_info["configuration"].get("font_size", numeric.font().pointSize())
        numeric.set_font_size(font_size)

        columns_visibility = window_info["configuration"].get("columns_visibility", {})
        if columns_visibility:
            numeric.channels.columnHeader.toggle_column(
                columns_visibility["raw"], numeric.channels.columnHeader.RawColumn
            )
            numeric.channels.columnHeader.toggle_column(
                columns_visibility["scaled"], numeric.channels.columnHeader.ScaledColumn
            )
            numeric.channels.columnHeader.toggle_column(
                columns_visibility["unit"], numeric.channels.columnHeader.UnitColumn
            )

        header_and_controls_visible = window_info["configuration"].get("header_and_controls_visible", True)
        if not header_and_controls_visible:
            numeric.controls.setHidden(True)
            numeric.channels.columnHeader.setHidden(True)

        sorting = window_info["configuration"].get("sorting", {})
        if sorting:
            enabled = sorting["enabled"]
            if enabled:
                numeric.channels.columnHeader.backend.sort_reversed = not sorting["reversed"]
            else:
                numeric.channels.columnHeader.backend.sort_reversed = sorting["reversed"]

            numeric.channels.columnHeader.backend.sorting_enabled = sorting["enabled"]
            numeric.channels.columnHeader.backend.sort_column(sorting["sort_column"])

            if not enabled:
                numeric.channels.columnHeader.backend.reorder(
                    [s["name"] for s in window_info["configuration"]["channels"]]
                )

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

        if self.subplots_link:
            gps.timestamp_changed_signal.connect(self.set_cursor)

        return w, False

    def _load_plot_window(self, window_info):
        geometry = window_info.get("geometry", None)

        # patterns
        pattern_info = window_info["configuration"].get("pattern", {})
        if pattern_info:
            if self.comparison:
                plot_signals = {}
                for file_index, file in enumerate(self.iter_files()):
                    new_plot_signals = extract_signals_using_pattern(
                        mdf=file.mdf,
                        channels_db=None,
                        pattern_info=pattern_info,
                        ignore_value2text_conversions=self.ignore_value2text_conversions,
                        uuid=file.uuid,
                    )

                    plot_signals |= new_plot_signals

            else:
                plot_signals = extract_signals_using_pattern(
                    mdf=self.mdf,
                    channels_db=None,
                    pattern_info=pattern_info,
                    ignore_value2text_conversions=self.ignore_value2text_conversions,
                    uuid=self.uuid,
                )

            mime_data = None
            for sig in plot_signals.values():
                sig.ranges = copy_ranges(pattern_info["ranges"])
            descriptions = {}

        else:
            if self.comparison:
                mdfs = [file.mdf for file in self.iter_files()]
            else:
                mdfs = [self.mdf]
            (
                mime_data,
                descriptions,
                found,
                not_found,
                computed,
            ) = build_mime_from_config(window_info["configuration"]["channels"], mdfs)

            plot_signals = {}
            measured_signals = {}
            for mdf in mdfs:
                origin_uuid = mdf.uuid
                origin_mdf = mdf.original_name.name

                measured_signals[origin_uuid] = mdf_measured = {}
                mdf_not_found = {uuid: item for uuid, item in not_found.items() if item["origin_uuid"] == origin_uuid}
                mdf_not_found_names = {item["name"]: item for item in mdf_not_found.values()}
                mdf_computed = {uuid: item for uuid, item in computed.items() if item["origin_uuid"] == origin_uuid}
                mdf_found = {uuid: item for uuid, item in found.items() if item["origin_uuid"] == origin_uuid}

                for (sig_uuid, sig_item), signal in zip(
                    mdf_found.items(),
                    mdf.select(
                        [(item["name"], item["group_index"], item["channel_index"]) for item in mdf_found.values()],
                        ignore_value2text_conversions=self.ignore_value2text_conversions,
                        copy_master=False,
                        validate=True,
                        raw=True,
                    ),
                    strict=False,
                ):
                    signal.flags &= ~signal.Flags.computed
                    signal.computation = {}
                    signal.color = sig_item["color"]
                    signal.group_index = sig_item["group_index"]
                    signal.channel_index = sig_item["channel_index"]
                    signal.origin_uuid = origin_uuid
                    signal.origin_mdf = origin_mdf
                    signal.name = sig_item["name"]
                    signal.mode = sig_item.get("mode", "phys")
                    signal.uuid = sig_uuid

                    mdf_measured[signal.name] = signal
                    plot_signals[sig_uuid] = signal

                matrix_components = []
                for nf_name in mdf_not_found_names:
                    name, indexes = parse_matrix_component(nf_name)
                    if indexes and name in mdf:
                        matrix_components.append((name, indexes))

                matrix_signals = {
                    str(matrix_element): sig
                    for sig, matrix_element in zip(
                        mdf.select(
                            [el[0] for el in matrix_components],
                            ignore_value2text_conversions=self.ignore_value2text_conversions,
                            copy_master=False,
                        ),
                        matrix_components,
                        strict=False,
                    )
                }

                for signal_mat, (_n, indexes) in zip(matrix_signals.values(), matrix_components, strict=False):
                    indexes_string = "".join(f"[{_index}]" for _index in indexes)
                    sig_name = f"{signal_mat.name}{indexes_string}"

                    if sig_name in mdf_not_found_names:
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

                        description = mdf_not_found_names[sig_name]
                        sig_uuid = description["uuid"]

                        signal.color = description["color"]
                        signal.flags &= ~signal.Flags.computed
                        signal.computation = {}
                        signal.origin_uuid = origin_uuid
                        signal.origin_mdf = origin_mdf
                        signal.name = sig_name
                        signal.mode = description.get("mode", "phys")
                        signal.uuid = sig_uuid

                        mdf_measured[signal.name] = signal

                        plot_signals[sig_uuid] = signal

                if mdf_measured:
                    all_timebase = np.unique(
                        np.concatenate(
                            list({id(sig.timestamps): sig.timestamps for sig in mdf_measured.values()}.values())
                        )
                    )
                else:
                    all_timebase = []

                required_channels = []
                for ch in mdf_computed.values():
                    required_channels.extend(get_required_from_computed(ch))

                required_channels = set(required_channels)

                required_channels = [
                    (channel, *mdf.whereis(channel)[0])
                    for channel in required_channels
                    if channel not in list(mdf_measured) and channel in mdf
                ]
                required_channels = {
                    sig.name: sig
                    for sig in mdf.select(
                        required_channels,
                        ignore_value2text_conversions=self.ignore_value2text_conversions,
                        copy_master=False,
                    )
                }

                required_channels.update(mdf_measured)

                for sig_uuid, channel in mdf_computed.items():
                    computation = channel["computation"]

                    signal = compute_signal(
                        computation,
                        required_channels,
                        all_timebase,
                        self.functions,
                        self.global_variables,
                    )
                    signal.color = channel["color"]
                    signal.flags |= signal.Flags.computed
                    signal.computation = channel["computation"]
                    signal.name = channel["name"]
                    signal.unit = channel["unit"]
                    signal.group_index = -1
                    signal.channel_index = -1
                    signal.origin_uuid = origin_uuid
                    signal.origin_mdf = origin_mdf
                    signal.uuid = sig_uuid

                    if channel["flags"] & Signal.Flags.user_defined_conversion:
                        signal.conversion = from_dict(channel["conversion"])
                        signal.flags |= signal.Flags.user_defined_conversion

                    if channel["flags"] & Signal.Flags.user_defined_name:
                        signal.original_name = channel["name"]
                        signal.name = channel.get("user_defined_name", "") or ""
                        signal.flags |= signal.Flags.user_defined_name

                    plot_signals[sig_uuid] = signal

                for uuid, description in mdf_not_found.items():
                    if uuid not in plot_signals:
                        sig = Signal([], [], name=description["name"])
                        sig.uuid = uuid

                        sig.origin_uuid = origin_uuid
                        sig.origin_mdf = origin_mdf
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

                        plot_signals[uuid] = sig

        signals = {
            sig_uuid: sig
            for sig_uuid, sig in plot_signals.items()
            if sig.samples.dtype.kind not in "SU" and not sig.samples.dtype.names and not len(sig.samples.shape) > 1
        }

        if not self.comparison:
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
            mdf = self.mdf

        else:
            events = []
            origin = next(self.iter_files()).mdf.start_time
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
        sub.pattern_modified.connect(self.window_pattern_modified)

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
        plot.show_overlapping_alias.connect(self._show_overlapping_alias)
        plot.show_properties.connect(self._show_info)

        plot.add_new_channels(signals, mime_data)

        # plot.show()

        plot.add_channels_request.connect(partial(self.add_new_channels, widget=plot))
        plot.edit_channel_request.connect(partial(self.edit_channel, widget=plot))

        if self.subplots_link:
            plot.x_range_changed_signal.connect(self.set_x_range)
            plot.cursor_moved_signal.connect(self.set_cursor)
            plot.region_removed_signal.connect(self.remove_region)
            plot.region_moved_signal.connect(self.set_region)
            plot.splitter_moved.connect(self.set_splitter)

            for i, mdi in enumerate(self.mdi_area.subWindowList()):
                widget = mdi.widget()
                if isinstance(widget, Plot):
                    plot.plot.viewbox.setXRange(*widget.plot.viewbox.viewRange()[0], padding=0, update=True)
                    break

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

        if self.comparison:
            plot.channel_selection.setColumnHidden(plot.channel_selection.OriginColumn, False)

        if "common_axis_y_range" in window_info["configuration"]:
            plot.plot.common_axis_y_range = tuple(window_info["configuration"]["common_axis_y_range"])

        # keep compatibility with older asammdf versions
        if "channels_header_sizes" in window_info["configuration"]:
            width, sizes = window_info["configuration"]["channels_header_sizes"]
            current_width = sum(plot.splitter.sizes())
            plot.splitter.setSizes([width, max(current_width - width, 50)])
            plot.channel_selection.set_header_sizes(sizes)

        elif "channels_header" in window_info["configuration"]:
            width, sizes = window_info["configuration"]["channels_header"]
            current_width = sum(plot.splitter.sizes())
            plot.splitter.setSizes([width, max(current_width - width, 50)])
            for i, size in enumerate(sizes):
                plot.channel_selection.setColumnWidth(i, size)

        # keep compatibility with older asammdf versions
        if "channels_header_columns_visiblity" in window_info["configuration"]:
            plot.channel_selection.set_header_columns_visibility(
                window_info["configuration"]["channels_header_columns_visiblity"]
            )

        elif "channels_header_columns_visible" in window_info["configuration"]:
            for i, visible in enumerate(window_info["configuration"]["channels_header_columns_visible"]):
                plot.channel_selection.setColumnHidden(i, not visible)

        hide_missing = window_info["configuration"].get("hide_missing_channels", False)
        hide_disabled = window_info["configuration"].get("hide_disabled_channels", False)
        if hide_missing or hide_disabled:
            plot.channel_selection.hide_missing_channels = hide_missing
            plot.channel_selection.hide_disabled_channels = hide_disabled
            plot.channel_selection.update_hidden_states()

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

            ranges = {sig.name: copy_ranges(pattern_info["ranges"]) for sig in signals_}
            signals_ = [(sig.name, sig.group_index, sig.channel_index) for sig in signals_]

        else:
            required = set(window_info["configuration"]["channels"])

            signals_ = [
                (name, *self.mdf.whereis(name)[0])
                for name in window_info["configuration"]["channels"]
                if name in self.mdf
            ]

            ranges = window_info["configuration"].get("ranges", {})

            if not signals_:
                return None, False

        signals = self.mdf.to_dataframe(
            channels=signals_,
            time_from_zero=False,
            ignore_value2text_conversions=self.ignore_value2text_conversions,
            use_interpolation=QtCore.QSettings().value("tabular_interpolation", True, type=bool),
        )

        found = set(signals.columns)
        dim = len(signals.index)

        for name in sorted(required - found):
            vals = np.empty(dim)
            vals.fill(np.nan)
            signals[name] = pd.Series(vals, index=signals.index)

        tabular = Tabular(
            signals,
            ranges=ranges,
            start=self.mdf.header.start_time,
            parent=self,
            owner=self,
        )
        tabular.pattern = pattern_info

        sub = MdiSubWindow(parent=self)
        sub.setWidget(tabular)
        tabular.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.sigClosed.connect(self.window_closed_handler)
        sub.titleModified.connect(self.window_closed_handler)
        sub.pattern_modified.connect(self.window_pattern_modified)

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

        widget = self._add_lin_bus_trace_window(ranges)

        sections_width = window_info["configuration"].get("header_sections_width", [])
        if sections_width:
            for i, width in enumerate(sections_width):
                widget.tree.columnHeader.setColumnWidth(i, width)
                widget.tree.dataView.setColumnWidth(i, width)

            widget.tree.dataView.updateGeometry()
            widget.tree.columnHeader.updateGeometry()

        return None, False

    def _load_xy_window(self, window_info):
        geometry = window_info.get("geometry", None)

        x, y = window_info["configuration"]["channels"]

        if x in self.mdf:
            (x,) = self.mdf.select(
                [(x, *self.mdf.whereis(x)[0])],
                ignore_value2text_conversions=True,
                copy_master=False,
                validate=True,
                raw=False,
            )
        else:
            x = None

        if y in self.mdf:
            (y,) = self.mdf.select(
                [(y, *self.mdf.whereis(y)[0])],
                ignore_value2text_conversions=True,
                copy_master=False,
                validate=True,
                raw=False,
            )
        else:
            y = None

        xy = XY(x, y, color=window_info["configuration"]["color"])

        sub = MdiSubWindow(parent=self)
        sub.setWidget(xy)
        xy.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
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

        xy.add_channels_request.connect(partial(self.add_new_channels, widget=xy))
        if self.subplots_link:
            xy.timestamp_changed_signal.connect(self.set_cursor)

        return w, xy

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
                elif widget:
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
                        widget.region_moved_signal.disconnect(self.set_region)
                    except:
                        pass
                    try:
                        widget.splitter_moved.disconnect(self.set_splitter)
                    except:
                        pass
                elif widget:
                    try:
                        widget.timestamp_changed_signal.disconnect(self.set_cursor)
                    except:
                        pass

    def set_cursor(self, widget, pos):
        if self._busy:
            return
        else:
            self._busy = True

        if not self.subplots_link:
            self._busy = False
            return

        active_window = self.mdi_area.currentSubWindow()
        if active_window is None:
            self._busy = False
            return

        active_widget = active_window.widget()

        if widget is not active_widget:
            self._busy = False
            return

        for mdi in self.mdi_area.subWindowList():
            wid = mdi.widget()
            if wid is not widget:
                try:
                    wid.set_timestamp(pos)
                except:
                    print(format_exc())

        self._busy = False

    def set_x_range(self, widget, x_range):
        if self._busy:
            return
        else:
            self._busy = True

        if not self.subplots_link:
            self._busy = False
            return

        if not isinstance(x_range, (tuple, list)):
            self._busy = False
            return

        if not len(x_range) == 2:
            self._busy = False
            return

        if np.any(np.isnan(x_range)) or not np.all(np.isfinite(x_range)):
            self._busy = False
            return

        for mdi in self.mdi_area.subWindowList():
            wid = mdi.widget()
            if wid is not widget and isinstance(wid, Plot):
                try:
                    wid._inhibit_x_range_changed_signal = True
                    wid.plot.viewbox.setXRange(*x_range, padding=0, update=True)
                    wid._inhibit_x_range_changed_signal = False
                except:
                    print(format_exc())

        self._busy = False

    def set_region(self, widget, region):
        if self._busy:
            return
        else:
            self._busy = True

        if not self.subplots_link:
            self._busy = False
            return

        active_window = self.mdi_area.currentSubWindow()
        if active_window is None:
            self._busy = False
            return

        active_widget = active_window.widget()

        if widget is not active_widget:
            self._busy = False
            return

        for mdi in self.mdi_area.subWindowList():
            wid = mdi.widget()
            if isinstance(wid, Plot) and wid is not widget:
                try:
                    if wid.plot.region is None:
                        event = QtGui.QKeyEvent(
                            QtCore.QEvent.Type.KeyPress,
                            QtCore.Qt.Key.Key_R,
                            QtCore.Qt.KeyboardModifier.NoModifier,
                        )
                        wid.plot.keyPressEvent(event)
                    wid.plot.region.setRegion(region)
                except:
                    print(format_exc())

        self._busy = False

    def set_splitter(self, widget, selection_width):
        if self._busy:
            return
        else:
            self._busy = True

        if not self.subplots_link:
            self._busy = False
            return

        if self._splitter_source is None:
            self._splitter_source = widget
            for mdi in self.mdi_area.subWindowList():
                wid = mdi.widget()
                if isinstance(wid, Plot) and wid is not widget:
                    if selection_width is not None:
                        try:
                            total_size = sum(wid.splitter.sizes())
                            if total_size > selection_width:
                                wid.splitter.setSizes([selection_width, total_size - selection_width])
                        except:
                            print(format_exc())

            self._splitter_source = None

        self._busy = False

    def update_comparison_windows(self):
        if not self.comparison:
            return

        uuids = {file.mdf.uuid for file in self.iter_files()}

        windows = list(self.mdi_area.subWindowList())
        for window in windows:
            widget = window.widget()
            widget.update_missing_signals(uuids)

    def update_functions(self, original_definitions, modified_definitions, new_global_variables):
        self.global_variables = new_global_variables
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

                            if new_global_variables or function in changed:
                                try:
                                    item.signal.computation["function"] = translation[
                                        item.signal.computation["function"]
                                    ]

                                    func_name = item.signal.computation["function"]
                                    definition = self.functions[func_name]
                                    _globals = generate_python_function_globals()
                                    exec(definition.replace("\t", "    "), _globals)
                                    func = _globals[func_name]

                                    parameters = list(inspect.signature(func).parameters)[:-1]
                                    args = {name: [] for name in parameters}
                                    for arg_name, alternatives in zip(
                                        parameters,
                                        item.signal.computation["args"].values(),
                                        strict=False,
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

    def iter_files(self):
        if isinstance(self.files, QtWidgets.QMdiArea):
            for file_index, file_window in enumerate(self.files.subWindowList()):
                if widget := file_window.widget():
                    yield widget
        else:
            for file_index in range(self.files.count()):
                if widget := self.files.widget(file_index):
                    yield widget

    def _show_overlapping_alias(self, sig):
        group_index, index, uuid = sig.group_index, sig.channel_index, sig.origin_uuid
        file_info = self.file_by_uuid(uuid)
        if file_info:
            _, file = file_info
            try:
                channel = file.mdf.get_channel_metadata(group=group_index, index=index)
                info = (channel.data_type, channel.byte_offset, channel.bit_count)
                position = (group_index, index)
                alias = {}
                for gp_index, gp in enumerate(file.mdf.groups):
                    for ch_index, ch in enumerate(gp.channels):
                        if (gp_index, ch_index) != position and (ch.data_type, ch.byte_offset, ch.bit_count) == info:
                            alias[ch.name] = (gp_index, ch_index)

                if alias:
                    alias_text = "\n".join(
                        f"{name} - group {gp_index} index {ch_index}" for name, (gp_index, ch_index) in alias.items()
                    )
                    MessageBox.information(
                        self,
                        f"{channel.name} - other overlapping alias",
                        f"{channel.name} has the following overlapping alias channels:\n\n{alias_text}",
                    )
                else:
                    MessageBox.information(
                        self,
                        f"{channel.name} - no other overlapping alias",
                        f"No other overlapping alias channels found for {channel.name}",
                    )

            except MdfException:
                print(format_exc())

    def _show_info(self, sig):
        group_index, index = sig.group_index, sig.channel_index
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
        if self.comparison:
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
            "You have modified bookmarks.\n\nDo you want to save the changes in the measurement file?\n",
        )

        if result == MessageBox.StandardButton.No:
            return

        _password = self.mdf._mdf._password

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

    def window_pattern_modified(self, pattern, window_id):
        for window in self.mdi_area.subWindowList():
            if id(window) == window_id:
                wid = window.widget()
                wid.pattern = pattern
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
                elif isinstance(wid, Tabular):
                    window_config["type"] = "Tabular"

                del wid
                window.close()

                self.load_window(window_config)
                break

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

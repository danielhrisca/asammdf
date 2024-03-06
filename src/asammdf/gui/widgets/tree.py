from datetime import datetime
from enum import IntFlag
from functools import lru_cache
import json
import os
import re

import numpy as np
from pyqtgraph import functions as fn
from PySide6 import QtCore, QtGui, QtWidgets

from ...blocks.conversion_utils import from_dict, to_dict
from ...blocks.utils import extract_mime_names
from ...signal import Signal
from .. import utils
from ..dialogs.advanced_search import AdvancedSearch
from ..dialogs.conversion_editor import ConversionEditor
from ..dialogs.messagebox import MessageBox
from ..dialogs.range_editor import RangeEditor
from ..utils import (
    copy_ranges,
    get_color_using_ranges,
    get_colors_using_ranges,
    SCROLLBAR_STYLE,
    unique_ranges,
    value_as_str,
)
from .tree_item import MinimalTreeItem

NOT_FOUND = 0xFFFFFFFF


def substitude_mime_uuids(mime, uuid=None, force=False, random_uuid=False):
    if not mime:
        return mime

    new_mime = []

    for item in mime:
        if item["type"] == "channel":
            if force or item["origin_uuid"] is None:
                item["origin_uuid"] = uuid
            new_mime.append(item)
        else:
            item["channels"] = substitude_mime_uuids(item["channels"], uuid, force=force, random_uuid=random_uuid)
            if force or item["origin_uuid"] is None:
                item["origin_uuid"] = uuid
            new_mime.append(item)

        if random_uuid:
            item["uuid"] = os.urandom(6).hex()

    return new_mime


def add_children(
    widget,
    channels,
    channel_dependencies,
    signals,
    entries=None,
    origin_uuid=None,
    version="4.11",
):
    children = []
    if entries is not None:
        channels_ = [channels[i] for _, i in entries]
    else:
        channels_ = channels

    for ch in channels_:
        if ch.added:
            continue

        entry = ch.entry

        child = MinimalTreeItem(entry, ch.name, strings=[ch.name], origin_uuid=origin_uuid)

        dep = channel_dependencies[entry[1]]
        if version >= "4.00":
            if dep and isinstance(dep[0], tuple):
                child.setFlags(
                    child.flags() | QtCore.Qt.ItemFlag.ItemIsAutoTristate | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                )

                add_children(
                    child,
                    channels,
                    channel_dependencies,
                    signals,
                    dep,
                    origin_uuid=origin_uuid,
                )

        if entry in signals:
            child.setCheckState(0, QtCore.Qt.CheckState.Checked)
        else:
            child.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

        ch.added = True
        children.append(child)

    widget.addChildren(children)


def add_new_items(tree, root, items, pos):
    for item in items:
        new_item = item.copy()
        new_widget = tree.itemWidget(item, 1).copy()
        new_item.widget = new_widget
        new_widget.item = new_item

        if pos is None:
            root.addChild(new_item)
        else:
            root.insertChild(pos, new_item)
            pos += 1

        tree.setItemWidget(new_item, 1, new_widget)

        if item.type() == ChannelsTreeItem.Group:
            child_items = [item.child(i) for i in range(item.childCount())]
            add_new_items(tree, new_item, child_items, None)


def valid_drop_target(target, item):
    if target is None:
        return True

    while target.parent():
        if target.parent() is item:
            return False
        target = target.parent()

    return True


def validate_drag_items(root, items, not_allowed):
    valid_items = []

    for item in items:
        if item.type() == ChannelsTreeItem.Info:
            continue

        idx = root.indexOfChild(item)
        if idx >= 0:
            parents = []
            parent = item.parent()
            while parent:
                parents.append(parent)
                parent = parent.parent()

            for parent in parents:
                if parent in not_allowed:
                    break
                elif parent and parent.type() == ChannelsTreeItem.Group:
                    if parent.pattern:
                        not_allowed.append(parent)
                        break

            else:
                not_allowed.append(item)
                valid_items.append(item)

    for item in valid_items:
        pos = items.index(item)
        items.pop(pos)

    for i in range(root.childCount()):
        child = root.child(i)
        if child.childCount():
            valid_items.extend(validate_drag_items(child, items, not_allowed))

    return valid_items


def get_data(plot, items, uuids_only=False):
    data = []

    for item in items:
        if item.type() == ChannelsTreeItem.Group:
            children = [item.child(i) for i in range(item.childCount())]

            if uuids_only:
                data.extend(get_data(plot, children, uuids_only))
            else:
                group = plot.channel_group_item_to_config(item)
                group["uuid"] = os.urandom(6).hex()
                group["channels"] = get_data(plot, children, uuids_only)

                data.append(group)

        elif item.type() == ChannelsTreeItem.Channel:
            if uuids_only:
                data.append(item.uuid)
            else:
                channel = plot.channel_item_to_config(item)
                channel["uuid"] = os.urandom(6).hex()
                channel["group_index"], channel["channel_index"] = item.entry

                data.append(channel)

    return data


class TreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSortingEnabled(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)
        self.setUniformRowHeights(True)

        self.mode = "Natural sort"

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key.Key_Space:
            event.accept()
            selected_items = self.selectedItems()
            if not selected_items:
                return
            elif len(selected_items) == 1:
                item = selected_items[0]
                checked = item.checkState(0)
                if checked == QtCore.Qt.CheckState.Checked:
                    item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(0, QtCore.Qt.CheckState.Checked)
            else:
                if any(item.checkState(0) == QtCore.Qt.CheckState.Unchecked for item in selected_items):
                    checked = QtCore.Qt.CheckState.Checked
                else:
                    checked = QtCore.Qt.CheckState.Unchecked
                for item in selected_items:
                    item.setCheckState(0, checked)
        else:
            super().keyPressEvent(event)

    def startDrag(self, supportedActions):
        def get_data(item):
            data = []
            count = item.childCount()

            if count:
                for i in range(count):
                    child = item.child(i)

                    if child.childCount():
                        data.extend(get_data(child))
                    else:
                        if child.entry[1] != 0xFFFFFFFFFFFFFFFF:
                            data.append(
                                {
                                    "name": child.name,
                                    "group_index": child.entry[0],
                                    "channel_index": child.entry[1],
                                    "type": "channel",
                                    "ranges": [],
                                    "uuid": os.urandom(6).hex(),
                                    "origin_uuid": child.origin_uuid,
                                    "computed": False,
                                    "computation": None,
                                }
                            )

            else:
                if item.entry[1] != 0xFFFFFFFFFFFFFFFF:
                    data.append(
                        {
                            "name": item.name,
                            "group_index": item.entry[0],
                            "channel_index": item.entry[1],
                            "type": "channel",
                            "ranges": [],
                            "uuid": os.urandom(6).hex(),
                            "origin_uuid": item.origin_uuid,
                            "computed": False,
                            "computation": None,
                        }
                    )

            return data

        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = []
        for item in selected_items:
            data.extend(get_data(item))

        data = json.dumps(sorted(data, key=lambda x: (x["name"], x["group_index"], x["channel_index"]))).encode("utf-8")

        mimeData.setData("application/octet-stream-asammdf", QtCore.QByteArray(data))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec(QtCore.Qt.DropAction.MoveAction)


class FileTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, path, start_time, parent=None):
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()

        super().__init__(parent, [path, start_time])

    def __lt__(self, otherItem):
        column = self.treeWidget().sortColumn()

        if column == 1:
            val1 = datetime.fromisoformat(self.text(column))
            val2 = datetime.fromisoformat(otherItem.text(column))

            return val1 < val2
        else:
            return self.text(column) < otherItem.text(column)

    def __del__(self):
        self.entry = self.name = self.origin_uuid = None


class FileTreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setUniformRowHeights(True)

        self.mode = "Natural sort"


class Delegate(QtWidgets.QItemDelegate):
    def __init__(self, *args):
        super().__init__(*args)

    def paint(self, painter, option, index):
        model = index.model()

        item = self.parent().itemFromIndex(index)

        if item.type() == item.Channel:
            brush = model.data(index, QtCore.Qt.ItemDataRole.ForegroundRole)

            if brush is not None:
                color = brush.color()

                complementary = fn.mkColor("#000000")
                if complementary == color:
                    complementary = fn.mkColor("#FFFFFF")

                option.palette.setColor(QtGui.QPalette.ColorRole.Highlight, color)
                option.palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, complementary)

        super().paint(painter, option, index)


class ChannelsTreeFlags(IntFlag):
    DetailsEnabled = 1
    HideMissingChannels = 1 << 1
    HideDisabledChannels = 1 << 2
    CanDeleteItems = 1 << 3
    CanInsertItems = 1 << 4


class ChannelsTreeWidget(QtWidgets.QTreeWidget):
    itemsDeleted = QtCore.Signal(list)
    set_time_offset = QtCore.Signal(list)
    add_channels_request = QtCore.Signal(list)
    show_properties = QtCore.Signal(object)
    insert_computation = QtCore.Signal(str)
    edit_computation = QtCore.Signal(object)
    pattern_group_added = QtCore.Signal(object)
    compute_fft_request = QtCore.Signal(str)
    conversion_changed = QtCore.Signal(str, object)
    color_changed = QtCore.Signal(str, object)
    unit_changed = QtCore.Signal(str, str)
    name_changed = QtCore.Signal(str, str)
    visible_items_changed = QtCore.Signal()
    group_activation_changed = QtCore.Signal()
    double_click = QtCore.Signal(object, object)

    NameColumn = 0
    ValueColumn = 1
    UnitColumn = 2
    CommonAxisColumn = 3
    IndividualAxisColumn = 4

    def __init__(
        self,
        hide_missing_channels=False,
        hide_disabled_channels=False,
        plot=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.plot = plot
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        self.setUniformRowHeights(True)

        self.details_enabled = False
        self.hide_missing_channels = hide_missing_channels
        self.hide_disabled_channels = hide_disabled_channels
        self.filter_computed_channels = False
        self.can_delete_items = True

        self.setHeaderHidden(False)
        self.setColumnCount(5)
        # self.setHeaderLabels(["Name", "Value", "Unit", "\u27f0", "\u21A8"])
        self.setHeaderLabels(["Name", "Value", "Unit", "\u290a", "\u21A8"])
        self.setDragEnabled(True)
        self.setExpandsOnDoubleClick(False)

        self.setMinimumWidth(5)
        self.header().setMinimumSectionSize(10)
        self.header().resizeSection(self.CommonAxisColumn, 10)
        self.header().resizeSection(self.IndividualAxisColumn, 10)
        self.header().setSectionResizeMode(self.CommonAxisColumn, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.header().setSectionResizeMode(self.IndividualAxisColumn, QtWidgets.QHeaderView.ResizeMode.Fixed)

        self.header().setStretchLastSection(False)

        self.itemSelectionChanged.connect(self.item_selection_changed)

        self.itemCollapsed.connect(self.update_visibility_status)
        self.itemExpanded.connect(self.update_visibility_status)
        self.verticalScrollBar().valueChanged.connect(self.update_visibility_status)

        self.autoscroll_timer = QtCore.QTimer()
        self.autoscroll_timer.timeout.connect(self.autoscroll)
        self.autoscroll_timer.setInterval(33)
        self.context_menu_timer = QtCore.QTimer()
        self.context_menu_timer.timeout.connect(self.open_menu)
        self.context_menu_timer.setSingleShot(True)
        self.context_menu_pos = None
        self.context_menu = None
        self.autoscroll_mouse_pos = None
        self.drop_target = None
        self.idel = Delegate(self)
        self.setItemDelegate(self.idel)

        settings = QtCore.QSettings()

        background = QtGui.QColor(55, 55, 55).name()
        if settings.value("current_theme") == "Dark":
            self._dark = True
            self._font_size = self.font().pointSize()
            self._background = background
            self._style = SCROLLBAR_STYLE
            self.setStyleSheet(self._style.format(font_size=self._font_size, background=self._background))
        else:
            self._dark = False

    def autoscroll(self):
        step = max(
            (self.verticalScrollBar().maximum() - self.verticalScrollBar().minimum()) // 90,
            1,
        )

        if self.autoscroll_mouse_pos is not None:
            height = self.viewport().rect().height()
            y = self.autoscroll_mouse_pos

            if y <= 15:
                pos = max(self.verticalScrollBar().value() - step, 0)
                self.verticalScrollBar().setValue(pos)
            elif y >= height - 15:
                pos = min(
                    self.verticalScrollBar().value() + step,
                    self.verticalScrollBar().maximum(),
                )
                self.verticalScrollBar().setValue(pos)

    def startDrag(self, supportedActions):
        selected_items = validate_drag_items(self.invisibleRootItem(), self.selectedItems(), [])

        mimeData = QtCore.QMimeData()

        data = get_data(self.plot, selected_items, uuids_only=False)
        data = json.dumps(data).encode("utf-8")

        mimeData.setData("application/octet-stream-asammdf", QtCore.QByteArray(data))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec(QtCore.Qt.DropAction.MoveAction)

    def dragEnterEvent(self, e):
        self.autoscroll_timer.start()
        self.autoscroll_mouse_pos = e.answerRect().y()
        e.accept()

    def dragLeaveEvent(self, e):
        self.autoscroll_timer.stop()
        self.autoscroll_mouse_pos = None
        e.accept()

    def dragMoveEvent(self, e):
        self.autoscroll_mouse_pos = e.answerRect().y()
        e.accept()

    def dropEvent(self, e):
        self.autoscroll_timer.stop()
        self.autoscroll_mouse_pos = None
        self.drop_target = None

        if e.source() is self:
            item = self.itemAt(6, 6)
            self.blockSignals(True)
            super().dropEvent(e)
            self.blockSignals(False)
            self.scrollToItem(item)

        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                item = self.itemAt(e.pos())

                if item and item.type() == item.Info:
                    item = item.parent()
                self.drop_target = item

                self.add_channels_request.emit(names)
            else:
                super().dropEvent(e)

        self.refresh()
        self.update_channel_groups_count()

    def is_item_visible(self, item):
        return item._is_visible

    def item_selection_changed(self):
        selection = list(self.selectedItems())

        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while item := iterator.value():
            widget = self.itemWidget(item, 1)

            if widget:
                if item in selection:
                    widget.set_selected(True)
                    selection.remove(item)
                else:
                    widget.set_selected(False)

            iterator += 1

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key.Key_Delete and self.can_delete_items:
            event.accept()
            selected_items = self.selectedItems()
            deleted = list(set(get_data(self.plot, selected_items, uuids_only=True)))

            self.setUpdatesEnabled(False)
            self.clearSelection()

            root = self.invisibleRootItem()

            for item in selected_items:
                (item.parent() or root).removeChild(item)

            self.setUpdatesEnabled(True)
            self.refresh()

            if deleted:
                self.itemsDeleted.emit(deleted)
            self.update_channel_groups_count()

        elif key == QtCore.Qt.Key.Key_Insert and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.accept()
            if hasattr(self.plot.owner, "mdf"):
                mdf = self.plot.owner.mdf
                channels_db = None
            else:
                mdf = None
                channels_db = self.plot.owner.channels_db

            dlg = AdvancedSearch(
                mdf=mdf,
                show_add_window=False,
                show_apply=True,
                show_search=False,
                show_pattern=True,
                window_title="Add pattern based group",
                parent=self,
                channels_db=channels_db,
            )
            dlg.setModal(True)
            dlg.exec_()
            pattern = dlg.result

            if pattern:
                group = ChannelsTreeItem(ChannelsTreeItem.Group, name=pattern["name"], pattern=pattern)

                item = self.currentItem()

                if item is None:
                    self.addTopLevelItem(group)
                else:
                    parent = item.parent()
                    if parent:
                        current_parent = parent
                        can_add_child = True
                        while current_parent:
                            if current_parent.type() == ChannelsTreeItem.Group:
                                if current_parent.pattern:
                                    can_add_child = False
                                    break
                            current_parent = current_parent.parent()

                        if can_add_child:
                            index = parent.indexOfChild(item)
                            parent.insertChild(index, group)
                        else:
                            self.addTopLevelItem(group)
                    else:
                        index = self.indexOfTopLevelItem(item)
                        self.insertTopLevelItem(index, group)

                self.pattern_group_added.emit(group)

                self.refresh()
                self.update_channel_groups_count()

        elif key == QtCore.Qt.Key.Key_Insert and modifiers == QtCore.Qt.KeyboardModifier.ShiftModifier:
            event.accept()
            text, ok = QtWidgets.QInputDialog.getText(self, "Channel group name", "New channel group name:")
            if ok:
                group = ChannelsTreeItem(ChannelsTreeItem.Group, name=text)

                item = self.currentItem()

                if item is None:
                    self.addTopLevelItem(group)
                else:
                    parent = item.parent()
                    if parent:
                        index = parent.indexOfChild(item)
                        parent.insertChild(index, group)
                    else:
                        index = self.indexOfTopLevelItem(item)
                        self.insertTopLevelItem(index, group)

                self.refresh()
                self.update_channel_groups_count()

        elif key == QtCore.Qt.Key.Key_Space:
            event.accept()
            selected_items = self.selectedItems()
            if not selected_items:
                return
            elif len(selected_items) == 1:
                item = selected_items[0]
                checked = item.checkState(self.NameColumn)
                if checked == QtCore.Qt.CheckState.Checked:
                    item.setCheckState(self.NameColumn, QtCore.Qt.CheckState.Unchecked)
                    if self.hide_disabled_channels:
                        item.setHidden(True)
                else:
                    item.setCheckState(self.NameColumn, QtCore.Qt.CheckState.Checked)
            else:
                if any(item.checkState(self.NameColumn) == QtCore.Qt.CheckState.Unchecked for item in selected_items):
                    checked = QtCore.Qt.CheckState.Checked
                else:
                    checked = QtCore.Qt.CheckState.Unchecked
                for item in selected_items:
                    item.setCheckState(self.NameColumn, checked)
                    if self.hide_disabled_channels and checked == QtCore.Qt.CheckState.Unchecked:
                        item.setHidden(True)

        elif modifiers == QtCore.Qt.KeyboardModifier.NoModifier and key == QtCore.Qt.Key.Key_C:
            event.accept()
            selected_items = self.selectedItems()
            if not selected_items:
                return
            else:
                for item in selected_items:
                    if item.type() == item.Channel:
                        color = item.color
                        break
                else:
                    color = QtGui.QColor("#ffffff")

            color = QtWidgets.QColorDialog.getColor(color, parent=self)
            if color.isValid():
                for item in selected_items:
                    if item.type() != item.Info:
                        item.color = color

        elif modifiers == QtCore.Qt.KeyboardModifier.ControlModifier and key == QtCore.Qt.Key.Key_C:
            event.accept()
            selected_items = validate_drag_items(self.invisibleRootItem(), self.selectedItems(), [])
            data = get_data(self.plot, selected_items, uuids_only=False)
            data = substitude_mime_uuids(data, None, force=True)
            QtWidgets.QApplication.instance().clipboard().setText(json.dumps(data))

        elif modifiers == QtCore.Qt.KeyboardModifier.ControlModifier and key == QtCore.Qt.Key.Key_V:
            event.accept()
            try:
                data = QtWidgets.QApplication.instance().clipboard().text()
                data = json.loads(data)
                data = substitude_mime_uuids(data, random_uuid=True)
                self.add_channels_request.emit(data)
            except:
                pass

        elif modifiers == QtCore.Qt.KeyboardModifier.ControlModifier and key == QtCore.Qt.Key.Key_N:
            event.accept()
            selected_items = self.selectedItems()
            if not selected_items:
                return
            else:
                text = "\n".join(item.name for item in selected_items)

            QtWidgets.QApplication.instance().clipboard().setText(text)

        elif modifiers == QtCore.Qt.KeyboardModifier.ControlModifier and key == QtCore.Qt.Key.Key_R:
            event.accept()
            selected_items = self.selectedItems()
            if not selected_items:
                return

            if len(selected_items) == 1:
                item = selected_items[0]

                type = item.type()
                if type == ChannelsTreeItem.Group:
                    dlg = RangeEditor(f"channels from <{item._name}>", ranges=item.ranges, parent=self)
                    dlg.exec_()
                    if dlg.pressed_button == "apply":
                        item.set_ranges(dlg.result)
                        item.update_child_values()

                elif type == ChannelsTreeItem.Channel:
                    dlg = RangeEditor(item.signal.name, item.unit, item.ranges, parent=self)
                    dlg.exec_()
                    if dlg.pressed_button == "apply":
                        item.set_ranges(dlg.result)
                        item.set_value(item._value, update=True)

            else:
                ranges = []
                for item in selected_items:
                    ranges.extend(item.ranges)
                dlg = RangeEditor("<selected items>", ranges=unique_ranges(ranges), parent=self)
                dlg.exec_()
                if dlg.pressed_button == "apply":
                    for item in selected_items:
                        if item.type() == item.Channel:
                            item.set_ranges(copy_ranges(dlg.result))
                            item.set_value(item._value, update=True)

                        elif item.type() == item.Group:
                            item.set_ranges(copy_ranges(dlg.result))
                            item.update_child_values()

            self.refresh()
            self.plot.plot.update()

        elif (
            modifiers == (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier)
            and key == QtCore.Qt.Key.Key_C
        ):
            event.accept()
            selected_items = [
                item
                for item in self.selectedItems()
                if item.type() in (ChannelsTreeItem.Channel, ChannelsTreeItem.Group)
            ]
            if selected_items:
                item = selected_items[0]
                clipboard_text = item.get_display_properties()
                QtWidgets.QApplication.instance().clipboard().setText(clipboard_text)

        elif (
            modifiers == (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier)
            and key == QtCore.Qt.Key.Key_V
        ):
            event.accept()
            info = QtWidgets.QApplication.instance().clipboard().text()
            selected_items = self.selectedItems()
            if not selected_items:
                return

            try:
                info = json.loads(info)
            except:
                return

            if info["type"] == "channel":
                info["color"] = fn.mkColor(info["color"])

                for item in selected_items:
                    item.color = info["color"]
                    item.precision = info["precision"]
                    item.format = info["format"]

                    item.setCheckState(
                        self.IndividualAxisColumn,
                        QtCore.Qt.CheckState.Checked if info["individual_axis"] else QtCore.Qt.CheckState.Unchecked,
                    )
                    item.setCheckState(
                        self.CommonAxisColumn,
                        QtCore.Qt.CheckState.Checked if info["ylink"] else QtCore.Qt.CheckState.Unchecked,
                    )

                    if item.type() == ChannelsTreeItem.Channel:
                        plot = item.treeWidget().plot.plot
                        sig, index = plot.signal_by_uuid(item.uuid)
                        sig.y_range = info["y_range"]

                        item.set_ranges(info["ranges"])

                        if "conversion" in info:
                            item.set_conversion(from_dict(info["conversion"]))

                    elif item.type() == ChannelsTreeItem.Group:
                        item.set_ranges(info["ranges"])

            elif info["type"] == "group":
                for item in selected_items:
                    if item.type() in (ChannelsTreeItem.Channel, ChannelsTreeItem.Group):
                        item.set_ranges(info["ranges"])

        elif modifiers == QtCore.Qt.KeyboardModifier.ShiftModifier and key in (
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
        ):
            event.ignore()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.context_menu_timer.start(300)
            self.context_menu_pos = event.pos()

        if self.context_menu is not None:
            self.context_menu.close()
            self.context_menu = None

        # If there was a click performed on disabled item, then clear selection
        position = event.position()
        item = self.itemAt(position.x(), position.y())
        if item and item.isDisabled():
            self.clearSelection()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.context_menu_timer.stop()
            self.context_menu_pos = None

        position = event.position()
        self.double_click.emit(self.itemAt(position.x(), position.y()), event.button())

    def open_menu(self):
        position = self.context_menu_pos

        item = self.itemAt(position)

        count = 0
        enabled = 0
        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while cur_item := iterator.value():
            if cur_item.type() == ChannelsTreeItem.Channel:
                count += 1
                if cur_item.checkState(self.NameColumn) == QtCore.Qt.CheckState.Checked:
                    enabled += 1
            iterator += 1

        self.context_menu = menu = QtWidgets.QMenu()
        menu.addAction(self.tr(f"{count} items in the list, {enabled} enabled"))
        menu.addSeparator()
        menu.addAction(QtGui.QIcon(":/search.png"), "Search item")
        if item and item.type() in (ChannelsTreeItem.Channel, ChannelsTreeItem.Group):
            menu.addAction(QtGui.QIcon(":/down.png"), f"Find next {item.name}")
        menu.addSeparator()

        menu.addAction(self.tr("Add channel group [Shift+Insert]"))
        menu.addAction(self.tr("Add pattern based channel group [Ctrl+Insert]"))
        menu.addSeparator()

        menu.addAction(self.tr("Copy names [Ctrl+N]"))
        menu.addAction(self.tr("Copy names and values"))
        menu.addAction(self.tr("Copy display properties [Ctrl+Shift+C]"))
        menu.addAction(self.tr("Paste display properties [Ctrl+Shift+V]"))
        menu.addAction(self.tr("Copy channel structure [Ctrl+C]"))
        menu.addAction(self.tr("Paste channel structure [Ctrl+V]"))

        if item and item.type() == ChannelsTreeItem.Channel:
            menu.addAction(self.tr("Rename channel"))
        menu.addSeparator()

        submenu = QtWidgets.QMenu("Enable/disable")
        if item and item.type() == item.Group and item.isDisabled():
            submenu.addAction(self.tr("Activate group"))
        submenu.addAction(self.tr("Deactivate groups"))
        submenu.addAction(self.tr("Enable all"))
        submenu.addAction(self.tr("Disable all"))
        submenu.addAction(self.tr("Enable selected"))
        submenu.addAction(self.tr("Disable selected"))
        if item:
            submenu.addAction(self.tr("Disable all but this"))
        menu.addMenu(submenu)
        menu.addSeparator()
        submenu = QtWidgets.QMenu("Show/hide")

        if self.hide_disabled_channels:
            show_disabled_channels = "Show disabled items"
        else:
            show_disabled_channels = "Hide disabled items"
        submenu.addAction(self.tr(show_disabled_channels))
        if self.hide_missing_channels:
            show_missing_channels = "Show missing items"
        else:
            show_missing_channels = "Hide missing items"
        submenu.addAction(self.tr(show_missing_channels))
        submenu.addAction("Filter only computed channels")
        submenu.addAction("Un-filter computed channels")
        menu.addMenu(submenu)
        menu.addSeparator()

        menu.addAction(self.tr("Edit Y axis scaling [Ctrl+G]"))
        if item and item.type() == ChannelsTreeItem.Channel:
            menu.addAction(self.tr("Add to common Y axis"))
            menu.addAction(self.tr("Remove from common Y axis"))
        menu.addSeparator()

        menu.addAction(self.tr("Set color [C]"))
        menu.addAction(self.tr("Set random color"))
        menu.addAction(self.tr("Set precision"))
        menu.addAction(self.tr("Set color ranges [Ctrl+R]"))
        menu.addAction(self.tr("Set channel conversion"))
        menu.addAction(self.tr("Set channel comment"))
        menu.addAction(self.tr("Set unit"))
        if item and item.type() == ChannelsTreeItem.Channel and item.signal.flags & Signal.Flags.computed:
            menu.addSeparator()
            menu.addAction(self.tr("Edit this computed channel"))
        menu.addSeparator()

        if item and item.type() == ChannelsTreeItem.Channel:
            menu.addSeparator()
            submenu = QtWidgets.QMenu("Time shift")
            submenu.addAction(self.tr("Relative time base shift"))
            submenu.addAction(self.tr("Set time base start offset"))

            try:
                import scipy  # noqa: F401

                menu.addAction(self.tr("Compute FFT"))
            except ImportError:
                pass

            menu.addMenu(submenu)
            menu.addSeparator()
        if item:
            menu.addAction(self.tr("Delete [Del]"))
            menu.addSeparator()
        menu.addAction(self.tr("Toggle details"))
        if item and item.type() == ChannelsTreeItem.Channel:
            menu.addAction(self.tr("File/Computation properties"))
        elif item and item.type() == ChannelsTreeItem.Group:
            menu.addAction(self.tr("Edit group"))
            menu.addAction(self.tr("Group properties"))

        action = menu.exec(self.viewport().mapToGlobal(position))

        if action is None:
            return

        action_text = action.text()

        if action_text == "Copy names [Ctrl+N]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_N, QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self.keyPressEvent(event)

        elif action_text == "Copy names and values":
            texts = []
            t = self.plot.cursor_info.text()

            if t.startswith("<html>"):
                start = t.index("<p>") + 3
                stop = t.index("</p>")

                t = t[start:stop]

            for item in self.selectedItems():
                texts.append(
                    ", ".join(
                        [
                            item.text(item.NameColumn),
                            t,
                            f"{item.text(item.ValueColumn)}{item.text(item.UnitColumn)}",
                        ]
                    )
                )

            QtWidgets.QApplication.instance().clipboard().setText("\n".join(texts))

        elif action_text == "Activate group":
            # Selected Items will not be retrieved if there are disabled.
            # So Activating multiple groups will not be possible.
            # for item in self.selectedItems():
            if item.type() == item.Group and item.isDisabled():
                item.set_disabled(False)
            self.group_activation_changed.emit()

        elif action_text == "Deactivate groups":
            selectedItems = self.selectedItems()
            selectedItems.append(item)
            for item in selectedItems:
                if item.type() == item.Group and not item.isDisabled():
                    item.set_disabled(True)

            self.group_activation_changed.emit()

        elif action_text == "Set color [C]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_C,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Set random color":
            for item in self.selectedItems():
                if item.type() == ChannelsTreeItem.Channel:
                    while True:
                        rgb = os.urandom(3)
                        if 100 <= sum(rgb) <= 650:
                            break

                    item.color = f"#{rgb.hex()}"

        elif action_text == "Set color ranges [Ctrl+R]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_R,
                QtCore.Qt.KeyboardModifier.ControlModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Set channel conversion":
            selected_items = self.selectedItems()
            if not selected_items:
                return

            if len(selected_items) == 1:
                item = selected_items[0]
                if item.type() == ChannelsTreeItem.Channel:
                    conversion = item.signal.conversion
                    channel_name = item.name
                else:
                    conversion = None
                    channel_name = item.name
            else:
                conversion = None
                channel_name = "selected items"

            dlg = ConversionEditor(channel_name, conversion, parent=self)
            dlg.exec_()
            if dlg.pressed_button == "apply":
                conversion = dlg.conversion()

                for item in selected_items:
                    if item.type() in (
                        ChannelsTreeItem.Channel,
                        ChannelsTreeItem.Group,
                    ):
                        item.set_conversion(conversion)

        elif action_text == "Set channel comment":
            selected_items = self.selectedItems()
            if not selected_items:
                return

            for item in selected_items:
                if item.type() == ChannelsTreeItem.Channel:
                    comment = item.comment

            new_comment, ok = QtWidgets.QInputDialog.getMultiLineText(
                self,
                "Input new comment",
                "New comment:",
                comment,
            )

            if ok:
                for item in selected_items:
                    item.comment = new_comment

        elif action_text == "Copy channel structure [Ctrl+C]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_C,
                QtCore.Qt.KeyboardModifier.ControlModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Paste channel structure [Ctrl+V]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_V,
                QtCore.Qt.KeyboardModifier.ControlModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Copy display properties [Ctrl+Shift+C]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_C,
                QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Paste display properties [Ctrl+Shift+V]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_V,
                QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
            self.keyPressEvent(event)

        elif action_text == "Enable all":
            count = self.topLevelItemCount()
            for i in range(count):
                item = self.topLevelItem(i)
                if item.type() != item.Info:
                    item.setCheckState(self.NameColumn, QtCore.Qt.CheckState.Checked)

        elif action_text == "Disable all":
            count = self.topLevelItemCount()
            for i in range(count):
                item = self.topLevelItem(i)
                if item.type() != item.Info:
                    item.setCheckState(self.NameColumn, QtCore.Qt.CheckState.Unchecked)

        elif action_text == "Enable selected":
            selected_items = self.selectedItems()

            for item in selected_items:
                if item.type() in (item.Channel, item.Group):
                    item.setCheckState(item.NameColumn, QtCore.Qt.CheckState.Checked)

        elif action_text == "Disable selected":
            selected_items = self.selectedItems()

            for item in selected_items:
                if item.type() in (item.Channel, item.Group):
                    item.setCheckState(item.NameColumn, QtCore.Qt.CheckState.Unchecked)

        elif action_text == "Disable all but this":
            selected_items = self.selectedItems()

            count = self.topLevelItemCount()
            for i in range(count):
                item = self.topLevelItem(i)
                if item.type() != item.Info:
                    item.setCheckState(item.NameColumn, QtCore.Qt.CheckState.Unchecked)

            for item in selected_items:
                if item.type() == item.Channel:
                    item.setCheckState(item.NameColumn, QtCore.Qt.CheckState.Checked)
                elif item.type() == item.Group:
                    count = item.childCount()
                    for i in range(count):
                        child = item.child(i)
                        if child in selected_items:
                            break
                    else:
                        item.setCheckState(item.NameColumn, QtCore.Qt.CheckState.Checked)

        elif action_text == show_disabled_channels:
            self.hide_disabled_channels = not self.hide_disabled_channels
            self.update_hidden_states()

        elif action_text == show_missing_channels:
            self.hide_missing_channels = not self.hide_missing_channels
            self.update_hidden_states()

        elif action_text == "Filter only computed channels":
            self.filter_computed_channels = True
            self.update_hidden_states()

        elif action_text == "Un-filter computed channels":
            self.filter_computed_channels = False
            self.update_hidden_states()

        elif action_text == "Add to common Y axis":
            selected_items = self.selectedItems()

            if self.plot.locked:
                for item in selected_items:
                    if item.type() == ChannelsTreeItem.Channel:
                        item.signal.y_link = True
            else:
                for item in selected_items:
                    if item.type() == ChannelsTreeItem.Channel:
                        item.setCheckState(self.CommonAxisColumn, QtCore.Qt.CheckState.Checked)

        elif action_text == "Edit Y axis scaling [Ctrl+G]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_G,
                QtCore.Qt.KeyboardModifier.ControlModifier,
            )
            self.plot.keyPressEvent(event)

        elif action_text == "Remove from common Y axis":
            selected_items = self.selectedItems()

            if self.plot.locked:
                for item in selected_items:
                    if item.type() == ChannelsTreeItem.Channel:
                        item.signal.y_link = False
            else:
                for item in selected_items:
                    if item.type() == ChannelsTreeItem.Channel and item.signal.y_link:
                        item.setCheckState(self.CommonAxisColumn, QtCore.Qt.CheckState.Unchecked)

        elif action_text == "Set unit":
            selected_items = self.selectedItems()

            unit, ok = QtWidgets.QInputDialog.getText(self, "Set new unit", "Unit:")

            if ok:
                for item in selected_items:
                    if item.type() == ChannelsTreeItem.Channel:
                        item.unit = unit

        elif action_text == "Set precision":
            selected_items = self.selectedItems()

            if not selected_items:
                return

            for item in selected_items:
                if item.type() == item.Channel:
                    precision = item.precision
                    break
            else:
                precision = 3

            precision, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Set new precision (float decimals)",
                "Precision:",
                precision,
                -1,
                15,
                1,
            )

            if ok:
                for item in selected_items:
                    if item.type() == ChannelsTreeItem.Channel:
                        item.precision = precision
                    elif item.type() == ChannelsTreeItem.Group:
                        for channel_item in item.get_all_channel_items():
                            channel_item.precision = precision

                plot = self.plot.plot
                if plot.region is not None:
                    self.plot.range_modified(plot.region)
                else:
                    self.plot.cursor_moved(plot.cursor1)

        elif action_text in (
            "Relative time base shift",
            "Set time base start offset",
        ):
            selected_items = self.selectedItems()
            if selected_items:
                if action_text == "Relative time base shift":
                    offset, ok = QtWidgets.QInputDialog.getDouble(
                        self, "Relative offset [s]", "Offset [s]:", decimals=6
                    )
                    absolute = False
                else:
                    offset, ok = QtWidgets.QInputDialog.getDouble(
                        self,
                        "Absolute time start offset [s]",
                        "Offset [s]:",
                        decimals=6,
                    )
                    absolute = True
                if ok:
                    uuids = []

                    for item in selected_items:
                        if item.type() == ChannelsTreeItem.Channel:
                            uuids.append(item.uuid)

                    self.set_time_offset.emit([absolute, offset, *uuids])

        elif action_text == "Delete [Del]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Delete, QtCore.Qt.KeyboardModifier.NoModifier
            )
            self.keyPressEvent(event)

        elif action_text == "Toggle details":
            self.details_enabled = not self.details_enabled

            if self.details_enabled:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while item := iterator.value():
                    if item.type() == ChannelsTreeItem.Channel:
                        item.details = ChannelsTreeItem(
                            type=ChannelsTreeItem.Info,
                            name=item.details_text,
                            signal=item.signal,
                        )

                        item.addChild(item.details)

                        if count <= 200:
                            item.setExpanded(True)

                    iterator += 1

            else:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while item := iterator.value():
                    if item.type() == ChannelsTreeItem.Channel:
                        if item.details:
                            item.removeChild(item.details)
                            item.details = None

                    iterator += 1

        elif action_text in ("File/Computation properties", "Group properties"):
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                if item.type() == ChannelsTreeItem.Channel:
                    self.show_properties.emit(item.uuid)
                elif item.type() == ChannelsTreeItem.Group:
                    item.show_info()

        elif action_text == "Edit this computed channel":
            self.edit_computation.emit(item)

        elif action_text == "Compute FFT":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                if item.type() == ChannelsTreeItem.Channel:
                    self.compute_fft_request.emit(item.uuid)

        elif action_text == "Add channel group [Shift+Insert]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Insert, QtCore.Qt.KeyboardModifier.ShiftModifier
            )
            self.keyPressEvent(event)

        elif action_text == "Add pattern based channel group [Ctrl+Insert]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Insert, QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self.keyPressEvent(event)

        elif action_text == "Rename channel":
            text, ok = QtWidgets.QInputDialog.getText(
                self,
                "Rename channel",
                "New channel name:",
                text=item.text(item.NameColumn),
            )
            if ok and text.strip():
                text = text.strip()
                item.name = text

        elif action_text == "Edit group":
            if item.pattern:
                mdf = None
                if self.plot:
                    if hasattr(self.plot, "mdf"):
                        mdf = self.plot.mdf

                pattern = dict(item.pattern)
                pattern["ranges"] = copy_ranges(item.ranges)
                dlg = AdvancedSearch(
                    mdf,
                    show_add_window=False,
                    show_apply=True,
                    show_search=False,
                    window_title="Add pattern based group",
                    parent=self,
                    pattern=pattern,
                )
                dlg.setModal(True)
                dlg.exec_()
                pattern = dlg.result

                if pattern:
                    item.pattern = pattern
                    item.name = pattern["name"]
                    item.set_ranges(pattern["ranges"])

                    self.clearSelection()

                    count = item.childCount()
                    for i in range(count):
                        child = item.child(i)
                        child.setSelected(True)

                    event = QtGui.QKeyEvent(
                        QtCore.QEvent.Type.KeyPress,
                        QtCore.Qt.Key.Key_Delete,
                        QtCore.Qt.KeyboardModifier.NoModifier,
                    )
                    self.keyPressEvent(event)

                    self.pattern_group_added.emit(item)
                    self.refresh()
            else:
                text, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Edit channel group name",
                    "New channel group name:",
                    QtWidgets.QLineEdit.EchoMode.Normal,
                    item.name,
                )
                if ok:
                    item.name = text.strip()

        elif action_text == "Search item":
            pattern, ok = QtWidgets.QInputDialog.getText(self, "Search item", "Item name:")
            if ok and pattern:
                original_pattern = pattern
                wildcard = f"{os.urandom(6).hex()}_WILDCARD_{os.urandom(6).hex()}"
                pattern = pattern.replace("*", wildcard)
                pattern = re.escape(pattern)
                pattern = pattern.replace(wildcard, ".*")

                compiled_pattern = re.compile(pattern, flags=re.IGNORECASE)

                start_item = item or self
                iterator = QtWidgets.QTreeWidgetItemIterator(start_item)

                while item := iterator.value():
                    if (
                        item is not start_item
                        and item.type() in (ChannelsTreeItem.Channel, ChannelsTreeItem.Group)
                        and compiled_pattern.search(item.name)
                    ):
                        self.scrollToItem(item)
                        self.setCurrentItem(item)
                        return

                    iterator += 1

                if start_item is self:
                    MessageBox.warning(
                        self,
                        "No matches found",
                        f"No item matches the pattern\n{original_pattern}",
                    )
                else:
                    iterator = QtWidgets.QTreeWidgetItemIterator(self)

                    while item := iterator.value():
                        if item is start_item:
                            break

                        if item.type() in (
                            ChannelsTreeItem.Channel,
                            ChannelsTreeItem.Group,
                        ) and compiled_pattern.search(item.name):
                            self.scrollToItem(item)
                            self.setCurrentItem(item)
                            return

                        iterator += 1

                    MessageBox.warning(
                        self,
                        "No matches found",
                        f"No item matches the pattern\n{original_pattern}",
                    )

        elif action_text == f"Find next {item.name if item else ''}":
            start_item = item
            wildcard = f"{os.urandom(6).hex()}_WILDCARD_{os.urandom(6).hex()}"
            pattern = start_item.name.replace("*", wildcard)
            pattern = re.escape(pattern)
            pattern = pattern.replace(wildcard, ".*")

            compiled_pattern = re.compile(pattern, flags=re.IGNORECASE)

            iterator = QtWidgets.QTreeWidgetItemIterator(start_item)

            iterator += 1

            while item := iterator.value():
                if item.type() in (
                    ChannelsTreeItem.Channel,
                    ChannelsTreeItem.Group,
                ) and compiled_pattern.fullmatch(item.name):
                    self.scrollToItem(item)
                    self.setCurrentItem(item)
                    return

                iterator += 1

            iterator = QtWidgets.QTreeWidgetItemIterator(self)

            while item := iterator.value():
                if item is start_item:
                    break

                if item.type() in (
                    ChannelsTreeItem.Channel,
                    ChannelsTreeItem.Group,
                ) and compiled_pattern.search(item.name):
                    self.scrollToItem(item)
                    self.setCurrentItem(item)
                    return

                iterator += 1

        self.update_channel_groups_count()

    def refresh(self):
        self.updateGeometry()
        self.update_hidden_states()
        self.update_visibility_status()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        if self.updatesEnabled():
            super().resizeEvent(e)
            self.update_visibility_status()

    def set_font_size(self, size):
        if self._dark:
            self._font_size = size
            self.setStyleSheet(self._style.format(font_size=self._font_size, background=self._background))

    def update_channel_groups_count(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while item := iterator.value():
            if item.type() == ChannelsTreeItem.Group:
                item.count = item.childCount()
            iterator += 1

    def update_hidden_states(self):
        hide_missing_channels = self.hide_missing_channels
        hide_disabled_channels = self.hide_disabled_channels
        filter_computed = self.filter_computed_channels

        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while item := iterator.value():
            hidden = False

            if filter_computed:
                hidden = item.filter_computed()
            else:
                if item.type() == ChannelsTreeItem.Channel:
                    if hide_missing_channels and not item.exists:
                        hidden = True
                    if hide_disabled_channels and item.checkState(self.NameColumn) == QtCore.Qt.CheckState.Unchecked:
                        hidden = True
                else:
                    if hide_disabled_channels and item.checkState(self.NameColumn) == QtCore.Qt.CheckState.Unchecked:
                        hidden = True

            item.setHidden(hidden)

            iterator += 1

    def update_visibility_status(self, *args):
        tree_rect = self.viewport().rect()

        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while item := iterator.value():
            try:
                if item.type() == item.Channel:
                    rect = self.visualItemRect(item)
                    item._is_visible = rect.intersects(tree_rect)
                else:
                    item._is_visible = False
            except:
                print(item)

            iterator += 1

        self.visible_items_changed.emit()


class ChannelsTreeItem(QtWidgets.QTreeWidgetItem):
    Group = 2000
    Channel = 2001
    Info = 2002

    NameColumn = 0
    ValueColumn = 1
    UnitColumn = 2
    CommonAxisColumn = 3
    IndividualAxisColumn = 4

    def __init__(
        self,
        type,
        signal=None,
        name="",
        pattern=None,
        parent=None,
        uuid="",
        check=None,
        ranges=None,
        origin_uuid=None,
        background_color=None,
        expanded=False,
    ):
        super().__init__(parent, type)
        self.exists = True
        self.resolved_ranges = None
        self.ranges = []

        # bind cache to instance to avoid memory leaks
        self.get_color_using_ranges = lru_cache(maxsize=1024)(self._get_color_using_ranges)

        self._name = ""
        self._count = 0
        self._background_color = background_color
        self._current_background_color = background_color
        self._current_font_color = background_color

        if type == self.Group:
            self.pattern = None
            self.set_pattern(pattern)
            self._is_visible = True
            self.uuid = uuid
            self.origin_uuid = origin_uuid or os.urandom(6).hex()

            self.count = 0

            self.setFlags(
                QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsAutoTristate
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsDropEnabled
            )

            self.setCheckState(self.NameColumn, QtCore.Qt.CheckState.Unchecked)
            self.name = name.split("\t[")[0]

            # Store Previous State (Disabled/Enabled)
            self._previous_state = self.isDisabled()

        elif type == self.Channel:
            self.signal = signal
            self.details = None

            self._value_prefix = ""
            self._value = "n.a."
            self.uuid = signal.uuid
            self.origin_uuid = signal.origin_uuid
            self.set_ranges(ranges or [])
            self.resolved_ranges = None

            kind = signal.phys_samples.dtype.kind
            # if len(signal.samples) and signal.conversion:
            #     kind = signal.conversion.convert(signal.samples[:1]).dtype.kind
            # else:
            #     kind = signal.samples.dtype.kind

            self.kind = kind

            tooltip = getattr(signal, "tooltip", "") or f"{signal.name}\n{signal.comment}"
            if signal.source:
                details = signal.source.get_details()
            else:
                details = ""
            self.details_text = details or "\tSource not available"
            self.setToolTip(self.NameColumn, tooltip)
            self.setToolTip(self.ValueColumn, "")
            self.setToolTip(self.UnitColumn, "unit")
            self.setToolTip(self.CommonAxisColumn, "common axis")
            self.setToolTip(self.IndividualAxisColumn, "individual axis")

            self.setText(self.UnitColumn, signal.unit)

            if kind in "SUVui" or self.precision == -1:
                self.fmt = "{}"
            else:
                self.fmt = f"{{:.{self.precision}f}}"

            self.entry = signal.group_index, signal.channel_index

            self.setText(self.NameColumn, self.name)
            self.setForeground(self.NameColumn, signal.color)
            self.setForeground(self.ValueColumn, signal.color)
            self.setForeground(self.CommonAxisColumn, signal.color)
            self.setForeground(self.IndividualAxisColumn, signal.color)
            self.setForeground(self.UnitColumn, signal.color)

            self._is_visible = True

            self.setFlags(
                QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
            )

            if check is None:
                self.setCheckState(self.NameColumn, QtCore.Qt.CheckState.Checked)
            else:
                self.setCheckState(self.NameColumn, check)

            self.setCheckState(self.CommonAxisColumn, QtCore.Qt.CheckState.Unchecked)
            self.setCheckState(self.IndividualAxisColumn, QtCore.Qt.CheckState.Unchecked)

            if signal.flags & Signal.Flags.computed:
                font = self.font(0)
                font.setItalic(True)
                for column in (self.NameColumn, self.ValueColumn, self.UnitColumn):
                    self.setFont(column, font)

            if signal.group_index == NOT_FOUND:
                self.does_not_exist()

        elif type == self.Info:
            self.name = name
            self.color = signal.color
            self.uuid = uuid
            self.origin_uuid = origin_uuid

        self.setTextAlignment(self.ValueColumn, QtCore.Qt.AlignmentFlag.AlignRight)

    def __repr__(self):
        t = self.type()
        if t == 2000:
            type = "Group"
        elif t == 2001:
            type = "Channel"
        elif type == 2002:
            type = "Info"
        else:
            type = t

        return f"ChannelTreeItem(type={type}, name={self.name}, uuid={self.uuid}, origin_uuid={self.origin_uuid})"

    @property
    def color(self):
        type = self.type()
        if type == self.Channel:
            return self.signal.color
        else:
            return ""

    @color.setter
    def color(self, value):
        if self.type() == self.Channel:
            value = fn.mkColor(value)
            color_name = value.name()

            self.signal.color = value
            self.signal.pen = fn.mkPen(color_name)
            self.signal.color_name = color_name
            self.setForeground(self.NameColumn, value)
            self.setForeground(self.ValueColumn, value)
            self.setForeground(self.CommonAxisColumn, value)
            self.setForeground(self.IndividualAxisColumn, value)
            self.setForeground(self.UnitColumn, value)

            if self.details is not None:
                self.details.color = value

            tree = self.treeWidget()
            if tree:
                tree.color_changed.emit(self.uuid, value)
        elif self.type() == self.Group:
            count = self.childCount()
            for row in range(count):
                child = self.child(row)
                child.color = value
        elif self.type() == self.Info:
            value = fn.mkColor(value)

            self.setForeground(self.NameColumn, value)
            self.setForeground(self.ValueColumn, value)
            self.setForeground(self.CommonAxisColumn, value)
            self.setForeground(self.IndividualAxisColumn, value)
            self.setForeground(self.UnitColumn, value)

    @property
    def comment(self):
        type = self.type()
        if type == self.Channel:
            return self.signal.comment
        else:
            return ""

    @comment.setter
    def comment(self, value):
        value = value or ""
        type = self.type()
        if type == self.Channel:
            self.signal.comment = value
            self.signal.flags |= Signal.Flags.user_defined_comment

            tooltip = getattr(self.signal, "tooltip", "") or f"{self.signal.name}\n{value}"
            self.setToolTip(self.NameColumn, tooltip)

    def copy(self):
        type = self.type()

        if type == self.Channel:
            return ChannelsTreeItem(
                type,
                signal=self.signal,
                check=self.checkState(self.NameColumn),
            )

        elif type == self.Group:
            return ChannelsTreeItem(
                type,
                name=self.name,
                pattern=self.pattern,
                uuid=self.uuid,
            )
        else:
            return ChannelsTreeItem(
                type,
                name=self.name,
            )

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = value
        if self.type() == self.Group:
            if self.pattern:
                if self._count:
                    self.setText(self.ValueColumn, f"{self._count} matches")
                else:
                    self.setText(self.ValueColumn, "no matches")
            else:
                self.setText(self.ValueColumn, f"{self._count} items")

    def does_not_exist(self, exists=False):
        if exists == self.exists:
            return

        if self.type() == self.Channel:
            if utils.ERROR_ICON is None:
                utils.ERROR_ICON = QtGui.QIcon()
                utils.ERROR_ICON.addPixmap(QtGui.QPixmap(":/error.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

                utils.NO_ICON = QtGui.QIcon()

            if not exists:
                icon = utils.ERROR_ICON
            else:
                icon = utils.NO_ICON
            self.setIcon(self.NameColumn, icon)

        self.exists = exists

    def filter_computed(self):
        if self.type() == self.Channel:
            hide = not (self.signal.flags & Signal.Flags.computed)
        else:
            hide = all(self.child(i).filter_computed() for i in range(self.childCount()))
        return hide

    def get_all_channel_items(self):
        children = []
        count = self.childCount()
        for i in range(count):
            item = self.child(i)

            type = item.type()
            if type == self.Group:
                children.extend(item.get_all_channel_items())
            elif type == self.Channel:
                children.append(item)

        return children

    @property
    def format(self):
        if self.type() == self.Channel:
            return self.signal.format
        else:
            return "phys"

    @format.setter
    def format(self, format):
        if self.type() == self.Channel:
            self.signal.format = format
            self.set_fmt(format)

            self.set_value(update=True)
        elif self.type() == self.Group:
            count = self.childCount()
            for row in range(count):
                child = self.child(row)
                child.set_fmt(format)

    def _get_color_using_ranges(self, value, pen=False):
        return get_color_using_ranges(value, self.get_ranges(), self.signal.color, pen=pen)

    def get_display_properties(self):
        if self.type() == ChannelsTreeItem.Group:
            info = {
                "type": "group",
                "ranges": copy_ranges(self.ranges),
            }

        elif self.type() == ChannelsTreeItem.Channel:
            info = {
                "type": "channel",
                "color": self.color.name(),
                "precision": self.precision,
                "ylink": self.checkState(self.CommonAxisColumn) == QtCore.Qt.CheckState.Checked,
                "individual_axis": self.checkState(self.IndividualAxisColumn) == QtCore.Qt.CheckState.Checked,
                "format": self.format,
                "ranges": copy_ranges(self.ranges),
            }

            if self.signal.flags & Signal.Flags.user_defined_conversion:
                info["conversion"] = to_dict(self.signal.conversion)

            plot = self.treeWidget().plot.plot

            sig, index = plot.signal_by_uuid(self.uuid)

            info["y_range"] = tuple(float(e) for e in sig.y_range)

        for range_info in info["ranges"]:
            range_info["background_color"] = range_info["background_color"].name()
            range_info["font_color"] = range_info["font_color"].name()

        return json.dumps(info)

    def get_ranges(self, tree=None):
        if self.resolved_ranges is None:
            tree = tree or self.treeWidget()
            if tree:
                parent = self.parent()
                if parent is None:
                    self.resolved_ranges = self.ranges
                else:
                    self.resolved_ranges = [*self.ranges, *parent.get_ranges(tree)]
            else:
                return self.ranges
        return self.resolved_ranges

    @property
    def mode(self):
        if self.type() == self.Channel:
            return self.signal.mode
        else:
            return "phys"

    @mode.setter
    def mode(self, mode):
        if self.type() == self.Channel:
            self.signal.mode = mode

            self.set_value(update=True)

    @property
    def name(self):
        type = self.type()
        if type == self.Channel:
            return self.signal.name
        else:
            return self._name

    @name.setter
    def name(self, text):
        text = text or ""
        type = self.type()
        if type == self.Group:
            self._name = text
        elif type == self.Channel:
            if text != self.signal.name:
                if self.signal.original_name is None:
                    self.signal.original_name = self.signal.name
                self.signal.name = text

                if not self.signal.flags & Signal.Flags.computed:
                    self.signal.flags |= Signal.Flags.user_defined_name

        else:
            self._name = text
        self.setText(self.NameColumn, self.name)

        if type == self.Channel:
            tree = self.treeWidget()
            if tree:
                tree.name_changed.emit(self.uuid, text)

    @property
    def precision(self):
        if self.type() == self.Channel:
            return self.signal.precision
        else:
            return 3

    @precision.setter
    def precision(self, precision):
        if self.type() == self.Channel:
            self.signal.precision = precision
            if self.kind == "f":
                if precision >= 0:
                    self.fmt = f"{{:.{self.precision}f}}"
                else:
                    self.fmt = "{}"
            self.set_value(update=True)
        elif self.type() == self.Group:
            count = self.childCount()
            for row in range(count):
                child = self.child(row)
                child.precision = precision

    def reset_resolved_ranges(self):
        self.resolved_ranges = None

        count = self.childCount()
        for row in range(count):
            item = self.child(row)
            item.reset_resolved_ranges()

    def set_conversion(self, conversion):
        if self.type() == self.Channel:
            self.signal.conversion = conversion
            self.signal.flags |= Signal.Flags.user_defined_conversion

            self.signal.text_conversion = None

            if self.signal.conversion:
                samples = self.signal.conversion.convert(self.signal.samples, as_bytes=True)
                if samples.dtype.kind not in "SUV":
                    nans = np.isnan(samples)
                    if np.any(nans):
                        self.signal.raw_samples = self.signal.samples[~nans]
                        self.signal.phys_samples = samples[~nans]
                        self.signal.timestamps = self.signal.timestamps[~nans]
                        self.signal.samples = self.signal.samples[~nans]
                    else:
                        self.signal.raw_samples = self.signal.samples
                        self.signal.phys_samples = samples
                else:
                    self.signal.text_conversion = self.signal.conversion
                    self.signal.phys_samples = self.signal.raw_samples = self.signal.samples

                self.unit = conversion.unit
            else:
                self.signal.phys_samples = self.signal.raw_samples = self.signal.samples

            self.set_value(update=True)

            tree = self.treeWidget()
            if tree:
                tree.conversion_changed.emit(self.uuid, conversion)

        elif self.type() == self.Group:
            count = self.childCount()
            for i in range(count):
                child = self.child(i)
                child.set_conversion(conversion)

    def set_disabled(self, disabled, preserve_subgroup_state=True):
        if self.type() == self.Channel:
            for col in range(self.columnCount()):
                self.setForeground(col, QtCore.Qt.BrushStyle.NoBrush)
                self.setBackground(col, QtCore.Qt.BrushStyle.NoBrush)
                self._current_background_color = None
                self._current_font_color = None

            if self.parent().isDisabled() == disabled:
                self.setDisabled(disabled)
                if self.details is not None:
                    self.details.setDisabled(disabled)

                if disabled:
                    self.signal.enable = False
                else:
                    enable = self.checkState(self.NameColumn) == QtCore.Qt.CheckState.Checked
                    self.signal.enable = enable
                    color = self.signal.color

                    self.setForeground(self.NameColumn, color)
                    self.setForeground(self.ValueColumn, color)
                    self.setForeground(self.CommonAxisColumn, color)
                    self.setForeground(self.IndividualAxisColumn, color)
                    self.setForeground(self.UnitColumn, color)

        elif self.type() == self.Group:
            # If the group is subgroup (has a parent)
            if self.parent() and not self.parent().isDisabled():
                # And the action was triggered on it
                if preserve_subgroup_state:
                    # Save current state
                    self._previous_state = not self.isDisabled()
                # Restore state
                elif not disabled:
                    disabled = self._previous_state

            if disabled:
                self.setIcon(self.NameColumn, QtGui.QIcon(":/erase.png"))
            elif not self.parent() or (self.parent() and not self.parent().isDisabled()):
                self.setIcon(self.NameColumn, QtGui.QIcon(":/open.png"))

            self.setDisabled(disabled)

            count = self.childCount()
            for i in range(count):
                child = self.child(i)
                child.set_disabled(disabled, preserve_subgroup_state=False)

    def set_fmt(self, fmt):
        if self.type() == self.Channel:
            if self.kind in "SUV":
                self.fmt = "{}"
            elif self.kind == "f":
                self.fmt = f"{{:.{self.precision}f}}"
            else:
                if fmt == "hex":
                    self.fmt = "0x{:x}"
                elif fmt == "bin":
                    self.fmt = "0b{:b}"
                elif fmt == "phys":
                    self.fmt = "{}"

    def set_pattern(self, pattern):
        if pattern:
            self.setIcon(self.NameColumn, QtGui.QIcon(":/filter.png"))
            self.pattern = dict(pattern)
            self.pattern["ranges"] = copy_ranges(self.pattern["ranges"])
            for range_info in self.pattern["ranges"]:
                if isinstance(range_info["font_color"], str):
                    range_info["font_color"] = fn.mkColor(range_info["font_color"])
                if isinstance(range_info["background_color"], str):
                    range_info["background_color"] = fn.mkColor(range_info["background_color"])
        else:
            self.setIcon(self.NameColumn, QtGui.QIcon(":/open.png"))
            self.pattern = None

    def set_prefix(self, text=""):
        self._value_prefix = text

    def set_ranges(self, ranges):
        self.get_color_using_ranges.cache_clear()

        if utils.RANGE_INDICATOR_ICON is None:
            utils.RANGE_INDICATOR_ICON = QtGui.QIcon()
            utils.RANGE_INDICATOR_ICON.addPixmap(
                QtGui.QPixmap(":/paint.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off
            )

            utils.NO_ICON = QtGui.QIcon()
            utils.NO_ERROR_ICON = QtGui.QIcon()

        if ranges:
            self.setIcon(self.ValueColumn, utils.RANGE_INDICATOR_ICON)
            self.setToolTip(self.ValueColumn, f"{self.name}\nhas color ranges defined")
        else:
            self.setIcon(self.ValueColumn, utils.NO_ICON)
            self.setToolTip(self.ValueColumn, "")

            if self.type() == self.Channel:
                brush = fn.mkBrush(self._background_color.name())
                self.setBackground(self.NameColumn, brush)
                self.setBackground(self.ValueColumn, brush)
                self.setBackground(self.UnitColumn, brush)
                self.setBackground(self.CommonAxisColumn, brush)
                self.setBackground(self.IndividualAxisColumn, brush)
                self._current_background_color = self._background_color

                brush = fn.mkBrush(self.signal.color_name)
                self.setForeground(self.NameColumn, brush)
                self.setForeground(self.UnitColumn, brush)
                self.setForeground(self.ValueColumn, brush)
                self.setForeground(self.CommonAxisColumn, brush)
                self.setForeground(self.IndividualAxisColumn, brush)
                self._current_font_color = self.signal.color

        self.ranges = []
        for range_info in ranges:
            if isinstance(range_info["font_color"], str):
                range_info["font_color"] = fn.mkColor(range_info["font_color"])
            if isinstance(range_info["background_color"], str):
                range_info["background_color"] = fn.mkColor(range_info["background_color"])
            self.ranges.append(range_info)

        self.reset_resolved_ranges()

    def set_value(self, value=None, update=False, force=False):
        update_text = (value != self._value) or force
        if value is not None:
            if self._value == value and update is False:
                return
            else:
                self._value = value
        else:
            value = self._value

        default_background_color = None
        default_font_color = None

        ranges = self.get_ranges()

        if ranges and not self.isDisabled():
            new_background_color, new_font_color = get_colors_using_ranges(
                value,
                ranges=self.get_ranges(),
                default_background_color=default_background_color,
                default_font_color=default_font_color,
            )

            if new_background_color is None:
                if self._background_color != self._current_background_color:
                    brush = fn.mkBrush(self._background_color.name())
                    self.setBackground(self.NameColumn, brush)
                    self.setBackground(self.ValueColumn, brush)
                    self.setBackground(self.UnitColumn, brush)
                    self.setBackground(self.CommonAxisColumn, brush)
                    self.setBackground(self.IndividualAxisColumn, brush)
                    self._current_background_color = self._background_color
            else:
                if new_background_color != self._current_background_color:
                    brush = fn.mkBrush(new_background_color.name())
                    self.setBackground(self.NameColumn, brush)
                    self.setBackground(self.ValueColumn, brush)
                    self.setBackground(self.UnitColumn, brush)
                    self.setBackground(self.CommonAxisColumn, brush)
                    self.setBackground(self.IndividualAxisColumn, brush)
                    self._current_background_color = new_background_color

            if new_font_color is None:
                if self.signal.color != self._current_font_color:
                    brush = fn.mkBrush(self.signal.color_name)
                    self.setForeground(self.NameColumn, brush)
                    self.setForeground(self.UnitColumn, brush)
                    self.setForeground(self.ValueColumn, brush)
                    self.setForeground(self.CommonAxisColumn, brush)
                    self.setForeground(self.IndividualAxisColumn, brush)
                    self._current_font_color = self.signal.color
            else:
                if new_font_color != self.foreground(0).color():
                    brush = fn.mkBrush(new_font_color.name())
                    self.setForeground(self.NameColumn, brush)
                    self.setForeground(self.ValueColumn, brush)
                    self.setForeground(self.UnitColumn, brush)
                    self.setForeground(self.CommonAxisColumn, brush)
                    self.setForeground(self.IndividualAxisColumn, brush)
                    self._current_font_color = new_font_color

        if update_text:
            if value in ("", "n.a."):
                text = f"{self._value_prefix}{value}"
                self.setText(self.ValueColumn, text)
            else:
                if self.signal.text_conversion and self.mode == "phys":
                    value = self.signal.text_conversion.convert([value], as_bytes=True)[0]
                    if isinstance(value, bytes):
                        try:
                            text = value.decode("utf-8", errors="replace")
                        except:
                            text = value.decode("latin-1", errors="replace")
                        text = f"{self._value_prefix}{text}"
                    else:
                        text = f"{self._value_prefix}{value}"

                else:
                    string = value_as_str(
                        value,
                        self.format,
                        self.signal.plot_samples.dtype,
                        self.precision,
                    )

                    text = f"{self._value_prefix}{string}"

                try:
                    self.setText(self.ValueColumn, text)
                except (ValueError, TypeError):
                    self.setText(self.ValueColumn, f"{self._value_prefix}{value}")

    def show_info(self):
        if self.type() == self.Group:
            ChannnelGroupDialog(self.name, self.pattern, self.get_ranges(), self.treeWidget()).show()

    @property
    def unit(self):
        type = self.type()
        if type == self.Channel:
            return self.signal.unit if self.signal.mode != "raw" else ""
        else:
            return ""

    @unit.setter
    def unit(self, text):
        text = text or ""
        if self.type() == self.Channel and self.signal.unit != text:
            self.signal.unit = text
            self.signal.flags |= Signal.Flags.user_defined_unit
            self.setText(self.UnitColumn, text)
            tree = self.treeWidget()
            if tree:
                tree.unit_changed.emit(self.uuid, text)

    def update_child_values(self, tree=None):
        tree = tree or self.treeWidget()
        count = self.childCount()
        for i in range(count):
            item = self.child(i)
            item.update_child_values(tree)

        if self.type() == self.Channel:
            self.set_value(update=True)


class ChannnelGroupDialog(QtWidgets.QDialog):
    def __init__(self, name, pattern, ranges, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(QtCore.Qt.WindowType.Window)

        layout = QtWidgets.QGridLayout()
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        self.setLayout(layout)

        if pattern:
            self.setWindowTitle(f"<{name}> pattern group details")

            for i, key in enumerate(
                (
                    "name",
                    "pattern",
                    "match_type",
                    "case_sensitive",
                    "filter_type",
                    "filter_value",
                    "raw",
                )
            ):
                widget = QtWidgets.QLabel(str(pattern.get(key, "False")))

                if key == "raw":
                    key = "Use raw values"
                label = QtWidgets.QLabel(key.replace("_", " ").capitalize())
                label.setStyleSheet("color:rgb(97, 190, 226);")

                layout.addWidget(label, i, 0)
                layout.addWidget(widget, i, 1)
        else:
            self.setWindowTitle(f"<{name}> group details")

        # self.setStyleSheet('font: 8pt "Consolas";}')

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        self.setWindowIcon(icon)
        self.setMinimumWidth(500)
        self.adjustSize()

        # screen = QtWidgets.QApplication.desktop().screenGeometry()
        # self.move((screen.width() - 1200) // 2, (screen.height() - 600) // 2)


if __name__ == "__main__":
    pass

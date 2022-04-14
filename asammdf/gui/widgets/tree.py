# -*- coding: utf-8 -*-
from collections import defaultdict
from datetime import date, datetime
from functools import lru_cache
import json
import os
from struct import pack
from traceback import format_exc

from PySide6 import QtCore, QtGui, QtWidgets

from ..dialogs.advanced_search import AdvancedSearch
from ..utils import copy_ranges, get_color_using_ranges, get_colors_using_ranges, extract_mime_names
from .. import utils
from .channel_display import ChannelDisplay
from .channel_group_display import ChannelGroupDisplay
from .tree_item import TreeItem


NOT_FOUND = 0xFFFFFFFF


def substitude_mime_uuids(mime, uuid=None, force=False):
    if not mime:
        return mime

    new_mime = []

    for item in mime:
        if item["type"] == "channel":
            if force or item["origin_uuid"] is None:
                item["origin_uuid"] = uuid
            new_mime.append(item)
        else:
            item["channels"] = substitude_mime_uuids(
                item["channels"], uuid, force=force
            )
            if force or item["origin_uuid"] is None:
                item["origin_uuid"] = uuid
            new_mime.append(item)
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
        if ch.added == True:
            continue

        entry = ch.entry

        child = TreeItem(entry, ch.name, origin_uuid=origin_uuid)
        child.setText(0, ch.name)

        dep = channel_dependencies[entry[1]]
        if version >= "4.00":
            if dep and isinstance(dep[0], tuple):
                child.setFlags(
                    child.flags()
                    | QtCore.Qt.ItemIsAutoTristate
                    | QtCore.Qt.ItemIsUserCheckable
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
            child.setCheckState(0, QtCore.Qt.Checked)
        else:
            child.setCheckState(0, QtCore.Qt.Unchecked)

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
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setUniformRowHeights(True)

        self.mode = "Natural sort"

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            elif len(selected_items) == 1:
                item = selected_items[0]
                checked = item.checkState(0)
                if checked == QtCore.Qt.Checked:
                    item.setCheckState(0, QtCore.Qt.Unchecked)
                else:
                    item.setCheckState(0, QtCore.Qt.Checked)
            else:
                if any(
                    item.checkState(0) == QtCore.Qt.Unchecked for item in selected_items
                ):
                    checked = QtCore.Qt.Checked
                else:
                    checked = QtCore.Qt.Unchecked
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

        data = json.dumps(
            sorted(
                data, key=lambda x: (x["name"], x["group_index"], x["channel_index"])
            )
        ).encode("utf-8")

        mimeData.setData("application/octet-stream-asammdf", QtCore.QByteArray(data))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec_(QtCore.Qt.MoveAction)


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

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setUniformRowHeights(True)

        self.mode = "Natural sort"


class SearchTreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        self.setUniformRowHeights(True)

        self.can_delete_items = False

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key_Delete and self.can_delete_items:
            selected_items = self.selectedItems()

            root = self.invisibleRootItem()
            for item in selected_items:
                (item.parent() or root).removeChild(item)


class ChannelsTreeWidget(QtWidgets.QTreeWidget):
    itemsDeleted = QtCore.Signal(list)
    set_time_offset = QtCore.Signal(list)
    items_rearranged = QtCore.Signal(list)
    add_channels_request = QtCore.Signal(list)
    show_properties = QtCore.Signal(object)
    insert_computation = QtCore.Signal(str)
    pattern_group_added = QtCore.Signal(object)
    compute_fft_request = QtCore.Signal(str)
    color_changed = QtCore.Signal(str, str)
    unit_changed = QtCore.Signal(str, str)
    name_changed = QtCore.Signal(str, str)

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
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.setUniformRowHeights(True)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)
        self.details_enabled = False
        self.hide_missing_channels = hide_missing_channels
        self.hide_disabled_channels = hide_disabled_channels
        self.can_delete_items = True

        self.setHeaderHidden(False)
        self.setColumnCount(4)
        self.setHeaderLabels(["Name", "Value", "CA", "IA"])
        self.setDragEnabled(True)

        self.setMinimumWidth(5)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.header().resizeSection(2, 10)
        self.header().resizeSection(3, 10)
        self.header().setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        self.header().setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)

        self.header().setStretchLastSection(False)

        # self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        # self.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        # self.itemSelectionChanged.connect(self.item_selection_changed)

        self.itemCollapsed.connect(self.update_visibility_status)
        self.itemExpanded.connect(self.update_visibility_status)
        self.verticalScrollBar().valueChanged.connect(self.update_visibility_status)
        self.itemsDeleted.connect(self.update_visibility_status)

    def item_selection_changed(self):
        selection = list(self.selectedItems())

        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while True:
            item = iterator.value()
            if item is None:
                break
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

        if key == QtCore.Qt.Key_Delete and self.can_delete_items:
            selected_items = self.selectedItems()
            deleted = get_data(self.plot, selected_items, uuids_only=True)

            self.plot.ignore_selection_change = True

            root = self.invisibleRootItem()
            for item in selected_items:
                item_widget = self.itemWidget(item, 1)
                if hasattr(item_widget, "disconnect_slots"):
                    item_widget.disconnect_slots()
                (item.parent() or root).removeChild(item)
                item.widget = None
                item_widget.item = None

            self.refresh()

            self.plot.ignore_selection_change = False

            if deleted:
                self.itemsDeleted.emit(list(deleted))
                self.update_channel_groups_count()

        elif key == QtCore.Qt.Key_Insert and modifiers == QtCore.Qt.ControlModifier:

            dlg = AdvancedSearch(
                None,
                show_add_window=False,
                show_apply=True,
                show_search=False,
                window_title="Add pattern based group",
                parent=self,
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
                            if current_parent.type() ==  ChannelsTreeItem.Group:
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

        elif key == QtCore.Qt.Key_Insert and modifiers == QtCore.Qt.ShiftModifier:
            text, ok = QtWidgets.QInputDialog.getText(
                self, "Channel group name", "New channel group name:"
            )
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

        elif key == QtCore.Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            elif len(selected_items) == 1:
                item = selected_items[0]
                checked = item.checkState(0)
                if checked == QtCore.Qt.Checked:
                    item.setCheckState(0, QtCore.Qt.Unchecked)
                    if self.hide_disabled_channels:
                        item.setHidden(True)
                else:
                    item.setCheckState(0, QtCore.Qt.Checked)
            else:
                if any(
                    item.checkState(0) == QtCore.Qt.Unchecked for item in selected_items
                ):
                    checked = QtCore.Qt.Checked
                else:
                    checked = QtCore.Qt.Unchecked
                for item in selected_items:
                    item.setCheckState(0, checked)
                    if self.hide_disabled_channels and checked == QtCore.Qt.Unchecked:
                        item.setHidden(True)

        elif modifiers == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_C:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            self.itemWidget(selected_items[0], 1).keyPressEvent(event)

        elif modifiers == (
            QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier
        ) and key in (QtCore.Qt.Key_C, QtCore.Qt.Key_P):
            selected_items = self.selectedItems()
            if not selected_items:
                return
            self.itemWidget(selected_items[0], 1).keyPressEvent(event)

        elif modifiers == QtCore.Qt.ShiftModifier and key in (
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
        ):
            event.ignore()
        else:
            super().keyPressEvent(event)

    def startDrag(self, supportedActions):

        selected_items = validate_drag_items(
            self.invisibleRootItem(), self.selectedItems(), []
        )

        mimeData = QtCore.QMimeData()

        data = get_data(self.plot, selected_items, uuids_only=False)
        data = json.dumps(data).encode("utf-8")

        mimeData.setData("application/octet-stream-asammdf", QtCore.QByteArray(data))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec_(QtCore.Qt.MoveAction)

    def dragEnterEvent(self, e):
        e.accept()

    def dragLeaveEvent(self, e):
        e.accept()

    def dragMoveEvent(self, e):
        e.accept()

    def dropEvent(self, e):

        if e.source() is self:

            drop_item = self.itemAt(e.pos())

            # cannot move inside pattern channel group
            current_item = drop_item
            while current_item:
                if current_item.type() == ChannelsTreeItem.Group and current_item.pattern:
                    e.ignore()
                    return
                current_item = current_item.parent()

            uuids = get_data(self.plot, self.selectedItems(), uuids_only=True)

            super().dropEvent(e)
            #
            # selected_items = validate_drag_items(
            #     self.invisibleRootItem(), self.selectedItems(), []
            # )
            #
            # if drop_item is not None:
            #     selected_items = [
            #         item
            #         for item in selected_items
            #         if valid_drop_target(target=drop_item, item=item)
            #     ]
            #
            # uuids = get_data(self.plot, selected_items, uuids_only=True)
            #
            # if drop_item is None:
            #     add_new_items(
            #         self,
            #         self.invisibleRootItem(),
            #         selected_items,
            #         pos=None,
            #     )
            #
            # elif isinstance(drop_item, ChannelsTreeItem):
            #     parent = drop_item.parent()
            #
            #     if not parent:
            #         index = initial = self.indexOfTopLevelItem(self.itemAt(e.pos()))
            #         index_func = self.indexOfTopLevelItem
            #         root = self.invisibleRootItem()
            #     else:
            #         index = initial = parent.indexOfChild(drop_item)
            #         index_func = parent.indexOfChild
            #         root = parent
            #
            #     for it in selected_items:
            #
            #         idx = index_func(it)
            #         if 0 <= idx < initial:
            #             index -= 1
            #
            #     add_new_items(
            #         self,
            #         root,
            #         selected_items,
            #         pos=index,
            #     )
            #
            # elif isinstance(drop_item, ChannelsGroupTreeItem):
            #     add_new_items(
            #         self,
            #         drop_item,
            #         selected_items,
            #         pos=0,
            #     )
            #
            # root = self.invisibleRootItem()
            # for item in selected_items:
            #     item_widget = self.itemWidget(item, 1)
            #     item.widget = None
            #     item_widget.item = None
            #     if hasattr(item_widget, "disconnect_slots"):
            #         item_widget.disconnect_slots()
            #     (item.parent() or root).removeChild(item)

            self.update_channel_groups_count()
            self.items_rearranged.emit(list(uuids))

        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
            else:
                super().dropEvent(e)

    def open_menu(self, position):

        item = self.itemAt(position)

        count = 0
        enabled = 0
        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while iterator.value():
            cur_item = iterator.value()
            if cur_item.type() == ChannelsTreeItem.Channel:
                count += 1
                if cur_item.checkState(0) == QtCore.Qt.Checked:
                    enabled += 1
            iterator += 1

        menu = QtWidgets.QMenu()
        menu.addAction(self.tr(f"{count} items in the list, {enabled} enabled"))
        menu.addSeparator()

        menu.addAction(self.tr(f"Add channel group [Shift+Insert]"))
        menu.addAction(self.tr(f"Add pattern based channel group [Ctrl+Insert]"))
        menu.addSeparator()

        if item and item.type() == ChannelsTreeItem.Channel:
            menu.addAction(self.tr("Copy name (Ctrl+C)"))
            menu.addAction(self.tr("Copy display properties [Ctrl+Shift+C]"))
            menu.addAction(self.tr("Paste display properties [Ctrl+Shift+P]"))

        menu.addAction(self.tr("Copy channel structure"))
        menu.addAction(self.tr("Paste channel structure"))

        if item and item.type() == ChannelsTreeItem.Channel:
            menu.addAction(self.tr("Rename channel"))
            menu.addSeparator()
        elif item and item.type() == ChannelsTreeItem.Group:
            menu.addAction(self.tr("Rename group"))
            menu.addSeparator()

        menu.addAction(self.tr("Enable all"))
        menu.addAction(self.tr("Disable all"))
        if item:
            menu.addAction(self.tr("Enable all but this"))
        menu.addSeparator()
        if self.hide_disabled_channels:
            show_disabled_channels = "Show disabled items"
        else:
            show_disabled_channels = "Hide disabled items"
        menu.addAction(self.tr(show_disabled_channels))
        if self.hide_missing_channels:
            show_missing_channels = "Show missing items"
        else:
            show_missing_channels = "Hide missing items"
        menu.addAction(self.tr(show_missing_channels))
        menu.addSeparator()

        menu.addAction(self.tr("Edit Y axis scaling [Ctrl+G]"))
        if item and item.type() == ChannelsTreeItem.Channel:
            menu.addAction(self.tr("Add to common Y axis"))
            menu.addAction(self.tr("Remove from common Y axis"))
            menu.addSeparator()
            menu.addAction(self.tr("Set unit"))

        menu.addAction(self.tr("Set precision"))

        if item and item.type() == ChannelsTreeItem.Channel:
            menu.addSeparator()
            menu.addAction(self.tr("Relative time base shift"))
            menu.addAction(self.tr("Set time base start offset"))
            menu.addSeparator()
            menu.addAction(self.tr("Insert computation using this channel"))
            menu.addAction(self.tr("Compute FFT"))
            menu.addSeparator()
        if item:
            menu.addAction(self.tr("Delete [Del]"))
            menu.addSeparator()
        menu.addAction(self.tr("Toggle details"))
        if item and item.type() == ChannelsTreeItem.Channel:
            menu.addAction(self.tr("File/Computation properties"))
        elif item and item.type() == ChannelsTreeItem.Group:
            if item.pattern:
                menu.addAction(self.tr("Edit pattern"))
            menu.addAction(self.tr("Group properties"))

        action = menu.exec_(self.viewport().mapToGlobal(position))

        if action is None:
            return

        if action.text() == "Copy name [Ctrl+C]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_C, QtCore.Qt.ControlModifier
            )
            self.keyPressEvent(event)

        elif action.text() == "Copy channel structure":
            selected_items = validate_drag_items(
                self.invisibleRootItem(), self.selectedItems(), []
            )
            data = get_data(self.plot, selected_items, uuids_only=False)
            data = substitude_mime_uuids(data, None, force=True)
            QtWidgets.QApplication.instance().clipboard().setText(json.dumps(data))

        elif action.text() == "Paste channel structure":
            try:
                data = QtWidgets.QApplication.instance().clipboard().text()
                data = json.loads(data)
                self.add_channels_request.emit(data)
            except:
                print(format_exc())
                pass

        elif action.text() == "Copy display properties [Ctrl+Shift+C]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress,
                QtCore.Qt.Key_C,
                QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier,
            )
            self.keyPressEvent(event)

        elif action.text() == "Paste display properties [Ctrl+Shift+P]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress,
                QtCore.Qt.Key_P,
                QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier,
            )
            self.keyPressEvent(event)

        elif action.text() == "Enable all":
            count = self.topLevelItemCount()
            for i in range(count):
                item = self.topLevelItem(i)
                item.setCheckState(QtCore.Qt.Checked)

        elif action.text() == "Disable all":
            count = self.topLevelItemCount()
            for i in range(count):
                item = self.topLevelItem(i)
                item.setCheckState(QtCore.Qt.Unchecked)

        elif action.text() == "Enable all but this":
            selected_items = self.selectedItems()

            count = self.topLevelItemCount()
            for i in range(count):
                item = self.topLevelItem(i)
                item.setCheckState(QtCore.Qt.Unchecked)

            for item in selected_items:
                item.setCheckState(QtCore.Qt.Checked)

        elif action.text() == show_disabled_channels:
            if self.hide_missing_channels:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while True:
                    item = iterator.value()
                    if not item:
                        break

                    if (
                        item.type() == ChannelsTreeItem.Channel
                        and self.hide_missing_channels
                        and not item.exists
                    ):
                        pass
                    elif item.checkState(0) == QtCore.Qt.Unchecked:
                        item.setHidden(False)
                    iterator += 1
            else:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while True:
                    item = iterator.value()
                    if not item:
                        break

                    if item.type() == ChannelsTreeItem.Channel and item.checkState(0) == QtCore.Qt.Unchecked:
                        item.setHidden(True)

                    iterator += 1

            self.hide_disabled_channels = not self.hide_disabled_channels

        elif action.text() == show_missing_channels:
            if self.hide_missing_channels:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while True:
                    item = iterator.value()
                    if not item:
                        break

                    if (
                        item.type() == ChannelsTreeItem.Channel
                        and self.hide_disabled_channels
                        and item.checkState(0) == QtCore.Qt.Unchecked
                    ):
                        pass
                    else:
                        item.setHidden(False)
                    iterator += 1
            else:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while True:
                    item = iterator.value()
                    if not item:
                        break

                    if item.type() == ChannelsTreeItem.Channel and not item.exists:
                        item.setHidden(True)

                    iterator += 1

            self.hide_missing_channels = not self.hide_missing_channels

        elif action.text() == "Add to common Y axis":
            selected_items = self.selectedItems()

            for item in selected_items:
                if item.type() == ChannelsTreeItem.Channel:
                    item.setCheckState(2, QtCore.Qt.Checked)

        elif action.text() == "Edit Y axis scaling [Ctrl+G]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress,
                QtCore.Qt.Key_G,
                QtCore.Qt.ControlModifier,
            )
            self.plot.keyPressEvent(event)

        elif action.text() == "Remove from common Y axis":
            selected_items = self.selectedItems()

            for item in selected_items:
                if item.type() == ChannelsTreeItem.Channel:
                    item.setCheckState(2, QtCore.Qt.Unchecked)

        elif action.text() == "Set unit":
            selected_items = self.selectedItems()

            unit, ok = QtWidgets.QInputDialog.getText(None, "Set new unit", "Unit:")

            if ok:
                for item in selected_items:
                    if item.type() == ChannelsTreeItem.Channel:
                        item.set_unit(unit)
                        item.update_information()

        elif action.text() == "Set precision":
            selected_items = self.selectedItems()

            precision, ok = QtWidgets.QInputDialog.getInt(
                self, "Set new precision (float decimals)", "Precision:", 3, -1, 15, 1
            )

            if ok:

                for item in selected_items:
                    if item.type() == ChannelsTreeItem.Channel:
                        item.set_precision(precision)
                        item.update_information()
                    elif item.type() == ChannelsTreeItem.Group:
                        for channel_item in item.get_all_channel_items():
                            channel_item.set_precision(precision)
                            channel_item.update_information()

        elif action.text() in (
            "Relative time base shift",
            "Set time base start offset",
        ):
            selected_items = self.selectedItems()
            if selected_items:

                if action.text() == "Relative time base shift":
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

                    self.set_time_offset.emit([absolute, offset] + uuids)

        elif action.text() == "Delete [Del]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_Delete, QtCore.Qt.NoModifier
            )
            self.keyPressEvent(event)

        elif action.text() == "Toggle details":
            self.details_enabled = not self.details_enabled

            iterator = QtWidgets.QTreeWidgetItemIterator(self)
            while iterator.value():
                item = iterator.value()
                if item.type() == ChannelsTreeItem.Channel:
                    widget = self.itemWidget(item, 1)
                    widget.details.setVisible(self.details_enabled)
                    widget.update()
                    item.setSizeHint(1, widget.sizeHint())
                    widget.repaint()
                iterator += 1

            sizes = self.parent().parent().sizes()
            if self.details_enabled:
                sizes[0] += 1
                sizes[1] -= 1
            else:
                sizes[0] -= 1
                sizes[1] += 1
            self.parent().parent().setSizes(sizes)

        elif action.text() in ("File/Computation properties", "Group properties"):
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                if item.type() == ChannelsTreeItem.Channel:
                    self.show_properties.emit(item.uuid)
                elif item.type() == ChannelsTreeItem.Group:
                    item.show_info()

        elif action.text() == "Insert computation using this channel":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                if item.type() == ChannelsTreeItem.Channel:
                    self.insert_computation.emit(item.name)

        elif action.text() == "Compute FFT":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                if item.type() == ChannelsTreeItem.Channel:
                    self.compute_fft_request.emit(item.uuid)

        elif action.text() == "Add channel group [Shift+Insert]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_Insert, QtCore.Qt.ShiftModifier
            )
            self.keyPressEvent(event)

        elif action.text() == "Add pattern based channel group [Ctrl+Insert]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_Insert, QtCore.Qt.ControlModifier
            )
            self.keyPressEvent(event)

        elif action.text() == "Rename group":
            text, ok = QtWidgets.QInputDialog.getText(
                self, "Rename group", "New channel group name:"
            )
            if ok and text.strip():
                text = text.strip()
                item.name = text

        elif action.text() == "Rename channel":
            text, ok = QtWidgets.QInputDialog.getText(
                self, "Rename channel", "New channel name:"
            )
            if ok and text.strip():
                text = text.strip()
                item.name = text

        elif action.text() == "Edit pattern":
            pattern = dict(item.pattern)
            pattern["ranges"] = copy_ranges(item.ranges)
            dlg = AdvancedSearch(
                None,
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
                item.set_ranges(pattern["ranges"])

                self.clearSelection()

                count = item.childCount()
                for i in range(count):
                    child = item.child(i)
                    child.setSelected(True)

                event = QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress, QtCore.Qt.Key_Delete, QtCore.Qt.NoModifier
                )
                self.keyPressEvent(event)

                self.pattern_group_added.emit(item)
                self.refresh()

        self.update_channel_groups_count()

    def update_channel_groups_count(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while True:
            item = iterator.value()
            if item is None:
                break
            if item.type() == ChannelsTreeItem.Group:
                item.count = item.childCount()
            iterator += 1

    def update_hidden_states(self):
        hide_missing_channels = self.hide_missing_channels
        hide_disabled_channels = self.hide_disabled_channels

        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while True:
            item = iterator.value()
            if item is None:
                break

            hidden = False

            if item.type() == ChannelsTreeItem.Channel:
                if hide_missing_channels and not item.exists:
                    hidden = True
                if (
                    hide_disabled_channels
                    and item.checkState(0) == QtCore.Qt.Unchecked
                ):
                    hidden = True
            else:
                if (
                    hide_disabled_channels
                    and item.checkState(0) == QtCore.Qt.Unchecked
                ):
                    hidden = True

            item.setHidden(hidden)

            iterator += 1

    def refresh(self):
        self.updateGeometry()
        self.update_visibility_status()

    def update_visibility_status(self, *args):

        tree_rect = self.viewport().rect()

        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while True:
            item = iterator.value()
            if item is None:
                break
            rect = self.visualItemRect(item)
            item._is_visible = rect.intersects(tree_rect)

            iterator += 1

    def is_item_visible(self, item):
        return item._is_visible

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        self.update_visibility_status()


class ChannelsTreeItem(QtWidgets.QTreeWidgetItem):

    Group = 2000
    Channel = 2001
    Info = 2002

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
    ):
        super().__init__(parent, type)
        self.exists = True
        self.resolved_ranges = None
        self.ranges = []

        self._name = ""

        if type == self.Group:

            self.pattern = None
            self.set_pattern(pattern)
            self._is_visible = True
            self.uuid = uuid
            self.origin_uuid = origin_uuid

            self.count = 0

            self.setFlags(
                QtCore.Qt.ItemIsUserCheckable
                | QtCore.Qt.ItemIsEnabled
                | QtCore.Qt.ItemIsAutoTristate
                | QtCore.Qt.ItemIsDragEnabled
            )

            self.setCheckState(0, QtCore.Qt.Checked)
            self.name = name.split("\t[")[0]

        elif type == self.Channel:
            self.signal = signal

            self._value_prefix = ""
            self._value = "n.a."
            self._precision = -1
            self.uuid = signal.uuid
            self.origin_uuid = signal.origin_uuid
            self.set_ranges(ranges or [])
            self.resolved_ranges = None

            if len(signal.samples) and signal.conversion:
                kind = signal.conversion.convert(signal.samples[:1]).dtype.kind
            else:
                kind = signal.samples.dtype.kind

            self.kind = kind

            tooltip = getattr(signal, "tooltip", "") or f"{signal.name}\n{signal.comment}"
            if signal.source:
                details = signal.source.get_details()
            else:
                details = ""
            self.details = details or "\tSource not available"
            self.setToolTip(0, tooltip)
            self.setToolTip(1, tooltip)
            self.setToolTip(2, tooltip)
            self.setToolTip(3, tooltip)

            if kind in "SUVui" or self._precision == -1:
                self.fmt = "{}"
            else:
                self.fmt = f"{{:.{self._precision}f}}"

            # if sig.computed:
            #     font = QtGui.QFont()
            #     font.setItalic(True)
            #     it.name.setFont(font)

            self.entry = signal.group_index, signal.channel_index

            self.setText(0, self.name)
            self.setForeground(0, signal.pen.color())
            self.setForeground(1, signal.pen.color())

            self._is_visible = True

            self.setFlags(
                self.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDragEnabled
            )

            if check is None:
                self.setCheckState(0, QtCore.Qt.Checked)
            else:
                self.setCheckState(0, check)

            self.setCheckState(2, QtCore.Qt.Unchecked)
            self.setCheckState(3, QtCore.Qt.Unchecked)

            if signal.group_index == NOT_FOUND:
                self.does_not_exist()

        elif type == self.Info:
            self.name = name
            self.uuid = uuid
            self.origin_uuid = origin_uuid

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
            self.signal.color = value
            self.setForeground(0, value)
            self.setForeground(1, value)
            tree = self.treeWidget()
            if tree:
                tree.color_changed.emit(self.uuid, value)

    def copy(self):
        type = self.type()

        if type == self.Channel:
            return ChannelsTreeItem(
                type,
                signal=self.signal,
                check=self.checkState(0),
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

    def does_not_exist(self, exists=False):
        if exists == self.exists:
            return

        if self.type() == self.Channel:
            if utils.ERROR_ICON is None:
                utils.ERROR_ICON = QtGui.QIcon()
                utils.ERROR_ICON.addPixmap(
                    QtGui.QPixmap(":/error.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                )

                utils.NO_ICON = QtGui.QIcon()

            if not exists:
                icon = utils.ERROR_ICON
            else:
                icon = utils.NO_ICON
            self.setIcon(0, icon)

        self.exists = exists

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

    @lru_cache(maxsize=1024)
    def get_color_using_ranges(self, value, pen=False):
        return get_color_using_ranges(
            value, self.get_ranges(), self._font_color, pen=pen
        )

    def get_ranges(self, tree=None):
        tree = tree or self.treeWidget()
        if tree:
            parent = self.parent()
            if parent is None:
                return self.ranges
            else:
                return [*self.ranges, *parent.get_ranges(tree)]
        else:
            return []

    @property
    def name(self):
        type = self.type()
        if type == self.Group:
            if self.pattern:
                if self.count:
                    return f"{self._name} [{self.count} matches]"
                else:
                    return f"{self._name} [no matches]"
            else:
                return f"{self._name} [{self.count} items]"
        elif type == self.Channel:
            return f"{self.signal.name} ({self.signal.unit})"
        else:
            return self._name

    @name.setter
    def name(self, text):
        type = self.type()
        if type == self.Group:
            self._name = text
        elif type == self.Channel:
            self.signal.name = text
        else:
            self._name = text
        self.setText(0, self.name)

        tree = self.treeWidget()
        if tree:
            tree.name_changed.emit(self.uuid, text)

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        self._precision = precision
        if self.kind == "f" and precision >= 0:
            self.fmt = f"{{:.{self._precision}f}}"
        else:
            self.fmt = "{}"

    def reset_resolved_ranges(self):
        self.resolved_ranges = None

        count = self.childCount()
        for row in range(count):
            item = self.child(row)
            item.reset_resolved_ranges()

    def set_fmt(self, fmt):
        if self.kind in "SUV":
            self.fmt = "{}"
        elif self.kind == "f":
            self.fmt = f"{{:.{self._precision}f}}"
        else:
            if fmt == "hex":
                self.fmt = "0x{:X}"
            elif fmt == "bin":
                self.fmt = "0b{:b}"
            elif fmt == "phys":
                self.fmt = "{}"

    def set_pattern(self, pattern):
        if pattern:
            self.setIcon(0, QtGui.QIcon(":/filter.png"))
            self.pattern = dict(pattern)
            self.pattern["ranges"] = copy_ranges(self.pattern["ranges"])
            for range_info in self.pattern["ranges"]:
                if isinstance(range_info["font_color"], str):
                    range_info["font_color"] = QtGui.QColor(range_info["font_color"])
                if isinstance(range_info["background_color"], str):
                    range_info["background_color"] = QtGui.QColor(
                        range_info["background_color"]
                    )
        else:
            self.setIcon(0, QtGui.QIcon(":/open.png"))
            self.pattern = None

    def set_prefix(self, text=""):
        self._value_prefix = text

    def set_ranges(self, ranges):
        self.get_color_using_ranges.cache_clear()

        if utils.RANGE_INDICATOR_ICON is None:
            utils.RANGE_INDICATOR_ICON = QtGui.QIcon()
            utils.RANGE_INDICATOR_ICON.addPixmap(
                QtGui.QPixmap(":/paint.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
            )

            utils.NO_ICON = QtGui.QIcon()

        if ranges:
            self.setIcon(1, utils.RANGE_INDICATOR_ICON)
        else:
            self.setIcon(1, utils.NO_ICON)

        self.ranges = []
        for range_info in ranges:
            if isinstance(range_info["font_color"], str):
                range_info["font_color"] = QtGui.QColor(range_info["font_color"])
            if isinstance(range_info["background_color"], str):
                range_info["background_color"] = QtGui.QColor(
                    range_info["background_color"]
                )
            self.ranges.append(range_info)

        self.resolved_ranges = None

    def set_value(self, value=None, update=False, force=False):
        update_text = value != self._value
        if value is not None:
            if self._value == value and update is False:
                return
            else:
                self._value = value
        else:
            value = self._value

        default_background_color = None
        default_font_color = None

        new_background_color, new_font_color = get_colors_using_ranges(
            value,
            ranges=self.get_ranges(),
            default_background_color=default_background_color,
            default_font_color=default_font_color,
        )

        if new_background_color is None:
            self.setData(0, QtCore.Qt.BackgroundRole, None)
            self.setData(1, QtCore.Qt.BackgroundRole, None)
            self.setData(2, QtCore.Qt.BackgroundRole, None)
            self.setData(3, QtCore.Qt.BackgroundRole, None)
        else:
            self.setBackground(0, new_background_color)
            self.setBackground(1, new_background_color)
            self.setBackground(2, new_background_color)
            self.setBackground(3, new_background_color)

        if new_font_color is None:
            self.setForeground(0, self.signal.color)
            self.setForeground(1, self.signal.color)
        else:
            self.setForeground(0, new_font_color)
            self.setForeground(1, new_font_color)

        if update_text:

            if value in ("", "n.a."):
                text = f"{self._value_prefix}{value}"
                self.setText(1, text)
            else:
                text = f"{self._value_prefix}{self.fmt}".format(value)

                try:
                    self.setText(1, text)
                except (ValueError, TypeError):
                    self.setText(1, f"{self._value_prefix}{value}")

    def show_info(self):
        if self.type() == self.Group:
            ChannnelGroupDialog(self.name, self.pattern, self.get_ranges(), self.treeWidget()).show()

    @property
    def unit(self):
        type = self.type()
        if type == self.Channel:
            return self.signal.unit
        else:
            return ""

    @unit.setter
    def unit(self, text):
        if self.type() == self.Channel:
            self.signal.unit = text
            self.setText(0, self.name)
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

        self.setWindowFlags(QtCore.Qt.Window)

        layout = QtWidgets.QGridLayout()
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        self.setLayout(layout)

        if pattern:
            self.setWindowTitle(f"<{name}> pattern group details")

            for i, key in enumerate(
                ("name", "pattern", "match_type", "filter_type", "filter_value", "raw")
            ):
                widget = QtWidgets.QLabel(str(pattern[key]))

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
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        self.setWindowIcon(icon)
        self.setMinimumWidth(500)
        self.adjustSize()

        # screen = QtWidgets.QApplication.desktop().screenGeometry()
        # self.move((screen.width() - 1200) // 2, (screen.height() - 600) // 2)


if __name__ == "__main__":
    pass

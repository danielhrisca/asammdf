# -*- coding: utf-8 -*-

from collections import defaultdict
from datetime import date, datetime
import json
from struct import pack

from PyQt5 import QtCore, QtGui, QtWidgets

from ..dialogs.advanced_search import AdvancedSearch
from ..utils import copy_ranges, extract_mime_names
from .channel_display import ChannelDisplay
from .channel_group_display import ChannelGroupDisplay
from .tree_item import TreeItem


def add_children(
    widget,
    channels,
    channel_dependencies,
    signals,
    entries=None,
    mdf_uuid=None,
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

        child = TreeItem(entry, ch.name, mdf_uuid=mdf_uuid)
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
                    mdf_uuid=mdf_uuid,
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
        new_widget.item = new_item

        if pos is None:
            root.addChild(new_item)
        else:
            root.insertChild(pos, new_item)
            pos += 1

        tree.setItemWidget(new_item, 1, new_widget)

        if isinstance(item, ChannelsGroupTreeItem):
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
                elif isinstance(parent, ChannelsGroupTreeItem):
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


def get_data(items, uuids_only=False):
    data = []

    if items:
        tree = items[0].treeWidget()

    for item in items:
        if isinstance(item, ChannelsGroupTreeItem):
            children = [item.child(i) for i in range(item.childCount())]
            if uuids_only:
                data.extend(get_data(children, uuids_only))
            else:
                if item.pattern:
                    data.append(
                        (
                            item.name,
                            item.pattern,
                            [],
                            None,
                            "group",
                            [],
                        )
                    )
                else:
                    data.append(
                        (
                            item.name,
                            None,
                            get_data(children, uuids_only),
                            None,
                            "group",
                            [],
                        )
                    )

        else:
            if uuids_only:
                data.append(tree.itemWidget(item, 1).uuid)
            else:
                widget = item.treeWidget().itemWidget(item, 1)
                if item.entry == (-1, -1):
                    info = {
                        "name": item.name,
                        "computation": item.computation,
                        "computed": True,
                        "unit": widget._unit,
                        "color": widget.color,
                    }
                else:
                    info = item.name

                ranges = copy_ranges(widget.ranges)

                for range_info in ranges:
                    range_info["background_color"] = range_info[
                        "background_color"
                    ].name()
                    range_info["font_color"] = range_info["font_color"].name()

                data.append(
                    (
                        info,
                        *item.entry,
                        item.mdf_uuid,
                        "channel",
                        ranges,
                    )
                )
    return data


class TreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

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
                                (
                                    child.name,
                                    *child.entry,
                                    child.mdf_uuid,
                                    "channel",
                                    [],
                                )
                            )
            else:
                if item.entry[1] != 0xFFFFFFFFFFFFFFFF:
                    data.append(
                        (
                            item.name,
                            *item.entry,
                            item.mdf_uuid,
                            "channel",
                            [],
                        )
                    )

            return data

        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = []
        for item in selected_items:
            data.extend(get_data(item))

        data = json.dumps(sorted(data)).encode("utf-8")

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
        self.entry = self.name = self.mdf_uuid = None


class FileTreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setUniformRowHeights(True)

        self.mode = "Natural sort"


class ChannelsTreeWidget(QtWidgets.QTreeWidget):
    itemsDeleted = QtCore.pyqtSignal(list)
    set_time_offset = QtCore.pyqtSignal(list)
    items_rearranged = QtCore.pyqtSignal(list)
    add_channels_request = QtCore.pyqtSignal(list)
    show_properties = QtCore.pyqtSignal(object)
    insert_computation = QtCore.pyqtSignal(str)
    pattern_group_added = QtCore.pyqtSignal(object)
    compute_fft_request = QtCore.pyqtSignal(str)

    def __init__(
        self, hide_missing_channels=False, hide_disabled_channels=False, *args, **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.setUniformRowHeights(False)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)
        self.details_enabled = False
        self.hide_missing_channels = hide_missing_channels
        self.hide_disabled_channels = hide_disabled_channels
        self.can_delete_items = True

        self.setHeaderHidden(True)
        self.setColumnCount(2)
        self.setDragEnabled(True)

        # self.setColumnWidth(2,40)
        # self.setColumnWidth(3,10)
        self.header().setStretchLastSection(False)

        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.itemSelectionChanged.connect(self.item_selection_changed)

        self.itemExpanded.connect(self.update_visibility_status)
        self.verticalScrollBar().valueChanged.connect(self.update_visibility_status)
        self.itemsDeleted.connect(self.update_visibility_status)

        # self.header().hideSection(0)
        self._moved = []

    def item_selection_changed(self):
        selection = list(self.selectedItems())

        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
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
            deleted = get_data(selected_items, uuids_only=True)

            root = self.invisibleRootItem()
            for item in selected_items:
                item_widget = self.itemWidget(item, 1)
                if hasattr(item_widget, "disconnect_slots"):
                    item_widget.disconnect_slots()
                (item.parent() or root).removeChild(item)

            self.refresh()

            if deleted:
                self.itemsDeleted.emit(list(deleted))

        elif key == QtCore.Qt.Key_Insert and modifiers == QtCore.Qt.ControlModifier:

            dlg = AdvancedSearch(
                {},
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
                group = ChannelsGroupTreeItem(pattern["name"], pattern)
                widget = ChannelGroupDisplay(pattern["name"], pattern, item=group)

                item = self.currentItem()

                if item is None:
                    self.addTopLevelItem(group)
                else:
                    parent = item.parent()
                    if parent:
                        current_parent = parent
                        can_add_child = True
                        while current_parent:
                            if isinstance(current_parent, ChannelsGroupTreeItem):
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

                self.setItemWidget(group, 1, widget)
                self.pattern_group_added.emit(group)

                self.refresh()

        elif key == QtCore.Qt.Key_Insert and modifiers == QtCore.Qt.ShiftModifier:
            text, ok = QtWidgets.QInputDialog.getText(
                self, "Channel group name", "New channel group name:"
            )
            if ok:
                group = ChannelsGroupTreeItem(text)
                widget = ChannelGroupDisplay(text, item=group)

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

                self.setItemWidget(group, 1, widget)
                self.refresh()

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

        else:
            super().keyPressEvent(event)

        self.update_channel_groups_count()

    def startDrag(self, supportedActions):

        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = get_data(selected_items, uuids_only=False)
        data = json.dumps(data).encode("utf-8")

        mimeData.setData("application/octet-stream-asammdf", QtCore.QByteArray(data))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec_(QtCore.Qt.MoveAction)

    def dragEnterEvent(self, e):
        e.accept()

    def dragMoveEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        if e.source() is self:

            drop_item = self.itemAt(e.pos())

            # cannot move inside pattern channel group
            current_item = drop_item
            while current_item:
                if isinstance(current_item, ChannelsGroupTreeItem):
                    if current_item.pattern:
                        e.ignore()
                        return
                current_item = current_item.parent()

            selected_items = validate_drag_items(
                self.invisibleRootItem(), self.selectedItems(), []
            )

            if drop_item is not None:
                selected_items = [
                    item
                    for item in selected_items
                    if valid_drop_target(target=drop_item, item=item)
                ]

            uuids = get_data(selected_items, uuids_only=True)

            if drop_item is None:
                add_new_items(
                    self,
                    self.invisibleRootItem(),
                    selected_items,
                    pos=None,
                )

            elif isinstance(drop_item, ChannelsTreeItem):
                parent = drop_item.parent()

                if not parent:
                    index = initial = self.indexOfTopLevelItem(self.itemAt(e.pos()))
                    index_func = self.indexOfTopLevelItem
                    root = self.invisibleRootItem()
                else:
                    index = initial = parent.indexOfChild(drop_item)
                    index_func = parent.indexOfChild
                    root = parent

                for it in selected_items:

                    idx = index_func(it)
                    if 0 <= idx < initial:
                        index -= 1

                add_new_items(
                    self,
                    root,
                    selected_items,
                    pos=index,
                )

            elif isinstance(drop_item, ChannelsGroupTreeItem):
                add_new_items(
                    self,
                    drop_item,
                    selected_items,
                    pos=0,
                )

            root = self.invisibleRootItem()
            for item in selected_items:
                item_widget = self.itemWidget(item, 1)
                if hasattr(item_widget, "disconnect_slots"):
                    item_widget.disconnect_slots()
                (item.parent() or root).removeChild(item)

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
        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while iterator.value():
            cur_item = iterator.value()
            if isinstance(cur_item, ChannelsTreeItem):
                count += 1
            iterator += 1

        menu = QtWidgets.QMenu()
        menu.addAction(self.tr(f"{count} items in the list"))
        menu.addSeparator()

        menu.addAction(self.tr(f"Add channel group [Shift+Insert]"))
        menu.addAction(self.tr(f"Add pattern based channel group [Ctrl+Insert]"))
        menu.addSeparator()

        if isinstance(item, ChannelsTreeItem):
            menu.addAction(self.tr("Copy name (Ctrl+C)"))
            menu.addAction(self.tr("Copy display properties [Ctrl+Shift+C]"))
            menu.addAction(self.tr("Paste display properties [Ctrl+Shift+P]"))
            menu.addAction(self.tr("Rename channel"))
            menu.addSeparator()
        elif isinstance(item, ChannelsGroupTreeItem):
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

        if isinstance(item, ChannelsTreeItem):
            menu.addAction(self.tr("Add to common Y axis"))
            menu.addAction(self.tr("Remove from common Y axis"))
            menu.addSeparator()
            menu.addAction(self.tr("Set unit"))
            menu.addAction(self.tr("Set precision"))
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
        if isinstance(item, ChannelsTreeItem):
            menu.addAction(self.tr("File/Computation properties"))
        elif isinstance(item, ChannelsGroupTreeItem):
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
            if isinstance(item, ChannelsTreeItem):
                self.itemWidget(item, 1).keyPressEvent(event)

        elif action.text() == "Copy display properties [Ctrl+Shift+C]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress,
                QtCore.Qt.Key_C,
                QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier,
            )
            if isinstance(item, ChannelsTreeItem):
                self.itemWidget(item, 1).keyPressEvent(event)

        elif action.text() == "Paste display properties [Ctrl+Shift+P]":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress,
                QtCore.Qt.Key_P,
                QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier,
            )
            if isinstance(item, ChannelsTreeItem):
                self.itemWidget(item, 1).keyPressEvent(event)

        elif action.text() == "Enable all":
            iterator = QtWidgets.QTreeWidgetItemIterator(self)
            while iterator.value():
                item = iterator.value()
                item.setCheckState(0, QtCore.Qt.Checked)

                iterator += 1

        elif action.text() == "Disable all":
            iterator = QtWidgets.QTreeWidgetItemIterator(self)
            while iterator.value():
                item = iterator.value()
                item.setCheckState(0, QtCore.Qt.Unchecked)

                iterator += 1

        elif action.text() == "Enable all but this":
            selected_items = self.selectedItems()

            iterator = QtWidgets.QTreeWidgetItemIterator(self)
            while iterator.value():
                item = iterator.value()
                if item in selected_items:
                    item.setCheckState(0, QtCore.Qt.Unchecked)
                else:
                    item.setCheckState(0, QtCore.Qt.Checked)

                iterator += 1

        elif action.text() == show_disabled_channels:
            if self.hide_disabled_channels:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while iterator.value():
                    item = iterator.value()
                    widget = self.itemWidget(item, 1)
                    if (
                        widget
                        and isinstance(widget, ChannelDisplay)
                        and self.hide_missing_channels
                        and not widget.exists
                    ):
                        pass
                    else:
                        item.setHidden(False)
                    iterator += 1
            else:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while iterator.value():
                    item = iterator.value()
                    if item.checkState(0) == QtCore.Qt.Unchecked:
                        item.setHidden(True)

                    iterator += 1

            self.hide_disabled_channels = not self.hide_disabled_channels

        elif action.text() == show_missing_channels:
            if self.hide_missing_channels:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while iterator.value():
                    item = iterator.value()
                    widget = self.itemWidget(item, 1)
                    if (
                        widget
                        and isinstance(widget, ChannelDisplay)
                        and self.hide_disabled_channels
                        and item.checkState(0) == QtCore.Qt.Unchecked
                    ):
                        pass
                    else:
                        item.setHidden(False)
                    iterator += 1
            else:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while iterator.value():
                    item = iterator.value()
                    widget = self.itemWidget(item, 1)
                    if isinstance(widget, ChannelDisplay) and not widget.exists:
                        item.setHidden(True)

                    iterator += 1

            self.hide_missing_channels = not self.hide_missing_channels

        elif action.text() == "Add to common Y axis":
            selected_items = self.selectedItems()

            iterator = QtWidgets.QTreeWidgetItemIterator(self)
            while iterator.value():
                item = iterator.value()
                if item in selected_items and isinstance(item, ChannelsTreeItem):
                    widget = self.itemWidget(item, 1)
                    widget.ylink.setCheckState(QtCore.Qt.Checked)
                iterator += 1

        elif action.text() == "Remove from common Y axis":
            selected_items = self.selectedItems()
            iterator = QtWidgets.QTreeWidgetItemIterator(self)
            while iterator.value():
                item = iterator.value()
                if item in selected_items and isinstance(item, ChannelsTreeItem):
                    widget = self.itemWidget(item, 1)
                    widget.ylink.setCheckState(QtCore.Qt.Unchecked)
                iterator += 1

        elif action.text() == "Set unit":
            selected_items = self.selectedItems()

            unit, ok = QtWidgets.QInputDialog.getText(None, "Set new unit", "Unit:")

            if ok:

                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while iterator.value():
                    item = iterator.value()
                    if item in selected_items and isinstance(item, ChannelsTreeItem):
                        widget = self.itemWidget(item, 1)
                        widget.set_unit(unit)
                        widget.update_information()
                    iterator += 1

        elif action.text() == "Set precision":
            selected_items = self.selectedItems()

            precision, ok = QtWidgets.QInputDialog.getInt(
                self, "Set new precision (float decimals)", "Precision:", 3, -1, 15, 1
            )

            if ok:

                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while iterator.value():
                    item = iterator.value()
                    if item in selected_items and isinstance(item, ChannelsTreeItem):
                        widget = self.itemWidget(item, 1)
                        widget.set_precision(precision)
                        widget.update_information()
                    iterator += 1

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

                    iterator = QtWidgets.QTreeWidgetItemIterator(self)
                    while iterator.value():
                        item = iterator.value()
                        if item in selected_items and isinstance(
                            item, ChannelsTreeItem
                        ):
                            widget = self.itemWidget(item, 1)
                            uuids.append(widget.uuid)
                        iterator += 1

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
                if isinstance(item, ChannelsTreeItem):
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
                if isinstance(item, ChannelsTreeItem):
                    self.show_properties.emit(self.itemWidget(item, 1).uuid)
                elif isinstance(item, ChannelsGroupTreeItem):
                    item.show_info()

        elif action.text() == "Insert computation using this channel":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                if isinstance(item, ChannelsTreeItem):
                    self.insert_computation.emit(self.itemWidget(item, 1)._name)

        elif action.text() == "Compute FFT":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                if isinstance(item, ChannelsTreeItem):
                    widget = self.itemWidget(item, 1)
                    self.compute_fft_request.emit(widget.uuid)

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
                self.itemWidget(item, 1).name.setText(text)

        elif action.text() == "Rename channel":
            text, ok = QtWidgets.QInputDialog.getText(
                self, "Rename channel", "New channel name:"
            )
            if ok and text.strip():
                text = text.strip()
                item.name = text
                self.itemWidget(item, 1).set_name(text)
                self.itemWidget(item, 1).update_information()

        elif action.text() == "Edit pattern":
            widget = self.itemWidget(item, 1)
            pattern = dict(item.pattern)
            pattern["ranges"] = copy_ranges(widget.ranges)
            dlg = AdvancedSearch(
                {},
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
                widget.set_ranges(pattern["ranges"])
                widget.set_pattern(pattern)

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
        while iterator.value():
            item = iterator.value()
            if isinstance(item, ChannelsGroupTreeItem):
                widget = self.itemWidget(item, 1)
                widget.count = item.childCount()
            iterator += 1

    def update_hidden_states(self):
        hide_missing_channels = self.hide_missing_channels
        hide_disabled_channels = self.hide_disabled_channels

        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            widget = self.itemWidget(item, 1)
            hidden = False
            if widget:
                if isinstance(widget, ChannelDisplay):
                    if hide_missing_channels and not widget.exists:
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
        while iterator.value():
            item = iterator.value()
            rect = self.visualItemRect(item)
            item._is_visible = rect.intersects(tree_rect)

            iterator += 1

    def is_item_visible(self, item):
        return item._is_visible


class ChannelsTreeItem(QtWidgets.QTreeWidgetItem):

    color_changed = QtCore.pyqtSignal(object, str)
    enable_changed = QtCore.pyqtSignal(object, int)
    ylink_changed = QtCore.pyqtSignal(object, int)
    individual_axis_changed = QtCore.pyqtSignal(object, int)

    def __init__(
        self,
        entry,
        name="",
        computation=None,
        parent=None,
        mdf_uuid=None,
        category="channel",
        texts=("", ""),
        check=None,
    ):
        super().__init__(parent, list(texts))

        self.entry = entry
        self.name = name
        self.computation = computation
        self.mdf_uuid = mdf_uuid
        self.category = category
        self._is_visible = True

        self.setFlags(
            self.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
        )

        if check is None:
            self.setCheckState(0, QtCore.Qt.Checked)
        else:
            self.setCheckState(0, check)

    def copy(self):
        x = ChannelsTreeItem(
            self.entry,
            self.name,
            self.computation,
            mdf_uuid=self.mdf_uuid,
            category=self.category,
            check=self.checkState(0),
        )
        return x

    def get_ranges(self):
        tree = self.treeWidget()
        if tree:
            widget = tree.itemWidget(self, 1)
            parent = self.parent()
            if widget is None:
                if parent is None:
                    return []
                else:
                    return parent.get_ranges(tree)
            else:
                if parent is None:
                    return widget.ranges
                else:
                    return [*widget.ranges, *parent.get_ranges(tree)]
        else:
            return []

    def update_child_values(self, tree=None):
        tree = tree or self.treeWidget()
        widget = tree.itemWidget(self, 1)
        widget.set_value(update=True)


class ChannelsGroupTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, name="", pattern=None):
        super().__init__(["", ""])
        self.name = name.split("\t[")[0]
        self.pattern = pattern
        self._is_visible = True

        self.setFlags(
            self.flags()
            | QtCore.Qt.ItemIsUserCheckable
            | QtCore.Qt.ItemIsEnabled
            | QtCore.Qt.ItemIsAutoTristate
        )

        self.setCheckState(0, QtCore.Qt.Checked)

    def copy(self):
        x = ChannelsGroupTreeItem(self.name, self.pattern)
        return x

    def show_info(self):
        widget = self.treeWidget().itemWidget(self, 1)
        if widget:
            ranges = widget.ranges
            name = widget._name or self.name
        else:
            ranges = []
            name = self.name
        ChannnelGroupDialog(name, self.pattern, ranges, self.treeWidget()).show()

    def get_ranges(self, tree=None):
        tree = tree or self.treeWidget()
        widget = tree.itemWidget(self, 1)
        parent = self.parent()
        if widget is None:
            if parent is None:
                return []
            else:
                return parent.get_ranges(tree)
        else:
            if parent is None:
                return widget.ranges
            else:
                return [*widget.ranges, *parent.get_ranges(tree)]

    def update_child_values(self, tree=None):
        tree = tree or self.treeWidget()
        count = self.childCount()
        for i in range(count):
            item = self.child(i)
            item.update_child_values(tree)


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

        screen = QtWidgets.QApplication.desktop().screenGeometry()
        self.move((screen.width() - 1200) // 2, (screen.height() - 600) // 2)


if __name__ == "__main__":
    pass

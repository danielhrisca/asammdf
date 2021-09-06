# -*- coding: utf-8 -*-

from struct import pack
from datetime import datetime, date
import json

from PyQt5 import QtCore, QtGui, QtWidgets

from ..utils import extract_mime_names
from .channel_display import ChannelDisplay
from .channel_group_display import ChannelGroupDisplay
from collections import defaultdict


def add_new_items(tree, root, items, pos):

    for item in items:

        new_item = item.copy()
        new_widget = tree.itemWidget(item, 1).copy()

        if pos is None:
            root.addChild(new_item)
        else:
            root.insertChild(pos, new_item)
            pos += 1

        tree.setItemWidget(new_item, 1, new_widget)

        if isinstance(item, ChannelsGroupTreeItem):
            child_items = [
                item.child(i)
                for i in range(item.childCount())
            ]

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
                data.append(
                    (
                        item.name,
                        None,
                        get_data(children, uuids_only),
                        None,
                        "group",
                    )
                )

        else:
            if uuids_only:
                data.append(tree.itemWidget(item, 1).uuid)
            else:
                if item.entry == (-1, -1):
                    widget = item.treeWidget().itemWidget(item, 1)
                    info = {
                        "name": item.name,
                        "computation": item.computation,
                        "computed": True,
                        "unit": widget._unit,
                        "color": widget.color,
                    }
                else:
                    info = item.name
                data.append(
                    (
                        info,
                        *item.entry,
                        item.mdf_uuid,
                        "channel"
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
                        )
                    )

            return data

        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = []
        for item in selected_items:
            data.extend(get_data(item))

        data = json.dumps(sorted(data)).encode('utf-8')

        mimeData.setData(
            "application/octet-stream-asammdf", QtCore.QByteArray(data)
        )

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

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.setUniformRowHeights(False)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)
        self.details_enabled = False
        self._has_hidden_items = False
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
            for item in self.selectedItems():
                item_widget = self.itemWidget(item, 1)
                if hasattr(item_widget, "disconnect_slots"):
                    item_widget.disconnect_slots()
                (item.parent() or root).removeChild(item)

            if deleted:
                self.itemsDeleted.emit(list(deleted))

        elif key == QtCore.Qt.Key_Insert and modifiers == QtCore.Qt.ShiftModifier:
            text, ok = QtWidgets.QInputDialog.getText(self, 'Channel group name', 'New channel group name:')
            if ok:
                group = ChannelsGroupTreeItem(text)
                widget = ChannelGroupDisplay(text)

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

        elif key == QtCore.Qt.Key_Space:
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

        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = get_data(selected_items, uuids_only=False)

        from pprint import pprint

        data = json.dumps(data).encode('utf-8')

        mimeData.setData(
            "application/octet-stream-asammdf", QtCore.QByteArray(data)
        )

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec_(QtCore.Qt.MoveAction)

    def dragEnterEvent(self, e):
        e.accept()

    def dragMoveEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )

        if e.source() is self:

            drop_item = self.itemAt(e.pos())

            selected_items = validate_drag_items(
                self.invisibleRootItem(),
                self.selectedItems(),
                []
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
            count += 1
            iterator += 1

        menu = QtWidgets.QMenu()
        menu.addAction(self.tr(f"{count} items in the list"))
        menu.addSeparator()

        menu = QtWidgets.QMenu()
        menu.addAction(self.tr(f"Add channel group"))
        menu.addSeparator()

        if isinstance(item, ChannelsTreeItem):
            menu.addAction(self.tr("Copy name (Ctrl+C)"))
            menu.addAction(self.tr("Copy display properties (Ctrl+Shift+C)"))
            menu.addAction(self.tr("Paste display properties (Ctrl+Shift+P)"))
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
        if self._has_hidden_items:
            show_hide = "Show disabled items"
        else:
            show_hide = "Hide disabled items"
        menu.addAction(self.tr(show_hide))
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
            menu.addSeparator()
        if item:
            menu.addAction(self.tr("Delete (Del)"))
            menu.addSeparator()
        menu.addAction(self.tr("Toggle details"))
        if isinstance(item, ChannelsTreeItem):
            menu.addAction(self.tr("File/Computation properties"))

        action = menu.exec_(self.viewport().mapToGlobal(position))

        if action is None:
            return

        if action.text() == "Copy name (Ctrl+C)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_C, QtCore.Qt.ControlModifier
            )
            if isinstance(item, ChannelsTreeItem):
                self.itemWidget(item, 1).keyPressEvent(event)

        elif action.text() == "Copy display properties (Ctrl+Shift+C)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress,
                QtCore.Qt.Key_C,
                QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier,
            )
            if isinstance(item, ChannelsTreeItem):
                self.itemWidget(item, 1).keyPressEvent(event)

        elif action.text() == "Paste display properties (Ctrl+Shift+P)":
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

        elif action.text() == show_hide:
            if self._has_hidden_items:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while iterator.value():
                    item = iterator.value()
                    item.setHidden(False)
                    iterator += 1
            else:
                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while iterator.value():
                    item = iterator.value()
                    if item.checkState(0) == QtCore.Qt.Unchecked:
                        item.setHidden(True)
                    iterator += 1

            self._has_hidden_items = not self._has_hidden_items

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
                        widget.update()
                    iterator += 1

        elif action.text() == "Set precision":
            selected_items = self.selectedItems()

            precision, ok = QtWidgets.QInputDialog.getInt(
                None, "Set new precision (float decimals)", "Precision:"
            )

            if ok and 0 <= precision <= 15:

                iterator = QtWidgets.QTreeWidgetItemIterator(self)
                while iterator.value():
                    item = iterator.value()
                    if item in selected_items and isinstance(item, ChannelsTreeItem):
                        widget = self.itemWidget(item, 1)
                        widget.set_precision(precision)
                        widget.update()
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
                        if item in selected_items and isinstance(item, ChannelsTreeItem):
                            widget = self.itemWidget(item, 1)
                            uuids.append(widget.uuid)
                        iterator += 1

                    self.set_time_offset.emit([absolute, offset] + uuids)

        elif action.text() == "Delete (Del)":
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
                    item.setSizeHint(widget.sizeHint())
                iterator += 1

        elif action.text() == "File/Computation properties":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                if isinstance(item, ChannelsTreeItem):
                    self.show_properties.emit(self.itemWidget(item, 1).uuid)

        elif action.text() == "Insert computation using this channel":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                if isinstance(item, ChannelsTreeItem):
                    self.insert_computation.emit(self.itemWidget(item, 1)._name)

        elif action.text() == "Add channel group":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_Insert, QtCore.Qt.ShiftModifier
            )
            self.keyPressEvent(event)

        elif action.text() == "Rename group":
            text, ok = QtWidgets.QInputDialog.getText(self, 'Rename group', 'New channel group name:')
            if ok and text.strip():
                text = text.strip()
                item.name = text
                self.itemWidget(item, 1).name.setText(text)

        elif action.text() == "Rename channel":
            text, ok = QtWidgets.QInputDialog.getText(self, 'Rename channel', 'New channel name:')
            if ok and text.strip():
                text = text.strip()
                item.name = text
                self.itemWidget(item, 1).set_name(text)
                self.itemWidget(item, 1).update()


class ChannelsTreeItem(QtWidgets.QTreeWidgetItem):

    color_changed = QtCore.pyqtSignal(object, str)
    enable_changed = QtCore.pyqtSignal(object, int)
    ylink_changed = QtCore.pyqtSignal(object, int)
    individual_axis_changed = QtCore.pyqtSignal(object, int)

    def __init__(self, entry, name="", computation=None, parent=None, mdf_uuid=None, category="channel", texts=("",""), check=None):
        super().__init__(parent, list(texts))

        self.entry = entry
        self.name = name
        self.computation = computation
        self.mdf_uuid = mdf_uuid
        self.category = category

        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)

        if check is None:
            self.setCheckState(0, QtCore.Qt.Checked)
        else:
            self.setCheckState(0, check)

    def copy(self):
        x = ChannelsTreeItem(self.entry, self.name, self.computation, mdf_uuid=self.mdf_uuid, category=self.category, check=self.checkState(0))
        return x


class ChannelsGroupTreeItem(QtWidgets.QTreeWidgetItem):

    def __init__(self, name=""):
        super().__init__(["", ""])
        self.name = name

        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsTristate)

        self.setCheckState(0, QtCore.Qt.Checked)

    def copy(self):
        x = ChannelsGroupTreeItem(self.name)
        return x



if __name__ == "__main__":
    pass
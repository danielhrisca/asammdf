# -*- coding: utf-8 -*-

from struct import pack
from datetime import datetime, date

from PyQt5 import QtCore, QtGui, QtWidgets

from ..utils import extract_mime_names
from .channel_display import ChannelDisplay
from .channel_group_display import ChannelGroupDisplay
from collections import defaultdict


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
    data = set()

    if items:
        tree = items[0].treeWidget()

    for item in items:
        count = item.childCount()

        if count:
            for i in range(count):
                child = item.child(i)

                if child.childCount():
                    children = [child.child(i) for i in range(child.childCount())]
                    data = data | get_data(children, uuids_only)
                else:
                    if isinstance(child, ChannelsTreeItem):
                        if uuids_only:
                            data.add(tree.itemWidget(child, 1).uuid)
                        else:

                            name = child.name.encode("utf-8")
                            entry = child.entry
                            if entry[1] != 0xFFFFFFFFFFFFFFFF:
                                data.add(
                                    (
                                        str(child.mdf_uuid).encode("ascii"),
                                        name,
                                        entry[0],
                                        entry[1],
                                        len(name),
                                    )
                                )
        else:
            if isinstance(item, ChannelsTreeItem):
                if uuids_only:
                    data.add(tree.itemWidget(item, 1).uuid)
                else:
                    name = item.name.encode("utf-8")
                    entry = item.entry
                    if entry[1] != 0xFFFFFFFFFFFFFFFF:
                        data.add(
                            (
                                str(item.mdf_uuid).encode("ascii"),
                                name,
                                entry[0],
                                entry[1],
                                len(name),
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
            data = set()
            count = item.childCount()

            if count:
                for i in range(count):
                    child = item.child(i)

                    if child.childCount():
                        data = data | get_data(child)
                    else:

                        name = child.name.encode("utf-8")
                        entry = child.entry
                        if entry[1] != 0xFFFFFFFFFFFFFFFF:
                            data.add(
                                (
                                    str(child.mdf_uuid).encode("ascii"),
                                    name,
                                    entry[0],
                                    entry[1],
                                    len(name),
                                )
                            )
            else:
                name = item.name.encode("utf-8")
                entry = item.entry
                if entry[1] != 0xFFFFFFFFFFFFFFFF:
                    data.add(
                        (
                            str(item.mdf_uuid).encode("ascii"),
                            name,
                            entry[0],
                            entry[1],
                            len(name),
                        )
                    )

            return data

        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = set()
        for item in selected_items:
            data = data | get_data(item)

        data = [
            pack(
                f"<12s3q{name_length}s",
                uuid,
                group_index,
                channel_index,
                name_length,
                name,
            )
            for uuid, name, group_index, channel_index, name_length in sorted(data)
        ]

        mimeData.setData(
            "application/octet-stream-asammdf", QtCore.QByteArray(b"".join(data))
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
        # self.header().hideSection(0)
        self._moved = []

    def keyPressEvent(self, event):
        key = event.key()

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

        data = [
            pack(
                f"<12s3q{name_length}s",
                uuid,
                group_index,
                channel_index,
                name_length,
                name,
            )
            for uuid, name, group_index, channel_index, name_length in sorted(data)
        ]

        mimeData.setData(
            "application/octet-stream-asammdf", QtCore.QByteArray(b"".join(data))
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

            selectedItems = validate_drag_items(
                self.invisibleRootItem(),
                self.selectedItems(),
                []
            )

            items = []
            valid_items = []
            drop_item = self.itemAt(e.pos())
            if drop_item is None:

                for it in selectedItems:
                    items.append((it.copy(), self.itemWidget(it, 1).copy()))

                for item, widget in items:
                    if widget:
                        item.setSizeHint(1, widget.sizeHint())

                root = self.invisibleRootItem()
                for item in selectedItems:
                    (item.parent() or root).removeChild(item)

                self.addTopLevelItems([elem[0] for elem in items])

                uuids = []
                for item, widget in items:
                    if widget:
                        if isinstance(widget, ChannelDisplay):
                            uuids.append(widget.uuid)
                        self.setItemWidget(item, 1, widget)

            else:

                if isinstance(drop_item, ChannelsTreeItem):
                    parent = drop_item.parent()

                    if not parent:
                        index = initial = self.indexOfTopLevelItem(self.itemAt(e.pos()))
                        index_func = self.indexOfTopLevelItem
                        insert_func = self.insertTopLevelItems
                    else:
                        index = initial = parent.indexOfChild(drop_item)
                        index_func = parent.indexOfChild
                        insert_func = parent.insertChildren

                    for it in selectedItems:

                        if not valid_drop_target(target=drop_item, item=it):
                            continue

                        valid_items.append(it)
                        idx = index_func(it)
                        if 0 <= idx < initial:
                            index -= 1

                        items.append((it.copy(), self.itemWidget(it, 1).copy()))

                    for item, widget in items:
                        if widget:
                            item.setSizeHint(1, widget.sizeHint())

                    root = self.invisibleRootItem()
                    for item in valid_items:
                        (item.parent() or root).removeChild(item)

                    insert_func(index, [elem[0] for elem in items])

                    uuids = []
                    for item, widget in items:
                        if widget:
                            if isinstance(widget, ChannelDisplay):
                                uuids.append(widget.uuid)
                            self.setItemWidget(item, 1, widget)
                else:
                    for it in selectedItems:

                        if not valid_drop_target(target=drop_item, item=it):
                            continue

                        valid_items.append(it)
                        items.append((it.copy(), self.itemWidget(it, 1).copy()))

                    for item, widget in items:
                        if widget:
                            item.setSizeHint(1, widget.sizeHint())

                    root = self.invisibleRootItem()
                    for item in valid_items:
                        (item.parent() or root).removeChild(item)

                    drop_item.insertChildren(0, [elem[0] for elem in items])

                    uuids = []
                    for item, widget in items:
                        if widget:
                            if isinstance(widget, ChannelDisplay):
                                uuids.append(widget.uuid)
                            self.setItemWidget(item, 1, widget)

            self.items_rearranged.emit(uuids)
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

        if item:
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
            menu.addAction(self.tr("Delete (Del)"))
            menu.addSeparator()
        menu.addAction(self.tr("Toggle details"))
        if item:
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
            text, ok = QtWidgets.QInputDialog.getText(self, 'Channel group name', 'New channel group name:')
            if ok:
                group = ChannelsGroupTreeItem(text)
                widget = ChannelGroupDisplay(text)

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

    def __init__(self, entry, name="", computation=None, parent=None, mdf_uuid=None, category="channel", texts=("","")):
        super().__init__(parent, list(texts))

        self.entry = entry
        self.name = name
        self.computation = computation
        self.mdf_uuid = mdf_uuid
        self.category = category

        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)

        self.setCheckState(0, QtCore.Qt.Checked)

    def copy(self):
        x = ChannelsTreeItem(self.entry, self.name, self.computation, mdf_uuid=self.mdf_uuid, category=self.category)
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
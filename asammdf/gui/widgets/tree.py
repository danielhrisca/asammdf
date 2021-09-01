# -*- coding: utf-8 -*-

from struct import pack
from datetime import datetime, date

from PyQt5 import QtCore, QtGui, QtWidgets


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
    items_rearranged = QtCore.pyqtSignal()
    add_channels_request = QtCore.pyqtSignal(list)
    show_properties = QtCore.pyqtSignal(object)
    insert_computation = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.setUniformRowHeights(True)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)
        self.details_enabled = False
        self._has_hidden_items = False

        self.setHeaderHidden(True)
        self.setColumnCount(2)
        self.setDragEnabled(True)

        # self.setColumnWidth(2,40)
        # self.setColumnWidth(3,10)
        self.header().setStretchLastSection(False)

        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        # self.header().hideSection(0)

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
        print('start drag')
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

        print('execut', b"".join(data))
        print(bin(self.dragDropMode()))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec_(QtCore.Qt.MoveAction)

    def dragEnterEvent(self, e):
        print("DragEnter")
        e.accept()

    def dragMoveEvent(self, e):
        print("DragMove")
        e.accept()

    def dropEvent(self, e):
        print('drop', e)
        if e.source() is self:
            print('a')
            super().dropEvent(e)
            self.items_rearranged.emit()
        else:
            print('b')
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
            else:
                super().dropEvent(e)

    def open_menu(self, position):

        item = self.itemAt(position)
        if item is None:
            return

        count = 0
        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while iterator.value():
            count += 1
            iterator += 1

        menu = QtWidgets.QMenu()
        menu.addAction(self.tr(f"{count} items in the list"))
        menu.addSeparator()
        menu.addAction(self.tr("Copy name (Ctrl+C)"))
        menu.addAction(self.tr("Copy display properties (Ctrl+Shift+C)"))
        menu.addAction(self.tr("Paste display properties (Ctrl+Shift+P)"))
        menu.addSeparator()
        menu.addAction(self.tr("Enable all"))
        menu.addAction(self.tr("Disable all"))
        menu.addAction(self.tr("Enable all but this"))
        menu.addSeparator()
        if self._has_hidden_items:
            show_hide = "Show disabled items"
        else:
            show_hide = "Hide disabled items"
        menu.addAction(self.tr(show_hide))
        menu.addSeparator()

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
        menu.addAction(self.tr("File/Computation properties"))

        action = menu.exec_(self.viewport().mapToGlobal(position))

        if action is None:
            return

        if action.text() == "Copy name (Ctrl+C)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_C, QtCore.Qt.ControlModifier
            )
            self.itemWidget(item).keyPressEvent(event)

        elif action.text() == "Copy display properties (Ctrl+Shift+C)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress,
                QtCore.Qt.Key_C,
                QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier,
            )
            self.itemWidget(item).keyPressEvent(event)

        elif action.text() == "Paste display properties (Ctrl+Shift+P)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress,
                QtCore.Qt.Key_P,
                QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier,
            )
            self.itemWidget(item).keyPressEvent(event)

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
                for i in range(self.count()):
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
                if item in selected_items:
                    widget = self.itemWidget(item, 1)
                    widget.ylink.setCheckState(QtCore.Qt.Checked)
                iterator += 1

        elif action.text() == "Remove from common Y axis":
            selected_items = self.selectedItems()
            iterator = QtWidgets.QTreeWidgetItemIterator(self)
            while iterator.value():
                item = iterator.value()
                if item in selected_items:
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
                    if item in selected_items:
                        widget = self.itemWidget(item, 1)
                        widget.unit = unit
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
                    if item in selected_items:
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
                        if item in selected_items:
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
                widget = self.itemWidget(item, 1)
                widget.details.setVisible(self.details_enabled)
                item.setSizeHint(widget.sizeHint())
                iterator += 1

        elif action.text() == "File/Computation properties":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                self.show_properties.emit(self.itemWidget(item, 1).uuid)

        elif action.text() == "Insert computation using this channel":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                self.insert_computation.emit(self.itemWidget(item, 1)._name)


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
        # self.setCheckState(2, QtCore.Qt.Unchecked)
        # self.setCheckState(3, QtCore.Qt.Unchecked)



if __name__ == "__main__":
    pass
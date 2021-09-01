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

        menu = QtWidgets.QMenu()
        menu.addAction(self.tr(f"{'TO DO'} items in the list"))
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
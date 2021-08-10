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


if __name__ == "__main__":
    pass
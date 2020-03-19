# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from struct import pack


class TreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setUniformRowHeights(True)

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

    def mouseMoveEvent(self, e):

        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = []

        for item in selected_items:

            count = item.childCount()

            if count:
                for i in range(count):
                    child = item.child(i)

                    name = child.name.encode("utf-8")
                    entry = child.entry

                    data.append(
                        pack(f"<3q{len(name)}s", entry[0], entry[1], len(name), name)
                    )
            else:

                name = item.name.encode("utf-8")
                entry = item.entry
                if entry[1] != 0xFFFFFFFFFFFFFFFF:
                    data.append(
                        pack(f"<3q{len(name)}s", entry[0], entry[1], len(name), name)
                    )

        mimeData.setData(
            "application/octet-stream-asammdf", QtCore.QByteArray(b"".join(data))
        )

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(e.pos() - self.rect().topLeft())

        drag.exec_(QtCore.Qt.MoveAction)

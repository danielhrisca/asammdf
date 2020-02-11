# -*- coding: utf-8 -*-
import json
from struct import pack

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from ..utils import extract_mime_names


class NumericTreeWidget(QtWidgets.QTreeWidget):
    add_channels_request = QtCore.pyqtSignal(list)
    items_rearranged = QtCore.pyqtSignal()
    items_deleted = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)

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
        elif (
            event.key() == QtCore.Qt.Key_Delete
            and event.modifiers() == QtCore.Qt.NoModifier
        ):
            selected = reversed(self.selectedItems())
            names = [item.text(0) for item in selected]
            for item in selected:
                if item.parent() is None:
                    index = self.indexFromItem(item).row()
                    self.takeTopLevelItem(index)
                else:
                    item.parent().removeChild(item)
            self.items_deleted.emit(names)
        else:
            super().keyPressEvent(event)

    def startDrag(self, supportedActions):
        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = []

        for item in selected_items:

            entry = item.entry

            if entry == (-1, -1):
                info = {
                    "name": item.name,
                    "computation": {},
                }
                info = json.dumps(info).encode("utf-8")
            else:
                info = item.name.encode("utf-8")

            data.append(pack(f"<3q{len(info)}s", entry[0], entry[1], len(info), info))

        mimeData.setData(
            "application/octet-stream-asammdf", QtCore.QByteArray(b"".join(data))
        )

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec(QtCore.Qt.CopyAction)

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):

        if e.source() is self:
            super().dropEvent(e)
            self.items_rearranged.emit()
        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
            else:
                super().dropEvent(e)

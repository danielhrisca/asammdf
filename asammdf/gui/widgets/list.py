# -*- coding: utf-8 -*-

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from struct import pack
import json

from ..utils import extract_mime_names


class ListWidget(QtWidgets.QListWidget):

    itemsDeleted = QtCore.pyqtSignal(list)
    set_time_offset = QtCore.pyqtSignal(list)
    items_rearranged = QtCore.pyqtSignal()
    add_channels_request = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)

        self.setAlternatingRowColors(True)

        self.can_delete_items = True
        self.setAcceptDrops(True)
        self.show()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key_Delete and self.can_delete_items:
            selected_items = self.selectedItems()
            deleted = []
            for item in selected_items:
                row = self.row(item)
                deleted.append(self.itemWidget(item).uuid)
                self.takeItem(row)
            if deleted:
                self.itemsDeleted.emit(deleted)
        elif key == QtCore.Qt.Key_Space and modifiers == QtCore.Qt.NoModifier:
            selected_items = self.selectedItems()
            if not selected_items:
                return

            states = [
                self.itemWidget(item).display.checkState() for item in selected_items
            ]

            if any(state == QtCore.Qt.Unchecked for state in states):
                state = QtCore.Qt.Checked
            else:
                state = QtCore.Qt.Unchecked
            for item in selected_items:
                wid = self.itemWidget(item)
                wid.display.setCheckState(state)

        elif key == QtCore.Qt.Key_Space and modifiers == QtCore.Qt.ControlModifier:
            selected_items = self.selectedItems()
            if not selected_items:
                return

            states = [
                self.itemWidget(item).individual_axis.checkState()
                for item in selected_items
            ]

            if any(state == QtCore.Qt.Unchecked for state in states):
                state = QtCore.Qt.Checked
            else:
                state = QtCore.Qt.Unchecked
            for item in selected_items:
                wid = self.itemWidget(item)
                wid.individual_axis.setCheckState(state)

        elif modifiers == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_C:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            self.itemWidget(selected_items[0]).keyPressEvent(event)

        elif modifiers == (
            QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier
        ) and key in (QtCore.Qt.Key_C, QtCore.Qt.Key_P):
            selected_items = self.selectedItems()
            if not selected_items:
                return
            self.itemWidget(selected_items[0]).keyPressEvent(event)

        else:
            super().keyPressEvent(event)
            self.parent().keyPressEvent(event)

    def startDrag(self, supportedActions):
        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = []

        for item in selected_items:

            entry = item.entry
            computation = item.computation

            widget = self.itemWidget(item)

            color = widget.color
            unit = widget.unit

            if entry == (-1, -1):
                info = {
                    "name": item.name,
                    "computation": computation,
                    "computed": True,
                    "unit": unit,
                    "color": color,
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
        if e.mimeData().hasFormat("application/octet-stream-asammdf"):
            e.accept()
        super().dragEnterEvent(e)

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

    def open_menu(self, position):

        item = self.itemAt(position)
        if item is None:
            return

        menu = QtWidgets.QMenu()
        menu.addAction(self.tr("Copy name (Ctrl+C)"))
        menu.addAction(self.tr("Copy display properties (Ctrl+Shift+C)"))
        menu.addAction(self.tr("Paste display properties (Ctrl+Shift+P)"))
        menu.addSeparator()
        menu.addAction(self.tr("Enable all"))
        menu.addAction(self.tr("Disable all"))
        menu.addAction(self.tr("Enable all but this"))
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
        menu.addAction(self.tr("Delete (Del)"))

        action = menu.exec_(self.viewport().mapToGlobal(position))

        if action is None:
            return

        if action.text() == "Copy name (Ctrl+C)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_C, QtCore.Qt.ControlModifier,
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
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                widget.display.setCheckState(QtCore.Qt.Checked)

        elif action.text() == "Disable all":
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                widget.display.setCheckState(QtCore.Qt.Unchecked)

        elif action.text() == "Enable all but this":
            selected_items = self.selectedItems()
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                if item in selected_items:
                    widget.display.setCheckState(QtCore.Qt.Unchecked)
                else:
                    widget.display.setCheckState(QtCore.Qt.Checked)

        elif action.text() == "Add to common Y axis":
            selected_items = self.selectedItems()
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                if item in selected_items:
                    widget.ylink.setCheckState(QtCore.Qt.Checked)

        elif action.text() == "Remove from common Y axis":
            selected_items = self.selectedItems()
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                if item in selected_items:
                    widget.ylink.setCheckState(QtCore.Qt.Unchecked)

        elif action.text() == "Set unit":
            selected_items = self.selectedItems()

            unit, ok = QtWidgets.QInputDialog.getText(None, "Set new unit", "Unit:",)

            if ok:

                selected_items = self.selectedItems()
                for i in range(self.count()):
                    item = self.item(i)
                    widget = self.itemWidget(item)
                    if item in selected_items:
                        widget.unit = unit
                        widget.update()

        elif action.text() == "Set precision":
            selected_items = self.selectedItems()

            precision, ok = QtWidgets.QInputDialog.getInt(
                None, "Set new precision (float decimals)", "Precision:",
            )

            if ok and 0 <= precision <= 15:

                for i in range(self.count()):
                    item = self.item(i)
                    widget = self.itemWidget(item)
                    if item in selected_items:
                        widget.set_precision(precision)
                        widget.update()

        elif action.text() in ("Relative time base shift", "Set time base start offset"):
            selected_items = self.selectedItems()
            if selected_items:

                if action.text() == "Relative time base shift":
                    offset, ok = QtWidgets.QInputDialog.getDouble(
                        self, "Relative offset [s]", "Offset [s]:",
                    )
                    absolute = False
                else:
                    offset, ok = QtWidgets.QInputDialog.getDouble(
                        self, "Absolute time start offset [s]", "Offset [s]:",
                    )
                    absolute = True
                if ok:
                    uuids = []

                    for i in range(self.count()):
                        item = self.item(i)
                        widget = self.itemWidget(item)
                        if item in selected_items:

                            uuids.append(widget.uuid)
                    self.set_time_offset.emit([absolute, offset, ] + uuids)

        elif action.text() == "Delete (Del)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_Delete, QtCore.Qt.NoModifier,
            )
            self.keyPressEvent(event)


class MinimalListWidget(QtWidgets.QListWidget):

    itemsDeleted = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.setAlternatingRowColors(True)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)

        self.setAcceptDrops(True)
        self.show()

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Delete:
            selected_items = self.selectedItems()
            deleted = []
            for item in selected_items:
                row = self.row(item)
                deleted.append(row)
                self.takeItem(row)
            if deleted:
                self.itemsDeleted.emit(deleted)
        else:
            super().keyPressEvent(event)

    def open_menu(self, position):

        item = self.itemAt(position)
        if item is None:
            return

        menu = QtWidgets.QMenu()
        menu.addAction(self.tr("Delete (Del)"))

        action = menu.exec_(self.viewport().mapToGlobal(position))

        if action is None:
            return

        if action.text() == "Delete (Del)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_Delete, QtCore.Qt.NoModifier,
            )
            self.keyPressEvent(event)

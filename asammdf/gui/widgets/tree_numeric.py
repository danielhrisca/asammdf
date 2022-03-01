# -*- coding: utf-8 -*-
import json
from struct import pack

from PySide6 import QtCore, QtGui, QtWidgets

from ..dialogs.range_editor import RangeEditor
from ..utils import extract_mime_names


class NumericTreeWidget(QtWidgets.QTreeWidget):
    add_channels_request = QtCore.Signal(list)
    items_rearranged = QtCore.Signal()
    items_deleted = QtCore.Signal(list)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)

        self.header().sortIndicatorChanged.connect(self.handle_sorting_changed)
        self.itemDoubleClicked.connect(self.handle_item_double_click)

        self._handles_double_click = True
        self.set_double_clicked_enabled(True)

    def set_double_clicked_enabled(self, state):
        self._handles_double_click = bool(state)

    def keyPressEvent(self, event):
        key = event.key()
        if (
            event.key() == QtCore.Qt.Key_Delete
            and event.modifiers() == QtCore.Qt.NoModifier
        ):
            selected = reversed(self.selectedItems())
            names = [(item.origin_uuid, item.text(0)) for item in selected]
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
                    "computation": item.computation,
                }
            else:
                info = item.name

            ranges = [dict(e) for e in item.ranges]

            for range_info in ranges:
                range_info["color"] = range_info["color"].color().name()

            data.append(
                (
                    info,
                    *item.entry,
                    str(item.origin_uuid),
                    "channel",
                    ranges,
                )
            )

        data = json.dumps(data).encode("utf-8")

        mimeData.setData("application/octet-stream-asammdf", QtCore.QByteArray(data))

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

    def handle_item_double_click(self, item, column):
        if self._handles_double_click:
            dlg = RangeEditor(
                item.name,
                ranges=item.ranges,
                parent=self,
                brush=True,
            )
            dlg.exec_()
            if dlg.pressed_button == "apply":
                item.ranges = dlg.result
                item.check_signal_range()

    def handle_sorting_changed(self, index, order):
        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            iterator += 1

            item._sorting_column = index

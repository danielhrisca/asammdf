# -*- coding: utf-8 -*-
from math import ceil

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore


class NumericTreeWidget(QtWidgets.QTreeWidget):
    add_channels_request = QtCore.pyqtSignal(list)
    items_rearranged = QtCore.pyqtSignal()
    items_deleted = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

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
                if any(item.checkState(0) == QtCore.Qt.Unchecked for item in selected_items):
                    checked = QtCore.Qt.Checked
                else:
                    checked = QtCore.Qt.Unchecked
                for item in selected_items:
                    item.setCheckState(0, checked)
        elif event.key() == QtCore.Qt.Key_Delete and event.modifiers() == QtCore.Qt.NoModifier:
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

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        if e.source() is self:
            super().dropEvent(e)
            self.items_rearranged.emit()
        else:
            data = e.mimeData()
            if data.hasFormat('application/x-qabstractitemmodeldatalist'):
                data = bytes(data.data('application/x-qabstractitemmodeldatalist'))
                data = data.replace(b'\0', b'')
                names = []

                while data:
                    _1, _2, data = data.split(b'\n', 2)

                    size = int(ceil(data[0] / 2))
                    names.append(data[1:1+size].decode('utf-8'))
                    data = data[1+size:]
                self.add_channels_request.emit(names)

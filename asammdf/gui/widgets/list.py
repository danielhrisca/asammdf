# -*- coding: utf-8 -*-
from math import ceil

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

class ListWidget(QtWidgets.QListWidget):

    itemsDeleted = QtCore.pyqtSignal(list)
    items_rearranged = QtCore.pyqtSignal()
    add_channels_request = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)

        self.setAlternatingRowColors(True)

        self.can_delete_items = True
        self.setAcceptDrops(True)
        self.show()

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Delete and self.can_delete_items:
            selected_items = self.selectedItems()
            deleted = []
            for item in selected_items:
                row = self.row(item)
                deleted.append(row)
                self.takeItem(row)
            if deleted:
                self.itemsDeleted.emit(deleted)
        elif key == QtCore.Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return

            states = [
                self.itemWidget(item).display.checkState()
                for item in selected_items
            ]

            if any(state == QtCore.Qt.Unchecked for state in states):
                state = QtCore.Qt.Checked
            else:
                state = QtCore.Qt.Unchecked
            for item in selected_items:
                wid = self.itemWidget(item)
                wid.display.setCheckState(state)

        else:
            super().keyPressEvent(event)

    def startDrag(self, supportedActions):
        drag = QtGui.QDrag(self)
        t = [self.itemWidget(i).text() for i in self.selectedItems()]
        mimeData = self.model().mimeData(self.selectedIndexes())
        mimeData.setText(str(t))
        drag.setMimeData(mimeData)
        drag.exec(QtCore.Qt.CopyAction)

    def dragEnterEvent(self, e):
        e.accept()
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        if e.source() is self:
            super().dropEvent(e)
            self.items_rearranged.emit()
        else:
            data = e.mimeData()
            if data.hasFormat('application/x-qabstractitemmodeldatalist'):
                if data.hasFormat('text/plain'):
                    names = [
                        name.strip('"\'')
                        for name in data.text().strip('[]').split(', ')
                    ]
                else:
                    model = QtGui.QStandardItemModel()
                    model.dropMimeData(data, QtCore.Qt.CopyAction, 0,0, QtCore.QModelIndex())

                    names = [
                        model.item(row, 0).text()
                        for row in range(model.rowCount())
                    ]
                self.add_channels_request.emit(names)
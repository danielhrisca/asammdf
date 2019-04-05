# -*- coding: utf-8 -*-

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

class ListWidget(QtWidgets.QListWidget):

    itemsDeleted = QtCore.pyqtSignal(list)
    add_channel_request = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

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

    def dragEnterEvent(self, e):

        e.accept()

    def dropEvent(self, e):
        data = e.mimeData()
        if data.hasFormat('application/x-qabstractitemmodeldatalist'):
            data = bytes(data.data('application/x-qabstractitemmodeldatalist'))
            name = data.replace(b'\0', b'').split(b'\n')[-1][1:].decode('utf-8')

            self.add_channel_request.emit(name)

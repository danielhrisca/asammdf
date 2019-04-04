# -*- coding: utf-8 -*-

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

class ListWidget(QtWidgets.QListWidget):

    itemsDeleted = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.setAlternatingRowColors(True)

        self.can_delete_items = True

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

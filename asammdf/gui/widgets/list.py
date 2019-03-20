# -*- coding: utf-8 -*-

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class ListWidget(QListWidget):

    itemsDeleted = pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.setAlternatingRowColors(True)

        self.can_delete_items = True

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Delete and self.can_delete_items:
            selected_items = self.selectedItems()
            deleted = []
            for item in selected_items:
                row = self.row(item)
                deleted.append(row)
                self.takeItem(row)
            if deleted:
                self.itemsDeleted.emit(deleted)
        elif key == Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return

            states = [
                self.itemWidget(item).display.checkState()
                for item in selected_items
            ]

            if any(state == Qt.Unchecked for state in states):
                state = Qt.Checked
            else:
                state = Qt.Unchecked
            for item in selected_items:
                wid = self.itemWidget(item)
                wid.display.setCheckState(state)

        else:
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                widget = self.itemWidget(selected_items[0])
                widget.keyPressEvent(event)
            else:
                super().keyPressEvent(event)

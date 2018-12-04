# -*- coding: utf-8 -*-

try:
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5 import uic
    from ..ui import resource_qt5 as resource_rc

    QT = 5

except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    from PyQt4 import uic
    from ..ui import resource_qt4 as resource_rc

    QT = 4


class ListWidget(QListWidget):

    itemsDeleted = pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super(ListWidget, self).__init__(*args, **kwargs)

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
            try:
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
            except:
                pass
        else:
            super(ListWidget, self).keyPressEvent(event)

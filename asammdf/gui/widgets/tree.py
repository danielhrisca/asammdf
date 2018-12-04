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

class TreeWidget(QTreeWidget):
    def __init__(self, *args, **kwargs):

        super(TreeWidget, self).__init__(*args, **kwargs)

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            elif len(selected_items) == 1:
                item = selected_items[0]
                checked = item.checkState(0)
                if checked == Qt.Checked:
                    item.setCheckState(0, Qt.Unchecked)
                else:
                    item.setCheckState(0, Qt.Checked)
            else:
                if any(item.checkState(0) == Qt.Unchecked for item in selected_items):
                    checked = Qt.Checked
                else:
                    checked = Qt.Unchecked
                for item in selected_items:
                    item.setCheckState(0, checked)
        else:
            super(TreeWidget, self).keyPressEvent(event)

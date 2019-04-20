# -*- coding: utf-8 -*-
from math import ceil

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

class MdiAreaWidget(QtWidgets.QMdiArea):

    add_window_request = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setAcceptDrops(True)
        self.show()

    def dragEnterEvent(self, e):
        e.accept()
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        if e.source() is self:
            super().dropEvent(e)
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
                ret, ok = QtWidgets.QInputDialog.getItem(
                    None,
                    "Select window type",
                    "Type:",
                    ["Plot", "Numeric"],
                    0,
                    False,
                )
                if ok:
                    self.add_window_request.emit([ret, names])

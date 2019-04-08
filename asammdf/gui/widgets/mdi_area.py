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
                data = bytes(data.data('application/x-qabstractitemmodeldatalist'))

                data = data.replace(b'\0', b'')
                names = []

                while data:
                    _1, _2, data = data.split(b'\n', 2)

                    size = int(ceil(data[0] / 2))
                    names.append(data[1:1+size].decode('utf-8'))
                    data = data[1+size:]

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
                # for name in names:
                #     self.add_channel_request.emit(name)

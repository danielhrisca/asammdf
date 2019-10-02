# -*- coding: utf-8 -*-

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from struct import unpack


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
            if data.hasFormat('application/octet-stream-asammdf'):
                data = bytes(data.data('application/octet-stream-asammdf'))
                size = len(data)
                names = []
                pos = 0
                while pos < size:
                    group_index, channel_index, name_length = unpack('<3Q', data[pos: pos + 24])
                    pos += 24
                    name = data[pos: pos + name_length].decode('utf-8')
                    pos += name_length
                    names.append((name, group_index, channel_index))
                ret, ok = QtWidgets.QInputDialog.getItem(
                    None,
                    "Select window type",
                    "Type:",
                    ["Plot", "Numeric", "Tabular"],
                    0,
                    False,
                )
                if ok:
                    self.add_window_request.emit([ret, names])

    def tile_vertically(self):
        sub_windows = self.subWindowList()

        position = QtCore.QPoint(0, 0)

        width = self.width()
        height = self.height()
        ratio = height // len(sub_windows)

        for window in sub_windows:
            rect = QtCore.QRect(0, 0, width, ratio)

            window.setGeometry(rect)
            window.move(position)
            position.setY(position.y() + ratio)

    def tile_horizontally(self):
        sub_windows = self.subWindowList()

        position = QtCore.QPoint(0, 0)

        width = self.width()
        height = self.height()
        ratio = width // len(sub_windows)

        for window in sub_windows:
            rect = QtCore.QRect(0, 0, ratio, height)

            window.setGeometry(rect)
            window.move(position)
            position.setX(position.x() + ratio)

# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from PyQt5 import QtCore


class TreeItem(QtWidgets.QTreeWidgetItem):

    __slots__ = 'entry',

    def __init__(self, entry):

        super().__init__()

        self.entry = entry
        self.setFlags(self.flags() | QtCore.Qt.ItemIsDragEnabled)


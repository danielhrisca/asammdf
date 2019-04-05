# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from PyQt5 import QtCore


class TreeItem(QtWidgets.QTreeWidgetItem):

    __slots__ = 'entry', 'name'

    def __init__(self, entry, name=''):

        super().__init__()

        self.entry = entry
        self.name = name
        self.setFlags(self.flags() | QtCore.Qt.ItemIsDragEnabled)


# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets


class TreeItem(QtWidgets.QTreeWidgetItem):

    __slots__ = 'entry', 'name'

    def __init__(self, entry, name='', parent=None):

        super().__init__(parent)

        self.entry = entry
        self.name = name

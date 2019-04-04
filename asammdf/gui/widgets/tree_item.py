# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets


class TreeItem(QtWidgets.QTreeWidgetItem):

    __slots__ = 'entry',

    def __init__(self, entry):

        super().__init__()

        self.entry = entry

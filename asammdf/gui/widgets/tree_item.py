# -*- coding: utf-8 -*-

try:
    from PyQt5.QtWidgets import QTreeWidgetItem
except ImportError:
    from PyQt4.QtGui import QTreeWidgetItem


class TreeItem(QTreeWidgetItem):

    __slots__ = 'entry',

    def __init__(self, entry):

        super().__init__()

        self.entry = entry

# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QTreeWidgetItem


class TreeItem(QTreeWidgetItem):

    __slots__ = 'entry',

    def __init__(self, entry):

        super().__init__()

        self.entry = entry

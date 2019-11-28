# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets


class ListItem(QtWidgets.QListWidgetItem):

    __slots__ = "entry", "name", "computation"

    def __init__(self, entry, name="", computation=None, parent=None):

        super().__init__(parent)

        self.entry = entry
        self.name = name
        self.computation = computation

# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets


class ListItem(QtWidgets.QListWidgetItem):

    __slots__ = "entry", "name", "computation", "mdf_uuid"

    def __init__(self, entry, name="", computation=None, parent=None, mdf_uuid=None):

        super().__init__()

        self.entry = entry
        self.name = name
        self.computation = computation
        self.mdf_uuid = mdf_uuid

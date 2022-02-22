# -*- coding: utf-8 -*-

from PySide6 import QtWidgets


class ListItem(QtWidgets.QListWidgetItem):
    def __init__(self, entry, name="", computation=None, parent=None, mdf_uuid=None):

        super().__init__()

        self.entry = entry
        self.name = name
        self.computation = computation
        self.mdf_uuid = mdf_uuid

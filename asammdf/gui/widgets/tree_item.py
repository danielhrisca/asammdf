# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets


class TreeItem(QtWidgets.QTreeWidgetItem):

    __slots__ = "entry", "name"

    def __init__(self, entry, name="", parent=None, strings=None):

        super().__init__(parent, strings)

        self.entry = entry
        self.name = name

    def __lt__(self, otherItem):
        column = self.treeWidget().sortColumn()

        if column == 1:
            val1 = self.text(column)
            try:
                val1 = float(val1)
            except:
                pass

            val2 = otherItem.text(column)
            try:
                val2 = float(val2)
            except:
                pass

            try:
                return val1 < val2
            except:
                if isinstance(val1, float):
                    return True
                else:
                    return False
        else:
            return self.text(column) < otherItem.text(column)

    def __del__(self):
        self.entry = self.name = None

# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets


class TreeItem(QtWidgets.QTreeWidgetItem):

    __slots__ = "entry", "name", "mdf_uuid", "ranges"

    def __init__(self, entry, name="", parent=None, strings=None, mdf_uuid=None, computation=None, ranges=None):

        super().__init__(parent, strings)

        self.entry = entry
        self.name = name
        self.mdf_uuid = mdf_uuid
        self.computation = computation or {}
        self.ranges = ranges or []

    def __lt__(self, otherItem):
        column = self.treeWidget().sortColumn()

        if column == 1:
            val1 = self.text(column)
            try:
                val1 = float(val1)
            except:
                if val1.startswith('0x'):
                    try:
                        val1 = int(val1, 16)
                    except:
                        pass
                elif val1.startswith('0b'):
                    try:
                        val1 = int(val1, 2)
                    except:
                        pass

            val2 = otherItem.text(column)
            try:
                val2 = float(val2)
            except:
                if val2.startswith('0x'):
                    try:
                        val2 = int(val2, 16)
                    except:
                        pass
                elif val2.startswith('0b'):
                    try:
                        val2 = int(val2, 2)
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
        self.entry = self.name = self.mdf_uuid = None
        
    def value(self):
        val = self.text(1)
        try:
            val = float(val)
        except:
            if val.startswith('0x'):
                try:
                    val = int(val, 16)
                except:
                    pass
            elif val.startswith('0b'):
                try:
                    val = int(val, 2)
                except:
                    pass

        return val
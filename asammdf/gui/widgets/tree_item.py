# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets, QtGui, QtCore
from ..utils import get_colors_using_ranges


class TreeItem(QtWidgets.QTreeWidgetItem):

    __slots__ = "entry", "name", "mdf_uuid", "ranges"

    def __init__(self, entry, name="", parent=None, strings=None, mdf_uuid=None, computation=None, ranges=None):

        super().__init__(parent, strings)

        self.entry = entry
        self.name = name
        self.mdf_uuid = mdf_uuid
        self.computation = computation or {}
        self.ranges = ranges or []

        self._back_ground_color = self.background(0)
        self._font_color = self.foreground(0)

        self._current_background_color = self._back_ground_color
        self._current_font_color = self._font_color

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
        
    def check_signal_range(self, value=None):
        if value is None:
            value = self.text(1).strip()
            if value:
                try:
                    value = float(value)
                except:
                    if value.startswith('0x'):
                        try:
                            value = float(int(value, 16))
                        except:
                            pass
                    elif value.startswith('0b'):
                        try:
                            value = float(int(value, 2))
                        except:
                            pass
            else:
                value = None

        new_background_color, new_font_color = get_colors_using_ranges(
            value,
            ranges=self.ranges,
            default_background_color=self._back_ground_color,
            default_font_color=self._font_color,
        )

        self.setBackground(0, new_background_color)
        self.setBackground(1, new_background_color)
        self.setBackground(2, new_background_color)

        self.setForeground(0, new_font_color)
        self.setForeground(1, new_font_color)
        self.setForeground(2, new_font_color)

# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets, QtGui, QtCore


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
        
    def check_signal_range(self, value):
        value = self.text(1)
        try:
            value = float(value)
        except:
            value = None

        if self.ranges and value is not None:

            for range in self.ranges:
                background_color, font_color, op1, op2, value1, value2 = range.values()

                result = False

                if value1 is not None:
                    if op1 == '==':
                        result = value1 == value
                    elif op1 == '!=':
                        result = value1 != value
                    elif op1 == '<=':
                        result = value1 <= value
                    elif op1 == '<':
                        result = value1 < value
                    elif op1 == '>=':
                        result = value1 >= value
                    elif op1 == '>':
                        result = value1 > value

                    if not result:
                        continue

                if value2 is not None:
                    if op2 == '==':
                        result = value == value2
                    elif op2 == '!=':
                        result = value != value2
                    elif op2 == '<=':
                        result = value <= value2
                    elif op2 == '<':
                        result = value < value2
                    elif op2 == '>=':
                        result = value >= value2
                    elif op2 == '>':
                        result = value > value2

                    if not result:
                        continue

                if result:
                    new_background_color = QtGui.QBrush(background_color)
                    new_font_color = QtGui.QBrush(font_color)
                    break
            else:
                new_background_color = self._current_background_color
                new_font_color = self._current_font_color

            self.setBackground(0, new_background_color)
            self.setBackground(1, new_background_color)
            self.setBackground(2, new_background_color)

            self.setForeground(0, new_font_color)
            self.setForeground(1, new_font_color)
            self.setForeground(2, new_font_color)
        else:
            new_color = self._back_ground_color
            self.setBackground(0, new_color)
            self.setBackground(1, new_color)
            self.setBackground(2, new_color)

            new_color = self._font_color
            self.setForeground(0, new_color)
            self.setForeground(1, new_color)
            self.setForeground(2, new_color)

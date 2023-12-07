from time import perf_counter

from PySide6 import QtWidgets

from ..utils import get_colors_using_ranges


class MinimalTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(
        self,
        entry,
        name="",
        parent=None,
        strings=None,
        origin_uuid=None,
    ):
        super().__init__(parent, strings)
        self.entry = entry
        self.name = name
        self.origin_uuid = origin_uuid


class TreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(
        self,
        entry,
        name="",
        parent=None,
        strings=None,
        origin_uuid=None,
        computation=None,
        ranges=None,
    ):
        super().__init__(parent, strings)

        self.entry = entry
        self.name = name
        self.origin_uuid = origin_uuid
        self.computation = computation or {}
        self.ranges = ranges or []

        self._back_ground_color = self.background(0)
        self._font_color = self.foreground(0)

        self._current_background_color = self._back_ground_color
        self._current_font_color = self._font_color
        self._value = None
        self._sorting_column = 0

        self._t = perf_counter()

    def __lt__(self, other):
        if self._sorting_column == 1:
            self_value = self._value
            if self_value is None:
                return True

            other_value = other._value
            if other_value is None:
                return False

            if self_value.dtype.kind in "fui":
                if other_value.dtype.kind in "fui":
                    return self_value < other_value
                else:
                    return True

            else:
                if other_value.dtype.kind in "fui":
                    return False
                else:
                    return super().__lt__(other)

        else:
            return super().__lt__(other)

    def __del__(self):
        self.entry = self.name = self.origin_uuid = None

    def check_signal_range(self, value=None):
        if value is None:
            value = self.text(1).strip()
            if value:
                try:
                    value = float(value)
                except:
                    if value.startswith("0x"):
                        try:
                            value = float(int(value, 16))
                        except:
                            pass
                    elif value.startswith("0b"):
                        try:
                            value = float(int(value, 2))
                        except:
                            pass
            else:
                value = None

        default_background_color = self._back_ground_color
        default_font_color = self._font_color

        new_background_color, new_font_color = get_colors_using_ranges(
            value,
            ranges=self.ranges,
            default_background_color=default_background_color,
            default_font_color=default_font_color,
        )

        if new_background_color is not default_background_color:
            self.setBackground(0, new_background_color)
            self.setBackground(1, new_background_color)
            self.setBackground(2, new_background_color)

        if new_font_color is not default_font_color:
            self.setForeground(0, new_font_color)
            self.setForeground(1, new_font_color)
            self.setForeground(2, new_font_color)

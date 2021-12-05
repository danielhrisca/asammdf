# -*- coding: utf-8 -*-
import json

from PyQt5 import QtCore, QtGui, QtWidgets

from ..dialogs.range_editor import RangeEditor
from ..ui import resource_rc as resource_rc
from ..ui.channel_group_display_widget import Ui_ChannelGroupDisplay
from ..utils import copy_ranges


class ChannelGroupDisplay(Ui_ChannelGroupDisplay, QtWidgets.QWidget):
    def __init__(
        self,
        name="",
        pattern=None,
        count=0,
        ranges=None,
        item=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.double_clicked_enabled = True

        self._name = name.split("\t[")[0]
        self.pattern = None
        font = self.name.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.name.setFont(font)
        self.set_pattern(pattern)
        self.count = count
        self.set_ranges(ranges or [] if not pattern else pattern.get("ranges", []))
        self.item = item

    def set_double_clicked_enabled(self, state):
        self.double_clicked_enabled = state

    def set_pattern(self, pattern):
        if pattern:
            self._icon.setPixmap(QtGui.QPixmap(":/filter.png"))
            self.pattern = pattern
        else:
            self._icon.setPixmap(QtGui.QPixmap(":/open.png"))
            self.pattern = None

    def copy(self):
        new = ChannelGroupDisplay(
            self.name.text(),
            self.pattern,
            self.count,
            ranges=copy_ranges(self.ranges),
        )

        return new

    def set_selected(self, state):
        pass

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = value

        if self.pattern:
            if value:
                self.name.setText(f"{self._name}\t[{value} matches]")
            else:
                self.name.setText(f"{self._name}\t[no matches]")
        else:
            self.name.setText(f"{self._name}\t[{value} items]")

    def mouseDoubleClickEvent(self, event):
        if self.double_clicked_enabled:
            dlg = RangeEditor(
                f"channels from <{self._name}>", ranges=self.ranges, parent=self
            )
            dlg.exec_()
            if dlg.pressed_button == "apply":
                self.set_ranges(dlg.result)
                self.item.update_child_values()

    def set_ranges(self, ranges):
        if ranges:
            self.range_indicator.setHidden(False)
        else:
            self.range_indicator.setHidden(True)
        self.ranges = ranges

# -*- coding: utf-8 -*-
import json

from PyQt5 import QtCore, QtGui, QtWidgets

from ..ui import resource_rc as resource_rc
from ..ui.channel_group_display_widget import Ui_ChannelGroupDisplay


class ChannelGroupDisplay(Ui_ChannelGroupDisplay, QtWidgets.QWidget):

    def __init__(
        self,
        name="",
        pattern_group=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.name.setText(name)
        font = self.name.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.name.setFont(font)

    def set_pattern_group(self, state):
        if state:
            self._icon.setPixmap(QtGui.QPixmap(":/filter.png"))
            self.blocked = True
        else:
            self._icon.setPixmap(QtGui.QPixmap(":/open.png"))
            self.blocked = False


    def copy(self):
        new = ChannelGroupDisplay(
            self.name.text()
        )

        return new

    def set_selected(self, state):
        pass

# -*- coding: utf-8 -*-
from PySide6 import QtCore, QtGui, QtWidgets

from ..ui import resource_rc
from ..widgets.channel_group_info import ChannelGroupInfoWidget


class ChannelGroupInfoDialog(QtWidgets.QDialog):
    def __init__(self, mdf, group, index, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(QtCore.Qt.Window)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle(f"Channel group {index}")

        layout.addWidget(ChannelGroupInfoWidget(mdf, group, self))

        self.setStyleSheet('font: 8pt "Consolas";}')

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        self.setWindowIcon(icon)
        self.setGeometry(240, 60, 1200, 600)

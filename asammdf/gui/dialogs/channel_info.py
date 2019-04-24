# -*- coding: utf-8 -*-
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from ..ui import resource_rc as resource_rc

from ..widgets.channel_info import ChannelInfoWidget


class ChannelInfoDialog(QtWidgets.QDialog):
    def __init__(self, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(QtCore.Qt.Window)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle(channel.name)

        layout.addWidget(ChannelInfoWidget(channel, self))

        self.setStyleSheet('font: 8pt "Consolas";}')

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        self.setWindowIcon(icon)
        self.setGeometry(240, 60, 1200, 600)

        screen = QtWidgets.QApplication.desktop().screenGeometry()
        self.move((screen.width() - 1200) // 2, (screen.height() - 600) // 2)

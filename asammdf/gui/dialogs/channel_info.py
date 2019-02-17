# -*- coding: utf-8 -*-
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ..ui import resource_qt5 as resource_rc

from ..widgets.channel_info import ChannelInfoWidget


class ChannelInfoDialog(QDialog):
    def __init__(self, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(Qt.Window)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle(channel.name)

        layout.addWidget(ChannelInfoWidget(channel, self))

        self.setStyleSheet('font: 8pt "Consolas";}')

        icon = QIcon()
        icon.addPixmap(QPixmap(":/info.png"), QIcon.Normal, QIcon.Off)

        self.setWindowIcon(icon)
        self.setGeometry(240, 60, 1200, 600)

        screen = QApplication.desktop().screenGeometry()
        self.move((screen.width() - 1200) // 2, (screen.height() - 600) // 2)

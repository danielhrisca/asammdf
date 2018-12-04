# -*- coding: utf-8 -*-
import os

try:
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5 import uic
    from ..ui import resource_qt5 as resource_rc

    QT = 5

except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    from PyQt4 import uic
    from ..ui import resource_qt4 as resource_rc

    QT = 4

from ..widgets.channel_info import ChannelInfoWidget

HERE = os.path.dirname(os.path.realpath(__file__))


class ChannelInfoDialog(QDialog):
    def __init__(self, channel, *args, **kwargs):
        super(QDialog, self).__init__(*args, **kwargs)

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

from PySide6 import QtCore, QtGui, QtWidgets

from ..utils import StaticIcons
from ..widgets.channel_info import ChannelInfoWidget


class ChannelInfoDialog(QtWidgets.QDialog):
    def __init__(self, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(QtCore.Qt.WindowType.Window)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle(channel.name)

        layout.addWidget(ChannelInfoWidget(channel, self))

        self.setStyleSheet('{font: 8pt "Consolas";}')

        self.setWindowIcon(StaticIcons.icon("info"))
        self.setGeometry(240, 60, 1200, 600)

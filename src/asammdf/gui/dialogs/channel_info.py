from PySide6 import QtCore, QtGui, QtWidgets

from ..widgets.channel_info import ChannelInfoWidget


class ChannelInfoDialog(QtWidgets.QDialog):
    def __init__(self, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(QtCore.Qt.WindowType.Window)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle(channel.name)

        layout.addWidget(ChannelInfoWidget(channel, self))

        self.setStyleSheet('font: 8pt "Consolas";}')

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        self.setWindowIcon(icon)
        self.setGeometry(240, 60, 1200, 600)

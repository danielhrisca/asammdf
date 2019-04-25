# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets

from ..ui import resource_rc as resource_rc
from ..ui.channel_info_widget import Ui_ChannelInfo


class ChannelInfoWidget(Ui_ChannelInfo, QtWidgets.QWidget):
    def __init__(self, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.channel_label.setText(channel.metadata())

        if channel.conversion:
            self.conversion_label.setText(channel.conversion.metadata())

        if channel.source:
            self.source_label.setText(channel.source.metadata())

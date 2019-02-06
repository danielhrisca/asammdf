# -*- coding: utf-8 -*-

import os

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

HERE = os.path.dirname(os.path.realpath(__file__))


class ChannelInfoWidget(QWidget):
    def __init__(self, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "..", "ui", "channel_info_widget.ui"), self)

        self.channel_label.setText(channel.metadata())

        if channel.conversion:
            self.conversion_label.setText(channel.conversion.metadata())

        if channel.source:
            self.source_label.setText(channel.source.metadata())

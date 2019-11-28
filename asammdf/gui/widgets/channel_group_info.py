# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets

from ..ui import resource_rc as resource_rc
from ..ui.channel_group_info_widget import Ui_ChannelGroupInfo


class ChannelGroupInfoWidget(Ui_ChannelGroupInfo, QtWidgets.QWidget):
    def __init__(self, channel_group, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.channel_group_label.setText(channel_group.metadata())

        if hasattr(channel_group, "acq_source") and channel_group.acq_source:
            self.source_label.setText(channel_group.acq_source.metadata())

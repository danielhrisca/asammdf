import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from ...blocks.utils import csv_bytearray2hex
from ..ui.channel_group_info_widget import Ui_ChannelGroupInfo
from ..utils import BLUE
from ..widgets.list_item import ListItem


class ChannelGroupInfoWidget(Ui_ChannelGroupInfo, QtWidgets.QWidget):
    def __init__(self, mdf, group, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        channel_group = group.channel_group
        self.mdf = mdf
        self.group = group

        self.channel_group_label.setText(channel_group.metadata())

        if hasattr(channel_group, "acq_source") and channel_group.acq_source:
            self.source_label.setText(channel_group.acq_source.metadata())

        items = []
        for i, ch in enumerate(group.channels):
            item = ListItem(entry=i, name=ch.name)
            item.setText(item.name)
            items.append(item)

        items.sort(key=lambda x: x.name)

        for item in items:
            self.channels.addItem(item)

        self.scroll.valueChanged.connect(self._display)
        self.channels.currentRowChanged.connect(self.select_channel)

        self.byte_count = 0
        self.byte_offset = 0
        self.position = 0

        self.index_size = len(str(channel_group.cycles_nr))
        self.cycles = channel_group.cycles_nr
        if self.mdf.version >= "4.00":
            self.record_size = channel_group.samples_byte_nr + channel_group.invalidation_bytes_nr
        else:
            self.record_size = channel_group.samples_byte_nr

        self.wrap.stateChanged.connect(self.wrap_changed)
        self._display(self.position)

    def wrap_changed(self):
        if self.wrap.checkState() == QtCore.Qt.CheckState.Checked:
            self.display.setWordWrapMode(QtGui.QTextOption.WrapMode.WordWrap)
        else:
            self.display.setWordWrapMode(QtGui.QTextOption.WrapMode.NoWrap)
        self._display(self.position)

    def select_channel(self, row):
        item = self.channels.item(row)
        channel = self.group.channels[item.entry]
        self.byte_offset = channel.byte_offset
        byte_count = channel.bit_count + channel.bit_offset

        if byte_count % 8:
            byte_count += 8 - (byte_count % 8)
        self.byte_count = byte_count // 8

        self._display(self.position)

    def _display(self, position):
        self.display.clear()

        self.position = position

        record_offset = max(0, position * self.cycles // self.scroll.maximum())
        record_end = max(0, position * self.cycles // self.scroll.maximum() + 100)
        record_count = record_end - record_offset

        data = b"".join(
            e[0] for e in self.mdf._load_data(self.group, record_offset=record_offset, record_count=record_count)
        )

        data = pd.Series(list(np.frombuffer(data, dtype=f"({self.record_size},)u1")))
        data = list(csv_bytearray2hex(data))

        lines = [
            """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html><head><meta name="qrichtext" content="1" /><style type="text/css">
p, li { white-space: pre-wrap; }
</style></head><body style=" font-size:8pt; font-style:normal;">"""
        ]

        if self.byte_count == 0:
            template = f'<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-weight:600; color:{BLUE};">{{index: >{self.index_size}}}: </span>{{line}}</p>'
            for i, l in enumerate(data, record_offset):
                lines.append(template.format(index=i, line=l))
        else:
            template = f'<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-weight:600; color:{BLUE};">{{index: >{self.index_size}}}: </span>{{start}}<span style=" font-weight:600; color:#ff5500;">{{middle}}</span>{{end}}</p>'
            for i, l in enumerate(data, record_offset):
                lines.append(
                    template.format(
                        index=i,
                        start=l[: self.byte_offset * 3],
                        middle=l[self.byte_offset * 3 : self.byte_offset * 3 + self.byte_count * 3],
                        end=l[self.byte_offset * 3 + self.byte_count * 3 :],
                    )
                )

        self.display.appendHtml("\n".join(lines))

        if position == 0:
            self.display.verticalScrollBar().setSliderPosition(0)
        elif position == self.scroll.maximum():
            self.display.verticalScrollBar().setSliderPosition(self.display.verticalScrollBar().maximum())

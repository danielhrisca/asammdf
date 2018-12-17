# -*- coding: utf-8 -*-
import os

import pyqtgraph as pg
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

HERE = os.path.dirname(os.path.realpath(__file__))


class ChannelStats(QWidget):
    def __init__(self, *args, **kwargs):
        super(ChannelStats, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "..", "ui", "channel_stats.ui"), self)

        self.color = "#000000"
        self.fmt = "phys"
        self.name_template = '<html><head/><body><p><span style=" font-size:11pt; font-weight:600; color:{};">{}</span></p></body></html>'
        self._name = "Please select a single channel"

    def set_stats(self, stats):
        if stats:
            for name, value in stats.items():
                try:
                    if value.dtype.kind in "ui":
                        sign = "-" if value < 0 else ""
                        value = abs(value)
                        if self.fmt == "hex":
                            value = "{}0x{:X}".format(sign, value)
                        elif self.fmt == "bin":
                            value = "{}0b{:b}".format(sign, value)
                        else:
                            value = "{}{}".format(sign, value)
                    else:
                        value = "{:.6f}".format(value)
                except:
                    if isinstance(value, int):
                        sign = "-" if value < 0 else ""
                        value = abs(value)
                        if self.fmt == "hex":
                            value = "{}0x{:X}".format(sign, value)
                        elif self.fmt == "bin":
                            value = "{}0b{:b}".format(sign, value)
                        else:
                            value = "{}{}".format(sign, value)
                    elif isinstance(value, float):
                        value = "{:.6f}".format(value)
                    else:
                        value = value

                if name == "unit":
                    for i in range(1, 10):
                        label = self.findChild(QLabel, "unit{}".format(i))
                        label.setText(" {}".format(value))
                elif name == "name":
                    self._name = value
                    self.name.setText(
                        self.name_template.format(self.color, self._name)
                    )
                elif name == "color":
                    self.color = value
                    self.name.setText(
                        self.name_template.format(self.color, self._name)
                    )
                else:
                    label = self.findChild(QLabel, name)
                    label.setText(value)
        else:
            self.clear()

    def clear(self):
        self._name = "Please select a single channel"
        self.color = "#000000"
        self.name.setText(self.name_template.format(self.color, self._name))
        for group in (
            self.cursor_group,
            self.range_group,
            self.visible_group,
            self.overall_group,
        ):
            layout = group.layout()
            rows = layout.rowCount()
            for i in range(rows):
                label = layout.itemAtPosition(i, 1).widget()
                label.setText("")
            for i in range(rows // 2, rows):
                label = layout.itemAtPosition(i, 2).widget()
                label.setText("")

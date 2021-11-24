# -*- coding: utf-8 -*-
from PyQt5 import QtGui, QtWidgets

from ..ui import resource_rc as resource_rc
from ..ui.channel_stats import Ui_ChannelStats

MONOSPACE_FONT = None


class ChannelStats(Ui_ChannelStats, QtWidgets.QWidget):
    def __init__(self, xunit="s", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        global MONOSPACE_FONT

        if MONOSPACE_FONT is None:

            families = QtGui.QFontDatabase().families()
            for family in (
                "Consolas",
                "Liberation Mono",
                "DejaVu Sans Mono",
                "Droid Sans Mono",
                "Liberation Mono",
                "Roboto Mono",
                "Monaco",
                "Courier",
            ):
                if family in families:
                    MONOSPACE_FONT = family
                    break

        font = QtGui.QFont(MONOSPACE_FONT)
        self.setFont(font)

        self.xunit = xunit.strip()
        self.color = "#000000"
        self.fmt = "phys"
        self.name_template = '<html><head/><body><p><span style=" font-size:11pt; font-weight:600; color:{};">{}</span></p></body></html>'
        self._name = "Please select a single channel"
        self.name.setStyleSheet("background-color: transparent;")

        for i in range(10):
            label = self.findChild(QtWidgets.QLabel, f"xunit{i}")
            label.setText(f" {self.xunit}")

    def set_stats(self, stats):
        if stats:
            for name, value in stats.items():
                #                try:
                #                    if value.dtype.kind in "ui":
                #                        sign = "-" if value < 0 else ""
                #                        value = abs(value)
                #                        if self.fmt == "hex":
                #                            value = f"{sign}0x{value:X}"
                #                        elif self.fmt == "bin":
                #                            value = f"{sign}0b{value:b}"
                #                        else:
                #                            value = f"{sign}{value}"
                #                    else:
                #                        value = f"{value:.6f}"
                #                except AttributeError:
                #                    if isinstance(value, int):
                #                        sign = "-" if value < 0 else ""
                #                        value = abs(value)
                #                        if self.fmt == "hex":
                #                            value = f"{sign}0x{value:X}"
                #                        elif self.fmt == "bin":
                #                            value = f"{sign}0b{value:b}"
                #                        else:
                #                            value = f"{sign}{value}"
                #                    elif isinstance(value, float):
                #                        value = f"{value:.6f}"
                #                    else:
                #                        value = value

                if name == "unit":
                    for i in range(1, 20):
                        label = self.findChild(QtWidgets.QLabel, f"unit{i}")
                        label.setText(f" {value}")
                    self.selected_gradient_unit.setText(f" {value}/{self.xunit}")
                    self.visible_gradient_unit.setText(f" {value}/{self.xunit}")
                    self.overall_gradient_unit.setText(f" {value}/{self.xunit}")
                    self.selected_integral_unit.setText(f" {value}*{self.xunit}")
                    self.visible_integral_unit.setText(f" {value}*{self.xunit}")
                    self.overall_integral_unit.setText(f" {value}*{self.xunit}")
                elif name == "name":
                    self._name = value
                    self.name.setText(self.name_template.format(self.color, self._name))
                elif name == "color":
                    self.color = value
                    self.name.setText(self.name_template.format(self.color, self._name))
                else:
                    label = self.findChild(QtWidgets.QLabel, name)
                    label.setText(str(value))
        else:
            self.clear()

    def clear(self):
        self._name = "Please select a single channel"
        self.color = "#000000"
        self.name.setText(self.name_template.format(self.color, self._name))
        for k, group in enumerate(
            (
                self.cursor_group,
                self.range_group,
                self.visible_group,
                self.overall_group,
            )
        ):
            layout = group.layout()
            rows = layout.rowCount()
            for i in range(rows):
                label = layout.itemAtPosition(i, 1)
                if label is not None:
                    label = label.widget()
                    label.setText("")
                label = layout.itemAtPosition(i, 2)
                if label is not None:
                    label = label.widget()
                    if label.objectName().startswith("unit"):
                        label.setText("")

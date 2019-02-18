# -*- coding: utf-8 -*-
import os
from pathlib import Path

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

from ..ui import resource_qt5 as resource_rc

HERE = Path(__file__).resolve().parent

uifile_1 = HERE.joinpath("..", "ui", "channel_display_widget.ui")

cwd = Path.cwd()
os.chdir(HERE.joinpath("..", "ui"))
form_1, base_1 = uic.loadUiType(uifile_1, resource_suffix='_qt5', from_imports=True)
os.chdir(cwd)


class ChannelDisplay(base_1, form_1):

    color_changed = pyqtSignal(int, str)
    enable_changed = pyqtSignal(int, int)
    ylink_changed = pyqtSignal(int, int)

    __slots__ = (
        'color',
        '_value_prefix',
        '_value',
        '_name',
        'fmt',
        'index',
        'ranges',
        'unit',
        '_transparent',
    )

    def __init__(self, index, unit="", *args, **kwargs):
        super(base_1, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.color = "#ff0000"
        self._value_prefix = ""
        self._value = ""
        self._name = ""
        self.fmt = "{}"
        self.index = index
        self.ranges = {}
        self.unit = unit

        self._transparent = True

        self.color_btn.clicked.connect(self.select_color)
        self.display.stateChanged.connect(self.display_changed)
        self.ylink.stateChanged.connect(self.ylink_change)

        self.setToolTip(self._name)

    def display_changed(self, state):
        state = self.display.checkState()
        self.enable_changed.emit(self.index, state)

    def ylink_change(self, state):
        state = self.ylink.checkState()
        self.ylink_changed.emit(self.index, state)

    def select_color(self):
        color = QColorDialog.getColor(QColor(self.color)).name()
        self.setColor(color)

        self.color_changed.emit(self.index, color)

    def setFmt(self, fmt):
        if fmt == "hex":
            self.fmt = "0x{:X}"
        elif fmt == "bin":
            self.fmt = "0b{:b}"
        elif fmt == "phys":
            self.fmt = "{}"
        else:
            self.fmt = fmt

    def setColor(self, color):
        self.color = color
        self.setName(self._name)
        self.setValue(self._value)
        self.color_btn.setStyleSheet(f"background-color: {color};")

    def setName(self, text=""):
        self.setToolTip(text)
        self._name = text
        if len(text) <= 32:
            self.name.setText(
                f'<html><head/><body><p><span style=" color:{self.color};">{self._name} ({self.unit})</span></p></body></html>'
            )
        else:
            self.name.setText(
                f'<html><head/><body><p><span style=" color:{self.color};">{self._name[:29]}... ({self.unit})</span></p></body></html>'
            )

    def setPrefix(self, text=""):
        self._value_prefix = text

    def setValue(self, value):
        self._value = value
        if self.ranges and value not in ("", "n.a."):
            for (start, stop), color in self.ranges.items():
                if start <= value < stop:
                    self.setStyleSheet(f"background-color: {color};")
                    self._transparent = False
                    break
            else:
                self._transparent = True
                self.setStyleSheet("background-color: transparent;")
        elif not self._transparent:
            self.setStyleSheet("background-color: transparent;")
        template = '<html><head/><body><p><span style=" color:{{}};">{{}}{}</span></p></body></html>'
        if value not in ("", "n.a."):
            template = template.format(self.fmt)
        else:
            template = template.format("{}")
        try:
            self.value.setText(template.format(self.color, self._value_prefix, value))
        except (ValueError, TypeError):
            template = '<html><head/><body><p><span style=" color:{};">{}{}</span></p></body></html>'
            self.value.setText(template.format(self.color, self._value_prefix, value))

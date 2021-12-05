# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets

from ..ui import resource_rc as resource_rc
from ..ui.range_widget import Ui_RangeWidget


class RangeWidget(Ui_RangeWidget, QtWidgets.QWidget):
    add_channels_request = QtCore.pyqtSignal(list)
    timestamp_changed_signal = QtCore.pyqtSignal(object, float)

    def __init__(
        self,
        name,
        value1="",
        op1="==",
        value2="",
        op2="==",
        font_color="#ff0000",
        background_color="#00ff00",
        brush=False,
        *args,
        **kwargs,
    ):
        super(QtWidgets.QWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self._settings = QtCore.QSettings()

        self.name.setText(name)

        self.value1.textChanged.connect(self.value1_changed)
        self.value2.textChanged.connect(self.value2_changed)

        self.background_color_btn.clicked.connect(self.select_background_color)
        self.font_color_btn.clicked.connect(self.select_font_color)

        self.value1.setText(str(value1) if value1 is not None else "")
        self.value2.setText(str(value2) if value2 is not None else "")

        self.op1.setCurrentText(op1)
        self.op2.setCurrentText(op2)

        if isinstance(font_color, QtGui.QColor):
            font_color = font_color.name()
        elif isinstance(font_color, QtGui.QBrush):
            font_color = font_color.color().name()

        if isinstance(font_color, QtGui.QColor):
            font_color = font_color.name()
        elif isinstance(font_color, QtGui.QBrush):
            font_color = font_color.color().name()

        if isinstance(background_color, QtGui.QColor):
            background_color = background_color.name()
        elif isinstance(background_color, QtGui.QBrush):
            background_color = background_color.color().name()

        if isinstance(background_color, QtGui.QColor):
            background_color = background_color.name()
        elif isinstance(background_color, QtGui.QBrush):
            background_color = background_color.color().name()

        self.font_color = font_color
        self.background_color = background_color

        self.name.setStyleSheet(
            f"background-color: {background_color}; color: {font_color};"
        )
        self.background_color_btn.setStyleSheet(
            f"background-color: {background_color};"
        )
        self.font_color_btn.setStyleSheet(f"background-color: {font_color};")

    def value1_changed(self, text):
        if text.strip():
            self.op1.setEnabled(True)
        else:
            self.op1.setEnabled(False)

    def value2_changed(self, text):
        if text.strip():
            self.op2.setEnabled(True)
        else:
            self.op2.setEnabled(False)

    def select_background_color(self, event=None):
        color = self.background_color_btn.palette().button().color()
        color = QtWidgets.QColorDialog.getColor(color)
        if color.isValid():
            color = color.name()
            self.background_color = color
            self.background_color_btn.setStyleSheet(f"background-color: {color};")
            self.name.setStyleSheet(
                f"background-color: {self.background_color}; color: {self.font_color};"
            )

    def select_font_color(self, event=None):
        color = self.font_color_btn.palette().button().color()
        color = QtWidgets.QColorDialog.getColor(color)
        if color.isValid():
            color = color.name()
            self.font_color = color
            self.font_color_btn.setStyleSheet(f"background-color: {color};")
            self.name.setStyleSheet(
                f"background-color: {self.background_color}; color: {self.font_color};"
            )

    def to_dict(self, brush=False):
        value1 = self.value1.text().strip()
        if value1:
            try:
                value1 = float(value1)
            except:
                if value1.startswith("0x"):
                    try:
                        value1 = float(int(value1, 16))
                    except:
                        pass
                elif value1.startswith("0b"):
                    try:
                        value1 = float(int(value1, 2))
                    except:
                        pass

        else:
            value1 = None

        value2 = self.value2.text().strip()
        if value2:
            try:
                value2 = float(value2)
            except:
                if value2.startswith("0x"):
                    try:
                        value2 = float(int(value2, 16))
                    except:
                        pass
                elif value2.startswith("0b"):
                    try:
                        value2 = float(int(value2, 2))
                    except:
                        pass
        else:
            value2 = None

        font_color = self.font_color_btn.palette().button().color()
        background_color = self.background_color_btn.palette().button().color()
        if brush:
            background_color = QtGui.QBrush(background_color)
            font_color = QtGui.QBrush(font_color)

        return {
            "background_color": background_color,
            "font_color": font_color,
            "op1": self.op1.currentText(),
            "op2": self.op2.currentText(),
            "value1": value1,
            "value2": value2,
        }

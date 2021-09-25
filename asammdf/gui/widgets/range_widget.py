# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtWidgets, QtGui

from ..ui import resource_rc as resource_rc
from ..ui.range_widget import Ui_RangeWidget


class RangeWidget(Ui_RangeWidget, QtWidgets.QWidget):
    add_channels_request = QtCore.pyqtSignal(list)
    timestamp_changed_signal = QtCore.pyqtSignal(object, float)

    def __init__(self, name, value1="", op1='==', value2="", op2="==", color="#000000", *args, **kwargs):
        super(QtWidgets.QWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self._settings = QtCore.QSettings()

        self.name.setText(name)

        self.value1.textChanged.connect(self.value1_changed)
        self.value2.textChanged.connect(self.value2_changed)

        self.color_btn.clicked.connect(self.select_color)

        self.value1.setText(str(value1) if value1 is not None else "")
        self.value2.setText(str(value2) if value2 is not None else "")

        self.op1.setCurrentText(op1)
        self.op2.setCurrentText(op2)

        if isinstance(color, QtGui.QColor):
            color = color.name()

        self.color_btn.setStyleSheet(f"background-color: {color};")

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

    def select_color(self, event=None):
        color = self.color_btn.palette().button().color()
        color = QtWidgets.QColorDialog.getColor(color)
        if color.isValid():
            color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {color};")

    def to_dict(self):
        value1 = self.value1.text().strip()
        if value1:
            try:
                value1 = float(value1)
            except:
                if value1.startswith('0x'):
                    try:
                        value1 = float(int(value1, 16))
                    except:
                        value1 = None
                elif value1.startswith('0b'):
                    try:
                        value1 = float(int(value1, 2))
                    except:
                        value1 = None
                else:
                    value1 = None

        else:
            value1 = None

        value2 = self.value2.text().strip()
        if value2:
            try:
                value2 = float(value2)
            except:
                if value2.startswith('0x'):
                    try:
                        value2 = float(int(value2, 16))
                    except:
                        value2 = None
                elif value2.startswith('0b'):
                    try:
                        value2 = float(int(value2, 2))
                    except:
                        value2 = None
                else:
                    value2 = None
        else:
            value2 = None
            
        return {
            "color": self.color_btn.palette().button().color(),
            "op1": self.op1.currentText(),
            "op2": self.op2.currentText(),
            "value1": value1,
            "value2": value2,
        }

# -*- coding: utf-8 -*-
import os

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


class DefineChannel(QDialog):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "..", "ui", "define_channel_dialog.ui"), self)

        self.channels = {ch.name: ch for ch in channels}
        self.result = {}
        self.pressed_button = None

        self.op1_type = None
        self.op1_value = None
        self.op2_type = None
        self.op2_value = None

        self.operand1.addItems(sorted(self.channels))
        self.operand1.insertItem(0, 'CONSTANT')
        self.operand1.setCurrentIndex(-1)

        self.operand2.addItems(sorted(self.channels))
        self.operand2.insertItem(0, 'CONSTANT')
        self.operand2.setCurrentIndex(-1)

        self.op.addItems(
            ['+', '-', '/', '//', '*', '>>', '<<', '**', '^']
        )

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)
        self.operand1.currentIndexChanged.connect(self.op1_changed)
        self.operand2.currentIndexChanged.connect(self.op2_changed)

    def op1_changed(self, index):
        if self.operand1.currentText() == 'CONSTANT':
            self.op1_type = QComboBox()
            self.op1_type.addItems(['int', 'float'])
            self.op1_type.setCurrentIndex(-1)
            self.op1_type.currentIndexChanged.connect(self.op1_constant_changed)

            self.gridLayout.addWidget(self.op1_type, 0, 2)

        else:
            if self.op1_value is not None:
                self.op1_value.setParent(None)
                self.op1_value = None
            if self.op1_type is not None:
                self.op1_type.setParent(None)
                self.op1_type = None

    def op2_changed(self, index):
        if self.operand2.currentText() == 'CONSTANT':
            self.op2_type = QComboBox()
            self.op2_type.addItems(['int', 'float'])
            self.op2_type.setCurrentIndex(-1)
            self.op2_type.currentIndexChanged.connect(self.op2_constant_changed)

            self.gridLayout.addWidget(self.op2_type, 2, 2)

        else:
            if self.op2_value is not None:
                self.op2_value.setParent(None)
                self.op2_value = None
            if self.op2_type is not None:
                self.op2_type.setParent(None)
                self.op2_type = None

    def op1_constant_changed(self, index):
        if self.op1_type.currentText() == 'int':
            if self.op1_value is not None:
                self.op1_value.setParent(None)
                self.op1_value = None
            self.op1_value = QSpinBox()
            self.op1_value.setRange(-2147483648, 2147483647)
            self.gridLayout.addWidget(self.op1_value, 0, 3)
        else:
            if self.op1_value is not None:
                self.op1_value.setParent(None)
                self.op1_value = None
            self.op1_value = QDoubleSpinBox()
            self.op1_value.setRange(-2**64, 2**64-1)
            self.gridLayout.addWidget(self.op1_value, 0, 3)

    def op2_constant_changed(self, index):
        if self.op2_type.currentText() == 'int':
            if self.op2_value is not None:
                self.op2_value.setParent(None)
                self.op2_value = None
            self.op2_value = QSpinBox()
            self.op2_value.setRange(-2147483648, 2147483647)
            self.gridLayout.addWidget(self.op2_value, 2, 3)
        else:
            if self.op2_value is not None:
                self.op2_value.setParent(None)
                self.op2_value = None
            self.op2_value = QDoubleSpinBox()
            self.op2_value.setRange(-2**64, 2**64-1)
            self.gridLayout.addWidget(self.op2_value, 2, 3)

    def apply(self, event):

        for row in range(100):
            try:
                start = self.table.cellWidget(row, 0).value()
                stop = self.table.cellWidget(row, 1).value()
                button = self.table.cellWidget(row, 2)
                color = button.palette().button().color().name()
            except:
                continue
            else:
                self.result[(start, stop)] = color
        self.pressed_button = "apply"
        self.close()

    def cancel(self, event):
        self.result = {}
        self.pressed_button = "cancel"
        self.close()

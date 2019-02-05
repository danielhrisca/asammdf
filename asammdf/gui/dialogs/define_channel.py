# -*- coding: utf-8 -*-
import os

import numpy as np

from ...signal import Signal as AsamSignal

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


OPS_TO_STR = {
    '+': '__add__',
    '-': '__sub__',
    '/': '__div__',
    '//': '__floordiv__',
    '*': '__mul__',
    '%': '__mod__',
    '**': '__pow__',
    '^': '__xor__',
    '>>': '__rshift__',
    '<<': '__lsift__',
    '&': '__and__',
    '|': '__or__',
}


class DefineChannel(QDialog):
    def __init__(self, channels, all_timebase, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "..", "ui", "define_channel_dialog.ui"), self)

        self.channels = {ch.name: ch for ch in channels}
        self.result = None
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
        self.all_timebase = all_timebase

        self.op.addItems(
            [
                '+ (add)',
                '- (substract)',
                '/ (divide)',
                '// (floor divide)',
                '* (multiply)',
                '% (modulo)',
                '** (power)',
                '^ (xor)',
                '& (and)',
                '| (or)',
                '>> (right shift)',
                '<< (left shift)',
            ]
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
        if self.operand1.currentIndex() == -1:
            QMessageBox.warning(
                None,
                "Can't compute new channel",
                "Must select operand 1 first",
            )
            return

        if self.operand2.currentIndex() == -1:
            QMessageBox.warning(
                None,
                "Can't compute new channel",
                "Must select operand 2 first",
            )
            return

        if self.op1_type is not None and self.op1_type.currentIndex() == -1:
            QMessageBox.warning(
                None,
                "Can't compute new channel",
                "Must select operand 1 type first",
            )
            return

        if self.op2_type is not None and self.op2_type.currentIndex() == -1:
            QMessageBox.warning(
                None,
                "Can't compute new channel",
                "Must select operand 2 type first",
            )

        if self.op.currentIndex() == -1:
            QMessageBox.warning(
                None,
                "Can't compute new channel",
                "Must select operator",
            )

        operand1 = self.operand1.currentText()
        operand2 = self.operand2.currentText()

        if operand1 == 'CONSTANT':
            if self.op1_type.currentText() == 'int':
                operand1 = int(self.op1_value.value())
            else:
                operand1 = float(self.op1_value.value())
            operand1_str = str(operand1)
        else:
            operand1 = self.channels[operand1]
            operand1_str = operand1.name

        if operand2 == 'CONSTANT':
            if self.op2_type.currentText() == 'int':
                operand2 = int(self.op2_value.value())
            else:
                operand2 = float(self.op2_value.value())
            operand2_str = str(operand2)
        else:
            operand2 = self.channels[operand2]
            operand2_str = operand2.name

        op = self.op.currentText().split(' ')[0]

        self.result = eval(f'operand1 {op} operand2')
        if not hasattr(self.result, 'name'):
            self.result = AsamSignal(
                name='_',
                samples=np.ones(len(self.all_timebase))*self.result,
                timestamps=self.all_timebase,
            )

        name = self.name.text()

        if not name:
            name = f'COMP_{operand1_str}{OPS_TO_STR[op]}{operand2_str}'

        self.result.name = name
        self.result.unit = self.unit.text()
        self.result.enabled = True

        self.pressed_button = "apply"
        self.close()

    def cancel(self, event):
        self.result = None
        self.pressed_button = "cancel"
        self.close()

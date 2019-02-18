# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np

from ...signal import Signal as AsamSignal
from ..ui import resource_qt5 as resource_rc
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

HERE = Path(__file__).resolve().parent


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
        uic.loadUi(HERE.joinpath("..", "ui", "define_channel_dialog.ui"), self)

        self.channels = {ch.name: ch for ch in channels}
        self.result = None
        self.pressed_button = None

        self.op1_type = None
        self.op1_value = None
        self.op2_type = None
        self.op2_value = None

        self.func_arg1 = None
        self.func_arg2 = None

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

        self.function.addItems(
            sorted(
                [
                    'arccos',
                    'arcsin',
                    'arctan',
                    'cos',
                    'deg2rad',
                    'degrees',
                    'rad2deg',
                    'radians',
                    'sin',
                    'tan',
                    'ceil',
                    'floor',
                    'rint',
                    'around',
                    'fix',
                    'trunc',
                    'cumprod',
                    'cumsum',
                    'diff',
                    'gradient',
                    'exp',
                    'log10',
                    'log',
                    'log2',
                    'absolute',
                    'cbrt',
                    'clip',
                    'sqrt',
                    'square',
                ]
            )
        )
        self.function.setCurrentIndex(-1)
        self.channel.addItems(sorted(self.channels))
        self.channel.setCurrentIndex(-1)

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)
        self.operand1.currentIndexChanged.connect(self.op1_changed)
        self.operand2.currentIndexChanged.connect(self.op2_changed)

        self.apply_function_btn.clicked.connect(self.apply_function)
        self.function.currentIndexChanged.connect(self.function_changed)

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

        try:
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
        except:
            self.result = None

        self.pressed_button = "apply"
        self.close()

    def function_changed(self, index):
        function = self.function.currentText()
        if function in [
            'arccos',
            'arcsin',
            'arctan',
            'cos',
            'deg2rad',
            'degrees',
            'rad2deg',
            'radians',
            'sin',
            'tan',
            'floor',
            'rint',
            'fix',
            'trunc',
            'cumprod',
            'cumsum',
            'diff',
            'exp',
            'log10',
            'log',
            'log2',
            'absolute',
            'cbrt',
            'sqrt',
            'square',
        ]:
            if self.func_arg1 is not None:
                self.func_arg1.setParent(None)
                self.func_arg1 = None
                self.func_arg2.setParent(None)
                self.func_arg2 = None
        else:
            if self.func_arg1 is None:
                self.func_arg1 = QDoubleSpinBox()
                self.func_arg1.setRange(-2**64, 2**64-1)
                self.gridLayout_2.addWidget(self.func_arg1, 0, 2)

                self.func_arg2 = QDoubleSpinBox()
                self.func_arg2.setRange(-2**64, 2**64-1)
                self.gridLayout_2.addWidget(self.func_arg2, 0, 3)

            if function == 'round':
                self.func_arg2.setEnabled(False)
            else:
                self.func_arg2.setEnabled(True)


    def apply_function(self, event):
        if self.function.currentIndex() == -1:
            QMessageBox.warning(
                None,
                "Can't compute new channel",
                "Must select a function first",
            )
            return

        if self.channel.currentIndex() == -1:
            QMessageBox.warning(
                None,
                "Can't compute new channel",
                "Must select a channel first",
            )
            return

        function = self.function.currentText()
        channel_name = self.channel.currentText()

        channel = self.channels[channel_name]
        func = getattr(np, function)

        try:

            if function in [
                'arccos',
                'arcsin',
                'arctan',
                'cos',
                'deg2rad',
                'degrees',
                'rad2deg',
                'radians',
                'sin',
                'tan',
                'floor',
                'rint',
                'fix',
                'trunc',
                'cumprod',
                'cumsum',
                'diff',
                'exp',
                'log10',
                'log',
                'log2',
                'absolute',
                'cbrt',
                'sqrt',
                'square',
                'gradient',
            ]:

                samples = func(channel.samples)
                if function == 'diff':
                    timestamps = channel.timestamps[1:]
                else:
                    timestamps = channel.timestamps
                name = f'{function}_{channel.name}'

            elif function == 'around':
                decimals = int(self.func_arg1.value())
                samples = func(channel.samples, decimals)
                timestamps = channel.timestamps
                name = f'{function}_{channel.name}_{decimals}'
            elif function == 'clip':
                lower = float(self.func_arg1.value())
                upper = float(self.func_arg2.value())
                samples = func(channel.samples, lower, upper)
                timestamps = channel.timestamps
                name = f'{function}_{channel.name}_{lower}_{upper}'

            name = self.function_name.text() or name
            unit = self.function_unit.text() or channel.unit

            self.result = AsamSignal(
                samples=samples,
                timestamps=timestamps,
                name=name,
                unit=unit,
            )
            self.result.enabled = True

        except Exception as err:
            QMessageBox.critical(
                None,
                "Function error",
                str(err),
            )
            self.result = None

        self.pressed_button = "apply"
        self.close()

    def cancel(self, event):
        self.result = None
        self.pressed_button = "cancel"
        self.close()

# -*- coding: utf-8 -*-

from functools import reduce
import re
from traceback import format_exc

from numexpr import evaluate
import numpy as np
from PyQt5 import QtWidgets

from ...signal import Signal as AsamSignal
from ..ui import resource_rc as resource_rc
from ..ui.define_channel_dialog import Ui_ComputedChannel

OPS_TO_STR = {
    "+": "__add__",
    "-": "__sub__",
    "/": "__div__",
    "//": "__floordiv__",
    "*": "__mul__",
    "%": "__mod__",
    "**": "__pow__",
    "^": "__xor__",
    ">>": "__rshift__",
    "<<": "__lsift__",
    "&": "__and__",
    "|": "__or__",
}


SIG_RE = re.compile(r"\{\{(?!\}\})(?P<name>.*?)\}\}")


class DefineChannel(Ui_ComputedChannel, QtWidgets.QDialog):
    def __init__(self, channels, all_timebase, name="", mdf=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.channels = {ch.name: ch for ch in channels}
        self.result = None
        self.pressed_button = None

        self.op1_type = None
        self.op1_value = None
        self.op2_type = None
        self.op2_value = None

        self.func_arg1 = None
        self.func_arg2 = None

        self.mdf = mdf

        self.operand1.addItems(sorted(self.channels))
        self.operand1.insertItem(0, "CONSTANT")
        if name:
            try:
                self.operand1.setCurrentText(name)
            except:
                self.operand1.setCurrentIndex(-1)
        else:
            self.operand1.setCurrentIndex(-1)

        self.operand2.addItems(sorted(self.channels))
        self.operand2.insertItem(0, "CONSTANT")
        self.operand2.setCurrentIndex(-1)
        self.all_timebase = all_timebase

        self.op.addItems(
            [
                "+ (add)",
                "- (substract)",
                "/ (divide)",
                "// (floor divide)",
                "* (multiply)",
                "% (modulo)",
                "** (power)",
                "^ (xor)",
                "& (and)",
                "| (or)",
                ">> (right shift)",
                "<< (left shift)",
                "average",
                "maximum",
                "minimum",
            ]
        )

        self.function.addItems(
            sorted(
                [
                    "absolute",
                    "arccos",
                    "arcsin",
                    "arctan",
                    "around",
                    "cbrt",
                    "ceil",
                    "clip",
                    "cos",
                    "cumprod",
                    "cumsum",
                    "deg2rad",
                    "degrees",
                    "diff",
                    "exp",
                    "fix",
                    "floor",
                    "gradient",
                    "log",
                    "log10",
                    "log2",
                    "rad2deg",
                    "radians",
                    "rint",
                    "sin",
                    "sqrt",
                    "square",
                    "tan",
                    "trunc",
                ]
            )
        )
        self.function.setCurrentIndex(-1)
        self.channel.addItems(sorted(self.channels))

        if name:
            try:
                self.channel.setCurrentText(name)
            except:
                self.channel.setCurrentIndex(-1)
        else:
            self.channel.setCurrentIndex(-1)

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)
        self.operand1.currentIndexChanged.connect(self.op1_changed)
        self.operand2.currentIndexChanged.connect(self.op2_changed)

        self.function.currentIndexChanged.connect(self.function_changed)

    def op1_changed(self, index):
        if self.operand1.currentText() == "CONSTANT":
            self.op1_type = QtWidgets.QComboBox()
            self.op1_type.addItems(["int", "float"])
            self.op1_type.setCurrentIndex(-1)
            self.op1_type.currentIndexChanged.connect(self.op1_constant_changed)

            self.computation_grid_layout.addWidget(self.op1_type, 0, 2)

        else:
            if self.op1_value is not None:
                self.op1_value.setParent(None)
                self.op1_value = None
            if self.op1_type is not None:
                self.op1_type.setParent(None)
                self.op1_type = None

    def op2_changed(self, index):
        if self.operand2.currentText() == "CONSTANT":
            self.op2_type = QtWidgets.QComboBox()
            self.op2_type.addItems(["int", "float"])
            self.op2_type.setCurrentIndex(-1)
            self.op2_type.currentIndexChanged.connect(self.op2_constant_changed)

            self.computation_grid_layout.addWidget(self.op2_type, 2, 2)

        else:
            if self.op2_value is not None:
                self.op2_value.setParent(None)
                self.op2_value = None
            if self.op2_type is not None:
                self.op2_type.setParent(None)
                self.op2_type = None

    def op1_constant_changed(self, index):
        if self.op1_type.currentText() == "int":
            if self.op1_value is not None:
                self.op1_value.setParent(None)
                self.op1_value = None
            self.op1_value = QtWidgets.QSpinBox()
            self.op1_value.setRange(-2147483648, 2147483647)
            self.computation_grid_layout.addWidget(self.op1_value, 0, 3)
        else:
            if self.op1_value is not None:
                self.op1_value.setParent(None)
                self.op1_value = None
            self.op1_value = QtWidgets.QDoubleSpinBox()
            self.op1_value.setDecimals(6)
            self.op1_value.setRange(-(2 ** 64), 2 ** 64 - 1)
            self.computation_grid_layout.addWidget(self.op1_value, 0, 3)

    def op2_constant_changed(self, index):
        if self.op2_type.currentText() == "int":
            if self.op2_value is not None:
                self.op2_value.setParent(None)
                self.op2_value = None
            self.op2_value = QtWidgets.QSpinBox()
            self.op2_value.setRange(-2147483648, 2147483647)
            self.computation_grid_layout.addWidget(self.op2_value, 2, 3)
        else:
            if self.op2_value is not None:
                self.op2_value.setParent(None)
                self.op2_value = None
            self.op2_value = QtWidgets.QDoubleSpinBox()
            self.op2_value.setDecimals(6)
            self.op2_value.setRange(-(2 ** 64), 2 ** 64 - 1)
            self.computation_grid_layout.addWidget(self.op2_value, 2, 3)

    def apply_simple_computation(self):
        if self.operand1.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select operand 1 first"
            )
            return

        if self.operand2.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select operand 2 first"
            )
            return

        if self.op1_type is not None and self.op1_type.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select operand 1 type first"
            )
            return

        if self.op2_type is not None and self.op2_type.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select operand 2 type first"
            )

        if self.op.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select operator"
            )

        operand1 = self.operand1.currentText()
        operand2 = self.operand2.currentText()

        if operand1 == "CONSTANT":
            if self.op1_type.currentText() == "int":
                operand1 = int(self.op1_value.value())
            else:
                operand1 = float(self.op1_value.value())
            operand1_str = str(operand1)
        else:
            operand1_sig = self.channels[operand1]
            operand1 = operand1_sig.physical()
            operand1.computed = operand1_sig.computed
            operand1.computation = operand1_sig.computation
            operand1_str = operand1.name

        if operand2 == "CONSTANT":
            if self.op2_type.currentText() == "int":
                operand2 = int(self.op2_value.value())
            else:
                operand2 = float(self.op2_value.value())
            operand2_str = str(operand2)
        else:
            operand2_sig = self.channels[operand2]
            operand2 = operand2_sig.physical()
            operand2.computed = operand2_sig.computed
            operand2.computation = operand2_sig.computation
            operand2_str = operand2.name

        op = self.op.currentText().split(" ")[0]

        try:
            if op in OPS_TO_STR:
                self.result = eval(f"operand1 {op} operand2")
                print(f"operand1 {op} operand2", operand1, operand2, op)
                print(eval(f"operand1 {op} operand2"))
            elif op == "average":
                self.result = (operand1 + operand2) / 2
            elif op in ("maximum", "minimum"):
                if isinstance(operand1, (int, float)) and isinstance(
                    operand2, (int, float)
                ):
                    fnc = min if op == "minimum" else max
                    self.result = fnc(operand1, operand2)
                elif isinstance(operand1, (int, float)):
                    fnc = np.minimum if op == "minimum" else np.maximum
                    operand1 = [operand1]
                    self.result = operand2.copy()
                    self.result.samples = fnc(self.result.samples, operand1)
                elif isinstance(operand2, (int, float)):
                    fnc = np.minimum if op == "minimum" else np.maximum
                    operand2 = [operand2]
                    self.result = operand1.copy()
                    self.result.samples = fnc(self.result.samples, operand2)
                else:
                    fnc = np.minimum if op == "minimum" else np.maximum
                    t = np.union1d(operand1.timestamps, operand2.timestamps)
                    operand1 = operand1.interp(t)
                    operand2 = operand2.interp(t)
                    self.result = AsamSignal(
                        fnc(operand1.samples, operand2.samples), t, name="_"
                    )
            if not hasattr(self.result, "name"):
                self.result = AsamSignal(
                    name="_",
                    samples=np.ones(len(self.all_timebase)) * self.result,
                    timestamps=self.all_timebase,
                )

            name = self.name.text()

            if not name:
                name = (
                    f"COMP_{operand1_str}{OPS_TO_STR.get(op, f'_{op}_')}{operand2_str}"
                )

            self.result.name = name
            self.result.unit = self.unit.text()
            self.result.enable = True
            self.result.computed = True
            self.result.computation = {
                "type": "arithmetic",
                "op": op,
            }

            if isinstance(operand1, (int, float)):
                self.result.computation["operand1"] = operand1
            else:
                if operand1.computed:
                    self.result.computation["operand1"] = operand1.computation
                else:
                    self.result.computation["operand1"] = operand1.name

            if isinstance(operand2, (int, float)):
                self.result.computation["operand2"] = operand2
            else:
                if operand2.computed:
                    self.result.computation["operand2"] = operand2.computation
                else:
                    self.result.computation["operand2"] = operand2.name
        except:
            print(format_exc())
            QtWidgets.QMessageBox.critical(
                self, "Simple computation apply error", format_exc()
            )
            self.result = None

        self.pressed_button = "apply"
        self.close()

    def function_changed(self, index):
        function = self.function.currentText()
        if function in [
            "arccos",
            "arcsin",
            "arctan",
            "cos",
            "deg2rad",
            "degrees",
            "rad2deg",
            "radians",
            "sin",
            "tan",
            "floor",
            "rint",
            "fix",
            "trunc",
            "cumprod",
            "cumsum",
            "diff",
            "exp",
            "log10",
            "log",
            "log2",
            "absolute",
            "cbrt",
            "sqrt",
            "square",
        ]:
            if self.func_arg1 is not None:
                self.func_arg1.setParent(None)
                self.func_arg1 = None
                self.func_arg2.setParent(None)
                self.func_arg2 = None
        else:
            if self.func_arg1 is None:
                self.func_arg1 = QtWidgets.QDoubleSpinBox()
                self.func_arg1.setDecimals(6)
                self.func_arg1.setRange(-(2 ** 64), 2 ** 64 - 1)
                self.gridLayout_4.addWidget(self.func_arg1, 0, 2)

                self.func_arg2 = QtWidgets.QDoubleSpinBox()
                self.func_arg2.setDecimals(6)
                self.func_arg2.setRange(-(2 ** 64), 2 ** 64 - 1)
                self.gridLayout_4.addWidget(self.func_arg2, 0, 3)

            if function == "round":
                self.func_arg2.setEnabled(False)
            else:
                self.func_arg2.setEnabled(True)

    def apply_function(self):
        if self.function.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select a function first"
            )
            return

        if self.channel.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select a channel first"
            )
            return

        function = self.function.currentText()
        channel_name = self.channel.currentText()

        channel_sig = self.channels[channel_name]
        channel = channel_sig.physical()
        channel.computed = channel_sig.computed
        channel.computation = channel_sig.computation
        func = getattr(np, function)

        try:

            if function in [
                "arccos",
                "arcsin",
                "arctan",
                "cos",
                "deg2rad",
                "degrees",
                "rad2deg",
                "radians",
                "sin",
                "tan",
                "floor",
                "rint",
                "fix",
                "trunc",
                "cumprod",
                "cumsum",
                "diff",
                "exp",
                "log10",
                "log",
                "log2",
                "absolute",
                "cbrt",
                "sqrt",
                "square",
                "gradient",
            ]:

                samples = func(channel.samples)
                if function == "diff":
                    timestamps = channel.timestamps[1:]
                else:
                    timestamps = channel.timestamps
                name = f"{function}_{channel.name}"
                args = []

            elif function == "around":
                decimals = int(self.func_arg1.value())
                samples = func(channel.samples, decimals)
                timestamps = channel.timestamps
                name = f"{function}_{channel.name}_{decimals}"
                args = [decimals]
            elif function == "clip":
                lower = float(self.func_arg1.value())
                upper = float(self.func_arg2.value())
                samples = func(channel.samples, lower, upper)
                timestamps = channel.timestamps
                name = f"{function}_{channel.name}_{lower}_{upper}"
                args = [lower, upper]

            name = self.function_name.text() or name
            unit = self.function_unit.text() or channel.unit

            self.result = AsamSignal(
                samples=samples, timestamps=timestamps, name=name, unit=unit
            )
            self.result.enabled = True
            self.result.computation = {
                "type": "function",
                "channel": channel.computation or channel_name,
                "name": function,
                "args": args,
            }

        except:
            QtWidgets.QMessageBox.critical(self, "Function error", format_exc())
            self.result = None

        self.pressed_button = "apply"

    def apply(self, event):
        if self.tabs.currentIndex() == 0:
            self.apply_simple_computation()
        elif self.tabs.currentIndex() == 1:
            self.apply_function()
        else:
            self.apply_expression()
        self.close()

    def apply_expression(self):
        expression_string = self.expression.toPlainText().strip()
        expression_string = "".join(expression_string.splitlines())
        names = [match.group("name") for match in SIG_RE.finditer(expression_string)]
        positions = [
            (i, match.start(), match.end())
            for i, match in enumerate(SIG_RE.finditer(expression_string))
        ]
        positions.reverse()

        expression = expression_string
        for idx, start, end in positions:
            expression = expression[:start] + f"X_{idx}" + expression[end:]

        if names:
            try:
                names = [
                    (None, *self.mdf.whereis(name)[0])
                    for name in names
                    if name in self.mdf
                ]
                signals = self.mdf.select(names)
                common_timebase = reduce(
                    np.union1d, [sig.timestamps for sig in signals]
                )
                signals = {
                    f"X_{i}": sig.interp(common_timebase).samples
                    for i, sig in enumerate(signals)
                }

                samples = evaluate(expression, local_dict=signals)

                self.result = AsamSignal(
                    name=self.expression_name.text() or "expression",
                    unit=self.expression_unit.text().strip(),
                    samples=samples,
                    timestamps=common_timebase,
                )
                self.result.enabled = True
                self.result.computed = True
                self.result.computation = {
                    "type": "expression",
                    "expression": expression_string,
                }

            except:
                print(format_exc())
                QtWidgets.QMessageBox.critical(
                    self, "Function apply error", format_exc()
                )
                self.result = None

        else:
            self.result = None

    def cancel(self, event):
        self.result = None
        self.pressed_button = "cancel"
        self.close()

# -*- coding: utf-8 -*-
from functools import partial
import os
import re
from traceback import format_exc

import numpy as np
from PySide6 import QtWidgets

from ..ui import resource_rc
from ..ui.define_channel_dialog import Ui_ComputedChannel
from .advanced_search import AdvancedSearch

SIG_RE = re.compile(r"\{\{(?!\}\})(?P<name>.*?)\}\}")

SHORT_OPS = [
    "+",
    "-",
    "/",
    "//",
    "*",
    "%",
    "**",
    "^",
    "&",
    "|",
    ">>",
    "<<",
]


FUNCTIONS = [
    "absolute",
    "arccos",
    "arcsin",
    "arctan",
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
    "round",
    "sin",
    "sqrt",
    "square",
    "tan",
    "trunc",
]

ARGS_COUNT = {
    **{name: 0 for name in FUNCTIONS},
    "clip": 2,
    "round": 1,
}


MULTIPLE_ARGS_FUNCTIONS = {
    "clip",
    "round",
}


class DefineChannel(Ui_ComputedChannel, QtWidgets.QDialog):
    def __init__(
        self,
        mdf,
        name="",
        computation=None,
        computed_signals=None,
        origin_uuid=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.mdf = mdf
        self.result = None
        self.pressed_button = None
        self.computed_signals = computed_signals or {}
        self.origin_uuid = origin_uuid or (mdf.uuid if mdf else os.urandom(6).hex())

        self.operand1_float.setMaximum(np.inf)
        self.operand1_float.setMinimum(-np.inf)
        self.operand1_integer.setMaximum(2**31 - 1)
        self.operand1_integer.setMinimum(-(2**31) + 1)

        self.operand2_float.setMaximum(np.inf)
        self.operand2_float.setMinimum(-np.inf)
        self.operand2_integer.setMaximum(2**31 - 1)
        self.operand2_integer.setMinimum(-(2**31) + 1)

        self.first_function_argument.setMaximum(np.inf)
        self.first_function_argument.setMinimum(-np.inf)
        self.second_function_argument.setMaximum(np.inf)
        self.second_function_argument.setMinimum(-np.inf)

        for widget in (
            self.apply_btn,
            self.cancel_btn,
            self.operand1_search_btn,
            self.operand2_search_btn,
            self.function_search_btn,
            self.expression_search_btn,
        ):
            widget.setDefault(False)
            widget.setAutoDefault(False)

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
            ]
        )

        self.function.addItems(sorted(FUNCTIONS))
        self.function.setCurrentIndex(-1)

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)

        self.operand1_search_btn.clicked.connect(
            partial(self.search, text_widget=self.operand1_name)
        )
        self.operand2_search_btn.clicked.connect(
            partial(self.search, text_widget=self.operand2_name)
        )
        self.function_search_btn.clicked.connect(
            partial(self.search, text_widget=self.function_channel)
        )
        self.expression_search_btn.clicked.connect(
            partial(self.search, text_widget=self.expression)
        )

        self.function.currentIndexChanged.connect(self.function_changed)

        if computation is None:

            if name:
                self.operand1_name.setText(name)
                self.function_channel.setText(name)

        else:

            self.name.setText(
                computation.get("channel_name", computation.get("channel", ""))
            )
            self.unit.setText(computation.get("channel_unit", ""))
            self.comment.setPlainText(computation.get("channel_comment", ""))

            if computation["type"] == "arithmetic":
                if isinstance(computation["operand1"], str):
                    self.operand1_as_signal.setChecked(True)
                    if not isinstance(computation["operand1"], dict):
                        self.operand1_name.setText(computation["operand1"])
                elif isinstance(computation["operand1"], int):
                    self.operand1_as_integer.setChecked(True)
                    self.operand1_integer.setValue(computation["operand1"])
                else:
                    self.operand1_as_float.setChecked(True)
                    self.operand1_float.setValue(computation["operand1"])

                if isinstance(computation["operand2"], str):
                    self.operand2_as_signal.setChecked(True)
                    if not isinstance(computation["operand2"], dict):
                        self.operand2_name.setText(computation["operand2"])
                elif isinstance(computation["operand2"], int):
                    self.operand2_as_integer.setChecked(True)
                    self.operand2_integer.setValue(computation["operand2"])
                else:
                    self.operand2_as_float.setChecked(True)
                    self.operand2_float.setValue(computation["operand2"])

                self.op.setCurrentIndex(SHORT_OPS.index(computation["op"]))

            elif computation["type"] == "function":
                self.tabs.setCurrentIndex(1)
                self.function.setCurrentText(computation["name"])
                if not isinstance(computation["channel"], dict):
                    self.function_channel.setText(computation["channel"])

                for arg, arg_value in zip(
                    [self.first_function_argument, self.second_function_argument],
                    computation["args"],
                ):
                    arg.setValue(arg_value)

            else:
                self.tabs.setCurrentIndex(2)
                self.expression.setPlainText(computation["expression"])

    def apply_simple_computation(self):

        if (
            self.operand1_as_signal.isChecked()
            and not self.operand1_name.text().strip()
        ):
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select operand 1 first"
            )
            return

        if (
            self.operand2_as_signal.isChecked()
            and not self.operand2_name.text().strip()
        ):
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select operand 2 first"
            )
            return

        if self.op.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select operator"
            )
            return

        if self.operand1_as_signal.isChecked():
            operand1 = self.operand1_name.text().strip()
            if operand1 in self.computed_signals:
                operand1 = self.computed_signals[operand1].computation
        elif self.operand1_as_integer.isChecked():
            operand1 = self.opernad1_integer.value()
        else:
            operand1 = self.operand1_float.value()

        if self.operand2_as_signal.isChecked():
            operand2 = self.operand2_name.text().strip()
            if operand2 in self.computed_signals:
                operand2 = self.computed_signals[operand2].computation
        elif self.operand2_as_integer.isChecked():
            operand2 = self.operand2_integer.value()
        else:
            operand2 = self.operand2_float.value()

        op = self.op.currentText().split(" ")[0]

        self.result = {
            "type": "channel",
            "common_axis": False,
            "individual_axis": False,
            "enabled": True,
            "mode": "phys",
            "fmt": "{:.3f}",
            "format": "phys",
            "precision": 3,
            "ranges": [],
            "y_range": [0, 1],
            "unit": self.unit.text().strip(),
            "computed": True,
            "color": f"#{os.urandom(3).hex()}",
            "origin_uuid": self.origin_uuid,
            "uuid": os.urandom(6).hex(),
            "group_index": -1,
            "channel_index": -1,
            "name": self.name.text().strip() or f"{operand1}_{op}_{operand2}",
            "computation": {
                "type": "arithmetic",
                "op": op,
                "operand1": operand1,
                "operand2": operand2,
                "channel_name": self.name.text().strip()
                or f"{operand1}_{op}_{operand2}",
                "channel_unit": self.unit.text().strip(),
                "channel_comment": self.comment.toPlainText().strip(),
            },
        }

        self.pressed_button = "apply"
        self.close()

    def function_changed(self, index):
        function = self.function.currentText()
        self.help.setText(getattr(np, function).__doc__)

        args = ARGS_COUNT[function]
        if args == 0:
            self.first_function_argument.setEnabled(False)
            self.second_function_argument.setEnabled(False)
        elif args == 1:
            self.first_function_argument.setEnabled(True)
            self.second_function_argument.setEnabled(False)
        else:
            self.first_function_argument.setEnabled(True)
            self.second_function_argument.setEnabled(True)

    def apply_function(self):
        if self.function.currentIndex() == -1:
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select a function first"
            )
            return

        if not self.function_channel.text().strip():
            QtWidgets.QMessageBox.warning(
                None, "Can't compute new channel", "Must select a channel first"
            )
            return

        function = self.function.currentText()
        function_channel = self.function_channel.text().strip()

        args_count = ARGS_COUNT[function]
        if args_count == 0:
            args = []
        elif args_count == 1:
            args = [self.first_function_argument.value()]
        else:
            args = [
                self.first_function_argument.value(),
                self.second_function_argument.value(),
            ]

        if function_channel in self.computed_signals:
            function_channel = self.computed_signals[function_channel].computation

        self.result = {
            "type": "channel",
            "common_axis": False,
            "individual_axis": False,
            "enabled": True,
            "mode": "phys",
            "fmt": "{:.3f}",
            "format": "phys",
            "precision": 3,
            "ranges": [],
            "y_range": [0, 1],
            "unit": self.unit.text().strip(),
            "computed": True,
            "color": f"#{os.urandom(3).hex()}",
            "uuid": os.urandom(6).hex(),
            "origin_uuid": self.origin_uuid,
            "group_index": -1,
            "channel_index": -1,
            "name": self.name.text().strip() or f"{function}({function_channel})",
            "computation": {
                "type": "function",
                "channel": function_channel,
                "name": function,
                "args": args,
                "channel_name": self.name.text().strip()
                or f"{function}({function_channel})",
                "channel_unit": self.unit.text().strip(),
                "channel_comment": self.comment.toPlainText().strip(),
            },
        }

        self.pressed_button = "apply"
        self.close()

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

        self.result = {
            "type": "channel",
            "common_axis": False,
            "individual_axis": False,
            "enabled": True,
            "mode": "phys",
            "fmt": "{:.3f}",
            "format": "phys",
            "precision": 3,
            "ranges": [],
            "y_range": [0, 1],
            "unit": self.unit.text().strip(),
            "computed": True,
            "color": f"#{os.urandom(3).hex()}",
            "uuid": os.urandom(6).hex(),
            "origin_uuid": self.origin_uuid,
            "group_index": -1,
            "channel_index": -1,
            "name": self.name.text().strip() or "expression",
            "computation": {
                "type": "expression",
                "expression": expression_string,
                "channel_name": self.name.text().strip() or "expression",
                "channel_unit": self.unit.text().strip(),
                "channel_comment": self.comment.toPlainText().strip(),
            },
        }

        self.pressed_button = "apply"
        self.close()

    def cancel(self, event):
        self.result = None
        self.pressed_button = "cancel"
        self.close()

    def search(self, *args, text_widget=None):
        dlg = AdvancedSearch(
            self.mdf,
            show_add_window=False,
            show_apply=True,
            apply_text="Select channel",
            show_pattern=False,
            parent=self,
            return_names=True,
            computed_signals=self.computed_signals,
        )
        dlg.setModal(True)
        dlg.exec_()
        result, pattern_window = dlg.result, dlg.pattern_window

        if result:
            text = list(result)[0]

            if text_widget is self.expression:
                text_widget.insertPlainText("{{" + text + "}}")
            elif text_widget is not None:
                text_widget.setText(text)

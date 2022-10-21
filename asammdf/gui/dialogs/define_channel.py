# -*- coding: utf-8 -*-
from functools import partial
import os
import re
import string
from traceback import format_exc

import numpy as np
from PySide6 import QtWidgets

from ..ui import resource_rc
from ..ui.define_channel_dialog import Ui_ComputedChannel
from ..utils import computation_to_python_function, generate_python_function
from .advanced_search import AdvancedSearch
from .error_dialog import ErrorDialog

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
        self.python_function_search_btn.clicked.connect(
            partial(self.search, text_widget=self.python_function)
        )
        self.check_syntax_btn.clicked.connect(self.check_syntax)

        self.function.currentIndexChanged.connect(self.function_changed)

        self.tabs.setTabVisible(0, False)
        self.tabs.setTabVisible(1, False)
        self.tabs.setTabVisible(2, False)

        if computation is None:

            if name:
                self.operand1_name.setText(name)
                self.function_channel.setText(name)

        else:

            computation = computation_to_python_function(computation)

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

            elif computation["type"] == "expression":
                self.tabs.setCurrentIndex(2)
                self.expression.setPlainText(computation["expression"])

            elif computation["type"] == "python_function":
                self.tabs.setCurrentIndex(3)
                self.python_function.setPlainText(computation["definition"])

        self.python_function.setPlaceholderText(
            """The channel definition is written as a Python function.
The device signal names must be placed inside double curly braces: {{MAIN_CLOCK}}.
Use the 'return' statement to return a value, otherwise 'None' will automatically be returned by the function.

Here is a minimalistic example:


if {{MAIN_CLOCK}} > 5000:
    return 0
else:
    avg = ({{p_FL}} + {{p_FR}} + {{p_RL}} + {{p_RR}}) / 4

    if avg > 20.5 and {{VehicleSpeed}} < 100:
        return avg
    else:
        return avg + 9
"""
        )

    def apply(self, event):
        if self.tabs.currentIndex() == 0:
            self.apply_simple_computation()
        elif self.tabs.currentIndex() == 1:
            self.apply_function()
        elif self.tabs.currentIndex() == 2:
            self.apply_expression()
        elif self.tabs.currentIndex() == 3:
            self.apply_python_function()
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

    def apply_python_function(self):

        if not self.check_syntax(hidden=True):
            return

        function = self.python_function.toPlainText().strip()
        name = self.name.text().strip() or f"PythonFunction_{os.urandom(6).hex()}"

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
            "name": name,
            "computation": {
                "type": "python_function",
                "definition": function,
                "channel_name": name,
                "channel_unit": self.unit.text().strip(),
                "channel_comment": self.comment.toPlainText().strip(),
            },
        }

        self.pressed_button = "apply"
        self.close()

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

    def cancel(self, event):
        self.result = None
        self.pressed_button = "cancel"
        self.close()

    def check_syntax(self, event=None, hidden=False):

        allowed_chars = string.ascii_letters + string.digits
        (
            func,
            arg_names,
            func_name,
            trace,
            function_source,
        ) = generate_python_function(
            self.python_function.toPlainText().replace("\t", "    "),
            "".join(
                ch if ch in allowed_chars else "_" for ch in self.name.text().strip()
            ),
        )

        if trace is not None:
            ErrorDialog(
                title="Virtual channel definition check",
                message="The syntax is not correct. The following error was found",
                trace=f"{trace}\n\nin the function\n\n{function_source}",
                parent=self,
            ).exec()
            return False

        try:
            func()
        except ZeroDivisionError:
            pass
        except:
            trace = format_exc()
            for i, var_name in enumerate(arg_names, 1):
                trace = trace.replace(f"_v{i}_", f"{{{{{var_name}}}}}")
                function_source = function_source.replace(
                    f"_v{i}_", f"{{{{{var_name}}}}}"
                )

            ErrorDialog(
                title="Virtual channel definition check",
                message="The syntax is not correct. The following error was found",
                trace=f"{trace}\n\nin the function\n\n{function_source}",
                parent=self,
            ).exec()

            return False

        else:
            if not hidden:
                QtWidgets.QMessageBox.information(
                    self,
                    "Virtual channel definition check",
                    "The function definition appears to be correct.",
                )

            return True

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
            if text_widget in (self.expression, self.python_function):
                for text in list(result):
                    text_widget.insertPlainText("{{" + text + "}} ")
            elif text_widget is not None:
                text_widget.setText(list(result)[0])

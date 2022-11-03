# -*- coding: utf-8 -*-
import inspect
import os
import re
from functools import partial
from traceback import format_exc

from PySide6 import QtWidgets, QtGui

from ...signal import Signal
from ..ui import resource_rc
from ..ui.define_channel_dialog import Ui_ComputedChannel
from ..utils import computation_to_python_function
from .advanced_search import AdvancedSearch
from .error_dialog import ErrorDialog

SIG_RE = re.compile(r"\{\{(?!\}\})(?P<name>.*?)\}\}")


class DefineChannel(Ui_ComputedChannel, QtWidgets.QDialog):
    def __init__(
        self,
        mdf,
        computation=None,
        computed_signals=None,
        origin_uuid=None,
        functions=None,
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

        self.arg_widgets = []

        for widget in (
            self.apply_btn,
            self.cancel_btn,
        ):
            widget.setDefault(False)
            widget.setAutoDefault(False)

        self._functions = functions or {}

        self.functions.addItems(sorted(self._functions))
        self.functions.setCurrentIndex(-1)
        self.functions.currentTextChanged.connect(self.function_changed)

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)
        self.show_definition_btn.clicked.connect(self.show_definition)

        self.trigger_search_btn.clicked.connect(self.search)

        if computation:

            computation = computation_to_python_function(computation)

            self.name.setText(
                computation.get("channel_name", computation.get("channel", ""))
            )
            self.unit.setText(computation.get("channel_unit", ""))
            self.comment.setPlainText(computation.get("channel_comment", ""))

            if computation["triggering"] == "triggering_on_all":
                self.triggering_on_all.setChecked(True)
            elif computation["triggering"] == "triggering_on_channel":
                self.triggering_on_channel.setChecked(True)
                self.trigger_interval.setValue(float(computation["triggering_value"]))
            elif computation["triggering"] == "triggering_on_interval":
                self.triggering_on_interval.setChecked(True)
                self.trigger_channel.setText(computation["triggering_value"])

            self.functions.setCurrentText(computation["function"])

            for i, name in enumerate(computation["args"].values()):
                self.arg_widgets[i][1].setText(name)

    def apply(self):

        if not self.functions.currentIndex() >= 0:
            return

        name = self.name.text().strip() or f"Function_{os.urandom(6).hex()}"

        if self.triggering_on_all.isChecked():
            triggering = "triggering_on_all"
            triggering_value = "all"
        elif self.triggering_on_interval.isChecked():
            triggering = "triggering_on_interval"
            triggering_value = self.trigger_interval.value()
        else:
            triggering = "triggering_on_channel"
            triggering_value = self.trigger_channel.text().strip()

        fargs = {}
        for i, (label, line_edit, button) in enumerate(self.arg_widgets):
            fargs[label.text()] = line_edit.text()

        self.result = {
            "type": "channel",
            "common_axis": False,
            "individual_axis": False,
            "enabled": True,
            "mode": "phys",
            "fmt": "{:.3f}",
            "format": "phys",
            "precision": 3,
            "flags": Signal.Flags.no_flags,
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
                "args": fargs,
                "type": "python_function",
                "definition": "",
                "channel_name": name,
                "function": self.functions.currentText(),
                "channel_unit": self.unit.text().strip(),
                "channel_comment": self.comment.toPlainText().strip(),
                "triggering": triggering,
                "triggering_value": triggering_value,

            },
        }

        self.pressed_button = "apply"
        self.close()

    def cancel(self, event):
        self.result = None
        self.pressed_button = "cancel"
        self.close()

    def function_changed(self, name):
        for widgets in self.arg_widgets:
            for widget in widgets:
                self.arg_layout.removeWidget(widget)
                widget.setParent(None)

        self.arg_widgets.clear()

        definition = self._functions[name]
        exec(definition)
        func = locals()[name]

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/search.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )

        parameters = list(inspect.signature(func).parameters)[:-1]
        for i, arg_name in enumerate(parameters, 2):
            label = QtWidgets.QLabel(arg_name)
            self.arg_layout.addWidget(label, i, 0)
            line_edit = QtWidgets.QLineEdit()
            self.arg_layout.addWidget(line_edit, i, 1)
            button = QtWidgets.QPushButton("")
            button.setIcon(icon)
            button.clicked.connect(partial(self.search_argument, index=i-2))
            self.arg_layout.addWidget(button, i, 2)

            self.arg_widgets.append((label, line_edit, button))

    def search_argument(self, *args, index=0):
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
            self.arg_widgets[index][1].setText(list(result)[0])

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
            self.trigger_channel.setText(list(result)[0])

    def show_definition(self, *args):
        function = self.functions.currentText()
        definition = self._functions[self.functions.currentText()]

        QtWidgets.QMessageBox.information(
            self,
            f'{function} definition',
            definition
        )

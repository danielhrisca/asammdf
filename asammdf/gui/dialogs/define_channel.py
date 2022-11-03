# -*- coding: utf-8 -*-
import os
import re
from traceback import format_exc

from PySide6 import QtWidgets

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
        name="",
        computation=None,
        computed_signals=None,
        origin_uuid=None,
        functions=(),
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

        for widget in (
            self.apply_btn,
            self.cancel_btn,
        ):
            widget.setDefault(False)
            widget.setAutoDefault(False)

        self.functions.addItems(sorted(functions))
        self.functions.setCurrentIndex(-1)

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)

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

    def apply(self):

        if not self.functions.currentIndex() >= 0:
            return

        name = self.name.text().strip() or f"PythonFunction_{os.urandom(6).hex()}"

        if self.triggering_on_all.isChecked():
            triggering = "triggering_on_all"
            triggering_value = "all"
        elif self.triggering_on_interval.isChecked():
            triggering = "triggering_on_interval"
            triggering_value = self.trigger_interval.value()
        else:
            triggering = "triggering_on_channel"
            triggering_value = self.trigger_channel.text().strip()

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

from functools import partial
import inspect
import os
import re

from PySide6 import QtCore, QtGui, QtWidgets

from ...signal import Signal
from ..ui.define_channel_dialog import Ui_ComputedChannel
from ..utils import (
    computation_to_python_function,
    generate_python_function_globals,
    generate_python_variables,
)
from ..widgets.python_highlighter import PythonHighlighter
from .advanced_search import AdvancedSearch
from .messagebox import MessageBox

SIG_RE = re.compile(r"\{\{(?!\}\})(?P<name>.*?)\}\}")


class DefineChannel(Ui_ComputedChannel, QtWidgets.QDialog):
    def __init__(
        self,
        mdf,
        computation=None,
        origin_uuid=None,
        functions=None,
        global_variables=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.trigger_interval.setMinimum(1e-9)

        global_variables = global_variables or ""
        _in_globals = generate_python_function_globals()
        ret = generate_python_variables(global_variables, in_globals=_in_globals)
        self.global_variables = {name: val for name, val in _in_globals.items() if isinstance(val, (int, float))}

        self.setWindowFlags(QtCore.Qt.WindowType.WindowMinMaxButtonsHint | self.windowFlags())

        self.mdf = mdf
        self.result = None
        self.pressed_button = None
        self.origin_uuid = origin_uuid or (mdf.uuid if mdf else os.urandom(6).hex())

        self.arg_widgets = []
        spacer = QtWidgets.QSpacerItem(
            20, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.arg_layout.addItem(spacer, len(self.arg_widgets) + 2, 0)
        self.arg_widgets.append(spacer)

        for widget in (
            self.apply_btn,
            self.cancel_btn,
        ):
            widget.setDefault(False)
            widget.setAutoDefault(False)

        self._functions = functions or {}
        self.info = None
        self.signature = None

        self.functions.addItems(sorted(self._functions))
        self.functions.setCurrentIndex(-1)
        self.functions.currentTextChanged.connect(self.function_changed)
        self.functions.currentIndexChanged.connect(self.function_changed)

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)
        self.show_definition_btn.clicked.connect(self.show_definition)

        self.trigger_search_btn.clicked.connect(self.search)

        self.global_variable_timestamps_shift.currentTextChanged.connect(
            self.update_global_variable_timestamps_description
        )

        self.computation = computation
        self.global_variable_timestamps_shift.addItems(sorted(self.global_variables))

        if computation:
            computation = computation_to_python_function(computation)

            self.name.setText(computation.get("channel_name", computation.get("channel", "")))
            self.unit.setText(computation.get("channel_unit", ""))
            self.comment.setPlainText(computation.get("channel_comment", ""))

            use_raw_values = computation.get("use_raw_values", {})

            if computation["triggering"] == "triggering_on_all":
                self.triggering_on_all.setChecked(True)

            elif computation["triggering"] == "triggering_on_channel":
                self.triggering_on_channel.setChecked(True)
                self.trigger_channel.setText(computation["triggering_value"])

            elif computation["triggering"] == "triggering_on_interval":
                self.triggering_on_interval.setChecked(True)
                self.trigger_interval.setValue(float(computation["triggering_value"]))

            if computation["function"] in self._functions:
                self.functions.setCurrentText(computation["function"])

                for i, arg_name in enumerate(list(self.signature.parameters)[:-1]):
                    names = computation["args"][arg_name]
                    self.arg_widgets[i][1].insertPlainText("\n".join(names))
                    self.arg_widgets[i][3].setChecked(use_raw_values.get(arg_name, False))

            if computation.get("computation_mode", "sample_by_sample") == "sample_by_sample":
                self.sample_by_sample.setChecked(True)
            else:
                self.complete_signal.setChecked(True)

            if "time_stamps_shift" in computation:
                time_stamps_shift = computation["time_stamps_shift"]
                if time_stamps_shift["type"] == "fixed":
                    self.shift_type_fixed.setChecked(True)
                else:
                    self.shift_type_global.setChecked(True)

                self.fixed_timestamps_shift.setValue(time_stamps_shift["fixed_timestamps_shift"])
                self.global_variable_timestamps_shift.setCurrentText(
                    time_stamps_shift["global_variable_timestamps_shift"]
                )

        self.showMaximized()

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
        raw_info = {}
        for i, (label, text_edit, button, check, *_) in enumerate(self.arg_widgets[:-1]):
            names = text_edit.toPlainText().splitlines()
            names = [line.strip() for line in names if line.strip()]
            fargs[label.text()] = names
            raw_info[label.text()] = check.isChecked()

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
                "use_raw_values": raw_info,
                "type": "python_function",
                "definition": "",
                "channel_name": name,
                "function": self.functions.currentText(),
                "channel_unit": self.unit.text().strip(),
                "channel_comment": self.comment.toPlainText().strip(),
                "triggering": triggering,
                "triggering_value": triggering_value,
                "computation_mode": "sample_by_sample" if self.sample_by_sample.isChecked() else "complete_signal",
                "time_stamps_shift": {
                    "fixed_timestamps_shift": self.fixed_timestamps_shift.value(),
                    "global_variable_timestamps_shift": self.global_variable_timestamps_shift.currentText() or "",
                    "type": "fixed" if self.shift_type_fixed.isChecked() else "global",
                },
            },
        }

        self.pressed_button = "apply"
        self.close()

    def cancel(self, event):
        self.result = None
        self.pressed_button = "cancel"
        self.close()

    def function_changed(self, *args):
        name = self.functions.currentText()
        for widgets in self.arg_widgets[:-1]:
            for widget in widgets:
                self.arg_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()

        self.arg_layout.removeItem(self.arg_widgets[-1])

        self.arg_widgets.clear()

        definition = self._functions[name]
        exec(definition.replace("\t", "    "))
        func = locals()[name]

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/search.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        self.signature = inspect.signature(func)

        parameters = list(self.signature.parameters)[:-1]
        for i, arg_name in enumerate(parameters):
            row = i * 3 + 2  # function name and arguments label raws

            label = QtWidgets.QLabel(arg_name)
            self.arg_layout.addWidget(label, row, 0)
            text_edit = QtWidgets.QPlainTextEdit()
            self.arg_layout.addWidget(text_edit, row, 1)
            button = QtWidgets.QPushButton("")
            button.setIcon(icon)
            button.clicked.connect(partial(self.search_argument, index=i))
            self.arg_layout.addWidget(button, row, 2)
            check = QtWidgets.QCheckBox("use raw values")
            self.arg_layout.addWidget(check, row+1, 1)
            self.arg_widgets.append((label, text_edit, button, check))

            if i < len(parameters) - 1:
                line = QtWidgets.QFrame()
                line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
                self.arg_layout.addWidget(line, row + 2, 0, 1, 3)
                self.arg_widgets.append([label, text_edit, button, check, line])
            else:
                self.arg_widgets.append((label, text_edit, button, check))

        spacer = QtWidgets.QSpacerItem(
            20, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding
        )

        self.arg_layout.addItem(spacer, len(self.arg_widgets) + 2, 0)
        self.arg_widgets.append(spacer)

    def search_argument(self, *args, index=0):
        dlg = AdvancedSearch(
            self.mdf,
            show_add_window=False,
            show_apply=True,
            apply_text="Select channel",
            show_pattern=False,
            parent=self,
            return_names=True,
        )
        dlg.setModal(True)
        dlg.exec_()
        result, pattern_window = dlg.result, dlg.pattern_window

        if result:
            lines = [self.arg_widgets[index][1].toPlainText(), *list(result)]
            self.arg_widgets[index][1].setPlainText("\n".join(lines))

    def search(self, *args, text_widget=None):
        dlg = AdvancedSearch(
            self.mdf,
            show_add_window=False,
            show_apply=True,
            apply_text="Select channel",
            show_pattern=False,
            parent=self,
            return_names=True,
        )
        dlg.setModal(True)
        dlg.exec_()
        result, pattern_window = dlg.result, dlg.pattern_window

        if result:
            self.trigger_channel.setText(list(result)[0])

    def show_definition(self, *args):
        function = self.functions.currentText()
        if function:
            definition = self._functions[self.functions.currentText()]

            # keep a reference otherwise the window gets closed
            self.info = info = QtWidgets.QPlainTextEdit(definition)
            PythonHighlighter(info.document())
            info.setReadOnly(True)
            info.setLineWrapMode(info.NoWrap)
            info.setWindowFlags(QtCore.Qt.WindowType.WindowMinMaxButtonsHint | info.windowFlags())
            info.setWindowTitle(f"{function} definition")
            info.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

            p = info.palette()
            for active in (QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorGroup.Inactive):
                p.setColor(active, QtGui.QPalette.ColorRole.Base, QtGui.QColor("#131314"))
                p.setColor(active, QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#ffffff"))
                p.setColor(active, QtGui.QPalette.ColorRole.Text, QtGui.QColor("#ffffff"))
            info.setPalette(p)

            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            info.setWindowIcon(icon)

            info.show()
            rect = info.geometry()
            rect.setWidth(600)
            rect.setHeight(400)
            info.setGeometry(rect)

        else:
            if self.computation:
                function = self.computation["function"]
                MessageBox.warning(
                    self,
                    f"{function} definition missing",
                    f"The function {function} was not found in the Functions manager",
                )
            else:
                MessageBox.warning(
                    self,
                    "No function selected",
                    "Please select one of the fucntion defined in the Functions manager",
                )

    def update_global_variable_timestamps_description(self, text):
        if text:
            if text in self.global_variables:
                self.global_variable_timstamps_shift_description.setText(f"{text} = {self.global_variables[text]:.9f}s")
            else:
                self.global_variable_timstamps_shift_description.setText(f"{text} not found => default shift is 0s")
        else:
            self.global_variable_timstamps_shift_description.setText("default shift 0s")

import json
import os
from pathlib import Path

from natsort import natsorted
from PySide6 import QtCore, QtGui, QtWidgets

from ..dialogs.error_dialog import ErrorDialog
from ..dialogs.messagebox import MessageBox
from ..ui.functions_manager import Ui_FunctionsManager
from ..utils import (
    check_generated_function,
    generate_python_function,
    generate_python_function_globals,
    generate_python_variables,
)
from .python_highlighter import PythonHighlighter


class FunctionsManager(Ui_FunctionsManager, QtWidgets.QWidget):
    def __init__(self, definitions, channels=None, selected_definition="", global_variables="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.original_globals = global_variables

        self.channels = channels or {}

        definitions.pop("__global_variables__", None)

        for info in definitions.values():
            info["original_definition"] = info["current_definition"] = info["definition"] = info["definition"].replace(
                "\t", "    "
            )

        self.definitions = definitions

        self.globals_definition.setPlaceholderText(
            """The global variables definition is written as Python code.

Here is a minimalistic example:

PI = 3.1415 # float value
magic_number =  9  # integer value
initial_samples = [3, 5, 6]  # list of integers
"""
        )
        self.globals_definition.setPlainText(global_variables)
        self.globals_definition.setTabStopDistance(
            QtGui.QFontMetricsF(self.globals_definition.font()).horizontalAdvance(" ") * 4
        )

        p = self.globals_definition.palette()
        for active in (QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorGroup.Inactive):
            p.setColor(active, QtGui.QPalette.ColorRole.Base, QtGui.QColor("#131314"))
            p.setColor(active, QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#ffffff"))
            p.setColor(active, QtGui.QPalette.ColorRole.Text, QtGui.QColor("#ffffff"))
        self.globals_definition.setPalette(p)
        self.globals_highlighter = PythonHighlighter(self.globals_definition.document())

        self.function_definition.setPlaceholderText(
            """The virtual channel definition is written as a Python function.
Use the 'return' statement to return a value, otherwise 'None' will automatically be returned by the function.
The last function argument must be 't=0'.

Here is a minimalistic example:

def MyAverage(main_clock=0, p_FL=0, p_FR=0, p_RL=0, p_RR=0, vehicle_speed=0, t=0): 
    if main_clock > 5000:
        return 0
    else:
        avg = (p_FL + p_FR + p_RL + p_RR) / 4
        
        if avg > 20.5 and vehicle_speed < 100:
            return avg
        else:
            return avg + 9
"""
        )
        self.function_definition.setTabStopDistance(
            QtGui.QFontMetricsF(self.function_definition.font()).horizontalAdvance(" ") * 4
        )

        p = self.function_definition.palette()
        for active in (QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorGroup.Inactive):
            p.setColor(active, QtGui.QPalette.ColorRole.Base, QtGui.QColor("#131314"))
            p.setColor(active, QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#ffffff"))
            p.setColor(active, QtGui.QPalette.ColorRole.Text, QtGui.QColor("#ffffff"))
        self.function_definition.setPalette(p)

        self.func_highlighter = PythonHighlighter(self.function_definition.document())

        self.functions_list.minimal_menu = True
        self.functions_list.all_texts = True
        self.functions_list.placeholder_text = "Press the + button to add a new function definition"
        self.functions_list.user_editable = True
        self.functions_list.setAlternatingRowColors(True)
        self.functions_list.setIconSize(QtCore.QSize(16, 16))
        self.functions_list.itemsDeleted.connect(self.definitions_deleted)

        self.add_btn.clicked.connect(self.add_definition)
        self.check_syntax_btn.clicked.connect(self.check_syntax)
        self.check_globals_syntax_btn.clicked.connect(self.check_globals_syntax)
        self.erase_btn.clicked.connect(self.erase_definitions)
        self.export_btn.clicked.connect(self.export_definitions)
        self.import_btn.clicked.connect(self.import_definitions)
        self.store_btn.clicked.connect(self.store_definition)
        self.load_original_function_btn.clicked.connect(self.load_original_function_definition)
        self.load_original_globals_btn.clicked.connect(self.load_original_globals_definition)

        self.tabs.currentChanged.connect(self.tabs_changed)

        self.refresh_functions_list(selected_definition=selected_definition)

        self.functions_list.currentItemChanged.connect(self.definition_selection_changed)

        for button in (
            self.add_btn,
            self.check_syntax_btn,
            self.check_globals_syntax_btn,
            self.erase_btn,
            self.import_btn,
            self.export_btn,
        ):
            button.setDefault(False)
            button.setAutoDefault(False)

        self.showMaximized()

    def add_definition(self):
        if previous := self.functions_list.currentItem():
            name = previous.text()
            previous.setIcon(QtGui.QIcon())
            info = self.definitions[name]
            info["current_definition"] = self.function_definition.toPlainText().replace("\t", "    ")

            ok, _ = self.check_syntax(silent=True)
            if ok:
                if info["current_definition"] != info["definition"]:
                    previous.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxInformation))
                else:
                    previous.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_CommandLink))
            else:
                previous.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxCritical))

        counter = 1
        while True:
            name = f"Function{counter}"
            if name in self.definitions:
                counter += 1
            else:
                break

        self.definitions[name] = {
            "definition": f"def {name}(t=0):\n    return 0",
            "current_definition": f"def {name}(t=0):\n    return 0",
            "uuid": os.urandom(6).hex(),
        }

        self.refresh_functions_list(selected_definition=name)

    def check_globals_syntax(self, silent=False):
        trace = generate_python_variables(self.globals_definition.toPlainText())

        if trace:
            ErrorDialog(
                title="Global variables definition check",
                message="The syntax is not correct. The following error was found",
                trace=trace,
                parent=self,
            ).exec()

        elif not silent:
            MessageBox.information(
                self,
                "Global variables definition check",
                "The global variables definition appears to be correct.",
            )

        return not bool(trace)

    def check_syntax(self, silent=False, definition=None, globals_definition=None):
        if definition is None:
            definition = self.function_definition.toPlainText().replace("\t", "    ")

        if globals_definition is None:
            globals_definition = self.globals_definition.toPlainText()

        _globals = generate_python_function_globals()

        generate_python_variables(globals_definition, in_globals=_globals)

        for info in self.definitions.values():
            generate_python_function(info["definition"], in_globals=_globals)

        function_source = definition
        func, trace = generate_python_function(function_source, in_globals=_globals)

        return check_generated_function(func, trace, function_source, silent, parent=self)

    def definitions_deleted(self, deleted):
        count = self.functions_list.count()
        names = {self.functions_list.item(row).text() for row in range(count)}

        deleted = [name for name in self.definitions if name not in names]

        for name in deleted:
            del self.definitions[name]

        if not self.definitions:
            self.function_definition.setPlainText("")

    def definition_selection_changed(self, current, previous):
        if previous:
            name = previous.text()
            previous.setIcon(QtGui.QIcon())
            info = self.definitions[name]
            info["current_definition"] = self.function_definition.toPlainText().replace("\t", "    ")

            ok, _ = self.check_syntax(silent=True)
            if ok:
                if info["current_definition"] != info["definition"]:
                    previous.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxInformation))
                else:
                    previous.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_CommandLink))
            else:
                previous.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxCritical))

        if current:
            name = current.text()
            info = self.definitions[name]
            self.function_definition.setPlainText(info["current_definition"])

            ok, _ = self.check_syntax(silent=True)
            if ok:
                if info["current_definition"] != info["definition"]:
                    current.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxInformation))
                else:
                    current.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_CommandLink))
            else:
                current.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxCritical))

    def erase_definitions(self):
        self.functions_list.clear()
        self.definitions = {}
        self.function_definition.setPlainText("")

    def export_definitions(self, *args):
        self.refresh_definitions()

        for name, info in self.definitions.items():
            if info["current_definition"] != info["definition"]:
                result = MessageBox.question(
                    self, "Unsaved function definitions", "Do you want to review the functions before exporting them?"
                )
                if result == MessageBox.StandardButton.No:
                    break
                else:
                    for row in range(self.functions_list.count()):
                        item = self.functions_list.item(row)
                        if item.text() == name:
                            self.functions_list.setCurrentRow(row)
                            self.functions_list.scrollToItem(item)
                            break
                    return

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select functions definition export file",
            "",
            "Functions definition file (*.def)",
        )

        if file_name and Path(file_name).suffix.lower() == ".def":
            definitions = {name: info["current_definition"] for name, info in self.definitions.items()}
            definitions["__global_variables__"] = self.globals_definition.toPlainText().replace("\t", "    ")
            Path(file_name).write_text(json.dumps(definitions, indent=2))

    def import_definitions(self, *args):
        self.refresh_definitions()

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select functions definition file",
            "",
            "Functions definitions file (*.def)",
            "Functions definitions file (*.def)",
        )

        if file_name and Path(file_name).suffix.lower() == ".def":
            file_name = Path(file_name)

            with open(file_name) as infile:
                info = json.load(infile)

            self.globals_definition.setPlainText(info.pop("__global_variables__", "").replace("\t", "    "))

            for name, definition in info.items():
                definition = definition.replace("\t", "    ")

                if name in self.definitions:
                    self.definitions[name]["definition"] = self.definitions[name]["current_definition"] = definition
                else:
                    self.definitions[name] = {
                        "definition": definition,
                        "current_definition": definition,
                        "uuid": os.urandom(6).hex(),
                    }

            self.refresh_functions_list()

    def load_original_globals_definition(self):
        self.globals_definition.setPlainText(self.original_globals)

    def load_original_function_definition(self):
        item = self.functions_list.currentItem()
        if not item:
            return

        current_name = item.text()
        info = self.definitions.pop(current_name)
        original_definition = info["original_definition"]
        info["current_definition"] = original_definition

        self.function_definition.setPlainText(original_definition)

        self.definitions[current_name] = info

        self.refresh_functions_list()

    def refresh_definitions(self):
        _globals = generate_python_function_globals()

        generate_python_variables(self.globals_definition.toPlainText(), in_globals=_globals)

        for info in self.definitions.values():
            generate_python_function(info["current_definition"], in_globals=_globals)

        function_source = self.function_definition.toPlainText().replace("\t", "    ")
        func, trace = generate_python_function(function_source, in_globals=_globals)

        if func is not None:
            name = func.__name__
            if name in self.definitions:
                self.definitions[name]["current_definition"] = self.function_definition.toPlainText().replace(
                    "\t", "    "
                )

    def refresh_functions_list(self, selected_definition=""):
        self.functions_list.blockSignals(True)
        self.functions_list.clear()

        if self.definitions:
            items = []
            names = natsorted(self.definitions)

            for name in names:
                info = self.definitions[name]
                ok, func = self.check_syntax(silent=True, definition=info["definition"])

                if ok:
                    if info["current_definition"] != info["definition"]:
                        items.append(
                            QtWidgets.QListWidgetItem(
                                self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxInformation),
                                name,
                                self.functions_list,
                            )
                        )
                    else:
                        items.append(
                            QtWidgets.QListWidgetItem(
                                self.style().standardIcon(QtWidgets.QStyle.SP_CommandLink),
                                name,
                                self.functions_list,
                            )
                        )
                else:
                    items.append(
                        QtWidgets.QListWidgetItem(
                            self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxCritical),
                            name,
                            self.functions_list,
                        )
                    )

            self.functions_list.blockSignals(False)

            if selected_definition:
                if selected_definition in names:
                    self.functions_list.setCurrentRow(names.index(selected_definition))
                elif names:
                    self.functions_list.setCurrentRow(0)
        else:
            self.functions_list.blockSignals(False)

    def store_definition(self, *args):
        item = self.functions_list.currentItem()
        if not item:
            return

        current_name = item.text()
        info = self.definitions.pop(current_name)
        info["definition"] = info["current_definition"] = self.function_definition.toPlainText().replace("\t", "    ")

        ok, func = self.check_syntax(silent=True)
        if ok:
            func_name = func.__name__

            if current_name != func_name and func_name in self.definitions:
                MessageBox.information(
                    self,
                    "Invalid function name",
                    f'The name "{func_name}" is already given to another function.\nThe function names must be unique',
                )

            else:
                current_name = func_name

        self.definitions[current_name] = info

        self.refresh_functions_list(selected_definition=current_name)

    def tabs_changed(self, index):
        self.check_globals_syntax(silent=True)

import json
import os
from pathlib import Path

from natsort import natsorted
from PySide6 import QtGui, QtWidgets

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

        self.channels = channels or {}

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

        self.functions_list.currentItemChanged.connect(self.definition_selection_changed)
        self.functions_list.itemsDeleted.connect(self.definitions_deleted)

        self.add_btn.clicked.connect(self.add_definition)
        self.check_syntax_btn.clicked.connect(self.check_syntax)
        self.check_globals_syntax_btn.clicked.connect(self.check_globals_syntax)
        self.erase_btn.clicked.connect(self.erase_definitions)
        self.export_btn.clicked.connect(self.export_definitions)
        self.import_btn.clicked.connect(self.import_definitions)
        self.store_btn.clicked.connect(self.store_definition)

        self.tabs.currentChanged.connect(self.tabs_changed)

        if self.definitions:
            self.functions_list.clear()
            names = natsorted(self.definitions)
            self.functions_list.addItems(names)
            if selected_definition in names:
                self.functions_list.setCurrentRow(names.index(selected_definition))
            elif names:
                self.functions_list.setCurrentRow(0)

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
        counter = 1
        while True:
            name = f"Function{counter}"
            if name in self.definitions:
                counter += 1
            else:
                break

        self.definitions[name] = {
            "definition": f"def {name}(t=0):\n    return 0",
            "uuid": os.urandom(6).hex(),
        }

        self.functions_list.clear()
        names = natsorted(self.definitions)
        self.functions_list.addItems(names)

        row = names.index(name)
        self.functions_list.setCurrentRow(row + 1)

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

    def check_syntax(self, silent=False):
        _globals = generate_python_function_globals()

        generate_python_variables(self.globals_definition.toPlainText(), in_globals=_globals)

        for info in self.definitions.values():
            generate_python_function(info["definition"], in_globals=_globals)

        function_source = self.function_definition.toPlainText().replace("\t", "    ")
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
        if current:
            name = current.text()

            definition = self.definitions[name]
            self.function_definition.setPlainText(definition["definition"])

    def erase_definitions(self):
        self.functions_list.clear()
        self.definitions = {}
        self.function_definition.setPlainText("")

    def export_definitions(self, *args):
        self.refresh_definitions()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select functions definition export file",
            "",
            "Functions definition file (*.def)",
        )

        if file_name and Path(file_name).suffix.lower() == ".def":
            definitions = {name: info["definition"] for name, info in self.definitions.items()}
            definitions["__global_variables__"] = self.globals_definition.toPlainText()
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

            for name, definition in info.items():
                if name == "__global_variables__":
                    self.globals_definition.setPlainText(definition)

                if name in self.definitions:
                    self.definitions[name]["definition"] = definition
                else:
                    self.definitions[name] = {
                        "definition": definition,
                        "uuid": os.urandom(6).hex(),
                    }

            self.functions_list.clear()
            names = natsorted(self.definitions)
            self.functions_list.addItems(names)
            if names:
                self.functions_list.setCurrentRow(0)

    def refresh_definitions(self):
        _globals = generate_python_function_globals()

        generate_python_variables(self.globals_definition.toPlainText(), in_globals=_globals)

        for info in self.definitions.values():
            generate_python_function(info["definition"], in_globals=_globals)

        function_source = self.function_definition.toPlainText().replace("\t", "    ")
        func, trace = generate_python_function(function_source, in_globals=_globals)

        if func is not None:
            name = func.__name__
            if name in self.definitions:
                self.definitions[name]["definition"] = self.function_definition.toPlainText()

    def store_definition(self, *args):
        ok, func = self.check_syntax(silent=True)
        if ok:
            item = self.functions_list.currentItem()
            name = func.__name__

            if name != item.text() and name in self.definitions:
                MessageBox.information(
                    self,
                    "Invalid function name",
                    f'The name "{name}" is already given to another function.\n' "The function names must be unique",
                )
                return

            info = self.definitions.pop(item.text())
            info["definition"] = self.function_definition.toPlainText()
            self.definitions[func.__name__] = info

            self.functions_list.clear()
            names = natsorted(self.definitions)
            self.functions_list.addItems(names)

            row = names.index(name)
            self.functions_list.setCurrentRow(row)

    def tabs_changed(self, index):
        self.check_globals_syntax(silent=True)

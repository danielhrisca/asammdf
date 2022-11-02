# -*- coding: utf-8 -*-
import os
import string
from functools import partial
from traceback import format_exc

import numpy as np
from natsort import natsorted
from PySide6 import QtCore, QtGui, QtWidgets

from ..ui.functions_manager import Ui_FunctionsManager
from ..utils import ErrorDialog, generate_python_function
from ..dialogs.simple_search import SimpleSearch


class FunctionsManager(Ui_FunctionsManager, QtWidgets.QWidget):

    def __init__(self, definitions, channels=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.channels = channels or {}

        self.definitions = definitions

        self.function_definition.setPlaceholderText(
            """The virtual channel definition is written as a Python function.
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
        self.function_definition.setTabStopDistance(
            QtGui.QFontMetricsF(self.function_definition.font()).horizontalAdvance(" ")
            * 4
        )

        self.functions_list.minimal_menu = True
        self.functions_list.all_texts = True
        self.functions_list.placeholder_text = (
            "Press the + button to add a new virtual channel"
        )
        self.functions_list.user_editable = True
        self.functions_list.setAlternatingRowColors(True)

        self.functions_list.currentItemChanged.connect(self.definition_selection_changed)
        self.functions_list.itemsDeleted.connect(self.definitions_deleted)

        self.function_name.editingFinished.connect(self.function_name_edited)

        self.add_btn.clicked.connect(self.add_definition)
        self.check_syntax_btn.clicked.connect(self.check_syntax)
        self.erase_btn.clicked.connect(self.erase_definitions)
        self.search_btn.clicked.connect(self.search)

        if self.definitions:
            self.functions_list.clear()
            names = natsorted(self.definitions)
            self.functions_list.addItems(names)
            self.functions_list.setCurrentRow(0)

        for button in (self.add_btn, self.check_syntax_btn, self.erase_btn, self.import_btn, self.export_btn, self.search_btn):
            button.setDefault(False)
            button.setAutoDefault(False)

    def add_definition(self):
        counter = 1
        while True:
            name = f"Function{counter}"
            if name in self.definitions:
                counter += 1
            else:
                break

        self.definitions[name] = {
            "definition": "",
            "uuid": os.urandom(6).hex(),
        }

        self.functions_list.clear()
        names = natsorted(self.definitions)
        self.functions_list.addItems(names)

        row = names.index(name)
        self.functions_list.setCurrentRow(row)

    def function_name_edited(self):
        item = self.functions_list.currentItem()
        if not item:
            return

        current_name = item.text()
        new_name = self.function_name.text().strip()

        if not new_name:
            QtWidgets.QMessageBox.information(
                self,
                "Invalid function name",
                "The function name cannot be an empty string",
            )
            self.function_name.setText(current_name)

        elif new_name != current_name and new_name in self.definitions:
            QtWidgets.QMessageBox.information(
                self,
                "Invalid function name",
                f'The name "{new_name}" is already given to another function.\n'
                "The function names must be unique",
            )

            self.function_name.setText(current_name)

        else:
            item.setText(new_name)
            del self.definitions[current_name]
            self.definitions[new_name] = {
                "definition": self.function_definition.toPlainText(),
                "name": new_name,
            }

            self.functions_list.clear()
            names = natsorted(self.definitions)
            self.functions_list.addItems(names)

            row = names.index(new_name)
            self.functions_list.setCurrentRow(row)

    def check_syntax(self):

        allowed_chars = string.ascii_letters + string.digits
        (
            func,
            arg_names,
            get_data_args,
            func_name,
            trace,
            function_source,
        ) = generate_python_function(
            self.function_definition.toPlainText().replace("\t", "    "),
            "".join(
                ch if ch in allowed_chars else "_"
                for ch in self.function_name.text().strip()
            ),
        )

        if trace is not None:
            ErrorDialog(
                self.logger,
                title="Function definition check",
                message="The syntax is not correct. The following error was found",
                trace=f"{trace}\n\nin the function\n\n{function_source}",
            ).exec()
            return

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
                self.logger,
                title="Function definition check",
                message="The syntax is not correct. The following error was found",
                trace=f"{trace}\n\nin the function\n\n{function_source}",
            ).exec()

        else:
            QtWidgets.QMessageBox.information(
                self,
                "Function definition check",
                "The function definition appears to be correct.",
            )

    def definitions_deleted(self, deleted):
        count = self.functions_list.count()
        names = set(self.functions_list.item(row).text() for row in range(count))

        deleted = [name for name in self.definitions if name not in names]

        for name in deleted:
            del self.definitions[name]

        if not self.definitions:
            self.function_name.setText("")
            self.function_definition.setPlainText("")

    def definition_selection_changed(self, current, previous):

        if previous:
            previous_uuid = self.definitions[previous.text()]["uuid"]
            name = self.function_name.text()

            self.definitions[name] = {
                "definition": self.function_definition.toPlainText(),
                "uuid": previous_uuid,
            }

        if current:
            name = current.text()

            definition = self.definitions[name]
            self.function_name.setText("")
            self.function_definition.setPlainText(definition['definition'])
            self.function_name.setText(name)

    def erase_definitions(self):
        self.functions_list.clear()
        self.definitions = {}
        self.function_name.setText("")
        self.function_definition.setPlainText("")

    def refresh_definitions(self):
        name = self.function_name.text().strip()

        self.definitions[name]["definition"] = self.function_definition.toPlainText()

    def search(self):
        dlg = SimpleSearch(self.channels, parent=self)
        dlg.setModal(True)
        dlg.exec()
        result = dlg.result
        if result:
            for name in result:
                self.function_definition.insertPlainText(f" {{{{{name}}}}}")

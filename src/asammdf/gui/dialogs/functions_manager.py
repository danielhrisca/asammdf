from copy import deepcopy
import os

from PySide6 import QtCore, QtWidgets

from ..widgets.functions_manager import FunctionsManager


class FunctionsManagerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        definitions,
        channels=None,
        selected_definition="",
        prefix="",
        *args,
        **kwargs,
    ):
        definitions = definitions or {}
        definitions = {
            name: {
                "definition": definition,
                "uuid": os.urandom(6).hex(),
            }
            for name, definition in definitions.items()
        }

        self.original_definitions = {}
        self.modified_definitions = {}

        for name, info in definitions.items():
            self.original_definitions[info["uuid"]] = {
                "name": name,
                "definition": info["definition"],
            }

        super().__init__(*args, **kwargs)

        self.setObjectName("FunctionsManagerDialog")
        self.resize(404, 294)
        self.setSizeGripEnabled(True)
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.widget = FunctionsManager(deepcopy(definitions), channels, selected_definition)

        self.verticalLayout.addWidget(self.widget)

        self.horLayout = QtWidgets.QHBoxLayout(self)

        spacer = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.horLayout.addSpacerItem(spacer)
        self.horLayout.addWidget(self.apply_btn)
        self.horLayout.addWidget(self.cancel_btn)

        self.verticalLayout.addLayout(self.horLayout)

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)
        self.pressed_button = "cancel"

        if prefix:
            self.setWindowTitle(f"{prefix} - Functions Manager")
        else:
            self.setWindowTitle("Functions Manager")

    def apply(self, *args):
        self.pressed_button = "apply"
        self.modified_definitions = {}

        self.widget.refresh_definitions()

        for name, info in self.widget.definitions.items():
            self.modified_definitions[info["uuid"]] = {
                "name": name,
                "definition": info["definition"],
            }

        self.close()

    def cancel(self, *args):
        self.pressed_button = "cancel"
        self.close()

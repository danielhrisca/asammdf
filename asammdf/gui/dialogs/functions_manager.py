# -*- coding: utf-8 -*-
from traceback import format_exc

from PySide6 import QtCore, QtWidgets

from ..widgets.functions_manager import FunctionsManager


class FunctionsManagerDialog(QtWidgets.QDialog):

    def __init__(
        self,
        definitions,
        channels=None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.setObjectName("FunctionsManagerDialog")
        self.resize(404, 294)
        self.setSizeGripEnabled(True)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.widget = FunctionsManager(definitions, channels)

        self.verticalLayout.addWidget(self.widget)

        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        self.verticalLayout.addWidget(self.buttonBox)

        self.buttonBox.accepted.connect(self.apply)
        self.buttonBox.rejected.connect(self.dismiss)

        self.setWindowTitle("Functions Manager")

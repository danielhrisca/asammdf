# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets, QtGui

from ..ui import resource_rc as resource_rc
from ..ui.error_dialog import Ui_ErrorDialog
from ..widgets.collapsiblebox import CollapsibleBox


class ErrorDialog(Ui_ErrorDialog, QtWidgets.QDialog):
    def __init__(
        self, title, message, trace, *args, **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.error_box = CollapsibleBox(title="Full error traceback")
        self.layout.insertWidget(0, self.error_box)

        lay = QtWidgets.QVBoxLayout()
        label = QtWidgets.QTextEdit(trace)
        label.setReadOnly(True)
        lay.addWidget(label)

        self.error_box.setContentLayout(lay)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/error.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        self.setWindowIcon(icon)

        self.setWindowTitle(title)

        self.error_message.setText(message)

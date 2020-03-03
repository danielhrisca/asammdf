# -*- coding: utf-8 -*-
from time import sleep
from threading import Thread

from PyQt5 import QtWidgets, QtGui, QtCore

from ..ui import resource_rc as resource_rc
from ..ui.error_dialog import Ui_ErrorDialog
from ..widgets.collapsiblebox import CollapsibleBox


class ErrorDialog(Ui_ErrorDialog, QtWidgets.QDialog):
    def __init__(self, title, message, trace, *args, **kwargs):
        if 'remote' in kwargs:
            remote = kwargs.pop('remote')
            if 'timeout' in kwargs:
                timeout = kwargs.pop('timeout')
            else:
                timeout = 60
        else:
            remote = False

        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.trace = QtWidgets.QTextEdit()
        self.layout.insertWidget(0, self.trace)
        self.trace.setText(trace)
        self.trace.setReadOnly(True)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/error.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )

        self.setWindowIcon(icon)

        self.setWindowTitle(title)

        self.error_message.setText(message)

        self.copy_to_clipboard_btn.clicked.connect(self.copy_to_clipboard)

        self.layout.setStretch(0, 1)
        self.layout.setStretch(1, 0)
        self.layout.setStretch(2, 0)

        if remote:
            self._timeout = timeout

            self._thread = Thread(target=self.count_down, args=())
            self._thread.start()

            QtCore.QTimer.singleShot(self._timeout * 1000, self.close)

    def copy_to_clipboard(self, event):
        text = (
            f"Error: {self.error_message.text()}\n\nDetails: {self.trace.toPlainText()}"
        )
        QtWidgets.QApplication.instance().clipboard().setText(text)


    def count_down(self):
        while self._timeout > 0:
            sleep(1)
            self._timeout -= 1
            self.status.setText(f"This window will close in {self._timeout:02}s")


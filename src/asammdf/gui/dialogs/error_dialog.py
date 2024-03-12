import os

from PySide6 import QtCore, QtGui, QtWidgets

from ..ui.error_dialog import Ui_ErrorDialog


class ErrorDialog(Ui_ErrorDialog, QtWidgets.QDialog):
    def __init__(self, title, message, trace, *args, **kwargs):
        remote = kwargs.pop("remote", False)
        timeout = kwargs.pop("timeout", int(os.environ.get("ASAMMDF_ERROR_DIALOG_TIMEOUT", 90)))
        logger = kwargs.pop("logger", None)

        super().__init__(*args, **kwargs)
        self.setupUi(self)

        if logger is not None:
            logger.error(f">>> {title}\n{message}\n{trace}")
        print(trace)

        self.trace = QtWidgets.QTextEdit()

        families = QtGui.QFontDatabase().families()
        for family in (
            "Consolas",
            "Liberation Mono",
            "DejaVu Sans Mono",
            "Droid Sans Mono",
            "Liberation Mono",
            "Roboto Mono",
            "Monaco",
            "Courier",
        ):
            if family in families:
                break

        font = QtGui.QFont(family)
        self.trace.setFont(font)
        self.layout.insertWidget(2, self.trace)
        self.trace.hide()
        self.trace.setText(trace)
        self.trace.setReadOnly(True)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/error.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        self.setWindowIcon(icon)

        self.setWindowTitle(title)

        self.error_message.setText(message)

        self.copy_to_clipboard_btn.clicked.connect(self.copy_to_clipboard)
        self.show_trace_btn.clicked.connect(self.show_trace)

        self.layout.setStretch(0, 0)
        self.layout.setStretch(1, 0)
        self.layout.setStretch(2, 1)

        self._timeout = timeout

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.count_down)

        if timeout > 0:
            self.status.setText(f"This window will be closed in {self._timeout}s\nAbort the countdown - [F1]")
            self.timer.start(1000)

    def copy_to_clipboard(self, event):
        text = f"Error: {self.error_message.text()}\n\nDetails: {self.trace.toPlainText()}"
        QtWidgets.QApplication.instance().clipboard().setText(text)

    def count_down(self):
        if self._timeout > 0:
            self._timeout -= 1
            self.status.setText(f"This window will be closed in {self._timeout}s\nAbort the countdown - [F1]")
        else:
            self.close()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_F1:
            self.timer.stop()
            self.status.clear()
            event.accept()
        else:
            super().keyPressEvent(event)

    def show_trace(self, event):
        if self.trace.isHidden():
            self.trace.show()
            self.show_trace_btn.setText("Hide error trace")
        else:
            self.trace.hide()
            self.show_trace_btn.setText("Show error trace")

from PySide6 import QtCore, QtWidgets

DEFAULT_TIMEOUT = 60


class MessageBox(QtWidgets.QMessageBox):
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.pop("timeout", DEFAULT_TIMEOUT)
        informative_text = kwargs.pop("informative_text", "")
        detailed_text = kwargs.pop("detailed_text", "")
        escapeButton = kwargs.pop("escapeButton", None)
        defaultButton = kwargs.pop("defaultButton", None)
        markdown = kwargs.pop("markdown", False)

        super().__init__(*args, **kwargs)

        self.original_text = self.text()
        if markdown:
            self.setTextFormat(QtCore.Qt.TextFormat.MarkdownText)

        if defaultButton is not None:
            self.setDefaultButton(defaultButton)

        if self.defaultButton() is not None:
            if markdown:
                self.setText(
                    f"""{self.original_text}

* * *

This message will be closed in {self.timeout}s
* Default button - [{self.defaultButton().text().strip('&')}]
* Abort the countdown - [F1]"""
                )
            else:
                self.setText(
                    f"{self.original_text}\n\nThis message will be closed in {self.timeout}s\n"
                    f'Default button - [{self.defaultButton().text().strip("&")}]\n'
                    "Abort the countdown - [F1]"
                )
        else:
            if markdown:
                self.setText(
                    f"""{self.original_text}

* * *

This message will be closed in {self.timeout}s
* Abort the countdown - [F1]"""
                )

            else:
                self.setText(
                    f"{self.original_text}\n\nThis message will be closed in {self.timeout}s\n"
                    "Abort the countdown - [F1]"
                )

        if informative_text:
            self.setInformativeText(informative_text)

        if detailed_text:
            self.setDetailedText(detailed_text)

        if escapeButton is not None:
            self.setEscapeButton(escapeButton)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_F1:
            self.timer.stop()
            self.setText(self.original_text)
            event.accept()
        else:
            super().keyPressEvent(event)

    def tick(self):
        self.timeout -= 1
        default = self.defaultButton()

        if self.timeout <= 0:
            self.timer.stop()
            if default is not None:
                default.animateClick()
            else:
                self.done(0)
        else:
            if self.defaultButton() is not None:
                if self.textFormat() == QtCore.Qt.TextFormat.MarkdownText:
                    self.setText(
                        f"""{self.original_text}

* * *

This message will be closed in {self.timeout}s
* Default button - [{self.defaultButton().text().strip('&')}]
* Abort the countdown - [F1]"""
                    )
                else:
                    self.setText(
                        f"{self.original_text}\n\nThis message will be closed in {self.timeout}s\n"
                        f'Default button - [{self.defaultButton().text().strip("&")}]\n'
                        "Abort the countdown - [F1]"
                    )
            else:
                if self.textFormat() == QtCore.Qt.TextFormat.MarkdownText:
                    self.setText(
                        f"""{self.original_text}

* * *

This message will be closed in {self.timeout}s
* Abort the countdown - [F1]"""
                    )

                else:
                    self.setText(
                        f"{self.original_text}\n\nThis message will be closed in {self.timeout}s\n"
                        "Abort the countdown - [F1]"
                    )

    @classmethod
    def about(
        cls,
        parent,
        title,
        text,
        buttons=QtWidgets.QMessageBox.StandardButton.Ok,
        defaultButton=QtWidgets.QMessageBox.StandardButton.Ok,
        escapeButton=QtWidgets.QMessageBox.StandardButton.Ok,
        timeout=DEFAULT_TIMEOUT,
        markdown=False,
        informative_text="",
        detailed_text="",
    ):
        msg = cls(
            QtWidgets.QMessageBox.Icon.NoIcon,
            title,
            text,
            buttons,
            parent,
            timeout=timeout,
            defaultButton=defaultButton,
            escapeButton=escapeButton,
            markdown=markdown,
            informative_text=informative_text,
            detailed_text=detailed_text,
        )
        return msg.exec()

    @classmethod
    def critical(
        cls,
        parent,
        title,
        text,
        buttons=QtWidgets.QMessageBox.StandardButton.Ok,
        defaultButton=QtWidgets.QMessageBox.StandardButton.Ok,
        escapeButton=QtWidgets.QMessageBox.StandardButton.Ok,
        timeout=DEFAULT_TIMEOUT,
        markdown=False,
        informative_text="",
        detailed_text="",
    ):
        msg = cls(
            QtWidgets.QMessageBox.Icon.Critical,
            title,
            text,
            buttons,
            parent,
            timeout=timeout,
            defaultButton=defaultButton,
            escapeButton=escapeButton,
            markdown=markdown,
            informative_text=informative_text,
            detailed_text=detailed_text,
        )

        return msg.exec()

    @classmethod
    def information(
        cls,
        parent,
        title,
        text,
        buttons=QtWidgets.QMessageBox.StandardButton.Ok,
        defaultButton=QtWidgets.QMessageBox.StandardButton.Ok,
        escapeButton=QtWidgets.QMessageBox.StandardButton.Ok,
        timeout=DEFAULT_TIMEOUT,
        markdown=False,
        informative_text="",
        detailed_text="",
    ):
        msg = cls(
            QtWidgets.QMessageBox.Icon.Information,
            title,
            text,
            buttons,
            parent,
            timeout=timeout,
            defaultButton=defaultButton,
            escapeButton=escapeButton,
            markdown=markdown,
            informative_text=informative_text,
            detailed_text=detailed_text,
        )
        return msg.exec()

    @classmethod
    def question(
        cls,
        parent,
        title,
        text,
        buttons=QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        defaultButton=QtWidgets.QMessageBox.StandardButton.No,
        escapeButton=QtWidgets.QMessageBox.StandardButton.No,
        timeout=DEFAULT_TIMEOUT,
        markdown=False,
        informative_text="",
        detailed_text="",
    ):
        msg = cls(
            QtWidgets.QMessageBox.Icon.Question,
            title,
            text,
            buttons,
            parent,
            timeout=timeout,
            defaultButton=defaultButton,
            escapeButton=escapeButton,
            markdown=markdown,
            informative_text=informative_text,
            detailed_text=detailed_text,
        )
        return msg.exec()

    @classmethod
    def warning(
        cls,
        parent,
        title,
        text,
        buttons=QtWidgets.QMessageBox.StandardButton.Ok,
        defaultButton=QtWidgets.QMessageBox.StandardButton.Ok,
        escapeButton=QtWidgets.QMessageBox.StandardButton.Ok,
        timeout=DEFAULT_TIMEOUT,
        markdown=False,
        informative_text="",
        detailed_text="",
    ):
        msg = cls(
            QtWidgets.QMessageBox.Icon.Warning,
            title,
            text,
            buttons,
            parent,
            timeout=timeout,
            defaultButton=defaultButton,
            escapeButton=escapeButton,
            markdown=markdown,
            informative_text=informative_text,
            detailed_text=detailed_text,
        )
        return msg.exec()

# -*- coding: utf-8 -*-
from pathlib import Path

from PySide6 import QtWidgets

from ...blocks.utils import extract_encryption_information
from ..ui import resource_rc
from ..ui.attachment import Ui_Attachment


class Attachment(Ui_Attachment, QtWidgets.QWidget):
    def __init__(self, index, mdf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.extract_btn.clicked.connect(self.extract)
        self.mdf = mdf
        self.index = index

    def extract(self, event=None):
        attachment = self.mdf.attachments[self.index]
        encryption_info = extract_encryption_information(attachment.comment)
        password = None
        if encryption_info.get("encrypted", False) and self.mdf._password is None:
            text, ok = QtWidgets.QInputDialog.getText(
                self,
                "Attachment password",
                "The attachment is encrypted. Please provide the password:",
                QtWidgets.QLineEdit.Password,
            )
            if ok and text:
                password = text

        data, file_path, md5_sum = self.mdf.extract_attachment(
            self.index, password=password
        )

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select extracted file",
            str(file_path),
            "All files (*.*)",
            "All files (*.*)",
        )
        if file_name:
            file_name = Path(file_name)
            file_name.write_bytes(data)

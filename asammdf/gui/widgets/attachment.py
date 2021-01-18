# -*- coding: utf-8 -*-
from hashlib import md5
from pathlib import Path

from PyQt5 import QtWidgets

from ...blocks import v4_constants as v4c
from ..ui import resource_rc as resource_rc
from ..ui.attachment import Ui_Attachment


class Attachment(Ui_Attachment, QtWidgets.QWidget):
    def __init__(self, attachment, decryption_function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.extract_btn.clicked.connect(self.extract)
        self.attachment = attachment
        self.decryption_function = decryption_function

    def extract(self, event=None):
        flags = self.attachment.flags
        file_path = Path(self.attachment.file_name).resolve()

        if flags & v4c.FLAG_AT_EMBEDDED:
            data = self.attachment.extract()
            if flags & v4c.FLAG_AT_ENCRYPTED and self.decryption_function is not None:
                try:
                    data = self.decryption_function(data)
                except:
                    pass
        else:
            if not file_path.exists():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Can't extract attachment",
                    f"The attachment <{file_path}> does not exist",
                )
                return

            if flags & v4c.FLAG_AT_MD5_VALID:
                data = file_path.read_bytes()

                md5_worker = md5()
                md5_worker.update(data)
                md5_sum = md5_worker.digest()

                if self.attachment.md5_sum != md5_sum:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Can't extract attachment - wrong checksum",
                        f'ATBLOCK md5sum="{self.attachment["md5_sum"]}" '
                        f"and external attachment data <{file_path}> has "
                        f'md5sum="{md5_sum}"',
                    )
                    return
            else:
                data = file_path.read_bytes()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select extracted file", "", "All files (*.*)", "All files (*.*)"
        )
        if file_name:
            file_name = Path(file_name)
            file_name.write_bytes(data)

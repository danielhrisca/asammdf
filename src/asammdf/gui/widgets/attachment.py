from pathlib import Path
from tempfile import gettempdir
import threading

from PySide6 import QtGui, QtWidgets

from ...blocks.utils import extract_encryption_information
from ...blocks.v4_constants import FLAG_AT_TO_STRING
from ..ui.attachment import Ui_Attachment

try:
    import sounddevice as sd
    import soundfile as sf

    current_frame = 0
except:
    current_frame = None


class Attachment(Ui_Attachment, QtWidgets.QWidget):
    def __init__(self, index, attachment_block, file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.extract_btn.clicked.connect(self.extract)
        self.file = file
        self.index = index

        self.number.setText(f"{index + 1}.")

        fields = []

        field = QtWidgets.QTreeWidgetItem()
        field.setText(0, "ATBLOCK address")
        field.setText(1, f"0x{attachment_block.address:X}")
        fields.append(field)

        field = QtWidgets.QTreeWidgetItem()
        field.setText(0, "File name")
        field.setText(1, str(attachment_block.file_name))
        fields.append(field)

        field = QtWidgets.QTreeWidgetItem()
        field.setText(0, "MIME type")
        field.setText(1, attachment_block.mime)
        fields.append(field)

        field = QtWidgets.QTreeWidgetItem()
        field.setText(0, "Comment")
        field.setText(1, attachment_block.comment)
        fields.append(field)

        field = QtWidgets.QTreeWidgetItem()
        field.setText(0, "Flags")
        if attachment_block.flags:
            flags = []
            for flag, string in FLAG_AT_TO_STRING.items():
                if attachment_block.flags & flag:
                    flags.append(string)
            text = f"{attachment_block.flags} [0x{attachment_block.flags:X}= {', '.join(flags)}]"
        else:
            text = "0"
        field.setText(1, text)
        fields.append(field)

        field = QtWidgets.QTreeWidgetItem()
        field.setText(0, "MD5 sum")
        field.setText(1, attachment_block.md5_sum.hex().upper())
        fields.append(field)

        size = attachment_block.original_size
        if size <= 1 << 10:
            text = f"{size} B"
        elif size <= 1 << 20:
            text = f"{size / 1024:.1f} KB"
        elif size <= 1 << 30:
            text = f"{size / 1024 / 1024:.1f} MB"
        else:
            text = f"{size / 1024 / 1024 / 1024:.1f} GB"

        field = QtWidgets.QTreeWidgetItem()
        field.setText(0, "Size")
        field.setText(1, text)
        fields.append(field)

        self.fields.addTopLevelItems(fields)

        self.audio_comment = None
        self.audio_progress = None

        if current_frame is not None:
            if attachment_block.file_name == "user_audio_comment.ogg" and attachment_block.mime == r"audio/ogg":
                self.audio_comment = attachment_block.extract()
                field = QtWidgets.QTreeWidgetItem()
                field.setText(0, "Audio comment")
                widget = QtWidgets.QWidget()
                layout = QtWidgets.QHBoxLayout()
                widget.setLayout(layout)

                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(":/play.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
                if icon.isNull():
                    play_audio_comment_btn = QtWidgets.QPushButton("Play comment")
                else:
                    play_audio_comment_btn = QtWidgets.QPushButton(widget)
                    play_audio_comment_btn.setIcon(icon)

                play_audio_comment_btn.clicked.connect(self.play_audio_comment)
                layout.addWidget(play_audio_comment_btn)

                self.audio_progress = QtWidgets.QProgressBar(widget)
                self.audio_progress.setValue(0)
                self.audio_progress.setTextVisible(False)
                layout.addWidget(self.audio_progress)

                layout.setStretch(1, 1)
                layout.setStretch(0, 0)

                self.fields.addTopLevelItem(field)
                self.fields.setItemWidget(field, 1, widget)

    def extract(self, event=None):
        attachment = self.file.mdf.attachments[self.index]
        encryption_info = extract_encryption_information(attachment.comment)
        password = None
        if encryption_info.get("encrypted", False) and self.file.mdf._mdf._password is None:
            text, ok = QtWidgets.QInputDialog.getText(
                self,
                "Attachment password",
                "The attachment is encrypted. Please provide the password:",
                QtWidgets.QLineEdit.EchoMode.Password,
            )
            if ok and text:
                password = text

        data, file_path, md5_sum = self.file.mdf.extract_attachment(self.index, password=password)

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

    def play_audio_comment(self):
        if self.audio_comment is None:
            return

        global current_frame

        tmp = Path(gettempdir()) / "daxil_audio_comment.ogg"
        tmp.write_bytes(self.audio_comment)

        data, fs = sf.read(tmp)
        size = len(data)

        current_frame = 0
        playback_finished = threading.Event()

        def callback(outdata, frames, time, status):
            global current_frame
            chunksize = min(len(data) - current_frame, frames)
            outdata[:chunksize] = data[current_frame : current_frame + chunksize]
            if chunksize < frames:
                outdata[chunksize:] = 0
                raise sd.CallbackStop()
            current_frame += chunksize
            self.audio_progress.setValue(int(100 * current_frame / size))

        stream = sd.OutputStream(
            samplerate=fs, channels=data.shape[1], callback=callback, finished_callback=playback_finished.set
        )
        with stream:
            playback_finished.wait()

        self.audio_progress.setValue(0)

        tmp.unlink()

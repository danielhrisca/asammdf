import numpy as np
from PySide6 import QtWidgets

from ..ui.vtt_widget import Ui_VTT_Widget


class VTTWidget(Ui_VTT_Widget, QtWidgets.QWidget):
    def __init__(
        self,
        value=0,
        text="",
        conversion=None,
        mode="text",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.text.setText(text)

        self.value.setMaximum(np.inf)
        self.value.setMinimum(-np.inf)
        self.value.setValue(value)

        self.conversion = conversion

        self.mode_switch.currentIndexChanged.connect(self.mode.setCurrentIndex)

        if mode == "text":
            self.mode_switch.setCurrentIndex(0)
        else:
            self.mode_switch.setCurrentIndex(1)

        self.conversion_btn.clicked.connect(self.edit_conversion)

    def edit_conversion(self):
        dlg = ConversionEditor(f"Raw={self.value.value()} referenced", self.conversion, parent=self)
        dlg.exec_()
        if dlg.pressed_button == "apply":
            self.conversion = dlg.conversion()

    def reference(self):
        if self.mode.currentIndex() == 0:
            return self.text.text().strip().encode("utf-8")
        else:
            return self.conversion


from ..dialogs.conversion_editor import ConversionEditor

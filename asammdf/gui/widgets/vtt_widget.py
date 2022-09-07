# -*- coding: utf-8 -*-
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..ui import resource_rc
from ..ui.vtt_widget import Ui_VTT_Widget


class VTTWidget(Ui_VTT_Widget, QtWidgets.QWidget):
    def __init__(
        self,
        value=0,
        text="",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.text.setText(text)

        self.value.setMaximum(np.inf)
        self.value.setMinimum(-np.inf)
        self.value.setValue(value)

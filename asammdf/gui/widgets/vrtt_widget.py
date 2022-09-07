# -*- coding: utf-8 -*-
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..ui import resource_rc
from ..ui.vrtt_widget import Ui_VRTT_Widget


class VRTTWidget(Ui_VRTT_Widget, QtWidgets.QWidget):
    def __init__(
        self,
        lower=0,
        upper=0,
        text="",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.text.setText(text)

        self.lower.setMaximum(np.inf)
        self.lower.setMinimum(-np.inf)
        self.lower.setValue(lower)

        self.upper.setMaximum(np.inf)
        self.upper.setMinimum(-np.inf)
        self.upper.setValue(upper)

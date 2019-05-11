# -*- coding: utf-8 -*-

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from ..ui import resource_rc as resource_rc
from ..ui.channel_display_widget import Ui_ChannelDiplay
from ..dialogs.range_editor import RangeEditor


class ChannelDisplay(Ui_ChannelDiplay, QtWidgets.QWidget):

    color_changed = QtCore.pyqtSignal(int, str)
    enable_changed = QtCore.pyqtSignal(int, int)
    ylink_changed = QtCore.pyqtSignal(int, int)
    individual_axis_changed = QtCore.pyqtSignal(int, int)

    __slots__ = (
        'color',
        '_value_prefix',
        '_value',
        '_name',
        'fmt',
        'index',
        'ranges',
        'unit',
        '_transparent',
    )

    def __init__(self, index, unit="", kind='f', precision=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.color = "#ff0000"
        self._value_prefix = ""
        self._value = ""
        self._name = ""

        self.index = index
        self.ranges = {}
        self.unit = unit.strip()
        self.kind = kind
        self.precision = precision

        self._transparent = True

        self.color_btn.clicked.connect(self.select_color)
        self.display.stateChanged.connect(self.display_changed)
        self.ylink.stateChanged.connect(self._ylink_changed)
        self.individual_axis.stateChanged.connect(self._individual_axis)

        self.fm = QtGui.QFontMetrics(self.name.font())

        self.setToolTip(self._name)

        if kind in 'SUVui':
            self.fmt = '{}'
        else:
            self.fmt = f'{{:.{self.precision}f}}'

    def set_precision(self, precision):
        if self.kind == 'f':
            self.precision = precision
            self.fmt = f'{{:.{self.precision}f}}'

    def display_changed(self, state):
        state = self.display.checkState()
        self.enable_changed.emit(self.index, state)

    def _individual_axis(self, state):
        state = self.individual_axis.checkState()
        self.individual_axis_changed.emit(self.index, state)

    def _ylink_changed(self, state):
        state = self.ylink.checkState()
        self.ylink_changed.emit(self.index, state)

    def mouseDoubleClickEvent(self, event):
        dlg = RangeEditor(self.unit, self.ranges)
        dlg.exec_()
        if dlg.pressed_button == "apply":
            self.ranges = dlg.result

    def select_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color))
        if color.isValid():
            self.set_color(color.name())

            self.color_changed.emit(self.index, color.name())

    def set_fmt(self, fmt):
        if self.kind in 'fSUV':
            pass
        else:
            if fmt == "hex":
                self.fmt = "0x{:X}"
            elif fmt == "bin":
                self.fmt = "0b{:b}"
            elif fmt == "phys":
                self.fmt = "{}"

    def set_color(self, color):
        self.color = color
        self.set_name(self._name)
        self.set_value(self._value)
        self.color_btn.setStyleSheet(f"background-color: {color};")

    def set_name(self, text=""):
        self.setToolTip(text)
        self._name = text

    def set_prefix(self, text=""):
        self._value_prefix = text

    def set_value(self, value):
        self._value = value
        if self.ranges and value not in ("", "n.a."):
            for (start, stop), color in self.ranges.items():
                if start <= value < stop:
                    self.setStyleSheet(f"background-color: {color};")
                    break
            else:
                self.setStyleSheet("background-color: transparent;")
        elif not self._transparent:
            self.setStyleSheet("background-color: transparent;")
        template = '{{}}{}'
        if value not in ("", "n.a."):
            template = template.format(self.fmt)
        else:
            template = template.format("{}")
        try:
            self.value.setText(template.format(self._value_prefix, value))
        except (ValueError, TypeError):
            template = '{}{}'
            self.value.setText(template.format(self._value_prefix, value))

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()
        if modifier == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_C:
            QtWidgets.QApplication.instance().clipboard().setText(self._name)

        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        width = self.name.size().width()
        if self.unit:
            self.name.setText(self.fm.elidedText(f'{self._name} ({self.unit})', QtCore.Qt.ElideMiddle, width))
        else:
            self.name.setText(self.fm.elidedText(self._name, QtCore.Qt.ElideMiddle, width))

    def text(self):
        return self._name
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..dialogs.range_editor import RangeEditor
from ..ui.channel_bar_display_widget import Ui_ChannelBarDisplay


class BarWidget(QtWidgets.QWidget):
    def __init__(self, range=(0, 100), over=20, color=None):
        super().__init__()
        self.range = range
        self.over = over
        self.max = self.range[1]
        self.resizeEvent(None)
        self.color = color
        self.value = 50.0

    def resizeEvent(self, event):
        width = self.size().width()
        parts = 6
        while True:
            px = width / parts
            if px >= 50:
                break
            else:
                parts -= 2
                if parts == 4:
                    break

        if isinstance(self.range[0], int):
            self.ticks = [int(e) for e in np.linspace(self.range[0], self.range[1], parts + 1, True).tolist()]
        else:
            self.ticks = np.linspace(self.range[0], self.range[1], parts + 1, True).tolist()

    def setValue(self, value):
        self.value = float(value)
        self.paintEvent(None)

    def set_color(self, color):
        self.color = color
        self.paintEvent(None)

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()

    def drawWidget(self, qp):
        font = QtGui.QFont("Serif", 7, QtGui.QFont.Weight.Light)
        qp.setFont(font)

        size = self.size()
        w = size.width()
        h = size.height()

        till = int((w / self.max) * self.value)
        full = int((w / self.max) * self.over)

        if self.value >= self.over:
            qp.setPen(QtGui.QColor(self.color))
            qp.setBrush(QtGui.QColor(self.color))
            qp.drawRect(0, 0, full, h)
            qp.setPen(QtGui.QColor(255, 175, 175))
            qp.setBrush(QtGui.QColor(255, 175, 175))
            qp.drawRect(full, 0, till - full, h)

        else:
            qp.setPen(QtGui.QColor(self.color))
            qp.setBrush(QtGui.QColor(self.color))
            qp.drawRect(0, 0, till, h)

        pen = QtGui.QPen(QtGui.QColor(20, 20, 20), 1, QtCore.Qt.PenStyle.SolidLine)

        qp.setPen(pen)
        qp.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        qp.drawRect(0, 0, w - 1, h - 1)

        for j, val in enumerate(self.ticks):
            val_pos = int(val / self.range[1] * w)

            val = str(val) if isinstance(val, int) else f"{val:.3f}"

            qp.drawLine(val_pos, 0, val_pos, 5)
            metrics = qp.fontMetrics().boundingRect(val)
            fw = metrics.width()
            fh = metrics.height()

            x, y = int(val_pos - fw / 2), 7 + fh
            x = max(2, x)
            x = min(x, w - fw - 2)
            qp.drawText(x, y, val)


class ChannelBarDisplay(Ui_ChannelBarDisplay, QtWidgets.QWidget):
    def __init__(
        self,
        uuid,
        value,
        range,
        over,
        color,
        unit="",
        precision=3,
        tooltip="",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.color = color
        self._value_prefix = ""
        self._value = value
        self._name = ""

        self.bar = BarWidget(range, over, color=self.color)
        self.layout.addWidget(self.bar)

        self.instant_value = QtWidgets.QLabel("")
        self.instant_value.setMinimumWidth(150)
        font = self.instant_value.font()
        font.setBold(True)
        font.setPointSize(16)
        self.instant_value.setFont(font)
        self.layout.addWidget(self.instant_value)

        self.layout.setStretch(0, 0)
        self.layout.setStretch(1, 0)
        self.layout.setStretch(2, 1)
        self.layout.setStretch(3, 0)

        self.uuid = uuid
        self.ranges = {}
        self.unit = unit.strip()
        self.precision = precision

        self._transparent = True
        self._tooltip = tooltip

        self.color_btn.clicked.connect(self.select_color)

        self.fm = QtGui.QFontMetrics(self.name.font())

        self.setToolTip(self._tooltip or self._name)

        self.kind = "f"
        if self.kind in "SUVui":
            self.fmt = "{}"
        else:
            self.fmt = f"{{:.{self.precision}f}}"

        self.show()

    def set_selected(self, on):
        palette = self.name.palette()
        if on:
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Text, brush)
        else:
            brush = QtGui.QBrush(QtGui.QColor(self.color))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Text, brush)

        self.name.setPalette(palette)

    def set_precision(self, precision):
        if self.kind == "f":
            self.precision = precision
            self.fmt = f"{{:.{self.precision}f}}"

    def mouseDoubleClickEvent(self, event):
        dlg = RangeEditor(self.unit, self.ranges)
        dlg.exec_()
        if dlg.pressed_button == "apply":
            self.ranges = dlg.result

    def select_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color))
        if color.isValid():
            self.set_color(color.name())

        self.bar.set_color(color)

    def set_fmt(self, fmt):
        if self.kind in "SUV":
            self.fmt = "{}"
        elif self.kind == "f":
            self.fmt = f"{{:.{self.precision}f}}"
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

        self.bar.set_color(color)

        palette = self.name.palette()

        brush = QtGui.QBrush(QtGui.QColor(color))
        brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
        palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Text, brush)

        self.name.setPalette(palette)
        self.instant_value.setPalette(palette)

    def set_name(self, text=""):
        self.setToolTip(self._tooltip or text)
        self._name = text

    def set_prefix(self, text=""):
        self._value_prefix = text

    def update(self):
        width = self.name.size().width()
        if self.unit:
            self.name.setText(
                self.fm.elidedText(f"{self._name} ({self.unit})", QtCore.Qt.TextElideMode.ElideMiddle, width)
            )
        else:
            self.name.setText(self.fm.elidedText(self._name, QtCore.Qt.TextElideMode.ElideMiddle, width))
        self.set_value(self._value, update=True)

    def set_value(self, value, update=False):
        if self._value == value and update is False:
            return

        self.bar.setValue(value)
        self.instant_value.setText(self.fmt.format(value))

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()
        if modifier == QtCore.Qt.KeyboardModifier.ControlModifier and key == QtCore.Qt.Key.Key_C:
            QtWidgets.QApplication.instance().clipboard().setText(self._name)

        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        width = self.name.size().width()
        if self.unit:
            self.name.setText(
                self.fm.elidedText(f"{self._name} ({self.unit})", QtCore.Qt.TextElideMode.ElideMiddle, width)
            )
        else:
            self.name.setText(self.fm.elidedText(self._name, QtCore.Qt.TextElideMode.ElideMiddle, width))

    def text(self):
        return self._name

    def does_not_exist(self):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/error.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.color_btn.setIcon(icon)
        self.color_btn.setFlat(True)
        self.color_btn.clicked.disconnect()

    def disconnect_slots(self):
        pass

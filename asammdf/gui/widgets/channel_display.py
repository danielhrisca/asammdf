# -*- coding: utf-8 -*-
import json

from PyQt5 import QtCore, QtGui, QtWidgets

from ..dialogs.range_editor import RangeEditor
from ..ui import resource_rc as resource_rc
from ..ui.channel_display_widget import Ui_ChannelDiplay


class ChannelDisplay(Ui_ChannelDiplay, QtWidgets.QWidget):

    color_changed = QtCore.pyqtSignal(object, str)
    enable_changed = QtCore.pyqtSignal(object, int)
    ylink_changed = QtCore.pyqtSignal(object, int)
    individual_axis_changed = QtCore.pyqtSignal(object, int)

    def __init__(
        self,
        uuid,
        unit="",
        kind="f",
        precision=3,
        tooltip="",
        details="",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.color = "#ff0000"
        self._value_prefix = ""
        self._value = ""
        self._name = ""

        self.details.setText(details or "\tSource not available")

        self.details.setVisible(False)

        self.uuid = uuid
        self.ranges = {}
        self.unit = unit.strip()
        self.kind = kind
        self.precision = precision

        self._transparent = True
        self._tooltip = tooltip

        self.color_btn.clicked.connect(self.select_color)
        self.display.stateChanged.connect(self.display_changed)
        self.ylink.stateChanged.connect(self._ylink_changed)
        self.individual_axis.stateChanged.connect(self._individual_axis)

        self.fm = QtGui.QFontMetrics(self.name.font())

        self.setToolTip(self._tooltip or self._name)

        if kind in "SUVui":
            self.fmt = "{}"
        else:
            self.fmt = f"{{:.{self.precision}f}}"

    def set_precision(self, precision):
        if self.kind == "f":
            self.precision = precision
            self.fmt = f"{{:.{self.precision}f}}"

    def display_changed(self, state):
        state = self.display.checkState()
        self.enable_changed.emit(self.uuid, state)

    def _individual_axis(self, state):
        state = self.individual_axis.checkState()
        self.individual_axis_changed.emit(self.uuid, state)

    def _ylink_changed(self, state):
        state = self.ylink.checkState()
        self.ylink_changed.emit(self.uuid, state)

    def mouseDoubleClickEvent(self, event):
        dlg = RangeEditor(self.unit, self.ranges)
        dlg.exec_()
        if dlg.pressed_button == "apply":
            self.ranges = dlg.result

    def select_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color))
        if color.isValid():
            self.set_color(color.name())

            self.color_changed.emit(self.uuid, color.name())

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

        palette = self.name.palette()

        brush = QtGui.QBrush(QtGui.QColor(color))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)

        self.name.setPalette(palette)

    def set_selected(self, on):
        palette = self.name.palette()
        if on:
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        else:
            brush = QtGui.QBrush(QtGui.QColor(self.color))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)

        self.name.setPalette(palette)

    def set_name(self, text=""):
        self.setToolTip(self._tooltip or text)
        self._name = text

    def set_prefix(self, text=""):
        self._value_prefix = text

    def update(self):
        width = self.name.size().width()
        if self.unit:
            self.name.setText(
                self.fm.elidedText(
                    f"{self._name} ({self.unit})", QtCore.Qt.ElideMiddle, width
                )
            )
        else:
            self.name.setText(
                self.fm.elidedText(self._name, QtCore.Qt.ElideMiddle, width)
            )
        self.set_value(self._value, update=True)

    def set_value(self, value, update=False):
        if self._value == value and update is False:
            return

        self._value = value
        if self.ranges and value not in ("", "n.a."):
            for (start, stop), color in self.ranges.items():
                if start <= value <= stop:
                    self.setStyleSheet(f"background-color: {color};")
                    break
            else:
                self.setStyleSheet("background-color: transparent;")
        elif not self._transparent:
            self.setStyleSheet("background-color: transparent;")
        template = "{{}}{}"
        if value not in ("", "n.a."):
            template = template.format(self.fmt)
        else:
            template = template.format("{}")
        try:
            self.value.setText(template.format(self._value_prefix, value))
        except (ValueError, TypeError):
            template = "{}{}"
            self.value.setText(template.format(self._value_prefix, value))

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()
        if modifier == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_C:
            QtWidgets.QApplication.instance().clipboard().setText(self._name)

        elif (
            modifier == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier)
            and key == QtCore.Qt.Key_C
        ):
            QtWidgets.QApplication.instance().clipboard().setText(
                self.get_display_properties()
            )

        elif (
            modifier == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier)
            and key == QtCore.Qt.Key_P
        ):
            info = QtWidgets.QApplication.instance().clipboard().text()
            try:
                info = json.loads(info)
                self.set_color(info["color"])
                self.color_changed.emit(self.uuid, info["color"])
                self.set_fmt(info["format"])
                self.individual_axis.setCheckState(
                    QtCore.Qt.Checked
                    if info["individual_axis"]
                    else QtCore.Qt.Unchecked
                )
                self.ylink.setCheckState(
                    QtCore.Qt.Checked if info["ylink"] else QtCore.Qt.Unchecked
                )
                self.set_precision(info["precision"])

                parent = self.parent().parent().parent().parent().parent()
                sig, index = parent.plot.signal_by_uuid(self.uuid)
                viewbox = parent.plot.view_boxes[index]
                viewbox.setYRange(info["min"], info["max"], padding=0)

                self.display.setCheckState(
                    QtCore.Qt.Checked if info["display"] else QtCore.Qt.Unchecked
                )

                self.ranges = {}

                for key, val in info["ranges"].items():
                    start, stop = [float(e) for e in key.split("|")]
                    self.ranges[(start, stop)] = val

            except:
                pass

        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        width = self.name.size().width()
        if self.unit:
            self.name.setText(
                self.fm.elidedText(
                    f"{self._name} ({self.unit})", QtCore.Qt.ElideMiddle, width
                )
            )
        else:
            self.name.setText(
                self.fm.elidedText(self._name, QtCore.Qt.ElideMiddle, width)
            )

    def text(self):
        return self._name

    def get_display_properties(self):
        info = {
            "color": self.color,
            "precision": self.precision,
            "ylink": self.ylink.checkState() == QtCore.Qt.Checked,
            "individual_axis": self.individual_axis.checkState() == QtCore.Qt.Checked,
            "format": "hex"
            if self.fmt.startswith("0x")
            else "bin"
            if self.fmt.startswith("0b")
            else "phys",
            "display": self.display.checkState() == QtCore.Qt.Checked,
            "ranges": {
                f"{start}|{stop}": val for (start, stop), val in self.ranges.items()
            },
        }

        parent = self.parent().parent().parent().parent().parent()

        sig, index = parent.plot.signal_by_uuid(self.uuid)

        min_, max_ = parent.plot.view_boxes[index].viewRange()[1]

        info["min"] = float(min_)
        info["max"] = float(max_)

        return json.dumps(info)

    def does_not_exist(self):
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/error.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.color_btn.setIcon(icon)
        self.color_btn.setFlat(True)
        self.color_btn.clicked.disconnect()

    def disconnect_slots(self):
        self.color_changed.disconnect()
        self.enable_changed.disconnect()
        self.ylink_changed.disconnect()
        self.individual_axis_changed.disconnect()

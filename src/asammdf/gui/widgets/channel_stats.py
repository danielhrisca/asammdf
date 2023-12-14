from copy import deepcopy

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..ui.channel_stats import Ui_ChannelStats

MONOSPACE_FONT = None


class ChannelStats(Ui_ChannelStats, QtWidgets.QWidget):
    precision_modified = QtCore.Signal()

    def __init__(self, xunit="s", precision=6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._settings = QtCore.QSettings()

        self.setupUi(self)

        global MONOSPACE_FONT

        if MONOSPACE_FONT is None:
            families = QtGui.QFontDatabase().families()
            for family in (
                "Consolas",
                "Liberation Mono",
                "DejaVu Sans Mono",
                "Droid Sans Mono",
                "Liberation Mono",
                "Roboto Mono",
                "Monaco",
                "Courier",
            ):
                if family in families:
                    MONOSPACE_FONT = family
                    break

        font = QtGui.QFont(MONOSPACE_FONT)
        self.setFont(font)

        self.xunit = xunit.strip()
        self.color = "#000000"
        self.fmt = "phys"
        self.name_template = '<html><head/><body><p><span style=" font-size:11pt; font-weight:600; color:{};">{}</span></p></body></html>'
        self._name = "Please select a single channel"
        self.name.setStyleSheet("background-color: transparent;")

        for i in range(10):
            label = self.findChild(QtWidgets.QLabel, f"xunit{i}")
            label.setText(f" {self.xunit}")

        self.precision.addItems(["Full float precision"] + [f"{i} float decimals" for i in range(16)])
        self.precision.setCurrentIndex(self._settings.value("stats_float_precision", 6, type=int) + 1)

        self.precision.currentIndexChanged.connect(self.set_float_precision)

    def set_stats(self, stats):
        if not stats:
            self.clear()
            return

        self._stats = deepcopy(stats)
        precision = self._settings.value("stats_float_precision", 6, type=int)
        fmt = f" {{:.{precision}f}}"

        color = stats["color"]
        if stats:
            for name, value in stats.items():
                if name == "unit":
                    for i in range(1, 23):
                        label = self.findChild(QtWidgets.QLabel, f"unit{i}")
                        if label:
                            label.setText(f" {value}")
                    self.selected_gradient_unit.setText(f" {value}/{self.xunit}")
                    self.visible_gradient_unit.setText(f" {value}/{self.xunit}")
                    self.overall_gradient_unit.setText(f" {value}/{self.xunit}")
                    self.selected_integral_unit.setText(f" {value}*{self.xunit}")
                    self.visible_integral_unit.setText(f" {value}*{self.xunit}")
                    self.overall_integral_unit.setText(f" {value}*{self.xunit}")
                elif name == "name":
                    self._name = value
                    self.name.setText(self.name_template.format(self.color, self._name))
                elif name == "color":
                    self.color = value
                    self.name.setText(self.name_template.format(color, self._name))

                elif name in ("region", "color"):
                    continue
                else:
                    label = self.findChild(QtWidgets.QLabel, name)

                    if precision >= 0:
                        if isinstance(value, (float, np.floating)):
                            label.setText(fmt.format(value))
                        else:
                            label.setText(str(value))
                    else:
                        label.setText(str(value))

            if stats["region"]:
                self.cursor_group.setHidden(True)
                self.region_group.setHidden(False)
            else:
                self.cursor_group.setHidden(False)
                self.region_group.setHidden(True)
        else:
            self.clear()

    def clear(self):
        self._name = "Please select a single channel"
        self.color = "#000000"
        self.name.setText(self.name_template.format(self.color, self._name))
        for k, group in enumerate(
            (
                self.cursor_group,
                self.region_group,
                self.visible_group,
                self.overall_group,
            )
        ):
            layout = group.layout()
            rows = layout.rowCount()
            for i in range(rows):
                label = layout.itemAtPosition(i, 1)
                if label is not None:
                    label = label.widget()
                    label.setText("")
                label = layout.itemAtPosition(i, 2)
                if label is not None:
                    label = label.widget()
                    if label.objectName().startswith("unit"):
                        label.setText("")

    def set_float_precision(self, index):
        self._settings.setValue("stats_float_precision", index - 1)
        self.precision_modified.emit()

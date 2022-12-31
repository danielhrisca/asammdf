# -*- coding: utf-8 -*-

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from asammdf.blocks import v4_constants as v4c
from asammdf.blocks.conversion_utils import from_dict

from ..ui import resource_rc
from ..ui.define_conversion_dialog import Ui_ConversionDialog
from ..widgets.vrtt_widget import VRTTWidget
from ..widgets.vtt_widget import VTTWidget


# from https://stackoverflow.com/a/53936965/11009349
def range_overlapping(x, y):
    if x.start == x.stop or y.start == y.stop:
        return False
    return x.start <= y.stop and y.start <= x.stop


class ConversionEditor(Ui_ConversionDialog, QtWidgets.QDialog):
    def __init__(self, channel_name="", conversion=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        for widget in (
            self.a,
            self.b,
            self.p1,
            self.p2,
            self.p3,
            self.p4,
            self.p5,
            self.p6,
        ):
            widget.setMaximum(np.inf)
            widget.setMinimum(-np.inf)

        original_conversion_color = QtGui.QColor("#62b2e2")
        bar = self.tabs.tabBar()

        if conversion is not None:

            self.name.setText(conversion["name"])
            self.unit.setText(conversion["unit"])
            self.comment.setPlainText(conversion["comment"])

            if conversion["conversion_type"] == v4c.CONVERSION_TYPE_LIN:
                self.tabs.setCurrentIndex(0)
                bar.setTabTextColor(0, original_conversion_color)

                self.a.setValue(conversion["a"])
                self.b.setValue(conversion["b"])

            elif conversion["conversion_type"] == v4c.CONVERSION_TYPE_RAT:
                self.tabs.setCurrentIndex(1)
                bar.setTabTextColor(1, original_conversion_color)

                self.p1.setValue(conversion["P1"])
                self.p2.setValue(conversion["P2"])
                self.p3.setValue(conversion["P3"])
                self.p4.setValue(conversion["P4"])
                self.p5.setValue(conversion["P5"])
                self.p6.setValue(conversion["P6"])

            elif conversion["conversion_type"] == v4c.CONVERSION_TYPE_TABX:
                self.tabs.setCurrentIndex(2)
                bar.setTabTextColor(2, original_conversion_color)

                self.vtt_default.setText(
                    conversion.referenced_blocks["default_addr"].decode(
                        "utf-8", errors="replace"
                    )
                )

                for i in range(conversion.ref_param_nr - 1):
                    widget = VTTWidget()

                    item = QtWidgets.QListWidgetItem()
                    item.setSizeHint(widget.sizeHint())
                    self.vtt_list.addItem(item)
                    self.vtt_list.setItemWidget(item, widget)

                    widget.value.setValue(conversion[f"val_{i}"])
                    widget.text.setText(
                        conversion.referenced_blocks[f"text_{i}"].decode(
                            "utf-8", errors="replace"
                        )
                    )

            elif conversion["conversion_type"] == v4c.CONVERSION_TYPE_RTABX:
                self.tabs.setCurrentIndex(3)
                bar.setTabTextColor(3, original_conversion_color)

                self.vrtt_default.setText(
                    conversion.referenced_blocks["default_addr"].decode(
                        "utf-8", errors="replace"
                    )
                )

                for i in range(conversion.ref_param_nr - 1):
                    widget = VRTTWidget()

                    item = QtWidgets.QListWidgetItem()
                    item.setSizeHint(widget.sizeHint())
                    self.vrtt_list.addItem(item)
                    self.vrtt_list.setItemWidget(item, widget)

                    widget.lower.setValue(conversion[f"lower_{i}"])
                    widget.upper.setValue(conversion[f"upper_{i}"])
                    widget.text.setText(
                        conversion.referenced_blocks[f"text_{i}"].decode(
                            "utf-8", errors="replace"
                        )
                    )

            elif conversion["conversion_type"] == v4c.CONVERSION_TYPE_NON:
                self.tabs.setCurrentIndex(4)
                bar.setTabTextColor(4, original_conversion_color)
        else:
            self.tabs.setCurrentIndex(4)
            bar.setTabTextColor(4, original_conversion_color)

        self.insert_btn.clicked.connect(self.insert)
        self.insert_vrtt_btn.clicked.connect(self.insert_vrtt)
        self.reset_btn.clicked.connect(self.reset)
        self.reset_vrtt_btn.clicked.connect(self.reset_vrtt)
        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)

        self.vtt_list.setUniformItemSizes(True)
        self.vtt_list.setAlternatingRowColors(False)
        self.vrtt_list.setUniformItemSizes(True)
        self.vrtt_list.setAlternatingRowColors(False)

        self.setWindowTitle(f"Edit {channel_name} conversion")

        self.pressed_button = None

        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)

    def apply(self, event):

        if self.tabs.currentIndex() == 1:
            if (self.p4.value(), self.p5.value(), self.p6.value()) == (0, 0, 0):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid conversion parameters",
                    "The rational conversion parameters P4, P5 and P6 cannot all be 0.",
                )
                return

        elif self.tabs.currentIndex() == 2:
            mapping = set()

            for i in range(self.vtt_list.count()):
                item = self.vtt_list.item(i)
                if item is None:
                    continue

                widget = self.vtt_list.itemWidget(item)

                value = int(widget.value.value())

                if value in mapping:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid conversion parameters",
                        f"The value-to-text conversion raw value {value} is defined multiple times.",
                    )
                    return
                else:
                    mapping.add(value)

        elif self.tabs.currentIndex() == 3:
            mapping = set()

            for i in range(self.vrtt_list.count()):
                item = self.vrtt_list.item(i)
                if item is None:
                    continue

                widget = self.vrtt_list.itemWidget(item)

                lower = int(widget.lower.value())
                upper = int(widget.upper.value())

                if upper < lower:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid conversion parameters",
                        f"The upper value must be greater or higher than the lower value. ({upper=} and {lower=}",
                    )
                    return

                else:
                    y = range(lower, upper)
                    for x in mapping:
                        if range_overlapping(x, y):
                            QtWidgets.QMessageBox.warning(
                                self,
                                "Invalid conversion parameters",
                                f"The ranges cannot overlap; {x} overlaps with {y}",
                            )
                            return
                        else:
                            mapping.add(y)

        self.pressed_button = "apply"
        self.close()

    def conversion(self):
        if self.pressed_button in (None, "cancel"):
            conversion = None

        else:
            if self.tabs.currentIndex() == 0:
                conversion = {
                    "name": self.name.text().strip(),
                    "unit": self.unit.text().strip(),
                    "comment": self.comment.toPlainText().strip(),
                    "a": self.a.value(),
                    "b": self.b.value(),
                }

            elif self.tabs.currentIndex() == 1:
                conversion = {
                    "name": self.name.text().strip(),
                    "unit": self.unit.text().strip(),
                    "comment": self.comment.toPlainText().strip(),
                    "P1": self.p1.value(),
                    "P2": self.p2.value(),
                    "P3": self.p3.value(),
                    "P4": self.p4.value(),
                    "P5": self.p5.value(),
                    "P6": self.p6.value(),
                }

            elif self.tabs.currentIndex() == 2:
                conversion = {
                    "name": self.name.text().strip(),
                    "unit": self.unit.text().strip(),
                    "comment": self.comment.toPlainText().strip(),
                    "default": self.vtt_default.text().strip().encode("utf-8"),
                }

                cntr = 0

                for i in range(self.vtt_list.count()):
                    item = self.vtt_list.item(i)
                    if item is None:
                        continue

                    widget = self.vtt_list.itemWidget(item)

                    value = int(widget.value.value())
                    text = widget.text.text().strip().encode("utf-8")
                    conversion[f"val_{cntr}"] = value
                    conversion[f"text_{cntr}"] = text
                    cntr += 1

            elif self.tabs.currentIndex() == 3:
                conversion = {
                    "name": self.name.text().strip(),
                    "unit": self.unit.text().strip(),
                    "comment": self.comment.toPlainText().strip(),
                    "default": self.vrtt_default.text().strip().encode("utf-8"),
                }

                cntr = 0

                for i in range(self.vrtt_list.count()):
                    item = self.vrtt_list.item(i)
                    if item is None:
                        continue

                    widget = self.vrtt_list.itemWidget(item)

                    lower = int(widget.lower.value())
                    upper = int(widget.upper.value())
                    text = widget.text.text().strip().encode("utf-8")
                    conversion[f"lower_{cntr}"] = lower
                    conversion[f"upper_{cntr}"] = upper
                    conversion[f"text_{cntr}"] = text
                    cntr += 1

            elif self.tabs.currentIndex() == 4:
                conversion = {
                    "name": self.name.text().strip(),
                    "unit": self.unit.text().strip(),
                    "comment": self.comment.toPlainText().strip(),
                }

            conversion = from_dict(conversion)

        return conversion

    def insert(self, event):
        widget = VTTWidget()

        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.vtt_list.addItem(item)
        self.vtt_list.setItemWidget(item, widget)

    def insert_vrtt(self, event):
        widget = VRTTWidget()

        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.vrtt_list.addItem(item)
        self.vrtt_list.setItemWidget(item, widget)

    def reset(self, event):
        self.vtt_list.clear()

    def reset_vrtt(self, event):
        self.vrtt_list.clear()

    def cancel(self, event):
        self.pressed_button = "cancel"
        self.close()

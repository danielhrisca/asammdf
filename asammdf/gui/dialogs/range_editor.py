# -*- coding: utf-8 -*-
import os

try:
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5 import uic
    from ..ui import resource_qt5 as resource_rc

    QT = 5

except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    from PyQt4 import uic
    from ..ui import resource_qt4 as resource_rc

    QT = 4
    
HERE = os.path.dirname(os.path.realpath(__file__))


class RangeEditor(QDialog):
    def __init__(self, unit="", ranges=None, *args, **kwargs):
        super(RangeEditor, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "..", "ui", "range_editor_dialog.ui"), self)

        self.unit = unit
        self.result = {}
        self.pressed_button = None

        if ranges:
            for i, (range, color) in enumerate(ranges.items()):
                self.cell_pressed(i, 0, range, color)

        self.table.cellPressed.connect(self.cell_pressed)
        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)
        self.reset_btn.clicked.connect(self.reset)

        self.setWindowTitle("Edit channel range colors")

    def cell_pressed(self, row, column, range=(0, 0), color="#000000"):

        for col in (0, 1):
            box = QDoubleSpinBox(self.table)
            box.setSuffix(" {}".format(self.unit))
            box.setRange(-10 ** 10, 10 ** 10)
            box.setDecimals(6)
            box.setValue(range[col])

            self.table.setCellWidget(row, col, box)

        button = QPushButton("", self.table)
        button.setStyleSheet("background-color: {};".format(color))
        self.table.setCellWidget(row, 2, button)
        button.clicked.connect(partial(self.select_color, button=button))

        button = QPushButton("Delete", self.table)
        self.table.setCellWidget(row, 3, button)
        button.clicked.connect(partial(self.delete_row, row=row))

    def delete_row(self, event, row):
        for column in range(4):
            self.table.setCellWidget(row, column, None)

    def select_color(self, event, button):
        color = button.palette().button().color()
        color = QColorDialog.getColor(color).name()
        button.setStyleSheet("background-color: {};".format(color))

    def apply(self, event):
        for row in range(100):
            try:
                start = self.table.cellWidget(row, 0).value()
                stop = self.table.cellWidget(row, 1).value()
                button = self.table.cellWidget(row, 2)
                color = button.palette().button().color().name()
            except:
                continue
            else:
                self.result[(start, stop)] = color
        self.pressed_button = "apply"
        self.close()

    def reset(self, event):
        for row in range(100):
            for column in range(4):
                self.table.setCellWidget(row, column, None)
        self.result = {}

    def cancel(self, event):
        self.result = {}
        self.pressed_button = "cancel"
        self.close()

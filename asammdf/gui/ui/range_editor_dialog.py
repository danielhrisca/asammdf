# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\DSUsers\uidn3651\02__PythonWorkspace\asammdf\asammdf\gui\ui\range_editor_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_RangeDialog(object):
    def setupUi(self, RangeDialog):
        RangeDialog.setObjectName("RangeDialog")
        RangeDialog.resize(697, 379)
        RangeDialog.setSizeGripEnabled(True)
        RangeDialog.setModal(True)
        self.gridLayout = QtWidgets.QGridLayout(RangeDialog)
        self.gridLayout.setObjectName("gridLayout")
        self.reset_btn = QtWidgets.QPushButton(RangeDialog)
        self.reset_btn.setObjectName("reset_btn")
        self.gridLayout.addWidget(self.reset_btn, 0, 1, 1, 1)
        self.cancel_btn = QtWidgets.QPushButton(RangeDialog)
        self.cancel_btn.setObjectName("cancel_btn")
        self.gridLayout.addWidget(self.cancel_btn, 3, 1, 1, 1)
        self.apply_btn = QtWidgets.QPushButton(RangeDialog)
        self.apply_btn.setObjectName("apply_btn")
        self.gridLayout.addWidget(self.apply_btn, 2, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            20, 271, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
        self.table = QtWidgets.QTableWidget(RangeDialog)
        self.table.setRowCount(100)
        self.table.setColumnCount(4)
        self.table.setObjectName("table")
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(3, item)
        self.table.horizontalHeader().setDefaultSectionSize(150)
        self.table.horizontalHeader().setMinimumSectionSize(30)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.gridLayout.addWidget(self.table, 0, 0, 4, 1)
        self.gridLayout.setColumnStretch(0, 1)

        self.retranslateUi(RangeDialog)
        QtCore.QMetaObject.connectSlotsByName(RangeDialog)
        RangeDialog.setTabOrder(self.table, self.apply_btn)
        RangeDialog.setTabOrder(self.apply_btn, self.reset_btn)
        RangeDialog.setTabOrder(self.reset_btn, self.cancel_btn)

    def retranslateUi(self, RangeDialog):
        _translate = QtCore.QCoreApplication.translate
        RangeDialog.setWindowTitle(_translate("RangeDialog", "Edit value range colors"))
        self.reset_btn.setText(_translate("RangeDialog", "Reset"))
        self.cancel_btn.setText(_translate("RangeDialog", "Cancel"))
        self.apply_btn.setText(_translate("RangeDialog", "Apply"))
        item = self.table.horizontalHeaderItem(0)
        item.setText(_translate("RangeDialog", "From"))
        item = self.table.horizontalHeaderItem(1)
        item.setText(_translate("RangeDialog", "To"))
        item = self.table.horizontalHeaderItem(2)
        item.setText(_translate("RangeDialog", "Set color"))
        item = self.table.horizontalHeaderItem(3)
        item.setText(_translate("RangeDialog", " "))

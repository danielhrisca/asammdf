# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'multi_search_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MultiSearchDialog(object):
    def setupUi(self, MultiSearchDialog):
        MultiSearchDialog.setObjectName("MultiSearchDialog")
        MultiSearchDialog.resize(1028, 549)
        MultiSearchDialog.setSizeGripEnabled(True)
        self.grid_layout = QtWidgets.QGridLayout(MultiSearchDialog)
        self.grid_layout.setContentsMargins(9, 9, 9, 9)
        self.grid_layout.setObjectName("grid_layout")
        self.status = QtWidgets.QLabel(MultiSearchDialog)
        self.status.setText("")
        self.status.setObjectName("status")
        self.grid_layout.addWidget(self.status, 9, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cancel_btn = QtWidgets.QPushButton(MultiSearchDialog)
        self.cancel_btn.setObjectName("cancel_btn")
        self.horizontalLayout.addWidget(self.cancel_btn)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.apply_btn = QtWidgets.QPushButton(MultiSearchDialog)
        self.apply_btn.setObjectName("apply_btn")
        self.horizontalLayout.addWidget(self.apply_btn)
        self.horizontalLayout.setStretch(1, 1)
        self.grid_layout.addLayout(self.horizontalLayout, 8, 0, 1, 6)
        self.match_kind = QtWidgets.QComboBox(MultiSearchDialog)
        self.match_kind.setObjectName("match_kind")
        self.match_kind.addItem("")
        self.match_kind.addItem("")
        self.grid_layout.addWidget(self.match_kind, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.grid_layout.addItem(spacerItem1, 6, 1, 1, 1)
        self.selection = MinimalListWidget(MultiSearchDialog)
        self.selection.setMinimumSize(QtCore.QSize(500, 0))
        self.selection.setObjectName("selection")
        self.grid_layout.addWidget(self.selection, 5, 2, 3, 4)
        self.add_btn = QtWidgets.QPushButton(MultiSearchDialog)
        self.add_btn.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/left.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.add_btn.setIcon(icon)
        self.add_btn.setObjectName("add_btn")
        self.grid_layout.addWidget(self.add_btn, 5, 1, 1, 1)
        self.label = QtWidgets.QLabel(MultiSearchDialog)
        self.label.setObjectName("label")
        self.grid_layout.addWidget(self.label, 1, 2, 1, 1)
        self.search_box = QtWidgets.QLineEdit(MultiSearchDialog)
        self.search_box.setText("")
        self.search_box.setObjectName("search_box")
        self.grid_layout.addWidget(self.search_box, 1, 0, 1, 1)
        self.show_measurement_list_btn = QtWidgets.QPushButton(MultiSearchDialog)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.show_measurement_list_btn.setIcon(icon1)
        self.show_measurement_list_btn.setObjectName("show_measurement_list_btn")
        self.grid_layout.addWidget(self.show_measurement_list_btn, 0, 2, 1, 2)
        self.matches = QtWidgets.QListWidget(MultiSearchDialog)
        self.matches.setObjectName("matches")
        self.grid_layout.addWidget(self.matches, 5, 0, 2, 1)
        self.grid_layout.setColumnStretch(0, 1)

        self.retranslateUi(MultiSearchDialog)
        self.match_kind.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MultiSearchDialog)
        MultiSearchDialog.setTabOrder(self.search_box, self.match_kind)
        MultiSearchDialog.setTabOrder(self.match_kind, self.selection)
        MultiSearchDialog.setTabOrder(self.selection, self.add_btn)
        MultiSearchDialog.setTabOrder(self.add_btn, self.cancel_btn)
        MultiSearchDialog.setTabOrder(self.cancel_btn, self.apply_btn)

    def retranslateUi(self, MultiSearchDialog):
        _translate = QtCore.QCoreApplication.translate
        MultiSearchDialog.setWindowTitle(_translate("MultiSearchDialog", "Dialog"))
        self.cancel_btn.setText(_translate("MultiSearchDialog", "Cancel"))
        self.apply_btn.setText(_translate("MultiSearchDialog", "Apply"))
        self.match_kind.setItemText(0, _translate("MultiSearchDialog", "Wildcard"))
        self.match_kind.setItemText(1, _translate("MultiSearchDialog", "Regex"))
        self.label.setText(_translate("MultiSearchDialog", "Final selection"))
        self.show_measurement_list_btn.setText(_translate("MultiSearchDialog", "Show measurement list"))


from asammdf.gui.widgets.list import MinimalListWidget
from . import resource_rc

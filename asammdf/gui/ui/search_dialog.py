# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'search_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SearchDialog(object):
    def setupUi(self, SearchDialog):
        SearchDialog.setObjectName("SearchDialog")
        SearchDialog.resize(625, 549)
        SearchDialog.setSizeGripEnabled(True)
        self.grid_layout = QtWidgets.QGridLayout(SearchDialog)
        self.grid_layout.setContentsMargins(9, 9, 9, 9)
        self.grid_layout.setObjectName("grid_layout")
        self.matches = QtWidgets.QListWidget(SearchDialog)
        self.matches.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.matches.setObjectName("matches")
        self.grid_layout.addWidget(self.matches, 5, 0, 4, 1)
        self.search_box = QtWidgets.QLineEdit(SearchDialog)
        self.search_box.setText("")
        self.search_box.setObjectName("search_box")
        self.grid_layout.addWidget(self.search_box, 1, 0, 1, 1)
        self.match_kind = QtWidgets.QComboBox(SearchDialog)
        self.match_kind.setObjectName("match_kind")
        self.match_kind.addItem("")
        self.match_kind.addItem("")
        self.grid_layout.addWidget(self.match_kind, 0, 0, 1, 1)
        self.selection = MinimalListWidget(SearchDialog)
        self.selection.setObjectName("selection")
        self.grid_layout.addWidget(self.selection, 5, 2, 4, 4)
        self.status = QtWidgets.QLabel(SearchDialog)
        self.status.setText("")
        self.status.setObjectName("status")
        self.grid_layout.addWidget(self.status, 10, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.grid_layout.addItem(spacerItem, 6, 1, 1, 1)
        self.label = QtWidgets.QLabel(SearchDialog)
        self.label.setObjectName("label")
        self.grid_layout.addWidget(self.label, 1, 2, 1, 1)
        self.add_btn = QtWidgets.QPushButton(SearchDialog)
        self.add_btn.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/left.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.add_btn.setIcon(icon)
        self.add_btn.setObjectName("add_btn")
        self.grid_layout.addWidget(self.add_btn, 5, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cancel_btn = QtWidgets.QPushButton(SearchDialog)
        self.cancel_btn.setObjectName("cancel_btn")
        self.horizontalLayout.addWidget(self.cancel_btn)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem1)
        self.add_window_btn = QtWidgets.QPushButton(SearchDialog)
        self.add_window_btn.setObjectName("add_window_btn")
        self.horizontalLayout.addWidget(self.add_window_btn)
        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem2)
        self.apply_btn = QtWidgets.QPushButton(SearchDialog)
        self.apply_btn.setObjectName("apply_btn")
        self.horizontalLayout.addWidget(self.apply_btn)
        self.horizontalLayout.setStretch(1, 1)
        self.grid_layout.addLayout(self.horizontalLayout, 9, 0, 1, 6)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(2, 1)

        self.retranslateUi(SearchDialog)
        self.match_kind.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(SearchDialog)
        SearchDialog.setTabOrder(self.search_box, self.matches)
        SearchDialog.setTabOrder(self.matches, self.match_kind)

    def retranslateUi(self, SearchDialog):
        _translate = QtCore.QCoreApplication.translate
        SearchDialog.setWindowTitle(_translate("SearchDialog", "Dialog"))
        self.matches.setSortingEnabled(True)
        self.match_kind.setItemText(0, _translate("SearchDialog", "Wildcard"))
        self.match_kind.setItemText(1, _translate("SearchDialog", "Regex"))
        self.label.setText(_translate("SearchDialog", "Final selection"))
        self.cancel_btn.setText(_translate("SearchDialog", "Cancel"))
        self.add_window_btn.setText(_translate("SearchDialog", "Add window"))
        self.apply_btn.setText(_translate("SearchDialog", "Apply"))


from asammdf.gui.widgets.list import MinimalListWidget
from . import resource_rc

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'search_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SearchDialog(object):
    def setupUi(self, SearchDialog):
        SearchDialog.setObjectName("SearchDialog")
        SearchDialog.resize(400, 300)
        SearchDialog.setSizeGripEnabled(True)
        self.grid_layout = QtWidgets.QGridLayout(SearchDialog)
        self.grid_layout.setObjectName("grid_layout")
        self.apply_all_btn = QtWidgets.QPushButton(SearchDialog)
        self.apply_all_btn.setObjectName("apply_all_btn")
        self.grid_layout.addWidget(self.apply_all_btn, 1, 1, 1, 1)
        self.match_kind = QtWidgets.QComboBox(SearchDialog)
        self.match_kind.setObjectName("match_kind")
        self.match_kind.addItem("")
        self.match_kind.addItem("")
        self.grid_layout.addWidget(self.match_kind, 0, 0, 1, 1)
        self.search_box = QtWidgets.QLineEdit(SearchDialog)
        self.search_box.setObjectName("search_box")
        self.grid_layout.addWidget(self.search_box, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 189, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.grid_layout.addItem(spacerItem, 4, 1, 1, 1)
        self.matches = QtWidgets.QListWidget(SearchDialog)
        self.matches.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.matches.setObjectName("matches")
        self.grid_layout.addWidget(self.matches, 2, 0, 4, 1)
        self.apply_btn = QtWidgets.QPushButton(SearchDialog)
        self.apply_btn.setObjectName("apply_btn")
        self.grid_layout.addWidget(self.apply_btn, 0, 1, 1, 1)
        self.status = QtWidgets.QLabel(SearchDialog)
        self.status.setText("")
        self.status.setObjectName("status")
        self.grid_layout.addWidget(self.status, 6, 0, 1, 1)
        self.cancel_btn = QtWidgets.QPushButton(SearchDialog)
        self.cancel_btn.setObjectName("cancel_btn")
        self.grid_layout.addWidget(self.cancel_btn, 5, 1, 1, 1)

        self.retranslateUi(SearchDialog)
        self.match_kind.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(SearchDialog)
        SearchDialog.setTabOrder(self.match_kind, self.search_box)
        SearchDialog.setTabOrder(self.search_box, self.matches)
        SearchDialog.setTabOrder(self.matches, self.apply_btn)
        SearchDialog.setTabOrder(self.apply_btn, self.apply_all_btn)
        SearchDialog.setTabOrder(self.apply_all_btn, self.cancel_btn)

    def retranslateUi(self, SearchDialog):
        _translate = QtCore.QCoreApplication.translate
        SearchDialog.setWindowTitle(_translate("SearchDialog", "Dialog"))
        self.apply_all_btn.setText(_translate("SearchDialog", "Apply All"))
        self.match_kind.setItemText(0, _translate("SearchDialog", "Wildcard"))
        self.match_kind.setItemText(1, _translate("SearchDialog", "Regex"))
        self.matches.setSortingEnabled(True)
        self.apply_btn.setText(_translate("SearchDialog", "Apply"))
        self.cancel_btn.setText(_translate("SearchDialog", "Cancel"))

import resource_rc

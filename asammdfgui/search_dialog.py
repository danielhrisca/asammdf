# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'search_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9
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
        self.search_box = QtWidgets.QLineEdit(SearchDialog)
        self.search_box.setObjectName("search_box")
        self.grid_layout.addWidget(self.search_box, 0, 0, 1, 1)
        self.apply_btn = QtWidgets.QPushButton(SearchDialog)
        self.apply_btn.setObjectName("apply_btn")
        self.grid_layout.addWidget(self.apply_btn, 0, 1, 2, 1)
        self.apply_all_btn = QtWidgets.QPushButton(SearchDialog)
        self.apply_all_btn.setObjectName("apply_all_btn")
        self.grid_layout.addWidget(self.apply_all_btn, 2, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 189, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.grid_layout.addItem(spacerItem, 3, 1, 1, 1)
        self.cancel_btn = QtWidgets.QPushButton(SearchDialog)
        self.cancel_btn.setObjectName("cancel_btn")
        self.grid_layout.addWidget(self.cancel_btn, 4, 1, 1, 1)
        self.matches = QtWidgets.QListWidget(SearchDialog)
        self.matches.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.matches.setObjectName("matches")
        self.grid_layout.addWidget(self.matches, 1, 0, 4, 1)

        self.retranslateUi(SearchDialog)
        QtCore.QMetaObject.connectSlotsByName(SearchDialog)

    def retranslateUi(self, SearchDialog):
        _translate = QtCore.QCoreApplication.translate
        SearchDialog.setWindowTitle(_translate("SearchDialog", "Dialog"))
        self.apply_btn.setText(_translate("SearchDialog", "Apply"))
        self.apply_all_btn.setText(_translate("SearchDialog", "Apply All"))
        self.cancel_btn.setText(_translate("SearchDialog", "Cancel"))
        self.matches.setSortingEnabled(True)

from asammdfgui import resource_rc

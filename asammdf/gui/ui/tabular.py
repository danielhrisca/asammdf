# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tabular.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TabularDisplay(object):
    def setupUi(self, TabularDisplay):
        TabularDisplay.setObjectName("TabularDisplay")
        TabularDisplay.resize(811, 560)
        self.verticalLayout = QtWidgets.QVBoxLayout(TabularDisplay)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tree = QtWidgets.QTreeWidget(TabularDisplay)
        font = QtGui.QFont()
        font.setFamily("Lucida Console")
        self.tree.setFont(font)
        self.tree.setUniformRowHeights(True)
        self.tree.setObjectName("tree")
        self.tree.headerItem().setText(0, "1")
        self.verticalLayout.addWidget(self.tree)

        self.retranslateUi(TabularDisplay)
        QtCore.QMetaObject.connectSlotsByName(TabularDisplay)

    def retranslateUi(self, TabularDisplay):
        _translate = QtCore.QCoreApplication.translate
        TabularDisplay.setWindowTitle(_translate("TabularDisplay", "Form"))



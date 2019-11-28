# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tabular_filter.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TabularFilter(object):
    def setupUi(self, TabularFilter):
        TabularFilter.setObjectName("TabularFilter")
        TabularFilter.resize(813, 26)
        TabularFilter.setMinimumSize(QtCore.QSize(0, 26))
        self.horizontalLayout = QtWidgets.QHBoxLayout(TabularFilter)
        self.horizontalLayout.setContentsMargins(2, 2, 2, 2)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.enabled = QtWidgets.QCheckBox(TabularFilter)
        self.enabled.setText("")
        self.enabled.setChecked(True)
        self.enabled.setObjectName("enabled")
        self.horizontalLayout.addWidget(self.enabled)
        self.relation = QtWidgets.QComboBox(TabularFilter)
        self.relation.setObjectName("relation")
        self.horizontalLayout.addWidget(self.relation)
        self.column = QtWidgets.QComboBox(TabularFilter)
        self.column.setMinimumSize(QtCore.QSize(300, 0))
        self.column.setObjectName("column")
        self.horizontalLayout.addWidget(self.column)
        self.op = QtWidgets.QComboBox(TabularFilter)
        self.op.setObjectName("op")
        self.horizontalLayout.addWidget(self.op)
        self.target = QtWidgets.QLineEdit(TabularFilter)
        self.target.setClearButtonEnabled(False)
        self.target.setObjectName("target")
        self.horizontalLayout.addWidget(self.target)

        self.retranslateUi(TabularFilter)
        QtCore.QMetaObject.connectSlotsByName(TabularFilter)

    def retranslateUi(self, TabularFilter):
        _translate = QtCore.QCoreApplication.translate
        TabularFilter.setWindowTitle(_translate("TabularFilter", "Form"))

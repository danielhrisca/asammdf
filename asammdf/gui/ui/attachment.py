# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'attachment.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Attachment(object):
    def setupUi(self, Attachment):
        Attachment.setObjectName("Attachment")
        Attachment.resize(717, 205)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Attachment)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.number = QtWidgets.QLabel(Attachment)
        self.number.setObjectName("number")
        self.horizontalLayout.addWidget(self.number)
        self.fields = QtWidgets.QTreeWidget(Attachment)
        self.fields.setMinimumSize(QtCore.QSize(0, 187))
        self.fields.setObjectName("fields")
        self.fields.header().setVisible(True)
        self.fields.header().setMinimumSectionSize(100)
        self.horizontalLayout.addWidget(self.fields)
        self.extract_btn = QtWidgets.QPushButton(Attachment)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/export.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.extract_btn.setIcon(icon)
        self.extract_btn.setObjectName("extract_btn")
        self.horizontalLayout.addWidget(self.extract_btn)
        self.horizontalLayout.setStretch(1, 1)

        self.retranslateUi(Attachment)
        QtCore.QMetaObject.connectSlotsByName(Attachment)

    def retranslateUi(self, Attachment):
        _translate = QtCore.QCoreApplication.translate
        Attachment.setWindowTitle(_translate("Attachment", "Form"))
        self.number.setText(_translate("Attachment", "Number"))
        self.fields.headerItem().setText(0, _translate("Attachment", "Item"))
        self.fields.headerItem().setText(1, _translate("Attachment", "Value"))
        self.extract_btn.setText(_translate("Attachment", "Extract"))

from . import resource_rc

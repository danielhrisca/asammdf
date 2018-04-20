# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.8.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PyMDFMainWindow(object):
    def setupUi(self, PyMDFMainWindow):
        PyMDFMainWindow.setObjectName("PyMDFMainWindow")
        PyMDFMainWindow.resize(800, 709)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        PyMDFMainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(PyMDFMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.toolBox = QtWidgets.QToolBox(self.centralwidget)
        self.toolBox.setObjectName("toolBox")
        self.page = QtWidgets.QWidget()
        self.page.setGeometry(QtCore.QRect(0, 0, 782, 623))
        self.page.setObjectName("page")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.page)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.open_file_btn = QtWidgets.QPushButton(self.page)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.open_file_btn.setIcon(icon1)
        self.open_file_btn.setObjectName("open_file_btn")
        self.horizontalLayout_2.addWidget(self.open_file_btn)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.files = QtWidgets.QTabWidget(self.page)
        self.files.setDocumentMode(False)
        self.files.setTabsClosable(True)
        self.files.setObjectName("files")
        self.verticalLayout_2.addWidget(self.files)
        self.toolBox.addItem(self.page, "")
        self.verticalLayout.addWidget(self.toolBox)
        PyMDFMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(PyMDFMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        PyMDFMainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(PyMDFMainWindow)
        self.statusbar.setObjectName("statusbar")
        PyMDFMainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(PyMDFMainWindow)
        self.toolBox.setCurrentIndex(0)
        self.files.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(PyMDFMainWindow)

    def retranslateUi(self, PyMDFMainWindow):
        _translate = QtCore.QCoreApplication.translate
        PyMDFMainWindow.setWindowTitle(_translate("PyMDFMainWindow", "PyMDF"))
        self.open_file_btn.setText(_translate("PyMDFMainWindow", "Open"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), _translate("PyMDFMainWindow", "Single files"))

from asammdfgui import resource_rc

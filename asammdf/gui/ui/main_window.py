# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.3.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QMainWindow, QMenuBar,
    QSizePolicy, QStackedWidget, QTabWidget, QVBoxLayout,
    QWidget)
from . import resource_rc

class Ui_PyMDFMainWindow(object):
    def setupUi(self, PyMDFMainWindow):
        if not PyMDFMainWindow.objectName():
            PyMDFMainWindow.setObjectName(u"PyMDFMainWindow")
        PyMDFMainWindow.resize(800, 723)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PyMDFMainWindow.sizePolicy().hasHeightForWidth())
        PyMDFMainWindow.setSizePolicy(sizePolicy)
        icon = QIcon()
        icon.addFile(u":/asammdf.png", QSize(), QIcon.Normal, QIcon.Off)
        PyMDFMainWindow.setWindowIcon(icon)
        self.action_memory_minimum = QAction(PyMDFMainWindow)
        self.action_memory_minimum.setObjectName(u"action_memory_minimum")
        self.action_memory_minimum.setCheckable(True)
        self.action_memory_full = QAction(PyMDFMainWindow)
        self.action_memory_full.setObjectName(u"action_memory_full")
        self.action_memory_full.setCheckable(True)
        self.action_memory_low = QAction(PyMDFMainWindow)
        self.action_memory_low.setObjectName(u"action_memory_low")
        self.action_memory_low.setCheckable(True)
        self.centralwidget = QWidget(PyMDFMainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setFrameShape(QFrame.NoFrame)
        self.stackedWidgetPage1 = QWidget()
        self.stackedWidgetPage1.setObjectName(u"stackedWidgetPage1")
        self.verticalLayout_2 = QVBoxLayout(self.stackedWidgetPage1)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.files = QTabWidget(self.stackedWidgetPage1)
        self.files.setObjectName(u"files")
        self.files.setDocumentMode(False)
        self.files.setTabsClosable(True)

        self.verticalLayout_2.addWidget(self.files)

        self.stackedWidget.addWidget(self.stackedWidgetPage1)

        self.verticalLayout.addWidget(self.stackedWidget)

        PyMDFMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(PyMDFMainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 21))
        PyMDFMainWindow.setMenuBar(self.menubar)

        self.retranslateUi(PyMDFMainWindow)

        self.stackedWidget.setCurrentIndex(0)
        self.files.setCurrentIndex(-1)


        QMetaObject.connectSlotsByName(PyMDFMainWindow)
    # setupUi

    def retranslateUi(self, PyMDFMainWindow):
        PyMDFMainWindow.setWindowTitle(QCoreApplication.translate("PyMDFMainWindow", u"asammdf", None))
        self.action_memory_minimum.setText(QCoreApplication.translate("PyMDFMainWindow", u"minimum", None))
#if QT_CONFIG(tooltip)
        self.action_memory_minimum.setToolTip(QCoreApplication.translate("PyMDFMainWindow", u"Minimal memory usage by loading only the nedded block addresses", None))
#endif // QT_CONFIG(tooltip)
        self.action_memory_full.setText(QCoreApplication.translate("PyMDFMainWindow", u"full", None))
#if QT_CONFIG(tooltip)
        self.action_memory_full.setToolTip(QCoreApplication.translate("PyMDFMainWindow", u"Load all blocks in the RAM", None))
#endif // QT_CONFIG(tooltip)
        self.action_memory_low.setText(QCoreApplication.translate("PyMDFMainWindow", u"low", None))
#if QT_CONFIG(tooltip)
        self.action_memory_low.setToolTip(QCoreApplication.translate("PyMDFMainWindow", u"Load metdata block in RAM but leave the samples on disk", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi


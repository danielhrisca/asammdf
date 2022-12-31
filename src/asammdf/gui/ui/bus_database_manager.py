# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'bus_database_manager.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QListWidgetItem, QPushButton,
    QSizePolicy, QSpacerItem, QTabWidget, QWidget)

from asammdf.gui.widgets.list import MinimalListWidget
from . import resource_rc

class Ui_BusDatabaseManager(object):
    def setupUi(self, BusDatabaseManager):
        if not BusDatabaseManager.objectName():
            BusDatabaseManager.setObjectName(u"BusDatabaseManager")
        BusDatabaseManager.resize(632, 267)
        self.gridLayout = QGridLayout(BusDatabaseManager)
        self.gridLayout.setObjectName(u"gridLayout")
        self.tabWidget = QTabWidget(BusDatabaseManager)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_4 = QGridLayout(self.tab)
        self.gridLayout_4.setSpacing(2)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setContentsMargins(2, 2, 2, 2)
        self.load_can_database_btn = QPushButton(self.tab)
        self.load_can_database_btn.setObjectName(u"load_can_database_btn")
        icon = QIcon()
        icon.addFile(u":/open.png", QSize(), QIcon.Normal, QIcon.Off)
        self.load_can_database_btn.setIcon(icon)

        self.gridLayout_4.addWidget(self.load_can_database_btn, 0, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.can_database_list = MinimalListWidget(self.tab)
        self.can_database_list.setObjectName(u"can_database_list")

        self.gridLayout_4.addWidget(self.can_database_list, 1, 0, 1, 2)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.gridLayout_8 = QGridLayout(self.tab_2)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_8.setContentsMargins(2, 2, 2, 2)
        self.load_lin_database_btn = QPushButton(self.tab_2)
        self.load_lin_database_btn.setObjectName(u"load_lin_database_btn")
        self.load_lin_database_btn.setIcon(icon)

        self.gridLayout_8.addWidget(self.load_lin_database_btn, 0, 0, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_8.addItem(self.horizontalSpacer_3, 0, 1, 1, 1)

        self.lin_database_list = MinimalListWidget(self.tab_2)
        self.lin_database_list.setObjectName(u"lin_database_list")

        self.gridLayout_8.addWidget(self.lin_database_list, 1, 0, 1, 2)

        self.tabWidget.addTab(self.tab_2, "")

        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)


        self.retranslateUi(BusDatabaseManager)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(BusDatabaseManager)
    # setupUi

    def retranslateUi(self, BusDatabaseManager):
        BusDatabaseManager.setWindowTitle(QCoreApplication.translate("BusDatabaseManager", u"Form", None))
        self.load_can_database_btn.setText(QCoreApplication.translate("BusDatabaseManager", u"Load CAN database", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("BusDatabaseManager", u"CAN", None))
        self.load_lin_database_btn.setText(QCoreApplication.translate("BusDatabaseManager", u"Load LIN database", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("BusDatabaseManager", u"LIN", None))
    # retranslateUi


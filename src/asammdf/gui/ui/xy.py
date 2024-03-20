# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'xy.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QWidget)
from . import resource_rc

class Ui_XYDisplay(object):
    def setupUi(self, XYDisplay):
        if not XYDisplay.objectName():
            XYDisplay.setObjectName(u"XYDisplay")
        XYDisplay.resize(739, 424)
        self.gridLayout = QGridLayout(XYDisplay)
        self.gridLayout.setObjectName(u"gridLayout")
        self.plot_layout = QHBoxLayout()
        self.plot_layout.setObjectName(u"plot_layout")

        self.gridLayout.addLayout(self.plot_layout, 0, 0, 1, 3)

        self.label = QLabel(XYDisplay)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)

        self.x_channel_edit = QLineEdit(XYDisplay)
        self.x_channel_edit.setObjectName(u"x_channel_edit")

        self.gridLayout.addWidget(self.x_channel_edit, 1, 1, 1, 1)

        self.x_search_btn = QPushButton(XYDisplay)
        self.x_search_btn.setObjectName(u"x_search_btn")
        icon = QIcon()
        icon.addFile(u":/search.png", QSize(), QIcon.Normal, QIcon.Off)
        self.x_search_btn.setIcon(icon)

        self.gridLayout.addWidget(self.x_search_btn, 1, 2, 1, 1)

        self.label_2 = QLabel(XYDisplay)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)

        self.y_channel_edit = QLineEdit(XYDisplay)
        self.y_channel_edit.setObjectName(u"y_channel_edit")

        self.gridLayout.addWidget(self.y_channel_edit, 2, 1, 1, 1)

        self.y_search_btn = QPushButton(XYDisplay)
        self.y_search_btn.setObjectName(u"y_search_btn")
        self.y_search_btn.setIcon(icon)

        self.gridLayout.addWidget(self.y_search_btn, 2, 2, 1, 1)


        self.retranslateUi(XYDisplay)

        QMetaObject.connectSlotsByName(XYDisplay)
    # setupUi

    def retranslateUi(self, XYDisplay):
        XYDisplay.setWindowTitle(QCoreApplication.translate("XYDisplay", u"Form", None))
        self.label.setText(QCoreApplication.translate("XYDisplay", u"X channel", None))
        self.x_search_btn.setText("")
        self.label_2.setText(QCoreApplication.translate("XYDisplay", u"Y channel", None))
        self.y_search_btn.setText("")
    # retranslateUi


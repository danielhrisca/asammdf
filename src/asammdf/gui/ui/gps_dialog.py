# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'gps_dialog.UI'
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
from PySide6.QtWidgets import (QApplication, QDialog, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QSpacerItem, QWidget)
from . import resource_rc

class Ui_GPSDialog(object):
    def setupUi(self, GPSDialog):
        if not GPSDialog.objectName():
            GPSDialog.setObjectName(u"GPSDialog")
        GPSDialog.resize(623, 190)
        icon = QIcon()
        icon.addFile(u":/globe.png", QSize(), QIcon.Normal, QIcon.Off)
        GPSDialog.setWindowIcon(icon)
        GPSDialog.setSizeGripEnabled(True)
        self.gridLayout = QGridLayout(GPSDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_3 = QLabel(GPSDialog)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 3)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.cancel_btn = QPushButton(GPSDialog)
        self.cancel_btn.setObjectName(u"cancel_btn")

        self.horizontalLayout.addWidget(self.cancel_btn)

        self.horizontalSpacer_2 = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.apply_btn = QPushButton(GPSDialog)
        self.apply_btn.setObjectName(u"apply_btn")

        self.horizontalLayout.addWidget(self.apply_btn)

        self.horizontalLayout.setStretch(0, 1)

        self.gridLayout.addLayout(self.horizontalLayout, 5, 0, 1, 3)

        self.latitude = QLineEdit(GPSDialog)
        self.latitude.setObjectName(u"latitude")

        self.gridLayout.addWidget(self.latitude, 2, 1, 1, 1)

        self.search_latitude_btn = QPushButton(GPSDialog)
        self.search_latitude_btn.setObjectName(u"search_latitude_btn")
        icon1 = QIcon()
        icon1.addFile(u":/search.png", QSize(), QIcon.Normal, QIcon.Off)
        self.search_latitude_btn.setIcon(icon1)

        self.gridLayout.addWidget(self.search_latitude_btn, 2, 2, 1, 1)

        self.label_2 = QLabel(GPSDialog)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)

        self.label = QLabel(GPSDialog)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)

        self.search_longitude_btn = QPushButton(GPSDialog)
        self.search_longitude_btn.setObjectName(u"search_longitude_btn")
        self.search_longitude_btn.setIcon(icon1)

        self.gridLayout.addWidget(self.search_longitude_btn, 3, 2, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 1, 0, 1, 1)

        self.longitude = QLineEdit(GPSDialog)
        self.longitude.setObjectName(u"longitude")

        self.gridLayout.addWidget(self.longitude, 3, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 4, 0, 1, 1)

        self.gridLayout.setRowStretch(4, 1)
        QWidget.setTabOrder(self.latitude, self.longitude)
        QWidget.setTabOrder(self.longitude, self.search_latitude_btn)
        QWidget.setTabOrder(self.search_latitude_btn, self.search_longitude_btn)
        QWidget.setTabOrder(self.search_longitude_btn, self.apply_btn)
        QWidget.setTabOrder(self.apply_btn, self.cancel_btn)

        self.retranslateUi(GPSDialog)

        QMetaObject.connectSlotsByName(GPSDialog)
    # setupUi

    def retranslateUi(self, GPSDialog):
        GPSDialog.setWindowTitle(QCoreApplication.translate("GPSDialog", u"GPS channels selection", None))
        self.label_3.setText(QCoreApplication.translate("GPSDialog", u"Search for the latitude and longitude channels:", None))
        self.cancel_btn.setText(QCoreApplication.translate("GPSDialog", u"Cancel", None))
        self.apply_btn.setText(QCoreApplication.translate("GPSDialog", u"Apply", None))
        self.search_latitude_btn.setText("")
        self.label_2.setText(QCoreApplication.translate("GPSDialog", u"Longitude", None))
        self.label.setText(QCoreApplication.translate("GPSDialog", u"Latitude", None))
        self.search_longitude_btn.setText("")
    # retranslateUi


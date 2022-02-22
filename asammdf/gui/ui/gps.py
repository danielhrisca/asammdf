# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'gps.ui'
##
## Created by: Qt User Interface Compiler version 6.2.3
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
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QHBoxLayout, QLabel,
    QSizePolicy, QSlider, QVBoxLayout, QWidget)
from . import resource_rc

class Ui_GPSDisplay(object):
    def setupUi(self, GPSDisplay):
        if not GPSDisplay.objectName():
            GPSDisplay.setObjectName(u"GPSDisplay")
        GPSDisplay.resize(674, 49)
        self.map_layout = QVBoxLayout(GPSDisplay)
        self.map_layout.setObjectName(u"map_layout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.timestamp = QDoubleSpinBox(GPSDisplay)
        self.timestamp.setObjectName(u"timestamp")
        self.timestamp.setDecimals(9)
        self.timestamp.setSingleStep(0.001000000000000)

        self.horizontalLayout_2.addWidget(self.timestamp)

        self.min_t = QLabel(GPSDisplay)
        self.min_t.setObjectName(u"min_t")

        self.horizontalLayout_2.addWidget(self.min_t)

        self.timestamp_slider = QSlider(GPSDisplay)
        self.timestamp_slider.setObjectName(u"timestamp_slider")
        self.timestamp_slider.setMaximum(99999)
        self.timestamp_slider.setOrientation(Qt.Horizontal)
        self.timestamp_slider.setInvertedAppearance(False)
        self.timestamp_slider.setInvertedControls(False)
        self.timestamp_slider.setTickPosition(QSlider.NoTicks)
        self.timestamp_slider.setTickInterval(1)

        self.horizontalLayout_2.addWidget(self.timestamp_slider)

        self.max_t = QLabel(GPSDisplay)
        self.max_t.setObjectName(u"max_t")

        self.horizontalLayout_2.addWidget(self.max_t)

        self.horizontalLayout_2.setStretch(2, 1)

        self.map_layout.addLayout(self.horizontalLayout_2)


        self.retranslateUi(GPSDisplay)

        QMetaObject.connectSlotsByName(GPSDisplay)
    # setupUi

    def retranslateUi(self, GPSDisplay):
        GPSDisplay.setWindowTitle(QCoreApplication.translate("GPSDisplay", u"Form", None))
        self.timestamp.setSuffix(QCoreApplication.translate("GPSDisplay", u"s", None))
        self.min_t.setText("")
        self.max_t.setText("")
    # retranslateUi


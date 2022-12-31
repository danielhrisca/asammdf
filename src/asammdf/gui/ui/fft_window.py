# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'fft_window.ui'
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
    QMainWindow, QMenuBar, QSizePolicy, QSpacerItem,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_FFTWindow(object):
    def setupUi(self, FFTWindow):
        if not FFTWindow.objectName():
            FFTWindow.setObjectName(u"FFTWindow")
        FFTWindow.resize(640, 480)
        self.centralwidget = QWidget(FFTWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout.addWidget(self.label_2)

        self.start_frequency = QDoubleSpinBox(self.centralwidget)
        self.start_frequency.setObjectName(u"start_frequency")
        self.start_frequency.setDecimals(3)
        self.start_frequency.setMinimum(0.001000000000000)
        self.start_frequency.setMaximum(1000000.000000000000000)
        self.start_frequency.setValue(1.000000000000000)

        self.verticalLayout.addWidget(self.start_frequency)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.verticalLayout.addWidget(self.label)

        self.end_frequency = QDoubleSpinBox(self.centralwidget)
        self.end_frequency.setObjectName(u"end_frequency")
        self.end_frequency.setDecimals(3)
        self.end_frequency.setMinimum(0.001000000000000)
        self.end_frequency.setMaximum(1000000.000000000000000)
        self.end_frequency.setValue(1000.000000000000000)

        self.verticalLayout.addWidget(self.end_frequency)

        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout.addWidget(self.label_3)

        self.frequency_step = QDoubleSpinBox(self.centralwidget)
        self.frequency_step.setObjectName(u"frequency_step")
        self.frequency_step.setDecimals(3)
        self.frequency_step.setMinimum(0.001000000000000)
        self.frequency_step.setMaximum(1000000.000000000000000)
        self.frequency_step.setValue(1.000000000000000)

        self.verticalLayout.addWidget(self.frequency_step)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.layout = QVBoxLayout()
        self.layout.setObjectName(u"layout")

        self.horizontalLayout.addLayout(self.layout)

        self.horizontalLayout.setStretch(1, 1)
        FFTWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(FFTWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 640, 21))
        FFTWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(FFTWindow)
        self.statusbar.setObjectName(u"statusbar")
        FFTWindow.setStatusBar(self.statusbar)

        self.retranslateUi(FFTWindow)

        QMetaObject.connectSlotsByName(FFTWindow)
    # setupUi

    def retranslateUi(self, FFTWindow):
        FFTWindow.setWindowTitle(QCoreApplication.translate("FFTWindow", u"MainWindow", None))
        self.label_2.setText(QCoreApplication.translate("FFTWindow", u"Start frequency", None))
        self.start_frequency.setSuffix(QCoreApplication.translate("FFTWindow", u"Hz", None))
        self.label.setText(QCoreApplication.translate("FFTWindow", u"End frequency", None))
        self.end_frequency.setSuffix(QCoreApplication.translate("FFTWindow", u"Hz", None))
        self.label_3.setText(QCoreApplication.translate("FFTWindow", u"Frequency step", None))
        self.frequency_step.setSuffix(QCoreApplication.translate("FFTWindow", u"Hz", None))
    # retranslateUi


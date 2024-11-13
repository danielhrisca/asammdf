# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'xy.ui'
##
## Created by: Qt User Interface Compiler version 6.7.3
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
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSlider, QVBoxLayout, QWidget)
from . import resource_rc

class Ui_XYDisplay(object):
    def setupUi(self, XYDisplay):
        if not XYDisplay.objectName():
            XYDisplay.setObjectName(u"XYDisplay")
        XYDisplay.resize(919, 487)
        self.verticalLayout = QVBoxLayout(XYDisplay)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(1, 1, 1, 1)
        self.plot_layout = QHBoxLayout()
        self.plot_layout.setSpacing(1)
        self.plot_layout.setObjectName(u"plot_layout")

        self.verticalLayout.addLayout(self.plot_layout)

        self.groupBox = QGroupBox(XYDisplay)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout = QHBoxLayout(self.groupBox)
        self.horizontalLayout.setSpacing(1)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(1, 1, 1, 1)
        self.timestamp = QDoubleSpinBox(self.groupBox)
        self.timestamp.setObjectName(u"timestamp")
        self.timestamp.setDecimals(9)
        self.timestamp.setMinimum(-999999999999.000000000000000)
        self.timestamp.setMaximum(9999999.000000000000000)

        self.horizontalLayout.addWidget(self.timestamp)

        self.min_t = QLabel(self.groupBox)
        self.min_t.setObjectName(u"min_t")

        self.horizontalLayout.addWidget(self.min_t)

        self.timestamp_slider = QSlider(self.groupBox)
        self.timestamp_slider.setObjectName(u"timestamp_slider")
        self.timestamp_slider.setMaximum(99999)
        self.timestamp_slider.setOrientation(Qt.Orientation.Horizontal)
        self.timestamp_slider.setTickInterval(1)

        self.horizontalLayout.addWidget(self.timestamp_slider)

        self.max_t = QLabel(self.groupBox)
        self.max_t.setObjectName(u"max_t")

        self.horizontalLayout.addWidget(self.max_t)

        self.horizontalLayout.setStretch(2, 1)

        self.verticalLayout.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(XYDisplay)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout = QGridLayout(self.groupBox_2)
        self.gridLayout.setSpacing(1)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(1, 1, 1, 1)
        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.x_channel_edit = QLineEdit(self.groupBox_2)
        self.x_channel_edit.setObjectName(u"x_channel_edit")

        self.gridLayout.addWidget(self.x_channel_edit, 0, 1, 1, 1)

        self.x_search_btn = QPushButton(self.groupBox_2)
        self.x_search_btn.setObjectName(u"x_search_btn")
        icon = QIcon()
        icon.addFile(u":/search.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.x_search_btn.setIcon(icon)

        self.gridLayout.addWidget(self.x_search_btn, 0, 2, 1, 1)

        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.y_channel_edit = QLineEdit(self.groupBox_2)
        self.y_channel_edit.setObjectName(u"y_channel_edit")

        self.gridLayout.addWidget(self.y_channel_edit, 1, 1, 1, 1)

        self.y_search_btn = QPushButton(self.groupBox_2)
        self.y_search_btn.setObjectName(u"y_search_btn")
        self.y_search_btn.setIcon(icon)

        self.gridLayout.addWidget(self.y_search_btn, 1, 2, 1, 1)

        self.gridLayout.setColumnStretch(1, 1)

        self.verticalLayout.addWidget(self.groupBox_2)

        self.verticalLayout.setStretch(0, 1)

        self.retranslateUi(XYDisplay)

        QMetaObject.connectSlotsByName(XYDisplay)
    # setupUi

    def retranslateUi(self, XYDisplay):
        XYDisplay.setWindowTitle(QCoreApplication.translate("XYDisplay", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("XYDisplay", u"Time stamp", None))
        self.timestamp.setSuffix(QCoreApplication.translate("XYDisplay", u"s", None))
        self.min_t.setText("")
        self.max_t.setText("")
        self.groupBox_2.setTitle(QCoreApplication.translate("XYDisplay", u"Channels", None))
        self.label.setText(QCoreApplication.translate("XYDisplay", u"X channel", None))
        self.x_search_btn.setText("")
        self.label_2.setText(QCoreApplication.translate("XYDisplay", u"Y channel", None))
        self.y_search_btn.setText("")
    # retranslateUi

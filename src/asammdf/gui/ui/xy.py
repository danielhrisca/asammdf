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
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSlider, QVBoxLayout,
    QWidget)
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
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(1, 1, 1, 1)
        self.timestamp_slider = QSlider(self.groupBox)
        self.timestamp_slider.setObjectName(u"timestamp_slider")
        self.timestamp_slider.setMaximum(99999)
        self.timestamp_slider.setOrientation(Qt.Orientation.Horizontal)
        self.timestamp_slider.setTickInterval(1)

        self.gridLayout_2.addWidget(self.timestamp_slider, 1, 2, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_3, 2, 0, 1, 1)

        self.max_t = QLabel(self.groupBox)
        self.max_t.setObjectName(u"max_t")

        self.gridLayout_2.addWidget(self.max_t, 0, 3, 2, 1)

        self.timestamp = QDoubleSpinBox(self.groupBox)
        self.timestamp.setObjectName(u"timestamp")
        self.timestamp.setDecimals(9)
        self.timestamp.setMinimum(-999999999999.000000000000000)
        self.timestamp.setMaximum(9999999.000000000000000)

        self.gridLayout_2.addWidget(self.timestamp, 0, 0, 2, 1)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_4, 3, 0, 1, 1)

        self.x = QLabel(self.groupBox)
        self.x.setObjectName(u"x")

        self.gridLayout_2.addWidget(self.x, 2, 1, 1, 2)

        self.y = QLabel(self.groupBox)
        self.y.setObjectName(u"y")

        self.gridLayout_2.addWidget(self.y, 3, 1, 1, 2)

        self.min_t = QLabel(self.groupBox)
        self.min_t.setObjectName(u"min_t")

        self.gridLayout_2.addWidget(self.min_t, 0, 1, 2, 1)


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

        self.label_5 = QLabel(self.groupBox_2)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)

        self.y_search_btn = QPushButton(self.groupBox_2)
        self.y_search_btn.setObjectName(u"y_search_btn")
        self.y_search_btn.setIcon(icon)

        self.gridLayout.addWidget(self.y_search_btn, 1, 2, 1, 1)

        self.y_channel_edit = QLineEdit(self.groupBox_2)
        self.y_channel_edit.setObjectName(u"y_channel_edit")

        self.gridLayout.addWidget(self.y_channel_edit, 1, 1, 1, 1)

        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.timestamps_mode = QComboBox(self.groupBox_2)
        self.timestamps_mode.addItem("")
        self.timestamps_mode.addItem("")
        self.timestamps_mode.addItem("")
        self.timestamps_mode.setObjectName(u"timestamps_mode")

        self.gridLayout.addWidget(self.timestamps_mode, 2, 1, 1, 1)


        self.verticalLayout.addWidget(self.groupBox_2)

        self.verticalLayout.setStretch(0, 1)

        self.retranslateUi(XYDisplay)

        QMetaObject.connectSlotsByName(XYDisplay)
    # setupUi

    def retranslateUi(self, XYDisplay):
        XYDisplay.setWindowTitle(QCoreApplication.translate("XYDisplay", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("XYDisplay", u"Time stamp", None))
        self.label_3.setText(QCoreApplication.translate("XYDisplay", u"X =", None))
        self.max_t.setText("")
        self.timestamp.setSuffix(QCoreApplication.translate("XYDisplay", u"s", None))
        self.label_4.setText(QCoreApplication.translate("XYDisplay", u"Y =", None))
        self.x.setText(QCoreApplication.translate("XYDisplay", u"n.a.", None))
        self.y.setText(QCoreApplication.translate("XYDisplay", u"n.a.", None))
        self.min_t.setText("")
        self.groupBox_2.setTitle(QCoreApplication.translate("XYDisplay", u"Channels", None))
        self.label.setText(QCoreApplication.translate("XYDisplay", u"X channel", None))
        self.x_search_btn.setText("")
        self.label_5.setText(QCoreApplication.translate("XYDisplay", u"Timestamps mode ", None))
        self.y_search_btn.setText("")
        self.label_2.setText(QCoreApplication.translate("XYDisplay", u"Y channel", None))
        self.timestamps_mode.setItemText(0, QCoreApplication.translate("XYDisplay", u"All unique timestamps", None))
        self.timestamps_mode.setItemText(1, QCoreApplication.translate("XYDisplay", u"X channel timestamps", None))
        self.timestamps_mode.setItemText(2, QCoreApplication.translate("XYDisplay", u"Y channel timestamps", None))

    # retranslateUi


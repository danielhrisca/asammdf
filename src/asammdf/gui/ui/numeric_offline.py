# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'numeric_offline.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSlider, QSpacerItem,
    QVBoxLayout, QWidget)
from . import resource_rc

class Ui_NumericDisplay(object):
    def setupUi(self, NumericDisplay):
        if not NumericDisplay.objectName():
            NumericDisplay.setObjectName(u"NumericDisplay")
        NumericDisplay.resize(681, 666)
        self.main_layout = QVBoxLayout(NumericDisplay)
        self.main_layout.setObjectName(u"main_layout")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.toggle_controls_btn = QPushButton(NumericDisplay)
        self.toggle_controls_btn.setObjectName(u"toggle_controls_btn")
        icon = QIcon()
        icon.addFile(u":/down.png", QSize(), QIcon.Normal, QIcon.Off)
        self.toggle_controls_btn.setIcon(icon)

        self.gridLayout_2.addWidget(self.toggle_controls_btn, 0, 3, 1, 1)

        self.label_5 = QLabel(NumericDisplay)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)

        self.float_precision = QComboBox(NumericDisplay)
        self.float_precision.setObjectName(u"float_precision")

        self.gridLayout_2.addWidget(self.float_precision, 0, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 0, 2, 1, 1)

        self.gridLayout_2.setColumnStretch(2, 1)

        self.main_layout.addLayout(self.gridLayout_2)

        self.time_group = QGroupBox(NumericDisplay)
        self.time_group.setObjectName(u"time_group")
        self.horizontalLayout_2 = QHBoxLayout(self.time_group)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.timestamp = QDoubleSpinBox(self.time_group)
        self.timestamp.setObjectName(u"timestamp")
        self.timestamp.setDecimals(9)
        self.timestamp.setSingleStep(0.001000000000000)

        self.horizontalLayout.addWidget(self.timestamp)

        self.min_t = QLabel(self.time_group)
        self.min_t.setObjectName(u"min_t")

        self.horizontalLayout.addWidget(self.min_t)

        self.timestamp_slider = QSlider(self.time_group)
        self.timestamp_slider.setObjectName(u"timestamp_slider")
        self.timestamp_slider.setMaximum(99999)
        self.timestamp_slider.setOrientation(Qt.Horizontal)
        self.timestamp_slider.setInvertedAppearance(False)
        self.timestamp_slider.setInvertedControls(False)
        self.timestamp_slider.setTickPosition(QSlider.NoTicks)
        self.timestamp_slider.setTickInterval(1)

        self.horizontalLayout.addWidget(self.timestamp_slider)

        self.max_t = QLabel(self.time_group)
        self.max_t.setObjectName(u"max_t")

        self.horizontalLayout.addWidget(self.max_t)

        self.horizontalLayout.setStretch(2, 1)

        self.horizontalLayout_2.addLayout(self.horizontalLayout)


        self.main_layout.addWidget(self.time_group)

        self.search_group = QGroupBox(NumericDisplay)
        self.search_group.setObjectName(u"search_group")
        self.gridLayout = QGridLayout(self.search_group)
        self.gridLayout.setObjectName(u"gridLayout")
        self.pattern_match = QLineEdit(self.search_group)
        self.pattern_match.setObjectName(u"pattern_match")

        self.gridLayout.addWidget(self.pattern_match, 0, 1, 1, 1)

        self.op = QComboBox(self.search_group)
        self.op.addItem("")
        self.op.addItem("")
        self.op.addItem("")
        self.op.addItem("")
        self.op.addItem("")
        self.op.addItem("")
        self.op.setObjectName(u"op")

        self.gridLayout.addWidget(self.op, 0, 2, 1, 1)

        self.match_type = QComboBox(self.search_group)
        self.match_type.addItem("")
        self.match_type.addItem("")
        self.match_type.setObjectName(u"match_type")

        self.gridLayout.addWidget(self.match_type, 1, 1, 1, 1)

        self.target = QLineEdit(self.search_group)
        self.target.setObjectName(u"target")

        self.gridLayout.addWidget(self.target, 0, 3, 1, 1)

        self.match_mode = QComboBox(self.search_group)
        self.match_mode.addItem("")
        self.match_mode.addItem("")
        self.match_mode.setObjectName(u"match_mode")

        self.gridLayout.addWidget(self.match_mode, 0, 0, 1, 1)

        self.case_sensitivity = QComboBox(self.search_group)
        self.case_sensitivity.addItem("")
        self.case_sensitivity.addItem("")
        self.case_sensitivity.setObjectName(u"case_sensitivity")

        self.gridLayout.addWidget(self.case_sensitivity, 2, 1, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.backward = QPushButton(self.search_group)
        self.backward.setObjectName(u"backward")
        icon1 = QIcon()
        icon1.addFile(u":/right.png", QSize(), QIcon.Normal, QIcon.Off)
        self.backward.setIcon(icon1)

        self.horizontalLayout_3.addWidget(self.backward)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_4)

        self.forward = QPushButton(self.search_group)
        self.forward.setObjectName(u"forward")
        icon2 = QIcon()
        icon2.addFile(u":/left.png", QSize(), QIcon.Normal, QIcon.Off)
        self.forward.setIcon(icon2)

        self.horizontalLayout_3.addWidget(self.forward)


        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 3, 2, 1)

        self.match = QLabel(self.search_group)
        self.match.setObjectName(u"match")

        self.gridLayout.addWidget(self.match, 3, 0, 1, 4)

        self.gridLayout.setColumnStretch(1, 1)

        self.main_layout.addWidget(self.search_group)

        QWidget.setTabOrder(self.timestamp, self.pattern_match)
        QWidget.setTabOrder(self.pattern_match, self.op)
        QWidget.setTabOrder(self.op, self.target)
        QWidget.setTabOrder(self.target, self.backward)
        QWidget.setTabOrder(self.backward, self.forward)

        self.retranslateUi(NumericDisplay)

        QMetaObject.connectSlotsByName(NumericDisplay)
    # setupUi

    def retranslateUi(self, NumericDisplay):
        NumericDisplay.setWindowTitle(QCoreApplication.translate("NumericDisplay", u"Form", None))
        self.toggle_controls_btn.setText(QCoreApplication.translate("NumericDisplay", u"Show controls", None))
        self.label_5.setText(QCoreApplication.translate("NumericDisplay", u"Float precision", None))
        self.time_group.setTitle(QCoreApplication.translate("NumericDisplay", u"Time stamp", None))
        self.timestamp.setSuffix(QCoreApplication.translate("NumericDisplay", u"s", None))
        self.min_t.setText("")
        self.max_t.setText("")
        self.search_group.setTitle(QCoreApplication.translate("NumericDisplay", u"Search for values", None))
        self.pattern_match.setPlaceholderText(QCoreApplication.translate("NumericDisplay", u"pattern", None))
        self.op.setItemText(0, QCoreApplication.translate("NumericDisplay", u"==", None))
        self.op.setItemText(1, QCoreApplication.translate("NumericDisplay", u"!=", None))
        self.op.setItemText(2, QCoreApplication.translate("NumericDisplay", u"<", None))
        self.op.setItemText(3, QCoreApplication.translate("NumericDisplay", u"<=", None))
        self.op.setItemText(4, QCoreApplication.translate("NumericDisplay", u">", None))
        self.op.setItemText(5, QCoreApplication.translate("NumericDisplay", u">=", None))

        self.match_type.setItemText(0, QCoreApplication.translate("NumericDisplay", u"Wildcard", None))
        self.match_type.setItemText(1, QCoreApplication.translate("NumericDisplay", u"Regex", None))

        self.target.setPlaceholderText(QCoreApplication.translate("NumericDisplay", u"target value", None))
        self.match_mode.setItemText(0, QCoreApplication.translate("NumericDisplay", u"Raw", None))
        self.match_mode.setItemText(1, QCoreApplication.translate("NumericDisplay", u"Scaled", None))

        self.case_sensitivity.setItemText(0, QCoreApplication.translate("NumericDisplay", u"Case insensitive", None))
        self.case_sensitivity.setItemText(1, QCoreApplication.translate("NumericDisplay", u"Case sensitive", None))

        self.backward.setText("")
        self.forward.setText("")
        self.match.setText("")
    # retranslateUi


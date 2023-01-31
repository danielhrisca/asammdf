# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'vrtt_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QStackedWidget, QWidget)

class Ui_VRTT_Widget(object):
    def setupUi(self, VRTT_Widget):
        if not VRTT_Widget.objectName():
            VRTT_Widget.setObjectName(u"VRTT_Widget")
        VRTT_Widget.resize(496, 46)
        self.horizontalLayout = QHBoxLayout(VRTT_Widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 2, -1, 2)
        self.label = QLabel(VRTT_Widget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.lower = QDoubleSpinBox(VRTT_Widget)
        self.lower.setObjectName(u"lower")
        self.lower.setMinimumSize(QSize(100, 0))
        self.lower.setDecimals(0)

        self.horizontalLayout.addWidget(self.lower)

        self.label_2 = QLabel(VRTT_Widget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.upper = QDoubleSpinBox(VRTT_Widget)
        self.upper.setObjectName(u"upper")
        self.upper.setMinimumSize(QSize(100, 0))
        self.upper.setDecimals(0)

        self.horizontalLayout.addWidget(self.upper)

        self.label_3 = QLabel(VRTT_Widget)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout.addWidget(self.label_3)

        self.mode_switch = QComboBox(VRTT_Widget)
        self.mode_switch.addItem("")
        self.mode_switch.addItem("")
        self.mode_switch.setObjectName(u"mode_switch")

        self.horizontalLayout.addWidget(self.mode_switch)

        self.mode = QStackedWidget(VRTT_Widget)
        self.mode.setObjectName(u"mode")
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.horizontalLayout_2 = QHBoxLayout(self.page)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.text = QLineEdit(self.page)
        self.text.setObjectName(u"text")

        self.horizontalLayout_2.addWidget(self.text)

        self.mode.addWidget(self.page)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.horizontalLayout_3 = QHBoxLayout(self.page_2)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.conversion_btn = QPushButton(self.page_2)
        self.conversion_btn.setObjectName(u"conversion_btn")

        self.horizontalLayout_3.addWidget(self.conversion_btn)

        self.mode.addWidget(self.page_2)

        self.horizontalLayout.addWidget(self.mode)


        self.retranslateUi(VRTT_Widget)

        self.mode.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(VRTT_Widget)
    # setupUi

    def retranslateUi(self, VRTT_Widget):
        VRTT_Widget.setWindowTitle(QCoreApplication.translate("VRTT_Widget", u"Form", None))
        self.label.setText(QCoreApplication.translate("VRTT_Widget", u"[", None))
        self.label_2.setText(QCoreApplication.translate("VRTT_Widget", u",", None))
        self.label_3.setText(QCoreApplication.translate("VRTT_Widget", u") ->", None))
        self.mode_switch.setItemText(0, QCoreApplication.translate("VRTT_Widget", u"Text", None))
        self.mode_switch.setItemText(1, QCoreApplication.translate("VRTT_Widget", u"Conversion", None))

        self.conversion_btn.setText(QCoreApplication.translate("VRTT_Widget", u"Referenced conversion", None))
    # retranslateUi


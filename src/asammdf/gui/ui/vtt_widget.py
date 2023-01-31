# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'vtt_widget.ui'
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

class Ui_VTT_Widget(object):
    def setupUi(self, VTT_Widget):
        if not VTT_Widget.objectName():
            VTT_Widget.setObjectName(u"VTT_Widget")
        VTT_Widget.resize(511, 26)
        self.horizontalLayout = QHBoxLayout(VTT_Widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 2, -1, 2)
        self.value = QDoubleSpinBox(VTT_Widget)
        self.value.setObjectName(u"value")
        self.value.setMinimumSize(QSize(100, 0))
        self.value.setDecimals(0)

        self.horizontalLayout.addWidget(self.value)

        self.label_2 = QLabel(VTT_Widget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.mode_switch = QComboBox(VTT_Widget)
        self.mode_switch.addItem("")
        self.mode_switch.addItem("")
        self.mode_switch.setObjectName(u"mode_switch")

        self.horizontalLayout.addWidget(self.mode_switch)

        self.mode = QStackedWidget(VTT_Widget)
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


        self.retranslateUi(VTT_Widget)

        self.mode.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(VTT_Widget)
    # setupUi

    def retranslateUi(self, VTT_Widget):
        VTT_Widget.setWindowTitle(QCoreApplication.translate("VTT_Widget", u"Form", None))
        self.label_2.setText(QCoreApplication.translate("VTT_Widget", u"->", None))
        self.mode_switch.setItemText(0, QCoreApplication.translate("VTT_Widget", u"Text", None))
        self.mode_switch.setItemText(1, QCoreApplication.translate("VTT_Widget", u"Conversion", None))

        self.text.setPlaceholderText("")
        self.conversion_btn.setText(QCoreApplication.translate("VTT_Widget", u"Referenced conversion", None))
    # retranslateUi


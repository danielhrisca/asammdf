# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'vtt_widget.ui'
##
## Created by: Qt User Interface Compiler version 6.2.0
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
    QLineEdit, QSizePolicy, QWidget)

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

        self.text = QLineEdit(VTT_Widget)
        self.text.setObjectName(u"text")

        self.horizontalLayout.addWidget(self.text)


        self.retranslateUi(VTT_Widget)

        QMetaObject.connectSlotsByName(VTT_Widget)
    # setupUi

    def retranslateUi(self, VTT_Widget):
        VTT_Widget.setWindowTitle(QCoreApplication.translate("VTT_Widget", u"Form", None))
        self.label_2.setText(QCoreApplication.translate("VTT_Widget", u"->", None))
    # retranslateUi


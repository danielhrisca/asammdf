# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'range_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QWidget)

class Ui_RangeWidget(object):
    def setupUi(self, RangeWidget):
        if not RangeWidget.objectName():
            RangeWidget.setObjectName(u"RangeWidget")
        RangeWidget.resize(673, 41)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(RangeWidget.sizePolicy().hasHeightForWidth())
        RangeWidget.setSizePolicy(sizePolicy)
        RangeWidget.setMinimumSize(QSize(0, 41))
        RangeWidget.setMaximumSize(QSize(16777215, 41))
        self.horizontalLayout = QHBoxLayout(RangeWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(RangeWidget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.value1 = QLineEdit(RangeWidget)
        self.value1.setObjectName(u"value1")

        self.horizontalLayout.addWidget(self.value1)

        self.op1 = QComboBox(RangeWidget)
        self.op1.addItem("")
        self.op1.addItem("")
        self.op1.addItem("")
        self.op1.addItem("")
        self.op1.addItem("")
        self.op1.addItem("")
        self.op1.setObjectName(u"op1")
        self.op1.setEnabled(False)

        self.horizontalLayout.addWidget(self.op1)

        self.name = QLabel(RangeWidget)
        self.name.setObjectName(u"name")
        self.name.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.name)

        self.op2 = QComboBox(RangeWidget)
        self.op2.addItem("")
        self.op2.addItem("")
        self.op2.addItem("")
        self.op2.addItem("")
        self.op2.addItem("")
        self.op2.addItem("")
        self.op2.setObjectName(u"op2")
        self.op2.setEnabled(False)

        self.horizontalLayout.addWidget(self.op2)

        self.value2 = QLineEdit(RangeWidget)
        self.value2.setObjectName(u"value2")

        self.horizontalLayout.addWidget(self.value2)

        self.label_2 = QLabel(RangeWidget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.background_color_btn = QPushButton(RangeWidget)
        self.background_color_btn.setObjectName(u"background_color_btn")
        self.background_color_btn.setMinimumSize(QSize(40, 0))

        self.horizontalLayout.addWidget(self.background_color_btn)

        self.label_3 = QLabel(RangeWidget)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout.addWidget(self.label_3)

        self.font_color_btn = QPushButton(RangeWidget)
        self.font_color_btn.setObjectName(u"font_color_btn")
        self.font_color_btn.setMinimumSize(QSize(40, 0))

        self.horizontalLayout.addWidget(self.font_color_btn)

        self.horizontalLayout.setStretch(3, 1)

        self.retranslateUi(RangeWidget)

        QMetaObject.connectSlotsByName(RangeWidget)
    # setupUi

    def retranslateUi(self, RangeWidget):
        RangeWidget.setWindowTitle(QCoreApplication.translate("RangeWidget", u"Form", None))
        self.label.setText(QCoreApplication.translate("RangeWidget", u"If", None))
        self.op1.setItemText(0, QCoreApplication.translate("RangeWidget", u"==", None))
        self.op1.setItemText(1, QCoreApplication.translate("RangeWidget", u"!=", None))
        self.op1.setItemText(2, QCoreApplication.translate("RangeWidget", u">", None))
        self.op1.setItemText(3, QCoreApplication.translate("RangeWidget", u">=", None))
        self.op1.setItemText(4, QCoreApplication.translate("RangeWidget", u"<", None))
        self.op1.setItemText(5, QCoreApplication.translate("RangeWidget", u"<=", None))

        self.name.setText(QCoreApplication.translate("RangeWidget", u"TextLabel", None))
        self.op2.setItemText(0, QCoreApplication.translate("RangeWidget", u"==", None))
        self.op2.setItemText(1, QCoreApplication.translate("RangeWidget", u"!=", None))
        self.op2.setItemText(2, QCoreApplication.translate("RangeWidget", u">", None))
        self.op2.setItemText(3, QCoreApplication.translate("RangeWidget", u">=", None))
        self.op2.setItemText(4, QCoreApplication.translate("RangeWidget", u"<", None))
        self.op2.setItemText(5, QCoreApplication.translate("RangeWidget", u"<=", None))

        self.label_2.setText(QCoreApplication.translate("RangeWidget", u"then set background", None))
        self.background_color_btn.setText("")
        self.label_3.setText(QCoreApplication.translate("RangeWidget", u"and font/curve", None))
        self.font_color_btn.setText("")
    # retranslateUi


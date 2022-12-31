# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'attachment.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QSizePolicy, QTreeWidget, QTreeWidgetItem,
    QWidget)
from . import resource_rc

class Ui_Attachment(object):
    def setupUi(self, Attachment):
        if not Attachment.objectName():
            Attachment.setObjectName(u"Attachment")
        Attachment.resize(717, 205)
        self.horizontalLayout = QHBoxLayout(Attachment)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.number = QLabel(Attachment)
        self.number.setObjectName(u"number")

        self.horizontalLayout.addWidget(self.number)

        self.fields = QTreeWidget(Attachment)
        self.fields.setObjectName(u"fields")
        self.fields.setMinimumSize(QSize(0, 187))
        self.fields.header().setVisible(True)
        self.fields.header().setMinimumSectionSize(100)

        self.horizontalLayout.addWidget(self.fields)

        self.extract_btn = QPushButton(Attachment)
        self.extract_btn.setObjectName(u"extract_btn")
        icon = QIcon()
        icon.addFile(u":/export.png", QSize(), QIcon.Normal, QIcon.Off)
        self.extract_btn.setIcon(icon)

        self.horizontalLayout.addWidget(self.extract_btn)

        self.horizontalLayout.setStretch(1, 1)

        self.retranslateUi(Attachment)

        QMetaObject.connectSlotsByName(Attachment)
    # setupUi

    def retranslateUi(self, Attachment):
        Attachment.setWindowTitle(QCoreApplication.translate("Attachment", u"Form", None))
        self.number.setText(QCoreApplication.translate("Attachment", u"Number", None))
        ___qtreewidgetitem = self.fields.headerItem()
        ___qtreewidgetitem.setText(1, QCoreApplication.translate("Attachment", u"Value", None));
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("Attachment", u"Item", None));
        self.extract_btn.setText(QCoreApplication.translate("Attachment", u"Extract", None))
    # retranslateUi


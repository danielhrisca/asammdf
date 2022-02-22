# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'error_dialog.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)
from . import resource_rc

class Ui_ErrorDialog(object):
    def setupUi(self, ErrorDialog):
        if not ErrorDialog.objectName():
            ErrorDialog.setObjectName(u"ErrorDialog")
        ErrorDialog.resize(622, 114)
        icon = QIcon()
        icon.addFile(u":/error.png", QSize(), QIcon.Normal, QIcon.Off)
        ErrorDialog.setWindowIcon(icon)
        ErrorDialog.setSizeGripEnabled(True)
        self.layout = QVBoxLayout(ErrorDialog)
        self.layout.setObjectName(u"layout")
        self.error_message = QLabel(ErrorDialog)
        self.error_message.setObjectName(u"error_message")

        self.layout.addWidget(self.error_message)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.horizontalLayout_2.addItem(self.verticalSpacer)


        self.layout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.show_trace_btn = QPushButton(ErrorDialog)
        self.show_trace_btn.setObjectName(u"show_trace_btn")

        self.horizontalLayout.addWidget(self.show_trace_btn)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.copy_to_clipboard_btn = QPushButton(ErrorDialog)
        self.copy_to_clipboard_btn.setObjectName(u"copy_to_clipboard_btn")

        self.horizontalLayout.addWidget(self.copy_to_clipboard_btn)

        self.horizontalLayout.setStretch(0, 1)

        self.layout.addLayout(self.horizontalLayout)

        self.status = QLabel(ErrorDialog)
        self.status.setObjectName(u"status")

        self.layout.addWidget(self.status)

        self.layout.setStretch(1, 1)

        self.retranslateUi(ErrorDialog)

        QMetaObject.connectSlotsByName(ErrorDialog)
    # setupUi

    def retranslateUi(self, ErrorDialog):
        ErrorDialog.setWindowTitle(QCoreApplication.translate("ErrorDialog", u"Dialog", None))
        self.error_message.setText("")
        self.show_trace_btn.setText(QCoreApplication.translate("ErrorDialog", u"Show error trace", None))
        self.copy_to_clipboard_btn.setText(QCoreApplication.translate("ErrorDialog", u"Copy to clipboard", None))
        self.status.setText("")
    # retranslateUi


# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'windows_selection_dialog.ui'
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QDialog,
    QDialogButtonBox, QFrame, QGroupBox, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_WindowSelectionDialog(object):
    def setupUi(self, WindowSelectionDialog):
        if not WindowSelectionDialog.objectName():
            WindowSelectionDialog.setObjectName(u"WindowSelectionDialog")
        WindowSelectionDialog.resize(453, 435)
        self.verticalLayout_2 = QVBoxLayout(WindowSelectionDialog)
        self.verticalLayout_2.setSpacing(1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(1, 1, 1, 1)
        self.groupBox = QGroupBox(WindowSelectionDialog)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setFlat(True)
        self.lay = QVBoxLayout(self.groupBox)
        self.lay.setSpacing(1)
        self.lay.setObjectName(u"lay")
        self.lay.setContentsMargins(1, 1, 1, 1)
        self.scrollArea = QScrollArea(self.groupBox)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setFrameShape(QFrame.Shape.NoFrame)
        self.scrollArea.setFrameShadow(QFrame.Shadow.Plain)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 439, 355))
        self.verticalLayout_3 = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_3.setSpacing(1)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(1, 1, 1, 1)
        self.selection_layout = QVBoxLayout()
        self.selection_layout.setSpacing(1)
        self.selection_layout.setObjectName(u"selection_layout")

        self.verticalLayout_3.addLayout(self.selection_layout)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)

        self.lay.addWidget(self.scrollArea)

        self.disable_channels = QCheckBox(self.groupBox)
        self.disable_channels.setObjectName(u"disable_channels")

        self.lay.addWidget(self.disable_channels)


        self.verticalLayout_2.addWidget(self.groupBox)

        self.buttonBox = QDialogButtonBox(WindowSelectionDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.verticalLayout_2.addWidget(self.buttonBox)


        self.retranslateUi(WindowSelectionDialog)
        self.buttonBox.accepted.connect(WindowSelectionDialog.accept)
        self.buttonBox.rejected.connect(WindowSelectionDialog.reject)

        QMetaObject.connectSlotsByName(WindowSelectionDialog)
    # setupUi

    def retranslateUi(self, WindowSelectionDialog):
        WindowSelectionDialog.setWindowTitle(QCoreApplication.translate("WindowSelectionDialog", u"Select window type", None))
        self.groupBox.setTitle(QCoreApplication.translate("WindowSelectionDialog", u"Available window types", None))
        self.disable_channels.setText(QCoreApplication.translate("WindowSelectionDialog", u"Disable newly added channels", None))
    # retranslateUi

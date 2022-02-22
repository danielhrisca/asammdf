# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'channel_bar_display_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_ChannelBarDisplay(object):
    def setupUi(self, ChannelBarDisplay):
        if not ChannelBarDisplay.objectName():
            ChannelBarDisplay.setObjectName(u"ChannelBarDisplay")
        ChannelBarDisplay.resize(154, 46)
        ChannelBarDisplay.setMinimumSize(QSize(40, 46))
        ChannelBarDisplay.setMaximumSize(QSize(16777215, 46))
        self.verticalLayout = QVBoxLayout(ChannelBarDisplay)
        self.verticalLayout.setSpacing(3)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.layout = QHBoxLayout()
        self.layout.setObjectName(u"layout")
        self.color_btn = QPushButton(ChannelBarDisplay)
        self.color_btn.setObjectName(u"color_btn")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.color_btn.sizePolicy().hasHeightForWidth())
        self.color_btn.setSizePolicy(sizePolicy)
        self.color_btn.setMinimumSize(QSize(16, 16))
        self.color_btn.setMaximumSize(QSize(16, 16))
        self.color_btn.setAutoFillBackground(False)
        self.color_btn.setFlat(False)

        self.layout.addWidget(self.color_btn)

        self.name = QLabel(ChannelBarDisplay)
        self.name.setObjectName(u"name")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.name.sizePolicy().hasHeightForWidth())
        self.name.setSizePolicy(sizePolicy1)
        self.name.setMinimumSize(QSize(130, 40))
        self.name.setMaximumSize(QSize(16777215, 40))
        self.name.setMouseTracking(False)
        self.name.setTextFormat(Qt.PlainText)
        self.name.setTextInteractionFlags(Qt.LinksAccessibleByMouse)

        self.layout.addWidget(self.name)


        self.verticalLayout.addLayout(self.layout)


        self.retranslateUi(ChannelBarDisplay)

        QMetaObject.connectSlotsByName(ChannelBarDisplay)
    # setupUi

    def retranslateUi(self, ChannelBarDisplay):
        ChannelBarDisplay.setWindowTitle(QCoreApplication.translate("ChannelBarDisplay", u"Form", None))
        self.color_btn.setText("")
        self.name.setText(QCoreApplication.translate("ChannelBarDisplay", u"MAIN CLOCK", None))
    # retranslateUi


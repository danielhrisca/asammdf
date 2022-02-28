# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'channel_group_display_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLayout,
    QSizePolicy, QVBoxLayout, QWidget)
from . import resource_rc

class Ui_ChannelGroupDisplay(object):
    def setupUi(self, ChannelGroupDisplay):
        if not ChannelGroupDisplay.objectName():
            ChannelGroupDisplay.setObjectName(u"ChannelGroupDisplay")
        ChannelGroupDisplay.resize(643, 35)
        ChannelGroupDisplay.setMinimumSize(QSize(40, 16))
        self.verticalLayout = QVBoxLayout(ChannelGroupDisplay)
        self.verticalLayout.setSpacing(3)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 0, 2, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self._icon = QLabel(ChannelGroupDisplay)
        self._icon.setObjectName(u"_icon")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self._icon.sizePolicy().hasHeightForWidth())
        self._icon.setSizePolicy(sizePolicy)
        self._icon.setMinimumSize(QSize(16, 16))
        self._icon.setMaximumSize(QSize(16, 16))
        self._icon.setBaseSize(QSize(0, 0))
        self._icon.setPixmap(QPixmap(u":/open.png"))
        self._icon.setScaledContents(True)

        self.horizontalLayout.addWidget(self._icon)

        self.range_indicator = QLabel(ChannelGroupDisplay)
        self.range_indicator.setObjectName(u"range_indicator")
        self.range_indicator.setMinimumSize(QSize(16, 16))
        self.range_indicator.setMaximumSize(QSize(16, 16))
        self.range_indicator.setPixmap(QPixmap(u":/paint.png"))
        self.range_indicator.setScaledContents(True)

        self.horizontalLayout.addWidget(self.range_indicator)

        self.name = QLabel(ChannelGroupDisplay)
        self.name.setObjectName(u"name")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.name.sizePolicy().hasHeightForWidth())
        self.name.setSizePolicy(sizePolicy1)
        self.name.setMinimumSize(QSize(0, 16))
        self.name.setMouseTracking(False)
        self.name.setTextFormat(Qt.PlainText)
        self.name.setTextInteractionFlags(Qt.LinksAccessibleByMouse)

        self.horizontalLayout.addWidget(self.name)

        self.horizontalLayout.setStretch(2, 1)

        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(ChannelGroupDisplay)

        QMetaObject.connectSlotsByName(ChannelGroupDisplay)
    # setupUi

    def retranslateUi(self, ChannelGroupDisplay):
        ChannelGroupDisplay.setWindowTitle(QCoreApplication.translate("ChannelGroupDisplay", u"Form", None))
        self._icon.setText("")
        self.range_indicator.setText("")
        self.name.setText("")
    # retranslateUi


# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'channel_display_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget)
from . import resource_rc

class Ui_ChannelDiplay(object):
    def setupUi(self, ChannelDiplay):
        if not ChannelDiplay.objectName():
            ChannelDiplay.setObjectName(u"ChannelDiplay")
        ChannelDiplay.resize(565, 56)
        ChannelDiplay.setMinimumSize(QSize(40, 16))
        self.verticalLayout = QVBoxLayout(ChannelDiplay)
        self.verticalLayout.setSpacing(3)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 2, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.color_btn = QPushButton(ChannelDiplay)
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

        self.horizontalLayout.addWidget(self.color_btn)

        self.range_indicator = QLabel(ChannelDiplay)
        self.range_indicator.setObjectName(u"range_indicator")
        self.range_indicator.setMinimumSize(QSize(16, 16))
        self.range_indicator.setMaximumSize(QSize(16, 16))
        self.range_indicator.setPixmap(QPixmap(u":/paint.png"))
        self.range_indicator.setScaledContents(True)

        self.horizontalLayout.addWidget(self.range_indicator)

        self.name = QLabel(ChannelDiplay)
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

        self.value = QLabel(ChannelDiplay)
        self.value.setObjectName(u"value")
        self.value.setMinimumSize(QSize(75, 0))
        self.value.setTextFormat(Qt.PlainText)
        self.value.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout.addWidget(self.value)

        self.ylink = QCheckBox(ChannelDiplay)
        self.ylink.setObjectName(u"ylink")

        self.horizontalLayout.addWidget(self.ylink)

        self.individual_axis = QCheckBox(ChannelDiplay)
        self.individual_axis.setObjectName(u"individual_axis")

        self.horizontalLayout.addWidget(self.individual_axis)

        self.horizontalLayout.setStretch(2, 1)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.details = QLabel(ChannelDiplay)
        self.details.setObjectName(u"details")

        self.verticalLayout.addWidget(self.details)


        self.retranslateUi(ChannelDiplay)

        QMetaObject.connectSlotsByName(ChannelDiplay)
    # setupUi

    def retranslateUi(self, ChannelDiplay):
        ChannelDiplay.setWindowTitle(QCoreApplication.translate("ChannelDiplay", u"Form", None))
        self.color_btn.setText("")
        self.range_indicator.setText("")
        self.name.setText(QCoreApplication.translate("ChannelDiplay", u"MAIN CLOCK", None))
        self.value.setText("")
#if QT_CONFIG(tooltip)
        self.ylink.setToolTip(QCoreApplication.translate("ChannelDiplay", u"enable common Y axis", None))
#endif // QT_CONFIG(tooltip)
        self.ylink.setText("")
        self.individual_axis.setText("")
        self.details.setText(QCoreApplication.translate("ChannelDiplay", u"details", None))
    # retranslateUi


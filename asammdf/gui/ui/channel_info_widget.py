# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'channel_info_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QLabel,
    QScrollArea, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)

class Ui_ChannelInfo(object):
    def setupUi(self, ChannelInfo):
        if not ChannelInfo.objectName():
            ChannelInfo.setObjectName(u"ChannelInfo")
        ChannelInfo.resize(926, 558)
        self.horizontalLayout = QHBoxLayout(ChannelInfo)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox = QGroupBox(ChannelInfo)
        self.groupBox.setObjectName(u"groupBox")
        font = QFont()
        font.setFamilies([u"Consolas"])
        font.setPointSize(8)
        self.groupBox.setFont(font)
        self.groupBox.setFlat(False)
        self.verticalLayout_4 = QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.scrollArea = QScrollArea(self.groupBox)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 277, 505))
        self.verticalLayout_3 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.channel_label = QLabel(self.scrollAreaWidgetContents)
        self.channel_label.setObjectName(u"channel_label")
        self.channel_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.verticalLayout_3.addWidget(self.channel_label)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_3)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_4.addWidget(self.scrollArea)


        self.horizontalLayout.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(ChannelInfo)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setFont(font)
        self.verticalLayout_6 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.scrollArea_2 = QScrollArea(self.groupBox_2)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 276, 505))
        self.verticalLayout_2 = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.conversion_label = QLabel(self.scrollAreaWidgetContents_2)
        self.conversion_label.setObjectName(u"conversion_label")
        self.conversion_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.verticalLayout_2.addWidget(self.conversion_label)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_6.addWidget(self.scrollArea_2)


        self.horizontalLayout.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(ChannelInfo)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setFont(font)
        self.verticalLayout_5 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.scrollArea_3 = QScrollArea(self.groupBox_3)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 277, 505))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents_3)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.source_label = QLabel(self.scrollAreaWidgetContents_3)
        self.source_label.setObjectName(u"source_label")
        self.source_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.verticalLayout.addWidget(self.source_label)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_2)

        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)

        self.verticalLayout_5.addWidget(self.scrollArea_3)


        self.horizontalLayout.addWidget(self.groupBox_3)


        self.retranslateUi(ChannelInfo)

        QMetaObject.connectSlotsByName(ChannelInfo)
    # setupUi

    def retranslateUi(self, ChannelInfo):
        ChannelInfo.setWindowTitle(QCoreApplication.translate("ChannelInfo", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("ChannelInfo", u"Channel", None))
        self.channel_label.setText(QCoreApplication.translate("ChannelInfo", u"Channel metadata", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("ChannelInfo", u"Conversion", None))
        self.conversion_label.setText(QCoreApplication.translate("ChannelInfo", u"No conversion", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("ChannelInfo", u"Source", None))
        self.source_label.setText(QCoreApplication.translate("ChannelInfo", u"No source", None))
    # retranslateUi


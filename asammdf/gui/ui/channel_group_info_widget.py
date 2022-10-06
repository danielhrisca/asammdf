# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'channel_group_info_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
    QScrollArea, QScrollBar, QSizePolicy, QSpacerItem,
    QSplitter, QTabWidget, QVBoxLayout, QWidget)

class Ui_ChannelGroupInfo(object):
    def setupUi(self, ChannelGroupInfo):
        if not ChannelGroupInfo.objectName():
            ChannelGroupInfo.setObjectName(u"ChannelGroupInfo")
        ChannelGroupInfo.resize(926, 558)
        self.verticalLayout_2 = QVBoxLayout(ChannelGroupInfo)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.tabs = QTabWidget(ChannelGroupInfo)
        self.tabs.setObjectName(u"tabs")
        font = QFont()
        font.setFamilies([u"Consolas"])
        font.setPointSize(8)
        self.tabs.setFont(font)
        self.ta_widet = QWidget()
        self.ta_widet.setObjectName(u"ta_widet")
        self.horizontalLayout = QHBoxLayout(self.ta_widet)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox = QGroupBox(self.ta_widet)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setFlat(False)
        self.verticalLayout_4 = QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.scrollArea = QScrollArea(self.groupBox)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 417, 461))
        self.verticalLayout_3 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.channel_group_label = QLabel(self.scrollAreaWidgetContents)
        self.channel_group_label.setObjectName(u"channel_group_label")
        self.channel_group_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)

        self.verticalLayout_3.addWidget(self.channel_group_label)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_3)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_4.addWidget(self.scrollArea)


        self.horizontalLayout.addWidget(self.groupBox)

        self.groupBox_3 = QGroupBox(self.ta_widet)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_5 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.scrollArea_3 = QScrollArea(self.groupBox_3)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 417, 461))
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

        self.tabs.addTab(self.ta_widet, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.horizontalLayout_3 = QHBoxLayout(self.tab_2)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.splitter = QSplitter(self.tab_2)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout_6 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.wrap = QCheckBox(self.layoutWidget)
        self.wrap.setObjectName(u"wrap")
        self.wrap.setChecked(True)

        self.verticalLayout_6.addWidget(self.wrap)

        self.channels = QListWidget(self.layoutWidget)
        self.channels.setObjectName(u"channels")

        self.verticalLayout_6.addWidget(self.channels)

        self.splitter.addWidget(self.layoutWidget)
        self.display = QPlainTextEdit(self.splitter)
        self.display.setObjectName(u"display")
        self.display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.display.setUndoRedoEnabled(False)
        self.display.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.display.setReadOnly(True)
        self.splitter.addWidget(self.display)

        self.horizontalLayout_3.addWidget(self.splitter)

        self.scroll = QScrollBar(self.tab_2)
        self.scroll.setObjectName(u"scroll")
        self.scroll.setMaximum(9999)
        self.scroll.setOrientation(Qt.Vertical)

        self.horizontalLayout_3.addWidget(self.scroll)

        self.tabs.addTab(self.tab_2, "")

        self.verticalLayout_2.addWidget(self.tabs)


        self.retranslateUi(ChannelGroupInfo)

        self.tabs.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(ChannelGroupInfo)
    # setupUi

    def retranslateUi(self, ChannelGroupInfo):
        ChannelGroupInfo.setWindowTitle(QCoreApplication.translate("ChannelGroupInfo", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("ChannelGroupInfo", u"Channel Group", None))
        self.channel_group_label.setText(QCoreApplication.translate("ChannelGroupInfo", u"Channel group metadata", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("ChannelGroupInfo", u"Source", None))
        self.source_label.setText(QCoreApplication.translate("ChannelGroupInfo", u"No source", None))
        self.tabs.setTabText(self.tabs.indexOf(self.ta_widet), QCoreApplication.translate("ChannelGroupInfo", u"Channel group CGBLOCK", None))
        self.wrap.setText(QCoreApplication.translate("ChannelGroupInfo", u"Text wrap", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_2), QCoreApplication.translate("ChannelGroupInfo", u"Raw bytes", None))
    # retranslateUi


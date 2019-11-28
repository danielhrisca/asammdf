# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'channel_group_info_widget.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ChannelGroupInfo(object):
    def setupUi(self, ChannelGroupInfo):
        ChannelGroupInfo.setObjectName("ChannelGroupInfo")
        ChannelGroupInfo.resize(926, 558)
        self.horizontalLayout = QtWidgets.QHBoxLayout(ChannelGroupInfo)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(ChannelGroupInfo)
        self.groupBox.setFlat(False)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.scrollArea = QtWidgets.QScrollArea(self.groupBox)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 423, 494))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.channel_group_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.channel_group_label.setTextInteractionFlags(
            QtCore.Qt.LinksAccessibleByMouse
            | QtCore.Qt.TextSelectableByKeyboard
            | QtCore.Qt.TextSelectableByMouse
        )
        self.channel_group_label.setObjectName("channel_group_label")
        self.verticalLayout_3.addWidget(self.channel_group_label)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout_3.addItem(spacerItem)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_4.addWidget(self.scrollArea)
        self.horizontalLayout.addWidget(self.groupBox)
        self.groupBox_3 = QtWidgets.QGroupBox(ChannelGroupInfo)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.scrollArea_3 = QtWidgets.QScrollArea(self.groupBox_3)
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 422, 494))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.source_label = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.source_label.setTextInteractionFlags(
            QtCore.Qt.LinksAccessibleByMouse
            | QtCore.Qt.TextSelectableByKeyboard
            | QtCore.Qt.TextSelectableByMouse
        )
        self.source_label.setObjectName("source_label")
        self.verticalLayout.addWidget(self.source_label)
        spacerItem1 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout.addItem(spacerItem1)
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)
        self.verticalLayout_5.addWidget(self.scrollArea_3)
        self.horizontalLayout.addWidget(self.groupBox_3)

        self.retranslateUi(ChannelGroupInfo)
        QtCore.QMetaObject.connectSlotsByName(ChannelGroupInfo)

    def retranslateUi(self, ChannelGroupInfo):
        _translate = QtCore.QCoreApplication.translate
        ChannelGroupInfo.setWindowTitle(_translate("ChannelGroupInfo", "Form"))
        self.groupBox.setTitle(_translate("ChannelGroupInfo", "Channel Group"))
        self.channel_group_label.setText(
            _translate("ChannelGroupInfo", "Channel group metadata")
        )
        self.groupBox_3.setTitle(_translate("ChannelGroupInfo", "Source"))
        self.source_label.setText(_translate("ChannelGroupInfo", "No source"))

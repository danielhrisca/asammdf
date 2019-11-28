# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\DSUsers\uidn3651\02__PythonWorkspace\asammdf\asammdf\gui\ui\define_channel_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ComputedChannel(object):
    def setupUi(self, ComputedChannel):
        ComputedChannel.setObjectName("ComputedChannel")
        ComputedChannel.resize(596, 400)
        ComputedChannel.setMaximumSize(QtCore.QSize(16777215, 400))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/plus.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        ComputedChannel.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(ComputedChannel)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(ComputedChannel)
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.operand1 = QtWidgets.QComboBox(self.groupBox)
        self.operand1.setMinimumSize(QtCore.QSize(380, 20))
        self.operand1.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.operand1.setObjectName("operand1")
        self.gridLayout.addWidget(self.operand1, 0, 0, 1, 2)
        self.op = QtWidgets.QComboBox(self.groupBox)
        self.op.setObjectName("op")
        self.gridLayout.addWidget(self.op, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            159, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout.addItem(spacerItem, 2, 4, 1, 1)
        self.operand2 = QtWidgets.QComboBox(self.groupBox)
        self.operand2.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.operand2.setObjectName("operand2")
        self.gridLayout.addWidget(self.operand2, 2, 0, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(
            29, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout.addItem(spacerItem1, 1, 1, 1, 1)
        self.apply_btn = QtWidgets.QPushButton(self.groupBox)
        self.apply_btn.setObjectName("apply_btn")
        self.gridLayout.addWidget(self.apply_btn, 6, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout.addItem(spacerItem2, 2, 3, 1, 1)
        self.name = QtWidgets.QLineEdit(self.groupBox)
        self.name.setInputMask("")
        self.name.setText("")
        self.name.setObjectName("name")
        self.gridLayout.addWidget(self.name, 3, 0, 1, 5)
        spacerItem3 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout.addItem(spacerItem3, 2, 2, 1, 1)
        self.line = QtWidgets.QFrame(self.groupBox)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 5, 0, 1, 5)
        self.unit = QtWidgets.QLineEdit(self.groupBox)
        self.unit.setObjectName("unit")
        self.gridLayout.addWidget(self.unit, 4, 0, 1, 1)
        self.gridLayout.setColumnStretch(1, 3)
        self.gridLayout.setColumnStretch(4, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(ComputedChannel)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.function_name = QtWidgets.QLineEdit(self.groupBox_2)
        self.function_name.setObjectName("function_name")
        self.gridLayout_2.addWidget(self.function_name, 2, 0, 1, 5)
        self.channel = QtWidgets.QComboBox(self.groupBox_2)
        self.channel.setObjectName("channel")
        self.gridLayout_2.addWidget(self.channel, 1, 0, 1, 2)
        spacerItem4 = QtWidgets.QSpacerItem(
            71, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem4, 5, 4, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(
            238, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem5, 5, 1, 1, 1)
        self.function = QtWidgets.QComboBox(self.groupBox_2)
        self.function.setMinimumSize(QtCore.QSize(380, 20))
        self.function.setObjectName("function")
        self.gridLayout_2.addWidget(self.function, 0, 0, 1, 2)
        spacerItem6 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem6, 5, 2, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.groupBox_2)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_2.addWidget(self.line_2, 4, 0, 1, 5)
        self.apply_function_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.apply_function_btn.setMinimumSize(QtCore.QSize(133, 23))
        self.apply_function_btn.setObjectName("apply_function_btn")
        self.gridLayout_2.addWidget(self.apply_function_btn, 5, 0, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem7, 5, 3, 1, 1)
        self.function_unit = QtWidgets.QLineEdit(self.groupBox_2)
        self.function_unit.setObjectName("function_unit")
        self.gridLayout_2.addWidget(self.function_unit, 3, 0, 1, 1)
        self.gridLayout_2.setColumnStretch(1, 3)
        self.gridLayout_2.setColumnStretch(4, 1)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem8 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem8)
        self.cancel_btn = QtWidgets.QPushButton(ComputedChannel)
        self.cancel_btn.setObjectName("cancel_btn")
        self.horizontalLayout.addWidget(self.cancel_btn)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(ComputedChannel)
        QtCore.QMetaObject.connectSlotsByName(ComputedChannel)
        ComputedChannel.setTabOrder(self.operand1, self.op)
        ComputedChannel.setTabOrder(self.op, self.operand2)
        ComputedChannel.setTabOrder(self.operand2, self.name)
        ComputedChannel.setTabOrder(self.name, self.unit)
        ComputedChannel.setTabOrder(self.unit, self.apply_btn)
        ComputedChannel.setTabOrder(self.apply_btn, self.function)
        ComputedChannel.setTabOrder(self.function, self.apply_function_btn)
        ComputedChannel.setTabOrder(self.apply_function_btn, self.cancel_btn)

    def retranslateUi(self, ComputedChannel):
        _translate = QtCore.QCoreApplication.translate
        ComputedChannel.setWindowTitle(
            _translate("ComputedChannel", "Define new channel")
        )
        self.groupBox.setTitle(_translate("ComputedChannel", "Computation"))
        self.apply_btn.setText(_translate("ComputedChannel", "Apply"))
        self.name.setPlaceholderText(_translate("ComputedChannel", "channel name"))
        self.unit.setPlaceholderText(_translate("ComputedChannel", "channel unit"))
        self.groupBox_2.setTitle(_translate("ComputedChannel", "Function"))
        self.function_name.setPlaceholderText(
            _translate("ComputedChannel", "channel name")
        )
        self.function.setToolTip(
            _translate("ComputedChannel", "see numpy documentation")
        )
        self.apply_function_btn.setText(_translate("ComputedChannel", "Apply"))
        self.function_unit.setPlaceholderText(
            _translate("ComputedChannel", "channel unit")
        )
        self.cancel_btn.setText(_translate("ComputedChannel", "Cancel"))


from . import resource_rc

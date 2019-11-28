# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'numeric.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_NumericDisplay(object):
    def setupUi(self, NumericDisplay):
        NumericDisplay.setObjectName("NumericDisplay")
        NumericDisplay.resize(480, 666)
        self.verticalLayout = QtWidgets.QVBoxLayout(NumericDisplay)
        self.verticalLayout.setObjectName("verticalLayout")
        self.channels = NumericTreeWidget(NumericDisplay)
        font = QtGui.QFont()
        font.setFamily("Lucida Console")
        self.channels.setFont(font)
        self.channels.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.channels.setAlternatingRowColors(True)
        self.channels.setColumnCount(3)
        self.channels.setObjectName("channels")
        self.verticalLayout.addWidget(self.channels)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(NumericDisplay)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.timestamp = QtWidgets.QDoubleSpinBox(NumericDisplay)
        self.timestamp.setDecimals(9)
        self.timestamp.setSingleStep(0.001)
        self.timestamp.setObjectName("timestamp")
        self.horizontalLayout.addWidget(self.timestamp)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.min_t = QtWidgets.QLabel(NumericDisplay)
        self.min_t.setText("")
        self.min_t.setObjectName("min_t")
        self.horizontalLayout_2.addWidget(self.min_t)
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_2.addItem(spacerItem)
        self.max_t = QtWidgets.QLabel(NumericDisplay)
        self.max_t.setText("")
        self.max_t.setObjectName("max_t")
        self.horizontalLayout_2.addWidget(self.max_t)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.timestamp_slider = QtWidgets.QSlider(NumericDisplay)
        self.timestamp_slider.setMaximum(9999)
        self.timestamp_slider.setOrientation(QtCore.Qt.Horizontal)
        self.timestamp_slider.setInvertedAppearance(False)
        self.timestamp_slider.setInvertedControls(False)
        self.timestamp_slider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.timestamp_slider.setTickInterval(1)
        self.timestamp_slider.setObjectName("timestamp_slider")
        self.verticalLayout.addWidget(self.timestamp_slider)
        self.groupBox = QtWidgets.QGroupBox(NumericDisplay)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 2, 1, 1)
        self.pattern = QtWidgets.QLineEdit(self.groupBox)
        self.pattern.setObjectName("pattern")
        self.gridLayout.addWidget(self.pattern, 1, 0, 1, 1)
        self.op = QtWidgets.QComboBox(self.groupBox)
        self.op.setObjectName("op")
        self.gridLayout.addWidget(self.op, 1, 1, 1, 1)
        self.target = QtWidgets.QLineEdit(self.groupBox)
        self.target.setObjectName("target")
        self.gridLayout.addWidget(self.target, 1, 2, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.backward = QtWidgets.QPushButton(self.groupBox)
        self.backward.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/right.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.backward.setIcon(icon)
        self.backward.setObjectName("backward")
        self.horizontalLayout_3.addWidget(self.backward)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_3.addItem(spacerItem1)
        self.forward = QtWidgets.QPushButton(self.groupBox)
        self.forward.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(
            QtGui.QPixmap(":/left.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.forward.setIcon(icon1)
        self.forward.setObjectName("forward")
        self.horizontalLayout_3.addWidget(self.forward)
        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 2, 1, 1)
        self.match = QtWidgets.QLabel(self.groupBox)
        self.match.setText("")
        self.match.setObjectName("match")
        self.gridLayout.addWidget(self.match, 2, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.verticalLayout.setStretch(0, 1)

        self.retranslateUi(NumericDisplay)
        QtCore.QMetaObject.connectSlotsByName(NumericDisplay)

    def retranslateUi(self, NumericDisplay):
        _translate = QtCore.QCoreApplication.translate
        NumericDisplay.setWindowTitle(_translate("NumericDisplay", "Form"))
        self.channels.headerItem().setText(0, _translate("NumericDisplay", "Name"))
        self.channels.headerItem().setText(1, _translate("NumericDisplay", "Value"))
        self.channels.headerItem().setText(2, _translate("NumericDisplay", "Unit"))
        self.label.setText(_translate("NumericDisplay", "Time"))
        self.timestamp.setSuffix(_translate("NumericDisplay", "s"))
        self.groupBox.setTitle(_translate("NumericDisplay", "Search for values"))
        self.label_2.setText(_translate("NumericDisplay", "Pattern"))
        self.label_3.setText(_translate("NumericDisplay", "target value"))


from asammdf.gui.widgets.tree_numeric import NumericTreeWidget
from . import resource_rc

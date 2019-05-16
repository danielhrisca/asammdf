# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'numeric.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_NumericDisplay(object):
    def setupUi(self, NumericDisplay):
        NumericDisplay.setObjectName("NumericDisplay")
        NumericDisplay.resize(392, 560)
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
        self.timestamp.setDecimals(3)
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
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
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


from asammdf.gui.widgets.tree_numeric import NumericTreeWidget

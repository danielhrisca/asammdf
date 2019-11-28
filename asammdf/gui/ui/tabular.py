# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tabular.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TabularDisplay(object):
    def setupUi(self, TabularDisplay):
        TabularDisplay.setObjectName("TabularDisplay")
        TabularDisplay.resize(821, 618)
        self.gridLayout_2 = QtWidgets.QGridLayout(TabularDisplay)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.add_filter_btn = QtWidgets.QPushButton(TabularDisplay)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/plus.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.add_filter_btn.setIcon(icon)
        self.add_filter_btn.setObjectName("add_filter_btn")
        self.gridLayout_2.addWidget(self.add_filter_btn, 3, 1, 1, 1)
        self.sort = QtWidgets.QCheckBox(TabularDisplay)
        self.sort.setObjectName("sort")
        self.gridLayout_2.addWidget(self.sort, 2, 6, 1, 1)
        self.apply_filters_btn = QtWidgets.QPushButton(TabularDisplay)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(
            QtGui.QPixmap(":/filter.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.apply_filters_btn.setIcon(icon1)
        self.apply_filters_btn.setObjectName("apply_filters_btn")
        self.gridLayout_2.addWidget(self.apply_filters_btn, 3, 4, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            200, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem, 3, 5, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(
            20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.gridLayout_2.addItem(spacerItem1, 3, 2, 1, 2)
        self.remove_prefix = QtWidgets.QCheckBox(TabularDisplay)
        self.remove_prefix.setObjectName("remove_prefix")
        self.gridLayout_2.addWidget(self.remove_prefix, 3, 6, 1, 1)
        self.tree_scroll = QtWidgets.QScrollBar(TabularDisplay)
        self.tree_scroll.setMaximum(9999)
        self.tree_scroll.setSingleStep(1)
        self.tree_scroll.setPageStep(10)
        self.tree_scroll.setOrientation(QtCore.Qt.Vertical)
        self.tree_scroll.setInvertedAppearance(False)
        self.tree_scroll.setObjectName("tree_scroll")
        self.gridLayout_2.addWidget(self.tree_scroll, 0, 8, 1, 1)
        self.prefix = QtWidgets.QComboBox(TabularDisplay)
        self.prefix.setMinimumSize(QtCore.QSize(200, 0))
        self.prefix.setEditable(True)
        self.prefix.setObjectName("prefix")
        self.gridLayout_2.addWidget(self.prefix, 3, 7, 1, 2)
        self.time_as_date = QtWidgets.QCheckBox(TabularDisplay)
        self.time_as_date.setObjectName("time_as_date")
        self.gridLayout_2.addWidget(self.time_as_date, 2, 7, 1, 2)
        self.tree = QtWidgets.QTreeWidget(TabularDisplay)
        font = QtGui.QFont()
        font.setFamily("Lucida Console")
        self.tree.setFont(font)
        self.tree.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tree.setUniformRowHeights(True)
        self.tree.setObjectName("tree")
        self.tree.headerItem().setText(0, "1")
        self.gridLayout_2.addWidget(self.tree, 0, 0, 1, 8)
        self.filters = ListWidget(TabularDisplay)
        self.filters.setObjectName("filters")
        self.gridLayout_2.addWidget(self.filters, 4, 0, 1, 9)
        self.groupBox = QtWidgets.QGroupBox(TabularDisplay)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 30))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.query = QtWidgets.QTextEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.query.sizePolicy().hasHeightForWidth())
        self.query.setSizePolicy(sizePolicy)
        self.query.setMinimumSize(QtCore.QSize(0, 7))
        font = QtGui.QFont()
        font.setFamily("Lucida Console")
        self.query.setFont(font)
        self.query.setStyleSheet("background-color: rgb(186, 186, 186);")
        self.query.setReadOnly(True)
        self.query.setObjectName("query")
        self.gridLayout.addWidget(self.query, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 5, 0, 1, 9)
        self.gridLayout_2.setColumnStretch(5, 1)
        self.gridLayout_2.setRowStretch(0, 5)

        self.retranslateUi(TabularDisplay)
        QtCore.QMetaObject.connectSlotsByName(TabularDisplay)

    def retranslateUi(self, TabularDisplay):
        _translate = QtCore.QCoreApplication.translate
        TabularDisplay.setWindowTitle(_translate("TabularDisplay", "Form"))
        self.add_filter_btn.setText(_translate("TabularDisplay", "Add new filter"))
        self.sort.setText(_translate("TabularDisplay", "Enable column sorting"))
        self.apply_filters_btn.setText(_translate("TabularDisplay", "Apply filters"))
        self.remove_prefix.setText(_translate("TabularDisplay", "Remove prefix"))
        self.time_as_date.setText(_translate("TabularDisplay", "Time as date"))
        self.groupBox.setTitle(_translate("TabularDisplay", "pandas DataFrame query"))


from asammdf.gui.widgets.list import ListWidget
from . import resource_rc

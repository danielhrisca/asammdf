# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'tabular.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
    QGroupBox, QHBoxLayout, QListWidgetItem, QPushButton,
    QSizePolicy, QSpacerItem, QTextEdit, QVBoxLayout,
    QWidget)

from asammdf.gui.widgets.list import MinimalListWidget
from . import resource_rc

class Ui_TabularDisplay(object):
    def setupUi(self, TabularDisplay):
        if not TabularDisplay.objectName():
            TabularDisplay.setObjectName(u"TabularDisplay")
        TabularDisplay.resize(821, 618)
        self.verticalLayout = QVBoxLayout(TabularDisplay)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.format_selection = QComboBox(TabularDisplay)
        self.format_selection.addItem("")
        self.format_selection.addItem("")
        self.format_selection.addItem("")
        self.format_selection.setObjectName(u"format_selection")

        self.horizontalLayout_2.addWidget(self.format_selection)

        self.float_precision = QComboBox(TabularDisplay)
        self.float_precision.setObjectName(u"float_precision")

        self.horizontalLayout_2.addWidget(self.float_precision)

        self.time_as_date = QCheckBox(TabularDisplay)
        self.time_as_date.setObjectName(u"time_as_date")

        self.horizontalLayout_2.addWidget(self.time_as_date)

        self.remove_prefix = QCheckBox(TabularDisplay)
        self.remove_prefix.setObjectName(u"remove_prefix")

        self.horizontalLayout_2.addWidget(self.remove_prefix)

        self.prefix = QComboBox(TabularDisplay)
        self.prefix.setObjectName(u"prefix")
        self.prefix.setMinimumSize(QSize(200, 0))
        self.prefix.setEditable(True)

        self.horizontalLayout_2.addWidget(self.prefix)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_3)

        self.toggle_filters_btn = QPushButton(TabularDisplay)
        self.toggle_filters_btn.setObjectName(u"toggle_filters_btn")
        icon = QIcon()
        icon.addFile(u":/down.png", QSize(), QIcon.Normal, QIcon.Off)
        self.toggle_filters_btn.setIcon(icon)

        self.horizontalLayout_2.addWidget(self.toggle_filters_btn)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.filters_group = QGroupBox(TabularDisplay)
        self.filters_group.setObjectName(u"filters_group")
        self.gridLayout_3 = QGridLayout(self.filters_group)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.add_filter_btn = QPushButton(self.filters_group)
        self.add_filter_btn.setObjectName(u"add_filter_btn")
        icon1 = QIcon()
        icon1.addFile(u":/plus.png", QSize(), QIcon.Normal, QIcon.Off)
        self.add_filter_btn.setIcon(icon1)

        self.gridLayout_3.addWidget(self.add_filter_btn, 0, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.apply_filters_btn = QPushButton(self.filters_group)
        self.apply_filters_btn.setObjectName(u"apply_filters_btn")
        icon2 = QIcon()
        icon2.addFile(u":/filter.png", QSize(), QIcon.Normal, QIcon.Off)
        self.apply_filters_btn.setIcon(icon2)

        self.gridLayout_3.addWidget(self.apply_filters_btn, 0, 2, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(559, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_2, 0, 3, 1, 1)

        self.filters = MinimalListWidget(self.filters_group)
        self.filters.setObjectName(u"filters")
        self.filters.setMaximumSize(QSize(16777215, 150))

        self.gridLayout_3.addWidget(self.filters, 1, 0, 1, 4)

        self.groupBox = QGroupBox(self.filters_group)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QSize(0, 30))
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.query = QTextEdit(self.groupBox)
        self.query.setObjectName(u"query")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.query.sizePolicy().hasHeightForWidth())
        self.query.setSizePolicy(sizePolicy1)
        self.query.setMinimumSize(QSize(0, 7))
        self.query.setMaximumSize(QSize(16777215, 150))
        font = QFont()
        font.setFamilies([u"Lucida Console"])
        self.query.setFont(font)
        self.query.setStyleSheet(u"background-color: rgb(186, 186, 186);")
        self.query.setReadOnly(True)

        self.gridLayout.addWidget(self.query, 0, 0, 1, 1)


        self.gridLayout_3.addWidget(self.groupBox, 2, 0, 1, 4)

        self.gridLayout_3.setColumnStretch(3, 1)

        self.verticalLayout.addWidget(self.filters_group)

        self.verticalLayout.setStretch(0, 1)
        QWidget.setTabOrder(self.format_selection, self.float_precision)
        QWidget.setTabOrder(self.float_precision, self.time_as_date)
        QWidget.setTabOrder(self.time_as_date, self.remove_prefix)
        QWidget.setTabOrder(self.remove_prefix, self.prefix)
        QWidget.setTabOrder(self.prefix, self.toggle_filters_btn)
        QWidget.setTabOrder(self.toggle_filters_btn, self.add_filter_btn)
        QWidget.setTabOrder(self.add_filter_btn, self.apply_filters_btn)
        QWidget.setTabOrder(self.apply_filters_btn, self.filters)
        QWidget.setTabOrder(self.filters, self.query)

        self.retranslateUi(TabularDisplay)

        QMetaObject.connectSlotsByName(TabularDisplay)
    # setupUi

    def retranslateUi(self, TabularDisplay):
        TabularDisplay.setWindowTitle(QCoreApplication.translate("TabularDisplay", u"Form", None))
        self.format_selection.setItemText(0, QCoreApplication.translate("TabularDisplay", u"phys", None))
        self.format_selection.setItemText(1, QCoreApplication.translate("TabularDisplay", u"hex", None))
        self.format_selection.setItemText(2, QCoreApplication.translate("TabularDisplay", u"bin", None))

        self.time_as_date.setText(QCoreApplication.translate("TabularDisplay", u"Time as date", None))
        self.remove_prefix.setText(QCoreApplication.translate("TabularDisplay", u"Remove prefix", None))
        self.toggle_filters_btn.setText(QCoreApplication.translate("TabularDisplay", u"Show filters", None))
        self.filters_group.setTitle(QCoreApplication.translate("TabularDisplay", u"Filters", None))
        self.add_filter_btn.setText(QCoreApplication.translate("TabularDisplay", u"Add new filter", None))
        self.apply_filters_btn.setText(QCoreApplication.translate("TabularDisplay", u"Apply filters", None))
        self.groupBox.setTitle(QCoreApplication.translate("TabularDisplay", u"pandas DataFrame query", None))
    # retranslateUi


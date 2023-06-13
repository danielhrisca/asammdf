# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'define_conversion_dialog.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDoubleSpinBox,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidgetItem, QPlainTextEdit, QPushButton, QSizePolicy,
    QSpacerItem, QStackedWidget, QTabWidget, QVBoxLayout,
    QWidget)

from asammdf.gui.widgets.list import MinimalListWidget
from . import resource_rc

class Ui_ConversionDialog(object):
    def setupUi(self, ConversionDialog):
        if not ConversionDialog.objectName():
            ConversionDialog.setObjectName(u"ConversionDialog")
        ConversionDialog.resize(937, 501)
        ConversionDialog.setMaximumSize(QSize(16777215, 16777215))
        icon = QIcon()
        icon.addFile(u":/plus.png", QSize(), QIcon.Normal, QIcon.Off)
        ConversionDialog.setWindowIcon(icon)
        ConversionDialog.setSizeGripEnabled(True)
        self.gridLayout = QGridLayout(ConversionDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_6 = QLabel(ConversionDialog)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)

        self.unit = QLineEdit(ConversionDialog)
        self.unit.setObjectName(u"unit")

        self.gridLayout.addWidget(self.unit, 4, 1, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_4, 5, 2, 1, 1)

        self.label_8 = QLabel(ConversionDialog)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout.addWidget(self.label_8, 3, 0, 1, 1)

        self.label_7 = QLabel(ConversionDialog)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 1, 0, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_3, 5, 1, 1, 1)

        self.cancel_btn = QPushButton(ConversionDialog)
        self.cancel_btn.setObjectName(u"cancel_btn")

        self.gridLayout.addWidget(self.cancel_btn, 5, 4, 1, 1)

        self.comment = QPlainTextEdit(ConversionDialog)
        self.comment.setObjectName(u"comment")

        self.gridLayout.addWidget(self.comment, 3, 1, 1, 4)

        self.apply_btn = QPushButton(ConversionDialog)
        self.apply_btn.setObjectName(u"apply_btn")

        self.gridLayout.addWidget(self.apply_btn, 5, 3, 1, 1)

        self.name = QLineEdit(ConversionDialog)
        self.name.setObjectName(u"name")

        self.gridLayout.addWidget(self.name, 1, 1, 1, 4)

        self.tabs = QTabWidget(ConversionDialog)
        self.tabs.setObjectName(u"tabs")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_5 = QGridLayout(self.tab)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label_10 = QLabel(self.tab)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_5.addWidget(self.label_10, 2, 0, 1, 1)

        self.b = QDoubleSpinBox(self.tab)
        self.b.setObjectName(u"b")
        self.b.setDecimals(6)

        self.gridLayout_5.addWidget(self.b, 2, 1, 1, 1)

        self.label_11 = QLabel(self.tab)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_5.addWidget(self.label_11, 0, 0, 1, 2)

        self.label_9 = QLabel(self.tab)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_5.addWidget(self.label_9, 1, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_5.addItem(self.horizontalSpacer, 1, 2, 1, 1)

        self.a = QDoubleSpinBox(self.tab)
        self.a.setObjectName(u"a")
        self.a.setMinimumSize(QSize(100, 0))
        self.a.setDecimals(6)

        self.gridLayout_5.addWidget(self.a, 1, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_5.addItem(self.verticalSpacer, 3, 0, 1, 1)

        self.tabs.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.gridLayout_6 = QGridLayout(self.tab_2)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.label_3 = QLabel(self.tab_2)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_6.addWidget(self.label_3, 1, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_6.addItem(self.verticalSpacer_2, 7, 0, 1, 1)

        self.p3 = QDoubleSpinBox(self.tab_2)
        self.p3.setObjectName(u"p3")
        self.p3.setDecimals(6)

        self.gridLayout_6.addWidget(self.p3, 3, 1, 1, 1)

        self.p1 = QDoubleSpinBox(self.tab_2)
        self.p1.setObjectName(u"p1")
        self.p1.setMinimumSize(QSize(100, 0))
        self.p1.setDecimals(6)

        self.gridLayout_6.addWidget(self.p1, 1, 1, 1, 1)

        self.p2 = QDoubleSpinBox(self.tab_2)
        self.p2.setObjectName(u"p2")
        self.p2.setDecimals(6)

        self.gridLayout_6.addWidget(self.p2, 2, 1, 1, 1)

        self.label_4 = QLabel(self.tab_2)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_6.addWidget(self.label_4, 2, 0, 1, 1)

        self.label_15 = QLabel(self.tab_2)
        self.label_15.setObjectName(u"label_15")

        self.gridLayout_6.addWidget(self.label_15, 6, 0, 1, 1)

        self.label_13 = QLabel(self.tab_2)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout_6.addWidget(self.label_13, 4, 0, 1, 1)

        self.p6 = QDoubleSpinBox(self.tab_2)
        self.p6.setObjectName(u"p6")
        self.p6.setDecimals(6)

        self.gridLayout_6.addWidget(self.p6, 6, 1, 1, 1)

        self.p5 = QDoubleSpinBox(self.tab_2)
        self.p5.setObjectName(u"p5")
        self.p5.setDecimals(6)

        self.gridLayout_6.addWidget(self.p5, 5, 1, 1, 1)

        self.label_5 = QLabel(self.tab_2)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_6.addWidget(self.label_5, 3, 0, 1, 1)

        self.p4 = QDoubleSpinBox(self.tab_2)
        self.p4.setObjectName(u"p4")
        self.p4.setDecimals(6)

        self.gridLayout_6.addWidget(self.p4, 4, 1, 1, 1)

        self.label_14 = QLabel(self.tab_2)
        self.label_14.setObjectName(u"label_14")

        self.gridLayout_6.addWidget(self.label_14, 5, 0, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_6.addItem(self.horizontalSpacer_2, 1, 2, 1, 1)

        self.label_2 = QLabel(self.tab_2)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_6.addWidget(self.label_2, 0, 0, 1, 3)

        self.tabs.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.gridLayout_2 = QGridLayout(self.tab_3)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.vtt_list = MinimalListWidget(self.tab_3)
        self.vtt_list.setObjectName(u"vtt_list")

        self.gridLayout_2.addWidget(self.vtt_list, 1, 0, 4, 4)

        self.vtt_mode = QStackedWidget(self.tab_3)
        self.vtt_mode.setObjectName(u"vtt_mode")
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.horizontalLayout = QHBoxLayout(self.page)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.vtt_default = QLineEdit(self.page)
        self.vtt_default.setObjectName(u"vtt_default")

        self.horizontalLayout.addWidget(self.vtt_default)

        self.vtt_mode.addWidget(self.page)
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.horizontalLayout_2 = QHBoxLayout(self.page_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.vtt_default_btn = QPushButton(self.page_2)
        self.vtt_default_btn.setObjectName(u"vtt_default_btn")

        self.horizontalLayout_2.addWidget(self.vtt_default_btn)

        self.vtt_mode.addWidget(self.page_2)

        self.gridLayout_2.addWidget(self.vtt_mode, 5, 2, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_3, 2, 4, 1, 1)

        self.insert_btn = QPushButton(self.tab_3)
        self.insert_btn.setObjectName(u"insert_btn")

        self.gridLayout_2.addWidget(self.insert_btn, 1, 4, 1, 1)

        self.label_16 = QLabel(self.tab_3)
        self.label_16.setObjectName(u"label_16")

        self.gridLayout_2.addWidget(self.label_16, 5, 0, 1, 1)

        self.label = QLabel(self.tab_3)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 4)

        self.reset_btn = QPushButton(self.tab_3)
        self.reset_btn.setObjectName(u"reset_btn")

        self.gridLayout_2.addWidget(self.reset_btn, 4, 4, 1, 1)

        self.vtt_default_mode = QComboBox(self.tab_3)
        self.vtt_default_mode.addItem("")
        self.vtt_default_mode.addItem("")
        self.vtt_default_mode.setObjectName(u"vtt_default_mode")

        self.gridLayout_2.addWidget(self.vtt_default_mode, 5, 1, 1, 1)

        self.tabs.addTab(self.tab_3, "")
        self.tab_4 = QWidget()
        self.tab_4.setObjectName(u"tab_4")
        self.gridLayout_3 = QGridLayout(self.tab_4)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.vrtt_list = MinimalListWidget(self.tab_4)
        self.vrtt_list.setObjectName(u"vrtt_list")

        self.gridLayout_3.addWidget(self.vrtt_list, 1, 0, 3, 4)

        self.label_12 = QLabel(self.tab_4)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout_3.addWidget(self.label_12, 0, 0, 1, 4)

        self.vrtt_mode = QStackedWidget(self.tab_4)
        self.vrtt_mode.setObjectName(u"vrtt_mode")
        self.page_3 = QWidget()
        self.page_3.setObjectName(u"page_3")
        self.horizontalLayout_3 = QHBoxLayout(self.page_3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.vrtt_default = QLineEdit(self.page_3)
        self.vrtt_default.setObjectName(u"vrtt_default")

        self.horizontalLayout_3.addWidget(self.vrtt_default)

        self.vrtt_mode.addWidget(self.page_3)
        self.page_4 = QWidget()
        self.page_4.setObjectName(u"page_4")
        self.horizontalLayout_4 = QHBoxLayout(self.page_4)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.vrtt_default_btn = QPushButton(self.page_4)
        self.vrtt_default_btn.setObjectName(u"vrtt_default_btn")

        self.horizontalLayout_4.addWidget(self.vrtt_default_btn)

        self.vrtt_mode.addWidget(self.page_4)

        self.gridLayout_3.addWidget(self.vrtt_mode, 4, 2, 1, 1)

        self.insert_vrtt_btn = QPushButton(self.tab_4)
        self.insert_vrtt_btn.setObjectName(u"insert_vrtt_btn")

        self.gridLayout_3.addWidget(self.insert_vrtt_btn, 1, 4, 1, 1)

        self.verticalSpacer_4 = QSpacerItem(20, 78, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer_4, 2, 4, 1, 1)

        self.reset_vrtt_btn = QPushButton(self.tab_4)
        self.reset_vrtt_btn.setObjectName(u"reset_vrtt_btn")

        self.gridLayout_3.addWidget(self.reset_vrtt_btn, 3, 4, 1, 1)

        self.label_17 = QLabel(self.tab_4)
        self.label_17.setObjectName(u"label_17")

        self.gridLayout_3.addWidget(self.label_17, 4, 0, 1, 1)

        self.vrtt_default_mode = QComboBox(self.tab_4)
        self.vrtt_default_mode.addItem("")
        self.vrtt_default_mode.addItem("")
        self.vrtt_default_mode.setObjectName(u"vrtt_default_mode")

        self.gridLayout_3.addWidget(self.vrtt_default_mode, 4, 1, 1, 1)

        self.tabs.addTab(self.tab_4, "")
        self.tab_5 = QWidget()
        self.tab_5.setObjectName(u"tab_5")
        self.verticalLayout = QVBoxLayout(self.tab_5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_18 = QLabel(self.tab_5)
        self.label_18.setObjectName(u"label_18")

        self.verticalLayout.addWidget(self.label_18)

        self.verticalSpacer_5 = QSpacerItem(20, 166, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_5)

        self.tabs.addTab(self.tab_5, "")

        self.gridLayout.addWidget(self.tabs, 0, 0, 1, 5)

        self.gridLayout.setRowStretch(0, 1)
        QWidget.setTabOrder(self.tabs, self.a)
        QWidget.setTabOrder(self.a, self.b)
        QWidget.setTabOrder(self.b, self.p1)
        QWidget.setTabOrder(self.p1, self.p2)
        QWidget.setTabOrder(self.p2, self.p3)
        QWidget.setTabOrder(self.p3, self.p4)
        QWidget.setTabOrder(self.p4, self.p5)
        QWidget.setTabOrder(self.p5, self.p6)
        QWidget.setTabOrder(self.p6, self.vtt_list)
        QWidget.setTabOrder(self.vtt_list, self.insert_btn)
        QWidget.setTabOrder(self.insert_btn, self.reset_btn)
        QWidget.setTabOrder(self.reset_btn, self.vtt_default_mode)
        QWidget.setTabOrder(self.vtt_default_mode, self.vtt_default)
        QWidget.setTabOrder(self.vtt_default, self.vrtt_list)
        QWidget.setTabOrder(self.vrtt_list, self.insert_vrtt_btn)
        QWidget.setTabOrder(self.insert_vrtt_btn, self.reset_vrtt_btn)
        QWidget.setTabOrder(self.reset_vrtt_btn, self.vrtt_default_mode)
        QWidget.setTabOrder(self.vrtt_default_mode, self.vrtt_default)
        QWidget.setTabOrder(self.vrtt_default, self.name)
        QWidget.setTabOrder(self.name, self.comment)
        QWidget.setTabOrder(self.comment, self.unit)
        QWidget.setTabOrder(self.unit, self.apply_btn)
        QWidget.setTabOrder(self.apply_btn, self.cancel_btn)
        QWidget.setTabOrder(self.cancel_btn, self.vrtt_default_btn)
        QWidget.setTabOrder(self.vrtt_default_btn, self.vtt_default_btn)

        self.retranslateUi(ConversionDialog)

        self.tabs.setCurrentIndex(0)
        self.vtt_mode.setCurrentIndex(0)
        self.vrtt_mode.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(ConversionDialog)
    # setupUi

    def retranslateUi(self, ConversionDialog):
        ConversionDialog.setWindowTitle(QCoreApplication.translate("ConversionDialog", u"Define new channel", None))
        self.label_6.setText(QCoreApplication.translate("ConversionDialog", u"Conversion unit", None))
        self.unit.setPlaceholderText("")
        self.label_8.setText(QCoreApplication.translate("ConversionDialog", u"Conversion comment", None))
        self.label_7.setText(QCoreApplication.translate("ConversionDialog", u"Conversion name", None))
        self.cancel_btn.setText(QCoreApplication.translate("ConversionDialog", u"Cancel", None))
        self.apply_btn.setText(QCoreApplication.translate("ConversionDialog", u"Apply", None))
        self.name.setInputMask("")
        self.name.setText("")
        self.name.setPlaceholderText("")
        self.label_10.setText(QCoreApplication.translate("ConversionDialog", u"Offset (b)", None))
        self.label_11.setText(QCoreApplication.translate("ConversionDialog", u"Y = a * X + b", None))
        self.label_9.setText(QCoreApplication.translate("ConversionDialog", u"Factor (a)", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab), QCoreApplication.translate("ConversionDialog", u"Linear", None))
        self.label_3.setText(QCoreApplication.translate("ConversionDialog", u"P1", None))
        self.label_4.setText(QCoreApplication.translate("ConversionDialog", u"P2", None))
        self.label_15.setText(QCoreApplication.translate("ConversionDialog", u"P6", None))
        self.label_13.setText(QCoreApplication.translate("ConversionDialog", u"P4", None))
        self.label_5.setText(QCoreApplication.translate("ConversionDialog", u"P3", None))
        self.label_14.setText(QCoreApplication.translate("ConversionDialog", u"P5", None))
        self.label_2.setText(QCoreApplication.translate("ConversionDialog", u"Y = (P1 * X^2 + P2 * X +P3) / (P4 * X^2 + P5 * X + P6)", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_2), QCoreApplication.translate("ConversionDialog", u"Rational", None))
        self.vtt_default_btn.setText(QCoreApplication.translate("ConversionDialog", u"Default conversion", None))
        self.insert_btn.setText(QCoreApplication.translate("ConversionDialog", u"Insert", None))
        self.label_16.setText(QCoreApplication.translate("ConversionDialog", u"Default", None))
        self.label.setText(QCoreApplication.translate("ConversionDialog", u"X -> Y (str/float) ", None))
        self.reset_btn.setText(QCoreApplication.translate("ConversionDialog", u"Reset", None))
        self.vtt_default_mode.setItemText(0, QCoreApplication.translate("ConversionDialog", u"Text", None))
        self.vtt_default_mode.setItemText(1, QCoreApplication.translate("ConversionDialog", u"Conversion", None))

        self.tabs.setTabText(self.tabs.indexOf(self.tab_3), QCoreApplication.translate("ConversionDialog", u"Value to Text/Conversion", None))
        self.label_12.setText(QCoreApplication.translate("ConversionDialog", u"[ .. X .. )  -> Y (str/float) ", None))
        self.vrtt_default_btn.setText(QCoreApplication.translate("ConversionDialog", u"Default conversion", None))
        self.insert_vrtt_btn.setText(QCoreApplication.translate("ConversionDialog", u"Insert", None))
        self.reset_vrtt_btn.setText(QCoreApplication.translate("ConversionDialog", u"Reset", None))
        self.label_17.setText(QCoreApplication.translate("ConversionDialog", u"Default", None))
        self.vrtt_default_mode.setItemText(0, QCoreApplication.translate("ConversionDialog", u"Text", None))
        self.vrtt_default_mode.setItemText(1, QCoreApplication.translate("ConversionDialog", u"Conversion", None))

        self.tabs.setTabText(self.tabs.indexOf(self.tab_4), QCoreApplication.translate("ConversionDialog", u"Value range to Text/Conversion", None))
        self.label_18.setText(QCoreApplication.translate("ConversionDialog", u"X -> X", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_5), QCoreApplication.translate("ConversionDialog", u"1:1", None))
    # retranslateUi


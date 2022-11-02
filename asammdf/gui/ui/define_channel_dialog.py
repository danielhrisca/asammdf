# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'define_channel_dialog.ui'
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
    QGridLayout, QGroupBox, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QRadioButton, QSizePolicy,
    QSpacerItem, QWidget)
from . import resource_rc

class Ui_ComputedChannel(object):
    def setupUi(self, ComputedChannel):
        if not ComputedChannel.objectName():
            ComputedChannel.setObjectName(u"ComputedChannel")
        ComputedChannel.resize(638, 382)
        ComputedChannel.setMaximumSize(QSize(16777215, 16777215))
        icon = QIcon()
        icon.addFile(u":/plus.png", QSize(), QIcon.Normal, QIcon.Off)
        ComputedChannel.setWindowIcon(icon)
        ComputedChannel.setSizeGripEnabled(True)
        self.gridLayout = QGridLayout(ComputedChannel)
        self.gridLayout.setObjectName(u"gridLayout")
        self.cancel_btn = QPushButton(ComputedChannel)
        self.cancel_btn.setObjectName(u"cancel_btn")

        self.gridLayout.addWidget(self.cancel_btn, 6, 4, 1, 1)

        self.label_8 = QLabel(ComputedChannel)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout.addWidget(self.label_8, 4, 0, 1, 1)

        self.functions = QComboBox(ComputedChannel)
        self.functions.setObjectName(u"functions")

        self.gridLayout.addWidget(self.functions, 1, 1, 1, 4)

        self.label_9 = QLabel(ComputedChannel)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout.addWidget(self.label_9, 1, 0, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_3, 6, 1, 1, 1)

        self.comment = QPlainTextEdit(ComputedChannel)
        self.comment.setObjectName(u"comment")

        self.gridLayout.addWidget(self.comment, 4, 1, 1, 4)

        self.unit = QLineEdit(ComputedChannel)
        self.unit.setObjectName(u"unit")

        self.gridLayout.addWidget(self.unit, 5, 1, 1, 1)

        self.label_7 = QLabel(ComputedChannel)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 2, 0, 1, 1)

        self.apply_btn = QPushButton(ComputedChannel)
        self.apply_btn.setObjectName(u"apply_btn")

        self.gridLayout.addWidget(self.apply_btn, 6, 3, 1, 1)

        self.name = QLineEdit(ComputedChannel)
        self.name.setObjectName(u"name")

        self.gridLayout.addWidget(self.name, 2, 1, 1, 4)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_7, 6, 2, 1, 1)

        self.groupBox_3 = QGroupBox(ComputedChannel)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_8 = QGridLayout(self.groupBox_3)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_8.addItem(self.horizontalSpacer_8, 1, 2, 1, 1)

        self.trigger_interval = QDoubleSpinBox(self.groupBox_3)
        self.trigger_interval.setObjectName(u"trigger_interval")
        self.trigger_interval.setDecimals(6)
        self.trigger_interval.setMinimum(0.000001000000000)
        self.trigger_interval.setMaximum(10.000000000000000)
        self.trigger_interval.setSingleStep(0.100000000000000)
        self.trigger_interval.setValue(0.010000000000000)

        self.gridLayout_8.addWidget(self.trigger_interval, 1, 1, 1, 1)

        self.trigger_channel = QLineEdit(self.groupBox_3)
        self.trigger_channel.setObjectName(u"trigger_channel")

        self.gridLayout_8.addWidget(self.trigger_channel, 4, 1, 1, 2)

        self.triggering_on_channel = QRadioButton(self.groupBox_3)
        self.triggering_on_channel.setObjectName(u"triggering_on_channel")

        self.gridLayout_8.addWidget(self.triggering_on_channel, 4, 0, 1, 1)

        self.triggering_on_interval = QRadioButton(self.groupBox_3)
        self.triggering_on_interval.setObjectName(u"triggering_on_interval")

        self.gridLayout_8.addWidget(self.triggering_on_interval, 1, 0, 1, 1)

        self.trigger_search_btn = QPushButton(self.groupBox_3)
        self.trigger_search_btn.setObjectName(u"trigger_search_btn")
        icon1 = QIcon()
        icon1.addFile(u":/search.png", QSize(), QIcon.Normal, QIcon.Off)
        self.trigger_search_btn.setIcon(icon1)

        self.gridLayout_8.addWidget(self.trigger_search_btn, 4, 3, 1, 1)

        self.triggering_on_all = QRadioButton(self.groupBox_3)
        self.triggering_on_all.setObjectName(u"triggering_on_all")
        self.triggering_on_all.setChecked(True)

        self.gridLayout_8.addWidget(self.triggering_on_all, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.groupBox_3, 0, 0, 1, 5)

        self.label_6 = QLabel(ComputedChannel)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 5, 0, 1, 1)


        self.retranslateUi(ComputedChannel)

        QMetaObject.connectSlotsByName(ComputedChannel)
    # setupUi

    def retranslateUi(self, ComputedChannel):
        ComputedChannel.setWindowTitle(QCoreApplication.translate("ComputedChannel", u"Define new channel", None))
        self.cancel_btn.setText(QCoreApplication.translate("ComputedChannel", u"Cancel", None))
        self.label_8.setText(QCoreApplication.translate("ComputedChannel", u"Computed channel comment", None))
        self.label_9.setText(QCoreApplication.translate("ComputedChannel", u"Function", None))
        self.unit.setPlaceholderText("")
        self.label_7.setText(QCoreApplication.translate("ComputedChannel", u"Computed channel name", None))
        self.apply_btn.setText(QCoreApplication.translate("ComputedChannel", u"Apply", None))
        self.name.setInputMask("")
        self.name.setText("")
        self.name.setPlaceholderText("")
        self.groupBox_3.setTitle(QCoreApplication.translate("ComputedChannel", u"Triggering", None))
        self.trigger_interval.setSuffix(QCoreApplication.translate("ComputedChannel", u"s", None))
        self.triggering_on_channel.setText(QCoreApplication.translate("ComputedChannel", u"on channel", None))
        self.triggering_on_interval.setText(QCoreApplication.translate("ComputedChannel", u"time interval", None))
        self.trigger_search_btn.setText("")
        self.triggering_on_all.setText(QCoreApplication.translate("ComputedChannel", u"all channels timestamps", None))
        self.label_6.setText(QCoreApplication.translate("ComputedChannel", u"Computed channel unit", None))
    # retranslateUi


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
    QFrame, QGridLayout, QGroupBox, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)
from . import resource_rc

class Ui_ComputedChannel(object):
    def setupUi(self, ComputedChannel):
        if not ComputedChannel.objectName():
            ComputedChannel.setObjectName(u"ComputedChannel")
        ComputedChannel.resize(723, 552)
        ComputedChannel.setMaximumSize(QSize(16777215, 16777215))
        icon = QIcon()
        icon.addFile(u":/plus.png", QSize(), QIcon.Normal, QIcon.Off)
        ComputedChannel.setWindowIcon(icon)
        ComputedChannel.setSizeGripEnabled(True)
        self.gridLayout = QGridLayout(ComputedChannel)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(4, 4, 4, 4)
        self.comment = QPlainTextEdit(ComputedChannel)
        self.comment.setObjectName(u"comment")

        self.gridLayout.addWidget(self.comment, 5, 1, 1, 4)

        self.apply_btn = QPushButton(ComputedChannel)
        self.apply_btn.setObjectName(u"apply_btn")

        self.gridLayout.addWidget(self.apply_btn, 7, 3, 1, 1)

        self.groupBox_3 = QGroupBox(ComputedChannel)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_8 = QGridLayout(self.groupBox_3)
        self.gridLayout_8.setSpacing(2)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_8.setContentsMargins(4, 4, 4, 4)
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

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_3, 7, 1, 1, 1)

        self.cancel_btn = QPushButton(ComputedChannel)
        self.cancel_btn.setObjectName(u"cancel_btn")

        self.gridLayout.addWidget(self.cancel_btn, 7, 4, 1, 1)

        self.label_6 = QLabel(ComputedChannel)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 6, 0, 1, 1)

        self.label_8 = QLabel(ComputedChannel)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.gridLayout.addWidget(self.label_8, 5, 0, 1, 1)

        self.unit = QLineEdit(ComputedChannel)
        self.unit.setObjectName(u"unit")

        self.gridLayout.addWidget(self.unit, 6, 1, 1, 1)

        self.label_7 = QLabel(ComputedChannel)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 3, 0, 1, 1)

        self.name = QLineEdit(ComputedChannel)
        self.name.setObjectName(u"name")

        self.gridLayout.addWidget(self.name, 3, 1, 1, 4)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_7, 7, 2, 1, 1)

        self.groupBox = QGroupBox(ComputedChannel)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout = QVBoxLayout(self.groupBox)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.scrollArea = QScrollArea(self.groupBox)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setFrameShape(QFrame.NoFrame)
        self.scrollArea.setLineWidth(0)
        self.scrollArea.setWidgetResizable(True)
        self.args_widget = QWidget()
        self.args_widget.setObjectName(u"args_widget")
        self.args_widget.setGeometry(QRect(0, 0, 713, 69))
        self.arg_layout = QGridLayout(self.args_widget)
        self.arg_layout.setSpacing(2)
        self.arg_layout.setObjectName(u"arg_layout")
        self.arg_layout.setContentsMargins(9, 4, 4, 4)
        self.show_definition_btn = QPushButton(self.args_widget)
        self.show_definition_btn.setObjectName(u"show_definition_btn")
        icon2 = QIcon()
        icon2.addFile(u":/info.png", QSize(), QIcon.Normal, QIcon.Off)
        self.show_definition_btn.setIcon(icon2)

        self.arg_layout.addWidget(self.show_definition_btn, 0, 2, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.arg_layout.addItem(self.horizontalSpacer, 1, 1, 1, 1)

        self.label_9 = QLabel(self.args_widget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setMinimumSize(QSize(155, 0))

        self.arg_layout.addWidget(self.label_9, 0, 0, 1, 1)

        self.functions = QComboBox(self.args_widget)
        self.functions.setObjectName(u"functions")

        self.arg_layout.addWidget(self.functions, 0, 1, 1, 1)

        self.label = QLabel(self.args_widget)
        self.label.setObjectName(u"label")

        self.arg_layout.addWidget(self.label, 1, 0, 1, 1)

        self.scrollArea.setWidget(self.args_widget)

        self.verticalLayout.addWidget(self.scrollArea)


        self.gridLayout.addWidget(self.groupBox, 2, 0, 1, 5)

        self.groupBox_2 = QGroupBox(ComputedChannel)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setHorizontalSpacing(2)
        self.gridLayout_2.setVerticalSpacing(12)
        self.gridLayout_2.setContentsMargins(4, 4, 4, 4)
        self.sample_by_sample = QRadioButton(self.groupBox_2)
        self.sample_by_sample.setObjectName(u"sample_by_sample")
        self.sample_by_sample.setChecked(True)

        self.gridLayout_2.addWidget(self.sample_by_sample, 0, 0, 1, 1)

        self.label_24 = QLabel(self.groupBox_2)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setTextFormat(Qt.RichText)
        self.label_24.setWordWrap(False)

        self.gridLayout_2.addWidget(self.label_24, 0, 1, 1, 1)

        self.complete_signal = QRadioButton(self.groupBox_2)
        self.complete_signal.setObjectName(u"complete_signal")
        self.complete_signal.setChecked(False)

        self.gridLayout_2.addWidget(self.complete_signal, 1, 0, 1, 1)

        self.label_25 = QLabel(self.groupBox_2)
        self.label_25.setObjectName(u"label_25")

        self.gridLayout_2.addWidget(self.label_25, 1, 1, 1, 1)

        self.gridLayout_2.setColumnStretch(1, 1)

        self.gridLayout.addWidget(self.groupBox_2, 1, 0, 1, 5)

        self.gridLayout.setRowStretch(2, 1)
        QWidget.setTabOrder(self.triggering_on_all, self.triggering_on_interval)
        QWidget.setTabOrder(self.triggering_on_interval, self.trigger_interval)
        QWidget.setTabOrder(self.trigger_interval, self.triggering_on_channel)
        QWidget.setTabOrder(self.triggering_on_channel, self.trigger_channel)
        QWidget.setTabOrder(self.trigger_channel, self.trigger_search_btn)
        QWidget.setTabOrder(self.trigger_search_btn, self.sample_by_sample)
        QWidget.setTabOrder(self.sample_by_sample, self.complete_signal)
        QWidget.setTabOrder(self.complete_signal, self.scrollArea)
        QWidget.setTabOrder(self.scrollArea, self.functions)
        QWidget.setTabOrder(self.functions, self.show_definition_btn)
        QWidget.setTabOrder(self.show_definition_btn, self.name)
        QWidget.setTabOrder(self.name, self.comment)
        QWidget.setTabOrder(self.comment, self.unit)
        QWidget.setTabOrder(self.unit, self.apply_btn)
        QWidget.setTabOrder(self.apply_btn, self.cancel_btn)

        self.retranslateUi(ComputedChannel)

        QMetaObject.connectSlotsByName(ComputedChannel)
    # setupUi

    def retranslateUi(self, ComputedChannel):
        ComputedChannel.setWindowTitle(QCoreApplication.translate("ComputedChannel", u"Define new channel", None))
        self.apply_btn.setText(QCoreApplication.translate("ComputedChannel", u"Apply", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("ComputedChannel", u"Triggering", None))
        self.trigger_interval.setSuffix(QCoreApplication.translate("ComputedChannel", u"s", None))
        self.triggering_on_channel.setText(QCoreApplication.translate("ComputedChannel", u"on channel", None))
        self.triggering_on_interval.setText(QCoreApplication.translate("ComputedChannel", u"time interval", None))
        self.trigger_search_btn.setText("")
        self.triggering_on_all.setText(QCoreApplication.translate("ComputedChannel", u"all channels timestamps", None))
        self.cancel_btn.setText(QCoreApplication.translate("ComputedChannel", u"Cancel", None))
        self.label_6.setText(QCoreApplication.translate("ComputedChannel", u"Computed channel unit", None))
        self.label_8.setText(QCoreApplication.translate("ComputedChannel", u"Computed channel comment", None))
        self.unit.setPlaceholderText("")
        self.label_7.setText(QCoreApplication.translate("ComputedChannel", u"Computed channel name", None))
        self.name.setInputMask("")
        self.name.setText("")
        self.name.setPlaceholderText("")
        self.groupBox.setTitle(QCoreApplication.translate("ComputedChannel", u"Function", None))
        self.show_definition_btn.setText("")
        self.label_9.setText(QCoreApplication.translate("ComputedChannel", u"Name", None))
        self.label.setText(QCoreApplication.translate("ComputedChannel", u"Arguments:", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("ComputedChannel", u"Computation mode", None))
        self.sample_by_sample.setText(QCoreApplication.translate("ComputedChannel", u"sample by sample", None))
        self.label_24.setText(QCoreApplication.translate("ComputedChannel", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" font-family:'MS Shell Dlg 2'; font-size:8pt;\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">the function will be called several times for each of the individual time stamp in the time base</li>\n"
"<li style=\" font-family:'MS Shell Dlg 2'; font-size:8pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">earch argument will receive the signal sample at the"
                        " current time stamp</li>\n"
"<li style=\" font-family:'MS Shell Dlg 2'; font-size:8pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">the<span style=\" font-weight:700; font-style:italic;\"> t</span> argument will receive the current time stamp value</li></ul></body></html>", None))
        self.complete_signal.setText(QCoreApplication.translate("ComputedChannel", u"complete signal", None))
        self.label_25.setText(QCoreApplication.translate("ComputedChannel", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" font-family:'MS Shell Dlg 2'; font-size:8pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">the function will be called only once</li>\n"
"<li style=\" font-family:'MS Shell Dlg 2'; font-size:8pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">each argument will receive the complete signal ndarray</li></ul>\n"
"<ul style=\"margin-top: 0px; margin-bottom:"
                        " 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" font-family:'MS Shell Dlg 2'; font-size:8pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">the<span style=\" font-style:italic;\"> </span><span style=\" font-weight:700; font-style:italic;\">t</span><span style=\" font-style:italic;\"> </span>argument will receive the complete time base ndarray</li></ul></body></html>", None))
    # retranslateUi


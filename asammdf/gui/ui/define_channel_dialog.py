# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'define_channel_dialog.ui'
##
## Created by: Qt User Interface Compiler version 6.3.0
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QGridLayout,
    QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QSizePolicy, QSpacerItem, QTabWidget, QTextEdit,
    QWidget)
from . import resource_rc

class Ui_ComputedChannel(object):
    def setupUi(self, ComputedChannel):
        if not ComputedChannel.objectName():
            ComputedChannel.setObjectName(u"ComputedChannel")
        ComputedChannel.resize(937, 384)
        ComputedChannel.setMaximumSize(QSize(16777215, 16777215))
        icon = QIcon()
        icon.addFile(u":/plus.png", QSize(), QIcon.Normal, QIcon.Off)
        ComputedChannel.setWindowIcon(icon)
        ComputedChannel.setSizeGripEnabled(True)
        self.gridLayout = QGridLayout(ComputedChannel)
        self.gridLayout.setObjectName(u"gridLayout")
        self.apply_btn = QPushButton(ComputedChannel)
        self.apply_btn.setObjectName(u"apply_btn")

        self.gridLayout.addWidget(self.apply_btn, 1, 1, 1, 1)

        self.cancel_btn = QPushButton(ComputedChannel)
        self.cancel_btn.setObjectName(u"cancel_btn")

        self.gridLayout.addWidget(self.cancel_btn, 1, 2, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_3, 1, 0, 1, 1)

        self.tabs = QTabWidget(ComputedChannel)
        self.tabs.setObjectName(u"tabs")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.computation_grid_layout = QGridLayout(self.tab)
        self.computation_grid_layout.setObjectName(u"computation_grid_layout")
        self.op = QComboBox(self.tab)
        self.op.setObjectName(u"op")

        self.computation_grid_layout.addWidget(self.op, 1, 0, 1, 1)

        self.operand2 = QComboBox(self.tab)
        self.operand2.setObjectName(u"operand2")
        self.operand2.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.computation_grid_layout.addWidget(self.operand2, 2, 0, 1, 2)

        self.horizontalSpacer = QSpacerItem(29, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.computation_grid_layout.addItem(self.horizontalSpacer, 1, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.computation_grid_layout.addItem(self.verticalSpacer, 5, 0, 1, 1)

        self.operand1 = QComboBox(self.tab)
        self.operand1.setObjectName(u"operand1")
        self.operand1.setMinimumSize(QSize(380, 20))
        self.operand1.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.computation_grid_layout.addWidget(self.operand1, 0, 0, 1, 2)

        self.horizontalSpacer_2 = QSpacerItem(92, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.computation_grid_layout.addItem(self.horizontalSpacer_2, 4, 2, 1, 1)

        self.unit = QLineEdit(self.tab)
        self.unit.setObjectName(u"unit")

        self.computation_grid_layout.addWidget(self.unit, 4, 0, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.computation_grid_layout.addItem(self.horizontalSpacer_5, 4, 3, 1, 1)

        self.name = QLineEdit(self.tab)
        self.name.setObjectName(u"name")

        self.computation_grid_layout.addWidget(self.name, 3, 0, 1, 4)

        self.computation_grid_layout.setColumnStretch(2, 1)
        self.computation_grid_layout.setColumnStretch(3, 1)
        self.tabs.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.function_layout = QGridLayout(self.tab_2)
        self.function_layout.setObjectName(u"function_layout")
        self.channel = QComboBox(self.tab_2)
        self.channel.setObjectName(u"channel")

        self.function_layout.addWidget(self.channel, 1, 0, 1, 2)

        self.horizontalSpacer_10 = QSpacerItem(92, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.function_layout.addItem(self.horizontalSpacer_10, 1, 2, 1, 1)

        self.horizontalSpacer_11 = QSpacerItem(184, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.function_layout.addItem(self.horizontalSpacer_11, 3, 1, 1, 1)

        self.function_unit = QLineEdit(self.tab_2)
        self.function_unit.setObjectName(u"function_unit")

        self.function_layout.addWidget(self.function_unit, 3, 0, 1, 1)

        self.function = QComboBox(self.tab_2)
        self.function.setObjectName(u"function")
        self.function.setMinimumSize(QSize(380, 20))

        self.function_layout.addWidget(self.function, 0, 0, 1, 2)

        self.function_name = QLineEdit(self.tab_2)
        self.function_name.setObjectName(u"function_name")

        self.function_layout.addWidget(self.function_name, 2, 0, 1, 3)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.function_layout.addItem(self.horizontalSpacer_6, 1, 3, 1, 1)

        self.help = QTextEdit(self.tab_2)
        self.help.setObjectName(u"help")

        self.function_layout.addWidget(self.help, 4, 0, 1, 4)

        self.tabs.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.gridLayout_2 = QGridLayout(self.tab_3)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.expression_name = QLineEdit(self.tab_3)
        self.expression_name.setObjectName(u"expression_name")

        self.gridLayout_2.addWidget(self.expression_name, 2, 0, 1, 2)

        self.expression_unit = QLineEdit(self.tab_3)
        self.expression_unit.setObjectName(u"expression_unit")

        self.gridLayout_2.addWidget(self.expression_unit, 3, 0, 1, 1)

        self.label = QLabel(self.tab_3)
        self.label.setObjectName(u"label")
        self.label.setTextFormat(Qt.RichText)

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 2)

        self.horizontalSpacer_4 = QSpacerItem(234, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_4, 3, 1, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_3, 4, 0, 1, 1)

        self.expression = QPlainTextEdit(self.tab_3)
        self.expression.setObjectName(u"expression")

        self.gridLayout_2.addWidget(self.expression, 1, 0, 1, 2)

        self.tabs.addTab(self.tab_3, "")

        self.gridLayout.addWidget(self.tabs, 0, 0, 1, 3)


        self.retranslateUi(ComputedChannel)

        self.tabs.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(ComputedChannel)
    # setupUi

    def retranslateUi(self, ComputedChannel):
        ComputedChannel.setWindowTitle(QCoreApplication.translate("ComputedChannel", u"Define new channel", None))
        self.apply_btn.setText(QCoreApplication.translate("ComputedChannel", u"Apply", None))
        self.cancel_btn.setText(QCoreApplication.translate("ComputedChannel", u"Cancel", None))
        self.unit.setPlaceholderText(QCoreApplication.translate("ComputedChannel", u"channel unit", None))
        self.name.setInputMask("")
        self.name.setText("")
        self.name.setPlaceholderText(QCoreApplication.translate("ComputedChannel", u"channel name", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab), QCoreApplication.translate("ComputedChannel", u"Simple computation", None))
        self.function_unit.setPlaceholderText(QCoreApplication.translate("ComputedChannel", u"channel unit", None))
#if QT_CONFIG(tooltip)
        self.function.setToolTip(QCoreApplication.translate("ComputedChannel", u"see numpy documentation", None))
#endif // QT_CONFIG(tooltip)
        self.function_name.setPlaceholderText(QCoreApplication.translate("ComputedChannel", u"channel name", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_2), QCoreApplication.translate("ComputedChannel", u"Function", None))
        self.expression_name.setPlaceholderText(QCoreApplication.translate("ComputedChannel", u"channel name", None))
        self.expression_unit.setPlaceholderText(QCoreApplication.translate("ComputedChannel", u"channel unit", None))
        self.label.setText(QCoreApplication.translate("ComputedChannel", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'MS Shell Dlg 2'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The expression is evaluated using the <span style=\" font-style:italic;\">numexpr </span>library. Have a look at the <a href=\"https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/user_guide.html\"><span style=\" text-decoration: underline; color:#0000ff;\">Numexpr documentation</span></a> for supported operators and function.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:"
                        "0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">The signal names must be enclosed by double curly braces: <span style=\" font-weight:600; color:#ff0000;\">{{</span>Signal_name<span style=\" font-weight:600; color:#ff0000;\">}}</span></p></body></html>", None))
        self.expression.setPlaceholderText(QCoreApplication.translate("ComputedChannel", u"enter your expression here. Example: ({{Sig1}} + {{Sig2}}) / ({{Sig3}} - {{Sig4}}) + 7.8", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_3), QCoreApplication.translate("ComputedChannel", u"Expression", None))
    # retranslateUi


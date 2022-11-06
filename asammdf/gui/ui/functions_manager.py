# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'functions_manager.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QGroupBox, QHBoxLayout,
    QListWidgetItem, QPlainTextEdit, QPushButton, QSizePolicy,
    QSpacerItem, QWidget)

from asammdf.gui.widgets.list import MinimalListWidget
from . import resource_rc

class Ui_FunctionsManager(object):
    def setupUi(self, FunctionsManager):
        if not FunctionsManager.objectName():
            FunctionsManager.setObjectName(u"FunctionsManager")
        FunctionsManager.resize(749, 703)
        self.gridLayout_2 = QGridLayout(FunctionsManager)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(10)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.add_btn = QPushButton(FunctionsManager)
        self.add_btn.setObjectName(u"add_btn")
        icon = QIcon()
        icon.addFile(u":/plus.png", QSize(), QIcon.Normal, QIcon.Off)
        self.add_btn.setIcon(icon)
        self.add_btn.setIconSize(QSize(24, 24))

        self.horizontalLayout_2.addWidget(self.add_btn)

        self.erase_btn = QPushButton(FunctionsManager)
        self.erase_btn.setObjectName(u"erase_btn")
        icon1 = QIcon()
        icon1.addFile(u":/erase.png", QSize(), QIcon.Normal, QIcon.Off)
        self.erase_btn.setIcon(icon1)
        self.erase_btn.setIconSize(QSize(24, 24))

        self.horizontalLayout_2.addWidget(self.erase_btn)

        self.import_btn = QPushButton(FunctionsManager)
        self.import_btn.setObjectName(u"import_btn")
        icon2 = QIcon()
        icon2.addFile(u":/open.png", QSize(), QIcon.Normal, QIcon.Off)
        self.import_btn.setIcon(icon2)
        self.import_btn.setIconSize(QSize(24, 24))

        self.horizontalLayout_2.addWidget(self.import_btn)

        self.export_btn = QPushButton(FunctionsManager)
        self.export_btn.setObjectName(u"export_btn")
        icon3 = QIcon()
        icon3.addFile(u":/save.png", QSize(), QIcon.Normal, QIcon.Off)
        self.export_btn.setIcon(icon3)
        self.export_btn.setIconSize(QSize(24, 24))

        self.horizontalLayout_2.addWidget(self.export_btn)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_5)


        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)

        self.function_group = QGroupBox(FunctionsManager)
        self.function_group.setObjectName(u"function_group")
        self.gridLayout = QGridLayout(self.function_group)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.check_syntax_btn = QPushButton(self.function_group)
        self.check_syntax_btn.setObjectName(u"check_syntax_btn")
        icon4 = QIcon()
        icon4.addFile(u":/checkmark.png", QSize(), QIcon.Normal, QIcon.Off)
        self.check_syntax_btn.setIcon(icon4)

        self.horizontalLayout_3.addWidget(self.check_syntax_btn)

        self.store_btn = QPushButton(self.function_group)
        self.store_btn.setObjectName(u"store_btn")
        icon5 = QIcon()
        icon5.addFile(u":/shift_down.png", QSize(), QIcon.Normal, QIcon.Off)
        self.store_btn.setIcon(icon5)

        self.horizontalLayout_3.addWidget(self.store_btn)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)


        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 2)

        self.function_definition = QPlainTextEdit(self.function_group)
        self.function_definition.setObjectName(u"function_definition")
        font = QFont()
        font.setFamilies([u"Consolas"])
        self.function_definition.setFont(font)
        self.function_definition.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.function_definition.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.function_definition.setTabStopDistance(140.000000000000000)

        self.gridLayout.addWidget(self.function_definition, 0, 0, 1, 2)

        self.gridLayout.setRowStretch(0, 1)

        self.gridLayout_2.addWidget(self.function_group, 0, 1, 2, 1)

        self.functions_list = MinimalListWidget(FunctionsManager)
        self.functions_list.setObjectName(u"functions_list")
        self.functions_list.setMinimumSize(QSize(300, 0))

        self.gridLayout_2.addWidget(self.functions_list, 1, 0, 1, 1)

        self.gridLayout_2.setColumnStretch(0, 1)
        self.gridLayout_2.setColumnStretch(1, 2)

        self.retranslateUi(FunctionsManager)

        QMetaObject.connectSlotsByName(FunctionsManager)
    # setupUi

    def retranslateUi(self, FunctionsManager):
        FunctionsManager.setWindowTitle(QCoreApplication.translate("FunctionsManager", u"Form", None))
        self.add_btn.setText(QCoreApplication.translate("FunctionsManager", u"Add", None))
#if QT_CONFIG(tooltip)
        self.erase_btn.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.erase_btn.setText(QCoreApplication.translate("FunctionsManager", u"Delete all", None))
        self.import_btn.setText(QCoreApplication.translate("FunctionsManager", u"Load definitions", None))
        self.export_btn.setText(QCoreApplication.translate("FunctionsManager", u"Save definitions", None))
        self.function_group.setTitle(QCoreApplication.translate("FunctionsManager", u"Function definition", None))
        self.check_syntax_btn.setText(QCoreApplication.translate("FunctionsManager", u"Check syntax", None))
        self.store_btn.setText(QCoreApplication.translate("FunctionsManager", u"Store function changes", None))
        self.function_definition.setPlaceholderText(QCoreApplication.translate("FunctionsManager", u"function code", None))
    # retranslateUi


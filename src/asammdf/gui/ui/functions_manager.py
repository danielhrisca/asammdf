# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'functions_manager.ui'
##
## Created by: Qt User Interface Compiler version 6.7.3
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
    QSpacerItem, QTabWidget, QVBoxLayout, QWidget)

from asammdf.gui.widgets.list import MinimalListWidget
from . import resource_rc

class Ui_FunctionsManager(object):
    def setupUi(self, FunctionsManager):
        if not FunctionsManager.objectName():
            FunctionsManager.setObjectName(u"FunctionsManager")
        FunctionsManager.resize(1035, 703)
        self.verticalLayout = QVBoxLayout(FunctionsManager)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(1, 1, 1, 1)
        self.tabs = QTabWidget(FunctionsManager)
        self.tabs.setObjectName(u"tabs")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_2 = QGridLayout(self.tab)
        self.gridLayout_2.setSpacing(1)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(1, 1, 1, 1)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(1)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.add_btn = QPushButton(self.tab)
        self.add_btn.setObjectName(u"add_btn")
        icon = QIcon()
        icon.addFile(u":/plus.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.add_btn.setIcon(icon)
        self.add_btn.setIconSize(QSize(24, 24))

        self.horizontalLayout_2.addWidget(self.add_btn)

        self.erase_btn = QPushButton(self.tab)
        self.erase_btn.setObjectName(u"erase_btn")
        icon1 = QIcon()
        icon1.addFile(u":/erase.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.erase_btn.setIcon(icon1)
        self.erase_btn.setIconSize(QSize(24, 24))

        self.horizontalLayout_2.addWidget(self.erase_btn)

        self.import_btn = QPushButton(self.tab)
        self.import_btn.setObjectName(u"import_btn")
        icon2 = QIcon()
        icon2.addFile(u":/open.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.import_btn.setIcon(icon2)
        self.import_btn.setIconSize(QSize(24, 24))

        self.horizontalLayout_2.addWidget(self.import_btn)

        self.export_btn = QPushButton(self.tab)
        self.export_btn.setObjectName(u"export_btn")
        icon3 = QIcon()
        icon3.addFile(u":/save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.export_btn.setIcon(icon3)
        self.export_btn.setIconSize(QSize(24, 24))

        self.horizontalLayout_2.addWidget(self.export_btn)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_5)


        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)

        self.function_group = QGroupBox(self.tab)
        self.function_group.setObjectName(u"function_group")
        self.gridLayout = QGridLayout(self.function_group)
        self.gridLayout.setSpacing(1)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(1, 1, 1, 1)
        self.function_definition = QPlainTextEdit(self.function_group)
        self.function_definition.setObjectName(u"function_definition")
        font = QFont()
        font.setFamilies([u"Consolas"])
        self.function_definition.setFont(font)
        self.function_definition.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.function_definition.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.function_definition.setTabStopDistance(140.000000000000000)

        self.gridLayout.addWidget(self.function_definition, 0, 0, 1, 2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(1)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.check_syntax_btn = QPushButton(self.function_group)
        self.check_syntax_btn.setObjectName(u"check_syntax_btn")
        icon4 = QIcon()
        icon4.addFile(u":/checkmark.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.check_syntax_btn.setIcon(icon4)

        self.horizontalLayout_3.addWidget(self.check_syntax_btn)

        self.store_btn = QPushButton(self.function_group)
        self.store_btn.setObjectName(u"store_btn")
        icon5 = QIcon()
        icon5.addFile(u":/shift_down.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.store_btn.setIcon(icon5)

        self.horizontalLayout_3.addWidget(self.store_btn)

        self.load_original_function_btn = QPushButton(self.function_group)
        self.load_original_function_btn.setObjectName(u"load_original_function_btn")
        icon6 = QIcon()
        icon6.addFile(u":/shift_up.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.load_original_function_btn.setIcon(icon6)

        self.horizontalLayout_3.addWidget(self.load_original_function_btn)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)


        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 0, 1, 2)


        self.gridLayout_2.addWidget(self.function_group, 0, 1, 2, 1)

        self.functions_list = MinimalListWidget(self.tab)
        self.functions_list.setObjectName(u"functions_list")
        self.functions_list.setMinimumSize(QSize(300, 0))

        self.gridLayout_2.addWidget(self.functions_list, 1, 0, 1, 1)

        self.gridLayout_2.setColumnStretch(1, 1)
        self.tabs.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.gridLayout_3 = QGridLayout(self.tab_2)
        self.gridLayout_3.setSpacing(1)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(1, 1, 1, 1)
        self.horizontalSpacer = QSpacerItem(637, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer, 1, 2, 1, 1)

        self.globals_definition = QPlainTextEdit(self.tab_2)
        self.globals_definition.setObjectName(u"globals_definition")
        self.globals_definition.setTabStopDistance(140.000000000000000)

        self.gridLayout_3.addWidget(self.globals_definition, 0, 0, 1, 3)

        self.check_globals_syntax_btn = QPushButton(self.tab_2)
        self.check_globals_syntax_btn.setObjectName(u"check_globals_syntax_btn")
        self.check_globals_syntax_btn.setIcon(icon4)

        self.gridLayout_3.addWidget(self.check_globals_syntax_btn, 1, 0, 1, 1)

        self.load_original_globals_btn = QPushButton(self.tab_2)
        self.load_original_globals_btn.setObjectName(u"load_original_globals_btn")
        self.load_original_globals_btn.setIcon(icon6)

        self.gridLayout_3.addWidget(self.load_original_globals_btn, 1, 1, 1, 1)

        self.tabs.addTab(self.tab_2, "")

        self.verticalLayout.addWidget(self.tabs)


        self.retranslateUi(FunctionsManager)

        self.tabs.setCurrentIndex(0)


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
        self.function_definition.setPlaceholderText(QCoreApplication.translate("FunctionsManager", u"function code", None))
        self.check_syntax_btn.setText(QCoreApplication.translate("FunctionsManager", u"Check syntax", None))
        self.store_btn.setText(QCoreApplication.translate("FunctionsManager", u"Store function changes", None))
        self.load_original_function_btn.setText(QCoreApplication.translate("FunctionsManager", u"Load original function", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab), QCoreApplication.translate("FunctionsManager", u"Functions", None))
        self.check_globals_syntax_btn.setText(QCoreApplication.translate("FunctionsManager", u"Check syntax", None))
        self.load_original_globals_btn.setText(QCoreApplication.translate("FunctionsManager", u"Load original globals", None))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_2), QCoreApplication.translate("FunctionsManager", u"Global variables", None))
    # retranslateUi


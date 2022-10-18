# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'multi_search_dialog.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QSizePolicy, QSpacerItem,
    QWidget)

from asammdf.gui.widgets.list import MinimalListWidget
from . import resource_rc

class Ui_MultiSearchDialog(object):
    def setupUi(self, MultiSearchDialog):
        if not MultiSearchDialog.objectName():
            MultiSearchDialog.setObjectName(u"MultiSearchDialog")
        MultiSearchDialog.resize(1028, 549)
        MultiSearchDialog.setSizeGripEnabled(True)
        self.grid_layout = QGridLayout(MultiSearchDialog)
        self.grid_layout.setObjectName(u"grid_layout")
        self.grid_layout.setContentsMargins(9, 9, 9, 9)
        self.status = QLabel(MultiSearchDialog)
        self.status.setObjectName(u"status")

        self.grid_layout.addWidget(self.status, 9, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.cancel_btn = QPushButton(MultiSearchDialog)
        self.cancel_btn.setObjectName(u"cancel_btn")

        self.horizontalLayout.addWidget(self.cancel_btn)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.apply_btn = QPushButton(MultiSearchDialog)
        self.apply_btn.setObjectName(u"apply_btn")

        self.horizontalLayout.addWidget(self.apply_btn)

        self.horizontalLayout.setStretch(1, 1)

        self.grid_layout.addLayout(self.horizontalLayout, 8, 0, 1, 6)

        self.match_kind = QComboBox(MultiSearchDialog)
        self.match_kind.addItem("")
        self.match_kind.addItem("")
        self.match_kind.setObjectName(u"match_kind")

        self.grid_layout.addWidget(self.match_kind, 0, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.grid_layout.addItem(self.verticalSpacer_2, 6, 1, 1, 1)

        self.selection = MinimalListWidget(MultiSearchDialog)
        self.selection.setObjectName(u"selection")
        self.selection.setMinimumSize(QSize(500, 0))

        self.grid_layout.addWidget(self.selection, 5, 2, 3, 4)

        self.add_btn = QPushButton(MultiSearchDialog)
        self.add_btn.setObjectName(u"add_btn")
        self.add_btn.setFocusPolicy(Qt.TabFocus)
        icon = QIcon()
        icon.addFile(u":/left.png", QSize(), QIcon.Normal, QIcon.Off)
        self.add_btn.setIcon(icon)

        self.grid_layout.addWidget(self.add_btn, 5, 1, 1, 1)

        self.label = QLabel(MultiSearchDialog)
        self.label.setObjectName(u"label")

        self.grid_layout.addWidget(self.label, 1, 2, 1, 1)

        self.search_box = QLineEdit(MultiSearchDialog)
        self.search_box.setObjectName(u"search_box")

        self.grid_layout.addWidget(self.search_box, 1, 0, 1, 1)

        self.show_measurement_list_btn = QPushButton(MultiSearchDialog)
        self.show_measurement_list_btn.setObjectName(u"show_measurement_list_btn")
        icon1 = QIcon()
        icon1.addFile(u":/info.png", QSize(), QIcon.Normal, QIcon.Off)
        self.show_measurement_list_btn.setIcon(icon1)

        self.grid_layout.addWidget(self.show_measurement_list_btn, 0, 2, 1, 2)

        self.matches = QListWidget(MultiSearchDialog)
        self.matches.setObjectName(u"matches")

        self.grid_layout.addWidget(self.matches, 5, 0, 2, 1)

        self.grid_layout.setColumnStretch(0, 1)
        QWidget.setTabOrder(self.search_box, self.match_kind)
        QWidget.setTabOrder(self.match_kind, self.matches)
        QWidget.setTabOrder(self.matches, self.add_btn)
        QWidget.setTabOrder(self.add_btn, self.selection)
        QWidget.setTabOrder(self.selection, self.show_measurement_list_btn)
        QWidget.setTabOrder(self.show_measurement_list_btn, self.apply_btn)
        QWidget.setTabOrder(self.apply_btn, self.cancel_btn)

        self.retranslateUi(MultiSearchDialog)

        self.match_kind.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MultiSearchDialog)
    # setupUi

    def retranslateUi(self, MultiSearchDialog):
        MultiSearchDialog.setWindowTitle(QCoreApplication.translate("MultiSearchDialog", u"Dialog", None))
        self.status.setText("")
        self.cancel_btn.setText(QCoreApplication.translate("MultiSearchDialog", u"Cancel", None))
        self.apply_btn.setText(QCoreApplication.translate("MultiSearchDialog", u"Apply", None))
        self.match_kind.setItemText(0, QCoreApplication.translate("MultiSearchDialog", u"Wildcard", None))
        self.match_kind.setItemText(1, QCoreApplication.translate("MultiSearchDialog", u"Regex", None))

        self.add_btn.setText("")
        self.label.setText(QCoreApplication.translate("MultiSearchDialog", u"Final selection", None))
        self.search_box.setText("")
        self.show_measurement_list_btn.setText(QCoreApplication.translate("MultiSearchDialog", u"Show measurement list", None))
    # retranslateUi


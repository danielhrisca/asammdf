# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'simple_search_dialog.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QTreeWidget, QTreeWidgetItem, QWidget)
import resource_rc

class Ui_SimpleSearchDialog(object):
    def setupUi(self, SimpleSearchDialog):
        if not SimpleSearchDialog.objectName():
            SimpleSearchDialog.setObjectName(u"SimpleSearchDialog")
        SimpleSearchDialog.resize(1183, 421)
        SimpleSearchDialog.setSizeGripEnabled(True)
        self.gridLayout = QGridLayout(SimpleSearchDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.matches = QTreeWidget(SimpleSearchDialog)
        self.matches.setObjectName(u"matches")
        self.matches.setFocusPolicy(Qt.TabFocus)
        self.matches.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.matches.setSortingEnabled(True)

        self.gridLayout.addWidget(self.matches, 3, 0, 2, 1)

        self.status = QLabel(SimpleSearchDialog)
        self.status.setObjectName(u"status")

        self.gridLayout.addWidget(self.status, 9, 0, 1, 1)

        self.selection = QTreeWidget(SimpleSearchDialog)
        self.selection.setObjectName(u"selection")
        self.selection.setSortingEnabled(True)

        self.gridLayout.addWidget(self.selection, 3, 2, 2, 1)

        self.label = QLabel(SimpleSearchDialog)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 2, 2, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)


        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 2, 1, 1)

        self.add_btn = QPushButton(SimpleSearchDialog)
        self.add_btn.setObjectName(u"add_btn")
        icon = QIcon()
        icon.addFile(u":/left.png", QSize(), QIcon.Normal, QIcon.Off)
        self.add_btn.setIcon(icon)

        self.gridLayout.addWidget(self.add_btn, 3, 1, 1, 1)

        self.search_box = QLineEdit(SimpleSearchDialog)
        self.search_box.setObjectName(u"search_box")

        self.gridLayout.addWidget(self.search_box, 2, 0, 1, 1)

        self.match_kind = QComboBox(SimpleSearchDialog)
        self.match_kind.addItem("")
        self.match_kind.addItem("")
        self.match_kind.setObjectName(u"match_kind")

        self.gridLayout.addWidget(self.match_kind, 0, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 4, 1, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.cancel_btn = QPushButton(SimpleSearchDialog)
        self.cancel_btn.setObjectName(u"cancel_btn")

        self.horizontalLayout.addWidget(self.cancel_btn)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.apply_btn = QPushButton(SimpleSearchDialog)
        self.apply_btn.setObjectName(u"apply_btn")

        self.horizontalLayout.addWidget(self.apply_btn)

        self.horizontalLayout.setStretch(1, 1)

        self.gridLayout.addLayout(self.horizontalLayout, 6, 0, 1, 3)

        self.comment = QLabel(SimpleSearchDialog)
        self.comment.setObjectName(u"comment")

        self.gridLayout.addWidget(self.comment, 5, 0, 1, 3)

        self.case_sensitivity = QComboBox(SimpleSearchDialog)
        self.case_sensitivity.addItem("")
        self.case_sensitivity.addItem("")
        self.case_sensitivity.setObjectName(u"case_sensitivity")

        self.gridLayout.addWidget(self.case_sensitivity, 1, 0, 1, 1)

        QWidget.setTabOrder(self.match_kind, self.case_sensitivity)
        QWidget.setTabOrder(self.case_sensitivity, self.search_box)
        QWidget.setTabOrder(self.search_box, self.matches)
        QWidget.setTabOrder(self.matches, self.add_btn)
        QWidget.setTabOrder(self.add_btn, self.selection)
        QWidget.setTabOrder(self.selection, self.apply_btn)
        QWidget.setTabOrder(self.apply_btn, self.cancel_btn)

        self.retranslateUi(SimpleSearchDialog)

        self.match_kind.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(SimpleSearchDialog)
    # setupUi

    def retranslateUi(self, SimpleSearchDialog):
        SimpleSearchDialog.setWindowTitle(QCoreApplication.translate("SimpleSearchDialog", u"Dialog", None))
        ___qtreewidgetitem = self.matches.headerItem()
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("SimpleSearchDialog", u"Channel", None));
        self.status.setText("")
        ___qtreewidgetitem1 = self.selection.headerItem()
        ___qtreewidgetitem1.setText(0, QCoreApplication.translate("SimpleSearchDialog", u"Channel", None));
        self.label.setText(QCoreApplication.translate("SimpleSearchDialog", u"Final selection", None))
        self.add_btn.setText("")
        self.search_box.setText("")
        self.match_kind.setItemText(0, QCoreApplication.translate("SimpleSearchDialog", u"Wildcard", None))
        self.match_kind.setItemText(1, QCoreApplication.translate("SimpleSearchDialog", u"Regex", None))

        self.cancel_btn.setText(QCoreApplication.translate("SimpleSearchDialog", u"Cancel", None))
        self.apply_btn.setText(QCoreApplication.translate("SimpleSearchDialog", u"Apply", None))
        self.comment.setText("")
        self.case_sensitivity.setItemText(0, QCoreApplication.translate("SimpleSearchDialog", u"Case insensitive", None))
        self.case_sensitivity.setItemText(1, QCoreApplication.translate("SimpleSearchDialog", u"Case sensitive", None))

    # retranslateUi


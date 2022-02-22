# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'values_dialog.ui'
##
## Created by: Qt User Interface Compiler version 6.2.3
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
from PySide6.QtWidgets import (QApplication, QDialog, QGridLayout, QHeaderView,
    QPushButton, QSizePolicy, QSpacerItem, QTableWidget,
    QTableWidgetItem, QWidget)

class Ui_RangeDialog(object):
    def setupUi(self, RangeDialog):
        if not RangeDialog.objectName():
            RangeDialog.setObjectName(u"RangeDialog")
        RangeDialog.resize(697, 379)
        RangeDialog.setSizeGripEnabled(True)
        RangeDialog.setModal(True)
        self.gridLayout = QGridLayout(RangeDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.reset_btn = QPushButton(RangeDialog)
        self.reset_btn.setObjectName(u"reset_btn")

        self.gridLayout.addWidget(self.reset_btn, 0, 1, 1, 1)

        self.cancel_btn = QPushButton(RangeDialog)
        self.cancel_btn.setObjectName(u"cancel_btn")

        self.gridLayout.addWidget(self.cancel_btn, 3, 1, 1, 1)

        self.apply_btn = QPushButton(RangeDialog)
        self.apply_btn.setObjectName(u"apply_btn")

        self.gridLayout.addWidget(self.apply_btn, 2, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 271, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 1, 1, 1, 1)

        self.table = QTableWidget(RangeDialog)
        if (self.table.columnCount() < 4):
            self.table.setColumnCount(4)
        __qtablewidgetitem = QTableWidgetItem()
        self.table.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.table.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.table.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.table.setHorizontalHeaderItem(3, __qtablewidgetitem3)
        if (self.table.rowCount() < 100):
            self.table.setRowCount(100)
        self.table.setObjectName(u"table")
        self.table.setRowCount(100)
        self.table.setColumnCount(4)
        self.table.horizontalHeader().setMinimumSectionSize(30)
        self.table.horizontalHeader().setDefaultSectionSize(150)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.gridLayout.addWidget(self.table, 0, 0, 4, 1)

        self.gridLayout.setColumnStretch(0, 1)
        QWidget.setTabOrder(self.table, self.apply_btn)
        QWidget.setTabOrder(self.apply_btn, self.reset_btn)
        QWidget.setTabOrder(self.reset_btn, self.cancel_btn)

        self.retranslateUi(RangeDialog)

        QMetaObject.connectSlotsByName(RangeDialog)
    # setupUi

    def retranslateUi(self, RangeDialog):
        RangeDialog.setWindowTitle(QCoreApplication.translate("RangeDialog", u"Edit value range colors", None))
        self.reset_btn.setText(QCoreApplication.translate("RangeDialog", u"Reset", None))
        self.cancel_btn.setText(QCoreApplication.translate("RangeDialog", u"Cancel", None))
        self.apply_btn.setText(QCoreApplication.translate("RangeDialog", u"Apply", None))
        ___qtablewidgetitem = self.table.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("RangeDialog", u"From", None));
        ___qtablewidgetitem1 = self.table.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("RangeDialog", u"To", None));
        ___qtablewidgetitem2 = self.table.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("RangeDialog", u"Set color", None));
    # retranslateUi


# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'range_editor_dialog.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QGridLayout,
    QListWidgetItem, QPushButton, QSizePolicy, QSpacerItem,
    QWidget)

from asammdf.gui.widgets.list import MinimalListWidget

class Ui_RangeDialog(object):
    def setupUi(self, RangeDialog):
        if not RangeDialog.objectName():
            RangeDialog.setObjectName(u"RangeDialog")
        RangeDialog.resize(1300, 329)
        RangeDialog.setSizeGripEnabled(True)
        RangeDialog.setModal(True)
        self.gridLayout = QGridLayout(RangeDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.ranges = MinimalListWidget(RangeDialog)
        self.ranges.setObjectName(u"ranges")
        self.ranges.setDragDropMode(QAbstractItemView.InternalMove)
        self.ranges.setDefaultDropAction(Qt.MoveAction)

        self.gridLayout.addWidget(self.ranges, 0, 0, 7, 1)

        self.verticalSpacer = QSpacerItem(20, 271, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 4, 1, 1, 1)

        self.reset_btn = QPushButton(RangeDialog)
        self.reset_btn.setObjectName(u"reset_btn")

        self.gridLayout.addWidget(self.reset_btn, 3, 1, 1, 1)

        self.cancel_btn = QPushButton(RangeDialog)
        self.cancel_btn.setObjectName(u"cancel_btn")

        self.gridLayout.addWidget(self.cancel_btn, 6, 1, 1, 1)

        self.apply_btn = QPushButton(RangeDialog)
        self.apply_btn.setObjectName(u"apply_btn")

        self.gridLayout.addWidget(self.apply_btn, 5, 1, 1, 1)

        self.insert_btn = QPushButton(RangeDialog)
        self.insert_btn.setObjectName(u"insert_btn")

        self.gridLayout.addWidget(self.insert_btn, 0, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 1, 1, 1, 1)

        self.gridLayout.setRowStretch(4, 1)
        self.gridLayout.setColumnStretch(0, 1)
        QWidget.setTabOrder(self.ranges, self.insert_btn)
        QWidget.setTabOrder(self.insert_btn, self.reset_btn)
        QWidget.setTabOrder(self.reset_btn, self.apply_btn)
        QWidget.setTabOrder(self.apply_btn, self.cancel_btn)

        self.retranslateUi(RangeDialog)

        QMetaObject.connectSlotsByName(RangeDialog)
    # setupUi

    def retranslateUi(self, RangeDialog):
        RangeDialog.setWindowTitle(QCoreApplication.translate("RangeDialog", u"Edit value range colors", None))
        self.reset_btn.setText(QCoreApplication.translate("RangeDialog", u"Reset", None))
        self.cancel_btn.setText(QCoreApplication.translate("RangeDialog", u"Cancel", None))
        self.apply_btn.setText(QCoreApplication.translate("RangeDialog", u"Apply", None))
        self.insert_btn.setText(QCoreApplication.translate("RangeDialog", u"Insert range", None))
    # retranslateUi


# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'database_item.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QSizePolicy, QWidget)

class Ui_DatabaseItemUI(object):
    def setupUi(self, DatabaseItemUI):
        if not DatabaseItemUI.objectName():
            DatabaseItemUI.setObjectName(u"DatabaseItemUI")
        DatabaseItemUI.resize(291, 24)
        self.horizontalLayout = QHBoxLayout(DatabaseItemUI)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(2, 2, 2, 2)
        self.bus = QComboBox(DatabaseItemUI)
        self.bus.setObjectName(u"bus")

        self.horizontalLayout.addWidget(self.bus)

        self.database = QLabel(DatabaseItemUI)
        self.database.setObjectName(u"database")

        self.horizontalLayout.addWidget(self.database)

        self.horizontalLayout.setStretch(1, 1)

        self.retranslateUi(DatabaseItemUI)

        QMetaObject.connectSlotsByName(DatabaseItemUI)
    # setupUi

    def retranslateUi(self, DatabaseItemUI):
        DatabaseItemUI.setWindowTitle(QCoreApplication.translate("DatabaseItemUI", u"Form", None))
        self.database.setText(QCoreApplication.translate("DatabaseItemUI", u"TextLabel", None))
    # retranslateUi


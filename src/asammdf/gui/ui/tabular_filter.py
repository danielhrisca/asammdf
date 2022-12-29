# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'tabular_filter.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout,
    QLineEdit, QSizePolicy, QWidget)

class Ui_TabularFilter(object):
    def setupUi(self, TabularFilter):
        if not TabularFilter.objectName():
            TabularFilter.setObjectName(u"TabularFilter")
        TabularFilter.resize(813, 26)
        TabularFilter.setMinimumSize(QSize(0, 26))
        self.horizontalLayout = QHBoxLayout(TabularFilter)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(2, 2, 2, 2)
        self.enabled = QCheckBox(TabularFilter)
        self.enabled.setObjectName(u"enabled")
        self.enabled.setChecked(True)

        self.horizontalLayout.addWidget(self.enabled)

        self.relation = QComboBox(TabularFilter)
        self.relation.setObjectName(u"relation")

        self.horizontalLayout.addWidget(self.relation)

        self.column = QComboBox(TabularFilter)
        self.column.setObjectName(u"column")
        self.column.setMinimumSize(QSize(300, 0))

        self.horizontalLayout.addWidget(self.column)

        self.op = QComboBox(TabularFilter)
        self.op.setObjectName(u"op")

        self.horizontalLayout.addWidget(self.op)

        self.target = QLineEdit(TabularFilter)
        self.target.setObjectName(u"target")
        self.target.setClearButtonEnabled(False)

        self.horizontalLayout.addWidget(self.target)


        self.retranslateUi(TabularFilter)

        QMetaObject.connectSlotsByName(TabularFilter)
    # setupUi

    def retranslateUi(self, TabularFilter):
        TabularFilter.setWindowTitle(QCoreApplication.translate("TabularFilter", u"Form", None))
        self.enabled.setText("")
    # retranslateUi


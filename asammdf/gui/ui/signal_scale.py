# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'signal_scale.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QDoubleSpinBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSpacerItem, QWidget)
from . import resource_rc

class Ui_ScaleDialog(object):
    def setupUi(self, ScaleDialog):
        if not ScaleDialog.objectName():
            ScaleDialog.setObjectName(u"ScaleDialog")
        ScaleDialog.resize(706, 414)
        self.gridLayout = QGridLayout(ScaleDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_12 = QLabel(ScaleDialog)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_12, 0, 0, 1, 1)

        self.plot = QLabel(ScaleDialog)
        self.plot.setObjectName(u"plot")
        self.plot.setMinimumSize(QSize(350, 350))
        self.plot.setMaximumSize(QSize(350, 350))
        self.plot.setFrameShape(QFrame.Box)
        self.plot.setLineWidth(2)

        self.gridLayout.addWidget(self.plot, 0, 1, 3, 1)

        self.groupBox = QGroupBox(ScaleDialog)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_3 = QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.fit_btn = QPushButton(self.groupBox)
        self.fit_btn.setObjectName(u"fit_btn")
        icon = QIcon()
        icon.addFile(u":/fit.png", QSize(), QIcon.Normal, QIcon.Off)
        self.fit_btn.setIcon(icon)

        self.gridLayout_3.addWidget(self.fit_btn, 3, 0, 1, 1)

        self.y_bottom = QDoubleSpinBox(self.groupBox)
        self.y_bottom.setObjectName(u"y_bottom")
        self.y_bottom.setMinimum(0.000000000000000)
        self.y_bottom.setMaximum(1.000000000000000)

        self.gridLayout_3.addWidget(self.y_bottom, 5, 1, 1, 1)

        self.zoom_in_btn = QPushButton(self.groupBox)
        self.zoom_in_btn.setObjectName(u"zoom_in_btn")
        icon1 = QIcon()
        icon1.addFile(u":/zoom-in.png", QSize(), QIcon.Normal, QIcon.Off)
        self.zoom_in_btn.setIcon(icon1)

        self.gridLayout_3.addWidget(self.zoom_in_btn, 1, 0, 1, 1)

        self.horizontalSpacer_5 = QSpacerItem(168, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_5, 1, 1, 1, 1)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_3.addWidget(self.label_4, 4, 0, 1, 1)

        self.label_8 = QLabel(self.groupBox)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_3.addWidget(self.label_8, 5, 0, 1, 1)

        self.zoom_out_btn = QPushButton(self.groupBox)
        self.zoom_out_btn.setObjectName(u"zoom_out_btn")
        icon2 = QIcon()
        icon2.addFile(u":/zoom-out.png", QSize(), QIcon.Normal, QIcon.Off)
        self.zoom_out_btn.setIcon(icon2)

        self.gridLayout_3.addWidget(self.zoom_out_btn, 2, 0, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer_3, 7, 0, 1, 1)

        self.groupBox_3 = QGroupBox(self.groupBox)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_4 = QGridLayout(self.groupBox_3)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_5 = QLabel(self.groupBox_3)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_4.addWidget(self.label_5, 0, 0, 1, 1)

        self.target_max = QDoubleSpinBox(self.groupBox_3)
        self.target_max.setObjectName(u"target_max")
        self.target_max.setDecimals(3)
        self.target_max.setMinimum(0.000000000000000)
        self.target_max.setMaximum(1.000000000000000)

        self.gridLayout_4.addWidget(self.target_max, 0, 1, 1, 1)

        self.label_9 = QLabel(self.groupBox_3)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_4.addWidget(self.label_9, 1, 0, 1, 1)

        self.target_min = QDoubleSpinBox(self.groupBox_3)
        self.target_min.setObjectName(u"target_min")
        self.target_min.setDecimals(3)
        self.target_min.setMinimum(0.000000000000000)
        self.target_min.setMaximum(1.000000000000000)

        self.gridLayout_4.addWidget(self.target_min, 1, 1, 1, 1)


        self.gridLayout_3.addWidget(self.groupBox_3, 0, 0, 1, 2)

        self.y_top = QDoubleSpinBox(self.groupBox)
        self.y_top.setObjectName(u"y_top")
        self.y_top.setMinimum(0.000000000000000)
        self.y_top.setMaximum(1.000000000000000)

        self.gridLayout_3.addWidget(self.y_top, 4, 1, 1, 1)

        self.label_10 = QLabel(self.groupBox)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_3.addWidget(self.label_10, 6, 0, 1, 1)

        self.offset = QDoubleSpinBox(self.groupBox)
        self.offset.setObjectName(u"offset")
        self.offset.setDecimals(3)
        self.offset.setMinimum(0.000000000000000)
        self.offset.setMaximum(1.000000000000000)

        self.gridLayout_3.addWidget(self.offset, 6, 1, 1, 1)


        self.gridLayout.addWidget(self.groupBox, 0, 2, 3, 1)

        self.verticalSpacer = QSpacerItem(20, 299, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 1, 0, 1, 1)

        self.label_11 = QLabel(ScaleDialog)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_11, 2, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_6)

        self.cancel_btn = QPushButton(ScaleDialog)
        self.cancel_btn.setObjectName(u"cancel_btn")
        icon3 = QIcon()
        icon3.addFile(u":/erase.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cancel_btn.setIcon(icon3)

        self.horizontalLayout.addWidget(self.cancel_btn)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_7)

        self.apply_btn = QPushButton(ScaleDialog)
        self.apply_btn.setObjectName(u"apply_btn")
        icon4 = QIcon()
        icon4.addFile(u":/checkmark.png", QSize(), QIcon.Normal, QIcon.Off)
        self.apply_btn.setIcon(icon4)

        self.horizontalLayout.addWidget(self.apply_btn)

        self.horizontalLayout.setStretch(0, 1)

        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 3)


        self.retranslateUi(ScaleDialog)

        QMetaObject.connectSlotsByName(ScaleDialog)
    # setupUi

    def retranslateUi(self, ScaleDialog):
        ScaleDialog.setWindowTitle(QCoreApplication.translate("ScaleDialog", u"Dialog", None))
        self.label_12.setText(QCoreApplication.translate("ScaleDialog", u"100%", None))
        self.plot.setText("")
        self.groupBox.setTitle(QCoreApplication.translate("ScaleDialog", u"Y scaling", None))
        self.fit_btn.setText(QCoreApplication.translate("ScaleDialog", u"Fit vertically", None))
        self.zoom_in_btn.setText(QCoreApplication.translate("ScaleDialog", u"Zoom in", None))
        self.label_4.setText(QCoreApplication.translate("ScaleDialog", u"Y top", None))
        self.label_8.setText(QCoreApplication.translate("ScaleDialog", u"Y bottom", None))
        self.zoom_out_btn.setText(QCoreApplication.translate("ScaleDialog", u"Zoom out", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("ScaleDialog", u"Expected signal values", None))
        self.label_5.setText(QCoreApplication.translate("ScaleDialog", u"Signal max", None))
        self.label_9.setText(QCoreApplication.translate("ScaleDialog", u"Signal min", None))
        self.label_10.setText(QCoreApplication.translate("ScaleDialog", u"0 value position", None))
        self.offset.setSuffix(QCoreApplication.translate("ScaleDialog", u"%", None))
        self.label_11.setText(QCoreApplication.translate("ScaleDialog", u"0%", None))
        self.cancel_btn.setText(QCoreApplication.translate("ScaleDialog", u"Cancel", None))
        self.apply_btn.setText(QCoreApplication.translate("ScaleDialog", u"Apply", None))
    # retranslateUi


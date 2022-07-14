# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'signal_scale.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDoubleSpinBox,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QSpacerItem,
    QWidget)
from . import resource_rc

class Ui_ScaleDialog(object):
    def setupUi(self, ScaleDialog):
        if not ScaleDialog.objectName():
            ScaleDialog.setObjectName(u"ScaleDialog")
        ScaleDialog.resize(1091, 676)
        self.gridLayout_3 = QGridLayout(ScaleDialog)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_6)

        self.cancel_btn = QPushButton(ScaleDialog)
        self.cancel_btn.setObjectName(u"cancel_btn")
        icon = QIcon()
        icon.addFile(u":/erase.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cancel_btn.setIcon(icon)

        self.horizontalLayout.addWidget(self.cancel_btn)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_7)

        self.apply_btn = QPushButton(ScaleDialog)
        self.apply_btn.setObjectName(u"apply_btn")
        icon1 = QIcon()
        icon1.addFile(u":/checkmark.png", QSize(), QIcon.Normal, QIcon.Off)
        self.apply_btn.setIcon(icon1)

        self.horizontalLayout.addWidget(self.apply_btn)

        self.horizontalLayout.setStretch(0, 1)

        self.gridLayout_3.addLayout(self.horizontalLayout, 5, 0, 1, 2)

        self.verticalSpacer = QSpacerItem(20, 135, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer, 4, 1, 1, 1)

        self.groupBox_2 = QGroupBox(ScaleDialog)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout = QGridLayout(self.groupBox_2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.fit_btn = QPushButton(self.groupBox_2)
        self.fit_btn.setObjectName(u"fit_btn")
        icon2 = QIcon()
        icon2.addFile(u":/fit.png", QSize(), QIcon.Normal, QIcon.Off)
        self.fit_btn.setIcon(icon2)

        self.gridLayout.addWidget(self.fit_btn, 2, 0, 1, 1)

        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_3, 2, 1, 1, 1)

        self.zoom_out_btn = QPushButton(self.groupBox_2)
        self.zoom_out_btn.setObjectName(u"zoom_out_btn")
        icon3 = QIcon()
        icon3.addFile(u":/zoom-out.png", QSize(), QIcon.Normal, QIcon.Off)
        self.zoom_out_btn.setIcon(icon3)

        self.gridLayout.addWidget(self.zoom_out_btn, 1, 0, 1, 1)

        self.label = QLabel(self.groupBox_2)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label, 1, 1, 1, 1)

        self.fast_shift_up_btn = QPushButton(self.groupBox_2)
        self.fast_shift_up_btn.setObjectName(u"fast_shift_up_btn")
        icon4 = QIcon()
        icon4.addFile(u":/shift_up.png", QSize(), QIcon.Normal, QIcon.Off)
        self.fast_shift_up_btn.setIcon(icon4)

        self.gridLayout.addWidget(self.fast_shift_up_btn, 5, 0, 1, 1)

        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)

        self.shift_down_btn = QPushButton(self.groupBox_2)
        self.shift_down_btn.setObjectName(u"shift_down_btn")
        icon5 = QIcon()
        icon5.addFile(u":/down.png", QSize(), QIcon.Normal, QIcon.Off)
        self.shift_down_btn.setIcon(icon5)

        self.gridLayout.addWidget(self.shift_down_btn, 4, 0, 1, 1)

        self.shift_up_btn = QPushButton(self.groupBox_2)
        self.shift_up_btn.setObjectName(u"shift_up_btn")
        icon6 = QIcon()
        icon6.addFile(u":/up.png", QSize(), QIcon.Normal, QIcon.Off)
        self.shift_up_btn.setIcon(icon6)

        self.gridLayout.addWidget(self.shift_up_btn, 3, 0, 1, 1)

        self.zoom_in_btn = QPushButton(self.groupBox_2)
        self.zoom_in_btn.setObjectName(u"zoom_in_btn")
        icon7 = QIcon()
        icon7.addFile(u":/zoom-in.png", QSize(), QIcon.Normal, QIcon.Off)
        self.zoom_in_btn.setIcon(icon7)

        self.gridLayout.addWidget(self.zoom_in_btn, 0, 0, 1, 1)

        self.fast_shift_down_btn = QPushButton(self.groupBox_2)
        self.fast_shift_down_btn.setObjectName(u"fast_shift_down_btn")
        icon8 = QIcon()
        icon8.addFile(u":/shift_down.png", QSize(), QIcon.Normal, QIcon.Off)
        self.fast_shift_down_btn.setIcon(icon8)

        self.gridLayout.addWidget(self.fast_shift_down_btn, 6, 0, 1, 1)


        self.gridLayout_3.addWidget(self.groupBox_2, 2, 1, 1, 1)

        self.plot = QLabel(ScaleDialog)
        self.plot.setObjectName(u"plot")
        self.plot.setMinimumSize(QSize(750, 600))
        self.plot.setMaximumSize(QSize(750, 600))
        self.plot.setFrameShape(QFrame.Box)
        self.plot.setLineWidth(2)

        self.gridLayout_3.addWidget(self.plot, 1, 0, 4, 1)

        self.groupBox_3 = QGroupBox(ScaleDialog)
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


        self.gridLayout_3.addWidget(self.groupBox_3, 1, 1, 1, 1)

        self.groupBox_4 = QGroupBox(ScaleDialog)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.gridLayout_2 = QGridLayout(self.groupBox_4)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_10 = QLabel(self.groupBox_4)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_2.addWidget(self.label_10, 3, 0, 1, 1)

        self.label_8 = QLabel(self.groupBox_4)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_2.addWidget(self.label_8, 1, 0, 1, 1)

        self.offset = QDoubleSpinBox(self.groupBox_4)
        self.offset.setObjectName(u"offset")
        self.offset.setDecimals(1)
        self.offset.setMinimum(0.000000000000000)
        self.offset.setMaximum(1.000000000000000)

        self.gridLayout_2.addWidget(self.offset, 3, 1, 1, 1)

        self.label_4 = QLabel(self.groupBox_4)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_2.addWidget(self.label_4, 0, 0, 1, 1)

        self.label_13 = QLabel(self.groupBox_4)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout_2.addWidget(self.label_13, 2, 0, 1, 1)

        self.scaling = QDoubleSpinBox(self.groupBox_4)
        self.scaling.setObjectName(u"scaling")
        self.scaling.setDecimals(6)

        self.gridLayout_2.addWidget(self.scaling, 2, 1, 1, 1)

        self.y_top = QLabel(self.groupBox_4)
        self.y_top.setObjectName(u"y_top")

        self.gridLayout_2.addWidget(self.y_top, 0, 1, 1, 1)

        self.y_bottom = QLabel(self.groupBox_4)
        self.y_bottom.setObjectName(u"y_bottom")

        self.gridLayout_2.addWidget(self.y_bottom, 1, 1, 1, 1)


        self.gridLayout_3.addWidget(self.groupBox_4, 3, 1, 1, 1)

        self.signal = QComboBox(ScaleDialog)
        self.signal.setObjectName(u"signal")

        self.gridLayout_3.addWidget(self.signal, 0, 0, 1, 2)


        self.retranslateUi(ScaleDialog)

        QMetaObject.connectSlotsByName(ScaleDialog)
    # setupUi

    def retranslateUi(self, ScaleDialog):
        ScaleDialog.setWindowTitle(QCoreApplication.translate("ScaleDialog", u"Dialog", None))
        self.cancel_btn.setText(QCoreApplication.translate("ScaleDialog", u"Cancel", None))
        self.apply_btn.setText(QCoreApplication.translate("ScaleDialog", u"Apply", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("ScaleDialog", u"Keyboard shortcuts", None))
        self.fit_btn.setText(QCoreApplication.translate("ScaleDialog", u"Fit vertically", None))
        self.label_3.setText(QCoreApplication.translate("ScaleDialog", u"F", None))
        self.zoom_out_btn.setText(QCoreApplication.translate("ScaleDialog", u"Zoom out", None))
        self.label.setText(QCoreApplication.translate("ScaleDialog", u"O", None))
        self.fast_shift_up_btn.setText(QCoreApplication.translate("ScaleDialog", u"Fast shift up", None))
        self.label_2.setText(QCoreApplication.translate("ScaleDialog", u"I", None))
        self.shift_down_btn.setText(QCoreApplication.translate("ScaleDialog", u"Shift down", None))
        self.shift_up_btn.setText(QCoreApplication.translate("ScaleDialog", u"Shift up", None))
        self.zoom_in_btn.setText(QCoreApplication.translate("ScaleDialog", u"Zoom in", None))
        self.fast_shift_down_btn.setText(QCoreApplication.translate("ScaleDialog", u"Fast shift down", None))
        self.plot.setText("")
        self.groupBox_3.setTitle(QCoreApplication.translate("ScaleDialog", u"Expected signal values", None))
        self.label_5.setText(QCoreApplication.translate("ScaleDialog", u"Signal max", None))
        self.label_9.setText(QCoreApplication.translate("ScaleDialog", u"Signal min", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("ScaleDialog", u"Screen levels", None))
        self.label_10.setText(QCoreApplication.translate("ScaleDialog", u"0 level on screen", None))
        self.label_8.setText(QCoreApplication.translate("ScaleDialog", u"Y axis bottom value", None))
        self.offset.setSuffix(QCoreApplication.translate("ScaleDialog", u"%", None))
        self.label_4.setText(QCoreApplication.translate("ScaleDialog", u"Y axis top value", None))
        self.label_13.setText(QCoreApplication.translate("ScaleDialog", u"Scaling [units/screen]", None))
        self.y_top.setText("")
        self.y_bottom.setText("")
    # retranslateUi


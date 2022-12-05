# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'channel_stats.ui'
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
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QComboBox, QGridLayout,
    QGroupBox, QLabel, QScrollArea, QSizePolicy,
    QSpacerItem, QTextEdit, QVBoxLayout, QWidget)

class Ui_ChannelStats(object):
    def setupUi(self, ChannelStats):
        if not ChannelStats.objectName():
            ChannelStats.setObjectName(u"ChannelStats")
        ChannelStats.resize(315, 852)
        font = QFont()
        font.setFamilies([u"Roboto Condensed"])
        font.setPointSize(10)
        ChannelStats.setFont(font)
        self.verticalLayout_2 = QVBoxLayout(ChannelStats)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(2, 2, 2, 2)
        self.scrollArea = QScrollArea(ChannelStats)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 309, 846))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.gridLayout = QGridLayout()
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.visible_group = QGroupBox(self.scrollAreaWidgetContents)
        self.visible_group.setObjectName(u"visible_group")
        self.gridLayout_3 = QGridLayout(self.visible_group)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setHorizontalSpacing(2)
        self.gridLayout_3.setVerticalSpacing(0)
        self.gridLayout_3.setContentsMargins(2, 2, 2, 2)
        self.visible_start = QLabel(self.visible_group)
        self.visible_start.setObjectName(u"visible_start")
        self.visible_start.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_start, 0, 1, 1, 1)

        self.unit7 = QLabel(self.visible_group)
        self.unit7.setObjectName(u"unit7")
        self.unit7.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.unit7, 8, 2, 1, 1)

        self.label_10 = QLabel(self.visible_group)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_3.addWidget(self.label_10, 4, 0, 1, 1)

        self.label_35 = QLabel(self.visible_group)
        self.label_35.setObjectName(u"label_35")

        self.gridLayout_3.addWidget(self.label_35, 7, 0, 1, 1)

        self.unit5 = QLabel(self.visible_group)
        self.unit5.setObjectName(u"unit5")
        self.unit5.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.unit5, 3, 2, 1, 1)

        self.visible_stop = QLabel(self.visible_group)
        self.visible_stop.setObjectName(u"visible_stop")
        self.visible_stop.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_stop, 1, 1, 1, 1)

        self.label_19 = QLabel(self.visible_group)
        self.label_19.setObjectName(u"label_19")

        self.gridLayout_3.addWidget(self.label_19, 8, 0, 1, 1)

        self.label_9 = QLabel(self.visible_group)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_3.addWidget(self.label_9, 3, 0, 1, 1)

        self.visible_rms = QLabel(self.visible_group)
        self.visible_rms.setObjectName(u"visible_rms")
        self.visible_rms.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_rms, 6, 1, 1, 1)

        self.unit17 = QLabel(self.visible_group)
        self.unit17.setObjectName(u"unit17")
        self.unit17.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.unit17, 7, 2, 1, 1)

        self.label_16 = QLabel(self.visible_group)
        self.label_16.setObjectName(u"label_16")

        self.gridLayout_3.addWidget(self.label_16, 2, 0, 1, 1)

        self.visible_min = QLabel(self.visible_group)
        self.visible_min.setObjectName(u"visible_min")
        self.visible_min.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_min, 3, 1, 1, 1)

        self.visible_std = QLabel(self.visible_group)
        self.visible_std.setObjectName(u"visible_std")
        self.visible_std.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_std, 7, 1, 1, 1)

        self.xunit4 = QLabel(self.visible_group)
        self.xunit4.setObjectName(u"xunit4")
        self.xunit4.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.xunit4, 0, 2, 1, 1)

        self.label_39 = QLabel(self.visible_group)
        self.label_39.setObjectName(u"label_39")

        self.gridLayout_3.addWidget(self.label_39, 9, 0, 1, 1)

        self.label_8 = QLabel(self.visible_group)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_3.addWidget(self.label_8, 1, 0, 1, 1)

        self.label_7 = QLabel(self.visible_group)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_3.addWidget(self.label_7, 0, 0, 1, 1)

        self.unit6 = QLabel(self.visible_group)
        self.unit6.setObjectName(u"unit6")
        self.unit6.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.unit6, 4, 2, 1, 1)

        self.xunit6 = QLabel(self.visible_group)
        self.xunit6.setObjectName(u"xunit6")
        self.xunit6.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.xunit6, 2, 2, 1, 1)

        self.visible_delta = QLabel(self.visible_group)
        self.visible_delta.setObjectName(u"visible_delta")
        self.visible_delta.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_delta, 8, 1, 1, 1)

        self.label_31 = QLabel(self.visible_group)
        self.label_31.setObjectName(u"label_31")

        self.gridLayout_3.addWidget(self.label_31, 6, 0, 1, 1)

        self.unit12 = QLabel(self.visible_group)
        self.unit12.setObjectName(u"unit12")
        self.unit12.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.unit12, 5, 2, 1, 1)

        self.label_30 = QLabel(self.visible_group)
        self.label_30.setObjectName(u"label_30")

        self.gridLayout_3.addWidget(self.label_30, 5, 0, 1, 1)

        self.visible_max = QLabel(self.visible_group)
        self.visible_max.setObjectName(u"visible_max")
        self.visible_max.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_max, 4, 1, 1, 1)

        self.visible_delta_t = QLabel(self.visible_group)
        self.visible_delta_t.setObjectName(u"visible_delta_t")
        self.visible_delta_t.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_delta_t, 2, 1, 1, 1)

        self.visible_average = QLabel(self.visible_group)
        self.visible_average.setObjectName(u"visible_average")
        self.visible_average.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_average, 5, 1, 1, 1)

        self.xunit5 = QLabel(self.visible_group)
        self.xunit5.setObjectName(u"xunit5")
        self.xunit5.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.xunit5, 1, 2, 1, 1)

        self.unit13 = QLabel(self.visible_group)
        self.unit13.setObjectName(u"unit13")
        self.unit13.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.unit13, 6, 2, 1, 1)

        self.label_40 = QLabel(self.visible_group)
        self.label_40.setObjectName(u"label_40")

        self.gridLayout_3.addWidget(self.label_40, 10, 0, 1, 1)

        self.visible_gradient_unit = QLabel(self.visible_group)
        self.visible_gradient_unit.setObjectName(u"visible_gradient_unit")
        self.visible_gradient_unit.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_gradient_unit, 9, 2, 1, 1)

        self.visible_integral_unit = QLabel(self.visible_group)
        self.visible_integral_unit.setObjectName(u"visible_integral_unit")
        self.visible_integral_unit.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_integral_unit, 10, 2, 1, 1)

        self.visible_gradient = QLabel(self.visible_group)
        self.visible_gradient.setObjectName(u"visible_gradient")
        self.visible_gradient.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_gradient, 9, 1, 1, 1)

        self.visible_integral = QLabel(self.visible_group)
        self.visible_integral.setObjectName(u"visible_integral")
        self.visible_integral.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.visible_integral, 10, 1, 1, 1)

        self.gridLayout_3.setColumnStretch(1, 1)

        self.gridLayout.addWidget(self.visible_group, 4, 0, 1, 2)

        self.overall_group = QGroupBox(self.scrollAreaWidgetContents)
        self.overall_group.setObjectName(u"overall_group")
        self.gridLayout_2 = QGridLayout(self.overall_group)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setHorizontalSpacing(2)
        self.gridLayout_2.setVerticalSpacing(0)
        self.gridLayout_2.setContentsMargins(2, 2, 2, 2)
        self.xunit7 = QLabel(self.overall_group)
        self.xunit7.setObjectName(u"xunit7")
        self.xunit7.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.xunit7, 0, 2, 1, 1)

        self.overall_delta = QLabel(self.overall_group)
        self.overall_delta.setObjectName(u"overall_delta")
        self.overall_delta.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_delta, 9, 1, 1, 1)

        self.overall_max = QLabel(self.overall_group)
        self.overall_max.setObjectName(u"overall_max")
        self.overall_max.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_max, 5, 1, 1, 1)

        self.overall_stop = QLabel(self.overall_group)
        self.overall_stop.setObjectName(u"overall_stop")
        self.overall_stop.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_stop, 1, 1, 1, 1)

        self.label_43 = QLabel(self.overall_group)
        self.label_43.setObjectName(u"label_43")

        self.gridLayout_2.addWidget(self.label_43, 9, 0, 1, 1)

        self.label_6 = QLabel(self.overall_group)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_2.addWidget(self.label_6, 5, 0, 1, 1)

        self.overall_std = QLabel(self.overall_group)
        self.overall_std.setObjectName(u"overall_std")
        self.overall_std.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_std, 8, 1, 1, 1)

        self.label_3 = QLabel(self.overall_group)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)

        self.label_32 = QLabel(self.overall_group)
        self.label_32.setObjectName(u"label_32")

        self.gridLayout_2.addWidget(self.label_32, 6, 0, 1, 1)

        self.unit9 = QLabel(self.overall_group)
        self.unit9.setObjectName(u"unit9")
        self.unit9.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.unit9, 5, 2, 1, 1)

        self.label_41 = QLabel(self.overall_group)
        self.label_41.setObjectName(u"label_41")

        self.gridLayout_2.addWidget(self.label_41, 10, 0, 1, 1)

        self.overall_rms = QLabel(self.overall_group)
        self.overall_rms.setObjectName(u"overall_rms")
        self.overall_rms.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_rms, 7, 1, 1, 1)

        self.xunit8 = QLabel(self.overall_group)
        self.xunit8.setObjectName(u"xunit8")
        self.xunit8.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.xunit8, 1, 2, 1, 1)

        self.overall_average = QLabel(self.overall_group)
        self.overall_average.setObjectName(u"overall_average")
        self.overall_average.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_average, 6, 1, 1, 1)

        self.unit8 = QLabel(self.overall_group)
        self.unit8.setObjectName(u"unit8")
        self.unit8.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.unit8, 4, 2, 1, 1)

        self.label_4 = QLabel(self.overall_group)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)

        self.label_5 = QLabel(self.overall_group)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_2.addWidget(self.label_5, 4, 0, 1, 1)

        self.unit18 = QLabel(self.overall_group)
        self.unit18.setObjectName(u"unit18")
        self.unit18.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.unit18, 8, 2, 1, 1)

        self.overall_integral_unit = QLabel(self.overall_group)
        self.overall_integral_unit.setObjectName(u"overall_integral_unit")
        self.overall_integral_unit.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_integral_unit, 11, 2, 1, 1)

        self.unit19 = QLabel(self.overall_group)
        self.unit19.setObjectName(u"unit19")
        self.unit19.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.unit19, 9, 2, 1, 1)

        self.overall_gradient_unit = QLabel(self.overall_group)
        self.overall_gradient_unit.setObjectName(u"overall_gradient_unit")
        self.overall_gradient_unit.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_gradient_unit, 10, 2, 1, 1)

        self.unit14 = QLabel(self.overall_group)
        self.unit14.setObjectName(u"unit14")
        self.unit14.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.unit14, 6, 2, 1, 1)

        self.label_33 = QLabel(self.overall_group)
        self.label_33.setObjectName(u"label_33")

        self.gridLayout_2.addWidget(self.label_33, 7, 0, 1, 1)

        self.unit15 = QLabel(self.overall_group)
        self.unit15.setObjectName(u"unit15")
        self.unit15.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.unit15, 7, 2, 1, 1)

        self.label_36 = QLabel(self.overall_group)
        self.label_36.setObjectName(u"label_36")

        self.gridLayout_2.addWidget(self.label_36, 8, 0, 1, 1)

        self.label_42 = QLabel(self.overall_group)
        self.label_42.setObjectName(u"label_42")

        self.gridLayout_2.addWidget(self.label_42, 11, 0, 1, 1)

        self.overall_start = QLabel(self.overall_group)
        self.overall_start.setObjectName(u"overall_start")
        self.overall_start.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_start, 0, 1, 1, 1)

        self.overall_min = QLabel(self.overall_group)
        self.overall_min.setObjectName(u"overall_min")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overall_min.sizePolicy().hasHeightForWidth())
        self.overall_min.setSizePolicy(sizePolicy)
        self.overall_min.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_min, 4, 1, 1, 1)

        self.overall_integral = QLabel(self.overall_group)
        self.overall_integral.setObjectName(u"overall_integral")
        self.overall_integral.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_integral, 11, 1, 1, 1)

        self.overall_gradient = QLabel(self.overall_group)
        self.overall_gradient.setObjectName(u"overall_gradient")
        self.overall_gradient.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_gradient, 10, 1, 1, 1)

        self.label_44 = QLabel(self.overall_group)
        self.label_44.setObjectName(u"label_44")

        self.gridLayout_2.addWidget(self.label_44, 3, 0, 1, 1)

        self.overall_delta_t = QLabel(self.overall_group)
        self.overall_delta_t.setObjectName(u"overall_delta_t")
        self.overall_delta_t.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.overall_delta_t, 3, 1, 1, 1)

        self.xunit9 = QLabel(self.overall_group)
        self.xunit9.setObjectName(u"xunit9")
        self.xunit9.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.xunit9, 3, 2, 1, 1)

        self.gridLayout_2.setColumnStretch(1, 1)

        self.gridLayout.addWidget(self.overall_group, 5, 0, 1, 2)

        self.cursor_group = QGroupBox(self.scrollAreaWidgetContents)
        self.cursor_group.setObjectName(u"cursor_group")
        self.cursor_group.setFlat(False)
        self.cursor_group.setCheckable(False)
        self.gridLayout_4 = QGridLayout(self.cursor_group)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setHorizontalSpacing(2)
        self.gridLayout_4.setVerticalSpacing(0)
        self.gridLayout_4.setContentsMargins(2, 2, 2, 2)
        self.cursor_t = QLabel(self.cursor_group)
        self.cursor_t.setObjectName(u"cursor_t")
        self.cursor_t.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_4.addWidget(self.cursor_t, 0, 1, 1, 1)

        self.label_12 = QLabel(self.cursor_group)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout_4.addWidget(self.label_12, 0, 0, 1, 1)

        self.cursor_value = QLabel(self.cursor_group)
        self.cursor_value.setObjectName(u"cursor_value")
        self.cursor_value.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_4.addWidget(self.cursor_value, 1, 1, 1, 1)

        self.label_11 = QLabel(self.cursor_group)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_4.addWidget(self.label_11, 1, 0, 1, 1)

        self.xunit0 = QLabel(self.cursor_group)
        self.xunit0.setObjectName(u"xunit0")
        self.xunit0.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_4.addWidget(self.xunit0, 0, 2, 1, 1)

        self.unit1 = QLabel(self.cursor_group)
        self.unit1.setObjectName(u"unit1")
        self.unit1.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_4.addWidget(self.unit1, 1, 2, 1, 1)

        self.gridLayout_4.setColumnStretch(1, 1)

        self.gridLayout.addWidget(self.cursor_group, 2, 0, 1, 2)

        self.name = QTextEdit(self.scrollAreaWidgetContents)
        self.name.setObjectName(u"name")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(1)
        sizePolicy1.setHeightForWidth(self.name.sizePolicy().hasHeightForWidth())
        self.name.setSizePolicy(sizePolicy1)
        self.name.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.name.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.name.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.name.setReadOnly(True)

        self.gridLayout.addWidget(self.name, 0, 0, 1, 2)

        self.region_group = QGroupBox(self.scrollAreaWidgetContents)
        self.region_group.setObjectName(u"region_group")
        self.gridLayout_5 = QGridLayout(self.region_group)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setHorizontalSpacing(2)
        self.gridLayout_5.setVerticalSpacing(0)
        self.gridLayout_5.setContentsMargins(2, 2, 2, 2)
        self.label_13 = QLabel(self.region_group)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout_5.addWidget(self.label_13, 0, 0, 1, 1)

        self.selected_integral_unit = QLabel(self.region_group)
        self.selected_integral_unit.setObjectName(u"selected_integral_unit")
        self.selected_integral_unit.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_integral_unit, 13, 2, 1, 1)

        self.selected_rms = QLabel(self.region_group)
        self.selected_rms.setObjectName(u"selected_rms")
        self.selected_rms.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_rms, 9, 1, 1, 1)

        self.label_29 = QLabel(self.region_group)
        self.label_29.setObjectName(u"label_29")

        self.gridLayout_5.addWidget(self.label_29, 9, 0, 1, 1)

        self.selected_gradient_unit = QLabel(self.region_group)
        self.selected_gradient_unit.setObjectName(u"selected_gradient_unit")
        self.selected_gradient_unit.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_gradient_unit, 12, 2, 1, 1)

        self.selected_start = QLabel(self.region_group)
        self.selected_start.setObjectName(u"selected_start")
        self.selected_start.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_start, 0, 1, 1, 1)

        self.label_15 = QLabel(self.region_group)
        self.label_15.setObjectName(u"label_15")

        self.gridLayout_5.addWidget(self.label_15, 1, 0, 1, 1)

        self.selected_delta_t = QLabel(self.region_group)
        self.selected_delta_t.setObjectName(u"selected_delta_t")
        self.selected_delta_t.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_delta_t, 3, 1, 1, 1)

        self.label_2 = QLabel(self.region_group)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_5.addWidget(self.label_2, 4, 0, 1, 1)

        self.label_14 = QLabel(self.region_group)
        self.label_14.setObjectName(u"label_14")

        self.gridLayout_5.addWidget(self.label_14, 6, 0, 1, 1)

        self.label = QLabel(self.region_group)
        self.label.setObjectName(u"label")

        self.gridLayout_5.addWidget(self.label, 8, 0, 1, 1)

        self.selected_stop = QLabel(self.region_group)
        self.selected_stop.setObjectName(u"selected_stop")
        self.selected_stop.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_stop, 1, 1, 1, 1)

        self.unit16 = QLabel(self.region_group)
        self.unit16.setObjectName(u"unit16")
        self.unit16.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.unit16, 10, 2, 1, 1)

        self.label_34 = QLabel(self.region_group)
        self.label_34.setObjectName(u"label_34")

        self.gridLayout_5.addWidget(self.label_34, 10, 0, 1, 1)

        self.xunit1 = QLabel(self.region_group)
        self.xunit1.setObjectName(u"xunit1")
        self.xunit1.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.xunit1, 0, 2, 1, 1)

        self.xunit3 = QLabel(self.region_group)
        self.xunit3.setObjectName(u"xunit3")
        self.xunit3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.xunit3, 3, 2, 1, 1)

        self.label_21 = QLabel(self.region_group)
        self.label_21.setObjectName(u"label_21")

        self.gridLayout_5.addWidget(self.label_21, 3, 0, 1, 1)

        self.selected_max = QLabel(self.region_group)
        self.selected_max.setObjectName(u"selected_max")
        self.selected_max.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_max, 7, 1, 1, 1)

        self.label_17 = QLabel(self.region_group)
        self.label_17.setObjectName(u"label_17")

        self.gridLayout_5.addWidget(self.label_17, 7, 0, 1, 1)

        self.label_37 = QLabel(self.region_group)
        self.label_37.setObjectName(u"label_37")

        self.gridLayout_5.addWidget(self.label_37, 12, 0, 1, 1)

        self.selected_std = QLabel(self.region_group)
        self.selected_std.setObjectName(u"selected_std")
        self.selected_std.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_std, 10, 1, 1, 1)

        self.label_18 = QLabel(self.region_group)
        self.label_18.setObjectName(u"label_18")

        self.gridLayout_5.addWidget(self.label_18, 11, 0, 1, 1)

        self.xunit2 = QLabel(self.region_group)
        self.xunit2.setObjectName(u"xunit2")
        self.xunit2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.xunit2, 1, 2, 1, 1)

        self.unit11 = QLabel(self.region_group)
        self.unit11.setObjectName(u"unit11")
        self.unit11.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.unit11, 9, 2, 1, 1)

        self.selected_min = QLabel(self.region_group)
        self.selected_min.setObjectName(u"selected_min")
        self.selected_min.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_min, 6, 1, 1, 1)

        self.label_38 = QLabel(self.region_group)
        self.label_38.setObjectName(u"label_38")

        self.gridLayout_5.addWidget(self.label_38, 13, 0, 1, 1)

        self.unit3 = QLabel(self.region_group)
        self.unit3.setObjectName(u"unit3")
        self.unit3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.unit3, 7, 2, 1, 1)

        self.selected_delta = QLabel(self.region_group)
        self.selected_delta.setObjectName(u"selected_delta")
        self.selected_delta.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_delta, 11, 1, 1, 1)

        self.selected_integral = QLabel(self.region_group)
        self.selected_integral.setObjectName(u"selected_integral")
        self.selected_integral.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_integral, 13, 1, 1, 1)

        self.unit4 = QLabel(self.region_group)
        self.unit4.setObjectName(u"unit4")
        self.unit4.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.unit4, 11, 2, 1, 1)

        self.selected_average = QLabel(self.region_group)
        self.selected_average.setObjectName(u"selected_average")
        self.selected_average.setAutoFillBackground(False)
        self.selected_average.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_average, 8, 1, 1, 1)

        self.selected_gradient = QLabel(self.region_group)
        self.selected_gradient.setObjectName(u"selected_gradient")
        self.selected_gradient.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_gradient, 12, 1, 1, 1)

        self.unit10 = QLabel(self.region_group)
        self.unit10.setObjectName(u"unit10")
        self.unit10.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.unit10, 8, 2, 1, 1)

        self.unit2 = QLabel(self.region_group)
        self.unit2.setObjectName(u"unit2")
        self.unit2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.unit2, 6, 2, 1, 1)

        self.label_20 = QLabel(self.region_group)
        self.label_20.setObjectName(u"label_20")

        self.gridLayout_5.addWidget(self.label_20, 5, 0, 1, 1)

        self.selected_left = QLabel(self.region_group)
        self.selected_left.setObjectName(u"selected_left")
        self.selected_left.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_left, 4, 1, 1, 1)

        self.selected_right = QLabel(self.region_group)
        self.selected_right.setObjectName(u"selected_right")
        self.selected_right.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.selected_right, 5, 1, 1, 1)

        self.unit21 = QLabel(self.region_group)
        self.unit21.setObjectName(u"unit21")
        self.unit21.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.unit21, 4, 2, 1, 1)

        self.unit22 = QLabel(self.region_group)
        self.unit22.setObjectName(u"unit22")
        self.unit22.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_5.addWidget(self.unit22, 5, 2, 1, 1)

        self.gridLayout_5.setColumnStretch(1, 1)

        self.gridLayout.addWidget(self.region_group, 3, 0, 1, 2)

        self.precision = QComboBox(self.scrollAreaWidgetContents)
        self.precision.setObjectName(u"precision")

        self.gridLayout.addWidget(self.precision, 1, 1, 1, 1)

        self.label_22 = QLabel(self.scrollAreaWidgetContents)
        self.label_22.setObjectName(u"label_22")

        self.gridLayout.addWidget(self.label_22, 1, 0, 1, 1)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)

        self.verticalLayout.addLayout(self.gridLayout)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_2.addWidget(self.scrollArea)


        self.retranslateUi(ChannelStats)

        QMetaObject.connectSlotsByName(ChannelStats)
    # setupUi

    def retranslateUi(self, ChannelStats):
        ChannelStats.setWindowTitle(QCoreApplication.translate("ChannelStats", u"Form", None))
        self.visible_group.setTitle(QCoreApplication.translate("ChannelStats", u"Visible region", None))
        self.visible_start.setText("")
        self.unit7.setText("")
        self.label_10.setText(QCoreApplication.translate("ChannelStats", u"Max", None))
        self.label_35.setText(QCoreApplication.translate("ChannelStats", u"STD", None))
        self.unit5.setText("")
        self.visible_stop.setText("")
        self.label_19.setText(QCoreApplication.translate("ChannelStats", u"\u0394", None))
        self.label_9.setText(QCoreApplication.translate("ChannelStats", u"Min", None))
        self.visible_rms.setText("")
        self.unit17.setText("")
        self.label_16.setText(QCoreApplication.translate("ChannelStats", u"\u0394t", None))
        self.visible_min.setText("")
        self.visible_std.setText("")
        self.xunit4.setText(QCoreApplication.translate("ChannelStats", u" s", None))
        self.label_39.setText(QCoreApplication.translate("ChannelStats", u"Gradient", None))
        self.label_8.setText(QCoreApplication.translate("ChannelStats", u"t2", None))
        self.label_7.setText(QCoreApplication.translate("ChannelStats", u"t1", None))
        self.unit6.setText("")
        self.xunit6.setText(QCoreApplication.translate("ChannelStats", u" s", None))
        self.visible_delta.setText("")
        self.label_31.setText(QCoreApplication.translate("ChannelStats", u"RMS", None))
        self.unit12.setText("")
        self.label_30.setText(QCoreApplication.translate("ChannelStats", u"Avg", None))
        self.visible_max.setText("")
        self.visible_delta_t.setText("")
        self.visible_average.setText("")
        self.xunit5.setText(QCoreApplication.translate("ChannelStats", u" s", None))
        self.unit13.setText("")
        self.label_40.setText(QCoreApplication.translate("ChannelStats", u"Integral", None))
        self.visible_gradient_unit.setText("")
        self.visible_integral_unit.setText("")
        self.visible_gradient.setText("")
        self.visible_integral.setText("")
        self.overall_group.setTitle(QCoreApplication.translate("ChannelStats", u"Overall", None))
        self.xunit7.setText(QCoreApplication.translate("ChannelStats", u" s", None))
        self.overall_delta.setText("")
        self.overall_max.setText("")
        self.overall_stop.setText("")
        self.label_43.setText(QCoreApplication.translate("ChannelStats", u"\u0394", None))
        self.label_6.setText(QCoreApplication.translate("ChannelStats", u"Max", None))
        self.overall_std.setText("")
        self.label_3.setText(QCoreApplication.translate("ChannelStats", u"t1", None))
        self.label_32.setText(QCoreApplication.translate("ChannelStats", u"Avg", None))
        self.unit9.setText("")
        self.label_41.setText(QCoreApplication.translate("ChannelStats", u"Grad", None))
        self.overall_rms.setText("")
        self.xunit8.setText(QCoreApplication.translate("ChannelStats", u" s", None))
        self.overall_average.setText("")
        self.unit8.setText("")
        self.label_4.setText(QCoreApplication.translate("ChannelStats", u"t2", None))
        self.label_5.setText(QCoreApplication.translate("ChannelStats", u"Min", None))
        self.unit18.setText("")
        self.overall_integral_unit.setText("")
        self.unit19.setText("")
        self.overall_gradient_unit.setText("")
        self.unit14.setText("")
        self.label_33.setText(QCoreApplication.translate("ChannelStats", u"RMS", None))
        self.unit15.setText("")
        self.label_36.setText(QCoreApplication.translate("ChannelStats", u"STD", None))
        self.label_42.setText(QCoreApplication.translate("ChannelStats", u"Integral", None))
        self.overall_start.setText("")
        self.overall_min.setText("")
        self.overall_integral.setText("")
        self.overall_gradient.setText("")
        self.label_44.setText(QCoreApplication.translate("ChannelStats", u"\u0394t", None))
        self.overall_delta_t.setText("")
        self.xunit9.setText(QCoreApplication.translate("ChannelStats", u" s", None))
        self.cursor_group.setTitle(QCoreApplication.translate("ChannelStats", u"Cursor", None))
        self.cursor_t.setText("")
        self.label_12.setText(QCoreApplication.translate("ChannelStats", u"Time", None))
        self.cursor_value.setText("")
        self.label_11.setText(QCoreApplication.translate("ChannelStats", u"Value", None))
        self.xunit0.setText(QCoreApplication.translate("ChannelStats", u" s", None))
        self.unit1.setText("")
        self.name.setHtml(QCoreApplication.translate("ChannelStats", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"</style></head><body style=\" font-family:'Roboto Condensed'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:'Consolas';\"><br /></p></body></html>", None))
        self.region_group.setTitle(QCoreApplication.translate("ChannelStats", u"Selected region", None))
        self.label_13.setText(QCoreApplication.translate("ChannelStats", u"t1", None))
        self.selected_integral_unit.setText("")
        self.selected_rms.setText("")
        self.label_29.setText(QCoreApplication.translate("ChannelStats", u"RMS", None))
        self.selected_gradient_unit.setText("")
        self.selected_start.setText("")
        self.label_15.setText(QCoreApplication.translate("ChannelStats", u"t2", None))
        self.selected_delta_t.setText("")
        self.label_2.setText(QCoreApplication.translate("ChannelStats", u"First", None))
        self.label_14.setText(QCoreApplication.translate("ChannelStats", u"Min", None))
        self.label.setText(QCoreApplication.translate("ChannelStats", u"Avg", None))
        self.selected_stop.setText("")
        self.unit16.setText("")
        self.label_34.setText(QCoreApplication.translate("ChannelStats", u"STD", None))
        self.xunit1.setText(QCoreApplication.translate("ChannelStats", u" s", None))
        self.xunit3.setText(QCoreApplication.translate("ChannelStats", u" s", None))
        self.label_21.setText(QCoreApplication.translate("ChannelStats", u"\u0394t", None))
        self.selected_max.setText("")
        self.label_17.setText(QCoreApplication.translate("ChannelStats", u"Max", None))
        self.label_37.setText(QCoreApplication.translate("ChannelStats", u"Gradient", None))
        self.selected_std.setText("")
        self.label_18.setText(QCoreApplication.translate("ChannelStats", u"\u0394", None))
        self.xunit2.setText(QCoreApplication.translate("ChannelStats", u" s", None))
        self.unit11.setText("")
        self.selected_min.setText("")
        self.label_38.setText(QCoreApplication.translate("ChannelStats", u"Integral", None))
        self.unit3.setText("")
        self.selected_delta.setText("")
        self.selected_integral.setText("")
        self.unit4.setText("")
        self.selected_average.setText("")
        self.selected_gradient.setText("")
        self.unit10.setText("")
        self.unit2.setText("")
        self.label_20.setText(QCoreApplication.translate("ChannelStats", u"Last", None))
        self.selected_left.setText("")
        self.selected_right.setText("")
        self.unit21.setText("")
        self.unit22.setText("")
        self.label_22.setText(QCoreApplication.translate("ChannelStats", u"Precisison", None))
    # retranslateUi


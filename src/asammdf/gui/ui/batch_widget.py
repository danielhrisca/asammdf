# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'batch_widget.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListView,
    QListWidget, QListWidgetItem, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSpacerItem, QSplitter,
    QStackedWidget, QTabWidget, QTextEdit, QTreeWidgetItem,
    QVBoxLayout, QWidget)

from asammdf.gui.widgets.list import MinimalListWidget
from asammdf.gui.widgets.tree import TreeWidget
from . import resource_rc

class Ui_batch_widget(object):
    def setupUi(self, batch_widget):
        if not batch_widget.objectName():
            batch_widget.setObjectName(u"batch_widget")
        batch_widget.resize(1312, 718)
        self.gridLayout_9 = QGridLayout(batch_widget)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.splitter = QSplitter(batch_widget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.aspects = QTabWidget(self.splitter)
        self.aspects.setObjectName(u"aspects")
        self.aspects.setTabPosition(QTabWidget.West)
        self.aspects.setDocumentMode(False)
        self.concatenate_tab = QWidget()
        self.concatenate_tab.setObjectName(u"concatenate_tab")
        self.gridLayout_7 = QGridLayout(self.concatenate_tab)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.concatenate_sync = QCheckBox(self.concatenate_tab)
        self.concatenate_sync.setObjectName(u"concatenate_sync")
        self.concatenate_sync.setChecked(True)

        self.gridLayout_7.addWidget(self.concatenate_sync, 0, 0, 1, 2)

        self.concatenate_add_samples_origin = QCheckBox(self.concatenate_tab)
        self.concatenate_add_samples_origin.setObjectName(u"concatenate_add_samples_origin")
        self.concatenate_add_samples_origin.setChecked(True)

        self.gridLayout_7.addWidget(self.concatenate_add_samples_origin, 1, 0, 1, 2)

        self.label_11 = QLabel(self.concatenate_tab)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_7.addWidget(self.label_11, 2, 0, 1, 1)

        self.concatenate_format = QComboBox(self.concatenate_tab)
        self.concatenate_format.setObjectName(u"concatenate_format")

        self.gridLayout_7.addWidget(self.concatenate_format, 2, 1, 1, 1)

        self.line_5 = QFrame(self.concatenate_tab)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setFrameShape(QFrame.HLine)
        self.line_5.setFrameShadow(QFrame.Sunken)

        self.gridLayout_7.addWidget(self.line_5, 3, 0, 1, 2)

        self.label_12 = QLabel(self.concatenate_tab)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout_7.addWidget(self.label_12, 4, 0, 1, 1)

        self.concatenate_compression = QComboBox(self.concatenate_tab)
        self.concatenate_compression.setObjectName(u"concatenate_compression")

        self.gridLayout_7.addWidget(self.concatenate_compression, 4, 1, 1, 1)

        self.concatenate_split = QCheckBox(self.concatenate_tab)
        self.concatenate_split.setObjectName(u"concatenate_split")
        self.concatenate_split.setChecked(True)

        self.gridLayout_7.addWidget(self.concatenate_split, 5, 0, 1, 1)

        self.line_6 = QFrame(self.concatenate_tab)
        self.line_6.setObjectName(u"line_6")
        self.line_6.setFrameShape(QFrame.HLine)
        self.line_6.setFrameShadow(QFrame.Sunken)

        self.gridLayout_7.addWidget(self.line_6, 7, 0, 1, 2)

        self.concatenate_split_size = QDoubleSpinBox(self.concatenate_tab)
        self.concatenate_split_size.setObjectName(u"concatenate_split_size")
        self.concatenate_split_size.setMaximum(4.000000000000000)

        self.gridLayout_7.addWidget(self.concatenate_split_size, 6, 1, 1, 1)

        self.label_10 = QLabel(self.concatenate_tab)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_7.addWidget(self.label_10, 6, 0, 1, 1)

        self.concatenate_btn = QPushButton(self.concatenate_tab)
        self.concatenate_btn.setObjectName(u"concatenate_btn")
        icon = QIcon()
        icon.addFile(u":/plus.png", QSize(), QIcon.Normal, QIcon.Off)
        self.concatenate_btn.setIcon(icon)

        self.gridLayout_7.addWidget(self.concatenate_btn, 9, 1, 1, 1)

        self.verticalSpacer_4 = QSpacerItem(20, 2, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_7.addItem(self.verticalSpacer_4, 8, 1, 1, 1)

        self.gridLayout_7.setColumnStretch(1, 1)
        self.aspects.addTab(self.concatenate_tab, icon, "")
        self.convert_tab = QWidget()
        self.convert_tab.setObjectName(u"convert_tab")
        self.verticalLayout_5 = QVBoxLayout(self.convert_tab)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.scrollArea_3 = QScrollArea(self.convert_tab)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 810, 674))
        self.horizontalLayout_2 = QHBoxLayout(self.scrollAreaWidgetContents_3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.verticalLayout_14 = QVBoxLayout()
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.filter_view = QComboBox(self.scrollAreaWidgetContents_3)
        self.filter_view.addItem("")
        self.filter_view.addItem("")
        self.filter_view.addItem("")
        self.filter_view.setObjectName(u"filter_view")

        self.verticalLayout_14.addWidget(self.filter_view)

        self.filter_tree = TreeWidget(self.scrollAreaWidgetContents_3)
        self.filter_tree.setObjectName(u"filter_tree")

        self.verticalLayout_14.addWidget(self.filter_tree)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.load_filter_list_btn = QPushButton(self.scrollAreaWidgetContents_3)
        self.load_filter_list_btn.setObjectName(u"load_filter_list_btn")
        icon1 = QIcon()
        icon1.addFile(u":/open.png", QSize(), QIcon.Normal, QIcon.Off)
        self.load_filter_list_btn.setIcon(icon1)

        self.horizontalLayout_5.addWidget(self.load_filter_list_btn)

        self.save_filter_list_btn = QPushButton(self.scrollAreaWidgetContents_3)
        self.save_filter_list_btn.setObjectName(u"save_filter_list_btn")
        icon2 = QIcon()
        icon2.addFile(u":/save.png", QSize(), QIcon.Normal, QIcon.Off)
        self.save_filter_list_btn.setIcon(icon2)

        self.horizontalLayout_5.addWidget(self.save_filter_list_btn)

        self.clear_filter_btn = QPushButton(self.scrollAreaWidgetContents_3)
        self.clear_filter_btn.setObjectName(u"clear_filter_btn")
        icon3 = QIcon()
        icon3.addFile(u":/erase.png", QSize(), QIcon.Normal, QIcon.Off)
        self.clear_filter_btn.setIcon(icon3)

        self.horizontalLayout_5.addWidget(self.clear_filter_btn)

        self.advanced_serch_filter_btn = QPushButton(self.scrollAreaWidgetContents_3)
        self.advanced_serch_filter_btn.setObjectName(u"advanced_serch_filter_btn")
        icon4 = QIcon()
        icon4.addFile(u":/search.png", QSize(), QIcon.Normal, QIcon.Off)
        self.advanced_serch_filter_btn.setIcon(icon4)

        self.horizontalLayout_5.addWidget(self.advanced_serch_filter_btn)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_5)


        self.verticalLayout_14.addLayout(self.horizontalLayout_5)


        self.horizontalLayout_2.addLayout(self.verticalLayout_14)

        self.verticalLayout_13 = QVBoxLayout()
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.label_3 = QLabel(self.scrollAreaWidgetContents_3)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_13.addWidget(self.label_3)

        self.selected_filter_channels = QListWidget(self.scrollAreaWidgetContents_3)
        self.selected_filter_channels.setObjectName(u"selected_filter_channels")
        self.selected_filter_channels.setViewMode(QListView.ListMode)
        self.selected_filter_channels.setUniformItemSizes(True)
        self.selected_filter_channels.setSortingEnabled(True)

        self.verticalLayout_13.addWidget(self.selected_filter_channels)


        self.horizontalLayout_2.addLayout(self.verticalLayout_13)

        self.verticalLayout_15 = QVBoxLayout()
        self.verticalLayout_15.setSpacing(2)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_15.setContentsMargins(2, 2, 2, 2)
        self.cut_group = QGroupBox(self.scrollAreaWidgetContents_3)
        self.cut_group.setObjectName(u"cut_group")
        self.cut_group.setCheckable(True)
        self.cut_group.setChecked(False)
        self.gridLayout_23 = QGridLayout(self.cut_group)
        self.gridLayout_23.setSpacing(2)
        self.gridLayout_23.setObjectName(u"gridLayout_23")
        self.gridLayout_23.setContentsMargins(2, 2, 2, 2)
        self.label_62 = QLabel(self.cut_group)
        self.label_62.setObjectName(u"label_62")

        self.gridLayout_23.addWidget(self.label_62, 0, 0, 1, 1)

        self.cut_stop = QDoubleSpinBox(self.cut_group)
        self.cut_stop.setObjectName(u"cut_stop")
        self.cut_stop.setDecimals(6)
        self.cut_stop.setMaximum(999999.999998999992386)

        self.gridLayout_23.addWidget(self.cut_stop, 1, 1, 1, 1)

        self.label_63 = QLabel(self.cut_group)
        self.label_63.setObjectName(u"label_63")

        self.gridLayout_23.addWidget(self.label_63, 1, 0, 1, 1)

        self.cut_start = QDoubleSpinBox(self.cut_group)
        self.cut_start.setObjectName(u"cut_start")
        self.cut_start.setDecimals(6)
        self.cut_start.setMaximum(999999.999998999992386)

        self.gridLayout_23.addWidget(self.cut_start, 0, 1, 1, 1)

        self.whence = QCheckBox(self.cut_group)
        self.whence.setObjectName(u"whence")

        self.gridLayout_23.addWidget(self.whence, 3, 0, 1, 2)

        self.cut_time_from_zero = QCheckBox(self.cut_group)
        self.cut_time_from_zero.setObjectName(u"cut_time_from_zero")

        self.gridLayout_23.addWidget(self.cut_time_from_zero, 4, 0, 1, 2)


        self.verticalLayout_15.addWidget(self.cut_group)

        self.resample_group = QGroupBox(self.scrollAreaWidgetContents_3)
        self.resample_group.setObjectName(u"resample_group")
        self.resample_group.setCheckable(True)
        self.resample_group.setChecked(False)
        self.gridLayout_24 = QGridLayout(self.resample_group)
        self.gridLayout_24.setSpacing(2)
        self.gridLayout_24.setObjectName(u"gridLayout_24")
        self.gridLayout_24.setContentsMargins(2, 2, 2, 2)
        self.raster_type_step = QRadioButton(self.resample_group)
        self.raster_type_step.setObjectName(u"raster_type_step")
        self.raster_type_step.setChecked(True)

        self.gridLayout_24.addWidget(self.raster_type_step, 0, 0, 1, 1)

        self.raster = QDoubleSpinBox(self.resample_group)
        self.raster.setObjectName(u"raster")
        self.raster.setMinimumSize(QSize(0, 0))
        self.raster.setDecimals(6)
        self.raster.setMinimum(0.000001000000000)

        self.gridLayout_24.addWidget(self.raster, 0, 1, 1, 1)

        self.raster_type_channel = QRadioButton(self.resample_group)
        self.raster_type_channel.setObjectName(u"raster_type_channel")

        self.gridLayout_24.addWidget(self.raster_type_channel, 2, 0, 1, 1)

        self.raster_channel = QComboBox(self.resample_group)
        self.raster_channel.setObjectName(u"raster_channel")
        self.raster_channel.setEnabled(False)
        self.raster_channel.setInsertPolicy(QComboBox.InsertAtBottom)
        self.raster_channel.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)

        self.gridLayout_24.addWidget(self.raster_channel, 2, 1, 1, 1)

        self.raster_search_btn = QPushButton(self.resample_group)
        self.raster_search_btn.setObjectName(u"raster_search_btn")
        self.raster_search_btn.setIcon(icon4)

        self.gridLayout_24.addWidget(self.raster_search_btn, 2, 2, 1, 1)

        self.resample_time_from_zero = QCheckBox(self.resample_group)
        self.resample_time_from_zero.setObjectName(u"resample_time_from_zero")

        self.gridLayout_24.addWidget(self.resample_time_from_zero, 3, 0, 1, 3)

        self.gridLayout_24.setColumnStretch(1, 1)

        self.verticalLayout_15.addWidget(self.resample_group)

        self.groupBox_11 = QGroupBox(self.scrollAreaWidgetContents_3)
        self.groupBox_11.setObjectName(u"groupBox_11")
        self.verticalLayout_21 = QVBoxLayout(self.groupBox_11)
        self.verticalLayout_21.setSpacing(2)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.verticalLayout_21.setContentsMargins(2, 2, 2, 2)
        self.output_format = QComboBox(self.groupBox_11)
        self.output_format.addItem("")
        self.output_format.addItem("")
        self.output_format.addItem("")
        self.output_format.addItem("")
        self.output_format.addItem("")
        self.output_format.addItem("")
        self.output_format.setObjectName(u"output_format")

        self.verticalLayout_21.addWidget(self.output_format)

        self.output_options = QStackedWidget(self.groupBox_11)
        self.output_options.setObjectName(u"output_options")
        self.MDF_2 = QWidget()
        self.MDF_2.setObjectName(u"MDF_2")
        self.gridLayout_25 = QGridLayout(self.MDF_2)
        self.gridLayout_25.setObjectName(u"gridLayout_25")
        self.mdf_split = QCheckBox(self.MDF_2)
        self.mdf_split.setObjectName(u"mdf_split")
        self.mdf_split.setChecked(True)

        self.gridLayout_25.addWidget(self.mdf_split, 3, 0, 1, 2)

        self.mdf_compression = QComboBox(self.MDF_2)
        self.mdf_compression.setObjectName(u"mdf_compression")

        self.gridLayout_25.addWidget(self.mdf_compression, 2, 1, 1, 1)

        self.label_39 = QLabel(self.MDF_2)
        self.label_39.setObjectName(u"label_39")

        self.gridLayout_25.addWidget(self.label_39, 2, 0, 1, 1)

        self.label_40 = QLabel(self.MDF_2)
        self.label_40.setObjectName(u"label_40")

        self.gridLayout_25.addWidget(self.label_40, 0, 0, 1, 1)

        self.line_17 = QFrame(self.MDF_2)
        self.line_17.setObjectName(u"line_17")
        self.line_17.setFrameShape(QFrame.HLine)
        self.line_17.setFrameShadow(QFrame.Sunken)

        self.gridLayout_25.addWidget(self.line_17, 1, 0, 1, 2)

        self.label_38 = QLabel(self.MDF_2)
        self.label_38.setObjectName(u"label_38")

        self.gridLayout_25.addWidget(self.label_38, 4, 0, 1, 1)

        self.mdf_version = QComboBox(self.MDF_2)
        self.mdf_version.setObjectName(u"mdf_version")
        self.mdf_version.setMinimumSize(QSize(0, 0))

        self.gridLayout_25.addWidget(self.mdf_version, 0, 1, 1, 1)

        self.groupBox_12 = QGroupBox(self.MDF_2)
        self.groupBox_12.setObjectName(u"groupBox_12")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_12.sizePolicy().hasHeightForWidth())
        self.groupBox_12.setSizePolicy(sizePolicy)
        self.gridLayout_26 = QGridLayout(self.groupBox_12)
        self.gridLayout_26.setObjectName(u"gridLayout_26")
        self.scramble_btn = QPushButton(self.groupBox_12)
        self.scramble_btn.setObjectName(u"scramble_btn")
        icon5 = QIcon()
        icon5.addFile(u":/scramble.png", QSize(), QIcon.Normal, QIcon.Off)
        self.scramble_btn.setIcon(icon5)

        self.gridLayout_26.addWidget(self.scramble_btn, 1, 0, 1, 1)

        self.horizontalSpacer_24 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_26.addItem(self.horizontalSpacer_24, 1, 1, 1, 1)

        self.label_64 = QLabel(self.groupBox_12)
        self.label_64.setObjectName(u"label_64")
        sizePolicy.setHeightForWidth(self.label_64.sizePolicy().hasHeightForWidth())
        self.label_64.setSizePolicy(sizePolicy)
        self.label_64.setWordWrap(True)

        self.gridLayout_26.addWidget(self.label_64, 0, 0, 1, 2)

        self.gridLayout_26.setRowStretch(0, 1)
        self.gridLayout_26.setColumnStretch(1, 1)

        self.gridLayout_25.addWidget(self.groupBox_12, 5, 0, 1, 2)

        self.mdf_split_size = QDoubleSpinBox(self.MDF_2)
        self.mdf_split_size.setObjectName(u"mdf_split_size")
        self.mdf_split_size.setMaximum(4.000000000000000)

        self.gridLayout_25.addWidget(self.mdf_split_size, 4, 1, 1, 1)

        self.output_options.addWidget(self.MDF_2)
        self.HDF5_2 = QWidget()
        self.HDF5_2.setObjectName(u"HDF5_2")
        self.gridLayout_17 = QGridLayout(self.HDF5_2)
        self.gridLayout_17.setSpacing(2)
        self.gridLayout_17.setObjectName(u"gridLayout_17")
        self.gridLayout_17.setContentsMargins(2, 2, 2, 2)
        self.empty_channels = QComboBox(self.HDF5_2)
        self.empty_channels.setObjectName(u"empty_channels")

        self.gridLayout_17.addWidget(self.empty_channels, 8, 1, 1, 1)

        self.verticalSpacer_12 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_17.addItem(self.verticalSpacer_12, 9, 0, 1, 1)

        self.label_71 = QLabel(self.HDF5_2)
        self.label_71.setObjectName(u"label_71")

        self.gridLayout_17.addWidget(self.label_71, 7, 0, 1, 1)

        self.label_66 = QLabel(self.HDF5_2)
        self.label_66.setObjectName(u"label_66")

        self.gridLayout_17.addWidget(self.label_66, 8, 0, 1, 1)

        self.export_compression = QComboBox(self.HDF5_2)
        self.export_compression.setObjectName(u"export_compression")

        self.gridLayout_17.addWidget(self.export_compression, 7, 1, 1, 1)

        self.line_32 = QFrame(self.HDF5_2)
        self.line_32.setObjectName(u"line_32")
        self.line_32.setFrameShape(QFrame.HLine)
        self.line_32.setFrameShadow(QFrame.Sunken)

        self.gridLayout_17.addWidget(self.line_32, 4, 0, 1, 2)

        self.raw = QCheckBox(self.HDF5_2)
        self.raw.setObjectName(u"raw")

        self.gridLayout_17.addWidget(self.raw, 3, 0, 1, 2)

        self.time_as_date = QCheckBox(self.HDF5_2)
        self.time_as_date.setObjectName(u"time_as_date")

        self.gridLayout_17.addWidget(self.time_as_date, 2, 0, 1, 2)

        self.time_from_zero = QCheckBox(self.HDF5_2)
        self.time_from_zero.setObjectName(u"time_from_zero")

        self.gridLayout_17.addWidget(self.time_from_zero, 1, 0, 1, 2)

        self.single_time_base = QCheckBox(self.HDF5_2)
        self.single_time_base.setObjectName(u"single_time_base")

        self.gridLayout_17.addWidget(self.single_time_base, 0, 0, 1, 2)

        self.use_display_names = QCheckBox(self.HDF5_2)
        self.use_display_names.setObjectName(u"use_display_names")

        self.gridLayout_17.addWidget(self.use_display_names, 5, 0, 1, 2)

        self.reduce_memory_usage = QCheckBox(self.HDF5_2)
        self.reduce_memory_usage.setObjectName(u"reduce_memory_usage")

        self.gridLayout_17.addWidget(self.reduce_memory_usage, 6, 0, 1, 2)

        self.output_options.addWidget(self.HDF5_2)
        self.MAT_2 = QWidget()
        self.MAT_2.setObjectName(u"MAT_2")
        self.gridLayout_18 = QGridLayout(self.MAT_2)
        self.gridLayout_18.setSpacing(2)
        self.gridLayout_18.setObjectName(u"gridLayout_18")
        self.gridLayout_18.setContentsMargins(2, 2, 2, 2)
        self.label_41 = QLabel(self.MAT_2)
        self.label_41.setObjectName(u"label_41")

        self.gridLayout_18.addWidget(self.label_41, 10, 0, 1, 1)

        self.label_72 = QLabel(self.MAT_2)
        self.label_72.setObjectName(u"label_72")

        self.gridLayout_18.addWidget(self.label_72, 7, 0, 1, 1)

        self.label_73 = QLabel(self.MAT_2)
        self.label_73.setObjectName(u"label_73")

        self.gridLayout_18.addWidget(self.label_73, 9, 0, 1, 1)

        self.export_compression_mat = QComboBox(self.MAT_2)
        self.export_compression_mat.setObjectName(u"export_compression_mat")

        self.gridLayout_18.addWidget(self.export_compression_mat, 7, 1, 1, 1)

        self.verticalSpacer_13 = QSpacerItem(20, 2, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_18.addItem(self.verticalSpacer_13, 11, 0, 1, 1)

        self.line_33 = QFrame(self.MAT_2)
        self.line_33.setObjectName(u"line_33")
        self.line_33.setFrameShape(QFrame.HLine)
        self.line_33.setFrameShadow(QFrame.Sunken)

        self.gridLayout_18.addWidget(self.line_33, 4, 0, 1, 2)

        self.empty_channels_mat = QComboBox(self.MAT_2)
        self.empty_channels_mat.setObjectName(u"empty_channels_mat")

        self.gridLayout_18.addWidget(self.empty_channels_mat, 8, 1, 1, 1)

        self.mat_format = QComboBox(self.MAT_2)
        self.mat_format.setObjectName(u"mat_format")

        self.gridLayout_18.addWidget(self.mat_format, 9, 1, 1, 1)

        self.oned_as = QComboBox(self.MAT_2)
        self.oned_as.setObjectName(u"oned_as")

        self.gridLayout_18.addWidget(self.oned_as, 10, 1, 1, 1)

        self.label_74 = QLabel(self.MAT_2)
        self.label_74.setObjectName(u"label_74")

        self.gridLayout_18.addWidget(self.label_74, 8, 0, 1, 1)

        self.raw_mat = QCheckBox(self.MAT_2)
        self.raw_mat.setObjectName(u"raw_mat")

        self.gridLayout_18.addWidget(self.raw_mat, 3, 0, 1, 2)

        self.use_display_names_mat = QCheckBox(self.MAT_2)
        self.use_display_names_mat.setObjectName(u"use_display_names_mat")

        self.gridLayout_18.addWidget(self.use_display_names_mat, 5, 0, 1, 2)

        self.reduce_memory_usage_mat = QCheckBox(self.MAT_2)
        self.reduce_memory_usage_mat.setObjectName(u"reduce_memory_usage_mat")

        self.gridLayout_18.addWidget(self.reduce_memory_usage_mat, 6, 0, 1, 2)

        self.time_as_date_mat = QCheckBox(self.MAT_2)
        self.time_as_date_mat.setObjectName(u"time_as_date_mat")

        self.gridLayout_18.addWidget(self.time_as_date_mat, 2, 0, 1, 2)

        self.time_from_zero_mat = QCheckBox(self.MAT_2)
        self.time_from_zero_mat.setObjectName(u"time_from_zero_mat")

        self.gridLayout_18.addWidget(self.time_from_zero_mat, 1, 0, 1, 2)

        self.single_time_base_mat = QCheckBox(self.MAT_2)
        self.single_time_base_mat.setObjectName(u"single_time_base_mat")

        self.gridLayout_18.addWidget(self.single_time_base_mat, 0, 0, 1, 2)

        self.output_options.addWidget(self.MAT_2)
        self.CSV = QWidget()
        self.CSV.setObjectName(u"CSV")
        self.gridLayout_2 = QGridLayout(self.CSV)
        self.gridLayout_2.setSpacing(2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(2, 2, 2, 2)
        self.escapechar = QLineEdit(self.CSV)
        self.escapechar.setObjectName(u"escapechar")
        self.escapechar.setMaxLength(1)

        self.gridLayout_2.addWidget(self.escapechar, 10, 1, 1, 1)

        self.doublequote = QCheckBox(self.CSV)
        self.doublequote.setObjectName(u"doublequote")
        self.doublequote.setChecked(True)

        self.gridLayout_2.addWidget(self.doublequote, 9, 0, 1, 1)

        self.quoting = QComboBox(self.CSV)
        self.quoting.addItem("")
        self.quoting.addItem("")
        self.quoting.addItem("")
        self.quoting.addItem("")
        self.quoting.setObjectName(u"quoting")

        self.gridLayout_2.addWidget(self.quoting, 13, 1, 1, 1)

        self.raw_csv = QCheckBox(self.CSV)
        self.raw_csv.setObjectName(u"raw_csv")

        self.gridLayout_2.addWidget(self.raw_csv, 3, 0, 1, 2)

        self.label_15 = QLabel(self.CSV)
        self.label_15.setObjectName(u"label_15")

        self.gridLayout_2.addWidget(self.label_15, 10, 0, 1, 1)

        self.empty_channels_csv = QComboBox(self.CSV)
        self.empty_channels_csv.setObjectName(u"empty_channels_csv")

        self.gridLayout_2.addWidget(self.empty_channels_csv, 7, 1, 1, 1)

        self.use_display_names_csv = QCheckBox(self.CSV)
        self.use_display_names_csv.setObjectName(u"use_display_names_csv")

        self.gridLayout_2.addWidget(self.use_display_names_csv, 5, 0, 1, 2)

        self.delimiter = QLineEdit(self.CSV)
        self.delimiter.setObjectName(u"delimiter")
        self.delimiter.setMaxLength(1)
        self.delimiter.setClearButtonEnabled(False)

        self.gridLayout_2.addWidget(self.delimiter, 8, 1, 1, 1)

        self.quotechar = QLineEdit(self.CSV)
        self.quotechar.setObjectName(u"quotechar")
        self.quotechar.setMaxLength(1)

        self.gridLayout_2.addWidget(self.quotechar, 12, 1, 1, 1)

        self.label_2 = QLabel(self.CSV)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_2.addWidget(self.label_2, 8, 0, 1, 1)

        self.line_34 = QFrame(self.CSV)
        self.line_34.setObjectName(u"line_34")
        self.line_34.setFrameShape(QFrame.HLine)
        self.line_34.setFrameShadow(QFrame.Sunken)

        self.gridLayout_2.addWidget(self.line_34, 4, 0, 1, 2)

        self.label_67 = QLabel(self.CSV)
        self.label_67.setObjectName(u"label_67")

        self.gridLayout_2.addWidget(self.label_67, 7, 0, 1, 1)

        self.label_6 = QLabel(self.CSV)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_2.addWidget(self.label_6, 13, 0, 1, 1)

        self.time_as_date_csv = QCheckBox(self.CSV)
        self.time_as_date_csv.setObjectName(u"time_as_date_csv")

        self.gridLayout_2.addWidget(self.time_as_date_csv, 2, 0, 1, 2)

        self.lineterminator = QLineEdit(self.CSV)
        self.lineterminator.setObjectName(u"lineterminator")

        self.gridLayout_2.addWidget(self.lineterminator, 11, 1, 1, 1)

        self.time_from_zero_csv = QCheckBox(self.CSV)
        self.time_from_zero_csv.setObjectName(u"time_from_zero_csv")

        self.gridLayout_2.addWidget(self.time_from_zero_csv, 1, 0, 1, 2)

        self.single_time_base_csv = QCheckBox(self.CSV)
        self.single_time_base_csv.setObjectName(u"single_time_base_csv")

        self.gridLayout_2.addWidget(self.single_time_base_csv, 0, 0, 1, 2)

        self.label_4 = QLabel(self.CSV)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_2.addWidget(self.label_4, 11, 0, 1, 1)

        self.label_5 = QLabel(self.CSV)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_2.addWidget(self.label_5, 12, 0, 1, 1)

        self.add_units = QCheckBox(self.CSV)
        self.add_units.setObjectName(u"add_units")

        self.gridLayout_2.addWidget(self.add_units, 6, 0, 1, 2)

        self.output_options.addWidget(self.CSV)
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.output_options.addWidget(self.page)

        self.verticalLayout_21.addWidget(self.output_options)


        self.verticalLayout_15.addWidget(self.groupBox_11)

        self.groupBox = QGroupBox(self.scrollAreaWidgetContents_3)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(2, 2, 2, 2)
        self.modify_output_folder = QLineEdit(self.groupBox)
        self.modify_output_folder.setObjectName(u"modify_output_folder")
        self.modify_output_folder.setClearButtonEnabled(True)

        self.gridLayout.addWidget(self.modify_output_folder, 0, 0, 1, 1)

        self.modify_output_folder_btn = QPushButton(self.groupBox)
        self.modify_output_folder_btn.setObjectName(u"modify_output_folder_btn")
        self.modify_output_folder_btn.setIcon(icon1)

        self.gridLayout.addWidget(self.modify_output_folder_btn, 0, 1, 1, 1)


        self.verticalLayout_15.addWidget(self.groupBox)

        self.verticalSpacer_14 = QSpacerItem(20, 2, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_15.addItem(self.verticalSpacer_14)

        self.apply_btn = QPushButton(self.scrollAreaWidgetContents_3)
        self.apply_btn.setObjectName(u"apply_btn")
        icon6 = QIcon()
        icon6.addFile(u":/checkmark.png", QSize(), QIcon.Normal, QIcon.Off)
        self.apply_btn.setIcon(icon6)

        self.verticalLayout_15.addWidget(self.apply_btn)

        self.verticalLayout_15.setStretch(4, 1)

        self.horizontalLayout_2.addLayout(self.verticalLayout_15)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)

        self.verticalLayout_5.addWidget(self.scrollArea_3)

        icon7 = QIcon()
        icon7.addFile(u":/convert.png", QSize(), QIcon.Normal, QIcon.Off)
        self.aspects.addTab(self.convert_tab, icon7, "")
        self.stack_tab = QWidget()
        self.stack_tab.setObjectName(u"stack_tab")
        self.gridLayout_8 = QGridLayout(self.stack_tab)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.stack_sync = QCheckBox(self.stack_tab)
        self.stack_sync.setObjectName(u"stack_sync")
        self.stack_sync.setChecked(True)

        self.gridLayout_8.addWidget(self.stack_sync, 0, 0, 1, 2)

        self.stack_add_samples_origin = QCheckBox(self.stack_tab)
        self.stack_add_samples_origin.setObjectName(u"stack_add_samples_origin")
        self.stack_add_samples_origin.setChecked(True)

        self.gridLayout_8.addWidget(self.stack_add_samples_origin, 1, 0, 1, 2)

        self.label_26 = QLabel(self.stack_tab)
        self.label_26.setObjectName(u"label_26")

        self.gridLayout_8.addWidget(self.label_26, 2, 0, 1, 1)

        self.stack_format = QComboBox(self.stack_tab)
        self.stack_format.setObjectName(u"stack_format")

        self.gridLayout_8.addWidget(self.stack_format, 2, 1, 1, 1)

        self.line_8 = QFrame(self.stack_tab)
        self.line_8.setObjectName(u"line_8")
        self.line_8.setFrameShape(QFrame.HLine)
        self.line_8.setFrameShadow(QFrame.Sunken)

        self.gridLayout_8.addWidget(self.line_8, 3, 0, 1, 2)

        self.label_25 = QLabel(self.stack_tab)
        self.label_25.setObjectName(u"label_25")

        self.gridLayout_8.addWidget(self.label_25, 4, 0, 1, 1)

        self.stack_compression = QComboBox(self.stack_tab)
        self.stack_compression.setObjectName(u"stack_compression")

        self.gridLayout_8.addWidget(self.stack_compression, 4, 1, 1, 1)

        self.stack_split = QCheckBox(self.stack_tab)
        self.stack_split.setObjectName(u"stack_split")
        self.stack_split.setChecked(True)

        self.gridLayout_8.addWidget(self.stack_split, 5, 0, 1, 1)

        self.label_23 = QLabel(self.stack_tab)
        self.label_23.setObjectName(u"label_23")

        self.gridLayout_8.addWidget(self.label_23, 6, 0, 1, 1)

        self.stack_split_size = QDoubleSpinBox(self.stack_tab)
        self.stack_split_size.setObjectName(u"stack_split_size")
        self.stack_split_size.setMaximum(4.000000000000000)

        self.gridLayout_8.addWidget(self.stack_split_size, 6, 1, 1, 1)

        self.line_7 = QFrame(self.stack_tab)
        self.line_7.setObjectName(u"line_7")
        self.line_7.setFrameShape(QFrame.HLine)
        self.line_7.setFrameShadow(QFrame.Sunken)

        self.gridLayout_8.addWidget(self.line_7, 7, 0, 1, 2)

        self.verticalSpacer_7 = QSpacerItem(20, 502, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_8.addItem(self.verticalSpacer_7, 8, 1, 1, 1)

        self.stack_btn = QPushButton(self.stack_tab)
        self.stack_btn.setObjectName(u"stack_btn")
        icon8 = QIcon()
        icon8.addFile(u":/stack.png", QSize(), QIcon.Normal, QIcon.Off)
        self.stack_btn.setIcon(icon8)

        self.gridLayout_8.addWidget(self.stack_btn, 9, 1, 1, 1)

        self.gridLayout_8.setColumnStretch(1, 1)
        self.aspects.addTab(self.stack_tab, icon8, "")
        self.extract_bus_tab = QWidget()
        self.extract_bus_tab.setObjectName(u"extract_bus_tab")
        self.verticalLayout_4 = QVBoxLayout(self.extract_bus_tab)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.scrollArea = QScrollArea(self.extract_bus_tab)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 810, 674))
        self.gridLayout_3 = QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.groupBox_3 = QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_10 = QGridLayout(self.groupBox_3)
        self.gridLayout_10.setSpacing(2)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.gridLayout_10.setContentsMargins(2, 2, 2, 2)
        self.lineterminator_bus = QLineEdit(self.groupBox_3)
        self.lineterminator_bus.setObjectName(u"lineterminator_bus")

        self.gridLayout_10.addWidget(self.lineterminator_bus, 9, 2, 1, 1)

        self.line_13 = QFrame(self.groupBox_3)
        self.line_13.setObjectName(u"line_13")
        self.line_13.setFrameShape(QFrame.HLine)
        self.line_13.setFrameShadow(QFrame.Sunken)

        self.gridLayout_10.addWidget(self.line_13, 12, 1, 1, 3)

        self.empty_channels_bus = QComboBox(self.groupBox_3)
        self.empty_channels_bus.setObjectName(u"empty_channels_bus")

        self.gridLayout_10.addWidget(self.empty_channels_bus, 5, 2, 1, 2)

        self.delimiter_bus = QLineEdit(self.groupBox_3)
        self.delimiter_bus.setObjectName(u"delimiter_bus")
        self.delimiter_bus.setMaxLength(1)
        self.delimiter_bus.setClearButtonEnabled(False)

        self.gridLayout_10.addWidget(self.delimiter_bus, 6, 2, 1, 1)

        self.time_from_zero_bus = QCheckBox(self.groupBox_3)
        self.time_from_zero_bus.setObjectName(u"time_from_zero_bus")

        self.gridLayout_10.addWidget(self.time_from_zero_bus, 1, 1, 1, 2)

        self.escapechar_bus = QLineEdit(self.groupBox_3)
        self.escapechar_bus.setObjectName(u"escapechar_bus")
        self.escapechar_bus.setMaxLength(1)

        self.gridLayout_10.addWidget(self.escapechar_bus, 8, 2, 1, 1)

        self.label_7 = QLabel(self.groupBox_3)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_10.addWidget(self.label_7, 11, 1, 1, 1)

        self.bus_time_as_date = QCheckBox(self.groupBox_3)
        self.bus_time_as_date.setObjectName(u"bus_time_as_date")

        self.gridLayout_10.addWidget(self.bus_time_as_date, 2, 1, 1, 2)

        self.label_9 = QLabel(self.groupBox_3)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_10.addWidget(self.label_9, 8, 1, 1, 1)

        self.label_29 = QLabel(self.groupBox_3)
        self.label_29.setObjectName(u"label_29")

        self.gridLayout_10.addWidget(self.label_29, 5, 1, 1, 1)

        self.quoting_bus = QComboBox(self.groupBox_3)
        self.quoting_bus.addItem("")
        self.quoting_bus.addItem("")
        self.quoting_bus.addItem("")
        self.quoting_bus.addItem("")
        self.quoting_bus.setObjectName(u"quoting_bus")

        self.gridLayout_10.addWidget(self.quoting_bus, 11, 2, 1, 1)

        self.label_14 = QLabel(self.groupBox_3)
        self.label_14.setObjectName(u"label_14")

        self.gridLayout_10.addWidget(self.label_14, 10, 1, 1, 1)

        self.export_raster_bus = QDoubleSpinBox(self.groupBox_3)
        self.export_raster_bus.setObjectName(u"export_raster_bus")
        self.export_raster_bus.setDecimals(6)

        self.gridLayout_10.addWidget(self.export_raster_bus, 4, 2, 1, 2)

        self.add_units_bus = QCheckBox(self.groupBox_3)
        self.add_units_bus.setObjectName(u"add_units_bus")

        self.gridLayout_10.addWidget(self.add_units_bus, 3, 1, 1, 2)

        self.label_28 = QLabel(self.groupBox_3)
        self.label_28.setObjectName(u"label_28")

        self.gridLayout_10.addWidget(self.label_28, 4, 1, 1, 1)

        self.extract_bus_csv_btn = QPushButton(self.groupBox_3)
        self.extract_bus_csv_btn.setObjectName(u"extract_bus_csv_btn")
        icon9 = QIcon()
        icon9.addFile(u":/csv.png", QSize(), QIcon.Normal, QIcon.Off)
        self.extract_bus_csv_btn.setIcon(icon9)

        self.gridLayout_10.addWidget(self.extract_bus_csv_btn, 13, 1, 1, 3)

        self.single_time_base_bus = QCheckBox(self.groupBox_3)
        self.single_time_base_bus.setObjectName(u"single_time_base_bus")

        self.gridLayout_10.addWidget(self.single_time_base_bus, 0, 1, 1, 2)

        self.quotechar_bus = QLineEdit(self.groupBox_3)
        self.quotechar_bus.setObjectName(u"quotechar_bus")
        self.quotechar_bus.setMaxLength(1)

        self.gridLayout_10.addWidget(self.quotechar_bus, 10, 2, 1, 1)

        self.label_13 = QLabel(self.groupBox_3)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout_10.addWidget(self.label_13, 9, 1, 1, 1)

        self.doublequote_bus = QCheckBox(self.groupBox_3)
        self.doublequote_bus.setObjectName(u"doublequote_bus")
        self.doublequote_bus.setChecked(True)

        self.gridLayout_10.addWidget(self.doublequote_bus, 7, 1, 1, 2)

        self.label_8 = QLabel(self.groupBox_3)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_10.addWidget(self.label_8, 6, 1, 1, 1)


        self.gridLayout_3.addWidget(self.groupBox_3, 2, 1, 1, 1)

        self.output_info_bus = QTextEdit(self.scrollAreaWidgetContents)
        self.output_info_bus.setObjectName(u"output_info_bus")
        self.output_info_bus.setReadOnly(True)

        self.gridLayout_3.addWidget(self.output_info_bus, 0, 2, 3, 1)

        self.tabWidget = QTabWidget(self.scrollAreaWidgetContents)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayout = QVBoxLayout(self.tab)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.load_can_database_btn = QPushButton(self.tab)
        self.load_can_database_btn.setObjectName(u"load_can_database_btn")
        self.load_can_database_btn.setIcon(icon1)

        self.verticalLayout.addWidget(self.load_can_database_btn)

        self.can_database_list = MinimalListWidget(self.tab)
        self.can_database_list.setObjectName(u"can_database_list")

        self.verticalLayout.addWidget(self.can_database_list)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_2 = QVBoxLayout(self.tab_2)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(2, 2, 2, 2)
        self.load_lin_database_btn = QPushButton(self.tab_2)
        self.load_lin_database_btn.setObjectName(u"load_lin_database_btn")
        self.load_lin_database_btn.setIcon(icon1)

        self.verticalLayout_2.addWidget(self.load_lin_database_btn)

        self.lin_database_list = MinimalListWidget(self.tab_2)
        self.lin_database_list.setObjectName(u"lin_database_list")

        self.verticalLayout_2.addWidget(self.lin_database_list)

        self.tabWidget.addTab(self.tab_2, "")

        self.gridLayout_3.addWidget(self.tabWidget, 0, 0, 1, 2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.scrollAreaWidgetContents)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.prefix = QLineEdit(self.scrollAreaWidgetContents)
        self.prefix.setObjectName(u"prefix")
        self.prefix.setMinimumSize(QSize(0, 0))

        self.horizontalLayout.addWidget(self.prefix)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.horizontalLayout.setStretch(2, 1)

        self.gridLayout_3.addLayout(self.horizontalLayout, 1, 0, 1, 2)

        self.groupBox_2 = QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_5 = QGridLayout(self.groupBox_2)
        self.gridLayout_5.setSpacing(2)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(2, 2, 2, 2)
        self.extract_bus_compression = QComboBox(self.groupBox_2)
        self.extract_bus_compression.setObjectName(u"extract_bus_compression")
        self.extract_bus_compression.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.gridLayout_5.addWidget(self.extract_bus_compression, 0, 1, 1, 1)

        self.label_24 = QLabel(self.groupBox_2)
        self.label_24.setObjectName(u"label_24")

        self.gridLayout_5.addWidget(self.label_24, 1, 0, 1, 1)

        self.label__1 = QLabel(self.groupBox_2)
        self.label__1.setObjectName(u"label__1")

        self.gridLayout_5.addWidget(self.label__1, 0, 0, 1, 1)

        self.line_12 = QFrame(self.groupBox_2)
        self.line_12.setObjectName(u"line_12")
        self.line_12.setFrameShape(QFrame.HLine)
        self.line_12.setFrameShadow(QFrame.Sunken)

        self.gridLayout_5.addWidget(self.line_12, 3, 0, 1, 2)

        self.extract_bus_btn = QPushButton(self.groupBox_2)
        self.extract_bus_btn.setObjectName(u"extract_bus_btn")
        icon10 = QIcon()
        icon10.addFile(u":/down.png", QSize(), QIcon.Normal, QIcon.Off)
        self.extract_bus_btn.setIcon(icon10)

        self.gridLayout_5.addWidget(self.extract_bus_btn, 4, 0, 1, 2)

        self.extract_bus_format = QComboBox(self.groupBox_2)
        self.extract_bus_format.setObjectName(u"extract_bus_format")
        self.extract_bus_format.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.gridLayout_5.addWidget(self.extract_bus_format, 1, 1, 1, 1)

        self.label_27 = QLabel(self.groupBox_2)
        self.label_27.setObjectName(u"label_27")

        self.gridLayout_5.addWidget(self.label_27, 2, 0, 1, 1)


        self.gridLayout_3.addWidget(self.groupBox_2, 2, 0, 1, 1)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_4.addWidget(self.scrollArea)

        self.aspects.addTab(self.extract_bus_tab, icon10, "")
        self.splitter.addWidget(self.aspects)

        self.gridLayout_9.addWidget(self.splitter, 1, 0, 1, 1)

        self.list_layout = QVBoxLayout()
        self.list_layout.setObjectName(u"list_layout")
        self.files_list = MinimalListWidget(batch_widget)
        self.files_list.setObjectName(u"files_list")

        self.list_layout.addWidget(self.files_list)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.sort_by_start_time_btn = QPushButton(batch_widget)
        self.sort_by_start_time_btn.setObjectName(u"sort_by_start_time_btn")
        icon11 = QIcon()
        icon11.addFile(u":/clock.png", QSize(), QIcon.Normal, QIcon.Off)
        self.sort_by_start_time_btn.setIcon(icon11)

        self.verticalLayout_3.addWidget(self.sort_by_start_time_btn)

        self.sort_alphabetically_btn = QPushButton(batch_widget)
        self.sort_alphabetically_btn.setObjectName(u"sort_alphabetically_btn")
        icon12 = QIcon()
        icon12.addFile(u":/alphabetical_sorting.png", QSize(), QIcon.Normal, QIcon.Off)
        self.sort_alphabetically_btn.setIcon(icon12)

        self.verticalLayout_3.addWidget(self.sort_alphabetically_btn)


        self.list_layout.addLayout(self.verticalLayout_3)


        self.gridLayout_9.addLayout(self.list_layout, 1, 1, 1, 1)

        self.gridLayout_9.setColumnStretch(0, 2)
        self.gridLayout_9.setColumnStretch(1, 1)

        self.retranslateUi(batch_widget)

        self.aspects.setCurrentIndex(0)
        self.output_options.setCurrentIndex(0)
        self.quoting.setCurrentIndex(1)
        self.quoting_bus.setCurrentIndex(1)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(batch_widget)
    # setupUi

    def retranslateUi(self, batch_widget):
        batch_widget.setWindowTitle(QCoreApplication.translate("batch_widget", u"Form", None))
        self.concatenate_sync.setText(QCoreApplication.translate("batch_widget", u"Sync using measurements timestamps", None))
        self.concatenate_add_samples_origin.setText(QCoreApplication.translate("batch_widget", u"Add samples origin file", None))
        self.label_11.setText(QCoreApplication.translate("batch_widget", u"Output format", None))
        self.label_12.setText(QCoreApplication.translate("batch_widget", u"Compression", None))
        self.concatenate_split.setText(QCoreApplication.translate("batch_widget", u"Split data blocks", None))
        self.concatenate_split_size.setSuffix(QCoreApplication.translate("batch_widget", u"MB", None))
        self.label_10.setText(QCoreApplication.translate("batch_widget", u"Split size ", None))
        self.concatenate_btn.setText(QCoreApplication.translate("batch_widget", u"Concatenate", None))
        self.aspects.setTabText(self.aspects.indexOf(self.concatenate_tab), QCoreApplication.translate("batch_widget", u"Concatenate", None))
        self.filter_view.setItemText(0, QCoreApplication.translate("batch_widget", u"Natural sort", None))
        self.filter_view.setItemText(1, QCoreApplication.translate("batch_widget", u"Internal file structure", None))
        self.filter_view.setItemText(2, QCoreApplication.translate("batch_widget", u"Selected channels only", None))

        ___qtreewidgetitem = self.filter_tree.headerItem()
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("batch_widget", u"Channels", None));
#if QT_CONFIG(tooltip)
        self.filter_tree.setToolTip(QCoreApplication.translate("batch_widget", u"Double click channel to see extended information", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.load_filter_list_btn.setToolTip(QCoreApplication.translate("batch_widget", u"Load channel selection list", None))
#endif // QT_CONFIG(tooltip)
        self.load_filter_list_btn.setText("")
#if QT_CONFIG(tooltip)
        self.save_filter_list_btn.setToolTip(QCoreApplication.translate("batch_widget", u"Save channel selection list", None))
#endif // QT_CONFIG(tooltip)
        self.save_filter_list_btn.setText("")
#if QT_CONFIG(tooltip)
        self.clear_filter_btn.setToolTip(QCoreApplication.translate("batch_widget", u"Clear selection", None))
#endif // QT_CONFIG(tooltip)
        self.clear_filter_btn.setText("")
#if QT_CONFIG(tooltip)
        self.advanced_serch_filter_btn.setToolTip(QCoreApplication.translate("batch_widget", u"Search and select channels", None))
#endif // QT_CONFIG(tooltip)
        self.advanced_serch_filter_btn.setText("")
        self.label_3.setText(QCoreApplication.translate("batch_widget", u"All selected channels", None))
        self.cut_group.setTitle(QCoreApplication.translate("batch_widget", u"Cut", None))
        self.label_62.setText(QCoreApplication.translate("batch_widget", u"Start", None))
        self.cut_stop.setSuffix(QCoreApplication.translate("batch_widget", u"s", None))
        self.label_63.setText(QCoreApplication.translate("batch_widget", u"End", None))
        self.cut_start.setSuffix(QCoreApplication.translate("batch_widget", u"s", None))
        self.whence.setText(QCoreApplication.translate("batch_widget", u"Start relative to first time stamp", None))
        self.cut_time_from_zero.setText(QCoreApplication.translate("batch_widget", u"Time from 0s", None))
        self.resample_group.setTitle(QCoreApplication.translate("batch_widget", u"Resample", None))
        self.raster_type_step.setText(QCoreApplication.translate("batch_widget", u"step", None))
        self.raster.setSuffix(QCoreApplication.translate("batch_widget", u"s", None))
        self.raster_type_channel.setText(QCoreApplication.translate("batch_widget", u"channel", None))
#if QT_CONFIG(tooltip)
        self.raster_search_btn.setToolTip(QCoreApplication.translate("batch_widget", u"Search raster channel", None))
#endif // QT_CONFIG(tooltip)
        self.raster_search_btn.setText("")
        self.resample_time_from_zero.setText(QCoreApplication.translate("batch_widget", u"Time from 0s", None))
        self.groupBox_11.setTitle(QCoreApplication.translate("batch_widget", u"Ouput format", None))
        self.output_format.setItemText(0, QCoreApplication.translate("batch_widget", u"MDF", None))
        self.output_format.setItemText(1, QCoreApplication.translate("batch_widget", u"ASC", None))
        self.output_format.setItemText(2, QCoreApplication.translate("batch_widget", u"CSV", None))
        self.output_format.setItemText(3, QCoreApplication.translate("batch_widget", u"HDF5", None))
        self.output_format.setItemText(4, QCoreApplication.translate("batch_widget", u"MAT", None))
        self.output_format.setItemText(5, QCoreApplication.translate("batch_widget", u"Parquet", None))

        self.mdf_split.setText(QCoreApplication.translate("batch_widget", u"Split data blocks", None))
        self.label_39.setText(QCoreApplication.translate("batch_widget", u"Compression", None))
        self.label_40.setText(QCoreApplication.translate("batch_widget", u"Version", None))
        self.label_38.setText(QCoreApplication.translate("batch_widget", u"Split size ", None))
        self.groupBox_12.setTitle(QCoreApplication.translate("batch_widget", u"Scramble", None))
        self.scramble_btn.setText(QCoreApplication.translate("batch_widget", u"Scramble texts", None))
        self.label_64.setText(QCoreApplication.translate("batch_widget", u"Anonymize the measurements: scramble all texts and replace them with random strings", None))
        self.mdf_split_size.setSuffix(QCoreApplication.translate("batch_widget", u"MB", None))
        self.label_71.setText(QCoreApplication.translate("batch_widget", u"Compression", None))
        self.label_66.setText(QCoreApplication.translate("batch_widget", u"Empty channels", None))
        self.raw.setText(QCoreApplication.translate("batch_widget", u"Raw values", None))
        self.time_as_date.setText(QCoreApplication.translate("batch_widget", u"Time as date", None))
        self.time_from_zero.setText(QCoreApplication.translate("batch_widget", u"Time from 0s", None))
        self.single_time_base.setText(QCoreApplication.translate("batch_widget", u"Single time base", None))
        self.use_display_names.setText(QCoreApplication.translate("batch_widget", u"Use display names", None))
        self.reduce_memory_usage.setText(QCoreApplication.translate("batch_widget", u"Reduce  memory usage", None))
        self.label_41.setText(QCoreApplication.translate("batch_widget", u".mat oned_as", None))
        self.label_72.setText(QCoreApplication.translate("batch_widget", u"Compression", None))
        self.label_73.setText(QCoreApplication.translate("batch_widget", u".mat format", None))
        self.label_74.setText(QCoreApplication.translate("batch_widget", u"Empty channels", None))
        self.raw_mat.setText(QCoreApplication.translate("batch_widget", u"Raw values", None))
        self.use_display_names_mat.setText(QCoreApplication.translate("batch_widget", u"Use display names", None))
        self.reduce_memory_usage_mat.setText(QCoreApplication.translate("batch_widget", u"Reduce  memory usage", None))
        self.time_as_date_mat.setText(QCoreApplication.translate("batch_widget", u"Time as date", None))
        self.time_from_zero_mat.setText(QCoreApplication.translate("batch_widget", u"Time from 0s", None))
        self.single_time_base_mat.setText(QCoreApplication.translate("batch_widget", u"Single time base", None))
        self.escapechar.setInputMask("")
        self.escapechar.setPlaceholderText(QCoreApplication.translate("batch_widget", u"None", None))
        self.doublequote.setText(QCoreApplication.translate("batch_widget", u"Double quote", None))
        self.quoting.setItemText(0, QCoreApplication.translate("batch_widget", u"ALL", None))
        self.quoting.setItemText(1, QCoreApplication.translate("batch_widget", u"MINIMAL", None))
        self.quoting.setItemText(2, QCoreApplication.translate("batch_widget", u"NONNUMERIC", None))
        self.quoting.setItemText(3, QCoreApplication.translate("batch_widget", u"NONE", None))

        self.raw_csv.setText(QCoreApplication.translate("batch_widget", u"Raw values", None))
        self.label_15.setText(QCoreApplication.translate("batch_widget", u"Escape Char", None))
        self.use_display_names_csv.setText(QCoreApplication.translate("batch_widget", u"Use display names", None))
        self.delimiter.setText(QCoreApplication.translate("batch_widget", u",", None))
        self.quotechar.setText(QCoreApplication.translate("batch_widget", u"\"", None))
        self.label_2.setText(QCoreApplication.translate("batch_widget", u"Delimiter", None))
        self.label_67.setText(QCoreApplication.translate("batch_widget", u"Empty channels", None))
        self.label_6.setText(QCoreApplication.translate("batch_widget", u"Quoting", None))
        self.time_as_date_csv.setText(QCoreApplication.translate("batch_widget", u"Time as date", None))
        self.lineterminator.setText(QCoreApplication.translate("batch_widget", u"\\r\\n", None))
        self.time_from_zero_csv.setText(QCoreApplication.translate("batch_widget", u"Time from 0s", None))
        self.single_time_base_csv.setText(QCoreApplication.translate("batch_widget", u"Single time base", None))
        self.label_4.setText(QCoreApplication.translate("batch_widget", u"Line Terminator", None))
        self.label_5.setText(QCoreApplication.translate("batch_widget", u"Quote Char", None))
        self.add_units.setText(QCoreApplication.translate("batch_widget", u"Add units", None))
        self.groupBox.setTitle(QCoreApplication.translate("batch_widget", u"Output folder", None))
        self.modify_output_folder.setPlaceholderText(QCoreApplication.translate("batch_widget", u"please select an output folder", None))
        self.modify_output_folder_btn.setText("")
        self.apply_btn.setText(QCoreApplication.translate("batch_widget", u"Apply", None))
        self.aspects.setTabText(self.aspects.indexOf(self.convert_tab), QCoreApplication.translate("batch_widget", u"Modify && Export", None))
#if QT_CONFIG(tooltip)
        self.aspects.setTabToolTip(self.aspects.indexOf(self.convert_tab), QCoreApplication.translate("batch_widget", u"conv", None))
#endif // QT_CONFIG(tooltip)
        self.stack_sync.setText(QCoreApplication.translate("batch_widget", u"Sync using measurements timestamps", None))
        self.stack_add_samples_origin.setText(QCoreApplication.translate("batch_widget", u"Add samples origin file", None))
        self.label_26.setText(QCoreApplication.translate("batch_widget", u"Output format", None))
        self.label_25.setText(QCoreApplication.translate("batch_widget", u"Compression", None))
        self.stack_split.setText(QCoreApplication.translate("batch_widget", u"Split data blocks", None))
        self.label_23.setText(QCoreApplication.translate("batch_widget", u"Split size ", None))
        self.stack_split_size.setSuffix(QCoreApplication.translate("batch_widget", u"MB", None))
        self.stack_btn.setText(QCoreApplication.translate("batch_widget", u"Stack", None))
        self.aspects.setTabText(self.aspects.indexOf(self.stack_tab), QCoreApplication.translate("batch_widget", u"Stack", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("batch_widget", u"CSV", None))
        self.lineterminator_bus.setText(QCoreApplication.translate("batch_widget", u"\\r\\n", None))
        self.delimiter_bus.setText(QCoreApplication.translate("batch_widget", u",", None))
        self.time_from_zero_bus.setText(QCoreApplication.translate("batch_widget", u"Time from 0s", None))
        self.escapechar_bus.setInputMask("")
        self.escapechar_bus.setPlaceholderText(QCoreApplication.translate("batch_widget", u"None", None))
        self.label_7.setText(QCoreApplication.translate("batch_widget", u"Quoting", None))
        self.bus_time_as_date.setText(QCoreApplication.translate("batch_widget", u"Time as date", None))
        self.label_9.setText(QCoreApplication.translate("batch_widget", u"Escape Char", None))
        self.label_29.setText(QCoreApplication.translate("batch_widget", u"Empty channels", None))
        self.quoting_bus.setItemText(0, QCoreApplication.translate("batch_widget", u"ALL", None))
        self.quoting_bus.setItemText(1, QCoreApplication.translate("batch_widget", u"MINIMAL", None))
        self.quoting_bus.setItemText(2, QCoreApplication.translate("batch_widget", u"NONNUMERIC", None))
        self.quoting_bus.setItemText(3, QCoreApplication.translate("batch_widget", u"NONE", None))

        self.label_14.setText(QCoreApplication.translate("batch_widget", u"Quote Char", None))
        self.export_raster_bus.setSuffix(QCoreApplication.translate("batch_widget", u"s", None))
        self.add_units_bus.setText(QCoreApplication.translate("batch_widget", u"Add units", None))
        self.label_28.setText(QCoreApplication.translate("batch_widget", u"Raster", None))
        self.extract_bus_csv_btn.setText(QCoreApplication.translate("batch_widget", u"Export to CSV         ", None))
        self.single_time_base_bus.setText(QCoreApplication.translate("batch_widget", u"Single time base", None))
        self.quotechar_bus.setText(QCoreApplication.translate("batch_widget", u"\"", None))
        self.label_13.setText(QCoreApplication.translate("batch_widget", u"Line Terminator", None))
        self.doublequote_bus.setText(QCoreApplication.translate("batch_widget", u"Double quote", None))
        self.label_8.setText(QCoreApplication.translate("batch_widget", u"Delimiter", None))
        self.load_can_database_btn.setText(QCoreApplication.translate("batch_widget", u"Load CAN database", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("batch_widget", u"CAN", None))
        self.load_lin_database_btn.setText(QCoreApplication.translate("batch_widget", u"Load LIN database", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("batch_widget", u"LIN", None))
        self.label.setText(QCoreApplication.translate("batch_widget", u"Prefix", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("batch_widget", u"MDF", None))
        self.label_24.setText(QCoreApplication.translate("batch_widget", u"Version", None))
        self.label__1.setText(QCoreApplication.translate("batch_widget", u"Compression", None))
        self.extract_bus_btn.setText(QCoreApplication.translate("batch_widget", u"Extract Bus signals", None))
        self.label_27.setText("")
        self.aspects.setTabText(self.aspects.indexOf(self.extract_bus_tab), QCoreApplication.translate("batch_widget", u"Bus logging", None))
        self.sort_by_start_time_btn.setText(QCoreApplication.translate("batch_widget", u"Sort by start time", None))
        self.sort_alphabetically_btn.setText(QCoreApplication.translate("batch_widget", u"Sort alphabetically", None))
    # retranslateUi


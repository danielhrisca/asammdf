# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'file_widget.ui'
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
    QStackedWidget, QTabWidget, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget)

from asammdf.gui.widgets.list import MinimalListWidget
from asammdf.gui.widgets.tree import TreeWidget
from . import resource_rc

class Ui_file_widget(object):
    def setupUi(self, file_widget):
        if not file_widget.objectName():
            file_widget.setObjectName(u"file_widget")
        file_widget.resize(1034, 622)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(file_widget.sizePolicy().hasHeightForWidth())
        file_widget.setSizePolicy(sizePolicy)
        file_widget.setMinimumSize(QSize(1, 1))
        self.verticalLayout = QVBoxLayout(file_widget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.aspects = QTabWidget(file_widget)
        self.aspects.setObjectName(u"aspects")
        self.aspects.setTabPosition(QTabWidget.West)
        self.aspects.setDocumentMode(False)
        self.channels_tab = QWidget()
        self.channels_tab.setObjectName(u"channels_tab")
        self.verticalLayout_5 = QVBoxLayout(self.channels_tab)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.splitter = QSplitter(self.channels_tab)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.verticalLayoutWidget = QWidget(self.splitter)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.channels_layout = QVBoxLayout(self.verticalLayoutWidget)
        self.channels_layout.setObjectName(u"channels_layout")
        self.channels_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_view = QComboBox(self.verticalLayoutWidget)
        self.channel_view.addItem("")
        self.channel_view.addItem("")
        self.channel_view.addItem("")
        self.channel_view.setObjectName(u"channel_view")

        self.channels_layout.addWidget(self.channel_view)

        self.channels_tree = TreeWidget(self.verticalLayoutWidget)
        self.channels_tree.setObjectName(u"channels_tree")

        self.channels_layout.addWidget(self.channels_tree)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setObjectName(u"buttons_layout")
        self.load_channel_list_btn = QPushButton(self.verticalLayoutWidget)
        self.load_channel_list_btn.setObjectName(u"load_channel_list_btn")
        icon = QIcon()
        icon.addFile(u":/open.png", QSize(), QIcon.Normal, QIcon.Off)
        self.load_channel_list_btn.setIcon(icon)

        self.buttons_layout.addWidget(self.load_channel_list_btn)

        self.button_spacer1 = QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.buttons_layout.addItem(self.button_spacer1)

        self.save_channel_list_btn = QPushButton(self.verticalLayoutWidget)
        self.save_channel_list_btn.setObjectName(u"save_channel_list_btn")
        icon1 = QIcon()
        icon1.addFile(u":/save.png", QSize(), QIcon.Normal, QIcon.Off)
        self.save_channel_list_btn.setIcon(icon1)

        self.buttons_layout.addWidget(self.save_channel_list_btn)

        self.select_all_btn = QPushButton(self.verticalLayoutWidget)
        self.select_all_btn.setObjectName(u"select_all_btn")
        icon2 = QIcon()
        icon2.addFile(u":/checkmark.png", QSize(), QIcon.Normal, QIcon.Off)
        self.select_all_btn.setIcon(icon2)

        self.buttons_layout.addWidget(self.select_all_btn)

        self.clear_channels_btn = QPushButton(self.verticalLayoutWidget)
        self.clear_channels_btn.setObjectName(u"clear_channels_btn")
        icon3 = QIcon()
        icon3.addFile(u":/erase.png", QSize(), QIcon.Normal, QIcon.Off)
        self.clear_channels_btn.setIcon(icon3)

        self.buttons_layout.addWidget(self.clear_channels_btn)

        self.advanced_search_btn = QPushButton(self.verticalLayoutWidget)
        self.advanced_search_btn.setObjectName(u"advanced_search_btn")
        icon4 = QIcon()
        icon4.addFile(u":/search.png", QSize(), QIcon.Normal, QIcon.Off)
        self.advanced_search_btn.setIcon(icon4)

        self.buttons_layout.addWidget(self.advanced_search_btn)

        self.create_window_btn = QPushButton(self.verticalLayoutWidget)
        self.create_window_btn.setObjectName(u"create_window_btn")
        icon5 = QIcon()
        icon5.addFile(u":/graph.png", QSize(), QIcon.Normal, QIcon.Off)
        self.create_window_btn.setIcon(icon5)

        self.buttons_layout.addWidget(self.create_window_btn)

        self.horizontalSpacer_6 = QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.buttons_layout.addItem(self.horizontalSpacer_6)

        self.load_embedded_channel_list_btn = QPushButton(self.verticalLayoutWidget)
        self.load_embedded_channel_list_btn.setObjectName(u"load_embedded_channel_list_btn")
        icon6 = QIcon()
        icon6.addFile(u":/load_embed.png", QSize(), QIcon.Normal, QIcon.Off)
        self.load_embedded_channel_list_btn.setIcon(icon6)

        self.buttons_layout.addWidget(self.load_embedded_channel_list_btn)

        self.save_embedded_channel_list_btn = QPushButton(self.verticalLayoutWidget)
        self.save_embedded_channel_list_btn.setObjectName(u"save_embedded_channel_list_btn")
        icon7 = QIcon()
        icon7.addFile(u":/attach.png", QSize(), QIcon.Normal, QIcon.Off)
        self.save_embedded_channel_list_btn.setIcon(icon7)

        self.buttons_layout.addWidget(self.save_embedded_channel_list_btn)

        self.button_spacer2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.buttons_layout.addItem(self.button_spacer2)


        self.channels_layout.addLayout(self.buttons_layout)

        self.splitter.addWidget(self.verticalLayoutWidget)

        self.verticalLayout_5.addWidget(self.splitter)

        self.aspects.addTab(self.channels_tab, icon5, "")
        self.modify = QWidget()
        self.modify.setObjectName(u"modify")
        self.horizontalLayout = QHBoxLayout(self.modify)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.filter_view = QComboBox(self.modify)
        self.filter_view.addItem("")
        self.filter_view.addItem("")
        self.filter_view.addItem("")
        self.filter_view.setObjectName(u"filter_view")

        self.verticalLayout_2.addWidget(self.filter_view)

        self.filter_tree = TreeWidget(self.modify)
        self.filter_tree.setObjectName(u"filter_tree")

        self.verticalLayout_2.addWidget(self.filter_tree)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.load_filter_list_btn = QPushButton(self.modify)
        self.load_filter_list_btn.setObjectName(u"load_filter_list_btn")
        self.load_filter_list_btn.setIcon(icon)

        self.horizontalLayout_4.addWidget(self.load_filter_list_btn)

        self.save_filter_list_btn = QPushButton(self.modify)
        self.save_filter_list_btn.setObjectName(u"save_filter_list_btn")
        self.save_filter_list_btn.setIcon(icon1)

        self.horizontalLayout_4.addWidget(self.save_filter_list_btn)

        self.clear_filter_btn = QPushButton(self.modify)
        self.clear_filter_btn.setObjectName(u"clear_filter_btn")
        self.clear_filter_btn.setIcon(icon3)

        self.horizontalLayout_4.addWidget(self.clear_filter_btn)

        self.advanced_serch_filter_btn = QPushButton(self.modify)
        self.advanced_serch_filter_btn.setObjectName(u"advanced_serch_filter_btn")
        self.advanced_serch_filter_btn.setIcon(icon4)

        self.horizontalLayout_4.addWidget(self.advanced_serch_filter_btn)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.label = QLabel(self.modify)
        self.label.setObjectName(u"label")

        self.verticalLayout_6.addWidget(self.label)

        self.selected_filter_channels = QListWidget(self.modify)
        self.selected_filter_channels.setObjectName(u"selected_filter_channels")
        self.selected_filter_channels.setViewMode(QListView.ListMode)
        self.selected_filter_channels.setUniformItemSizes(True)
        self.selected_filter_channels.setSortingEnabled(True)

        self.verticalLayout_6.addWidget(self.selected_filter_channels)


        self.horizontalLayout.addLayout(self.verticalLayout_6)

        self.scrollArea = QScrollArea(self.modify)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 306, 619))
        self.verticalLayout_4 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setSpacing(2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.cut_group = QGroupBox(self.scrollAreaWidgetContents)
        self.cut_group.setObjectName(u"cut_group")
        self.cut_group.setCheckable(True)
        self.cut_group.setChecked(False)
        self.gridLayout_19 = QGridLayout(self.cut_group)
        self.gridLayout_19.setObjectName(u"gridLayout_19")
        self.gridLayout_19.setHorizontalSpacing(2)
        self.gridLayout_19.setContentsMargins(2, 2, 2, 2)
        self.label_59 = QLabel(self.cut_group)
        self.label_59.setObjectName(u"label_59")

        self.gridLayout_19.addWidget(self.label_59, 0, 0, 1, 1)

        self.cut_stop = QDoubleSpinBox(self.cut_group)
        self.cut_stop.setObjectName(u"cut_stop")
        self.cut_stop.setDecimals(6)
        self.cut_stop.setMinimum(-9999999999999999635896294965248.000000000000000)
        self.cut_stop.setMaximum(999999999999999983222784.000000000000000)

        self.gridLayout_19.addWidget(self.cut_stop, 1, 1, 1, 1)

        self.label_60 = QLabel(self.cut_group)
        self.label_60.setObjectName(u"label_60")

        self.gridLayout_19.addWidget(self.label_60, 1, 0, 1, 1)

        self.cut_start = QDoubleSpinBox(self.cut_group)
        self.cut_start.setObjectName(u"cut_start")
        self.cut_start.setDecimals(6)
        self.cut_start.setMinimum(-9999999999999999635896294965248.000000000000000)
        self.cut_start.setMaximum(999999999999999983222784.000000000000000)

        self.gridLayout_19.addWidget(self.cut_start, 0, 1, 1, 1)

        self.whence = QCheckBox(self.cut_group)
        self.whence.setObjectName(u"whence")

        self.gridLayout_19.addWidget(self.whence, 3, 0, 1, 2)

        self.cut_time_from_zero = QCheckBox(self.cut_group)
        self.cut_time_from_zero.setObjectName(u"cut_time_from_zero")

        self.gridLayout_19.addWidget(self.cut_time_from_zero, 4, 0, 1, 2)


        self.verticalLayout_3.addWidget(self.cut_group)

        self.resample_group = QGroupBox(self.scrollAreaWidgetContents)
        self.resample_group.setObjectName(u"resample_group")
        self.resample_group.setCheckable(True)
        self.resample_group.setChecked(False)
        self.gridLayout_21 = QGridLayout(self.resample_group)
        self.gridLayout_21.setObjectName(u"gridLayout_21")
        self.gridLayout_21.setHorizontalSpacing(2)
        self.gridLayout_21.setContentsMargins(2, 2, 2, 2)
        self.raster_type_step = QRadioButton(self.resample_group)
        self.raster_type_step.setObjectName(u"raster_type_step")
        self.raster_type_step.setChecked(True)

        self.gridLayout_21.addWidget(self.raster_type_step, 0, 0, 1, 1)

        self.raster = QDoubleSpinBox(self.resample_group)
        self.raster.setObjectName(u"raster")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.raster.sizePolicy().hasHeightForWidth())
        self.raster.setSizePolicy(sizePolicy1)
        self.raster.setMinimumSize(QSize(0, 0))
        self.raster.setDecimals(6)
        self.raster.setMinimum(0.000001000000000)

        self.gridLayout_21.addWidget(self.raster, 0, 1, 1, 1)

        self.raster_type_channel = QRadioButton(self.resample_group)
        self.raster_type_channel.setObjectName(u"raster_type_channel")

        self.gridLayout_21.addWidget(self.raster_type_channel, 2, 0, 1, 1)

        self.raster_channel = QComboBox(self.resample_group)
        self.raster_channel.setObjectName(u"raster_channel")
        self.raster_channel.setEnabled(False)
        self.raster_channel.setInsertPolicy(QComboBox.InsertAtBottom)
        self.raster_channel.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)

        self.gridLayout_21.addWidget(self.raster_channel, 2, 1, 1, 1)

        self.raster_search_btn = QPushButton(self.resample_group)
        self.raster_search_btn.setObjectName(u"raster_search_btn")
        self.raster_search_btn.setIcon(icon4)

        self.gridLayout_21.addWidget(self.raster_search_btn, 2, 2, 1, 1)

        self.resample_time_from_zero = QCheckBox(self.resample_group)
        self.resample_time_from_zero.setObjectName(u"resample_time_from_zero")

        self.gridLayout_21.addWidget(self.resample_time_from_zero, 3, 0, 1, 3)

        self.gridLayout_21.setColumnStretch(1, 1)

        self.verticalLayout_3.addWidget(self.resample_group)

        self.groupBox_10 = QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_10.setObjectName(u"groupBox_10")
        self.verticalLayout_20 = QVBoxLayout(self.groupBox_10)
        self.verticalLayout_20.setSpacing(2)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.verticalLayout_20.setContentsMargins(2, 2, 2, 2)
        self.output_format = QComboBox(self.groupBox_10)
        self.output_format.addItem("")
        self.output_format.addItem("")
        self.output_format.addItem("")
        self.output_format.addItem("")
        self.output_format.addItem("")
        self.output_format.addItem("")
        self.output_format.setObjectName(u"output_format")

        self.verticalLayout_20.addWidget(self.output_format)

        self.output_options = QStackedWidget(self.groupBox_10)
        self.output_options.setObjectName(u"output_options")
        self.MDF = QWidget()
        self.MDF.setObjectName(u"MDF")
        self.gridLayout_22 = QGridLayout(self.MDF)
        self.gridLayout_22.setObjectName(u"gridLayout_22")
        self.mdf_split_size = QDoubleSpinBox(self.MDF)
        self.mdf_split_size.setObjectName(u"mdf_split_size")
        self.mdf_split_size.setMaximum(4.000000000000000)

        self.gridLayout_22.addWidget(self.mdf_split_size, 4, 1, 1, 1)

        self.label_29 = QLabel(self.MDF)
        self.label_29.setObjectName(u"label_29")

        self.gridLayout_22.addWidget(self.label_29, 2, 0, 1, 1)

        self.line_14 = QFrame(self.MDF)
        self.line_14.setObjectName(u"line_14")
        self.line_14.setFrameShape(QFrame.HLine)
        self.line_14.setFrameShadow(QFrame.Sunken)

        self.gridLayout_22.addWidget(self.line_14, 1, 0, 1, 2)

        self.mdf_compression = QComboBox(self.MDF)
        self.mdf_compression.setObjectName(u"mdf_compression")

        self.gridLayout_22.addWidget(self.mdf_compression, 2, 1, 1, 1)

        self.label_27 = QLabel(self.MDF)
        self.label_27.setObjectName(u"label_27")

        self.gridLayout_22.addWidget(self.label_27, 0, 0, 1, 1)

        self.groupBox_9 = QGroupBox(self.MDF)
        self.groupBox_9.setObjectName(u"groupBox_9")
        self.gridLayout_20 = QGridLayout(self.groupBox_9)
        self.gridLayout_20.setObjectName(u"gridLayout_20")
        self.scramble_btn = QPushButton(self.groupBox_9)
        self.scramble_btn.setObjectName(u"scramble_btn")
        icon8 = QIcon()
        icon8.addFile(u":/scramble.png", QSize(), QIcon.Normal, QIcon.Off)
        self.scramble_btn.setIcon(icon8)

        self.gridLayout_20.addWidget(self.scramble_btn, 1, 0, 1, 1)

        self.horizontalSpacer_23 = QSpacerItem(2, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_20.addItem(self.horizontalSpacer_23, 1, 1, 1, 1)

        self.label_61 = QLabel(self.groupBox_9)
        self.label_61.setObjectName(u"label_61")
        self.label_61.setWordWrap(True)

        self.gridLayout_20.addWidget(self.label_61, 0, 0, 1, 2)


        self.gridLayout_22.addWidget(self.groupBox_9, 5, 0, 1, 2)

        self.label_28 = QLabel(self.MDF)
        self.label_28.setObjectName(u"label_28")

        self.gridLayout_22.addWidget(self.label_28, 4, 0, 1, 1)

        self.mdf_version = QComboBox(self.MDF)
        self.mdf_version.setObjectName(u"mdf_version")
        self.mdf_version.setMinimumSize(QSize(0, 0))

        self.gridLayout_22.addWidget(self.mdf_version, 0, 1, 1, 1)

        self.mdf_split = QCheckBox(self.MDF)
        self.mdf_split.setObjectName(u"mdf_split")
        self.mdf_split.setChecked(True)

        self.gridLayout_22.addWidget(self.mdf_split, 3, 0, 1, 2)

        self.output_options.addWidget(self.MDF)
        self.HDF5 = QWidget()
        self.HDF5.setObjectName(u"HDF5")
        self.gridLayout_2 = QGridLayout(self.HDF5)
        self.gridLayout_2.setSpacing(2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(2, 2, 2, 2)
        self.empty_channels = QComboBox(self.HDF5)
        self.empty_channels.setObjectName(u"empty_channels")

        self.gridLayout_2.addWidget(self.empty_channels, 8, 1, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_2.addItem(self.verticalSpacer_3, 9, 0, 1, 1)

        self.label_67 = QLabel(self.HDF5)
        self.label_67.setObjectName(u"label_67")

        self.gridLayout_2.addWidget(self.label_67, 7, 0, 1, 1)

        self.label_65 = QLabel(self.HDF5)
        self.label_65.setObjectName(u"label_65")

        self.gridLayout_2.addWidget(self.label_65, 8, 0, 1, 1)

        self.export_compression = QComboBox(self.HDF5)
        self.export_compression.setObjectName(u"export_compression")

        self.gridLayout_2.addWidget(self.export_compression, 7, 1, 1, 1)

        self.line_30 = QFrame(self.HDF5)
        self.line_30.setObjectName(u"line_30")
        self.line_30.setFrameShape(QFrame.HLine)
        self.line_30.setFrameShadow(QFrame.Sunken)

        self.gridLayout_2.addWidget(self.line_30, 4, 0, 1, 2)

        self.single_time_base = QCheckBox(self.HDF5)
        self.single_time_base.setObjectName(u"single_time_base")

        self.gridLayout_2.addWidget(self.single_time_base, 0, 0, 1, 2)

        self.time_from_zero = QCheckBox(self.HDF5)
        self.time_from_zero.setObjectName(u"time_from_zero")

        self.gridLayout_2.addWidget(self.time_from_zero, 1, 0, 1, 2)

        self.time_as_date = QCheckBox(self.HDF5)
        self.time_as_date.setObjectName(u"time_as_date")

        self.gridLayout_2.addWidget(self.time_as_date, 2, 0, 1, 2)

        self.raw = QCheckBox(self.HDF5)
        self.raw.setObjectName(u"raw")

        self.gridLayout_2.addWidget(self.raw, 3, 0, 1, 2)

        self.use_display_names = QCheckBox(self.HDF5)
        self.use_display_names.setObjectName(u"use_display_names")

        self.gridLayout_2.addWidget(self.use_display_names, 5, 0, 1, 2)

        self.reduce_memory_usage = QCheckBox(self.HDF5)
        self.reduce_memory_usage.setObjectName(u"reduce_memory_usage")

        self.gridLayout_2.addWidget(self.reduce_memory_usage, 6, 0, 1, 2)

        self.output_options.addWidget(self.HDF5)
        self.MAT = QWidget()
        self.MAT.setObjectName(u"MAT")
        self.gridLayout_3 = QGridLayout(self.MAT)
        self.gridLayout_3.setSpacing(2)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(2, 2, 2, 2)
        self.label_19 = QLabel(self.MAT)
        self.label_19.setObjectName(u"label_19")

        self.gridLayout_3.addWidget(self.label_19, 10, 0, 1, 1)

        self.label_70 = QLabel(self.MAT)
        self.label_70.setObjectName(u"label_70")

        self.gridLayout_3.addWidget(self.label_70, 7, 0, 1, 1)

        self.label_69 = QLabel(self.MAT)
        self.label_69.setObjectName(u"label_69")

        self.gridLayout_3.addWidget(self.label_69, 9, 0, 1, 1)

        self.export_compression_mat = QComboBox(self.MAT)
        self.export_compression_mat.setObjectName(u"export_compression_mat")

        self.gridLayout_3.addWidget(self.export_compression_mat, 7, 1, 1, 1)

        self.verticalSpacer_4 = QSpacerItem(20, 2, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_3.addItem(self.verticalSpacer_4, 11, 0, 1, 1)

        self.line_31 = QFrame(self.MAT)
        self.line_31.setObjectName(u"line_31")
        self.line_31.setFrameShape(QFrame.HLine)
        self.line_31.setFrameShadow(QFrame.Sunken)

        self.gridLayout_3.addWidget(self.line_31, 4, 0, 1, 2)

        self.empty_channels_mat = QComboBox(self.MAT)
        self.empty_channels_mat.setObjectName(u"empty_channels_mat")

        self.gridLayout_3.addWidget(self.empty_channels_mat, 8, 1, 1, 1)

        self.mat_format = QComboBox(self.MAT)
        self.mat_format.setObjectName(u"mat_format")

        self.gridLayout_3.addWidget(self.mat_format, 9, 1, 1, 1)

        self.oned_as = QComboBox(self.MAT)
        self.oned_as.setObjectName(u"oned_as")

        self.gridLayout_3.addWidget(self.oned_as, 10, 1, 1, 1)

        self.label_68 = QLabel(self.MAT)
        self.label_68.setObjectName(u"label_68")

        self.gridLayout_3.addWidget(self.label_68, 8, 0, 1, 1)

        self.single_time_base_mat = QCheckBox(self.MAT)
        self.single_time_base_mat.setObjectName(u"single_time_base_mat")

        self.gridLayout_3.addWidget(self.single_time_base_mat, 0, 0, 1, 2)

        self.time_from_zero_mat = QCheckBox(self.MAT)
        self.time_from_zero_mat.setObjectName(u"time_from_zero_mat")

        self.gridLayout_3.addWidget(self.time_from_zero_mat, 1, 0, 1, 2)

        self.time_as_date_mat = QCheckBox(self.MAT)
        self.time_as_date_mat.setObjectName(u"time_as_date_mat")

        self.gridLayout_3.addWidget(self.time_as_date_mat, 2, 0, 1, 2)

        self.raw_mat = QCheckBox(self.MAT)
        self.raw_mat.setObjectName(u"raw_mat")

        self.gridLayout_3.addWidget(self.raw_mat, 3, 0, 1, 2)

        self.use_display_names_mat = QCheckBox(self.MAT)
        self.use_display_names_mat.setObjectName(u"use_display_names_mat")

        self.gridLayout_3.addWidget(self.use_display_names_mat, 5, 0, 1, 2)

        self.reduce_memory_usage_mat = QCheckBox(self.MAT)
        self.reduce_memory_usage_mat.setObjectName(u"reduce_memory_usage_mat")

        self.gridLayout_3.addWidget(self.reduce_memory_usage_mat, 6, 0, 1, 2)

        self.output_options.addWidget(self.MAT)
        self.CSV = QWidget()
        self.CSV.setObjectName(u"CSV")
        self.gridLayout = QGridLayout(self.CSV)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(2, 2, 2, 2)
        self.empty_channels_csv = QComboBox(self.CSV)
        self.empty_channels_csv.setObjectName(u"empty_channels_csv")

        self.gridLayout.addWidget(self.empty_channels_csv, 7, 1, 1, 2)

        self.label_66 = QLabel(self.CSV)
        self.label_66.setObjectName(u"label_66")

        self.gridLayout.addWidget(self.label_66, 7, 0, 1, 1)

        self.doublequote = QCheckBox(self.CSV)
        self.doublequote.setObjectName(u"doublequote")
        self.doublequote.setChecked(True)

        self.gridLayout.addWidget(self.doublequote, 9, 0, 1, 1)

        self.line_32 = QFrame(self.CSV)
        self.line_32.setObjectName(u"line_32")
        self.line_32.setFrameShape(QFrame.HLine)
        self.line_32.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line_32, 4, 0, 1, 3)

        self.time_from_zero_csv = QCheckBox(self.CSV)
        self.time_from_zero_csv.setObjectName(u"time_from_zero_csv")

        self.gridLayout.addWidget(self.time_from_zero_csv, 1, 0, 1, 3)

        self.quotechar = QLineEdit(self.CSV)
        self.quotechar.setObjectName(u"quotechar")
        self.quotechar.setMaxLength(1)

        self.gridLayout.addWidget(self.quotechar, 12, 1, 1, 2)

        self.label_4 = QLabel(self.CSV)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 11, 0, 1, 1)

        self.delimiter = QLineEdit(self.CSV)
        self.delimiter.setObjectName(u"delimiter")
        self.delimiter.setMaxLength(1)
        self.delimiter.setClearButtonEnabled(False)

        self.gridLayout.addWidget(self.delimiter, 8, 1, 1, 2)

        self.label_5 = QLabel(self.CSV)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 12, 0, 1, 1)

        self.single_time_base_csv = QCheckBox(self.CSV)
        self.single_time_base_csv.setObjectName(u"single_time_base_csv")

        self.gridLayout.addWidget(self.single_time_base_csv, 0, 0, 1, 3)

        self.label_2 = QLabel(self.CSV)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 8, 0, 1, 1)

        self.label_3 = QLabel(self.CSV)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 10, 0, 1, 1)

        self.use_display_names_csv = QCheckBox(self.CSV)
        self.use_display_names_csv.setObjectName(u"use_display_names_csv")

        self.gridLayout.addWidget(self.use_display_names_csv, 5, 0, 1, 3)

        self.label_6 = QLabel(self.CSV)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 13, 0, 1, 1)

        self.quoting = QComboBox(self.CSV)
        self.quoting.addItem("")
        self.quoting.addItem("")
        self.quoting.addItem("")
        self.quoting.addItem("")
        self.quoting.setObjectName(u"quoting")

        self.gridLayout.addWidget(self.quoting, 13, 1, 1, 2)

        self.raw_csv = QCheckBox(self.CSV)
        self.raw_csv.setObjectName(u"raw_csv")

        self.gridLayout.addWidget(self.raw_csv, 3, 0, 1, 3)

        self.lineterminator = QLineEdit(self.CSV)
        self.lineterminator.setObjectName(u"lineterminator")

        self.gridLayout.addWidget(self.lineterminator, 11, 1, 1, 2)

        self.escapechar = QLineEdit(self.CSV)
        self.escapechar.setObjectName(u"escapechar")
        self.escapechar.setMaxLength(1)

        self.gridLayout.addWidget(self.escapechar, 10, 1, 1, 2)

        self.time_as_date_csv = QCheckBox(self.CSV)
        self.time_as_date_csv.setObjectName(u"time_as_date_csv")

        self.gridLayout.addWidget(self.time_as_date_csv, 2, 0, 1, 3)

        self.add_units = QCheckBox(self.CSV)
        self.add_units.setObjectName(u"add_units")

        self.gridLayout.addWidget(self.add_units, 6, 0, 1, 3)

        self.output_options.addWidget(self.CSV)
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.output_options.addWidget(self.page)

        self.verticalLayout_20.addWidget(self.output_options)


        self.verticalLayout_3.addWidget(self.groupBox_10)

        self.verticalSpacer_2 = QSpacerItem(20, 2, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_2)

        self.apply_btn = QPushButton(self.scrollAreaWidgetContents)
        self.apply_btn.setObjectName(u"apply_btn")
        self.apply_btn.setIcon(icon2)

        self.verticalLayout_3.addWidget(self.apply_btn)

        self.verticalLayout_3.setStretch(3, 1)

        self.verticalLayout_4.addLayout(self.verticalLayout_3)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.horizontalLayout.addWidget(self.scrollArea)

        icon9 = QIcon()
        icon9.addFile(u":/convert.png", QSize(), QIcon.Normal, QIcon.Off)
        self.aspects.addTab(self.modify, icon9, "")
        self.extract_bus_tab = QWidget()
        self.extract_bus_tab.setObjectName(u"extract_bus_tab")
        self.verticalLayout_8 = QVBoxLayout(self.extract_bus_tab)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.scrollArea_2 = QScrollArea(self.extract_bus_tab)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 985, 596))
        self.gridLayout_7 = QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.groupBox_3 = QGroupBox(self.scrollAreaWidgetContents_2)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.gridLayout_6 = QGridLayout(self.groupBox_3)
        self.gridLayout_6.setSpacing(2)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_6.setContentsMargins(-1, 2, 2, 2)
        self.time_from_zero_bus = QCheckBox(self.groupBox_3)
        self.time_from_zero_bus.setObjectName(u"time_from_zero_bus")

        self.gridLayout_6.addWidget(self.time_from_zero_bus, 1, 0, 1, 3)

        self.quotechar_bus = QLineEdit(self.groupBox_3)
        self.quotechar_bus.setObjectName(u"quotechar_bus")
        self.quotechar_bus.setMaxLength(1)

        self.gridLayout_6.addWidget(self.quotechar_bus, 11, 2, 1, 1)

        self.export_raster_bus = QDoubleSpinBox(self.groupBox_3)
        self.export_raster_bus.setObjectName(u"export_raster_bus")
        self.export_raster_bus.setDecimals(6)

        self.gridLayout_6.addWidget(self.export_raster_bus, 4, 2, 1, 1)

        self.add_units_bus = QCheckBox(self.groupBox_3)
        self.add_units_bus.setObjectName(u"add_units_bus")

        self.gridLayout_6.addWidget(self.add_units_bus, 3, 0, 1, 3)

        self.single_time_base_bus = QCheckBox(self.groupBox_3)
        self.single_time_base_bus.setObjectName(u"single_time_base_bus")

        self.gridLayout_6.addWidget(self.single_time_base_bus, 0, 0, 1, 3)

        self.bus_time_as_date = QCheckBox(self.groupBox_3)
        self.bus_time_as_date.setObjectName(u"bus_time_as_date")

        self.gridLayout_6.addWidget(self.bus_time_as_date, 2, 0, 1, 3)

        self.doublequote_bus = QCheckBox(self.groupBox_3)
        self.doublequote_bus.setObjectName(u"doublequote_bus")
        self.doublequote_bus.setChecked(True)

        self.gridLayout_6.addWidget(self.doublequote_bus, 7, 0, 1, 3)

        self.label_8 = QLabel(self.groupBox_3)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_6.addWidget(self.label_8, 6, 0, 1, 1)

        self.label_11 = QLabel(self.groupBox_3)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_6.addWidget(self.label_11, 10, 0, 1, 2)

        self.label_23 = QLabel(self.groupBox_3)
        self.label_23.setObjectName(u"label_23")

        self.gridLayout_6.addWidget(self.label_23, 4, 0, 1, 1)

        self.escapechar_bus = QLineEdit(self.groupBox_3)
        self.escapechar_bus.setObjectName(u"escapechar_bus")
        self.escapechar_bus.setMaxLength(1)

        self.gridLayout_6.addWidget(self.escapechar_bus, 8, 2, 1, 1)

        self.lineterminator_bus = QLineEdit(self.groupBox_3)
        self.lineterminator_bus.setObjectName(u"lineterminator_bus")

        self.gridLayout_6.addWidget(self.lineterminator_bus, 10, 2, 1, 1)

        self.quoting_bus = QComboBox(self.groupBox_3)
        self.quoting_bus.addItem("")
        self.quoting_bus.addItem("")
        self.quoting_bus.addItem("")
        self.quoting_bus.addItem("")
        self.quoting_bus.setObjectName(u"quoting_bus")

        self.gridLayout_6.addWidget(self.quoting_bus, 12, 2, 1, 1)

        self.label_10 = QLabel(self.groupBox_3)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_6.addWidget(self.label_10, 11, 0, 1, 1)

        self.line_13 = QFrame(self.groupBox_3)
        self.line_13.setObjectName(u"line_13")
        self.line_13.setFrameShape(QFrame.HLine)
        self.line_13.setFrameShadow(QFrame.Sunken)

        self.gridLayout_6.addWidget(self.line_13, 13, 0, 1, 3)

        self.delimiter_bus = QLineEdit(self.groupBox_3)
        self.delimiter_bus.setObjectName(u"delimiter_bus")
        self.delimiter_bus.setMaxLength(1)
        self.delimiter_bus.setClearButtonEnabled(False)

        self.gridLayout_6.addWidget(self.delimiter_bus, 6, 2, 1, 1)

        self.label_25 = QLabel(self.groupBox_3)
        self.label_25.setObjectName(u"label_25")

        self.gridLayout_6.addWidget(self.label_25, 5, 0, 1, 2)

        self.label_7 = QLabel(self.groupBox_3)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_6.addWidget(self.label_7, 12, 0, 1, 1)

        self.label_9 = QLabel(self.groupBox_3)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_6.addWidget(self.label_9, 8, 0, 1, 1)

        self.extract_bus_csv_btn = QPushButton(self.groupBox_3)
        self.extract_bus_csv_btn.setObjectName(u"extract_bus_csv_btn")
        icon10 = QIcon()
        icon10.addFile(u":/csv.png", QSize(), QIcon.Normal, QIcon.Off)
        self.extract_bus_csv_btn.setIcon(icon10)

        self.gridLayout_6.addWidget(self.extract_bus_csv_btn, 14, 2, 1, 1)

        self.empty_channels_bus = QComboBox(self.groupBox_3)
        self.empty_channels_bus.setObjectName(u"empty_channels_bus")

        self.gridLayout_6.addWidget(self.empty_channels_bus, 5, 2, 1, 1)

        self.gridLayout_6.setColumnStretch(0, 1)

        self.gridLayout_7.addWidget(self.groupBox_3, 2, 1, 1, 1)

        self.output_info_bus = QTextEdit(self.scrollAreaWidgetContents_2)
        self.output_info_bus.setObjectName(u"output_info_bus")
        self.output_info_bus.setReadOnly(True)

        self.gridLayout_7.addWidget(self.output_info_bus, 0, 2, 3, 1)

        self.groupBox_2 = QGroupBox(self.scrollAreaWidgetContents_2)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_5 = QGridLayout(self.groupBox_2)
        self.gridLayout_5.setSpacing(2)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(2, 2, 2, 2)
        self.label__1 = QLabel(self.groupBox_2)
        self.label__1.setObjectName(u"label__1")

        self.gridLayout_5.addWidget(self.label__1, 0, 0, 1, 1)

        self.line_12 = QFrame(self.groupBox_2)
        self.line_12.setObjectName(u"line_12")
        self.line_12.setFrameShape(QFrame.HLine)
        self.line_12.setFrameShadow(QFrame.Sunken)

        self.gridLayout_5.addWidget(self.line_12, 3, 0, 1, 3)

        self.label_26 = QLabel(self.groupBox_2)
        self.label_26.setObjectName(u"label_26")

        self.gridLayout_5.addWidget(self.label_26, 2, 1, 1, 1)

        self.extract_bus_format = QComboBox(self.groupBox_2)
        self.extract_bus_format.setObjectName(u"extract_bus_format")
        self.extract_bus_format.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.gridLayout_5.addWidget(self.extract_bus_format, 1, 1, 1, 2)

        self.extract_bus_compression = QComboBox(self.groupBox_2)
        self.extract_bus_compression.setObjectName(u"extract_bus_compression")
        self.extract_bus_compression.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.gridLayout_5.addWidget(self.extract_bus_compression, 0, 1, 1, 2)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_5.addItem(self.horizontalSpacer_4, 4, 0, 1, 1)

        self.extract_bus_btn = QPushButton(self.groupBox_2)
        self.extract_bus_btn.setObjectName(u"extract_bus_btn")
        icon11 = QIcon()
        icon11.addFile(u":/down.png", QSize(), QIcon.Normal, QIcon.Off)
        self.extract_bus_btn.setIcon(icon11)

        self.gridLayout_5.addWidget(self.extract_bus_btn, 4, 1, 1, 2)

        self.label_24 = QLabel(self.groupBox_2)
        self.label_24.setObjectName(u"label_24")

        self.gridLayout_5.addWidget(self.label_24, 1, 0, 1, 1)


        self.gridLayout_7.addWidget(self.groupBox_2, 2, 0, 1, 1)

        self.tabWidget = QTabWidget(self.scrollAreaWidgetContents_2)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_4 = QGridLayout(self.tab)
        self.gridLayout_4.setSpacing(2)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setContentsMargins(2, 2, 2, 2)
        self.load_can_database_btn = QPushButton(self.tab)
        self.load_can_database_btn.setObjectName(u"load_can_database_btn")
        self.load_can_database_btn.setIcon(icon)

        self.gridLayout_4.addWidget(self.load_can_database_btn, 0, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_4.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.can_database_list = MinimalListWidget(self.tab)
        self.can_database_list.setObjectName(u"can_database_list")

        self.gridLayout_4.addWidget(self.can_database_list, 1, 0, 1, 2)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.gridLayout_8 = QGridLayout(self.tab_2)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_8.setContentsMargins(2, 2, 2, 2)
        self.load_lin_database_btn = QPushButton(self.tab_2)
        self.load_lin_database_btn.setObjectName(u"load_lin_database_btn")
        self.load_lin_database_btn.setIcon(icon)

        self.gridLayout_8.addWidget(self.load_lin_database_btn, 0, 0, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_8.addItem(self.horizontalSpacer_3, 0, 1, 1, 1)

        self.lin_database_list = MinimalListWidget(self.tab_2)
        self.lin_database_list.setObjectName(u"lin_database_list")

        self.gridLayout_8.addWidget(self.lin_database_list, 1, 0, 1, 2)

        self.tabWidget.addTab(self.tab_2, "")

        self.gridLayout_7.addWidget(self.tabWidget, 0, 0, 1, 2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_12 = QLabel(self.scrollAreaWidgetContents_2)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout_3.addWidget(self.label_12)

        self.prefix = QLineEdit(self.scrollAreaWidgetContents_2)
        self.prefix.setObjectName(u"prefix")
        self.prefix.setMinimumSize(QSize(0, 0))

        self.horizontalLayout_3.addWidget(self.prefix)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_5)

        self.horizontalLayout_3.setStretch(2, 1)

        self.gridLayout_7.addLayout(self.horizontalLayout_3, 1, 0, 1, 2)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_8.addWidget(self.scrollArea_2)

        self.aspects.addTab(self.extract_bus_tab, icon11, "")
        self.info_tab = QWidget()
        self.info_tab.setObjectName(u"info_tab")
        self.gridLayout_9 = QGridLayout(self.info_tab)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.info = QTreeWidget(self.info_tab)
        self.info.setObjectName(u"info")
        self.info.setUniformRowHeights(False)

        self.gridLayout_9.addWidget(self.info, 0, 0, 1, 1)

        icon12 = QIcon()
        icon12.addFile(u":/info.png", QSize(), QIcon.Normal, QIcon.Off)
        self.aspects.addTab(self.info_tab, icon12, "")
        self.attachments_tab = QWidget()
        self.attachments_tab.setObjectName(u"attachments_tab")
        self.verticalLayout_10 = QVBoxLayout(self.attachments_tab)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.attachments = QListWidget(self.attachments_tab)
        self.attachments.setObjectName(u"attachments")

        self.verticalLayout_10.addWidget(self.attachments)

        self.aspects.addTab(self.attachments_tab, icon7, "")

        self.verticalLayout.addWidget(self.aspects)


        self.retranslateUi(file_widget)

        self.aspects.setCurrentIndex(0)
        self.output_options.setCurrentIndex(0)
        self.quoting.setCurrentIndex(1)
        self.quoting_bus.setCurrentIndex(1)
        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(file_widget)
    # setupUi

    def retranslateUi(self, file_widget):
        file_widget.setWindowTitle(QCoreApplication.translate("file_widget", u"Form", None))
#if QT_CONFIG(tooltip)
        self.aspects.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.channel_view.setItemText(0, QCoreApplication.translate("file_widget", u"Natural sort", None))
        self.channel_view.setItemText(1, QCoreApplication.translate("file_widget", u"Internal file structure", None))
        self.channel_view.setItemText(2, QCoreApplication.translate("file_widget", u"Selected channels only", None))

        ___qtreewidgetitem = self.channels_tree.headerItem()
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("file_widget", u"Channels", None));
#if QT_CONFIG(tooltip)
        self.channels_tree.setToolTip(QCoreApplication.translate("file_widget", u"Double click channel to see extended information", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.load_channel_list_btn.setToolTip(QCoreApplication.translate("file_widget", u"Load display file", None))
#endif // QT_CONFIG(tooltip)
        self.load_channel_list_btn.setText("")
#if QT_CONFIG(tooltip)
        self.save_channel_list_btn.setToolTip(QCoreApplication.translate("file_widget", u"Save display file", None))
#endif // QT_CONFIG(tooltip)
        self.save_channel_list_btn.setText("")
#if QT_CONFIG(tooltip)
        self.select_all_btn.setToolTip(QCoreApplication.translate("file_widget", u"Select all the channels", None))
#endif // QT_CONFIG(tooltip)
        self.select_all_btn.setText("")
#if QT_CONFIG(tooltip)
        self.clear_channels_btn.setToolTip(QCoreApplication.translate("file_widget", u"Clear all selected channels", None))
#endif // QT_CONFIG(tooltip)
        self.clear_channels_btn.setText("")
#if QT_CONFIG(tooltip)
        self.advanced_search_btn.setToolTip(QCoreApplication.translate("file_widget", u"Search and select channels", None))
#endif // QT_CONFIG(tooltip)
        self.advanced_search_btn.setText("")
#if QT_CONFIG(tooltip)
        self.create_window_btn.setToolTip(QCoreApplication.translate("file_widget", u"Create window", None))
#endif // QT_CONFIG(tooltip)
        self.create_window_btn.setText("")
#if QT_CONFIG(tooltip)
        self.load_embedded_channel_list_btn.setToolTip(QCoreApplication.translate("file_widget", u"Load embedded display file", None))
#endif // QT_CONFIG(tooltip)
        self.load_embedded_channel_list_btn.setText("")
#if QT_CONFIG(tooltip)
        self.save_embedded_channel_list_btn.setToolTip(QCoreApplication.translate("file_widget", u"Embed display file", None))
#endif // QT_CONFIG(tooltip)
        self.save_embedded_channel_list_btn.setText("")
        self.aspects.setTabText(self.aspects.indexOf(self.channels_tab), QCoreApplication.translate("file_widget", u"Channels", None))
        self.filter_view.setItemText(0, QCoreApplication.translate("file_widget", u"Natural sort", None))
        self.filter_view.setItemText(1, QCoreApplication.translate("file_widget", u"Internal file structure", None))
        self.filter_view.setItemText(2, QCoreApplication.translate("file_widget", u"Selected channels only", None))

        ___qtreewidgetitem1 = self.filter_tree.headerItem()
        ___qtreewidgetitem1.setText(0, QCoreApplication.translate("file_widget", u"Channels", None));
#if QT_CONFIG(tooltip)
        self.filter_tree.setToolTip(QCoreApplication.translate("file_widget", u"Double click channel to see extended information", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.load_filter_list_btn.setToolTip(QCoreApplication.translate("file_widget", u"Load channel selection list", None))
#endif // QT_CONFIG(tooltip)
        self.load_filter_list_btn.setText("")
#if QT_CONFIG(tooltip)
        self.save_filter_list_btn.setToolTip(QCoreApplication.translate("file_widget", u"Save channel selection list", None))
#endif // QT_CONFIG(tooltip)
        self.save_filter_list_btn.setText("")
#if QT_CONFIG(tooltip)
        self.clear_filter_btn.setToolTip(QCoreApplication.translate("file_widget", u"Clear selection", None))
#endif // QT_CONFIG(tooltip)
        self.clear_filter_btn.setText("")
#if QT_CONFIG(tooltip)
        self.advanced_serch_filter_btn.setToolTip(QCoreApplication.translate("file_widget", u"Search and select channels", None))
#endif // QT_CONFIG(tooltip)
        self.advanced_serch_filter_btn.setText("")
        self.label.setText(QCoreApplication.translate("file_widget", u"All selected channels", None))
        self.cut_group.setTitle(QCoreApplication.translate("file_widget", u"Cut", None))
        self.label_59.setText(QCoreApplication.translate("file_widget", u"Start", None))
        self.cut_stop.setSuffix(QCoreApplication.translate("file_widget", u"s", None))
        self.label_60.setText(QCoreApplication.translate("file_widget", u"End", None))
        self.cut_start.setSuffix(QCoreApplication.translate("file_widget", u"s", None))
        self.whence.setText(QCoreApplication.translate("file_widget", u"Start relative to first time stamp", None))
        self.cut_time_from_zero.setText(QCoreApplication.translate("file_widget", u"Time from 0s", None))
        self.resample_group.setTitle(QCoreApplication.translate("file_widget", u"Resample", None))
        self.raster_type_step.setText(QCoreApplication.translate("file_widget", u"step", None))
        self.raster.setSuffix(QCoreApplication.translate("file_widget", u"s", None))
        self.raster_type_channel.setText(QCoreApplication.translate("file_widget", u"channel", None))
#if QT_CONFIG(tooltip)
        self.raster_search_btn.setToolTip(QCoreApplication.translate("file_widget", u"Search raster channel", None))
#endif // QT_CONFIG(tooltip)
        self.raster_search_btn.setText("")
        self.resample_time_from_zero.setText(QCoreApplication.translate("file_widget", u"Time from 0s", None))
        self.groupBox_10.setTitle(QCoreApplication.translate("file_widget", u"Ouput format", None))
        self.output_format.setItemText(0, QCoreApplication.translate("file_widget", u"MDF", None))
        self.output_format.setItemText(1, QCoreApplication.translate("file_widget", u"ASC", None))
        self.output_format.setItemText(2, QCoreApplication.translate("file_widget", u"CSV", None))
        self.output_format.setItemText(3, QCoreApplication.translate("file_widget", u"HDF5", None))
        self.output_format.setItemText(4, QCoreApplication.translate("file_widget", u"MAT", None))
        self.output_format.setItemText(5, QCoreApplication.translate("file_widget", u"Parquet", None))

        self.mdf_split_size.setSuffix(QCoreApplication.translate("file_widget", u"MB", None))
        self.label_29.setText(QCoreApplication.translate("file_widget", u"Compression", None))
        self.label_27.setText(QCoreApplication.translate("file_widget", u"Version", None))
        self.groupBox_9.setTitle(QCoreApplication.translate("file_widget", u"Scramble", None))
        self.scramble_btn.setText(QCoreApplication.translate("file_widget", u"Scramble texts", None))
        self.label_61.setText(QCoreApplication.translate("file_widget", u"Anonymize the measurements: scramble all texts and replace them with random strings", None))
        self.label_28.setText(QCoreApplication.translate("file_widget", u"Split size ", None))
        self.mdf_split.setText(QCoreApplication.translate("file_widget", u"Split data blocks", None))
        self.label_67.setText(QCoreApplication.translate("file_widget", u"Compression", None))
        self.label_65.setText(QCoreApplication.translate("file_widget", u"Empty channels", None))
        self.single_time_base.setText(QCoreApplication.translate("file_widget", u"Single time base", None))
        self.time_from_zero.setText(QCoreApplication.translate("file_widget", u"Time from 0s", None))
        self.time_as_date.setText(QCoreApplication.translate("file_widget", u"Time as date", None))
        self.raw.setText(QCoreApplication.translate("file_widget", u"Raw values", None))
        self.use_display_names.setText(QCoreApplication.translate("file_widget", u"Use display names", None))
        self.reduce_memory_usage.setText(QCoreApplication.translate("file_widget", u"Reduce  memory usage", None))
        self.label_19.setText(QCoreApplication.translate("file_widget", u".mat oned_as", None))
        self.label_70.setText(QCoreApplication.translate("file_widget", u"Compression", None))
        self.label_69.setText(QCoreApplication.translate("file_widget", u".mat format", None))
        self.label_68.setText(QCoreApplication.translate("file_widget", u"Empty channels", None))
        self.single_time_base_mat.setText(QCoreApplication.translate("file_widget", u"Single time base", None))
        self.time_from_zero_mat.setText(QCoreApplication.translate("file_widget", u"Time from 0s", None))
        self.time_as_date_mat.setText(QCoreApplication.translate("file_widget", u"Time as date", None))
        self.raw_mat.setText(QCoreApplication.translate("file_widget", u"Raw values", None))
        self.use_display_names_mat.setText(QCoreApplication.translate("file_widget", u"Use display names", None))
        self.reduce_memory_usage_mat.setText(QCoreApplication.translate("file_widget", u"Reduce  memory usage", None))
        self.label_66.setText(QCoreApplication.translate("file_widget", u"Empty channels", None))
        self.doublequote.setText(QCoreApplication.translate("file_widget", u"Double quote", None))
        self.time_from_zero_csv.setText(QCoreApplication.translate("file_widget", u"Time from 0s", None))
        self.quotechar.setText(QCoreApplication.translate("file_widget", u"\"", None))
        self.label_4.setText(QCoreApplication.translate("file_widget", u"Line Terminator", None))
        self.delimiter.setText(QCoreApplication.translate("file_widget", u",", None))
        self.label_5.setText(QCoreApplication.translate("file_widget", u"Quote Char", None))
        self.single_time_base_csv.setText(QCoreApplication.translate("file_widget", u"Single time base", None))
        self.label_2.setText(QCoreApplication.translate("file_widget", u"Delimiter", None))
        self.label_3.setText(QCoreApplication.translate("file_widget", u"Escape Char", None))
        self.use_display_names_csv.setText(QCoreApplication.translate("file_widget", u"Use display names", None))
        self.label_6.setText(QCoreApplication.translate("file_widget", u"Quoting", None))
        self.quoting.setItemText(0, QCoreApplication.translate("file_widget", u"ALL", None))
        self.quoting.setItemText(1, QCoreApplication.translate("file_widget", u"MINIMAL", None))
        self.quoting.setItemText(2, QCoreApplication.translate("file_widget", u"NONNUMERIC", None))
        self.quoting.setItemText(3, QCoreApplication.translate("file_widget", u"NONE", None))

        self.raw_csv.setText(QCoreApplication.translate("file_widget", u"Raw values", None))
        self.lineterminator.setText(QCoreApplication.translate("file_widget", u"\\r\\n", None))
        self.escapechar.setInputMask("")
        self.escapechar.setPlaceholderText(QCoreApplication.translate("file_widget", u"None", None))
        self.time_as_date_csv.setText(QCoreApplication.translate("file_widget", u"Time as date", None))
        self.add_units.setText(QCoreApplication.translate("file_widget", u"Add units", None))
        self.apply_btn.setText(QCoreApplication.translate("file_widget", u"Apply", None))
        self.aspects.setTabText(self.aspects.indexOf(self.modify), QCoreApplication.translate("file_widget", u"Modify && Export", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("file_widget", u"CSV", None))
        self.time_from_zero_bus.setText(QCoreApplication.translate("file_widget", u"Time from 0s", None))
        self.quotechar_bus.setText(QCoreApplication.translate("file_widget", u"\"", None))
        self.export_raster_bus.setSuffix(QCoreApplication.translate("file_widget", u"s", None))
        self.add_units_bus.setText(QCoreApplication.translate("file_widget", u"Add units", None))
        self.single_time_base_bus.setText(QCoreApplication.translate("file_widget", u"Single time base", None))
        self.bus_time_as_date.setText(QCoreApplication.translate("file_widget", u"Time as date", None))
        self.doublequote_bus.setText(QCoreApplication.translate("file_widget", u"Double quote", None))
        self.label_8.setText(QCoreApplication.translate("file_widget", u"Delimiter", None))
        self.label_11.setText(QCoreApplication.translate("file_widget", u"Line Terminator", None))
        self.label_23.setText(QCoreApplication.translate("file_widget", u"Raster", None))
        self.escapechar_bus.setInputMask("")
        self.escapechar_bus.setPlaceholderText(QCoreApplication.translate("file_widget", u"None", None))
        self.lineterminator_bus.setText(QCoreApplication.translate("file_widget", u"\\r\\n", None))
        self.quoting_bus.setItemText(0, QCoreApplication.translate("file_widget", u"ALL", None))
        self.quoting_bus.setItemText(1, QCoreApplication.translate("file_widget", u"MINIMAL", None))
        self.quoting_bus.setItemText(2, QCoreApplication.translate("file_widget", u"NONNUMERIC", None))
        self.quoting_bus.setItemText(3, QCoreApplication.translate("file_widget", u"NONE", None))

        self.label_10.setText(QCoreApplication.translate("file_widget", u"Quote Char", None))
        self.delimiter_bus.setText(QCoreApplication.translate("file_widget", u",", None))
        self.label_25.setText(QCoreApplication.translate("file_widget", u"Empty channels", None))
        self.label_7.setText(QCoreApplication.translate("file_widget", u"Quoting", None))
        self.label_9.setText(QCoreApplication.translate("file_widget", u"Escape Char", None))
        self.extract_bus_csv_btn.setText(QCoreApplication.translate("file_widget", u"Export to CSV         ", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("file_widget", u"MDF", None))
        self.label__1.setText(QCoreApplication.translate("file_widget", u"Compression", None))
        self.label_26.setText("")
        self.extract_bus_btn.setText(QCoreApplication.translate("file_widget", u"Extract Bus signals", None))
        self.label_24.setText(QCoreApplication.translate("file_widget", u"Version", None))
        self.load_can_database_btn.setText(QCoreApplication.translate("file_widget", u"Load CAN database", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("file_widget", u"CAN", None))
        self.load_lin_database_btn.setText(QCoreApplication.translate("file_widget", u"Load LIN database", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("file_widget", u"LIN", None))
        self.label_12.setText(QCoreApplication.translate("file_widget", u"Prefix", None))
        self.aspects.setTabText(self.aspects.indexOf(self.extract_bus_tab), QCoreApplication.translate("file_widget", u"Bus Logging", None))
        ___qtreewidgetitem2 = self.info.headerItem()
        ___qtreewidgetitem2.setText(1, QCoreApplication.translate("file_widget", u"Value", None));
        ___qtreewidgetitem2.setText(0, QCoreApplication.translate("file_widget", u"Cathegory", None));
        self.aspects.setTabText(self.aspects.indexOf(self.info_tab), QCoreApplication.translate("file_widget", u"Info", None))
        self.aspects.setTabText(self.aspects.indexOf(self.attachments_tab), QCoreApplication.translate("file_widget", u"Attachments", None))
    # retranslateUi


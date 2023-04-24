#!/usr/bin/env python
import pathlib
from test.asammdf.gui import QtCore, QtTest, QtWidgets
from test.asammdf.gui.test_base import DragAndDrop, TestBase
import time
from unittest import mock

from PySide6 import QtCore

from asammdf.gui.widgets.file import FileWidget


class TestPlotWidget(TestBase):
    # Note: Test Plot Widget through FileWidget.
    def setUp(self):
        super().setUp()
        self.widget = None

    def tearDown(self):
        if self.widget:
            self.widget.close()
            self.widget.destroy()
            self.widget.deleteLater()
        self.mc_ErrorDialog.reset_mock()
        super().tearDown()

    def test_Plot_DragAndDrop_fromFile_toPlot_0(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Drag and Drop channel from FileWidget.channel_tree to Plot
            - Drag and Drop same channel from FileWidget.channel_tree to Plot
        Evaluate:
            - Evaluate that two channels are added to Plot "channel_selection"
        """
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Event
        self.widget = FileWidget(
            measurement_file,
            True,  # with_dots
            True,  # subplots
            True,  # subplots_link
            False,  # ignore_value2text_conversions
            False,  # display_cg_name
            "line",  # line_interconnect
            1,  # password
            None,  # hide_missing_channels
            None,  # hide_disabled_channels
        )
        self.widget.showNormal()
        self.widget.activateWindow()
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        with mock.patch(
            "asammdf.gui.widgets.file.WindowSelectionDialog"
        ) as mc_WindowSelectionDialog:
            mc_WindowSelectionDialog.return_value.result.return_value = True
            mc_WindowSelectionDialog.return_value.selected_type.return_value = "Plot"
            # - Press PushButton "Create Window"
            QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.LeftButton)
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
            widget_types = sorted(
                map(
                    lambda w: w.widget().__class__.__name__,
                    self.widget.mdi_area.subWindowList(),
                )
            )
            self.assertIn("Plot", widget_types)

        # Select channel
        channel_tree = self.widget.channels_tree
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        item = channel_tree.topLevelItem(0)
        channel_name = item.text(0)
        drag_position = channel_tree.visualItemRect(item).center()
        drop_position = plot.channel_selection.viewport().rect().center()

        # PreEvaluation
        self.assertEqual(0, plot.channel_selection.topLevelItemCount())
        DragAndDrop(
            source_widget=channel_tree,
            destination_widget=plot.channel_selection,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        # Evaluate
        plot_channel = plot.channel_selection.topLevelItem(0)
        plot_channel_name = plot_channel.text(0)
        self.assertEqual(1, plot.channel_selection.topLevelItemCount())
        self.assertEqual(channel_name, plot_channel_name)

        drop_position = plot.plot.viewport().rect().center()
        DragAndDrop(
            source_widget=channel_tree,
            destination_widget=plot.plot,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        self.assertEqual(2, plot.channel_selection.topLevelItemCount())

    def test_Plot_DragAndDrop_fromFile_toPlot_1(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Select 3 channels from FileWidget.channel_tree
            - Drag and Drop channels from FileWidget.channel_tree to Plot
        Evaluate:
            - Evaluate that 3 channels are added to Plot "channel_selection"
        """
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Event
        self.widget = FileWidget(
            measurement_file,
            True,  # with_dots
            True,  # subplots
            True,  # subplots_link
            False,  # ignore_value2text_conversions
            False,  # display_cg_name
            "line",  # line_interconnect
            1,  # password
            None,  # hide_missing_channels
            None,  # hide_disabled_channels
        )
        self.widget.showNormal()
        self.widget.activateWindow()
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        with mock.patch(
            "asammdf.gui.widgets.file.WindowSelectionDialog"
        ) as mc_WindowSelectionDialog:
            mc_WindowSelectionDialog.return_value.result.return_value = True
            mc_WindowSelectionDialog.return_value.selected_type.return_value = "Plot"
            # - Press PushButton "Create Window"
            QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.LeftButton)
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
            widget_types = sorted(
                map(
                    lambda w: w.widget().__class__.__name__,
                    self.widget.mdi_area.subWindowList(),
                )
            )
            self.assertIn("Plot", widget_types)

        channel_tree = self.widget.channels_tree
        plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Select 3 from channels but not from beginning
        # Avoid $ChannelLog because it's empty channel
        iterator = QtWidgets.QTreeWidgetItemIterator(channel_tree)
        count = 6
        selected_channels = []
        item = None
        while iterator.value() and count:
            count -= 1
            if count > 2:
                iterator += 1
                continue
            item = iterator.value()
            item.setSelected(True)
            selected_channels.append(item.text(0))
            iterator += 1

        drag_position = channel_tree.visualItemRect(item).center()
        drop_position = plot.channel_selection.viewport().rect().center()

        # PreEvaluation
        self.assertEqual(0, plot.channel_selection.topLevelItemCount())
        DragAndDrop(
            source_widget=channel_tree,
            destination_widget=plot,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        # Evaluate
        self.assertEqual(3, plot.channel_selection.topLevelItemCount())
        iterator = QtWidgets.QTreeWidgetItemIterator(plot.channel_selection)
        plot_channels = []
        while iterator.value():
            item = iterator.value()
            plot_channels.append(item.text(0))
            iterator += 1
        self.assertListEqual(selected_channels, plot_channels)

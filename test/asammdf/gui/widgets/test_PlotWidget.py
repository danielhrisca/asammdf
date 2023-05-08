#!/usr/bin/env python
import pathlib
import sys
from test.asammdf.gui import QtCore, QtGui, QtTest, QtWidgets
from test.asammdf.gui.test_base import DragAndDrop, TestBase
import unittest
from unittest import mock

from PySide6 import QtCore

from asammdf.gui.widgets.file import FileWidget


class TestDoubleClick(TestBase):
    # Note: Test Plot Widget through FileWidget.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.measurement_file = str(pathlib.Path(cls.resource, "ASAP2_Demo_V171.mf4"))

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

    def test_DoubleClick_ChannelSelection(self):
        """
        Test Scope: Validate that doubleClick operation will activate de-activate channels.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Drag and Drop channel from FileWidget.channel_tree to Plot
            - Press mouse double click on added channel
            - Press mouse double click on added channel
        Evaluate:
            - Evaluate that channel checkbox is unchecked.
            - Evaluate that channel checkbox is checked.
        """
        # Event
        self.widget = FileWidget(
            self.measurement_file,
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

        # Pre-evaluation
        self.assertEqual(QtCore.Qt.Checked, plot_channel.checkState(0))

        # Press mouse double click on channel
        QtTest.QTest.mouseDClick(
            plot.channel_selection.viewport(),
            QtCore.Qt.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            plot.channel_selection.visualItemRect(plot_channel).center(),
        )
        self.processEvents(0.5)

        # Evaluate
        self.assertEqual(QtCore.Qt.Unchecked, plot_channel.checkState(0))

        # Press mouse double click on channel
        QtTest.QTest.mouseDClick(
            plot.channel_selection.viewport(),
            QtCore.Qt.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            plot.channel_selection.visualItemRect(plot_channel).center(),
        )
        self.assertEqual(QtCore.Qt.Checked, plot_channel.checkState(0))


class TestDragAndDrop(TestBase):
    # Note: Test Plot Widget through FileWidget.

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.measurement_file = str(pathlib.Path(cls.resource, "ASAP2_Demo_V171.mf4"))

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

    def test_Plot_ChannelSelection_DragAndDrop_fromFile_toPlot_0(self):
        """
        Test Scope:
            - Test DragAndDrop Action from FileWidget.channel_tree to Plot.channel_selection and to Plot.plot.
            - Channels are selection one by one.
            - Ensure that Drag and Drop Action allow channels to be added to Plot.channel_selection
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
        # Event
        self.widget = FileWidget(
            self.measurement_file,
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

    def test_Plot_ChannelSelection_DragAndDrop_fromFile_toPlot_1(self):
        """
        Test Scope:
            - Test DragAndDrop Action from FileWidget.channel_tree to Plot.channel_selection and to Plot.plot.
            - Multiple Channels are selected and dragged.
            - Ensure that Drag and Drop Action allow channels to be added to Plot.channel_selection
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
        # Event
        self.widget = FileWidget(
            self.measurement_file,
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

    @unittest.skipIf(sys.platform != "win32", "Test applicable just for Windows.")
    # Test is applicable just for Windows because of Drag and Drop implementation.
    # In order to perform this operation two threads are needed, one to startDrag operation and the other one to move
    # the cursor and Release/drop the item.
    # It may happen that drag operation or drop operation to lead on starting/stopping internal QTimers.
    # On Linux closing any QTimer/QThread in other one thread than parent leads to Segmentation Fault.
    # Windows behaves differently on startDrag operation.
    def test_Plot_ChannelSelection_DragAndDrop_fromPlotCS_toPlot(self):
        """
        Test Scope:
            - Test DragAndDrop Action from Plot.channel_selection to Plot.channel_selection
            - Case 0: Channels are selected one by one.
            - Case 1: Multiple Channels are selected.
            - Case 2: Create Channel Group. Drag channels inside the group one by one
            - Case 3: Create Channel Group. Drag multiple channels at once inside the group.
            - Case 4: Drag Group inside the Group
            - Case 5: Drag Group outside the Group
            - Ensure that Drag and Drop Action allow channels to be sorted to Plot.channel_selection
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Select 5 channels and DragAndDrop them to Plot.channel_selection.
                - One channel should be duplicated.
            - Case 0:
                - DragAndDrop first channel to 3rd position.
            - Case 1:
                - DragAndDrop 2nd and 3rd channels on last position.
            - Case 2:
                - Create ChannelGroup
                - DragAndDrop first channel inside the group.
                - DragAndDrop 2nd channel inside the group.
            - Case 3:
                - Create ChannelGroup
                - DragAndDrop last two channels inside the group.
            - Case 4:
                - Drag Group inside the Group
            - Case 5:
                - Drag Group outside the Group
        Evaluate:
            - Case 0:
                - Evaluate that channel changed the position in the tree.
            - Case 1:
                - Evaluate that channels changed their position in the tree.
            - Case 2:
                - Evaluate that channel group was created. New item is present on the tree.
                - Evaluate that channel was moved as child to this item. Item should no longer exist outside the group.
                - Evaluate that duplicated channel was moved inside the group, and there is still one more wo parent
                 on the tree
            - Case 3:
                - Evaluate that channels were moved inside the group and does no longer exist outside.
            - Case 4:
                - Evaluate that group was moved inside the group.
            - Case 5:
                - Evaluate that group was moved outside the group.
        """
        # Event
        self.widget = FileWidget(
            self.measurement_file,
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

        # Select 5 from channels but not from beginning
        # Avoid $ChannelLog because it's empty channel
        iterator = QtWidgets.QTreeWidgetItemIterator(channel_tree)
        count = -1
        selected_channels = []
        item = None
        while iterator.value():
            count += 1
            # Skip over first 3
            if count < 2:
                iterator += 1
                continue
            item = iterator.value()
            if item and count < 7:
                item.setSelected(True)
                item.setCheckState(0, QtCore.Qt.Checked)
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
        self.assertEqual(5, plot.channel_selection.topLevelItemCount())

        # Add one channel twice
        # Switch ComboBox to "Selected channels only"
        self.widget.channel_view.setCurrentText("Selected channels only")
        duplicated_channel = channel_tree.topLevelItem(0)
        duplicated_channel.setSelected(True)
        drag_position = channel_tree.visualItemRect(duplicated_channel).center()
        duplicated_channel = duplicated_channel.text(0)
        DragAndDrop(
            source_widget=channel_tree,
            destination_widget=plot,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        self.assertEqual(6, plot.channel_selection.topLevelItemCount())

        # Case 0:
        with self.subTest(
            "test_test_Plot_ChannelSelection_DragAndDrop_fromPlotCS_toPlot_0"
        ):
            # DragAndDrop first channel to 3rd position.
            first_channel = plot.channel_selection.topLevelItem(0)
            third_channel = plot.channel_selection.topLevelItem(2)
            # Get Positions
            drag_position = plot.channel_selection.visualItemRect(
                first_channel
            ).center()
            drop_position = plot.channel_selection.visualItemRect(
                third_channel
            ).center()
            # Get Names
            first_channel = first_channel.text(0)
            third_channel = third_channel.text(0)
            DragAndDrop(
                source_widget=plot.channel_selection,
                destination_widget=plot.channel_selection.viewport(),
                source_pos=drag_position,
                destination_pos=drop_position,
            )
            # Evaluate
            # First channel position was changed.
            new_first_channel = plot.channel_selection.topLevelItem(0).text(0)
            self.assertNotEqual(first_channel, new_first_channel)
            # Evaluate that first channel was moved to third channel position.
            new_third_channel = plot.channel_selection.topLevelItem(2).text(0)
            new_fourth_channel = plot.channel_selection.topLevelItem(1).text(0)
            self.assertEqual(first_channel, new_third_channel)
            self.assertEqual(third_channel, new_fourth_channel)

        # Case 1:
        with self.subTest(
            "test_test_Plot_ChannelSelection_DragAndDrop_fromPlotCS_toPlot_1"
        ):
            # DragAndDrop 2nd and 3rd channels on last position.
            second_channel = plot.channel_selection.topLevelItem(1)
            third_channel = plot.channel_selection.topLevelItem(2)
            last_channel = plot.channel_selection.topLevelItem(5)
            # Get Positions
            drag_position = plot.channel_selection.visualItemRect(
                third_channel
            ).center()
            drop_position = plot.channel_selection.visualItemRect(last_channel).center()
            # Select
            second_channel.setSelected(True)
            third_channel.setSelected(True)
            # Get Names
            second_channel = second_channel.text(0)
            third_channel = third_channel.text(0)
            DragAndDrop(
                source_widget=plot.channel_selection,
                destination_widget=plot.channel_selection.viewport(),
                source_pos=drag_position,
                destination_pos=drop_position,
            )
            # Evaluate
            new_second_channel = plot.channel_selection.topLevelItem(1).text(0)
            new_third_channel = plot.channel_selection.topLevelItem(2).text(0)
            self.assertNotEqual(second_channel, new_second_channel)
            self.assertNotEqual(third_channel, new_third_channel)
            new_fifth_channel = plot.channel_selection.topLevelItem(4).text(0)
            new_sixth_channel = plot.channel_selection.topLevelItem(5).text(0)
            self.assertEqual(second_channel, new_fifth_channel)
            self.assertEqual(third_channel, new_sixth_channel)

        # Case 2:
        with self.subTest(
            "test_test_Plot_ChannelSelection_DragAndDrop_fromPlotCS_toPlot_2"
        ):
            # Create Channel Group. Drag channels inside the group one by one
            with mock.patch(
                "asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText"
            ) as mc_getText:
                # Create Channel Group
                mc_getText.return_value = "FirstGroup", True
                QtTest.QTest.keySequence(
                    plot.channel_selection, QtGui.QKeySequence("Shift+Insert")
                )
                # PreEvaluation: Check if there is one extra-item
                self.assertEqual(7, plot.channel_selection.topLevelItemCount())
                # Get Group Position
                first_group = None
                for index in range(plot.channel_selection.topLevelItemCount()):
                    item = plot.channel_selection.topLevelItem(index)
                    if item.text(0) == "FirstGroup":
                        first_group = item
                        break
                else:
                    self.fail("FirstGroup is not present on Plot Channel Selection.")
                first_group.setExpanded(True)
                # Get First Item that will be moved
                first_channel = plot.channel_selection.topLevelItem(1)
                if first_channel.text(0) == duplicated_channel:
                    first_channel = plot.channel_selection.topLevelItem(2)

                drag_position = plot.channel_selection.visualItemRect(
                    first_channel
                ).center()
                drop_position = plot.channel_selection.visualItemRect(
                    first_group
                ).center()
                # Get Name of first channel
                first_channel = first_channel.text(0)
                # PreEvaluation: Ensure that group has no child
                self.assertEqual(0, first_group.childCount())
                DragAndDrop(
                    source_widget=plot.channel_selection,
                    destination_widget=plot.channel_selection.viewport(),
                    source_pos=drag_position,
                    destination_pos=drop_position,
                )
                # Evaluate
                self.assertEqual(1, first_group.childCount())
                self.assertEqual(first_channel, first_group.child(0).text(0))
                self.assertEqual(6, plot.channel_selection.topLevelItemCount())

                second_channel = None
                for index in range(plot.channel_selection.topLevelItemCount()):
                    item = plot.channel_selection.topLevelItem(index)
                    if item.text(0) == duplicated_channel:
                        second_channel = item
                        break
                else:
                    self.fail("Duplicate Channel is not found anymore.")
                drag_position = plot.channel_selection.visualItemRect(
                    second_channel
                ).center()
                # Now drop over the first item from group.
                drop_position = plot.channel_selection.visualItemRect(
                    first_group.child(0)
                ).center()
                DragAndDrop(
                    source_widget=plot.channel_selection,
                    destination_widget=plot.channel_selection.viewport(),
                    source_pos=drag_position,
                    destination_pos=drop_position,
                )
                # Evaluate
                self.assertEqual(2, first_group.childCount())
                self.assertEqual(duplicated_channel, first_group.child(1).text(0))
                self.assertEqual(5, plot.channel_selection.topLevelItemCount())

        # Case 3:
        with self.subTest(
            "test_test_Plot_ChannelSelection_DragAndDrop_fromPlotCS_toPlot_3"
        ):
            # Create Channel Group. Drag multiple channels inside the group
            with mock.patch(
                "asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText"
            ) as mc_getText:
                # Create Channel Group
                mc_getText.return_value = "SecondGroup", True
                QtTest.QTest.keySequence(
                    plot.channel_selection, QtGui.QKeySequence("Shift+Insert")
                )
                # PreEvaluation: Check if there is one extra-item
                self.assertEqual(6, plot.channel_selection.topLevelItemCount())
                # Get Group Position
                second_group = None
                for index in range(plot.channel_selection.topLevelItemCount()):
                    item = plot.channel_selection.topLevelItem(index)
                    if item.text(0) == "SecondGroup":
                        second_group = item
                        break
                else:
                    self.fail("SecondGroup is not present on Plot Channel Selection.")
                second_group.setExpanded(True)
                # Get Channels
                last_channel_0 = plot.channel_selection.topLevelItem(
                    plot.channel_selection.topLevelItemCount() - 1
                )
                last_channel_1 = plot.channel_selection.topLevelItem(
                    plot.channel_selection.topLevelItemCount() - 2
                )
                last_channel_0.setSelected(True)
                last_channel_1.setSelected(True)
                drag_position = plot.channel_selection.visualItemRect(
                    last_channel_1
                ).center()
                drop_position = plot.channel_selection.visualItemRect(
                    second_group
                ).center()
                DragAndDrop(
                    source_widget=plot.channel_selection,
                    destination_widget=plot.channel_selection.viewport(),
                    source_pos=drag_position,
                    destination_pos=drop_position,
                )
                # Evaluate
                self.assertEqual(2, second_group.childCount())
                self.assertEqual(4, plot.channel_selection.topLevelItemCount())

        # Case 4:
        with self.subTest(
            "test_test_Plot_ChannelSelection_DragAndDrop_fromPlotCS_toPlot_4"
        ):
            # Drag Group inside the Group
            # Get Group Positions
            first_group, second_group = None, None
            for index in range(plot.channel_selection.topLevelItemCount()):
                item = plot.channel_selection.topLevelItem(index)
                if item.text(0) == "FirstGroup":
                    first_group = item
                elif item.text(0) == "SecondGroup":
                    second_group = item
                if first_group and second_group:
                    break
            else:
                self.fail("Groups are not present on Plot Channel Selection.")
            drag_position = plot.channel_selection.visualItemRect(second_group).center()
            drop_position = plot.channel_selection.visualItemRect(first_group).center()
            DragAndDrop(
                source_widget=plot.channel_selection,
                destination_widget=plot.channel_selection.viewport(),
                source_pos=drag_position,
                destination_pos=drop_position,
            )
            # Evaluate
            self.assertEqual(3, first_group.childCount())
            self.assertEqual(3, plot.channel_selection.topLevelItemCount())

        # Case 5:
        with self.subTest(
            "test_test_Plot_ChannelSelection_DragAndDrop_fromPlotCS_toPlot_5"
        ):
            # Drag Group outside the Group
            # Get Group Positions
            first_group, second_group = None, None
            for index in range(plot.channel_selection.topLevelItemCount()):
                item = plot.channel_selection.topLevelItem(index)
                if item.text(0) == "FirstGroup":
                    first_group = item
                    break
            else:
                self.fail("FirstGroup is not present on Plot Channel Selection.")
            for index in range(first_group.childCount()):
                item = first_group.child(index)
                if item.text(0) == "SecondGroup":
                    second_group = item
                    break
            else:
                self.fail("SecondGroup is not present on Plot Channel Selection.")
            drag_position = plot.channel_selection.visualItemRect(second_group).center()
            drop_position = plot.channel_selection.rect().center()
            DragAndDrop(
                source_widget=plot.channel_selection,
                destination_widget=plot.channel_selection.viewport(),
                source_pos=drag_position,
                destination_pos=drop_position,
            )
            # Evaluate
            self.assertEqual(2, first_group.childCount())
            self.assertEqual(4, plot.channel_selection.topLevelItemCount())

    def test_Plot_ChannelSelection_DragAndDrop_fromPlot_toPlot(self):
        """
        Test Scope: Validate that channels can be dragged and dropped between Plot windows. (Ex: from Plot 0 to Plot 1)
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Select one channel and drag it to the 'Plot 0'
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Select one channel and drag it to the 'Plot 1'
            - Select channel from 'Plot 0' and drag it to the 'Plot 1'
        Evaluate:
            - Validate that channel from 'Plot 0' is added to 'Plot 1'
        """
        # Event
        self.widget = FileWidget(
            self.measurement_file,
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

        # Create New Plot Window
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
        plot_0 = self.widget.mdi_area.subWindowList()[0].widget()
        # Random Channels
        channel_0 = channel_tree.topLevelItem(8)
        channel_1 = channel_tree.topLevelItem(13)

        # Drag one Channel from FileWidget channel_tree to Plot_0
        drag_position = channel_tree.visualItemRect(channel_0).center()
        drop_position = plot_0.channel_selection.viewport().rect().center()

        # PreEvaluation
        self.assertEqual(0, plot_0.channel_selection.topLevelItemCount())
        DragAndDrop(
            source_widget=channel_tree,
            destination_widget=plot_0,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        self.assertEqual(1, plot_0.channel_selection.topLevelItemCount())

        # Create New Plot Window
        with mock.patch(
            "asammdf.gui.widgets.file.WindowSelectionDialog"
        ) as mc_WindowSelectionDialog:
            mc_WindowSelectionDialog.return_value.result.return_value = True
            mc_WindowSelectionDialog.return_value.selected_type.return_value = "Plot"
            # - Press PushButton "Create Window"
            QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.LeftButton)
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)

        plot_1 = self.widget.mdi_area.subWindowList()[1].widget()

        # Drag one Channel from FileWidget channel_tree to Plot_0
        drag_position = channel_tree.visualItemRect(channel_1).center()
        drop_position = plot_1.channel_selection.viewport().rect().center()

        # PreEvaluation
        self.assertEqual(0, plot_1.channel_selection.topLevelItemCount())
        DragAndDrop(
            source_widget=channel_tree,
            destination_widget=plot_1,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        self.assertEqual(1, plot_1.channel_selection.topLevelItemCount())

        # Select channel from 'Plot 0' and drag it to the 'Plot 1'
        # Drag one Channel from FileWidget channel_tree to Plot_0
        plot_0_channel = plot_0.channel_selection.topLevelItem(0)
        plot_0_channel_name = plot_0_channel.text(0)
        drag_position = plot_0.channel_selection.visualItemRect(plot_0_channel).center()
        drop_position = plot_1.channel_selection.viewport().rect().center()

        # PreEvaluation
        self.assertEqual(1, plot_1.channel_selection.topLevelItemCount())
        DragAndDrop(
            source_widget=plot_0.channel_selection,
            destination_widget=plot_1,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        # Evaluate
        plot_1_new_channel = plot_1.channel_selection.topLevelItem(1)
        plot_1_new_channel_name = plot_1_new_channel.text(0)
        self.assertEqual(2, plot_1.channel_selection.topLevelItemCount())
        self.assertEqual(plot_0_channel_name, plot_1_new_channel_name)

    def test_Plot_ChannelSelection_DragAndDrop_fromNumeric_toPlot(self):
        """
        Test Scope: Validate that channels can be dragged and dropped between Plot window and Numeric.
        Ex: from 'Numeric 0' to 'Plot 0'
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Select one channel and drag it to the 'Plot 0'
            - Press PushButton "Create Window"
                - Simulate that Numeric window is selected as window type.
            - Select one channel and drag it to the 'Numeric 0'
            - Select channel from 'Numeric 0' and drag it to the 'Plot 0'
        Evaluate:
            - Validate that channel from 'Numeric 0' is added to 'Plot 0'
        """
        # Event
        self.widget = FileWidget(
            self.measurement_file,
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

        # Create New Plot Window
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
        plot_0 = self.widget.mdi_area.subWindowList()[0].widget()
        # Random Channels
        channel_0 = channel_tree.topLevelItem(8)
        channel_1 = channel_tree.topLevelItem(0)

        # Drag one Channel from FileWidget channel_tree to Plot_0
        drag_position = channel_tree.visualItemRect(channel_0).center()
        drop_position = plot_0.channel_selection.viewport().rect().center()

        # PreEvaluation
        self.assertEqual(0, plot_0.channel_selection.topLevelItemCount())
        DragAndDrop(
            source_widget=channel_tree,
            destination_widget=plot_0,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        self.assertEqual(1, plot_0.channel_selection.topLevelItemCount())

        # Create New Numeric Window
        with mock.patch(
            "asammdf.gui.widgets.file.WindowSelectionDialog"
        ) as mc_WindowSelectionDialog:
            mc_WindowSelectionDialog.return_value.result.return_value = True
            mc_WindowSelectionDialog.return_value.selected_type.return_value = "Numeric"
            # - Press PushButton "Create Window"
            QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.LeftButton)
            # Evaluate
            self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
            widget_types = sorted(
                map(
                    lambda w: w.widget().__class__.__name__,
                    self.widget.mdi_area.subWindowList(),
                )
            )
            self.assertIn("Numeric", widget_types)

        numeric = self.widget.mdi_area.subWindowList()[1].widget()

        # Drag one Channel from FileWidget channel_tree to Numeric_0
        drag_position = channel_tree.visualItemRect(channel_1).center()
        drop_position = numeric.channels.dataView.viewport().rect().center()

        DragAndDrop(
            source_widget=channel_tree,
            destination_widget=numeric.channels.dataView,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        # Evaluate
        numeric_data = numeric.channels.dataView
        numeric_channel = numeric_data.model().data(numeric_data.model().index(0, 0))
        self.assertEqual(channel_1.text(0), numeric_channel)

        # Drag one Channel from Numeric_0 to Plot_0
        index = numeric_data.model().index(0, 0)
        drag_position = numeric_data.visualRect(index).center()
        drop_position = plot_0.channel_selection.viewport().rect().center()

        DragAndDrop(
            source_widget=numeric_data,
            destination_widget=plot_0.channel_selection,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        # Evaluate
        plot_0_new_channel = plot_0.channel_selection.topLevelItem(1)
        plot_0_new_channel_name = plot_0_new_channel.text(0)
        self.assertEqual(2, plot_0.channel_selection.topLevelItemCount())
        self.assertEqual(numeric_channel, plot_0_new_channel_name)


class TestShortcuts(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.measurement_file = str(pathlib.Path(cls.resource, "ASAP2_Demo_V171.mf4"))

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

    def test_Plot_Plot_Shortcut_Key_LeftRight(self):
        """
        Test Scope:
            Check that Arrow Keys: Left & Right ensure navigation on channels evolution.
            Ensure that navigation is working.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
            - Drag and Drop channels from FileWidget.channels_tree to Plot.channels_selection:
                # First
                - ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
                # Second
                - ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            - Send KeyClick Right 5 times
            - Send KeyClick Left 4 times
        Evaluate:
            - Evaluate values from `Value` column on Plot.channels_selection
            - Evaluate timestamp label
        """
        # Event
        self.widget = FileWidget(
            self.measurement_file,
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

        channels_tree = self.widget.channels_tree
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        channel_selection = plot.channel_selection

        # Select 5 from channels but not from beginning
        # Avoid $ChannelLog because it's empty channel
        iterator = QtWidgets.QTreeWidgetItemIterator(channels_tree)
        item = None
        while iterator.value():
            item = iterator.value()
            if item and item.text(0) in (
                "ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL",
                "ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL",
            ):
                item.setCheckState(0, QtCore.Qt.Checked)
                item.setSelected(True)
            iterator += 1

        drag_position = channels_tree.visualItemRect(item).center()
        drop_position = plot.channel_selection.viewport().rect().center()

        # PreEvaluation
        self.assertEqual(0, plot.channel_selection.topLevelItemCount())
        DragAndDrop(
            source_widget=channels_tree,
            destination_widget=plot,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        self.assertEqual(2, plot.channel_selection.topLevelItemCount())

        # Identify channels
        # First sample = 244 for ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
        # First sample = 23 for ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
        channel_14, channel_15 = None, None
        for _ in range(channel_selection.topLevelItemCount()):
            item = channel_selection.topLevelItem(_)
            if item.text(0) == "ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL":
                channel_14 = item
            else:
                channel_15 = item

        # Case 0:
        with self.subTest("test_Plot_Plot_Shortcut_Key_LeftRight_0"):
            # Select channel: ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            QtTest.QTest.mouseClick(
                channel_selection.viewport(),
                QtCore.Qt.LeftButton,
                QtCore.Qt.KeyboardModifiers(),
                channel_selection.visualItemRect(channel_15).center(),
            )
            plot.plot.setFocus()
            self.processEvents(0.1)

            self.assertEqual("25", channel_14.text(1))
            self.assertEqual("244", channel_15.text(1))

            # Send Key strokes
            for count in range(6):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("8", channel_14.text(1))
            self.assertEqual("6", channel_15.text(1))
            self.assertEqual("t = 0.082657s", plot.cursor_info.text())

            # Send Key strokes
            for count in range(5):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Left)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("21", channel_14.text(1))
            self.assertEqual("247", channel_15.text(1))
            self.assertEqual("t = 0.032657s", plot.cursor_info.text())

        # Case 1:
        with self.subTest("test_Plot_Plot_Shortcut_Key_LeftRight_1"):
            # Select channel: ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            QtTest.QTest.mouseClick(
                channel_selection.viewport(),
                QtCore.Qt.LeftButton,
                QtCore.Qt.KeyboardModifiers(),
                channel_selection.visualItemRect(channel_15).center(),
            )
            plot.plot.setFocus()
            self.processEvents(0.1)

            # Send Key strokes
            for count in range(6):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("5", channel_14.text(1))
            self.assertEqual("9", channel_15.text(1))
            self.assertEqual("t = 0.092657s", plot.cursor_info.text())

            # Send Key strokes
            for count in range(5):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Left)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("18", channel_14.text(1))
            self.assertEqual("250", channel_15.text(1))
            self.assertEqual("t = 0.042657s", plot.cursor_info.text())

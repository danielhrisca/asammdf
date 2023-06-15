#!/usr/bin/env python
import sys
from test.asammdf.gui.test_base import DragAndDrop
from test.asammdf.gui.widgets.test_BasePlotWidget import Pixmap, TestPlotWidget
import unittest
from unittest import mock

from PySide6 import QtCore, QtGui, QtTest, QtWidgets


class TestDoubleClick(TestPlotWidget):
    # Note: Test Plot Widget through FileWidget.

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
        # Open File Widget
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Create plot window
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        # Drag and Drop channel from FileWidget.channel_tree to Plot
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        plot_channel = self.add_channel_to_plot(plot=plot)

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


class TestDragAndDrop(TestPlotWidget):
    # Note: Test Plot Widget through FileWidget.

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
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)

        # Select channel
        channel_tree = self.widget.channels_tree
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        item = channel_tree.topLevelItem(0)
        channel_name = item.text(self.Column.NAME)
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
        plot_channel_name = plot_channel.text(self.Column.NAME)
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
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)

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
            plot_channels.append(item.text(self.Column.NAME))
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
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)

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
            first_channel = first_channel.text(self.Column.NAME)
            third_channel = third_channel.text(self.Column.NAME)
            DragAndDrop(
                source_widget=plot.channel_selection,
                destination_widget=plot.channel_selection.viewport(),
                source_pos=drag_position,
                destination_pos=drop_position,
            )
            # Evaluate
            # First channel position was changed.
            new_first_channel = plot.channel_selection.topLevelItem(0).text(
                self.Column.NAME
            )
            self.assertNotEqual(first_channel, new_first_channel)
            # Evaluate that first channel was moved to third channel position.
            new_third_channel = plot.channel_selection.topLevelItem(2).text(
                self.Column.NAME
            )
            new_fourth_channel = plot.channel_selection.topLevelItem(1).text(
                self.Column.NAME
            )
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
            second_channel = second_channel.text(self.Column.NAME)
            third_channel = third_channel.text(self.Column.NAME)
            DragAndDrop(
                source_widget=plot.channel_selection,
                destination_widget=plot.channel_selection.viewport(),
                source_pos=drag_position,
                destination_pos=drop_position,
            )
            # Evaluate
            new_second_channel = plot.channel_selection.topLevelItem(1).text(
                self.Column.NAME
            )
            new_third_channel = plot.channel_selection.topLevelItem(2).text(
                self.Column.NAME
            )
            self.assertNotEqual(second_channel, new_second_channel)
            self.assertNotEqual(third_channel, new_third_channel)
            new_fifth_channel = plot.channel_selection.topLevelItem(4).text(
                self.Column.NAME
            )
            new_sixth_channel = plot.channel_selection.topLevelItem(5).text(
                self.Column.NAME
            )
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
                    if item.text(self.Column.NAME) == "FirstGroup":
                        first_group = item
                        break
                else:
                    self.fail("FirstGroup is not present on Plot Channel Selection.")
                first_group.setExpanded(True)
                # Get First Item that will be moved
                first_channel = plot.channel_selection.topLevelItem(1)
                if first_channel.text(self.Column.NAME) == duplicated_channel:
                    first_channel = plot.channel_selection.topLevelItem(2)

                drag_position = plot.channel_selection.visualItemRect(
                    first_channel
                ).center()
                drop_position = plot.channel_selection.visualItemRect(
                    first_group
                ).center()
                # Get Name of first channel
                first_channel = first_channel.text(self.Column.NAME)
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
                self.assertEqual(
                    first_channel, first_group.child(0).text(self.Column.NAME)
                )
                self.assertEqual(6, plot.channel_selection.topLevelItemCount())

                second_channel = None
                for index in range(plot.channel_selection.topLevelItemCount()):
                    item = plot.channel_selection.topLevelItem(index)
                    if item.text(self.Column.NAME) == duplicated_channel:
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
                self.assertEqual(
                    duplicated_channel, first_group.child(1).text(self.Column.NAME)
                )
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
                    if item.text(self.Column.NAME) == "SecondGroup":
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
                if item.text(self.Column.NAME) == "FirstGroup":
                    first_group = item
                elif item.text(self.Column.NAME) == "SecondGroup":
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
                if item.text(self.Column.NAME) == "FirstGroup":
                    first_group = item
                    break
            else:
                self.fail("FirstGroup is not present on Plot Channel Selection.")
            for index in range(first_group.childCount()):
                item = first_group.child(index)
                if item.text(self.Column.NAME) == "SecondGroup":
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
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        # Create New Plot Window
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)

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
        self.create_window(window_type="Plot")
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
        plot_0_channel_name = plot_0_channel.text(self.Column.NAME)
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
        plot_1_new_channel_name = plot_1_new_channel.text(self.Column.NAME)
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
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        # Create New Plot Window
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)

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
        self.create_window(window_type="Numeric")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)

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
        self.assertEqual(channel_1.text(self.Column.NAME), numeric_channel)

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
        plot_0_new_channel_name = plot_0_new_channel.text(self.Column.NAME)
        self.assertEqual(2, plot_0.channel_selection.topLevelItemCount())
        self.assertEqual(numeric_channel, plot_0_new_channel_name)


class TestPushButtons(TestPlotWidget):
    def test_Plot_ChannelSelection_PushButton_ValuePanel(self):
        """
        Test Scope:
            Check that Value Panel label is visible or hidden according to Push Button
            Check that Value Panel label is updated according signal samples
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
            - Drag and Drop 2 channels from FileWidget.channels_tree to Plot.channels_selection
            - Press PushButton "Show selected channel value panel"
            - Press PushButton "Hide selected channel value panel"
            - Press PushButton "Show selected channel value panel"
            - Select one channel
            - Use navigation keys to change values
            - Select 2nd channel
            - Use navigation keys to change values
        Evaluate:
            - Evaluate that label is visible when button "Show selected channel value panel" is pressed.
            - Evaluate that label is hidden when button "Hide selected channel value panel" is pressed.
            - Evaluate that value from label is equal with: Channel Value + Channel Unit
            - Evaluate that value is updated according channel selected and current value
        """
        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Press PushButton "Create Window"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        # Add Channels to Plot
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        plot_channel_0 = self.add_channel_to_plot(
            plot=plot, channel_name="ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL"
        )
        self.assertEqual(1, plot.channel_selection.topLevelItemCount())
        plot_channel_1 = self.add_channel_to_plot(
            plot=plot, channel_name="ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL"
        )
        self.assertEqual(2, plot.channel_selection.topLevelItemCount())
        if plot.selected_channel_value.isVisible():
            # Press PushButton "Hide selected channel value panel"
            QtTest.QTest.mouseClick(
                plot.selected_channel_value_btn, QtCore.Qt.LeftButton
            )
        # Press PushButton "Show selected channel value panel"
        QtTest.QTest.mouseClick(plot.selected_channel_value_btn, QtCore.Qt.LeftButton)
        self.assertTrue(plot.selected_channel_value.isVisible())

        # Press PushButton "Hide selected channel value panel"
        QtTest.QTest.mouseClick(plot.selected_channel_value_btn, QtCore.Qt.LeftButton)
        self.assertFalse(plot.selected_channel_value.isVisible())
        # Press PushButton "Show selected channel value panel"
        QtTest.QTest.mouseClick(plot.selected_channel_value_btn, QtCore.Qt.LeftButton)
        self.assertTrue(plot.selected_channel_value.isVisible())

        # Select Channel
        QtTest.QTest.mouseClick(
            plot.channel_selection.viewport(),
            QtCore.Qt.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            plot.channel_selection.visualItemRect(plot_channel_0).center(),
        )

        # Evaluate
        plot_channel_0_value = plot_channel_0.text(self.Column.VALUE)
        plot_channel_0_unit = plot_channel_0.text(self.Column.UNIT)
        self.assertEqual(
            f"{plot_channel_0_value} {plot_channel_0_unit}",
            plot.selected_channel_value.text(),
        )

        # Event
        plot.plot.setFocus()
        self.processEvents(0.1)
        # Send Key strokes
        for _ in range(6):
            QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
            self.processEvents(0.1)
        self.processEvents(0.1)

        # Evaluate
        plot_channel_0_value = plot_channel_0.text(self.Column.VALUE)
        plot_channel_0_unit = plot_channel_0.text(self.Column.UNIT)
        self.assertEqual(
            f"{plot_channel_0_value} {plot_channel_0_unit}",
            plot.selected_channel_value.text(),
        )

        # Select 2nd Channel
        QtTest.QTest.mouseClick(
            plot.channel_selection.viewport(),
            QtCore.Qt.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            plot.channel_selection.visualItemRect(plot_channel_1).center(),
        )

        # Evaluate
        plot_channel_1_value = plot_channel_1.text(self.Column.VALUE)
        plot_channel_1_unit = plot_channel_1.text(self.Column.UNIT)
        self.assertEqual(
            f"{plot_channel_1_value} {plot_channel_1_unit}",
            plot.selected_channel_value.text(),
        )

        # Event
        plot.plot.setFocus()
        self.processEvents(0.1)
        # Send Key strokes
        for _ in range(6):
            QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
            self.processEvents(0.1)
        self.processEvents(0.1)

        # Evaluate
        plot_channel_1_value = plot_channel_1.text(self.Column.VALUE)
        plot_channel_1_unit = plot_channel_1.text(self.Column.UNIT)
        self.assertEqual(
            f"{plot_channel_1_value} {plot_channel_1_unit}",
            plot.selected_channel_value.text(),
        )

    def test_Plot_ChannelSelection_PushButton_FocusedMode(self):
        """
        Test Scope:
            Check if Plot is cleared when no channel is selected.
            Check if Plot is showing all channels when Focus Mode is disabled.
            Check if Plot is showing only one channel when Focus Mode is enabled.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
            - Press PushButton HideAxis (easy for evaluation)
            - Drag and Drop channels from FileWidget.channels_tree to Plot.channels_selection:
                # First
                - ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
                # Second
                - ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            - Press PushButton FocusMode
            - Press PushButton FocusMode
        Evaluate:
            - Evaluate that channels are displayed when FocusMode is disabled.
            - Evaluate that selected channels is displayed when FocusMode is enabled.
        """
        channel_0 = "ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL"
        channel_1 = "ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL"

        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Press PushButton "Create Window"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Press PushButton "Hide axis"
        if not plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(plot.hide_axes_btn, QtCore.Qt.LeftButton)

        # Save PixMap of clear plot
        clear_pixmap = plot.plot.viewport().grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

        # Add Channels to Plot
        plot_tree_channel_0 = self.add_channel_to_plot(
            plot=plot, channel_name=channel_0
        )
        self.assertEqual(1, plot.channel_selection.topLevelItemCount())
        plot_tree_channel_1 = self.add_channel_to_plot(
            plot=plot, channel_name=channel_1
        )
        self.assertEqual(2, plot.channel_selection.topLevelItemCount())

        # Identify PlotSignal
        plot_graph_channel_0, plot_graph_channel_1 = None, None
        for channel in plot.plot.signals:
            if channel.name == channel_0:
                plot_graph_channel_0 = channel
            elif channel.name == channel_1:
                plot_graph_channel_1 = channel

        if not plot.focused_mode_btn.isFlat():
            QtTest.QTest.mouseClick(plot.focused_mode_btn, QtCore.Qt.LeftButton)

        channels_present_pixmap = plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap=channels_present_pixmap))
        self.assertTrue(
            Pixmap.has_color(
                pixmap=channels_present_pixmap,
                color_name=plot_graph_channel_0.color_name,
            )
        )
        self.assertTrue(
            Pixmap.has_color(
                pixmap=channels_present_pixmap,
                color_name=plot_graph_channel_1.color_name,
            )
        )

        # Press Button Focus Mode
        QtTest.QTest.mouseClick(plot.focused_mode_btn, QtCore.Qt.LeftButton)

        # Evaluate
        focus_mode_clear_pixmap = plot.plot.viewport().grab()
        # No Channel is selected
        self.assertTrue(Pixmap.is_black(pixmap=focus_mode_clear_pixmap))

        # Select 2nd Channel
        QtTest.QTest.mouseClick(
            plot.channel_selection.viewport(),
            QtCore.Qt.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            plot.channel_selection.visualItemRect(plot_tree_channel_1).center(),
        )
        # Process flash until signal is present on plot.
        for _ in range(10):
            self.processEvents(timeout=0.01)
            focus_mode_channel_1_pixmap = plot.plot.viewport().grab()
            if Pixmap.has_color(
                pixmap=focus_mode_channel_1_pixmap,
                color_name=plot_graph_channel_1.color_name,
            ):
                break

        # Evaluate
        focus_mode_channel_1_pixmap = plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap=focus_mode_channel_1_pixmap))
        self.assertFalse(
            Pixmap.has_color(
                pixmap=focus_mode_channel_1_pixmap,
                color_name=plot_graph_channel_0.color_name,
            )
        )
        self.assertTrue(
            Pixmap.has_color(
                pixmap=focus_mode_channel_1_pixmap,
                color_name=plot_graph_channel_1.color_name,
            )
        )

    def test_Plot_ChannelSelection_PushButton_RegionDelta(self):
        """
        Test Scope:
            Check if computation is done when Region is selected.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
            - Press PushButton HideAxis (easy for evaluation)
            - Drag and Drop channels from FileWidget.channels_tree to Plot.channels_selection:
                # First
                - ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
                # Second
                - ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            - Press PushButton Delta (range is not active)
            - Press Key 'R' for range selection
            - Press PushButton Delta (range is active)
            - Move cursors
            - Press Key 'R' for range selection
            - Press PushButton Delta (range is not active)
        Evaluate:
            - Evaluate that 'delta' char is present on channel values when range selection is active
            - Evaluate that 'delta' is correctly computed and result is updated on channel value
        """
        channel_0 = "ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL"
        channel_1 = "ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL"

        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Press PushButton "Create Window"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Press PushButton "Hide axis"
        if not plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(plot.hide_axes_btn, QtCore.Qt.LeftButton)

        # Add Channels to Plot
        plot_tree_channel_0 = self.add_channel_to_plot(
            plot=plot, channel_name=channel_0
        )
        self.assertEqual(1, plot.channel_selection.topLevelItemCount())
        plot_tree_channel_1 = self.add_channel_to_plot(
            plot=plot, channel_name=channel_1
        )
        self.assertEqual(2, plot.channel_selection.topLevelItemCount())

        if not plot.delta_btn.isFlat():
            QtTest.QTest.mouseClick(plot.delta_btn, QtCore.Qt.LeftButton)
            self.processEvents()

        # Ensure that delta char is not present on channel values
        self.assertNotIn("Δ", plot_tree_channel_0.text(self.Column.VALUE))
        self.assertNotIn("Δ", plot_tree_channel_1.text(self.Column.VALUE))

        # Press PushButton Delta (range is not active)
        QtTest.QTest.mouseClick(plot.delta_btn, QtCore.Qt.LeftButton)
        self.processEvents()

        # Ensure that delta char is not present on channel values even if button is pressed
        self.assertNotIn("Δ", plot_tree_channel_0.text(self.Column.VALUE))
        self.assertNotIn("Δ", plot_tree_channel_1.text(self.Column.VALUE))

        # Press Key 'R' for range selection
        QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_R)
        self.processEvents(timeout=0.01)

        # Ensure that delta char is present on channel values even if button is pressed
        self.assertIn("Δ", plot_tree_channel_0.text(self.Column.VALUE))
        self.assertIn("Δ", plot_tree_channel_1.text(self.Column.VALUE))

        # Move cursor
        # Select channel_1
        QtTest.QTest.mouseClick(
            plot.channel_selection.viewport(),
            QtCore.Qt.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            plot.channel_selection.visualItemRect(plot_tree_channel_1).center(),
        )
        plot.plot.setFocus()
        self.processEvents(0.1)
        # Move a little bit in center of measurement
        for _ in range(15):
            QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
            self.processEvents(timeout=0.1)

        # Get current value: Ex: 'Δ = 8'. Get last number
        old_channel_0_value = int(
            plot_tree_channel_0.text(self.Column.VALUE).split(" ")[-1]
        )
        old_channel_1_value = int(
            plot_tree_channel_1.text(self.Column.VALUE).split(" ")[-1]
        )
        for count in range(5):
            QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
            self.processEvents(timeout=0.1)
            # Evaluate
            channel_0_value = int(
                plot_tree_channel_0.text(self.Column.VALUE).split(" ")[-1]
            )
            channel_1_value = int(
                plot_tree_channel_1.text(self.Column.VALUE).split(" ")[-1]
            )
            self.assertLess(old_channel_0_value, channel_0_value)
            self.assertGreater(old_channel_1_value, channel_1_value)
            old_channel_0_value = channel_0_value
            old_channel_1_value = channel_1_value

        # Press Key 'R' for range selection
        QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_R)
        self.processEvents(timeout=0.01)

        # Ensure that delta char is not present on channel values even if button is pressed
        self.assertNotIn("Δ", plot_tree_channel_0.text(self.Column.VALUE))
        self.assertNotIn("Δ", plot_tree_channel_1.text(self.Column.VALUE))


class TestShortcuts(TestPlotWidget):
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
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)

        plot = self.widget.mdi_area.subWindowList()[0].widget()
        channel_selection = plot.channel_selection
        channel_14 = self.add_channel_to_plot(
            plot=plot, channel_name="ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL"
        )
        channel_15 = self.add_channel_to_plot(
            plot=plot, channel_name="ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL"
        )
        self.assertEqual(2, plot.channel_selection.topLevelItemCount())

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

            self.assertEqual("25", channel_14.text(self.Column.VALUE))
            self.assertEqual("244", channel_15.text(self.Column.VALUE))

            # Send Key strokes
            for _ in range(6):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("8", channel_14.text(self.Column.VALUE))
            self.assertEqual("6", channel_15.text(self.Column.VALUE))
            self.assertEqual("t = 0.082657s", plot.cursor_info.text())

            # Send Key strokes
            for _ in range(5):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Left)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("21", channel_14.text(self.Column.VALUE))
            self.assertEqual("247", channel_15.text(self.Column.VALUE))
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
            for _ in range(6):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("5", channel_14.text(self.Column.VALUE))
            self.assertEqual("9", channel_15.text(self.Column.VALUE))
            self.assertEqual("t = 0.092657s", plot.cursor_info.text())

            # Send Key strokes
            for _ in range(5):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Left)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("18", channel_14.text(self.Column.VALUE))
            self.assertEqual("250", channel_15.text(self.Column.VALUE))
            self.assertEqual("t = 0.042657s", plot.cursor_info.text())

    @unittest.skip("Dev in progress.")
    def test_Plot_Plot_Shortcut_Key_R(self):
        """
        Test Scope:
            Check if Range Selection rectangle is painted over the plot.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Press PushButton "Create Window"
            - Press PushButton HideAxis (easy for evaluation)
            - Press Key R for range selection
            - Move Cursors
            - Press Key R for range selection
        Evaluate:
            - Evaluate that two cursors are available
            - Evaluate that new rectangle with different color is present
            - Evaluate that sum of rectangle areas is same with the one when plot is full black.
            - Evaluate that range selection disappear.
        """
        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Press PushButton "Create Window"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Press PushButton "Hide axis"
        if not plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(plot.hide_axes_btn, QtCore.Qt.LeftButton)

        # Save PixMap of clear plot
        clear_pixmap = plot.plot.viewport().grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

        # Get X position of Cursor
        cursors = Pixmap.cursors_x(clear_pixmap)
        # Evaluate that there is only one cursor
        self.assertEqual(1, len(cursors))

        # Press Key 'R' for range selection
        QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_R)
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(range_pixmap))

        # Get X position of Cursors
        cursors = Pixmap.cursors_x(range_pixmap)
        # Evaluate that two cursors are available
        self.assertEqual(2, len(cursors))

        # Evaluate that new rectangle with different color is present
        self.assertTrue(
            Pixmap.has_rectangle(pixmap=range_pixmap, color_name=Pixmap.COLOR_BLACK)
        )
        self.assertTrue(Pixmap.has_rectangle(pixmap=range_pixmap, color_name="????"))
        self.assertTrue(
            Pixmap.has_rectangle(pixmap=range_pixmap, color_name=Pixmap.COLOR_BLACK)
        )

        # Evaluate that sum of rectangle areas is same with the one when plot is full black.

        # Move Cursors

        # Press Key 'R' for range selection
        QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_R)
        self.processEvents(timeout=0.01)

        # Save PixMap of clear plot
        clear_pixmap = plot.plot.viewport().grab()
        self.assertTrue(self.pixmap_is_black(clear_pixmap))

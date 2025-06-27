#!/usr/bin/env python
from unittest import mock

from PySide6 import QtCore, QtGui, QtTest

from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget


class TestDoubleClick(TestPlotWidget):
    # Note: Test Plot Widget through FileWidget.
    def setUp(self):
        super().setUp()

        # Open File Widget
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Create a plot window
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        # Drag and Drop channel from FileWidget.channel_tree to Plot
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Press PushButton 'FocusMode' - disabled (easy for evaluation)
        if not self.plot.focused_mode_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.focused_mode_btn, QtCore.Qt.MouseButton.LeftButton)
        # Press PushButton "Hide axis"
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()

    def test_ChannelSelection(self):
        """
        Test Scope: Validate that doubleClick operation will activate deactivate channels.
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
        plot_channel = self.add_channel_to_plot()

        # Pre-evaluation
        self.assertEqual(QtCore.Qt.CheckState.Checked, plot_channel.checkState(0))

        # Press mouse double click on channel
        self.mouseDClick_WidgetItem(plot_channel)

        # Evaluate
        self.assertEqual(QtCore.Qt.CheckState.Unchecked, plot_channel.checkState(0))

        # Press mouse double click on channel
        self.mouseDClick_WidgetItem(plot_channel)
        self.assertEqual(QtCore.Qt.CheckState.Checked, plot_channel.checkState(0))

    def test_EnableDisable_Group(self):
        """
        Test Scope: Validate that doubleClick operation will activate deactivate groups.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Drag and Drop channels from FileWidget.channel_tree to Plot
            - Press 'Shift-Insert' in order to Insert Group
            - Drag and Drop some channels to new Group
            - Press mouse double click on added group
            - Press key Down few times
            - Press mouse double click on added group
            - Press key Down few times
        Evaluate:
            - Evaluate that group is disabled (not selectable anymore).
            - Evaluate that channel is enabled (selectable).
            - Evaluate that channel is not present on plot when is disabled.
        """
        # Event
        # Drag and Drop channel from FileWidget.channel_tree to Plot
        plot_channel_0 = self.add_channel_to_plot(channel_index=10)
        _ = self.add_channel_to_plot(channel_index=11)
        plot_channel_2 = self.add_channel_to_plot(channel_index=12)
        # Press 'Shift-Insert' in order to Insert Group
        # Create Channel Group. Drag channels inside the group one by one
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mc_getText:
            # Create Channel Group
            mc_getText.return_value = "FirstGroup", True
            QtTest.QTest.keySequence(self.plot.channel_selection, QtGui.QKeySequence("Shift+Insert"))
            self.processEvents()

            # Get Group Position
            for index in range(self.plot.channel_selection.topLevelItemCount()):
                item = self.plot.channel_selection.topLevelItem(index)
                if item.text(self.Column.NAME) == "FirstGroup":
                    first_group = item
                    break
            else:
                self.fail("FirstGroup is not present on Plot Channel Selection.")
            first_group.setExpanded(True)
            # Get the Name of the first channel
            first_channel = plot_channel_0.text(self.Column.NAME)
            # PreEvaluation: Ensure that group has no child
            self.assertEqual(0, first_group.childCount())
            self.move_item_inside_channels_tree_widget(src=plot_channel_0, dst=first_group)
            # PreEvaluate: Ensure that channel was added to group
            self.assertEqual(1, first_group.childCount())
            self.assertEqual(first_channel, first_group.child(0).text(self.Column.NAME))
            self.assertEqual(3, self.plot.channel_selection.topLevelItemCount())

            enabled_groups_pixmap = self.plot.plot.viewport().grab()
            # Evaluate
            self.assertTrue(
                Pixmap.has_color(
                    pixmap=enabled_groups_pixmap,
                    color_name=plot_channel_0.color,
                )
            )
            # Press mouse double click on Group
            self.mouseDClick_WidgetItem(first_group)

            # Evaluate
            for _ in range(10):
                self.processEvents(0.05)
            disabled_groups_pixmap = self.plot.plot.viewport().grab()
            self.assertFalse(
                Pixmap.has_color(
                    pixmap=disabled_groups_pixmap,
                    color_name=plot_channel_0.color,
                )
            )

            for _ in range(4):
                QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Down)
                self.processEvents()

            # Evaluate that item is still 2nd one because
            selectedItems = self.plot.channel_selection.selectedItems()
            self.assertEqual(1, len(selectedItems))
            selectedItem = selectedItems[0].text(self.Column.NAME)
            self.assertEqual(plot_channel_2.text(self.Column.NAME), selectedItem)

            # Press mouse double click on channel
            self.mouseDClick_WidgetItem(first_group)
            self.processEvents()

            for _ in range(4):
                QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Down)
                self.processEvents()

            # Evaluate that item is the one from the group
            selectedItems = self.plot.channel_selection.selectedItems()
            self.assertEqual(1, len(selectedItems))
            selectedItem = selectedItems[0].text(self.Column.NAME)
            self.assertEqual(plot_channel_0.text(self.Column.NAME), selectedItem)

            # Evaluate
            # After channels are enabled, they first flash a few times on plot.
            # ProcessEvents few times for channels to be present
            for _ in range(20):
                self.processEvents(0.05)

            enabled_groups_pixmap = self.plot.plot.viewport().grab()
            self.assertTrue(
                Pixmap.has_color(
                    pixmap=enabled_groups_pixmap,
                    color_name=plot_channel_0.color,
                ),
                msg=f"Color of channel {plot_channel_0.text(self.Column.NAME)} is not present on plot.",
            )

    # @unittest.skipIf(sys.platform == "win32", "fails on Windows")
    def test_EnableDisable_ParentGroup(self):
        """
        Test Scope:
        Validate that doubleClick operation will activate deactivate subgroups when parent is disabled.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Drag and Drop channels from FileWidget.channel_tree to Plot
            - Press 'Shift-Insert' in order to Insert Group: A
            - Press 'Shift-Insert' in order to Insert Group: B
            - Press 'Shift-Insert' in order to Insert Group: C
            - Drag and Drop some channels to new Group: A
            - Drag and Drop some channels to new Group: B
            - Drag and Drop some channels to new Group: C
            - Move Group C inside Group B
            - Move Group B inside Group A
            - Press mouse double click on added group: A
            - Press key Down few times
            - Press mouse double click on added group: A
            - Press key Down few times
        Evaluate:
            - Evaluate that subgroups are disabled (not selectable anymore).
            - Evaluate that channels are enabled (selectable).
            - Evaluate that channel is not present on plot when is disabled.
            - Evaluate channel color is darkGray when it's disabled.
        """
        # Event
        # Drag and Drop channel from FileWidget.channel_tree to Plot
        plot_channel_a = self.add_channel_to_plot(channel_index=10)
        plot_channel_b = self.add_channel_to_plot(channel_index=11)
        plot_channel_c = self.add_channel_to_plot(channel_index=12)
        plot_channel_d = self.add_channel_to_plot(channel_index=13)
        self.processEvents()
        # Press 'Shift-Insert' in order to Insert Group
        # Create Channel Group. Drag channels inside the group one by one
        groups = {}
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mc_getText:
            for group_name in ("A", "B", "C"):
                # Create Channel Group
                mc_getText.return_value = group_name, True
                QtTest.QTest.keySequence(self.plot.channel_selection, QtGui.QKeySequence("Shift+Insert"))
                self.processEvents()

            # Get Groups Position
            for index in range(self.plot.channel_selection.topLevelItemCount()):
                item = self.plot.channel_selection.topLevelItem(index)
                group_name = item.text(self.Column.NAME)
                if group_name in ("A", "B", "C"):
                    groups[group_name] = item
                    item.setExpanded(True)

            # Get the First Item that will be moved
            for group_name, plot_channel in zip(
                ("A", "B", "C"), (plot_channel_a, plot_channel_b, plot_channel_c), strict=False
            ):
                # PreEvaluation: Ensure that group has no child
                self.assertEqual(0, groups[group_name].childCount())
                self.move_item_inside_channels_tree_widget(src=plot_channel, dst=groups[group_name])
                self.assertEqual(1, groups[group_name].childCount())
                self.processEvents()

            # Move Group C inside Group B
            self.move_item_inside_channels_tree_widget(src=groups["C"], dst=groups["B"])
            # Move Group B inside Group A
            self.move_item_inside_channels_tree_widget(src=groups["B"], dst=groups["A"])
            groups["A"].setExpanded(True)
            groups["B"].setExpanded(True)
            groups["C"].setExpanded(True)

            self.processEvents()
            enabled_groups_pixmap = self.plot.plot.viewport().grab()
            # Evaluate
            for channel in (plot_channel_a, plot_channel_b, plot_channel_c):
                self.assertTrue(
                    Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=channel.color),
                    msg=f"Color of Channel: {channel.text(self.Column.NAME)} not present on 'plot'.",
                )
                color_name = channel.foreground(self.Column.NAME).color().name()
                self.assertNotEqual(color_name, "#808080")
            # Press mouse double click on Group A
            self.mouseDClick_WidgetItem(groups["A"])

            # Evaluate
            for _ in range(10):
                self.processEvents(0.05)
            disabled_groups_pixmap = self.plot.plot.viewport().grab()
            for channel in (plot_channel_a, plot_channel_b, plot_channel_c):
                self.assertFalse(
                    Pixmap.has_color(
                        pixmap=disabled_groups_pixmap,
                        color_name=channel.color,
                    ),
                    msg=f"Color of Channel: {channel.text(self.Column.NAME)} present on 'plot'.",
                )

            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Up)
            for _ in range(4):
                QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Down)
                self.processEvents()

            # Evaluate that item is still plot_channel_d
            selectedItems = self.plot.channel_selection.selectedItems()
            self.assertEqual(1, len(selectedItems))
            selectedItem = selectedItems[0].text(self.Column.NAME)
            self.assertEqual(plot_channel_d.text(self.Column.NAME), selectedItem)

            # Press mouse double click on channel
            self.mouseDClick_WidgetItem(groups["A"])
            self.processEvents()

            for _ in range(8):
                QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Down)
                self.processEvents()

            # Evaluate that item is the one from the group C
            selectedItems = self.plot.channel_selection.selectedItems()
            self.assertEqual(1, len(selectedItems))
            selectedItem = selectedItems[0].text(self.Column.NAME)
            self.assertEqual(plot_channel_c.text(self.Column.NAME), selectedItem)

            # Evaluate
            # After channels are enabled, they first flash a few times on self.plot.
            # ProcessEvents few times for channels to be present
            for _ in range(20):
                self.processEvents(0.05)

            enabled_groups_pixmap = self.plot.plot.viewport().grab()
            for channel in (
                plot_channel_a,
                plot_channel_b,
                plot_channel_c,
                plot_channel_d,
            ):
                self.assertTrue(
                    Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=channel.color),
                    msg=f"Color for Channel: {channel.text(self.Column.NAME)} not present on 'plot'",
                )

    # @unittest.skipIf(sys.platform == "win32", "fails on Windows")
    def test_EnableDisable_Subgroup(self):
        """
        Test Scope:
        Validate that doubleClick operation will activate deactivate subgroups and keep parent enabled.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Drag and Drop channels from FileWidget.channel_tree to Plot
            - Press 'Shift-Insert' in order to Insert Group: A
            - Press 'Shift-Insert' in order to Insert Group: B
            - Press 'Shift-Insert' in order to Insert Group: C
            - Drag and Drop some channels to new Group: A
            - Drag and Drop some channels to new Group: B
            - Drag and Drop some channels to new Group: C
            - Move Group C inside Group B
            - Move Group B inside Group A
            - Press mouse double click on added group: B
            - Press key Down few times
            - Press mouse double click on added group: B
            - Press key Down few times
        Evaluate:
            - Evaluate that subgroups are disabled (not selectable anymore).
            - Evaluate that channels are enabled (selectable).
            - Evaluate that channel is not present on plot when is disabled.
        """
        # Event
        # Drag and Drop channel from FileWidget.channel_tree to Plot
        plot_channel_a = self.add_channel_to_plot(channel_index=10)
        plot_channel_b = self.add_channel_to_plot(channel_index=11)
        plot_channel_c = self.add_channel_to_plot(channel_index=12)
        plot_channel_d = self.add_channel_to_plot(channel_index=13)

        # Press 'Shift-Insert' in order to Insert Group
        # Create Channel Group. Drag channels inside the group one by one
        groups = {}
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mc_getText:
            for group_name in ("A", "B", "C"):
                # Create Channel Group
                mc_getText.return_value = group_name, True
                QtTest.QTest.keySequence(self.plot.channel_selection, QtGui.QKeySequence("Shift+Insert"))
                self.processEvents()

            # Get Groups Position
            for index in range(self.plot.channel_selection.topLevelItemCount()):
                item = self.plot.channel_selection.topLevelItem(index)
                group_name = item.text(self.Column.NAME)
                if group_name in ("A", "B", "C"):
                    groups[group_name] = item
                    item.setExpanded(True)

            # Get the First Item that will be moved
            for group_name, plot_channel in zip(
                ("A", "B", "C"), (plot_channel_a, plot_channel_b, plot_channel_c), strict=False
            ):
                # PreEvaluation: Ensure that group has no child
                self.assertEqual(0, groups[group_name].childCount())
                self.move_item_inside_channels_tree_widget(src=plot_channel, dst=groups[group_name])
                self.assertEqual(1, groups[group_name].childCount())
                self.processEvents()

            # Move Group C inside Group B
            self.move_item_inside_channels_tree_widget(src=groups["C"], dst=groups["B"])
            # Move Group B inside Group A
            self.move_item_inside_channels_tree_widget(src=groups["B"], dst=groups["A"])

            groups["A"].setExpanded(True)
            groups["B"].setExpanded(True)
            groups["C"].setExpanded(True)

            enabled_groups_pixmap = self.plot.plot.viewport().grab()
            self.processEvents()
            # Evaluate
            for channel in (plot_channel_a, plot_channel_b, plot_channel_c):
                self.assertTrue(
                    Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=channel.color),
                    msg=f"Color of Channel: {channel.text(self.Column.NAME)} not present on 'plot'.",
                )
            # Press mouse double click on Group B
            self.mouseDClick_WidgetItem(groups["B"])
            disabled_groups_pixmap = self.plot.plot.viewport().grab()

            # Evaluate
            self.assertTrue(
                Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=plot_channel_a.color),
                msg=f"Color of Channel: {plot_channel_a.text(self.Column.NAME)} not present on 'plot'.",
            )
            for channel in (plot_channel_b, plot_channel_c):
                self.assertFalse(
                    Pixmap.has_color(
                        pixmap=disabled_groups_pixmap,
                        color_name=channel.color,
                    ),
                    msg=f"Color of Channel: {channel.text(self.Column.NAME)} present on 'plot'.",
                )

            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Up)
            for _ in range(4):
                QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Down)
                self.processEvents()

            # Press mouse double click on group B
            self.mouseDClick_WidgetItem(groups["B"])
            self.processEvents()

            for _ in range(8):
                QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Down)
                self.processEvents()

            # Evaluate
            # After channels are enabled, they first flash a few times on self.plot.
            # ProcessEvents few times for channels to be present
            for _ in range(20):
                self.processEvents(0.05)

            enabled_groups_pixmap = self.plot.plot.viewport().grab()
            for channel in (
                plot_channel_a,
                plot_channel_b,
                plot_channel_c,
                plot_channel_d,
            ):
                self.assertTrue(
                    Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=channel.color),
                    msg=f"Color for Channel: {channel.text(self.Column.NAME)} not present on 'plot'",
                )

    def test_EnableDisable_Preserve_Subgroup_State_0(self):
        """
        Test Scope:
            Validate that doubleClick operation will activate/deactivate subgroups and preserve state
            when parent is disabled/enabled.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Drag and Drop channels from FileWidget.channel_tree to Plot
            - Press 'Shift-Insert' in order to Insert Group: A
            - Press 'Shift-Insert' in order to Insert Group: B
            - Press 'Shift-Insert' in order to Insert Group: C
            - Drag and Drop some channels to new Group: A
            - Drag and Drop some channels to new Group: B
            - Drag and Drop some channels to new Group: C
            - Move Group C inside Group B
            - Move Group B inside Group A
            - Press mouse double click on added group: C
            - Press mouse double click on added group: B
            - Press key Down few times
            - Press mouse double click on added group: B
            - Press key Down few times
        Evaluate:
            - Evaluate that subgroups are disabled (not selectable anymore).
            - Evaluate that channel is not present on plot when is disabled.
            - Evaluate that subgroup C state is maintained after subgroup B is enabled.
        """
        # Event
        # Drag and Drop channel from FileWidget.channel_tree to Plot
        plot_channel_a = self.add_channel_to_plot(channel_index=10)
        plot_channel_b = self.add_channel_to_plot(channel_index=11)
        plot_channel_c = self.add_channel_to_plot(channel_index=12)
        plot_channel_d = self.add_channel_to_plot(channel_index=13)
        # Press 'Shift-Insert' in order to Insert Group
        # Create Channel Group. Drag channels inside the group one by one
        groups = {}
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mc_getText:
            for group_name in ("A", "B", "C"):
                # Create Channel Group
                mc_getText.return_value = group_name, True
                QtTest.QTest.keySequence(self.plot.channel_selection, QtGui.QKeySequence("Shift+Insert"))
                self.processEvents()

            # Get Groups Position
            for index in range(self.plot.channel_selection.topLevelItemCount()):
                item = self.plot.channel_selection.topLevelItem(index)
                group_name = item.text(self.Column.NAME)
                if group_name in ("A", "B", "C"):
                    groups[group_name] = item
                    item.setExpanded(True)

            # Get the First Item that will be moved
            for group_name, plot_channel in zip(
                ("A", "B", "C"), (plot_channel_a, plot_channel_b, plot_channel_c), strict=False
            ):
                # PreEvaluation: Ensure that group has no child
                self.assertEqual(0, groups[group_name].childCount())
                self.move_item_inside_channels_tree_widget(src=plot_channel, dst=groups[group_name])

                self.processEvents(0.05)
                self.assertEqual(1, groups[group_name].childCount())

            # Move Group C inside Group B
            self.move_item_inside_channels_tree_widget(src=groups["C"], dst=groups["B"])
            # Move Group B inside Group A
            self.move_item_inside_channels_tree_widget(src=groups["B"], dst=groups["A"])

            groups["A"].setExpanded(True)
            groups["B"].setExpanded(True)
            groups["C"].setExpanded(True)

            enabled_groups_pixmap = self.plot.plot.viewport().grab()
            self.processEvents()
            # Evaluate
            for channel in (plot_channel_a, plot_channel_b, plot_channel_c):
                self.assertTrue(
                    Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=channel.color),
                    msg=f"Color of Channel: {channel.text(self.Column.NAME)} not present on 'plot'.",
                )
            # Press mouse double click on Group C
            self.mouseDClick_WidgetItem(groups["C"])
            # Press mouse double click on Group B
            self.mouseDClick_WidgetItem(groups["B"])
            disabled_groups_pixmap = self.plot.plot.viewport().grab()

            # Evaluate
            self.assertTrue(
                Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=plot_channel_a.color),
                msg=f"Color of Channel: {plot_channel_a.text(self.Column.NAME)} not present on 'plot'.",
            )
            for channel in (plot_channel_b, plot_channel_c):
                self.assertFalse(
                    Pixmap.has_color(
                        pixmap=disabled_groups_pixmap,
                        color_name=channel.color,
                    ),
                    msg=f"Color of Channel: {channel.text(self.Column.NAME)} present on 'plot'.",
                )

            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Up)
            for _ in range(4):
                QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Down)
                self.processEvents()

            # Press mouse double click on group B
            self.mouseDClick_WidgetItem(groups["B"])
            self.processEvents()

            for _ in range(8):
                QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Down)
                self.processEvents()

            # Evaluate
            # After channels are enabled, they first flash a few times on self.plot.
            # ProcessEvents few times for channels to be present
            for _ in range(20):
                self.processEvents(0.05)

            enabled_groups_pixmap = self.plot.plot.viewport().grab()
            self.assertFalse(
                Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=plot_channel_c.color),
                msg=f"Color for Channel: {plot_channel_c.text(self.Column.NAME)} present on 'plot'",
            )
            for channel in (plot_channel_a, plot_channel_b, plot_channel_d):
                self.assertTrue(
                    Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=channel.color),
                    msg=f"Color for Channel: {channel.text(self.Column.NAME)} not present on 'plot'",
                )

    def test_EnableDisable_Preserve_Subgroup_State_1(self):
        """
        Test Scope:
            Validate that doubleClick operation will activate/deactivate subgroups and preserve state
            when parent is disabled/enabled.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
                - Simulate that Plot window is selected as window type.
            - Drag and Drop channels from FileWidget.channel_tree to Plot
            - Press 'Shift-Insert' in order to Insert Group: A
            - Press 'Shift-Insert' in order to Insert Group: B
            - Press 'Shift-Insert' in order to Insert Group: C
            - Drag and Drop some channels to new Group: A
            - Drag and Drop some channels to new Group: B
            - Drag and Drop some channels to new Group: C
            - Move Group C inside Group B
            - Move Group B inside Group A
            - Press mouse double click on added group: C
            - Press mouse double click on added group: B
            - Press key Down few times
            - Press mouse double click on added group: B
            - Press mouse click on group B:
                - Use supposed to simulate Deactivate group from ContextMenu for group: C
                but the bug is related to previous enabled group which remains selected.
            - Press key Down few times
        Evaluate:
            - Evaluate that subgroups are disabled (not selectable anymore).
            - Evaluate that channels are enabled (selectable).
            - Evaluate that channel is not present on plot when is disabled.
        """
        # Event
        # Drag and Drop channel from FileWidget.channel_tree to Plot
        plot_channel_a = self.add_channel_to_plot(channel_index=10)
        plot_channel_b = self.add_channel_to_plot(channel_index=11)
        plot_channel_c = self.add_channel_to_plot(channel_index=12)
        plot_channel_d = self.add_channel_to_plot(channel_index=13)
        # Press 'Shift-Insert' in order to Insert Group
        # Create Channel Group. Drag channels inside the group one by one
        groups = {}
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mc_getText:
            for group_name in ("A", "B", "C"):
                # Create Channel Group
                mc_getText.return_value = group_name, True
                QtTest.QTest.keySequence(self.plot.channel_selection, QtGui.QKeySequence("Shift+Insert"))
                self.processEvents()

            # Get Groups Position
            for index in range(self.plot.channel_selection.topLevelItemCount()):
                item = self.plot.channel_selection.topLevelItem(index)
                group_name = item.text(self.Column.NAME)
                if group_name in ("A", "B", "C"):
                    groups[group_name] = item
                    item.setExpanded(True)

            # Get the First Item that will be moved
            for group_name, plot_channel in zip(
                ("A", "B", "C"), (plot_channel_a, plot_channel_b, plot_channel_c), strict=False
            ):
                # PreEvaluation: Ensure that group has no child
                self.assertEqual(0, groups[group_name].childCount())
                self.move_item_inside_channels_tree_widget(src=plot_channel, dst=groups[group_name])

                self.processEvents(0.05)
                self.assertEqual(1, groups[group_name].childCount())

            # Move Group C inside Group B
            self.move_item_inside_channels_tree_widget(src=groups["C"], dst=groups["B"])
            # Move Group B inside Group A
            self.move_item_inside_channels_tree_widget(src=groups["B"], dst=groups["A"])

            groups["A"].setExpanded(True)
            groups["B"].setExpanded(True)
            groups["C"].setExpanded(True)

            enabled_groups_pixmap = self.plot.plot.viewport().grab()
            self.processEvents()
            # Evaluate
            for channel in (plot_channel_a, plot_channel_b, plot_channel_c):
                self.assertTrue(
                    Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=channel.color),
                    msg=f"Color of Channel: {channel.text(self.Column.NAME)} not present on 'plot'.",
                )
            # Press mouse double click on Group C
            self.mouseDClick_WidgetItem(groups["C"])
            # Press mouse double click on Group B
            self.mouseDClick_WidgetItem(groups["B"])
            disabled_groups_pixmap = self.plot.plot.viewport().grab()

            # Evaluate
            self.assertTrue(
                Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=plot_channel_a.color),
                msg=f"Color of Channel: {plot_channel_a.text(self.Column.NAME)} not present on 'plot'.",
            )
            for channel in (plot_channel_b, plot_channel_c):
                self.assertFalse(
                    Pixmap.has_color(
                        pixmap=disabled_groups_pixmap,
                        color_name=channel.color,
                    ),
                    msg=f"Color of Channel: {channel.text(self.Column.NAME)} present on 'plot'.",
                )

            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Up)
            for _ in range(4):
                QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Down)
                self.processEvents()

            # Press mouse double click on group B
            self.mouseDClick_WidgetItem(groups["B"])
            self.mouseClick_WidgetItem(groups["B"])
            self.processEvents()

            # Evaluate selected items
            selectedItems = self.plot.channel_selection.selectedItems()
            self.assertEqual(1, len(selectedItems))
            self.assertEqual("B", selectedItems[0].text(self.Column.NAME))

            # Disable group C from context-menu
            # Disclaimer: Group C is already disabled
            # Simple Click on group C is enough. The bug was that when you click disabled item,
            # the previous selected item remains in selectedItems.
            # Basically, the disabled item is not selectable and selection is kept.
            # In case that previous item was a group, and 'Disable groups' is selected from the context-menu
            # then, unintended parent and subgroups disable will happen.
            self.mouseClick_WidgetItem(groups["C"])

            # Evaluate that selection was changed.
            selectedItems = self.plot.channel_selection.selectedItems()
            self.assertEqual(0, len(selectedItems))

            for _ in range(8):
                QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Down)
                self.processEvents()

            # Evaluate
            # After channels are enabled, they first flash a few times on self.plot.
            # ProcessEvents few times for channels to be present
            for _ in range(20):
                self.processEvents(0.05)

            enabled_groups_pixmap = self.plot.plot.viewport().grab()
            self.assertFalse(
                Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=plot_channel_c.color),
                msg=f"Color for Channel: {plot_channel_c.text(self.Column.NAME)} present on 'plot'",
            )
            for channel in (plot_channel_a, plot_channel_b, plot_channel_d):
                self.assertTrue(
                    Pixmap.has_color(pixmap=enabled_groups_pixmap, color_name=channel.color),
                    msg=f"Color for Channel: {channel.text(self.Column.NAME)} not present on 'plot'",
                )

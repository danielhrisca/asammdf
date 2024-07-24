#!/usr/bin/env python
from PySide6 import QtCore, QtTest

from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget


class TestPushButtons(TestPlotWidget):
    def setUp(self):
        super().setUp()

        self.channel_0_name = "ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL"
        self.channel_1_name = "ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL"

        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Press PushButton "Create Window"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Press PushButton "Hide axis"
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)

        # Save PixMap of clear plot
        clear_pixmap = self.plot.plot.viewport().grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

        # Add Channels to Plot
        self.add_channels([self.channel_0_name, self.channel_1_name])
        self.plot_tree_channel_0 = self.channels[0]
        self.plot_tree_channel_1 = self.channels[1]
        self.assertEqual(2, self.plot.channel_selection.topLevelItemCount())

        # Identify PlotSignal
        self.plot_graph_channel_0, self.plot_graph_channel_1 = None, None
        for channel in self.plot.plot.signals:
            if channel.name == self.channel_0_name:
                self.plot_graph_channel_0 = channel
            elif channel.name == self.channel_1_name:
                self.plot_graph_channel_1 = channel

    def test_Plot_ChannelSelection_PushButton_ValuePanel(self):
        """
        Test Scope:
            Check that Value Panel label is visible or hidden according to Push Button
            Check that Value Panel label is updated according signal samples
        Precondition:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
            - Drag and Drop 2 channels from FileWidget.channels_tree to Plot.channels_selection
        Events:
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
        if self.plot.selected_channel_value.isVisible():
            # Press PushButton "Hide selected channel value panel"
            QtTest.QTest.mouseClick(self.plot.selected_channel_value_btn, QtCore.Qt.MouseButton.LeftButton)
        # Press PushButton "Show selected channel value panel"
        QtTest.QTest.mouseClick(self.plot.selected_channel_value_btn, QtCore.Qt.MouseButton.LeftButton)
        self.assertTrue(self.plot.selected_channel_value.isVisible())

        # Press PushButton "Hide selected channel value panel"
        QtTest.QTest.mouseClick(self.plot.selected_channel_value_btn, QtCore.Qt.MouseButton.LeftButton)
        self.assertFalse(self.plot.selected_channel_value.isVisible())
        # Press PushButton "Show selected channel value panel"
        QtTest.QTest.mouseClick(self.plot.selected_channel_value_btn, QtCore.Qt.MouseButton.LeftButton)
        self.assertTrue(self.plot.selected_channel_value.isVisible())

        # Select Channel
        self.mouseClick_WidgetItem(self.plot_tree_channel_0)

        # Evaluate
        plot_channel_0_value = self.plot_tree_channel_0.text(self.Column.VALUE)
        plot_channel_0_unit = self.plot_tree_channel_0.text(self.Column.UNIT)
        self.assertEqual(
            f"{plot_channel_0_value} {plot_channel_0_unit}",
            self.plot.selected_channel_value.text(),
        )

        # Event
        self.plot.plot.setFocus()
        self.processEvents(0.1)
        # Send Key strokes
        for _ in range(6):
            QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key.Key_Right)
            self.processEvents(0.1)
        self.processEvents(0.1)

        # Evaluate
        plot_channel_0_value = self.plot_tree_channel_0.text(self.Column.VALUE)
        plot_channel_0_unit = self.plot_tree_channel_0.text(self.Column.UNIT)
        self.assertEqual(
            f"{plot_channel_0_value} {plot_channel_0_unit}",
            self.plot.selected_channel_value.text(),
        )

        # Select 2nd Channel
        self.mouseClick_WidgetItem(self.plot_tree_channel_1)

        # Evaluate
        plot_channel_1_value = self.plot_tree_channel_1.text(self.Column.VALUE)
        plot_channel_1_unit = self.plot_tree_channel_1.text(self.Column.UNIT)
        self.assertEqual(
            f"{plot_channel_1_value} {plot_channel_1_unit}",
            self.plot.selected_channel_value.text(),
        )

        # Event
        self.plot.plot.setFocus()
        self.processEvents(0.1)
        # Send Key strokes
        for _ in range(6):
            QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key.Key_Right)
            self.processEvents(0.1)
        self.processEvents(0.1)

        # Evaluate
        plot_channel_1_value = self.plot_tree_channel_1.text(self.Column.VALUE)
        plot_channel_1_unit = self.plot_tree_channel_1.text(self.Column.UNIT)
        self.assertEqual(
            f"{plot_channel_1_value} {plot_channel_1_unit}",
            self.plot.selected_channel_value.text(),
        )

    def test_Plot_ChannelSelection_PushButton_FocusedMode(self):
        """
        Test Scope:
            Check if Plot is cleared when no channel is selected.
            Check if Plot is showing all channels when Focus Mode is disabled.
            Check if Plot is showing only one channel when Focus Mode is enabled.
        Precondition:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
            - Press PushButton HideAxis (easy for evaluation)
            - Drag and Drop channels from FileWidget.channels_tree to Plot.channels_selection:
                # First
                - ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
                # Second
                - ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
        Events:
            - Press PushButton FocusMode
            - Press PushButton FocusMode
        Evaluate:
            - Evaluate that channels are displayed when FocusMode is disabled.
            - Evaluate that selected channels is displayed when FocusMode is enabled.
        """
        # Events
        if not self.plot.focused_mode_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.focused_mode_btn, QtCore.Qt.MouseButton.LeftButton)

        channels_present_pixmap = self.plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap=channels_present_pixmap))
        self.assertTrue(
            Pixmap.has_color(
                pixmap=channels_present_pixmap,
                color_name=self.plot_graph_channel_0.color_name,
            )
        )
        self.assertTrue(
            Pixmap.has_color(
                pixmap=channels_present_pixmap,
                color_name=self.plot_graph_channel_1.color_name,
            )
        )

        # Press Button Focus Mode
        QtTest.QTest.mouseClick(self.plot.focused_mode_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate
        focus_mode_clear_pixmap = self.plot.plot.viewport().grab()
        # No Channel is selected
        self.assertTrue(Pixmap.is_black(pixmap=focus_mode_clear_pixmap))

        # Select 2nd Channel
        QtTest.QTest.mouseClick(
            self.plot.channel_selection.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            self.plot.channel_selection.visualItemRect(self.plot_tree_channel_1).center(),
        )
        # Process flash until signal is present on plot.
        for _ in range(10):
            self.processEvents(timeout=0.01)
            focus_mode_channel_1_pixmap = self.plot.plot.viewport().grab()
            if Pixmap.has_color(
                pixmap=focus_mode_channel_1_pixmap,
                color_name=self.plot_graph_channel_1.color_name,
            ):
                break

        # Evaluate
        focus_mode_channel_1_pixmap = self.plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap=focus_mode_channel_1_pixmap))
        self.assertFalse(
            Pixmap.has_color(
                pixmap=focus_mode_channel_1_pixmap,
                color_name=self.plot_graph_channel_0.color_name,
            )
        )
        self.assertTrue(
            Pixmap.has_color(
                pixmap=focus_mode_channel_1_pixmap,
                color_name=self.plot_graph_channel_1.color_name,
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
        if not self.plot.delta_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.delta_btn, QtCore.Qt.MouseButton.LeftButton)
            self.processEvents()

        # Ensure that delta char is not present on channel values
        self.assertNotIn("Δ", self.plot_tree_channel_0.text(self.Column.VALUE))
        self.assertNotIn("Δ", self.plot_tree_channel_1.text(self.Column.VALUE))

        # Press PushButton Delta (range is not active)
        QtTest.QTest.mouseClick(self.plot.delta_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()

        # Ensure that delta char is not present on channel values even if button is pressed
        self.assertNotIn("Δ", self.plot_tree_channel_0.text(self.Column.VALUE))
        self.assertNotIn("Δ", self.plot_tree_channel_1.text(self.Column.VALUE))

        # Press Key 'R' for range selection
        QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key.Key_R)
        self.processEvents(timeout=0.01)

        # Ensure that delta char is present on channel values even if button is pressed
        self.assertIn("Δ", self.plot_tree_channel_0.text(self.Column.VALUE))
        self.assertIn("Δ", self.plot_tree_channel_1.text(self.Column.VALUE))

        # Move cursor
        # Select channel_1
        QtTest.QTest.mouseClick(
            self.plot.channel_selection.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            self.plot.channel_selection.visualItemRect(self.plot_tree_channel_1).center(),
        )
        self.plot.plot.setFocus()
        self.processEvents(0.1)
        # Move a little bit in center of measurement
        for _ in range(15):
            QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key.Key_Right)
            self.processEvents(timeout=0.1)

        # Get current value: Ex: 'Δ = 8'. Get last number
        old_channel_0_value = int(self.plot_tree_channel_0.text(self.Column.VALUE).split(" ")[-1])
        old_channel_1_value = int(self.plot_tree_channel_1.text(self.Column.VALUE).split(" ")[-1])
        for count in range(5):
            QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key.Key_Right)
            self.processEvents(timeout=0.1)
            # Evaluate
            channel_0_value = int(self.plot_tree_channel_0.text(self.Column.VALUE).split(" ")[-1])
            channel_1_value = int(self.plot_tree_channel_1.text(self.Column.VALUE).split(" ")[-1])
            self.assertLess(old_channel_0_value, channel_0_value)
            self.assertGreater(old_channel_1_value, channel_1_value)
            old_channel_0_value = channel_0_value
            old_channel_1_value = channel_1_value

        # Press Key 'R' for range selection
        QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key.Key_R)
        self.processEvents(timeout=0.01)

        # Ensure that delta char is not present on channel values even if button is pressed
        self.assertNotIn("Δ", self.plot_tree_channel_0.text(self.Column.VALUE))
        self.assertNotIn("Δ", self.plot_tree_channel_1.text(self.Column.VALUE))

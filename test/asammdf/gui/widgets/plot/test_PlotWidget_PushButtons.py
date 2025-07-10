#!/usr/bin/env python

from PySide6 import QtCore, QtTest
from PySide6.QtWidgets import QPushButton as QBtn
from PySide6.QtWidgets import QTreeWidgetItemIterator

from asammdf.gui.serde import COLORS
from asammdf.gui.utils import BLUE
from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget


class TestPushButtons(TestPlotWidget):
    def setUp(self):
        super().setUp()

        self.channel_0_name = "ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL"
        self.channel_1_name = "ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL"

        # Open test file in File Widget
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

        # Press PushButton "Hide bookmarks"
        if not self.plot.bookmark_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.bookmark_btn, QtCore.Qt.MouseButton.LeftButton)

        if not self.plot.focused_mode_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.focused_mode_btn, QtCore.Qt.MouseButton.LeftButton)

        self.processEvents(0.1)
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

    def test_Plot_ChannelSelection_PushButton_ToggleBookmarks(self):
        """
        Precondition
            -

        Events:
            - Display 1 signal on plot
            - Press 3 times `Toggle bookmarks` button

        Evaluate:
            - Evaluate that bookmarks are not displayed before pressing `Toggle bookmarks` button
            - Evaluate that bookmarks are displayed after pressing `Toggle bookmarks` button first time
            - Evaluate that bookmarks are not displayed after pressing `Toggle bookmarks` button second time
        """
        # Hide channels
        iterator = QTreeWidgetItemIterator(self.plot.channel_selection)
        while iterator.value():
            item = iterator.value()
            if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            iterator += 1

        # Press `Toggle bookmarks` button
        QtTest.QTest.mouseClick(self.plot.bookmark_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents(0.1)

        # Get expected colors
        pg_colors = Pixmap.color_names_exclude_defaults(self.plot.plot.viewport().grab())
        bookmarks_colors = COLORS[: len(COLORS) - len(self.plot.plot.bookmarks) - 1 : -1]

        # Evaluate
        self.assertTrue(self.plot.show_bookmarks)
        for color in bookmarks_colors:
            self.assertIn(color, pg_colors)
        self.assertIn(BLUE, pg_colors)

        # Press `Toggle bookmarks` button
        QtTest.QTest.mouseClick(self.plot.bookmark_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()
        # Evaluate
        self.assertFalse(self.plot.show_bookmarks)
        self.assertTrue(Pixmap.is_black(self.plot.plot.viewport().grab()))

    def test_Plot_ChannelSelection_PushButton_HideAxes(self):
        """
        Events:
            - Display 2 signal on plot
            - Press 3 times `Show axes` button
        Evaluate:
            - Evaluate that bookmarks are not displayed before pressing `Hide axes` button
            - Evaluate that bookmarks are displayed after pressing `Hide axes` button first time
            - Evaluate that bookmarks are not displayed after pressing `Hide axes` button second time
            _ Evaluate that only y_axis color are the same as selected channel color
        """
        # Hide channels
        iterator = QTreeWidgetItemIterator(self.plot.channel_selection)
        while iterator.value():
            item = iterator.value()
            if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            iterator += 1

        if self.plot.plot.y_axis.isVisible() or self.plot.plot.x_axis.isVisible():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
            # Evaluate
            self.assertFalse(self.plot.plot.y_axis.isVisible())
            self.assertFalse(self.plot.plot.x_axis.isVisible())

        # Press `Show axes` button
        QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()

        # Evaluate
        self.assertTrue(self.plot.plot.y_axis.isVisible())
        self.assertTrue(self.plot.plot.x_axis.isVisible())

        with self.subTest("test pen color of axes"):
            self.assertNotEqual(self.plot_tree_channel_0.color.name(), self.plot_tree_channel_1.color.name())
            x_axis_color = self.plot.plot.x_axis.pen().color().name()

            # Event
            self.mouseClick_WidgetItem(self.plot_tree_channel_0)
            self.processEvents()

            # Evaluate
            self.assertEqual(x_axis_color, self.plot.plot.x_axis.pen().color().name())
            self.assertEqual(self.plot.plot.y_axis.pen().color().name(), self.plot_tree_channel_0.color.name())
            self.assertTrue(Pixmap.has_color(self.plot.plot.viewport().grab(), self.plot_tree_channel_0.color.name()))

            # Event
            self.mouseClick_WidgetItem(self.plot_tree_channel_1)
            self.processEvents()

            # Evaluate
            self.assertEqual(x_axis_color, self.plot.plot.x_axis.pen().color().name())
            self.assertEqual(self.plot.plot.y_axis.pen().color().name(), self.plot_tree_channel_1.color.name())
            self.assertTrue(Pixmap.has_color(self.plot.plot.viewport().grab(), self.plot_tree_channel_1.color.name()))

        # Press `Hide axes` button
        QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        # Evaluate
        self.assertFalse(self.plot.plot.y_axis.isVisible())
        self.assertFalse(self.plot.plot.x_axis.isVisible())
        self.assertTrue(Pixmap.is_black(self.plot.plot.viewport().grab()))

    def test_Plot_ChannelSelection_PushButton_Lock(self):
        """
        Events:
            - Display 2 signal on plot
            - Press `Lock` button to lock y-axis
            - Move signal using drag-and-drop technique on both axis
            - Press `Lock` button to unlock y-axis
            - Move signal using drag-and-drop technique on both axis

        Evaluate:
            - Evaluate that if y-axis is locked, signal cannot be moved only on y-axis and common axis column is hidden
            - Evaluate that if y-axis is unlocked, signal can be moved on both x and y axes
              and common axis column isn't hidden
        """

        def move_signal(x, y):
            def drag_and_drop():
                center = self.plot.plot.viewport().geometry().center()
                QtTest.QTest.mouseMove(self.plot.plot.viewport(), QtCore.QPoint(center.x() + x, center.y() + y))

                QtTest.QTest.mouseRelease(
                    self.plot.plot.viewport(),
                    QtCore.Qt.MouseButton.LeftButton,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                    QtCore.QPoint(center.x() + x, center.y() + y),
                    20,
                )

            self.mouseClick_WidgetItem(self.plot_tree_channel_0)
            QtCore.QTimer.singleShot(100, drag_and_drop)
            QtTest.QTest.mousePress(self.plot.plot.viewport(), QtCore.Qt.MouseButton.LeftButton)

        if self.plot.locked:
            QtTest.QTest.mouseClick(self.plot.lock_btn, QtCore.Qt.MouseButton.LeftButton)
            # Evaluate
            self.assertFalse(self.plot.locked)
            self.assertFalse(self.plot.channel_selection.isColumnHidden(self.plot.channel_selection.CommonAxisColumn))

        # Evaluate that signal wasn't moved on x-axis
        self.assertIsNone(self.plot_graph_channel_0.trim_info)

        # Press `Lock` button
        QtTest.QTest.mouseClick(self.plot.lock_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()
        sig_0_y_range = self.plot_graph_channel_0.y_range

        # move channels on plot graphics
        move_signal(50, 50)
        self.processEvents(0.5)

        # Evaluate
        self.assertTrue(self.plot.locked)
        self.assertTrue(self.plot.channel_selection.isColumnHidden(self.plot.channel_selection.CommonAxisColumn))
        self.assertTupleEqual(sig_0_y_range, self.plot_graph_channel_0.y_range)
        self.assertIsNotNone(self.plot_graph_channel_0.trim_info)

        # Press `Lock` button
        QtTest.QTest.mouseClick(self.plot.lock_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()

        # get trim info
        trim_info = self.plot_graph_channel_0.trim_info

        # move channels on plot graphics
        move_signal(-50, 50)
        self.processEvents(0.5)

        # Evaluate
        self.assertFalse(self.plot.locked)
        self.assertFalse(self.plot.channel_selection.isColumnHidden(self.plot.channel_selection.CommonAxisColumn))
        self.assertNotEqual(sig_0_y_range, self.plot_graph_channel_0.y_range)
        self.assertGreater(self.plot_graph_channel_0.trim_info[0], trim_info[0])  # start
        self.assertGreater(self.plot_graph_channel_0.trim_info[1], trim_info[1])  # stop
        self.assertEqual(self.plot_graph_channel_0.trim_info[2], trim_info[2])  # width

    def test_Plot_ChannelSelection_PushButtons_Zoom(self):
        """
        This method will test 4 buttons: zoom in, zoom out, undo zoom, redo zoom.
        Precondition:
            - Zoom history is clean - trim info must be None
            - Cursor is on center of plot. This step is required for easiest evaluation

        # Event
            - Press button `Zoom in`
            - Press button `Undo zoom`
            - Press button `Redo zoom`
            - Press button `Zoom out`
            - Press button `Zoom in`

        # Evaluate
            - Evaluate that `zoom` action was performed (zoom in will be tested at the end)
            - Evaluate that `undo zoom` action was performed
            - Evaluate that `redo zoom` action was performed and
             channels timestamp trim info is equal with its value before undo zoom action was performed
            - Evaluate that zoom `out action` was performed
            - Evaluate that zoom `in action` was performed

        """
        # Precondition
        self.assertIsNone(self.plot_graph_channel_0.trim_info)
        QtTest.QTest.mouseClick(self.plot.plot.viewport(), QtCore.Qt.MouseButton.LeftButton)

        buttons = self.plot.findChildren(QBtn)
        zoom_in_btn = "Zoom in"
        zoom_out_btn = "Zoom out"
        for btn in buttons:
            if btn.toolTip() == zoom_in_btn:
                zoom_in_btn = btn
                continue
            elif btn.toolTip() == zoom_out_btn:
                zoom_out_btn = btn
                continue

        # buttons was found
        self.assertIsInstance(zoom_in_btn, QBtn)
        self.assertIsInstance(zoom_out_btn, QBtn)

        # Event
        QtTest.QTest.mouseClick(zoom_in_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()
        trim_info_0 = self.plot_graph_channel_0.trim_info

        # Evaluate
        self.assertIsNotNone(trim_info_0)  # zoom was performed

        # Store previous zoom
        prev_zoom = trim_info_0[1] - trim_info_0[0]

        # Event
        QtTest.QTest.mouseClick(self.plot.undo_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()
        trim_info_1 = self.plot_graph_channel_0.trim_info

        # Evaluate
        self.assertGreater(trim_info_1[1] - trim_info_1[0], prev_zoom)

        # Store previous zoom
        prev_zoom = trim_info_1[1] - trim_info_1[0]

        # Event
        QtTest.QTest.mouseClick(self.plot.redo_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()
        trim_info_2 = self.plot_graph_channel_0.trim_info

        # Evaluate
        self.assertLess(trim_info_2[1] - trim_info_2[0], prev_zoom)
        self.assertTupleEqual(trim_info_0, trim_info_2)

        prev_zoom = trim_info_2[1] - trim_info_2[0]

        # Event
        QtTest.QTest.mouseClick(zoom_out_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()
        trim_info_3 = self.plot_graph_channel_0.trim_info

        # Evaluate
        self.assertGreater(trim_info_3[1] - trim_info_3[0], prev_zoom)

        # store previous zoom
        prev_zoom = trim_info_3[1] - trim_info_3[0]

        # Event
        QtTest.QTest.mouseClick(zoom_in_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents()
        trim_info_4 = self.plot_graph_channel_0.trim_info

        # Evaluate
        self.assertLess(trim_info_4[1] - trim_info_4[0], prev_zoom)
        self.assertTupleEqual(trim_info_2, trim_info_4)

    def test_Plot_ChannelSelection_PushButtons_CMD(self):
        """
        This method will test 6 actions from Cmd menu: Home, Honeywell, Fit, Stack, Increase font, Decrease font.

        Precondition:
            - No zoom actions performed. Y range for both signals are identical and timestamp trim info is None

        # Event
            -[0] Press `Cmd` button to open menu, after that select `Stack` action from menu
            -[1] Press `Cmd` button to open menu, after that select `Fit` action from menu
            -[2] Press `Cmd` button to open menu, after that select `Honeywell` action from menu
            -[3] Press `Cmd` button to open menu, after that select `Home` action from menu
            -[4] Press `Cmd` button to open menu, after that check common axis checkboxes
              and select `Stack` action from menu
            -[5] Press `Cmd` button to open menu, after that select `Increase font` action from menu
            -[6] Press `Cmd` button to open menu, after that select `Decrease font` action from menu

        # Evaluate
            -[0] Evaluate that signals Y ranges were modified,
              also Y range for first signal is less than Y range of second signal timestamp trim info is None
            -[1] Evaluate that signals Y ranges for both signals are identical, timestamp trim info is still None
            -[2] Evaluate that signals Y ranges for both signals are identical, timestamp trim info was modified,
              also trim info for both signals are identical
            -[3] Evaluate that signals Y ranges for both signals are identical, also timestamp trim info are identical,
              timestamp range became greater
            -[4] Evaluate that signals Y ranges for both signals are identical, also timestamp trim info are identical

            -[5] Evaluate that widget font size was increased
            -[6] Evaluate that widget font size was decreased
        """

        def click_on_cmd_action(btn: QBtn, action_text: str):
            for n, action in enumerate(btn.menu().actions(), 1):
                if action.text() == action_text:
                    break
            else:
                n = 0
            QtTest.QTest.mouseClick(btn, QtCore.Qt.MouseButton.LeftButton)
            for _ in range(n):
                QtTest.QTest.keyClick(btn.menu(), QtCore.Qt.Key.Key_Down)
                QtTest.QTest.qWait(100)
            QtTest.QTest.keyClick(btn.menu(), QtCore.Qt.Key.Key_Enter)

        # Precondition
        self.assertTupleEqual(self.plot_graph_channel_0.y_range, self.plot_graph_channel_1.y_range)
        self.assertIsNone(self.plot_graph_channel_0.trim_info)
        self.assertIsNone(self.plot_graph_channel_1.trim_info)

        buttons = self.plot.findChildren(QBtn)
        cmd_btn = "Cmd"
        for btn in buttons:
            if btn.text() == cmd_btn:
                cmd_btn = btn
                break

        self.processEvents(0.1)

        with self.subTest("PlotGraphics tests"):
            # Event
            click_on_cmd_action(cmd_btn, "Stack")
            self.processEvents(0.1)

            # Evaluate
            self.assertLess(self.plot_graph_channel_0.y_range[0], self.plot_graph_channel_1.y_range[0])
            self.assertLess(self.plot_graph_channel_0.y_range[1], self.plot_graph_channel_1.y_range[1])

            self.assertIsNone(self.plot_graph_channel_0.trim_info)  # zoom on timestamp axis wasn't performed
            self.assertIsNone(self.plot_graph_channel_1.trim_info)

            # Event
            click_on_cmd_action(cmd_btn, "Fit")
            self.processEvents(0.1)

            # Evaluate
            self.assertTupleEqual(self.plot_graph_channel_0.y_range, self.plot_graph_channel_1.y_range)
            self.assertIsNone(self.plot_graph_channel_0.trim_info)
            self.assertIsNone(self.plot_graph_channel_1.trim_info)

            # Event
            click_on_cmd_action(cmd_btn, "Honeywell")
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual(self.plot_graph_channel_0.y_range[0], self.plot_graph_channel_1.y_range[0])
            self.assertEqual(self.plot_graph_channel_0.y_range[1], self.plot_graph_channel_1.y_range[1])
            self.assertTupleEqual(self.plot_graph_channel_0.trim_info, self.plot_graph_channel_1.trim_info)

            # save previous trim info
            prev_trim_info = self.plot_graph_channel_0.trim_info[1] - self.plot_graph_channel_0.trim_info[0]
            # Event
            click_on_cmd_action(cmd_btn, "Home")
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual(self.plot_graph_channel_0.y_range[0], self.plot_graph_channel_1.y_range[0])
            self.assertEqual(self.plot_graph_channel_0.y_range[1], self.plot_graph_channel_1.y_range[1])
            self.assertLess(
                prev_trim_info, self.plot_graph_channel_0.trim_info[1] - self.plot_graph_channel_0.trim_info[0]
            )
            self.assertTupleEqual(self.plot_graph_channel_0.trim_info, self.plot_graph_channel_1.trim_info)

            # Event
            # Set checked individual axis for both channels
            common_axis_column = self.plot_tree_channel_0.CommonAxisColumn
            if self.plot_tree_channel_0.checkState(common_axis_column) == QtCore.Qt.CheckState.Unchecked:
                self.plot_tree_channel_0.setCheckState(common_axis_column, QtCore.Qt.CheckState.Checked)
            if self.plot_tree_channel_1.checkState(common_axis_column) == QtCore.Qt.CheckState.Unchecked:
                self.plot_tree_channel_1.setCheckState(common_axis_column, QtCore.Qt.CheckState.Checked)

            click_on_cmd_action(cmd_btn, "Stack")
            self.processEvents(0.1)

            # Evaluate
            self.assertTupleEqual(self.plot_graph_channel_0.y_range, self.plot_graph_channel_1.y_range)
            self.assertTupleEqual(self.plot_graph_channel_0.trim_info, self.plot_graph_channel_1.trim_info)

        with self.subTest("text actions tests"):
            prev_text_size = self.plot.font().pointSize()

            # Event
            click_on_cmd_action(cmd_btn, "Increase font")
            self.processEvents(0.1)

            # Evaluate
            self.assertGreater(self.plot.font().pointSize(), prev_text_size)

            prev_text_size = self.plot.font().pointSize()

            # Event
            click_on_cmd_action(cmd_btn, "Decrease font")
            self.processEvents(0.1)

            # Evaluate
            self.assertLess(self.plot.font().pointSize(), prev_text_size)

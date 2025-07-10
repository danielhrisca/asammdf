#!/usr/bin/env python
from unittest import mock

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from asammdf.gui.serde import COLORS
from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget


class TestPlotShortcuts(TestPlotWidget):
    def setUp(self):
        """
        Events:
            - Open measurement file.
            - Create plot sub-window.
            - Ensure that signal isn't with dots.
            - Ensure that there is no grid displayed.
            - Ensure that there are no bookmarks shown.

        Evaluate
            - Evaluate that there is one active sub-window.
        Returns
        -------

        """
        super().setUp()
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Preset for plot
        # Remove dots
        if self.plot.plot.with_dots:
            self.plot.plot.set_dots(False)
        # check if grid is available -> hide grid
        if not self.plot.hide_axes_btn.isFlat():
            QTest.mouseClick(self.plot.hide_axes_btn, Qt.MouseButton.LeftButton)
        # hide bookmarks if it's available
        if self.plot.show_bookmarks:
            self.plot.toggle_bookmarks(hide=True)
        # Ensure that plot is not` in Focused mode
        if self.plot.focused_mode:
            self.plot.toggle_focused_mode()
        # Ensure that plot cursor is not with circle
        if self.plot.plot.cursor1.show_circle:
            self.plot.plot.cursor1.show_circle = False

        # get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.plot))

        self.processEvents()

    def tearDown(self):
        # Ensure that at the end, button "No" is pressed for MessageBox question window
        with mock.patch("asammdf.gui.widgets.mdi_area.MessageBox.question") as mo_question:
            mo_question.return_value = QMessageBox.StandardButton.No
            super().tearDown()

    def test_statistics_shortcut(self):
        """
        Test Scope:
            To check if after pressing key "M", info about the selected channel is displayed properly,
                and after deleting all channels,
                buffer is clean and info about the last channel isn't displayed,

        Events:.
            - Select 3 signals and create a plot.
            - Press Key "M".
            - Select First channel.
            - Pres key "Down".
            - Delete all channels.

        Evaluate:
            - Evaluate that Statistics isn't visible by default.
            - Evaluate that Statistics is visible after pressing key "M".
            - Evaluate that displayed info is related to the first channel.
            - Evaluate that displayed info is related to the second channel.
            - Evaluate that buffer is clear, so the displayed message is the default one.
        """
        # Setup
        default_message = "Please select a single channel"
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35, 36, 37]))
        # Evaluate if info isn't visible before pressing key M
        self.assertFalse(self.plot.info.isVisible())

        # Event
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["statistics"]))
        # Evaluate if info is visible after pressing key M
        self.assertTrue(self.plot.info.isVisible())

        # click on a first channel
        self.mouseClick_WidgetItem(self.channels[0])
        # Evaluate
        with self.subTest("test_key_M_click_on_first_channel"):
            self.assertEqual(self.plot.info._name, self.channels[0].name)
            self.assertEqual(self.plot.info.color, self.channels[0].color.name())

        # Press key "Down"
        QTest.keySequence(self.plot.channel_selection, QKeySequence("Down"))
        # Evaluate
        with self.subTest("test_key_M_press_key_Down_on_channel_selection"):
            self.assertEqual(self.plot.info._name, self.channels[1].name)
            self.assertEqual(self.plot.info.color, self.channels[1].color.name())

        # delete all channels
        QTest.keySequence(self.plot, QKeySequence("Ctrl+A"))
        QTest.keySequence(self.plot.channel_selection, QKeySequence("Delete"))
        # Evaluate
        with self.subTest("test_key_M_delete_all_channels"):
            # The default message is displayed instead of channel name
            self.assertNotEqual(self.plot.info._name, default_message)

    def test_focused_mode_shortcut(self):
        """
        Test Scope:
            To check if is displayed only selected channel after pressing key "2"
        Events:
            - Select 3 signals and create a plot
            - Press Key "2"
            - Select first channel
            - Pres key "Down"
            - Select third channel
        Evaluate:
            - Evaluate that three signals are available.
            - Evaluate that plot is not black and contains colors of all 3 channels.
            - Evaluate that plot is black after pressing key "2".
            - Evaluate that plot contains only color of first channel after clicking on first channel.
            - Evaluate that plot contains only color of second channel after pressing key "Down".
            - Evaluate that plot contains only color of third channel after clicked on third channel.
            - Evaluate that plot contains colors of all 3 channels after pressing key "2".
        """
        # Setup
        if not self.plot.focused_mode_btn.isFlat():
            QTest.mouseClick(self.plot.focused_mode_btn, Qt.MouseButton.LeftButton)
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35, 36, 37]))
        colors = [ch.color.name() for ch in self.channels]

        # Evaluate
        # If all colors exists
        self.assertTrue(self.is_not_blinking(self.plot.plot, set(colors)))

        # case 0
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["focused_mode"]))
        self.processEvents(0.1)
        # Evaluate
        self.assertTrue(Pixmap.is_black(self.plot.plot.viewport().grab()))

        # case 1
        self.mouseClick_WidgetItem(self.channels[0])

        # Evaluate
        self.assertTrue(self.is_not_blinking(self.plot.plot, {colors[0]}))
        self.assertFalse(
            {colors[1], colors[2]}.issubset(Pixmap.color_names_exclude_defaults(self.plot.plot.viewport().grab()))
        )

        # case 2
        QTest.keyClick(self.plot.channel_selection, Qt.Key.Key_Down)

        # Evaluate
        self.assertTrue(self.is_not_blinking(self.plot.plot, {colors[1]}))
        self.assertFalse(
            {colors[0], colors[2]}.issubset(Pixmap.color_names_exclude_defaults(self.plot.plot.viewport().grab()))
        )

        # case 3
        self.mouseClick_WidgetItem(self.channels[2])

        # Evaluate
        self.assertTrue(self.is_not_blinking(self.plot.plot, {colors[2]}))
        self.assertFalse(
            {colors[1], colors[0]}.issubset(Pixmap.color_names_exclude_defaults(self.plot.plot.viewport().grab()))
        )

        # case 4 - exit focused mode
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["focused_mode"]))

        self.processEvents(0.1)
        # Evaluate
        self.assertTrue(self.is_not_blinking(self.plot.plot, set(colors)))

    def test_ascii__bin__hex__physical_shortcuts(self):
        """
        Test Scope:
            Check if values are converted to hex, bin, phys and ascii after pressing the combination of key
                "Ctrl+<H>|<B>|<P>|<T>"

        Events:
            - Display 1 signal on plot.
            - Press "Ctrl+B".
            - Press "Ctrl+H".
            - Press "Ctrl+P".
            - Press "Ctrl+T".

        Evaluate:
            - Evaluate that unit is changed to hex after pressing key "Ctrl+H".
            - Evaluate that unit is changed to bin after pressing key "Ctrl+B".
            - Evaluate that unit is changed to phys (int value) after pressing key "Ctrl+P".
            - Evaluate that unit is changed to ascii after pressing key "Ctrl+T".
        """
        # add channels to plot and select it
        self.assertIsNotNone(self.add_channels([35]))
        self.mouseClick_WidgetItem(self.channels[0])

        # Store text from the selected channel
        physical = self.plot.selected_channel_value.text()
        physical_hours = int(physical.split(" ")[0])
        # Press "Ctrl+B"
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["bin"]))
        bin_ = self.plot.selected_channel_value.text()
        bin_hours = int(bin_.split(" ")[0].replace(".", ""), 2)

        # Evaluate
        self.assertNotEqual(physical, bin_)
        self.assertEqual(physical_hours, bin_hours)

        # Press "Ctrl+H"
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["hex"]))
        hex_ = self.plot.selected_channel_value.text()
        # Convert hex value to Int
        hex_hours = int(hex_.split(" ")[0], 16)

        # Evaluate
        self.assertNotEqual(physical, hex_)
        self.assertIn("0x", hex_)
        self.assertEqual(physical_hours, hex_hours)

        # Press "Ctrl+P"
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["physical"]))
        new_physical = self.plot.selected_channel_value.text()
        new_physical_hours = int(new_physical.split(" ")[0])

        # Evaluate
        self.assertEqual(physical, new_physical)
        self.assertEqual(physical_hours, new_physical_hours)

        # Press "Ctrl+P"
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["ascii"]))
        ascii_value = self.plot.selected_channel_value.text()
        ascii_int = ascii(ascii_value.split(" ")[0])
        ascii_int = ascii_int.split("'")[1]
        ascii_int = "0" + ascii_int.split("\\")[1]
        ascii_int = int(ascii_int, 16)

        # Evaluate
        self.assertNotEqual(physical, ascii_value)
        self.assertEqual(physical_hours, ascii_int)

    def test_raw__scaled_samples_shortcuts(self):
        """
        Test Scope:
            Check functionality of key "Alt+R" and "Alt+S". They must convert samples to raw and scaled forms.
        Events:
            - Display 1 signal on plot
            - Press "Alt+R"
            - Press "Alt+S"
        Evaluate:
            - Evaluate that signal mode is raw and line style is DashLine after pressing key "Alt+R"
            - Evaluate that signal mode is phys and line style is SolidLine after pressing key "Alt+S"
        """
        # add channels to plot
        self.assertIsNotNone(self.add_channels(["ASAM.M.SCALAR.SBYTE.LINEAR_MUL_2"]))
        phys_value = self.plot.selected_channel_value.text().split()[0]
        expected_raw_value = float(phys_value) / self.channels[0].signal.conversion.a
        # Press "Alt+R"
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["raw_samples"]))
        self.processEvents(1)
        # Evaluate
        # Signal mode = raw
        self.assertEqual(self.channels[0].mode, "raw")
        # Signal line style = Dash line
        self.assertEqual(self.plot.plot.signals[0].pen.style(), Qt.PenStyle.DashLine)
        # Raw value
        self.assertEqual(expected_raw_value, float(self.plot.selected_channel_value.text()))

        # Press "Alt+S"
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["scaled_samples"]))
        self.processEvents(1)
        # Evaluate
        # Signal mode = raw
        self.assertEqual(self.channels[0].mode, "phys")
        # Signal line style = Dash line
        self.assertEqual(self.plot.plot.signals[0].pen.style(), Qt.PenStyle.SolidLine)

    def test_insert_bookmark_shortcut(self):
        """
        Test Scope:
            Check if bookmark is created after pressing key "Ctrl+I"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Press "Ctrl+I" and hit "Ok"
        Evaluate:
            - Evaluate that bookmarks are not displayed before pressing "Ctrl+I"
            - Evaluate that bookmarks are displayed after pressing "Ctrl+I"
                and the message of the last bookmark is the first element of the returned list of mock object
        """
        QTest.mouseClick(
            self.plot.plot.viewport(), Qt.MouseButton.LeftButton, pos=self.plot.plot.viewport().geometry().center()
        )
        self.processEvents()
        timestamp = self.plot.plot.cursor1.value()
        # mock for bookmark
        with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QInputDialog.getMultiLineText") as mo_getMultiLineText:
            mo_getMultiLineText.return_value = [self.id(), True]
            # Press "Ctrl+I"
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["insert_bookmark"]))

        mo_getMultiLineText.assert_called()

        # Destroy current widget
        with mock.patch("asammdf.gui.widgets.mdi_area.MessageBox.question") as mo_question:
            mo_question.return_value = QMessageBox.StandardButton.Yes
            self.destroy(self.widget)

        # Open file with new bookmark in new File Widget
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        self.processEvents(0.1)
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()
        self.processEvents(0.1)

        pg_colors = Pixmap.color_names_exclude_defaults(self.plot.plot.viewport().grab())
        bookmarks_colors = COLORS[: len(COLORS) - len(self.plot.plot.bookmarks) - 1 : -1]

        # Evaluate
        self.assertTrue(self.plot.show_bookmarks)
        self.assertEqual(self.plot.plot.bookmarks[len(self.plot.plot.bookmarks) - 1].message, self.id())
        self.assertEqual(self.plot.plot.bookmarks[len(self.plot.plot.bookmarks) - 1].value(), timestamp)
        for bookmark, color in zip(self.plot.plot.bookmarks, bookmarks_colors, strict=False):
            self.assertEqual(bookmark.color, color)
            self.assertIn(color, pg_colors)

    def test_toggle_bookmarks_shortcut(self):
        """
        Test Scope:
            Check functionality of key "Ctrl+I". It must toggle bookmarks visibility on plot.
        Events:
            - Display 1 signal on plot
            - Press 3 times "Alt+I"
        Evaluate:
            - Evaluate that bookmarks are not displayed before pressing "Alt+I"
            - Evaluate that bookmarks are displayed after pressing "Alt+I" first time
            - Evaluate that bookmarks are not displayed after pressing "Alt+I" second time
            - Evaluate that bookmarks are displayed after pressing "Alt+I" third time
        """
        # Evaluate
        self.assertFalse(self.plot.show_bookmarks)
        # Press "Al+I"
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["toggle_bookmarks"]))
        # Evaluate
        self.assertTrue(self.plot.show_bookmarks)
        # Press "Al+I"
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["toggle_bookmarks"]))
        # Evaluate
        self.assertFalse(self.plot.show_bookmarks)

    def test_edit_y_axis_scaling_shortcut(self):
        """
        Test Scope:
            Check if signal is changed his Y limits by setting it with ScaleDialog object called with "Ctrl+G"
        Events:
            - Create a plot widget with signals
            - Click on first channel
            - Press Ctrl+G
            - Set scaling and offset values
            - Hit "Ok"
        Evaluate:
            - Evaluate that Signal is on top and bottom part of plot
            - Evaluate that mock object was called
            - Evaluate that after pressing Ctrl+G, selected channel is situated on top and not on bottom part of plot
            - Evaluate if y_range of signal is identical with expected range
        """
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35]))
        self.processEvents()

        # Setup
        offset = 50.0
        scale = self.plot.plot.y_axis.range[1] - self.plot.plot.y_axis.range[0]
        y_bottom = -offset * scale / 100
        y_top = y_bottom + scale

        expected_y_range = y_bottom, y_top

        # Top
        pixmap = self.plot.plot.grab(QRect(0, 0, self.plot.plot.width(), int(self.plot.plot.height() / 2)))
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        # Bottom
        pixmap = self.plot.plot.grab(
            QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.width(), int(self.plot.plot.height() / 2))
        )
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[0].color.name()))

        # Click on the first channel
        self.mouseClick_WidgetItem(self.channels[0])
        with mock.patch("asammdf.gui.widgets.plot.ScaleDialog") as mo_ScaleDialog:
            mo_ScaleDialog.return_value.offset.value.return_value = offset
            mo_ScaleDialog.return_value.scaling.value.return_value = scale
            # Press Ctrl+Shift+C
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["edit_y_axis_scaling"]))

        # Evaluate

        # Evaluate that ScaleDialog object was created, and it's exec() method was called
        mo_ScaleDialog.return_value.exec.assert_called()
        self.processEvents(1)

        # Evaluate y_range tuple
        self.assertTupleEqual(self.plot.plot.signals[0].y_range, expected_y_range)

        # Top
        pixmap = self.plot.plot.grab(QRect(0, 0, self.plot.plot.width(), int(self.plot.plot.height() / 2)))
        # Evaluate plot, top half of plot must contain channel color
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        # Bottom
        pixmap = self.plot.plot.grab(
            QRect(0, int(self.plot.plot.height() / 2 + 2), self.plot.plot.width(), int(self.plot.plot.height() / 2 - 2))
        )
        # Evaluate plot, bottom half of plot must not contain channel color
        self.assertFalse(Pixmap.has_color(pixmap, self.channels[0].color.name()))

    def test_color_shortcut(self):
        """
        Test Scope:
            - Ensure that channel color is changed.

        Events:
            - Open Plot with two channels.
            - Press C.
            - Select one channel -> press C.
            - Select all channels -> press C.

        Evaluate:
            - Evaluate that color dialog is not open if the channel is not selected.
            - Evaluate that channel color is changed only for the selected channel.
        """
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35, 36, 37]))
        with self.subTest("test_WOSelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["set_color"]))
                mo_getColor.assert_not_called()

        with self.subTest("test_1SelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                # Setup
                self.mouseClick_WidgetItem(self.channels[1])
                previous_color = self.channels[1].color.name()
                color = QColor("magenta")
                mo_getColor.return_value = color
                # Event
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["set_color"]))
                # Evaluate
                mo_getColor.assert_called()
                self.assertNotEqual(previous_color, self.channels[1].color.name())
                self.assertEqual(color.name(), self.channels[1].color.name())

        with self.subTest("test_AllSelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                # Setup
                color = QColor("black")
                mo_getColor.return_value = color
                # Set selected both channels
                QTest.keySequence(self.plot.channel_selection, QKeySequence("Ctrl+A"))
                # store previous colors of channels
                previous_ch_35_color = self.channels[0].color.name()
                previous_ch_36_color = self.channels[1].color.name()
                previous_ch_37_color = self.channels[2].color.name()
                color = QColor("black")
                mo_getColor.return_value = color
                # Event
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["set_color"]))
                # Evaluate
                mo_getColor.assert_called()

                self.assertNotEqual(previous_ch_35_color, self.channels[0].color.name())
                self.assertNotEqual(previous_ch_36_color, self.channels[1].color.name())
                self.assertNotEqual(previous_ch_37_color, self.channels[2].color.name())
                self.assertEqual(color.name(), self.channels[0].color.name())
                self.assertEqual(color.name(), self.channels[1].color.name())
                self.assertEqual(color.name(), self.channels[2].color.name())

    def test_copy__paste_channel_structure_shortcut(self):
        """
        Test Scope:
            - Ensure that selected channel is copied to clipboard and pasted into a plot.
        Events:
            - Open Plot with 2 channels
            - Select first channel
            - Press Ctrl+C
            - Press Ctrl+V
            - Open new plot window
            - Press Ctrl+V
        Evaluate:
            - Evaluate that is one more channel in channel selection area with the same properties as selected channel
            - Evaluate that channel is inserted in new window with the same properties
        """
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35, 36]))
        self.mouseClick_WidgetItem(self.channels[0])
        # Press Ctrl+C -> Ctrl+V
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["copy_channel_structure"]))
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["paste_channel_structure"]))
        self.processEvents()
        # Evaluate that now is three channels available
        self.assertEqual(3, self.plot.channel_selection.topLevelItemCount())
        original_name = self.channels[0].name
        original_color_name = self.channels[0].color.name()
        original_origin_uuid = self.channels[0].origin_uuid
        original_signal = self.channels[0].signal
        replica = self.plot.channel_selection.topLevelItem(1)
        # Evaluate channels essential attributes
        self.assertEqual(original_name, replica.name)
        self.assertEqual(original_color_name, replica.color.name())
        self.assertEqual(original_origin_uuid, replica.origin_uuid)
        self.assertEqual(original_signal, replica.signal)

        # Uncheck channels
        ch = self.find_channel(self.widget.channels_tree, self.channels[0].name)
        ch.setCheckState(0, Qt.CheckState.Unchecked)
        ch = self.find_channel(self.widget.channels_tree, self.channels[1].name)
        ch.setCheckState(0, Qt.CheckState.Unchecked)
        # Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        # Evaluate that second window is created
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
        # Second window
        self.plot = self.widget.mdi_area.subWindowList()[1].widget()
        # Press Ctrl+V
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["paste_channel_structure"]))
        self.processEvents()
        # Evaluate that now is three channels available
        self.assertEqual(1, self.plot.channel_selection.topLevelItemCount())

        replica = self.plot.channel_selection.topLevelItem(0)
        # Evaluate channels essential attributes
        self.assertEqual(original_name, replica.name)
        self.assertEqual(original_color_name, replica.color.name())
        self.assertEqual(original_origin_uuid, replica.origin_uuid)
        self.assertEqual(original_signal, replica.signal)

    def test_copy__paste_display_properties_shortcuts(self):
        """
        Test Scope:
            Check if only display properties of selected channels is copied on another channel
                by shortcuts "Ctrl+ShiftC"->"Ctrl+Shift+V"
        Events:
            - Select 2 signals and create a plot
            - Click on first channel
            - Press key Ctrl+Shift+C
            - Click on second channel
            - Press key Ctrl+Shift+V
        Evaluate:
            - Evaluate that names and colors are different for both channels
            - Evaluate that after pressing shortcuts combination, names are different,
                but colors are the same for both channels
        """
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35, 36, 37]))
        # Evaluate precondition
        self.assertNotEqual(self.channels[1].name, self.channels[2].name)
        self.assertNotEqual(self.channels[1].color.name(), self.channels[2].color.name())

        # Click on channel 36
        self.mouseClick_WidgetItem(self.channels[1])
        # Press Ctrl+Shift+C
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["copy_display_properties"]))
        self.mouseClick_WidgetItem(self.channels[2])
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["paste_display_properties"]))

        # Evaluate
        self.assertNotEqual(self.channels[1].name, self.channels[2].name)
        self.assertEqual(self.channels[1].color.name(), self.channels[2].color.name())
        self.assertEqual(self.channels[1].precision, self.channels[2].precision)
        self.assertEqual(self.channels[1].format, self.channels[2].format)
        self.assertEqual(self.channels[1].signal.y_link, self.channels[2].signal.y_link)
        self.assertEqual(self.channels[1].signal.individual_axis, self.channels[2].signal.individual_axis)
        self.assertListEqual(self.channels[1].ranges, self.channels[2].ranges)
        self.assertEqual(self.channels[1].signal.y_range[0], self.channels[2].signal.y_range[0])
        self.assertEqual(self.channels[1].signal.y_range[1], self.channels[2].signal.y_range[1])

    def test_increase__decrease_font_shortcuts(self):
        """
        Test scope:
            Ensure that Ctrl+[ and Ctrl+] will change font size

        Events:
            - Press Ctrl+[
            - Press Ctrl+]

        Evaluate:
         - Evaluate that font size was increased after shortcut Ctrl+[ was pressed
         - Evaluate that font size was decreased after shortcut Ctrl+] was pressed
        """
        font_size = self.plot.font().pointSize()
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["decrease_font"]))
        self.assertLess(font_size, self.plot.font().pointSize())

        font_size = self.plot.font().pointSize()
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["increase_font"]))
        self.assertGreater(font_size, self.plot.font().pointSize())

    def test_navigate_trough_zoom_history_shortcuts(self):
        """
        Test Scope:
            Test functionality of keys Backspace, Shift+Backspace, Shift+W.
            Those keys must change zoom history

        Events:
            - Add one channel to plot
            - Perform some zoom actions
            - Press key Backspace
            - Press Shift+backspace
            - Press Shift+W

        Evaluate:
            - Evaluate that after pressing keys "Shift+W", y_range is equal with y_range at start
            - Evaluate that after pressing keys "Shift+Backspace", y_range is equal with y_range after zoom action
            - Evaluate that after pressing key "Backspace", y_range isn't equal with previous ranges
        """
        # Setup
        if not self.plot.focused_mode_btn.isFlat():
            QTest.mouseClick(self.plot.focused_mode_btn, Qt.MouseButton.LeftButton)
        self.plot.plot.viewbox.menu.set_x_zoom_mode()
        self.plot.plot.viewbox.menu.set_y_zoom_mode()
        self.plot.plot.viewbox.setMouseMode(self.plot.plot.viewbox.PanMode)
        self.plot.plot.setFocus()
        self.processEvents(0.1)
        # add channel to plot
        self.assertIsNotNone(self.add_channels([35]))

        x = round(self.plot.plot.width() / 2)
        y = round(self.plot.plot.height() / 2)
        # Y range at start
        y_range = self.plot.plot.signals[0].y_range

        # Events
        self.mouseClick_WidgetItem(self.channels[0])
        # Click in the center of the plot
        QTest.mouseClick(
            self.plot.plot.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(int(x), int(y)),
        )
        self.processEvents(0.01)
        # Rotate mouse wheel
        self.wheel_action(self.plot.plot.viewport(), x, y, 2)
        self.processEvents(0.01)

        # Click on plot
        QTest.mouseClick(
            self.plot.plot.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(int(x * 0.5), int(y * 0.8)),
        )
        self.processEvents(0.01)

        # Rotate mouse wheel
        self.wheel_action(self.plot.plot.viewport(), x, y, -3)
        self.processEvents(1)
        self.processEvents(0.1)
        new_y_range = self.plot.plot.signals[0].y_range

        # Evaluate
        self.assertNotEqual(new_y_range, y_range)

        # click on Backspace
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["undo_zoom"]))
        self.processEvents()
        undo_zoom_y_range = self.plot.plot.signals[0].y_range

        # click on Shift + Backspace
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["redo_zoom"]))
        self.processEvents()
        redo_zoom_y_range = self.plot.plot.signals[0].y_range

        # click on Shift + W
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["x_default_zoom"]))
        self.processEvents()
        default_zoom_y_range = self.plot.plot.signals[0].y_range

        # Evaluate
        # Shift+W
        self.assertEqual(y_range[0], default_zoom_y_range[0])
        self.assertEqual(y_range[1], default_zoom_y_range[1])
        # Shift + Backspace
        self.assertEqual(new_y_range[0], redo_zoom_y_range[0])
        self.assertEqual(new_y_range[1], redo_zoom_y_range[1])
        # Backspace
        self.assertNotIn(undo_zoom_y_range[0], [y_range[0], new_y_range[0]])
        self.assertNotIn(undo_zoom_y_range[1], [y_range[1], new_y_range[1]])

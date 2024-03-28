#!/usr/bin/env python
from math import floor
from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget
from unittest import mock

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from asammdf.gui.widgets.plot import PlotGraphics


class TestPlotShortcuts(TestPlotWidget):
    def setUp(self):
        """
        Events:
            Open test measurement file
            Set sort method: Natural sort
            Create new plot widget
            Add one channel to plot
        Evaluate:
            - Evaluate that one widget was created
            - Evaluate that one channel was added to plot
        """
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        self.add_channels([35], self.plot)

        # get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.plot))
        self.processEvents()

    def test_Plot_Plot_Shortcuts(self):
        """
        Test Scope:
            Check if all shortcut events is called
        """
        with self.subTest("test_Key_M"):
            expected_mo_call_count = 2
            # At start Statistics is hidden
            self.assertFalse(self.plot.info.isVisible())
            with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QSplitter.setSizes") as mo_setSizes:
                # press key "M"
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["statistics"]))
                self.assertTrue(self.plot.info.isVisible())
                # press key "M"
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["statistics"]))
                self.assertFalse(self.plot.info.isVisible())
            self.assertEqual(mo_setSizes.call_count, expected_mo_call_count)

        with self.subTest("test_Key_2"):
            # Setup
            previous_focused_mode_btn = self.plot.focused_mode_btn.isFlat()
            # Press 2
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["focused_mode"]))
            self.assertNotEqual(previous_focused_mode_btn, self.plot.focused_mode_btn.isFlat())

            # Update
            previous_focused_mode_btn = self.plot.focused_mode_btn.isFlat()
            # Press 2
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["focused_mode"]))
            self.assertNotEqual(previous_focused_mode_btn, self.plot.focused_mode_btn.isFlat())

        with mock.patch("asammdf.gui.widgets.plot.Plot.item_by_uuid"):
            with mock.patch.object(PlotGraphics, "get_axis") as mo_get_axis:
                mo_get_axis.assert_not_called()
                expected_mo_call_count = 0
                with self.subTest("test_Ctrl_B"):
                    QTest.keySequence(self.plot, QKeySequence(self.shortcuts["bin"]))
                    expected_mo_call_count += 1
                    self.assertEqual(mo_get_axis.call_count, expected_mo_call_count)
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "bin")
                #    self.assertEqual(mo_get_axis.return_value.format, "bin")
                with self.subTest("test_Ctrl_H"):
                    QTest.keySequence(self.plot, QKeySequence(self.shortcuts["hex"]))
                    expected_mo_call_count += 1
                    self.assertEqual(mo_get_axis.call_count, expected_mo_call_count)
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "hex")
                #    self.assertEqual(mo_get_axis.return_value.format, "hex")
                with self.subTest("test_Ctrl_P"):
                    QTest.keySequence(self.plot, QKeySequence(self.shortcuts["physical"]))
                    expected_mo_call_count += 1
                    self.assertEqual(mo_get_axis.call_count, expected_mo_call_count)
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "phys")
                #   self.assertEqual(mo_get_axis.return_value.format, "phys")
                with self.subTest("test_Ctrl_T"):
                    QTest.keySequence(self.plot, QKeySequence(self.shortcuts["ascii"]))
                    expected_mo_call_count += 1
                    self.assertEqual(mo_get_axis.call_count, expected_mo_call_count)
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "ascii")
                #   self.assertEqual(mo_get_axis.return_value.format, "ascii")

        with self.subTest("test_Key_R"):
            with mock.patch("asammdf.gui.widgets.tree.ChannelsTreeItem.set_prefix") as mo_set_prefix:
                with mock.patch("asammdf.gui.widgets.tree.ChannelsTreeItem.set_value") as mo_set_value:
                    QTest.keySequence(self.plot, QKeySequence(self.shortcuts["range"]))
            mo_set_prefix.assert_called()
            mo_set_value.assert_called()

        with self.subTest("test_Key_Ctrl_R"):
            with mock.patch("asammdf.gui.widgets.tree.ChannelsTreeWidget.keyPressEvent") as mo_keyPressEvent:
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["set_color_range"]))
            mo_keyPressEvent.assert_called()

        with mock.patch.object(self.plot.plot, "viewbox"):
            with mock.patch.object(self.plot.plot, "update"):
                with mock.patch.object(self.plot.plot, "get_axis") as mo_get_axis:
                    # Evaluate
                    with self.subTest("test_key_Alt_R"):
                        # Press "Alt+R"
                        QTest.keySequence(self.plot.plot, QKeySequence(self.shortcuts["raw_samples"]))
                        # Signal mode = raw
                        self.assertEqual(mo_get_axis.return_value.mode, "raw")
                        self.assertIsNone(mo_get_axis.return_value.picture)
                        mo_get_axis.return_value.update.assert_called()

                    with self.subTest("test_key_Alt_S"):
                        # Press "Alt+S"
                        QTest.keySequence(self.plot.plot, QKeySequence(self.shortcuts["scaled_samples"]))
                        # Signal mode = phys
                        self.assertEqual(mo_get_axis.return_value.mode, "phys")
                        self.assertIsNone(mo_get_axis.return_value.picture)
                        mo_get_axis.return_value.update.assert_called()

        # Final evaluation
        self.assertEqual(mo_get_axis.return_value.update.call_count, 2)
        self.assertEqual(mo_get_axis.call_count, 6)

        with self.subTest("test_Key_I"):
            with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QInputDialog.getMultiLineText") as mo_getMultiLineText:
                mo_getMultiLineText.return_value = self.id(), True
                with mock.patch.object(self.plot.plot, "bookmarks") as mo_bookmarks:
                    with mock.patch("asammdf.gui.widgets.plot.Bookmark") as mo_Bookmark:
                        with mock.patch.object(self.plot.plot, "viewbox") as mo_viewbox:
                            QTest.keySequence(self.plot.plot, QKeySequence(self.shortcuts["insert_bookmark"]))

            mo_bookmarks.append.assert_called_with(mo_Bookmark.return_value)
            mo_viewbox.addItem.assert_called_with(mo_bookmarks.__getitem__())
            self.assertEqual(mo_Bookmark.call_args[1]["message"], self.id())
            mo_getMultiLineText.assert_called()

        with self.subTest("test_Key_Alt_I"):
            bookmark_btn_previous_state = self.plot.bookmark_btn.isFlat()
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["toggle_bookmarks"]))
            self.assertEqual(self.plot.bookmark_btn.isFlat(), not bookmark_btn_previous_state)

        with self.subTest("test_Key_G"):
            offset = 10.0
            scale = 100.0
            y_bottom = -offset * scale / 100
            expected_y_range = (y_bottom, y_bottom + scale)
            # Adds one channel to plot
            self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
            self.processEvents()
            self.plot.channel_selection.topLevelItem(0).setSelected(True)
            self.processEvents()
            with mock.patch("asammdf.gui.widgets.plot.ScaleDialog") as mo_ScaleDialog:
                mo_ScaleDialog.return_value.exec.return_value = True
                mo_ScaleDialog.return_value.offset.value.return_value = offset
                mo_ScaleDialog.return_value.scaling.value.return_value = scale

                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["edit_y_axis_scaling"]))

            mo_ScaleDialog.return_value.exec.assert_called()
            self.assertTupleEqual(self.plot.plot.signals[0].y_range, expected_y_range)

        with mock.patch("asammdf.gui.widgets.plot.ChannelsTreeWidget.keyPressEvent") as mo_keyPressEvent:
            with self.subTest("test_Key_C"):
                # Event
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["set_color"]))
                # Evaluate
                mo_keyPressEvent.assert_called()

            with self.subTest("test_Ctrl_C"):
                # Event
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["copy_channel_structure"]))
                # Evaluate
                mo_keyPressEvent.assert_called()

            with self.subTest("test_Ctrl_Shift_C"):
                # Event
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["copy_display_properties"]))
                # Evaluate
                mo_keyPressEvent.assert_called()

        # Setup
        # Adds one channel to plot
        with mock.patch("asammdf.gui.widgets.plot.ChannelsTreeWidget.keyPressEvent") as mo_keyPressEvent:
            with self.subTest("test_Ctrl_V"):
                # Event
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["paste_channel_structure"]))
                # Evaluate
                mo_keyPressEvent.assert_called()

            with self.subTest("test_Ctrl_Shift_V"):
                # Event
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["paste_display_properties"]))
                # Evaluate
                mo_keyPressEvent.assert_called()

        with self.subTest("Shortcut_Key_Ctrl_BracketLeft"):
            # Setup
            # Adds one channel to plot
            self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
            self.processEvents()
            initial_size = self.plot.font().pointSize()
            with mock.patch("asammdf.gui.widgets.plot.ChannelsTreeWidget.set_font_size"):
                with mock.patch.object(self.plot.plot, "y_axis"):
                    with mock.patch.object(self.plot.plot, "x_axis"):
                        # Event
                        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["increase_font"]))
            # Evaluate
            self.assertEqual(initial_size, self.plot.font().pointSize() + 1)

        with self.subTest("Shortcut_Key_Ctrl_BracketRight"):
            # Setup
            # Adds one channel to plot
            self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
            self.processEvents()
            initial_size = self.plot.font().pointSize()
            with mock.patch("asammdf.gui.widgets.plot.ChannelsTreeWidget.set_font_size"):
                with mock.patch.object(self.plot.plot, "y_axis"):
                    with mock.patch.object(self.plot.plot, "x_axis"):
                        # Event
                        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["decrease_font"]))
            # Evaluate
            self.assertEqual(initial_size, self.plot.font().pointSize() - 1)

        with self.subTest("Shortcut_Key_Backspace"):
            # Setup
            # Adds one channel to plot
            self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
            self.processEvents()
            with mock.patch.object(self.plot, "undo_zoom") as mo_undo_zoom:
                # Event
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["undo_zoom"]))
            # Evaluate
            mo_undo_zoom.assert_called()

        with self.subTest("Shortcut_Key_Shift_Backspace"):
            # Setup
            # Adds one channel to plot
            self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
            self.processEvents()
            with mock.patch.object(self.plot, "redo_zoom") as mo_redo_zoom:
                # Event
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["redo_zoom"]))
            # Evaluate
            mo_redo_zoom.assert_called()

        with self.subTest("Shortcut_Key_Shift_W"):
            # Adds one channel to plot
            self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
            self.processEvents()
            with mock.patch.object(self.plot.plot, "set_y_range") as mo_set_y_range:
                with mock.patch.object(self.plot.plot, "viewbox") as mo_viewbox:
                    # Event
                    QTest.keySequence(self.plot, QKeySequence(self.shortcuts["x_default_zoom"]))
            # Evaluate
            mo_set_y_range.assert_called()
            mo_viewbox.setXRange.assert_called()
            self.assertFalse(self.plot.undo_btn.isEnabled())


class TestPlotShortcutsFunctionality(TestPlotWidget):
    def setUp(self):
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()
        # Remove dots
        if self.plot.plot.with_dots:
            self.plot.plot.set_dots(False)
        self.processEvents()
        # check if grid is available -> hide grid
        if not self.plot.hide_axes_btn.isFlat():
            QTest.mouseClick(self.plot.hide_axes_btn, Qt.MouseButton.LeftButton)
        # hide bookmarks if it's available
        if self.plot.show_bookmarks:
            self.plot.toggle_bookmarks(hide=True)

        # get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.plot))

        self.processEvents()

    def tearDown(self):
        super().tearDown()
        self.widget.destroy()
        with mock.patch("asammdf.gui.widgets.mdi_area.MessageBox.question") as mo_question:
            mo_question.return_value = QMessageBox.No

    def test_Plot_Plot_Shortcut_Key_M(self):
        """
        Test Scope:
            To check if after pressing key "M", info about selected channel is displayed properly
                and after deleting all channels, buffer is clean and info about last channel isn't displayed

        Events:.
            - Select 3 signals and create a plot
            - Press Key "M"
            - Select First channel
            - Pres key "Down"
            - Delete all channels

        Evaluate:
            - Evaluate that Statistics isn't visible by default
            - Evaluate that Statistics is visible after pressing key "M"
            - Evaluate that displayed info is related to first channel
            - Evaluate that displayed info is related to second channel
            - Evaluate that buffer is clear, so displayed message is the default one
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
            # default message is displayed instead of channel name
            self.assertNotEqual(self.plot.info._name, default_message)

    def test_Plot_Plot_Shortcut_Keys_Ctrl_H_Ctrl_B_Ctrl_P_Ctrl_T(self):
        """
        Test Scope:
            Check if values is converted to hex, bin, phys and ascii after pressing combination of key
                "Ctrl+<H>|<B>|<P>|<T>"

        Events:
            - Display 1 signal on plot
            - Press "Ctrl+B"
            - Press "Ctrl+H"
            - Press "Ctrl+P"
            - Press "Ctrl+T"

        Evaluate:
            - Evaluate that unit is changed to hex after pressing key "Ctrl+H"
            - Evaluate that unit is changed to bin after pressing key "Ctrl+B"
            - Evaluate that unit is changed to phys (int value) after pressing key "Ctrl+P"
            - Evaluate that unit is changed to ascii after pressing key "Ctrl+T"
        """
        # add channels to plot and select it
        self.assertIsNotNone(self.add_channels([35]))
        self.mouseClick_WidgetItem(self.channels[0])

        # Store text from selected channel
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

    def test_Plot_Plot_Shortcut_Key_Ctrl_R(self):
        """
        Test Scope:
            Check if color range is triggered after pressing key Ctrl+R
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Select signal
            - Press "Ctrl+R" -> set ranges from 0 to 40% of y value and colors green and red -> apply
            - Click on unchanged color part of signal on plot
            - Click on changed color part of signal on plot
            - Click Ctrl+G to shift plot from and to 40% of y range
        Evaluate:
            - Evaluate that plot is not black
            - Evaluate that plot selected channel value has channel color
            - Evaluate RangeEditor object was called
            - Evaluate that plot selected channel value area doesn't have red and green colors, only original one,
                when cursor not intersect red part of signal
            - Evaluate that after clicking on part that enter in selected ranges,
                plot selected channel value area has only red and green colors
            - Evaluate using Y axi scaling if plot is correctly painted: from 0 to 40% is red, from 40% is not affected
        """
        # Setup
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35]))

        self.widget.showMaximized()
        self.processEvents()
        if self.plot.selected_channel_value_btn.isFlat():
            QTest.mouseClick(self.plot.selected_channel_value_btn, Qt.MouseButton.LeftButton)
        self.mouseClick_WidgetItem(self.channels[0])
        # Evaluate
        self.assertTrue(Pixmap.has_color(self.plot.selected_channel_value.grab(), self.channels[0].color.name()))
        y_range = self.plot.plot.y_axis.range[1] - self.plot.plot.y_axis.range[0]
        offset = 40
        red_range = y_range * offset / 100
        green = QColor.fromRgbF(0.000000, 1.000000, 0.000000, 1.000000)
        red = QColor.fromRgbF(1.000000, 0.000000, 0.000000, 1.000000)

        range_editor_result = [
            {
                "background_color": green,
                "font_color": red,
                "op1": "<=",
                "op2": "<=",
                "value1": 0.0,
                "value2": red_range,
            }
        ]

        # Click on channel
        self.mouseClick_WidgetItem(self.channels[0])
        with mock.patch("asammdf.gui.widgets.tree.RangeEditor") as mo_RangeEditor:
            mo_RangeEditor.return_value.result = range_editor_result
            mo_RangeEditor.return_value.pressed_button = "apply"
            # Press "Alt+R"
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["set_color_range"]))

        # Evaluate
        mo_RangeEditor.assert_called()
        self.assertEqual(self.channels[0].ranges, range_editor_result)

        for _ in range(50):
            self.processEvents(0.01)
        self.avoid_blinking_issue(self.plot.channel_selection)

        # Evaluate that plot has only Green and Red colors
        self.assertTrue(Pixmap.has_color(self.plot.selected_channel_value.grab(), red))
        self.assertTrue(Pixmap.has_color(self.plot.selected_channel_value.grab(), green))
        self.assertFalse(Pixmap.has_color(self.plot.selected_channel_value.grab(), self.channels[0]))

        # Setup
        floor_ = floor(self.plot.plot.height() * offset / 1000)
        find_original_x = Pixmap.search_signal_extremes_by_ax(
            self.plot.plot.grab(QRect(0, floor_ * 2, self.plot.plot.width(), floor_)),
            signal_color=self.channels[0],
            ax="X",
        )[0]
        find_red_x = Pixmap.search_signal_extremes_by_ax(
            self.plot.plot.grab(QRect(0, self.plot.plot.height() - floor_ * 2, self.plot.plot.width(), floor_)),
            signal_color=red,
            ax="X",
        )[0]

        # Click on plot where was founded original color
        QTest.mouseClick(
            self.plot.plot.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifiers(),
            QPoint(find_original_x, int(self.plot.plot.height() / 2)),
        )
        self.processEvents()
        pm = self.plot.selected_channel_value.grab()

        # Evaluate
        self.assertFalse(Pixmap.has_color(pm, red))
        self.assertFalse(Pixmap.has_color(pm, green))
        self.assertTrue(Pixmap.has_color(pm, self.channels[0]))

        # Click on plot where was founded red color
        QTest.mouseClick(
            self.plot.plot.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifiers(),
            QPoint(find_red_x, int(self.plot.plot.height() / 2)),
        )
        self.processEvents()
        pm = self.plot.selected_channel_value.grab()

        # Evaluate
        self.assertTrue(Pixmap.has_color(pm, red))
        self.assertTrue(Pixmap.has_color(pm, green))
        self.assertFalse(Pixmap.has_color(pm, self.channels[0]))

        # self.mouseClick_WidgetItem(self.channels[0])
        # Evaluate plot
        with mock.patch("asammdf.gui.widgets.plot.ScaleDialog") as mo_ScaleDialog:
            mo_ScaleDialog.return_value.offset.value.return_value = 0.0
            mo_ScaleDialog.return_value.scaling.value.return_value = red_range
            # Press Ctrl+G
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["edit_y_axis_scaling"]))
            # Evaluate that ScaleDialog object was created, and it's a method exec() was called
            mo_ScaleDialog.return_value.exec.assert_called()
            self.avoid_blinking_issue(self.plot.channel_selection)

            for _ in range(50):
                self.processEvents()
            self.avoid_blinking_issue(self.plot.channel_selection)

            self.assertTrue(Pixmap.has_color(self.plot.plot.grab(), red))

            mo_ScaleDialog.return_value.offset.value.return_value = -offset
            mo_ScaleDialog.return_value.scaling.value.return_value = y_range
            # Press Ctrl+G
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["edit_y_axis_scaling"]))

            self.assertEqual(mo_ScaleDialog.return_value.exec.call_count, 2)
            self.avoid_blinking_issue(self.plot.channel_selection)

            for _ in range(50):
                self.processEvents()
            self.avoid_blinking_issue(self.plot.channel_selection)

            self.assertTrue(Pixmap.has_color(self.plot.plot.grab(), self.channels[0]))

            # Todo this test must be failed, add evaluation for second color in feature updates...

    def test_Plot_Plot_Shortcut_Keys_Alt_R_Alt_S(self):
        """
        Test Scope:
            Check functionality of key "Alt+I" and "Alt+S". They must convert samples to raw and scaled forms.
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
        for _ in range(50):
            self.processEvents()
        # Evaluate
        # Signal mode = raw
        self.assertEqual(self.channels[0].mode, "raw")
        # Signal line style = Dash line
        self.assertEqual(self.plot.plot.signals[0].pen.style(), Qt.PenStyle.DashLine)
        # Raw value
        self.assertEqual(expected_raw_value, float(self.plot.selected_channel_value.text()))

        # Press "Alt+S"
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["scaled_samples"]))
        # Evaluate
        # Signal mode = raw
        self.assertEqual(self.channels[0].mode, "phys")
        # Signal line style = Dash line
        self.assertEqual(self.plot.plot.signals[0].pen.style(), Qt.PenStyle.SolidLine)

    def test_Plot_Plot_Shortcut_Key_Ctrl_I(self):
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
        # mock for bookmark
        with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QInputDialog.getMultiLineText") as mo_getMultiLineText:
            mo_getMultiLineText.return_value = [self.id(), True]
            # Press "Ctrl+I"
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["insert_bookmark"]))
        # Evaluate
        mo_getMultiLineText.assert_called()
        self.assertTrue(self.plot.show_bookmarks)
        self.assertEqual(self.plot.plot.bookmarks[len(self.plot.plot.bookmarks) - 1].message, self.id())

    def test_Plot_Plot_Shortcut_Key_Alt_I(self):
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

    def test_Plot_Plot_Shortcut_Key_Ctrl_G(self):
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

        # Click on first channel
        self.mouseClick_WidgetItem(self.channels[0])
        with mock.patch("asammdf.gui.widgets.plot.ScaleDialog") as mo_ScaleDialog:
            mo_ScaleDialog.return_value.offset.value.return_value = offset
            mo_ScaleDialog.return_value.scaling.value.return_value = scale
            # Press Ctrl+Shift+C
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["edit_y_axis_scaling"]))
        # Evaluate that ScaleDialog object was created, and it's a method exec() was called
        mo_ScaleDialog.return_value.exec.assert_called()
        for _ in range(50):
            self.processEvents()
        self.avoid_blinking_issue(self.plot.channel_selection)

        # Top
        pixmap = self.plot.plot.grab(QRect(0, 0, self.plot.plot.width(), int(self.plot.plot.height() / 2)))
        # Evaluate plot
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        # Bottom
        pixmap = self.plot.plot.grab(
            QRect(0, int(self.plot.plot.height() / 2 + 1), self.plot.plot.width(), int(self.plot.plot.height() / 2))
        )
        self.assertFalse(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        # Evaluate y_range tuple
        self.assertTupleEqual(self.plot.plot.signals[0].y_range, expected_y_range)

    def test_Plot_Plot_Shortcut_Key_C(self):
        """
        Test Scope:
            - Ensure that channel color is changed.
        Events:
            - Open Plot with 2 channels
            - Press C
            - Select 1 Channel
            - Press C
            - Select 2 Channels
            - Press C
        Evaluate:
            - Evaluate that color dialog is not open if channel is not selected.
            - Evaluate that channel color is changed only for selected channel
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

    def test_Plot_Plot_Shortcut_Keys_Ctrl_C__Ctrl_V(self):
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

    def test_Plot_Plot_Shortcut_Keys_Ctrl_Shift_C__Ctrl_Shift_V(self):
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

    def test_Plot_Plot_Shortcut_Keys_Ctrl_Left_and_Right_Buckets(self):
        """
        Test scope:
            Ensure that  Ctrl+[ and Ctrl+] will change font size

        Events:
            - Press Ctrl+[
            - Press Ctrl+]

        Evaluate:
         - Evaluate that font size was decreased after shortcut Ctrl+[ was pressed
         - Evaluate that font size was increased after shortcut Ctrl+] was pressed
        """
        font_size = self.plot.font().pointSize()
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["decrease_font"]))
        self.assertLess(font_size, self.plot.font().pointSize())

        font_size = self.plot.font().pointSize()
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["increase_font"]))
        self.assertGreater(font_size, self.plot.font().pointSize())

    def test_Plot_Plot_Shortcut_Keys_Backspace__Shift_Backspace__Shift_W(self):
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
        # add channel to plot
        self.assertIsNotNone(self.add_channels([35]))
        # Toggle full screen
        self.widget.showMaximized()
        self.processEvents()
        x = self.plot.plot.width() / 2
        y = self.plot.plot.height() / 2
        # Y range at start
        y_range = self.plot.plot.signals[0].y_range

        # Events
        # Click on center of the plot
        QTest.mouseClick(
            self.plot.plot.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifiers(),
            QPoint(int(x), int(y)),
        )
        self.processEvents(0.01)

        # Rotate mouse wheel
        self.wheel_action(self.plot.plot.viewport(), x, y, -2)
        self.processEvents(0.01)

        # Click on plot
        QTest.mouseClick(
            self.plot.plot.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifiers(),
            QPoint(int(x * 0.2), int(y * 0.8)),
        )
        self.processEvents(0.01)

        # Rotate mouse wheel
        self.wheel_action(self.plot.plot.viewport(), x, y, 1)
        self.processEvents(0.01)

        new_y_range = self.plot.plot.signals[0].y_range
        # Evaluate
        self.assertNotEqual(new_y_range, y_range)

        # click on Backspace 2 times, idk why, but first hit didn't perform necessary action
        QTest.keySequence(self.plot, QKeySequence(self.shortcuts["undo_zoom"]))
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

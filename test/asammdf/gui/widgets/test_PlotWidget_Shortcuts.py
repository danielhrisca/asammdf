#!/usr/bin/env python
from math import floor
import os
from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget
from unittest import mock

from PySide6 import QtCore, QtGui, QtTest, QtWidgets

import asammdf.gui.widgets.plot


class TestPlotShortcuts(TestPlotWidget):
    def setUp(self):
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()

    def test_Plot_Plot_Shortcuts(self):
        """
        Test Scope:
            Check if all shortcut events is called
        Events:
            - Press key M
        Evaluate:
            - Evaluate that ...
        """
        with self.subTest("test_Key_M"):
            expected_mo_call_count = 2
            # At start Statistics is hidden
            self.assertFalse(self.plot.info.isVisible())
            with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QSplitter.setSizes") as mo_setSizes:
                # press key "M"
                QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_M)
                self.assertTrue(self.plot.info.isVisible())
                # press key "M"
                QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_M)
                self.assertFalse(self.plot.info.isVisible())
            self.assertEqual(mo_setSizes.call_count, expected_mo_call_count)

        with self.subTest("test_Key_2"):
            # Setup
            previous_focused_mode_btn = self.plot.focused_mode_btn.isFlat()
            # Press 2
            QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_2)
            self.assertNotEqual(previous_focused_mode_btn, self.plot.focused_mode_btn.isFlat())

            # Update
            previous_focused_mode_btn = self.plot.focused_mode_btn.isFlat()
            # Press 2
            QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_2)
            self.assertNotEqual(previous_focused_mode_btn, self.plot.focused_mode_btn.isFlat())

        with mock.patch("asammdf.gui.widgets.plot.Plot.item_by_uuid"):
            with mock.patch.object(asammdf.gui.widgets.plot.PlotGraphics, "get_axis") as mo_get_axis:
                mo_get_axis.assert_not_called()
                expected_mo_call_count = 0
                with self.subTest("test_Ctrl_B"):
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+B"))
                    expected_mo_call_count += 1
                    self.assertEqual(mo_get_axis.call_count, expected_mo_call_count)
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "bin")
                #    self.assertEqual(mo_get_axis.return_value.format, "bin")
                with self.subTest("test_Ctrl_H"):
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+H"))
                    expected_mo_call_count += 1
                    self.assertEqual(mo_get_axis.call_count, expected_mo_call_count)
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "hex")
                #    self.assertEqual(mo_get_axis.return_value.format, "hex")
                with self.subTest("test_Ctrl_P"):
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+P"))
                    expected_mo_call_count += 1
                    self.assertEqual(mo_get_axis.call_count, expected_mo_call_count)
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "phys")
                #   self.assertEqual(mo_get_axis.return_value.format, "phys")
                with self.subTest("test_Ctrl_T"):
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+T"))
                    expected_mo_call_count += 1
                    self.assertEqual(mo_get_axis.call_count, expected_mo_call_count)
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "ascii")
                #   self.assertEqual(mo_get_axis.return_value.format, "ascii")

        with self.subTest("test_Key_R"):
            with mock.patch("asammdf.gui.widgets.tree.ChannelsTreeItem.set_prefix") as mo_set_prefix:
                with mock.patch("asammdf.gui.widgets.tree.ChannelsTreeItem.set_value") as mo_set_value:
                    QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_R)
            mo_set_prefix.assert_called()
            mo_set_value.assert_called()

        with self.subTest("test_Key_Ctrl_R"):
            with mock.patch("asammdf.gui.widgets.tree.ChannelsTreeWidget.keyPressEvent") as mo_keyPressEvent:
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+R"))
            mo_keyPressEvent.assert_called()

        with mock.patch.object(self.plot.plot, "viewbox"):
            with mock.patch.object(self.plot.plot, "update"):
                with mock.patch.object(self.plot.plot, "get_axis") as mo_get_axis:
                    # Evaluate
                    with self.subTest("test_key_Alt_R"):
                        # Press "Alt+R"
                        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Alt+R"))
                        # Signal mode = raw
                        self.assertEqual(mo_get_axis.return_value.mode, "raw")
                        self.assertIsNone(mo_get_axis.return_value.picture)
                        mo_get_axis.return_value.update.assert_called()

                    with self.subTest("test_key_Alt_S"):
                        # Press "Alt+S"
                        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Alt+S"))
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
                            QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Ctrl+I"))

            mo_bookmarks.append.assert_called_with(mo_Bookmark.return_value)
            mo_viewbox.addItem.assert_called_with(mo_bookmarks.__getitem__())
            self.assertEqual(mo_Bookmark.call_args[1]["message"], self.id())
            mo_getMultiLineText.assert_called()

        with self.subTest("test_Key_Alt_I"):
            bookmark_btn_previous_state = self.plot.bookmark_btn.isFlat()
            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Alt+I"))  # ToDo is unittest???
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

                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+G"))

            mo_ScaleDialog.return_value.exec.assert_called()
            self.assertTupleEqual(self.plot.plot.signals[0].y_range, expected_y_range)

        with mock.patch("asammdf.gui.widgets.plot.ChannelsTreeWidget.keyPressEvent") as mo_keyPressEvent:
            with self.subTest("test_Key_C"):
                # Event
                QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_C)
                # Evaluate
                mo_keyPressEvent.assert_called()

            with self.subTest("test_Ctrl_C"):
                # Event
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+C"))
                # Evaluate
                mo_keyPressEvent.assert_called()

            with self.subTest("test_Ctrl_Shift_C"):
                # Event
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+Shift+C"))
                # Evaluate
                mo_keyPressEvent.assert_called()

        # Setup
        # Adds one channel to plot
        with mock.patch("asammdf.gui.widgets.plot.ChannelsTreeWidget.keyPressEvent") as mo_keyPressEvent:
            with self.subTest("test_Ctrl_V"):
                # Event
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+V"))
                # Evaluate
                mo_keyPressEvent.assert_called()

            with self.subTest("test_Ctrl_Shift_V"):
                # Event
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+Shift+V"))
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
                        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+["))
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
                        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+]"))
            # Evaluate
            self.assertEqual(initial_size, self.plot.font().pointSize() - 1)

        with self.subTest("Shortcut_Key_Backspace"):
            # Setup
            # Adds one channel to plot
            self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
            self.processEvents()
            with mock.patch.object(self.plot, "undo_zoom") as mo_undo_zoom:
                # Event
                QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_Backspace)
            # Evaluate
            mo_undo_zoom.assert_called()

        with self.subTest("Shortcut_Key_Shift_Backspace"):
            # Setup
            # Adds one channel to plot
            self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
            self.processEvents()
            with mock.patch.object(self.plot, "redo_zoom") as mo_redo_zoom:
                # Event
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence(QtGui.Qt.SHIFT + QtGui.Qt.Key_Backspace))
            # Evaluate
            mo_redo_zoom.assert_called()

        with self.subTest("Shortcut_Key_Shift_W"):
            # Adds one channel to plot
            self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
            self.processEvents()
            with mock.patch.object(self.plot.plot, "set_y_range") as mo_set_y_range:
                with mock.patch.object(self.plot.plot, "viewbox") as mo_viewbox:
                    # Event
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence(QtGui.Qt.SHIFT + QtGui.Qt.Key_W))
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
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        # hide bookmarks if it's available
        if self.plot.show_bookmarks:
            self.plot.toggle_bookmarks(hide=True)
        self.processEvents()
        # pixmap is not black
        # self.assertFalse(Pixmap.is_black(self.plot.plot.viewport().grab()))

    def tearDown(self):
        with mock.patch("asammdf.gui.widgets.mdi_area.MessageBox.question") as mo_question:
            mo_question.return_value = QtWidgets.QMessageBox.No
        super().tearDown()
        self.widget.destroy()

    def test_Plot_Plot_Shortcut_Key_M(self):
        """
        Test Scope:
            To check if is displayed info about selected channel and after deleting all channels buffer is clear
                by pressing key "M"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a plot
            - Press Key "M"
            - Select third channel
            - Select first channel
            - Pres key "Down"
        Evaluate:
            - Evaluate that displayed info is related to third channel
            - Evaluate that displayed info is related to first channel
            - Evaluate that displayed info is related to second channel
            - Evaluate that buffer is clear
        """
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35, 36, 37]))
        # #Evaluate if info isn't visible before pressing key M
        self.assertFalse(self.plot.info.isVisible())
        # Event
        QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_M)
        # #Evaluate if info is visible after pressing key M
        self.assertTrue(self.plot.info.isVisible())

        # click on a first channel
        with self.subTest("test_key_M_click_on_first_channel"):
            self.mouseClick_WidgetItem(self.channels[0])
            self.assertEqual(self.plot.info._name, self.channels[0].name)
            self.assertEqual(self.plot.info.color, self.channels[0].color.name())

        with self.subTest("test_key_M_press_key_Down_on_channel_selection"):
            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key_Down)
            self.assertEqual(self.plot.info._name, self.channels[1].name)
            self.assertEqual(self.plot.info.color, self.channels[1].color.name())

        # delete all channels
        with self.subTest("test_key_M_delete_all_channels"):
            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+A"))
            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key_Delete)

            # Not save value of the last selected channel
            self.assertNotEqual(self.plot.info._name, self.channels[2].name)

    def test_Plot_Plot_Shortcut_Key_2(self):
        """
        Test Scope:
            To check if is displayed only selected channel after pressing key "2"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a plot
            - Press Key "2"
            - Select first channel
            - Pres key "Down"
            - Select third channel
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black and contain colors of all 3 channels
            - Evaluate that plot is black after pressing key "2"
            - Evaluate that plot contains only color of first channel after clicked on first channel
            - Evaluate that plot contains only color of second channel after pressing key "Down"
            - Evaluate that plot contains only color of third channel after clicked on third channel
            - Evaluate that plot contains colors of all 3 channels after pressing key "2"
        """
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35, 36, 37]))
        pixmap = self.plot.plot.viewport().grab()
        # Evaluate
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[1].color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[2].color.name()))

        # case 0
        QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_2)
        for _ in range(1000):
            self.processEvents(0.01)
        # Evaluate
        pm = self.plot.plot.viewport().grab()
        colors = Pixmap.color_names(pm)

        # debug_ = Pixmap.color_map(pm)
        # for line in debug_.items():
        #     print(line)
        #     print()
        self.assertTrue(Pixmap.is_black(pm))

        # case 1
        self.mouseClick_WidgetItem(self.channels[0])
        for _ in range(50):
            self.avoid_blinking_issue(self.plot.channel_selection)
        pixmap = self.plot.plot.viewport().grab()
        # Evaluate
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        self.assertFalse(Pixmap.has_color(pixmap, self.channels[1].color.name()))
        self.assertFalse(Pixmap.has_color(pixmap, self.channels[2].color.name()))

        # case 2
        QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key_Down)
        for _ in range(50):
            self.avoid_blinking_issue(self.plot.channel_selection)
        pixmap = self.plot.plot.viewport().grab()
        # Evaluate
        self.assertFalse(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[1].color.name()))
        self.assertFalse(Pixmap.has_color(pixmap, self.channels[2].color.name()))

        # case 3
        self.mouseClick_WidgetItem(self.channels[2])
        for _ in range(50):
            self.avoid_blinking_issue(self.plot.channel_selection)
        pixmap = self.plot.plot.viewport().grab()
        # Evaluate
        self.assertFalse(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        self.assertFalse(Pixmap.has_color(pixmap, self.channels[1].color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[2].color.name()))

        # case 4
        QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_2)
        for _ in range(50):
            self.avoid_blinking_issue(self.plot.channel_selection)
        pixmap = self.plot.plot.viewport().grab()
        # Evaluate
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[1].color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[2].color.name()))

    def test_Plot_Plot_Shortcut_Keys_Ctrl_H_Ctrl_B_Ctrl_P_Ctrl_T(self):
        """
        Test Scope:
            Check if values is converted to int, hex, bin after pressing combination of key "Ctrl+<H>|<B>|<P>"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Press "Ctrl+B"
            - Press "Ctrl+H"
            - Press "Ctrl+P"
            - Press "Ctrl+T"

        Evaluate:
            - Evaluate that plot is not black
            - Evaluate that unit is changed to hex after pressing key "Ctrl+H"
            - Evaluate that unit is changed to bin after pressing key "Ctrl+B"
            - Evaluate that unit is changed to Int after pressing key "Ctrl+P"
        """
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35]))
        self.mouseClick_WidgetItem(self.channels[0])

        physical = self.plot.selected_channel_value.text()
        physical_hours = int(physical.split(" ")[0])
        # Press "Ctrl+B"
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+B"))
        bin_ = self.plot.selected_channel_value.text()
        bin_hours = int(bin_.split(" ")[0].replace(".", ""), 2)

        # Evaluate
        self.assertNotEqual(physical, bin_)
        self.assertEqual(physical_hours, bin_hours)

        # Press "Ctrl+H"
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+H"))
        hex_ = self.plot.selected_channel_value.text()
        # Convert hex value to Int
        hex_hours = int(hex_.split(" ")[0], 16)

        # Evaluate
        self.assertNotEqual(physical, hex_)
        self.assertIn("0x", hex_)
        self.assertEqual(physical_hours, hex_hours)

        # Press "Ctrl+P"
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+P"))
        new_physical = self.plot.selected_channel_value.text()
        new_physical_hours = int(new_physical.split(" ")[0])

        # Evaluate
        self.assertEqual(physical, new_physical)
        self.assertEqual(physical_hours, new_physical_hours)

        # Press "Ctrl+P"
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+T"))
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
            QtTest.QTest.mouseClick(self.plot.selected_channel_value_btn, QtCore.Qt.MouseButton.LeftButton)
        self.mouseClick_WidgetItem(self.channels[0])
        # Evaluate
        self.assertTrue(Pixmap.has_color(self.plot.selected_channel_value.grab(), self.channels[0].color.name()))
        y_range = self.plot.plot.y_axis.range[1] - self.plot.plot.y_axis.range[0]
        offset = 40
        red_range = y_range * offset / 100
        green = QtGui.QColor.fromRgbF(0.000000, 1.000000, 0.000000, 1.000000)
        red = QtGui.QColor.fromRgbF(1.000000, 0.000000, 0.000000, 1.000000)

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
            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+R"))

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
            self.plot.plot.grab(QtCore.QRect(0, floor_ * 2, self.plot.plot.width(), floor_)),
            signal_color=self.channels[0],
            ax="X",
        )[0]
        find_red_x = Pixmap.search_signal_extremes_by_ax(
            self.plot.plot.grab(QtCore.QRect(0, self.plot.plot.height() - floor_ * 2, self.plot.plot.width(), floor_)),
            signal_color=red,
            ax="X",
        )[0]

        # Click on plot where was founded original color
        QtTest.QTest.mouseClick(
            self.plot.plot.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            QtCore.QPoint(find_original_x, int(self.plot.plot.height() / 2)),
        )
        self.processEvents()
        pm = self.plot.selected_channel_value.grab()

        # Evaluate
        self.assertFalse(Pixmap.has_color(pm, red))
        self.assertFalse(Pixmap.has_color(pm, green))
        self.assertTrue(Pixmap.has_color(pm, self.channels[0]))

        # Click on plot where was founded red color
        QtTest.QTest.mouseClick(
            self.plot.plot.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            QtCore.QPoint(find_red_x, int(self.plot.plot.height() / 2)),
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
            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+G"))
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
            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+G"))

            self.assertEqual(mo_ScaleDialog.return_value.exec.call_count, 2)
            self.avoid_blinking_issue(self.plot.channel_selection)

            for _ in range(50):
                self.processEvents()
            self.avoid_blinking_issue(self.plot.channel_selection)

            self.assertTrue(Pixmap.has_color(self.plot.plot.grab(), self.channels[0]))

            # Todo this test must be failed, add evaluation for second color...
        # trans = []
        # buf = 0
        # op = (extremes[1] - extremes[0]) / len(self.channels[0].signal.samples) + extremes[0]
        # for x, sample in enumerate(self.channels[0].signal.samples):
        #     if red_range - 1 <= sample <= red_range:
        #         if x > buf + 1:
        #             trans.append(ceil(x * op))
        #         buf = x

    def test_Plot_Plot_Shortcut_Keys_Alt_R_Alt_S(self):
        """
        Test Scope:
            Check functionality of key "Alt+I" and "Alt+S"
        Events:
            - Open 'FileWidget' with valid measurement.
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
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Alt+R"))
        for _ in range(50):
            self.processEvents()
        # Evaluate
        # Signal mode = raw
        self.assertEqual(self.channels[0].mode, "raw")
        # Signal line style = Dash line
        self.assertEqual(self.plot.plot.signals[0].pen.style(), QtCore.Qt.PenStyle.DashLine)
        # Raw value
        self.assertEqual(expected_raw_value, float(self.plot.selected_channel_value.text()))

        # Press "Alt+S"
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Alt+S"))
        # Evaluate
        # Signal mode = raw
        self.assertEqual(self.channels[0].mode, "phys")
        # Signal line style = Dash line
        self.assertEqual(self.plot.plot.signals[0].pen.style(), QtCore.Qt.PenStyle.SolidLine)

    def test_Plot_Plot_Shortcut_Key_Ctrl_S(self):
        """
        Test Scope:
            Check if by pressing "Ctrl+S" is saved in the new measurement file only active channels
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a plot
            _ Deselect last channel
            - Mock getSaveFileName() object and set return value of this object a file path of the new measurement
            file
            - Press Key "Ctrl+S"
            - Open recently created measurement file in a new window
        Evaluate:
            - Evaluate that object getSaveFileName() was called after pressing combination "Ctrl+S"
            - Evaluate that in measurement file is saved only active channels
        """
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]))
        expected_items = [channel.name for channel in self.channels]
        expected_items.append("time")
        file_path = os.path.join(self.test_workspace, "file.mf4")
        self.create_window(window_type="Plot", channels_indexes=(20, 21, 22))
        self.processEvents()
        # mock for getSaveFileName object
        with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = (file_path, "")
            self.mouseClick_WidgetItem(self.channels[0])
            # Press Ctrl+S
            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+S"))
        # Evaluate
        mo_getSaveFileName.assert_called()

        # Open recently saved measurement file
        self.setUpFileWidget(measurement_file=file_path, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Evaluate
        for index in range(self.widget.channels_tree.topLevelItemCount()):
            self.assertIn(self.widget.channels_tree.topLevelItem(index).name, expected_items)
            if self.widget.channels_tree.topLevelItem(index).name != "time":
                expected_items.remove(self.widget.channels_tree.topLevelItem(index).name)
        self.assertListEqual(expected_items, ["time"])

    def test_Plot_Plot_Shortcut_Key_Ctrl_I(self):
        """
        Test Scope:
            Check if bookmark is created after pressing key "Ctrl+I"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Mock QInputDialog object
            - Press "Ctrl+I"
        Evaluate:
            - Evaluate that bookmarks are not displayed before pressing "Ctrl+I"
            - Evaluate that bookmarks are displayed after pressing "Ctrl+I"
                and the message of the last bookmark is the first element of the returned list of mock object
        """
        # mock for bookmark
        with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QInputDialog.getMultiLineText") as mo_getMultiLineText:
            mo_getMultiLineText.return_value = [self.id(), True]
            # Press "Ctrl+I"
            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+I"))
        # Evaluate
        mo_getMultiLineText.assert_called()
        self.assertTrue(self.plot.show_bookmarks)
        self.assertEqual(self.plot.plot.bookmarks[len(self.plot.plot.bookmarks) - 1].message, self.id())

    def test_Plot_Plot_Shortcut_Key_Alt_I(self):
        """
        Test Scope:
            Check functionality of key "Ctrl+I"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Mock QInputDialog object
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
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Alt+I"))
        # Evaluate
        self.assertTrue(self.plot.show_bookmarks)
        # Press "Al+I"
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Alt+I"))
        # Evaluate
        self.assertFalse(self.plot.show_bookmarks)

    def test_Plot_Plot_Shortcut_Key_Ctrl_G(self):
        """
        Test Scope:
            Check if signal is changed his Y limits by setting it with ScaleDialog object called with "Ctrl+G"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Create a plot widget with signals
            - Click on first channel
            - Mock ScaleDialog object
            - Set up return_value for scaling and offset objects
            - Press Ctrl+G
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
        pixmap = self.plot.plot.viewport().grab(
            QtCore.QRect(0, 0, self.plot.plot.width(), int(self.plot.plot.height() / 2))
        )
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        # Bottom
        pixmap = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.width(), int(self.plot.plot.height() / 2))
        )
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[0].color.name()))

        # Click on first channel
        self.mouseClick_WidgetItem(self.channels[0])
        with mock.patch("asammdf.gui.widgets.plot.ScaleDialog") as mo_ScaleDialog:
            mo_ScaleDialog.return_value.offset.value.return_value = offset
            mo_ScaleDialog.return_value.scaling.value.return_value = scale
            # Press Ctrl+Shift+C
            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+G"))
        # Evaluate that ScaleDialog object was created, and it's a method exec() was called
        mo_ScaleDialog.return_value.exec.assert_called()
        for _ in range(50):
            self.processEvents()
        self.avoid_blinking_issue(self.plot.channel_selection)

        # Top
        pixmap = self.plot.plot.viewport().grab(
            QtCore.QRect(0, 0, self.plot.plot.width(), int(self.plot.plot.height() / 2))
        )
        # Evaluate plot
        self.assertTrue(Pixmap.has_color(pixmap, self.channels[0].color.name()))
        # Bottom
        pixmap = self.plot.plot.viewport().grab(
            QtCore.QRect(
                0, int(self.plot.plot.height() / 2 + 1), self.plot.plot.width(), int(self.plot.plot.height() / 2)
            )
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
            - Mock getColor() object
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
                QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_C)
                mo_getColor.assert_not_called()

        with self.subTest("test_1SelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                # Setup
                self.mouseClick_WidgetItem(self.channels[1])
                previous_color = self.channels[1].color.name()
                color = QtGui.QColor("magenta")
                mo_getColor.return_value = color
                # Event
                QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_C)
                # Evaluate
                mo_getColor.assert_called()
                self.assertNotEqual(previous_color, self.channels[1].color.name())
                self.assertEqual(color.name(), self.channels[1].color.name())

        with self.subTest("test_AllSelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                # Setup
                color = QtGui.QColor("black")
                mo_getColor.return_value = color
                # Set selected both channels
                QtTest.QTest.keySequence(self.plot.channel_selection, QtGui.QKeySequence("Ctrl+A"))
                # store previous colors of channels
                previous_ch_35_color = self.channels[0].color.name()
                previous_ch_36_color = self.channels[1].color.name()
                previous_ch_37_color = self.channels[2].color.name()
                color = QtGui.QColor("black")
                mo_getColor.return_value = color
                # Event
                QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_C)
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
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+C"))
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+V"))
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
        ch.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        ch = self.find_channel(self.widget.channels_tree, self.channels[1].name)
        ch.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        # Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        # Evaluate that second window is created
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
        # Second window
        self.plot = self.widget.mdi_area.subWindowList()[1].widget()
        # Press Ctrl+V
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+V"))
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
            - Open 'FileWidget' with valid measurement.
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
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+Shift+C"))
        self.mouseClick_WidgetItem(self.channels[2])
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+Shift+V"))

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
        tests for Ctrl+[ and Ctrl+]
        """
        font_size = self.plot.font().pointSize()
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+]"))
        self.assertLess(font_size, self.plot.font().pointSize())

        font_size = self.plot.font().pointSize()
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+["))
        self.assertGreater(font_size, self.plot.font().pointSize())

    def test_Plot_Plot_Shortcut_Keys_Backspace__Shift_Backspace__Shift_W(self):
        """
        Test Scope:
            ...
        """
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35]))

        self.widget.showMaximized()

        self.processEvents()

        # Select line
        y_midd_line = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        color_map = Pixmap.color_map(y_midd_line)
        distance_in_pixels_0 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(color_map[0]):
            if x == self.channels[0].color.name():
                distance_in_pixels_0 = i - distance_in_pixels_0
                if distance_in_pixels_0 != i:
                    break
        self.assertGreater(distance_in_pixels_0, 0)

        # Press "O"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_O)

        self.processEvents()

        # Select line
        y_midd_line = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        color_map = Pixmap.color_map(y_midd_line)
        distance_in_pixels_1 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(color_map[0]):
            if x == self.channels[0].color.name():
                distance_in_pixels_1 = i - distance_in_pixels_1
                if distance_in_pixels_1 != i:
                    break
        self.assertGreater(distance_in_pixels_1, 0)

        # Press "O"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_O)
        self.processEvents()
        # Select line
        y_midd_line = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        color_map = Pixmap.color_map(y_midd_line)
        distance_in_pixels_2 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(color_map[0]):
            if x == self.channels[0].color.name():
                distance_in_pixels_2 = i - distance_in_pixels_2
                if distance_in_pixels_2 != i:
                    break
        self.assertGreater(distance_in_pixels_1, distance_in_pixels_2)

        # click on Backspace
        QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_Backspace)

        self.processEvents()
        # Select line
        y_midd_line = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        color_map = Pixmap.color_map(y_midd_line)
        distance_in_pixels_3 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(color_map[0]):
            if x == self.channels[0].color.name():
                distance_in_pixels_3 = i - distance_in_pixels_3
                if distance_in_pixels_3 != i:
                    break

        self.assertEqual(distance_in_pixels_1, distance_in_pixels_3)

        # click on Shift + Backspace
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence(QtGui.Qt.SHIFT + QtGui.Qt.Key_Backspace))

        self.processEvents()
        # Select line
        y_midd_line = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        color_map = Pixmap.color_map(y_midd_line)
        distance_in_pixels_3 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(color_map[0]):
            if x == self.channels[0].color.name():
                distance_in_pixels_3 = i - distance_in_pixels_3
                if distance_in_pixels_3 != i:
                    break

        self.assertEqual(distance_in_pixels_2, distance_in_pixels_3)

        # Press "O"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_O)
        # click on Shift + W
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence(QtGui.Qt.SHIFT + QtGui.Qt.Key_W))
        self.processEvents()

        # Select line
        y_midd_line = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        color_map = Pixmap.color_map(y_midd_line)
        distance_in_pixels_3 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(color_map[0]):
            if x == self.channels[0].color.name():
                distance_in_pixels_3 = i - distance_in_pixels_3
                if distance_in_pixels_3 != i:
                    break
        self.assertEqual(distance_in_pixels_0, distance_in_pixels_3)

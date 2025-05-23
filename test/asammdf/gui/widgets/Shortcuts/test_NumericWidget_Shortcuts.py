#!/usr/bin/env python\
import json
import os
import pathlib
from unittest import mock

from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtTest import QTest

from asammdf import mdf
from asammdf.gui.utils import copy_ranges
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget


class TestTableViewShortcuts(TestFileWidget):
    def setUp(self):
        """
        Events:
            - Open measurement file.
            - Create a Numeric window.

        Evaluate:
            - Evaluate that there is one active sub-window created.
        Returns
        -------

        """
        super().setUp()
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        self.create_window(window_type="Numeric")

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.numeric = self.widget.mdi_area.subWindowList()[0].widget()
        self.table_view = self.numeric.channels.dataView

        # get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.table_view))

    def test_delete_shortcut(self):
        """
        Test Scope:
            - Ensure that key Delete will remove selected channels

        Events:
            - Add some channels to Numeric widget.
            - Select first item and press key Delete.
            - Select all items and press key delete.

        Evaluate:
            - Evaluate that there are all items added to a numeric windget
            - Evaluate that first row is removed from a list after it became selected and key Delete is pressed
            - Evaluate that all rows after they became selected are removed by pressing key delete
        """
        # Setup
        self.processEvents()
        # Add one channel to the widget
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]), self.numeric)
        self.processEvents()
        channel_count = len(self.channels)

        # Events
        # Select first row
        self.table_view.selectRow(0)
        self.processEvents()
        # Press key Delete
        QTest.keySequence(self.table_view, QKeySequence(self.shortcuts["delete_items"]))

        # Evaluate
        self.assertEqual(len(self.table_view.backend.signals), channel_count - 1)

        # Events
        # select all items
        QTest.keySequence(self.table_view, QKeySequence("Ctrl+A"))
        # Press key Delete
        QTest.keySequence(self.table_view, QKeySequence(self.shortcuts["delete_items"]))

        # Evaluate
        self.assertEqual(len(self.table_view.backend.signals), 0)

    def test_set_color_range_shortcut(self):
        """
        Test Scope:
            Check if color range is triggered after pressing key Ctrl+R.
        Events:
            - Display a few signals on numeric widget.
            - Select first signal.
            - Press "Ctrl+R" -> ranges from 0 to 40% of y value and colors green and red -> apply.

        Evaluate:
            - Evaluate RangeEditor object was called
            - Evaluate that selected item has color range variable, and its value is equal to the expected value
        """
        # Setup
        value1 = 0.0
        value2 = 100
        # Add items to the widget
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]), self.numeric)
        self.processEvents()
        green = QColor.fromRgbF(0.000000, 1.000000, 0.000000, 1.000000)
        red = QColor.fromRgbF(1.000000, 0.000000, 0.000000, 1.000000)

        expected_value = [
            {
                "background_color": green,
                "font_color": red,
                "op1": "<=",
                "op2": "<=",
                "value1": value1,
                "value2": value2,
            }
        ]
        # Events
        # Select first row
        self.table_view.selectRow(0)
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.numeric.RangeEditor") as mo_RangeEditor:
            mo_RangeEditor.return_value.result = expected_value
            mo_RangeEditor.return_value.pressed_button = "apply"
            # Press "Alt+R"
            QTest.keySequence(self.table_view, QKeySequence(self.shortcuts["set_color_ranges"]))

        # Evaluate
        mo_RangeEditor.assert_called()
        row_0 = True
        for value in self.table_view.ranges.values():
            if row_0 is True:  # Evaluate range for first row
                self.assertDictEqual(value[0], expected_value[0])
                row_0 = False
            else:
                self.assertListEqual(value, [])

    def test_copy_display_properties_shortcut(self):
        """
        Test Scope:
            - Ensure that key "Ctrl+Shift+C" can copy display properties of selected item

        Events:
            - Open Numeric widget with one channel
            - Select this channel
            - Press Ctrl+Shift+C

        Evaluate:
            - Evaluate that there is added one item to widget
            - Evaluate that in clipboard is display properties of selected item
        """
        # Setup
        # Add item to the widget
        self.assertIsNotNone(self.add_channels(["ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL"]))
        channel = self.table_view.backend.signals[0]
        expected_ch_info = {
            "format": channel.format,
            "ranges": copy_ranges(self.table_view.ranges[channel.entry]),
            "type": "channel",
            "color": channel.color.name(),
            "precision": self.table_view.model().float_precision,
            "ylink": channel.signal.y_link,
            "individual_axis": channel.signal.individual_axis,
            "y_range": channel.y_range,
            "origin_uuid": channel.origin_uuid,
        }

        # expected_ch_info = json.dumps(expected_ch_info)
        # Events
        with mock.patch("asammdf.gui.widgets.numeric.QtWidgets.QApplication.instance") as mo_instance:
            # Click on a first channel
            self.table_view.selectRow(0)
            # Press Ctrl+Shift+C
            QTest.keySequence(self.table_view, QKeySequence(self.shortcuts["copy_display_properties"]))

        # Evaluate
        mo_instance.return_value.clipboard.return_value.setText.assert_called_with(json.dumps(expected_ch_info))

    def test_paste_display_properties_shortcut(self):
        """
        Test Scope:
            - Ensure that key "Ctrl+Shift+V" can paste from clipboard display properties to selected item.

        Events:
            - Open Numeric widget with two channels.
            - Copy display properties of the first item.
            - Select second item -> press Ctrl+Shift+C.

        Evaluate:
            - Evaluate that to the widget are added two items.
            - Evaluate that after selecting second channel and hit keys "Ctrl+Shift+V",
                the second channel has the same display properties with the first channel.
        """
        # Setup
        # Add items to the widget
        self.assertIsNotNone(
            self.add_channels(
                ["ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL", "ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL"]
            )
        )
        # Evaluate precondition
        signal_0 = self.table_view.backend.signals[0]
        expected_ch_info = {
            "format": signal_0.format,
            "ranges": copy_ranges(self.table_view.ranges[signal_0.entry]),
        }
        with mock.patch("asammdf.gui.widgets.numeric.QtWidgets.QApplication.instance") as mo_instance:
            mo_instance.return_value.clipboard.return_value.text.return_value = expected_ch_info

            # Click on a first channel
            self.table_view.selectRow(1)
            # Press Ctrl+Shift+C
            QTest.keySequence(self.table_view, QKeySequence(self.shortcuts["paste_display_properties"]))
        # Evaluate
        mo_instance.return_value.clipboard.return_value.text.assert_called()
        signal_1 = self.table_view.backend.signals[1]
        self.assertEqual(signal_1.format, signal_0.format)
        self.assertEqual(self.table_view.ranges[signal_1.entry], self.table_view.ranges[signal_0.entry])


class TestNumericShortcuts(TestFileWidget):
    def setUp(self):
        """
        Events:
            - Open measurement file.
            - Create a Numeric window.

        Evaluate:
            - Evaluate that there is one active sub-window created.
        Returns
        -------

        """
        super().setUp()
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        self.create_window(window_type="Numeric")

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.numeric = self.widget.mdi_area.subWindowList()[0].widget()
        self.table_view = self.numeric.channels.dataView

        # get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.numeric))

    def test_ascii__bin__hex__physical_shortcuts(self):
        """
        Test Scope:
            Check if values is converted to int, hex, bin after pressing combination of key "Ctrl+<H>|<B>|<P>"

        Events:
            - Display 1 signal on Numeric
            - Press "Ctrl+B"
            - Press "Ctrl+H"
            - Press "Ctrl+P"
            - Press "Ctrl+T"

        Evaluate:
            - Evaluate that signal format is changed to BIN after pressing key "Ctrl+B"
            - Evaluate that signal format is changed to HEX after pressing key "Ctrl+H"
            - Evaluate that signal format is changed to PHYS after pressing key "Ctrl+P"
            - Evaluate that signal format is changed to ASCII after pressing key "Ctrl+T"
        """
        # add channels to numeric
        self.assertIsNotNone(self.add_channels([35]))

        # Setup
        bin_format = "bin"
        hex_format = "hex"
        phys_format = "phys"
        ascii_format = "ascii"

        # Select first row (0)
        self.table_view.selectRow(0)
        self.processEvents()
        # Press "Ctrl+B"
        QTest.keySequence(self.numeric, QKeySequence(self.shortcuts["bin"]))
        self.processEvents()
        # Evaluate
        self.assertEqual(self.table_view.backend.signals[0].format, bin_format)
        self.assertEqual(self.table_view.model().format, bin_format)

        # Select first row (0)
        self.table_view.selectRow(0)
        # Press "Ctrl+H"
        QTest.keySequence(self.numeric, QKeySequence(self.shortcuts["hex"]))
        self.processEvents()
        # Evaluate
        self.assertEqual(self.table_view.backend.signals[0].format, hex_format)
        self.assertEqual(self.table_view.model().format, hex_format)

        # Select first row (0)
        self.table_view.selectRow(0)
        # Press "Ctrl+P"
        QTest.keySequence(self.numeric, QKeySequence(self.shortcuts["physical"]))
        self.processEvents()
        # Evaluate
        self.assertEqual(self.table_view.backend.signals[0].format, phys_format)
        self.assertEqual(self.table_view.model().format, phys_format)

        # Select first row (0)
        self.table_view.selectRow(0)
        # Press "Ctrl+T"
        QTest.keySequence(self.numeric, QKeySequence(self.shortcuts["ascii"]))
        self.processEvents()
        # Evaluate
        self.assertEqual(self.table_view.backend.signals[0].format, ascii_format)
        self.assertEqual(self.table_view.model().format, ascii_format)

    def test_move_timestamp_cursor_left__right_shortcuts(self):
        """
        Test Scope:
            Check that Arrow Keys: Left & Right ensure navigation on channels evolution.
            Ensure that navigation is working.
        Events:
            - Display one signals to the widget
            - Send KeyClick Right 25 times
            - Send KeyClick Left 15 times
        Evaluate:
            - Evaluate that timestamp slider value is equal to the expected value
        """
        # Setup
        # Add channels to numeric
        self.assertIsNotNone(self.add_channels(["ASAM_[13][0].M.MATRIX_DIM_16_1.UBYTE.IDENTICAL"]))

        left_clicks = 15
        right_clicks = 25
        # Get current timestamp slider value (TSV)
        ts_slider_value_at_start = self.numeric.timestamp_slider.value()
        # Add to TSV right clicks value and reduce it with left clicks value
        expected_ts_slider_value = ts_slider_value_at_start + right_clicks - left_clicks

        # Events
        # Select first row (0)
        self.table_view.selectRow(0)
        self.processEvents()
        # Press right key right_clicks time
        for _ in range(right_clicks):
            QTest.keySequence(self.numeric, QKeySequence(self.shortcuts["move_cursor_right_1x"]))
        # Press left key left_clicks time
        for _ in range(left_clicks):
            QTest.keySequence(self.numeric, QKeySequence(self.shortcuts["move_cursor_left_1x"]))

        # Evaluate
        self.assertEqual(self.numeric.timestamp_slider.value(), expected_ts_slider_value)

    def test_save_active_subplot_channels_shortcut(self):
        """
        Test Scope:
            Check if by pressing "Ctrl+S" is saved in the new measurement file only active channels.

        Events:
            - Select 3 signals and create a numeric.
            - Ensure that the first sub-window is selected.
            - Press Key "Ctrl+S" -> input name for new file -> Save
            - Open recently created measurement file in a new window
        Evaluate:
            - Evaluate that object getSaveFileName() was called after pressing combination "Ctrl+S".
            - Evaluate that in measurement file is saved only active channels.
        """
        # Setup
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]))
        # Create a new Numeric widget window with few different from first widget items
        self.create_window(window_type="Numeric", channels_indexes=(20, 21, 22))

        expected_channels = [channel.name for channel in self.channels]
        expected_channels.append("time")
        file_path = os.path.join(self.test_workspace, "file.mf4")

        # Events
        # Select first row (0)
        self.table_view.selectRow(0)

        # mock for getSaveFileName object
        with mock.patch("asammdf.gui.widgets.numeric.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = (file_path, "")
            # Press Ctrl+S
            QTest.keySequence(self.numeric, QKeySequence(self.shortcuts["save_active_subplot_channels"]))

        # Evaluate
        mo_getSaveFileName.assert_called()

        # get waved file as MDF
        process_bus_logging = ("process_bus_logging", True)
        mdf_file = mdf.MDF(file_path, process_bus_logging=process_bus_logging)

        # Evaluate
        for name in expected_channels:
            self.assertIn(name, mdf_file.channels_db.keys())
        mdf_file.close()

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
        font_size = self.numeric.font().pointSize()
        QTest.keySequence(self.numeric, QKeySequence(self.shortcuts["decrease_font"]))
        self.assertLess(font_size, self.numeric.font().pointSize())

        font_size = self.numeric.font().pointSize()
        QTest.keySequence(self.numeric, QKeySequence(self.shortcuts["increase_font"]))
        self.assertGreater(font_size, self.numeric.font().pointSize())

    def test_go_to_timestamp_shortcut(self):
        """
        Test scope:
            Ensure that Shift+G switch to selected timestamp

        Events:
            - Display one item to the widget.
            - Press Shift+G -> input some value -> "Ok".

        Evaluate:
             - Evaluate that there is added one item to the widget.
             - Evaluate that getDouble object was called.
             - Evaluate that timestamp value is almost equal with the inputted value.
        Returns
        -------

        """
        # Setup
        # Add one item to the widget
        self.assertIsNotNone(self.add_channels([10]))

        # Event
        with mock.patch("asammdf.gui.widgets.numeric.QtWidgets.QInputDialog.getDouble") as mo_getDouble:
            expected_pos = 2
            mo_getDouble.return_value = expected_pos, True
            # Press Shift+G
            QTest.keySequence(self.numeric, QKeySequence(self.shortcuts["go_to_timestamp"]))

        # Evaluate
        mo_getDouble.assert_called()
        self.assertAlmostEqual(self.numeric.timestamp.value(), expected_pos, delta=0.01)

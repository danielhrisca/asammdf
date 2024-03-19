#!/usr/bin/env python\
import json
import os
import pathlib
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget
from unittest import mock

from PySide6 import QtCore, QtGui, QtTest

from asammdf.gui.utils import copy_ranges


class TestTableViewWidgetShortcuts(TestFileWidget):
    def setUp(self):
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        self.create_window(window_type="Numeric")

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.numeric = self.widget.mdi_area.subWindowList()[0].widget()
        self.table_view = self.numeric.channels.dataView

    def test_TableViewWidget_Shortcut_Shift_Key_Delete(self):
        """
        test for shortcut Delete
        Returns
        -------

        """
        self.processEvents()
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]), self.numeric)
        self.processEvents()
        channel_count = len(self.channels)

        # Select first row
        self.table_view.selectRow(0)
        self.processEvents()
        # Press key Delete
        QtTest.QTest.keyClick(self.table_view, QtCore.Qt.Key_Delete)

        # Evaluate
        self.assertEqual(len(self.table_view.backend.signals), channel_count - 1)

        # select all items
        QtTest.QTest.keySequence(self.table_view, QtGui.QKeySequence("Ctrl+A"))
        # Press key Delete
        QtTest.QTest.keyClick(self.table_view, QtCore.Qt.Key_Delete)

        # Evaluate
        self.assertEqual(len(self.table_view.backend.signals), 0)

    def test_TableViewWidget_Shortcut_Shift_Key_Ctrl_R(self):
        """
        test for shortcut Ctrl+R
        Returns
        -------

        """
        value1 = 0.0
        value2 = 100
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]), self.numeric)
        self.processEvents()
        green = QtGui.QColor.fromRgbF(0.000000, 1.000000, 0.000000, 1.000000)
        red = QtGui.QColor.fromRgbF(1.000000, 0.000000, 0.000000, 1.000000)

        range_editor_result = [
            {
                "background_color": green,
                "font_color": red,
                "op1": "<=",
                "op2": "<=",
                "value1": value1,
                "value2": value2,
            }
        ]

        # Select first row
        self.table_view.selectRow(0)
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.numeric.RangeEditor") as mo_RangeEditor:
            mo_RangeEditor.return_value.result = range_editor_result
            mo_RangeEditor.return_value.pressed_button = "apply"
            # Press "Alt+R"
            QtTest.QTest.keySequence(self.table_view, QtGui.QKeySequence("Ctrl+R"))

        # Evaluate
        mo_RangeEditor.assert_called()
        row_0 = True
        for value in self.table_view.ranges.values():
            if row_0 is True:  # Evaluate range for first row
                self.assertDictEqual(value[0], range_editor_result[0])
                row_0 = False
            else:  #
                self.assertListEqual(value, [])

    def test_TableViewWidget_Shortcut_Key_Ctrl_Shift_C(self):
        """
        Test for Ctrl+Shift+C shortcut
        """
        self.assertIsNotNone(self.add_channels([10]))
        signal = self.table_view.backend.signals[0]
        expected_ch_info = {
            "format": signal.format,
            "ranges": copy_ranges(self.table_view.ranges[signal.entry]),
        }
        with mock.patch("asammdf.gui.widgets.numeric.QtWidgets.QApplication.instance") as mo_instance:
            # Click on first channel
            self.table_view.selectRow(0)
            # Press Ctrl+Shift+C
            QtTest.QTest.keySequence(self.table_view, QtGui.QKeySequence("Ctrl+Shift+C"))
        # Evaluate
        mo_instance.return_value.clipboard.return_value.setText.assert_called_with(json.dumps(expected_ch_info))

    def test_TableViewWidget_Shortcut_Key_Ctrl_Shift_V(self):
        """
        - Add 2 channels to numeric
        - Set clipboard text = display_properties of first channel
        - Click on item -> Press Ctrl+V

        Evaluate
            - display_properties of both channels must be equal
        """
        self.assertIsNotNone(self.add_channels([10, 11]))
        # Evaluate precondition
        signal_0 = self.table_view.backend.signals[0]
        expected_ch_info = {
            "format": signal_0.format,
            "ranges": copy_ranges(self.table_view.ranges[signal_0.entry]),
        }
        with mock.patch("asammdf.gui.widgets.numeric.QtWidgets.QApplication.instance") as mo_instance:
            mo_instance.return_value.clipboard.return_value.text.return_value = expected_ch_info

            # Click on first channel
            self.table_view.selectRow(1)
            # Press Ctrl+Shift+C
            QtTest.QTest.keySequence(self.table_view, QtGui.QKeySequence("Ctrl+Shift+V"))
        # Evaluate
        mo_instance.return_value.clipboard.return_value.text.assert_called()
        signal_1 = self.table_view.backend.signals[1]
        self.assertEqual(signal_1.format, signal_0.format)
        self.assertEqual(self.table_view.ranges[signal_1.entry], self.table_view.ranges[signal_0.entry])


class TestNumericWidgetShortcuts(TestFileWidget):
    def setUp(self):
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        self.create_window(window_type="Numeric")

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.numeric = self.widget.mdi_area.subWindowList()[0].widget()
        self.table_view = self.numeric.channels.dataView

    def tearDown(self):
        if self.widget:
            self.widget.destroy()
        super().tearDown()

    def test_NumericWidget_Shortcut_Keys_Ctrl_H_Ctrl_B_Ctrl_P_Ctrl_T(self):
        """
        Test Scope:
            Check if values is converted to int, hex, bin after pressing combination of key "Ctrl+<H>|<B>|<P>"
        Events:
            - Open 'FileWidget' with valid measurement.
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
        QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Ctrl+B"))
        self.processEvents()
        # Evaluate
        self.assertEqual(self.table_view.backend.signals[0].format, bin_format)
        self.assertEqual(self.table_view.model().format, bin_format)

        # Select first row (0)
        self.table_view.selectRow(0)
        # Press "Ctrl+H"
        QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Ctrl+H"))
        self.processEvents()
        # Evaluate
        self.assertEqual(self.table_view.backend.signals[0].format, hex_format)
        self.assertEqual(self.table_view.model().format, hex_format)

        # Select first row (0)
        self.table_view.selectRow(0)
        # Press "Ctrl+P"
        QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Ctrl+P"))
        self.processEvents()
        # Evaluate
        self.assertEqual(self.table_view.backend.signals[0].format, phys_format)
        self.assertEqual(self.table_view.model().format, phys_format)

        # Select first row (0)
        self.table_view.selectRow(0)
        # Press "Ctrl+T"
        QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Ctrl+T"))
        self.processEvents()
        # Evaluate
        self.assertEqual(self.table_view.backend.signals[0].format, ascii_format)
        self.assertEqual(self.table_view.model().format, ascii_format)

    def test_NumericWidget_Shortcut_Keys_Left_and_Rih_Arrows(self):
        """
        ...
        Returns
        -------

        """
        # add channels to numeric
        self.assertIsNotNone(self.add_channels([35]))

        # Setup
        left_arrow_clicks = 15
        right_arrow_clicks = 25
        ts_slider_value_at_start = self.numeric.timestamp_slider.value()
        expected_ts_slider_value = ts_slider_value_at_start + right_arrow_clicks - left_arrow_clicks
        # Select first row (0)
        self.table_view.selectRow(0)
        self.processEvents()

        for _ in range(right_arrow_clicks):
            QtTest.QTest.keyClick(self.numeric, QtCore.Qt.Key_Right)

        for _ in range(left_arrow_clicks):
            QtTest.QTest.keyClick(self.numeric, QtCore.Qt.Key_Left)

        # Evaluate
        self.assertEqual(self.numeric.timestamp_slider.value(), expected_ts_slider_value)

    def test_NumericWidget_Shortcut_Key_Ctrl_S(self):
        """
        Test Scope:
            Check if by pressing "Ctrl+S" is saved in the new measurement file only active channels
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a numeric
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
        # Select first row (0)
        self.table_view.selectRow(0)
        # QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Ctrl+A"))
        expected_items = [channel.name for channel in self.channels]
        expected_items.append("time")
        file_path = os.path.join(self.test_workspace, "file.mf4")
        # mock for getSaveFileName object
        with mock.patch("asammdf.gui.widgets.numeric.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = (file_path, "")
            # Press Ctrl+S
            QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Ctrl+S"))
        # Evaluate
        mo_getSaveFileName.assert_called()

        # Open recently saved measurement file
        self.setUpFileWidget(measurement_file=file_path, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Evaluate
        for index in range(self.widget.channels_tree.topLevelItemCount() - 1):
            self.assertIn(self.widget.channels_tree.topLevelItem(index).name, expected_items)
            if self.widget.channels_tree.topLevelItem(index).name != "time":
                expected_items.remove(self.widget.channels_tree.topLevelItem(index).name)

    def test_NumericWidget_Shortcut_Keys_Ctrl_Left_and_Right_Buckets(self):
        """
        tests for Ctrl+[ and Ctrl+]
        """
        font_size = self.numeric.font().pointSize()
        QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Ctrl+]"))
        self.assertLess(font_size, self.numeric.font().pointSize())

        font_size = self.numeric.font().pointSize()
        QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Ctrl+["))
        self.assertGreater(font_size, self.numeric.font().pointSize())

    def test_NumericWidget_Shortcut_Key_Shift_G(self):
        """

        Returns
        -------

        """
        self.assertIsNotNone(self.add_channels([10]))

        with mock.patch("asammdf.gui.widgets.numeric.QtWidgets.QInputDialog.getDouble") as mo_getDouble:
            expected_pos = 2
            mo_getDouble.return_value = expected_pos, True
            QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Shift+G"))
        mo_getDouble.assert_called()
        self.assertAlmostEqual(self.numeric.timestamp.value(), expected_pos, delta=0.01)

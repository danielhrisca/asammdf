#!/usr/bin/env python\
import json
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

    def test_Plot_Channel_Selection_Shortcut_Key_Ctrl_Shift_C(self):
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

    def test_Plot_Channel_Selection_Shortcut_Key_Ctrl_Shift_V(self):
        """
        - Add 2 channels to plot
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

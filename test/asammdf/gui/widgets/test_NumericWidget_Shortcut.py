#!/usr/bin/env python\
import pathlib
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget
from unittest import mock

from PySide6 import QtCore, QtGui, QtTest


class TestTableViewWidgetShortcuts(TestFileWidget):
    def setUp(self):
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        self.create_window(window_type="Numeric")

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.numeric = self.widget.mdi_area.subWindowList()[0].widget()

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
        self.numeric.channels.dataView.selectRow(0)
        self.processEvents()
        # Press key Delete
        QtTest.QTest.keyClick(self.numeric, QtCore.Qt.Key_Delete)

        # Evaluate
        self.assertEqual(len(self.numeric.channels.dataView.backend.signals), channel_count - 1)

        # select all items
        QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Ctrl+A"))
        # Press key Delete
        QtTest.QTest.keyClick(self.numeric, QtCore.Qt.Key_Delete)

        # Evaluate
        self.assertEqual(len(self.numeric.channels.dataView.backend.signals), 0)

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
        self.numeric.channels.dataView.selectRow(0)
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.numeric.RangeEditor") as mo_RangeEditor:
            mo_RangeEditor.return_value.result = range_editor_result
            mo_RangeEditor.return_value.pressed_button = "apply"
            # Press "Alt+R"
            QtTest.QTest.keySequence(self.numeric, QtGui.QKeySequence("Ctrl+R"))

        # Evaluate
        mo_RangeEditor.assert_called()
        row_0 = True
        for value in self.numeric.channels.dataView.ranges.values():
            if row_0 is True:       # Evaluate range for first row
                self.assertDictEqual(value[0], range_editor_result[0])
                row_0 = False
            else:           #
                self.assertListEqual(value, [])

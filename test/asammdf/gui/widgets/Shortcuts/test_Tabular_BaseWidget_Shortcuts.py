#!/usr/bin/env python\

from os import path
from pathlib import Path
from sys import platform
from unittest import mock, skipIf

import numpy
from PySide6 import QtGui, QtTest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication

from asammdf import mdf
from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget


class TestDataTableViewShortcuts(TestFileWidget):
    def setUp(self):
        """
        Events:
            - Open valid measurement file in ASAMMDF
            - Create a Tabular window with two channels on it

        Evaluate:
            - Evaluate that is one active sub-window
        Returns
        -------

        """
        super().setUp()
        # Open measurement file
        measurement_file = str(Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        # Create a tabular window
        self.create_window(window_type="Tabular", channels_indexes=(35, 36))
        # Evaluate
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.tabular = self.widget.mdi_area.subWindowList()[0].widget()
        self.dtw = self.tabular.tree.dataView
        # Load shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.dtw))
        self.processEvents(0.01)

    def test_set_color_range_shortcut(self):
        """
            Test Scope:
                Ensure that after pressing key Ctrl+R, range editor will be triggered

            Events:
                - Click on first channel
                - Press Ctrl+R
                - Add range
                - Ok

            Evaluate:
                - Evaluate that range editor was triggered after pressing key Ctrl+R
                - Evaluate that range was applied for only for selected signal
                - Evaluate that only cells with value within range are colored
        Returns
        -------

        """
        # Setup
        col = 1
        value1 = 0.0
        value2 = 150.0
        self.processEvents()
        green = QtGui.QColor.fromRgbF(0.000000, 1.000000, 0.000000, 1.000000)
        green_brush = QtGui.QBrush(green, QtGui.Qt.SolidPattern)
        red = QtGui.QColor.fromRgbF(1.000000, 0.000000, 0.000000, 1.000000)
        red_brush = QtGui.QBrush(red, QtGui.Qt.SolidPattern)
        range_editor_result = [
            {
                "background_color": green_brush,
                "font_color": red_brush,
                "op1": "<",
                "op2": "<",
                "value1": value1,
                "value2": value2,
            }
        ]

        # Select first row
        self.dtw.selectColumn(col)
        # QtTest.QTest.keySequence(self.dtw, QtGui.QKeySequence(self.shortcuts["set_color_range"]))
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.tabular_base.RangeEditor") as mo_RangeEditor:
            mo_RangeEditor.return_value.result = range_editor_result
            mo_RangeEditor.return_value.pressed_button = "apply"
            # Press "Ctrl+R"
            QtTest.QTest.keySequence(self.dtw, QtGui.QKeySequence(self.shortcuts["set_color_range"]))
            self.processEvents()
        self.dtw.selectColumn(col)
        self.processEvents(0.01)
        # Evaluate
        mo_RangeEditor.assert_called()
        col_0 = True
        for value in self.tabular.ranges.values():
            if col_0 is True:  # Evaluate range for first row
                self.assertDictEqual(value[0], range_editor_result[0])
                col_0 = False
            else:
                self.assertListEqual(value, [])

        # Evaluate cell color
        row = 0
        # Find regular cell
        for row in range(self.dtw.pgdf.df.values.size - 1):
            if self.dtw.pgdf.df.values[row][1] > value2:
                break

                # GoTo regular row
        self.dtw.selectRow(row)
        self.dtw.selectRow(row)
        self.processEvents(0.01)

        # Cell to evaluate
        pm = self.dtw.grab(
            QRect(
                self.dtw.columnViewportPosition(col),
                self.dtw.rowViewportPosition(row),
                self.dtw.columnWidth(col),
                self.dtw.rowHeight(row),
            )
        )
        # Evaluate
        self.assertFalse(Pixmap.has_color(pm, red))
        self.assertFalse(Pixmap.has_color(pm, green))

        # Find colored row
        for row in range(self.dtw.pgdf.df.values.size - 1):
            if value1 < self.dtw.pgdf.df.values[row][1] < value2:
                break

        # GoTo colored row
        self.dtw.selectRow(row)
        self.dtw.selectRow(row)
        self.processEvents(0.01)

        # Cell to evaluate
        pm = self.dtw.grab(
            QRect(
                self.dtw.columnViewportPosition(col),
                self.dtw.rowViewportPosition(row),
                self.dtw.columnWidth(col),
                self.dtw.rowHeight(row),
            )
        )
        # Evaluate
        self.assertTrue(Pixmap.has_color(pm, green))
        self.assertTrue(Pixmap.has_color(pm, red))


class TestTabularBaseShortcuts(TestFileWidget):
    def setUp(self):
        """
        Events:
            - Open valid measurement file in ASAMMDF
            - Create a Tabular window with two channels on it

        Evaluate:
            - Evaluate that is one active sub-window
        Returns
        -------

        """
        super().setUp()
        # Open measurement file
        measurement_file = str(Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        self.create_window(window_type="Tabular", channels_indexes=(35, 36))

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.tabular = self.widget.mdi_area.subWindowList()[0].widget()
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.tabular))
        self.processEvents(0.01)

    def test_bin__hex__physical_shortcuts(self):
        """
            Test scope:
                Ensure that data format is changed after pressing keys
                    - Ctrl+B -> to binary
                    - Ctrl+H -> to hexadecimal
                    - Ctrl+P -> to physical

            Events:
                - Press Ctrl+H
                - Press Ctrl+B
                - Press Ctrl+P

            Evaluate:
                - Evaluate that data format and value were changed to bin after pressing key Ctrl+H
                - Evaluate that data format and value were changed to hex after pressing key Ctrl+B
                - Evaluate that data format and value were changed to physical after pressing key Ctrl+P

        Returns
        -------

        """
        # Setup
        self.tabular.set_format("phys")
        data_view = self.tabular.tree.dataView
        # index for first cell of first signal
        index = data_view.indexAt(QPoint(data_view.columnViewportPosition(1), data_view.rowViewportPosition(0)))
        self.processEvents(0.01)
        # Store value from cell
        data = data_view.model().data(index)

        # Event
        # Press Ctrl+H
        QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence(self.shortcuts["hex"]))
        self.processEvents(0.1)
        # Store value from cell
        hex_data = data_view.model().data(index)

        # Evaluate
        self.assertEqual(self.tabular.format, "hex")
        self.assertIn("0x", hex_data)
        self.assertEqual(int(hex_data, 16), int(data))

        # Event
        # Press Ctrl+B
        QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence(self.shortcuts["bin"]))
        self.processEvents(0.1)
        # Store value from cell
        bin_data = data_view.model().data(index)

        # Evaluate
        self.assertEqual(self.tabular.format, "bin")
        self.assertIn(".", bin_data)
        self.assertEqual(int(bin_data.replace(".", ""), 2), int(data))

        # Event
        # Press Ctrl+P
        QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence(self.shortcuts["physical"]))
        self.processEvents(0.1)
        # Store value from cell
        phys_data = data_view.model().data(index)

        # Evaluate
        self.assertEqual(self.tabular.format, "phys")
        self.assertTrue(phys_data.isdecimal())
        self.assertEqual(int(phys_data), int(data))

    def test_save_active_subplot_channels_shortcut(self):
        """
            Test scope:
                Ensure that window Save as file was called
                    and all active subplots were saved into a new file

            Events:
                - Open second sub-window
                - Select first one
                - Press Ctrl+S
                - Set new name for file and click "Save" button

            Evaluate:
                - Evaluate that there are two active sub-windows
                - Evaluate that "Save as file" window was called
                - Evaluate that in new saved file is only channels from selected active sub-window
        Returns
        -------

        """
        # Setup
        self.create_window(window_type="Tabular", channels_indexes=(10, 11))
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 2)
        self.processEvents()
        expected_channels = ["time"]
        for key in self.tabular.ranges.keys():
            expected_channels.append(key)

        file_path = path.join(self.test_workspace, "file.mf4")
        # mock for getSaveFileName object
        with mock.patch("asammdf.gui.widgets.tabular_base.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = (file_path, "")
            # Press Ctrl+S
            QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence(self.shortcuts["save_active_subplot_channels"]))
        # Evaluate
        mo_getSaveFileName.assert_called()

        # get saved file as MDF
        process_bus_logging = ("process_bus_logging", True)
        mdf_file = mdf.MDF(file_path, process_bus_logging=process_bus_logging)

        # Evaluate
        for name in mdf_file.channels_db.keys():
            self.assertIn(name, expected_channels)
            expected_channels.remove(name)
        self.assertEqual(len(expected_channels), 0)
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
        font_size = self.tabular.tree.dataView.font().pointSize()
        QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence(self.shortcuts["decrease_font"]))
        self.assertLess(font_size, self.tabular.tree.dataView.font().pointSize())

        font_size = self.tabular.tree.dataView.font().pointSize()
        QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence(self.shortcuts["increase_font"]))
        self.assertGreater(font_size, self.tabular.tree.dataView.font().pointSize())


@skipIf(platform != "win32", "Failed on linux. Shortcut can copy only value for one cell")
class TestDataFrameViewerShortcuts(TestFileWidget):
    def setUp(self):
        """
            Events:
                - Open valid measurement file in ASAMMDF
                - Create a Tabular window with two channels on it

            Evaluate:
                - Evaluate that is one active sub-window
        Returns
        -------

        """
        super().setUp()
        # Open measurement file
        measurement_file = str(Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        self.create_window(window_type="Tabular", channels_indexes=(35, 36))

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.tabular = self.widget.mdi_area.subWindowList()[0].widget()
        self.dfw = self.tabular.tree
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.dfw))
        self.processEvents(0.01)

    def test_copy_shortcut(self):
        """
            Test scope:
                Ensure that Ctrl+C shortcut will copy value from selected row to clipboard

            Events:
                - Select one row.
                - Press Ctrl+C.

            Evaluate:
             - Evaluate that value from the selected row was copied to clipboard after shortcut Ctrl+C was pressed.
        Returns
        -------

        """

        def copy_header():
            df = self.dfw.pgdf.df
            copied_values = ""
            for name in df.columns:
                copied_values += "\t" + name
            return copied_values + "\n"

        def copy_row(r):
            df = self.dfw.pgdf.df
            fp = self.dfw.dataView.model().float_precision
            copied_values = ""
            for name in df.columns:
                x = df[name][r]
                # if x == r:
                #     continue
                if isinstance(x, numpy.floating):
                    if fp != -1:
                        x = f"\t{x:.16f}"
                    else:
                        x = f"\t{x:.16f}"
                else:
                    x = f"\t{x}"
                copied_values += x

            return copied_values + "\n"

        # -|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|-

        # Setup
        row = 0
        expected_data = copy_header() + "0" + copy_row(row)

        # Event
        self.dfw.dataView.selectRow(row)
        self.processEvents(0.1)
        QtTest.QTest.keySequence(self.dfw, QtGui.QKeySequence(self.shortcuts["copy"]))
        self.processEvents(1)
        data = QApplication.instance().clipboard().text()

        # Test failed, because of clipboard buffer on linux
        # Evaluate
        self.assertEqual(expected_data, data)

        # -|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|-

        # Setup 2
        expected_data = copy_header()
        for row in range(len(self.dfw.pgdf.df.values)):
            expected_data += str(row) + copy_row(row)

        # Event 2
        QtTest.QTest.keySequence(self.dfw, QtGui.QKeySequence("Ctrl+A"))
        self.processEvents()
        QtTest.QTest.keySequence(self.dfw, QtGui.QKeySequence(self.shortcuts["copy"]))
        self.processEvents(1)

        data = QApplication.instance().clipboard().text()

        # Evaluate
        self.assertEqual(expected_data, data)

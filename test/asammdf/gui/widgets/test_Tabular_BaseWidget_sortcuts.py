#!/usr/bin/env python\
import os
import pathlib
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget
from unittest import mock

from PySide6 import QtGui, QtTest


class TestDataTableViewWidgetShortcuts(TestFileWidget):
    def setUp(self):
        super().setUp()
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        self.create_window(window_type="Tabular", channels_indexes=(35, 36))

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.tabular = self.widget.mdi_area.subWindowList()[0].widget()
        self.dtw = self.tabular.tree.dataView
        self.processEvents(0.01)

    def test_DataTableViewWidget_Shortcut_Key_Ctrl_R(self):
        """
        test for shortcut Ctrl+R
        Returns
        -------

        """
        value1 = 0.0
        value2 = 100
        self.processEvents()
        # self.tabular.grab().save("D:\\tabular.png")
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
        self.dtw.selectColumn(1)
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.tabular_base.RangeEditor") as mo_RangeEditor:
            mo_RangeEditor.return_value.result = range_editor_result
            mo_RangeEditor.return_value.pressed_button = "apply"
            # Press "Alt+R"
            QtTest.QTest.keySequence(self.dtw, QtGui.QKeySequence("Ctrl+R"))

        # Evaluate
        mo_RangeEditor.assert_called()
        col_0 = True
        for value in self.tabular.ranges.values():
            if col_0 is True:  # Evaluate range for first row
                self.assertDictEqual(value[0], range_editor_result[0])
                col_0 = False
            else:  #
                self.assertListEqual(value, [])


class TestTabularBaseWidgetShortcuts(TestFileWidget):
    def setUp(self):
        super().setUp()
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        self.create_window(window_type="Tabular", channels_indexes=(35, 36))

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.tabular = self.widget.mdi_area.subWindowList()[0].widget()
        self.processEvents(0.01)

    def test_TabularBaseWidget_Shortcut_Keys_Ctrl_H__Ctrl_B__Ctrl_P(self):
        """
        test for Ctrl+H, Ctrl+B, Ctrl+P
        Returns
        -------

        """
        # Event
        QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence("Ctrl+H"))
        # Evaluate
        self.assertEqual(self.tabular.format, "hex")

        # Event
        QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence("Ctrl+B"))
        # Evaluate
        self.assertEqual(self.tabular.format, "bin")

        # Event
        QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence("Ctrl+P"))
        # Evaluate
        self.assertEqual(self.tabular.format, "phys")

    def test_TabularBaseWidget_Shortcut_Key_Ctrl_S(self):
        """
        test for shortcut Ctrl+S
        Returns
        -------

        """
        # Setup
        self.create_window(window_type="Tabular", channels_indexes=(10, 11))
        self.processEvents()
        expected_channels = ["time"]
        for key in self.tabular.ranges.keys():
            if "timestamps" not in key:
                expected_channels.append(key)

        file_path = os.path.join(self.test_workspace, "file.mf4")
        # mock for getSaveFileName object
        with mock.patch("asammdf.gui.widgets.tabular_base.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = (file_path, "")
            # Press Ctrl+S
            QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence("Ctrl+S"))
        # Evaluate
        mo_getSaveFileName.assert_called()

        # Open recently saved measurement file
        self.setUpFileWidget(measurement_file=file_path, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Evaluate
        for index in range(self.widget.channels_tree.topLevelItemCount() - 1):
            self.assertIn(self.widget.channels_tree.topLevelItem(index).name, expected_channels)
            if self.widget.channels_tree.topLevelItem(index).name != "time":
                expected_channels.remove(self.widget.channels_tree.topLevelItem(index).name)
        self.assertListEqual(expected_channels, ["time"])

    def test_TabularBaseWidget_Shortcut_Keys_Ctrl_Left_and_Right_Buckets(self):
        """
        tests for Ctrl+[ and Ctrl+]
        """
        font_size = self.tabular.tree.dataView.font().pointSize()
        QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence("Ctrl+]"))
        self.assertLess(font_size, self.tabular.tree.dataView.font().pointSize())

        font_size = self.tabular.tree.dataView.font().pointSize()
        QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence("Ctrl+["))
        self.assertGreater(font_size, self.tabular.tree.dataView.font().pointSize())

    def test_TabularBaseWidget_Shortcut_Key_Shift_G(self):
        """
        test gor Shift+G
        Returns
        -------

        """
        with mock.patch("asammdf.gui.widgets.tabular_base.QtWidgets.QInputDialog.getDouble") as mo_getDouble:
            expected_pos = 2
            mo_getDouble.return_value = expected_pos, True
            QtTest.QTest.keySequence(self.tabular, QtGui.QKeySequence("Shift+G"))
        mo_getDouble.assert_called()

        h = self.tabular.tree.dataView.height()
        index = self.tabular.tree.dataView.rowAt(h - 1)
        # self.tabular.tree.dataView.selectionModel().selection().indexes()
        self.assertIn(index, self.tabular.tree.dataView.selectedIndexes())


class TestDataFrameViewerWidgetShortcuts(TestFileWidget):
    def setUp(self):
        super().setUp()
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)

        self.create_window(window_type="Tabular", channels_indexes=(35, 36))

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.tabular = self.widget.mdi_area.subWindowList()[0].widget()
        self.dfw = self.tabular.tree
        self.processEvents(0.01)

    def test_DataFrameViewerWidget_Shortcut_Keys_Ctrl_C(self):
        """
        test for Ctrl+C
        Returns
        -------

        """
        with mock.patch("asammdf.gui.widgets.tabular_base.threading.Thread") as mo_Thread:
            # Click on first channel
            # Press Ctrl+Shift+C
            QtTest.QTest.keySequence(self.dfw, QtGui.QKeySequence("Ctrl+A"))
            QtTest.QTest.keySequence(self.dfw, QtGui.QKeySequence("Ctrl+C"))
        # Evaluate
        mo_Thread.return_value.start.assert_called()
        # Evaluate args

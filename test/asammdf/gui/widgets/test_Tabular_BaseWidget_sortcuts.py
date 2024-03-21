#!/usr/bin/env python\
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

        self.create_window(window_type="Tabular", channels_indexes=(35,))

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.tabular = self.widget.mdi_area.subWindowList()[0].widget()
        self.dtw = self.tabular.tree.dataView

    def test_DataTableViewWidget_Shortcut_Ctrl_R(self):
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
        # row_0 = True
        # for value in self.table_view.ranges.values():
        #     if row_0 is True:  # Evaluate range for first row
        #         self.assertDictEqual(value[0], range_editor_result[0])
        #         row_0 = False
        #     else:  #
        #         self.assertListEqual(value, [])

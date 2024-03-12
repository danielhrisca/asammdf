#!/usr/bin/env python\
import pathlib
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget

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

        Returns
        -------

        """
        # self.manual_use(self.widget)
        self.processEvents()
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]), self.numeric)
        self.processEvents()
        x = self.numeric.grab().save("D:\\numeric.png")
        channel_count = len(self.channels)
        # Click on last channel
        channel_0 = self.channels.pop()

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

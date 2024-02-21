#!/usr/bin/env python\

import pathlib
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget

from PySide6 import QtCore, QtGui, QtTest


class TestTreeWidgetShortcuts(TestFileWidget):
    def setUp(self):
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        self.tw = self.widget.channels_tree

    def test_TreeWidgetShortcut_Key_Space(self):
        """

        Returns
        -------

        """
        items_count = self.tw.topLevelItemCount()
        for _ in range(items_count - 1):
            # Evaluate that all items is ...
            self.assertEqual(self.tw.topLevelItem(_).checkState(0), QtCore.Qt.CheckState.Unchecked)

            # Select all impar items and hit the Space key
            if _ % 2 == 1:
                self.tw.topLevelItem(_).setSelected(True)
                QtTest.QTest.keyClick(self.tw, QtGui.Qt.Key_Space)
                self.tw.topLevelItem(_).setSelected(False)

        self.processEvents()

        # Evaluate that all impar items are checked
        for _ in range(items_count - 1):
            if _ % 2 == 1:
                self.assertEqual(self.tw.topLevelItem(_).checkState(0), QtCore.Qt.CheckState.Checked)

                # Select item and hit again the Space key
                self.tw.topLevelItem(_).setSelected(True)
                QtTest.QTest.keyClick(self.tw, QtGui.Qt.Key_Space)
                self.tw.topLevelItem(_).setSelected(False)
            else:
                self.assertEqual(self.tw.topLevelItem(_).checkState(0), QtCore.Qt.CheckState.Unchecked)

        # Evaluate that all items are unchecked
        for _ in range(items_count - 1):
            self.assertEqual(self.tw.topLevelItem(_).checkState(0), QtCore.Qt.CheckState.Unchecked)


class TestChannelsTreeWidgetShortcuts(TestFileWidget):
    def setUp(self): ...

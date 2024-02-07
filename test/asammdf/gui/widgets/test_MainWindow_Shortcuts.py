#!/usr/bin/env python

from asammdf.gui.widgets.file import FileWidget
from asammdf.gui.widgets.main import MainWindow
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget
from unittest import mock

from PySide6 import QtGui, QtTest, QtWidgets


class TestShortcuts(TestPlotWidget):
    def setUp(self):
        self.mw = MainWindow()
        self.mw.showNormal()
        self.processEvents()

    def destroyMW(self):
        if self.mw:
            self.mw.close()
            self.mw.deleteLater()

    def openFile(self, measurement_files):
        with mock.patch("asammdf.gui.widgets.main.QtWidgets.QFileDialog.getOpenFileNames") as mo_getOpenFileNames:
            if measurement_files:
                if isinstance(measurement_files, list):
                    mo_getOpenFileNames.return_value = measurement_files, ""
                    QtTest.QTest.keySequence(self.mw, QtGui.QKeySequence("Ctrl+O"))
                else:
                    mo_getOpenFileNames.return_value = [measurement_files], ""
                    QtTest.QTest.keySequence(self.mw, QtGui.QKeySequence("Ctrl+O"))
            else:
                mo_getOpenFileNames.return_value = [self.measurement_file], ""
                QtTest.QTest.keySequence(self.mw, QtGui.QKeySequence("Ctrl+O"))
        self.processEvents()
        mo_getOpenFileNames.assert_called()

    def tearDown(self):
        if self.mw:
            self.destroyMW()

    def test_Shortcut_Key_F11(self):
        """
        Events:
            - MainWindow is open by test "setUp" method
            - Open valid file
            - Send Key `F11`
            - Close MainWindow
        Expected:
            - Method "toggle_fullscreen" should be triggered as following trigger action
            - Ensure that FileWidget is closed too. No widget left open behind.
        :return:
        """
        # Setup
        with mock.patch.object(self.mw, "toggle_fullscreen", wraps=self.mw.toggle_fullscreen) as mock_showFullScreen:
            # Event
            self.openFile(measurement_files=None)
            # file widget
            self.widget = self.mw.files.widget(0)
            # Press F11
            QtTest.QTest.keyClick(self.widget, QtGui.Qt.Key_F11)
            self.processEvents()
            # Evaluate
            # Identify Partial Object linked to save_configuration
            mock_showFullScreen.assert_called()
        # Close MainWindow
        self.destroyMW()
        self.processEvents()
        # Evaluate
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(widget, FileWidget):
                self.assertFalse(widget.isVisible())

    def test_Shortcut_Ctrl_Key_O(self):
        """
        Events:
            - MainWindow is open by test "setUp" method
            - Open valid file
            - Press Ctrl+O, this action is triggered in openFile() method
            - Select a valid measurement file
        Expected:
            - Method "getOpenFileNames" should be triggered as following trigger action
            - Ensure that file is opened and available as file widget
        :return:
        """
        self.openFile(None)
        self.assertEqual(self.mw.files.count(), 1)

    def test_Shortcut_Key_F6(self):
        """
        Events:
            - MainWindow is open by test "setUp" method
            - Open valid file
            - Send Key `F11`
            - Close MainWindow
        Expected:
            - Method "FunctionsManagerDialog" should be triggered as following trigger action
        :return:
        """
        # Setup
        with mock.patch("asammdf.gui.widgets.main.FunctionsManagerDialog") as mock_FunctionsManagerDialog:
            # Event
            self.openFile(measurement_files=None)
            # Press F6
            QtTest.QTest.keyClick(self.mw, QtGui.Qt.Key_F6)
            self.processEvents()
        # Evaluate
        mock_FunctionsManagerDialog.assert_called()

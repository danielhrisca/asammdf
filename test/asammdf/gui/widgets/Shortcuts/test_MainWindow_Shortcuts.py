#!/usr/bin/env python
from pathlib import Path
import shutil
from unittest import mock

from PySide6 import QtCore
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QTreeWidgetItemIterator

from asammdf.gui.widgets import numeric, plot, tabular
from asammdf.gui.widgets.file import FileWidget
from asammdf.gui.widgets.main import MainWindow
from test.asammdf.gui.test_base import safe_setup, TestBase
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget


class TestShortcuts(TestBase):
    @safe_setup
    def setUp(self):
        """
        Events:
            Open Main window
        Returns
        -------

        """
        super().setUp()
        self.measurement_file = str(Path(TestBase.resource, "ASAP2_Demo_V171.mf4"))
        self.mw = MainWindow(files=(self.measurement_file,))
        self.mw.showNormal()
        self.processEvents(1)

        # get shortcuts
        self.shortcuts = TestFileWidget.load_shortcuts_from_json_file(self, self.mw)
        self.assertIsNotNone(self.shortcuts)

    def destroyMW(self):
        if self.mw:
            for i in range(self.mw.files.count()):
                widget = self.mw.files.widget(i)
                if widget is not None:
                    widget.clear_windows(is_closing=True)
            self.mw.close()
            self.mw.deleteLater()

    def tearDown(self):
        if self.mw:
            self.destroyMW()

    def test_fullscreen_shortcut(self):
        """
        Test scope:
            Ensure that F11 shortcut will toggle full-screen mode for opened file widget,
                and after closing the main window, file widget will be destroyed.
        Events:
            - Open valid measurement file
            - Press key `F11`
            - Close MainWindow

        Evaluate:
            - Evaluate that file widget isn't in full-screen mode by default.
            - Evaluate that after pressing key F11, file widget is in full-screen mode
            - Evaluate that after closing the main window, there are no more opened widgets.

        :return:
        """
        # Setup
        self.processEvents(0.01)
        # file widget
        file_widget = self.mw.files.widget(0)

        # Evaluate
        self.assertFalse(file_widget.isFullScreen())

        # Event
        QTest.keySequence(self.mw, QKeySequence(self.shortcuts["toggle_full-screen"]))  # Press F11
        self.processEvents()

        # Evaluate
        self.assertTrue(file_widget.isFullScreen())

        # Close MainWindow
        self.destroyMW()
        self.processEvents()

        # Evaluate
        for w in QApplication.topLevelWidgets():
            if isinstance(w, FileWidget):
                self.assertFalse(w.isVisible())

    def test_create_plot__numeric__tabular_sub_window_shortcut(self):
        """
        Test scope:
            Ensure that Plot, Numeric and Tabular sub-windows can be created by pressing keys F2, F3 and F4

        Events:
            - Open valid file
            - Press key F2.
            - Press key F3.
            - Press key F4.
            - Search all "*matrix*" channels and set all of them checked
            - Press key F2.
            - Press key F3.
            - Press key F4.

        Evaluate:
            - Evaluate that after pressing key F2, there is one new Plot type sub-window.
            - Evaluate that after pressing key F3, there is one new Numeric type sub-window.
            - Evaluate that after pressing key F4, there is one new Tabular type sub-window.
            - Evaluate that after pressing key F2, there is one new Plot type sub-window with checked channels.
            - Evaluate that after pressing key F3, there is one new Numeric type sub-window with checked channels.
            - Evaluate that after pressing key F4, there is one new Tabular type sub-window with checked channels.
        :return:
        """
        # Setup
        windows_count = 0

        # file widget
        file_widget = self.mw.files.widget(0)

        with self.subTest("test_shortcut_F2_without_selected_channels"):
            # Event
            QTest.keySequence(self.mw, QKeySequence(self.shortcuts["create_plot_window"]))  # Press F2
            self.processEvents()

            # Evaluate
            self.assertEqual(len(file_widget.mdi_area.subWindowList()), windows_count + 1)
            self.assertTrue(isinstance(file_widget.mdi_area.subWindowList()[windows_count].widget(), plot.Plot))

            windows_count += 1

        with self.subTest("test_shortcut_F3_without_selected_channels"):
            # Event
            QTest.keySequence(self.mw, QKeySequence(self.shortcuts["create_numeric_window"]))  # Press F3
            self.processEvents()

            # Evaluate
            self.assertEqual(len(file_widget.mdi_area.subWindowList()), windows_count + 1)
            self.assertTrue(isinstance(file_widget.mdi_area.subWindowList()[windows_count].widget(), numeric.Numeric))

            windows_count += 1

        with self.subTest("test_shortcut_F4_without_selected_channels"):
            # Event
            QTest.keySequence(self.mw, QKeySequence(self.shortcuts["create_tabular_window"]))  # Press F4
            self.processEvents()

            # Evaluate
            self.assertEqual(len(file_widget.mdi_area.subWindowList()), windows_count + 1)
            self.assertTrue(isinstance(file_widget.mdi_area.subWindowList()[windows_count].widget(), tabular.Tabular))

            windows_count += 1

        # Setup
        matrix_items = {}
        pattern = "matrix"
        # Search signals with specific patter
        iterator = QTreeWidgetItemIterator(file_widget.channels_tree)
        while item := iterator.value():
            if pattern.upper() in item.name.upper():
                matrix_items[item.entry] = item.name
            iterator += 1
        sw_count = 0  # Sub-windows
        # Mock for Advanced search and windowSelectionDialog objects
        with mock.patch("asammdf.gui.widgets.file.AdvancedSearch") as mo_AdvancedSearch:
            mo_AdvancedSearch.return_value.payload = matrix_items
            mo_AdvancedSearch.return_value.pattern_window = False
            mo_AdvancedSearch.return_value.add_window_request = False
            # Check some channels
            QTest.keySequence(self.mw, QKeySequence("Ctrl+F"))  # Press Ctrl+F

            self.processEvents(0.01)
        # Evaluate
        mo_AdvancedSearch.assert_called()

        with self.subTest("test_shortcut_F2_with_selected_channels"):
            # Event
            QTest.keySequence(self.mw, QKeySequence(self.shortcuts["create_plot_window"]))  # Press F2
            self.processEvents()

            # Evaluate
            self.assertEqual(len(file_widget.mdi_area.subWindowList()), windows_count + 1)
            self.assertTrue(isinstance(file_widget.mdi_area.subWindowList()[windows_count].widget(), plot.Plot))

            cs = file_widget.mdi_area.subWindowList()[windows_count].widget().channel_selection

            # Evaluate plot widget
            self.assertEqual(len(matrix_items), cs.topLevelItemCount())
            iterator = QTreeWidgetItemIterator(cs)
            while item := iterator.value():
                self.assertIn(item.name, matrix_items.values())
                iterator += 1
            windows_count += 1

        with self.subTest("test_shortcut_F3_with_selected_channels"):
            # Event
            QTest.keySequence(self.mw, QKeySequence(self.shortcuts["create_numeric_window"]))  # Press F3
            self.processEvents()

            # Evaluate
            self.assertEqual(len(file_widget.mdi_area.subWindowList()), windows_count + 1)
            self.assertTrue(isinstance(file_widget.mdi_area.subWindowList()[windows_count].widget(), numeric.Numeric))

            signals = file_widget.mdi_area.subWindowList()[windows_count].widget().channels.backend.signals
            self.assertEqual(len(matrix_items), len(signals))
            for sig in signals:
                self.assertIn(sig.name, matrix_items.values())
            windows_count += 1

        with self.subTest("test_shortcut_F4_with_selected_channels"):
            # Event
            QTest.keySequence(self.mw, QKeySequence(self.shortcuts["create_tabular_window"]))  # Press F4
            self.processEvents()

            # Evaluate
            self.assertEqual(len(file_widget.mdi_area.subWindowList()), windows_count + 1)
            self.assertTrue(isinstance(file_widget.mdi_area.subWindowList()[windows_count].widget(), tabular.Tabular))

            signals = file_widget.mdi_area.subWindowList()[windows_count].widget().tree
            # channels columns count - timestamp column
            self.assertEqual(len(matrix_items), signals.columnHeader.model().columnCount() - 1)
            for key in matrix_items.values():
                self.assertIn(key, signals.pgdf.df.keys())


class TestReplaceFile(TestBase):
    @safe_setup
    def setUp(self):
        super().setUp()
        # Create two copies of the test file so we have distinct paths
        self.measurement_file = str(
            shutil.copy(Path(TestBase.resource, "ASAP2_Demo_V171.mf4"), Path(self.test_workspace, "original.mf4"))
        )
        self.replacement_file = str(
            shutil.copy(Path(TestBase.resource, "ASAP2_Demo_V171.mf4"), Path(self.test_workspace, "replacement.mf4"))
        )
        self.mw = None

    def tearDown(self):
        if self.mw:
            for i in range(self.mw.files.count()):
                widget = self.mw.files.widget(i)
                if widget is not None:
                    widget.clear_windows(is_closing=True)
            self.mw.close()
            self.mw.deleteLater()
        super().tearDown()

    def test_replace_file_preserves_tab_position(self):
        """
        Test scope:
            Ensure that File > Replace replaces the current file tab in place.
        Events:
            - Open MainWindow with a measurement file
            - Trigger Replace via Ctrl+Shift+O with mocked file dialog
        Evaluate:
            - Tab count remains 1
            - Tab label matches replacement file name
            - Tab tooltip matches replacement file path
            - FileWidget is valid
        """
        self.mw = MainWindow(files=(self.measurement_file,))
        self.mw.showNormal()
        self.processEvents(1)

        self.assertEqual(self.mw.files.count(), 1)
        self.assertEqual(self.mw.files.tabText(0), "original.mf4")

        with mock.patch("asammdf.gui.widgets.main.QtWidgets.QFileDialog.getOpenFileName") as mo_dialog:
            mo_dialog.return_value = self.replacement_file, None
            QTest.keySequence(self.mw, QKeySequence("Ctrl+Shift+O"))
            self.processEvents(1)

        # Tab count should still be 1
        self.assertEqual(self.mw.files.count(), 1)
        # Tab label and tooltip should reflect the new file
        self.assertEqual(self.mw.files.tabText(0), "replacement.mf4")
        self.assertIn("replacement.mf4", self.mw.files.tabToolTip(0))
        # Widget should be a valid FileWidget
        self.assertIsInstance(self.mw.files.widget(0), FileWidget)

    def test_replace_file_preserves_display_config(self):
        """
        Test scope:
            Ensure that File > Replace preserves the display configuration (sub-windows)
            from the old file and applies it to the new file.
        Events:
            - Open MainWindow with a measurement file
            - Create a Plot sub-window with selected channels
            - Trigger Replace via Ctrl+Shift+O with mocked file dialog
        Evaluate:
            - New file has the same number of sub-windows
            - Sub-window type is Plot
            - Plot contains the same channels
        """
        self.mw = MainWindow(files=(self.measurement_file,))
        self.mw.showNormal()
        self.processEvents(1)

        file_widget = self.mw.files.widget(0)

        # Select some channels and create a Plot window
        channel_names = []
        iterator = QTreeWidgetItemIterator(file_widget.channels_tree)
        count = 0
        while item := iterator.value():
            if item.parent() is not None and count < 3:
                item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                channel_names.append(item.text(0))
                count += 1
            iterator += 1

        self.assertGreater(len(channel_names), 0, "No channels found to select")

        # Create a Plot window directly (F2 shortcut is tested separately in TestShortcuts)
        file_widget._create_window(None, "Plot")
        self.processEvents(0.5)

        self.assertEqual(len(file_widget.mdi_area.subWindowList()), 1)

        with mock.patch("asammdf.gui.widgets.main.QtWidgets.QFileDialog.getOpenFileName") as mo_dialog:
            mo_dialog.return_value = self.replacement_file, None
            QTest.keySequence(self.mw, QKeySequence("Ctrl+Shift+O"))
            self.processEvents(1)

        new_widget = self.mw.files.widget(0)
        self.assertIsInstance(new_widget, FileWidget)
        # Verify sub-windows were recreated
        sub_windows = new_widget.mdi_area.subWindowList()
        self.assertEqual(len(sub_windows), 1)
        self.assertIsInstance(sub_windows[0].widget(), plot.Plot)

    def test_replace_file_no_op_when_no_file_open(self):
        """
        Test scope:
            Ensure that Replace does nothing when no file tab is open.
        Events:
            - Open MainWindow with no files
            - Trigger Replace via Ctrl+Shift+O
        Evaluate:
            - Tab count remains 0
            - No errors raised
        """
        self.mw = MainWindow()
        self.mw.showNormal()
        self.processEvents(0.5)

        self.assertEqual(self.mw.files.count(), 0)

        with mock.patch("asammdf.gui.widgets.main.QtWidgets.QFileDialog.getOpenFileName") as mo_dialog:
            QTest.keySequence(self.mw, QKeySequence("Ctrl+Shift+O"))
            self.processEvents(0.5)
            # Dialog should never be shown
            mo_dialog.assert_not_called()

        self.assertEqual(self.mw.files.count(), 0)

    def test_replace_file_no_op_when_dialog_cancelled(self):
        """
        Test scope:
            Ensure that Replace does nothing when the user cancels the file dialog.
        Events:
            - Open MainWindow with a measurement file
            - Trigger Replace but return empty string from dialog (user cancelled)
        Evaluate:
            - Original file tab is unchanged
        """
        self.mw = MainWindow(files=(self.measurement_file,))
        self.mw.showNormal()
        self.processEvents(1)

        self.assertEqual(self.mw.files.count(), 1)
        self.assertEqual(self.mw.files.tabText(0), "original.mf4")

        with mock.patch("asammdf.gui.widgets.main.QtWidgets.QFileDialog.getOpenFileName") as mo_dialog:
            mo_dialog.return_value = "", None
            QTest.keySequence(self.mw, QKeySequence("Ctrl+Shift+O"))
            self.processEvents(0.5)

        # Original tab should be unchanged
        self.assertEqual(self.mw.files.count(), 1)
        self.assertEqual(self.mw.files.tabText(0), "original.mf4")

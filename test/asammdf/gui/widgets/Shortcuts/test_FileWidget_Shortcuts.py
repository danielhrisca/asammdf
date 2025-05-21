#!/usr/bin/env python\

from math import ceil, sqrt
import pathlib
from random import randint
from unittest import mock

from PySide6.QtCore import QRect
from PySide6.QtGui import QKeySequence, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QTreeWidgetItemIterator

from test.asammdf.gui.test_base import OpenMDF, Pixmap
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget


class TestFileWidgetShortcuts(TestFileWidget):
    """
    Test for F11 shortcut was moved to Main Window tests
    """

    def setUp(self):
        """
        Events:
            Open measurement file "ASAP2_Demo_V171.mf4" as FileWidget

        Evaluate:
            - Evaluate that widget didn't have active sub-windows
        """
        super().setUp()
        # Open measurement file
        self.measurement_file = str(pathlib.Path(TestFileWidget.resource, "ASAP2_Demo_V171.mf4"))
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)

        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 0)

        # Get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.widget))

    def test_search_and_select_channels_shortcut(self):
        """
        Test Scope:
            Check if advanced search widget is called by shortcut Ctrl+F and founded items are added to plot.

        Events:
            - Press "Ctrl+F" -> search signals -> Add channels -> 'New plot window' -> Ok
            - Press "Ctrl+F" -> search pattern-based signals -> Apply -> 'New pattern based plot window' -> Ok
            - Press "Ctrl+F" -> search signals -> Add channels -> 'Existing pattern based window' -> Ok

        Evaluate:
            - Evaluate that widget isn't full-screen at start
            - Evaluate that widget was switched to full-screen mode after key F11 was pressed

        """
        # Setup
        matrix_items = {}
        u_word_items = {}
        matrix_pattern = "matrix"
        u_word_pattern = "uWord"
        # Search signals with specific patter
        with OpenMDF(self.measurement_file) as mdf:
            for ch in mdf.iter_channels():
                if matrix_pattern.upper() in ch.name.upper():
                    matrix_items[(ch.group_index, ch.channel_index)] = ch.name
                if u_word_pattern.upper() in ch.name.upper():
                    u_word_items[(ch.group_index, ch.channel_index)] = ch.name

        sw_count = 0  # Sub-windows
        # Mock for Advanced search and windowSelectionDialog objects
        with (
            mock.patch("asammdf.gui.widgets.file.AdvancedSearch") as mo_AdvancedSearch,
            mock.patch("asammdf.gui.widgets.file.WindowSelectionDialog") as mo_WindowSelectionDialog,
        ):
            with self.subTest("test_search_shortcut_new_window"):
                mo_AdvancedSearch.return_value.result = matrix_items
                mo_AdvancedSearch.return_value.pattern_window = False
                mo_WindowSelectionDialog.return_value.dialog.return_value = 1  # Ok
                mo_WindowSelectionDialog.return_value.selected_type.return_value = "New plot window"
                mo_WindowSelectionDialog.return_value.disable_new_channels.return_value = False

                # Press Ctrl+F
                QTest.keySequence(self.widget, QKeySequence(self.shortcuts["search"]))
                self.processEvents(0.01)

                mo_AdvancedSearch.assert_called()
                mo_WindowSelectionDialog.assert_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), sw_count + 1)

                channel_selection = self.widget.mdi_area.subWindowList()[sw_count].widget().channel_selection
                sw_count += 1

                # Evaluate plot widget
                self.assertEqual(len(matrix_items), channel_selection.topLevelItemCount())
                iterator = QTreeWidgetItemIterator(channel_selection)
                while item := iterator.value():
                    self.assertIn(item.name, matrix_items.values())
                    iterator += 1

            with self.subTest("test_search_shortcut_pattern_window"):
                # Setup
                mo_AdvancedSearch.return_value.pattern_window = True
                mo_AdvancedSearch.return_value.result = {
                    "case_sensitive": False,
                    "filter_type": "Unspecified",
                    "filter_value": 0.0,
                    "integer_format": "phys",
                    "match_type": "Wildcard",
                    "name": matrix_pattern.upper(),
                    "pattern": f"*{matrix_pattern}*",
                    "ranges": [],
                    "raw": False,
                }
                mo_WindowSelectionDialog.return_value.dialog.return_value = 1  # Ok
                mo_WindowSelectionDialog.return_value.selected_type.return_value = "New pattern based plot window"
                mo_WindowSelectionDialog.return_value.disable_new_channels.return_value = False

                # Event
                # Press Ctrl+F
                QTest.keySequence(self.widget, QKeySequence(self.shortcuts["search"]))
                self.processEvents(0.01)
                mo_AdvancedSearch.assert_called()
                mo_WindowSelectionDialog.assert_called()

                self.assertEqual(len(self.widget.mdi_area.subWindowList()), sw_count + 1)
                self.assertEqual(self.widget.mdi_area.subWindowList()[sw_count].windowTitle(), f"*{matrix_pattern}*")

                channel_selection = self.widget.mdi_area.subWindowList()[sw_count].widget().channel_selection
                # Evaluate plot widget
                self.assertEqual(len(matrix_items), channel_selection.topLevelItemCount())
                iterator = QTreeWidgetItemIterator(channel_selection)
                while item := iterator.value():
                    self.assertIn(item.name, matrix_items.values())
                    iterator += 1

                # New setup
                mo_WindowSelectionDialog.return_value.dialog.return_value = 1  # Ok
                mo_WindowSelectionDialog.return_value.selected_type.return_value = f"*{matrix_pattern}*"
                mo_WindowSelectionDialog.return_value.disable_new_channels.return_value = False
                mo_AdvancedSearch.return_value.result = u_word_items
                mo_AdvancedSearch.return_value.pattern_window = False

                # Press Ctrl+F
                QTest.keySequence(self.widget, QKeySequence(self.shortcuts["search"]))
                self.processEvents(0.01)

                mo_AdvancedSearch.assert_called()
                mo_WindowSelectionDialog.assert_called()
                self.assertEqual(len(self.widget.mdi_area.subWindowList()), sw_count + 1)

                final_items = {**matrix_items, **u_word_items}
                channel_selection = self.widget.mdi_area.subWindowList()[sw_count].widget().channel_selection
                # Evaluate plot widget
                self.assertEqual(len(final_items), channel_selection.topLevelItemCount())
                iterator = QTreeWidgetItemIterator(channel_selection)
                while item := iterator.value():
                    self.assertIn(item.name, final_items.values())
                    iterator += 1

    def test_cascade__grid__vertically__horizontally_sub_windows_shortcuts(self):
        """
        Test Scope:
            - To check if sub-windows layout can be properly changed to vertically, horizontally and grid one
                using Shift V, Shift H and Shift_T shortcuts.

        Events:
            - Add random number of sub-windows, between 3 and 30.
            - Press "Shift+V".
            - Press "Shift+H".
            - Press "Shift+T".

        Evaluate:
            - For vertical layout:
                > each sub-window starts from the new x coordinate;
                > each sub-window starts from the same y coordinate;
                > each sub-window has the same length and width.
            - For horizontal layout:
                > each sub-window starts from the same x coordinate;
                > each sub-window starts from the new y coordinate;
                > each sub-window has the same length and width.
            - For grid layout:
                > each sub-window starts from ceil value of square root number of sub-windows for x coordinate;
                > each sub-window starts from round value of square root number of sub-windows for y coordinate;
                > there are only two sizes for sub-windows widths, one of sizes is step for sub-window starts by x-axis;
                > there are only two or three sizes for sub-windows heights,
                    one of sizes is step for sub-window starts by y-axis;
            - For cascade layout:
                > each sub-window starts from the new x coordinate;
                > all sub-windows is grouped in columns in order to fit all sub-windows top bar in mdi area;
                > all sub-windows must have the same width;
                > there are only two or three sizes for sub-windows heights.
        """

        def g(list_: list):
            x_ = {widget.geometry().x() for widget in list_}
            y_ = {widget.geometry().y() for widget in list_}
            w_ = {widget.geometry().width() for widget in list_}
            h_ = {widget.geometry().height() for widget in list_}
            return x_, y_, w_, h_

        def get_step(set_: set):
            list_ = list(set_)
            list_.sort()
            return list_[1] - list_[0]

        # Setup
        max_square = 5
        sub_windows = randint(5, max_square * (1 + max_square))
        for _ in range(sub_windows):
            self.create_window(window_type="Plot")
        self.processEvents(0.01)

        # Press Shift+V -|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|-
        QTest.keySequence(self.widget, QKeySequence(self.shortcuts["tile_sub_windows_vertically"]))
        self.processEvents(0.1)

        x, y, width, height = g(self.widget.mdi_area.subWindowList())

        # Evaluate
        self.assertEqual(len(x), sub_windows)
        self.assertEqual(len(y), 1)
        self.assertEqual(len(width), 1)
        self.assertEqual(len(height), 1)

        # Press Shift+H -|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|-
        QTest.keySequence(self.widget, QKeySequence(self.shortcuts["tile_sub_windows_horizontally"]))

        x, y, width, height = g(self.widget.mdi_area.subWindowList())

        # Evaluate
        self.assertEqual(len(x), 1)
        self.assertEqual(len(y), sub_windows)
        self.assertEqual(len(width), 1)
        self.assertEqual(len(height), 1)

        # Press Shift+T -|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|-
        QTest.keySequence(self.widget, QKeySequence(self.shortcuts["tile_sub_windows_in_a_grid"]))
        self.processEvents()
        x, y, width, height = g(self.widget.mdi_area.subWindowList())

        # Evaluate
        self.assertEqual(len(x), ceil(sqrt(sub_windows)))
        self.assertEqual(len(y), round(sqrt(sub_windows)))
        self.assertEqual(len(width), 2)
        self.assertIn(get_step(x), width)
        self.assertIn(len(height), range(2, 5))
        self.assertIn(get_step(y), height)

        # Press Shift+C -|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|-
        QTest.keySequence(self.widget, QKeySequence(self.shortcuts["cascade_sub_windows"]))
        self.processEvents(0.01)

        x, y, width, height = g(self.widget.mdi_area.subWindowList())

        # Evaluate
        self.assertEqual(len(x), sub_windows)
        self.assertGreaterEqual(len(y), ceil(sub_windows / 6))
        self.assertEqual(len(width), 1)
        self.assertIn(len(height), range(1, 4))

    def test_toggle_sub_windows_frame_shortcut(self):
        """
        Test Scope:
            Check if sub-windows frame was toggled after pressing keys "Shift+Alt+F"

        Events:
            - Press twice "Shift+Alt+F"

        Evaluate:
            - Evaluate that by default sub-windows are not frameless
            - Evaluate that sub-windows is frameless, and widget size was reduced
                after pressing "Shift+Alt+F" for the first time
            - Evaluate that sub-windows is not frameless, and widget size was increased
                after pressing "Shift+Alt+F" second time
        """
        # Setup
        self.create_window(window_type="Plot")
        self.processEvents(0.01)

        # Evaluate
        self.assertFalse(self.widget._frameless_windows)

        # Setup
        previous_size = self.widget.mdi_area.subWindowList()[0].widget().frameSize()

        # Press Shift+Alt+F first time
        QTest.keySequence(self.widget, QKeySequence(self.shortcuts["toggle_sub_windows_frames"]))

        # Evaluate
        self.assertTrue(self.widget._frameless_windows)
        self.assertLess(previous_size.width(), self.widget.mdi_area.subWindowList()[0].widget().frameSize().width())
        self.assertLess(previous_size.height(), self.widget.mdi_area.subWindowList()[0].widget().frameSize().height())

        # Setup
        previous_size = self.widget.mdi_area.subWindowList()[0].widget().frameSize()

        # Press Shift+Alt+F second time
        QTest.keySequence(self.widget, QKeySequence(self.shortcuts["toggle_sub_windows_frames"]))

        # Evaluate
        self.assertFalse(self.widget._frameless_windows)
        self.assertGreater(previous_size.width(), self.widget.mdi_area.subWindowList()[0].widget().frameSize().width())
        self.assertGreater(
            previous_size.height(), self.widget.mdi_area.subWindowList()[0].widget().frameSize().height()
        )

    def test_toggle_channel_list_shortcut(self):
        """
        Test Scope:
            Check if by pressing the combination of keys "Shift+L", visibility of a channel list is changed

        Events:
            - Press twice combination "Shift+L".

        Evaluate:
            - Evaluate that channel list is visible by default, and its width is greater than 0.
            - Evaluate that channel list is hidden and its width is equal to 0 after pressing first time .Shift+L.
            - Evaluate that channel list is visible and its width is greater than 0 after pressing second time Shift+L.
        """
        # Setup
        self.processEvents()

        # Evaluate
        self.assertTrue(self.widget.channel_view.isVisible())
        self.assertGreater(self.widget.splitter.sizes()[0], 0)

        # Press "Shift+L" first time
        QTest.keySequence(self.widget, QKeySequence(self.shortcuts["toggle_channel_view"]))
        self.processEvents()

        # Evaluate
        self.assertFalse(self.widget.channel_view.isVisible())
        self.assertEqual(self.widget.splitter.sizes()[0], 0)

        # Press "Shift+L" second time
        QTest.keySequence(self.widget, QKeySequence(self.shortcuts["toggle_channel_view"]))
        self.processEvents()

        # Evaluate
        self.assertTrue(self.widget.channel_view.isVisible())
        self.assertGreater(self.widget.splitter.sizes()[0], 0)

    def test_toggle_dots_shortcut(self):
        """
        Test Scope:
            Check if dots appear on plot after pressing key Period

        Events:
            - Press key "Period" <<.>> twice

        Evaluate:
            - Evaluate number of background colors, it must be greater when signal is without dots
        """
        # Setup
        self.create_window("Plot")
        self.add_channels(channels_list=[35])
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        # Remove bookmarks if plot is with bookmarks
        if not plot.bookmark_btn.isFlat():
            QTest.mouseClick(plot.bookmark_btn, Qt.MouseButton.LeftButton)
        if not plot.plot.with_dots:
            plot.plot.set_dots(True)
        # Remove grid if plot is with grid
        if not plot.hide_axes_btn.isFlat():
            QTest.mouseClick(plot.hide_axes_btn, Qt.MouseButton.LeftButton)
        self.processEvents(0.1)

        # Displayed signal
        sig = plot.plot.signals[0]
        # Coordinates for viewbox, second dot in the center of visual box
        x_step = (sig.timestamps[1] - sig.timestamps[0]) / 10
        y_step = (sig.samples[1] - sig.samples[0]) / 10
        x, y = sig.timestamps[1] - x_step, sig.samples[1] - y_step
        w, h = sig.timestamps[1] + x_step, sig.samples[1] + y_step
        # Set X and Y ranges for viewbox
        plot.plot.viewbox.setXRange(x, w, padding=0)
        plot.plot.viewbox.setYRange(y, h, padding=0)
        self.processEvents(0.1)
        # Get rect from the center of graphical plot
        rect = QRect(int(plot.plot.width() / 2) - 4, int(plot.plot.height() / 2) - 4, 8, 9)

        # Color map of selected rect
        pm_with_dots = Pixmap.color_map(plot.plot.grab(rect))
        # Black coverage
        with_dots_black_coverage = 0
        for cov in pm_with_dots.values():
            with_dots_black_coverage += cov.count(Pixmap.COLOR_BACKGROUND)

        # Event
        QTest.keySequence(self.widget, QKeySequence(self.shortcuts["set_line_style"]))
        self.processEvents(0.1)
        # Gat new pixmap
        pm_without_dots = Pixmap.color_map(plot.plot.grab(rect))
        # New black coverage
        without_dots_black_coverage = 0
        for cov in pm_without_dots.values():
            without_dots_black_coverage += cov.count(Pixmap.COLOR_BACKGROUND)

        # Evaluate
        self.assertGreater(without_dots_black_coverage, with_dots_black_coverage)

        # Event
        QTest.keySequence(self.widget, QKeySequence(self.shortcuts["set_line_style"]))
        self.processEvents(0.1)
        # Gat new pixmap
        pm_with_dots = Pixmap.color_map(plot.plot.grab(rect))
        # New black coverage
        new_with_dots_black_coverage = 0
        for cov in pm_with_dots.values():
            new_with_dots_black_coverage += cov.count(Pixmap.COLOR_BACKGROUND)

        # Evaluate
        self.assertEqual(new_with_dots_black_coverage, with_dots_black_coverage)

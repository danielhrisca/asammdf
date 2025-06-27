#!/usr/bin/env python\

import os
from unittest import mock

from PySide6.QtCore import QPoint, QRect, QSettings, Qt
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QTreeWidgetItemIterator

from asammdf import mdf
from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget


class TestPlotGraphicsShortcuts(TestPlotWidget):
    def setUp(self):
        """
        Events:
            Open test measurement file
            Set sort method: Natural sort
            Create new plot widget with following setup:
                - Without dots
                - Without grid
                - With hidden bookmarks
                - Cursor without circle
        Evaluate:
            - Evaluate that one widget was created
            - Evaluate that plot is black
        """
        super().setUp()
        settings = QSettings()
        settings.setValue("zoom_x_center_on_cursor", True)
        settings.setValue("plot_cursor_precision", 6)
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        # Evaluate
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        # Plot object
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Preset for plot
        # Remove dots
        if self.plot.plot.with_dots:
            self.plot.plot.set_dots(False)
        # check if grid is available -> hide grid
        if not self.plot.hide_axes_btn.isFlat():
            QTest.mouseClick(self.plot.hide_axes_btn, Qt.MouseButton.LeftButton)
        # hide bookmarks if it's available
        if self.plot.show_bookmarks:
            self.plot.toggle_bookmarks(hide=True)
        # Ensure that plot is not` in Focused mode
        if self.plot.focused_mode:
            self.plot.toggle_focused_mode()
        # Ensure that plot cursor is not with circle
        if self.plot.plot.cursor1.show_circle:
            self.plot.plot.cursor1.show_circle = False

        self.pg = self.plot.plot  # PlotGraphics object

        self.processEvents(0.01)
        # Evaluate that plot is black
        self.assertTrue(Pixmap.is_black(self.pg.grab()))

        # get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.pg))
        self.processEvents()

    def test_lock_unlock_range_shortcut(self):
        """
        Test Scope:
            Check if Range Selection cursor is locked/unlocked after pressing key Y.
        Events:
            - Set default cursor color, without horizontal line and circle, with 1px line width
            - Press Key Y for range selection
            - Move Cursors
            - Press Key R for range selection
        Evaluate:
            - Evaluate that at start is only one cursor available
            - Evaluate that two cursors are available after key Y was pressed
            - Evaluate that new rectangle with different color is present
            - Evaluate that sum of rectangle areas is same with the one when plot is full black.
            - Evaluate that range selection disappear.
        """
        # Setup for cursor
        self.widget.set_cursor_options(False, False, 1, Pixmap.COLOR_CURSOR)

        # Save PixMap of clean plot
        clean_pixmap = self.pg.grab()
        # Evaluate that plot is black
        self.assertTrue(Pixmap.is_black(clean_pixmap))

        # Get position of Cursor
        cursors = Pixmap.cursors_x(clean_pixmap)
        # Evaluate that there is only one cursor
        self.assertEqual(1, len(cursors))

        # Press Key 'Y' for range selection
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["toggle_range"]))
        self.processEvents(timeout=0.1)

        # Save PixMap of Range plot
        range_pixmap = self.pg.grab()
        self.assertFalse(Pixmap.is_black(range_pixmap))

        # Get X position of Cursors
        cursors = Pixmap.cursors_x(range_pixmap)
        # Evaluate that two cursors are available
        self.assertEqual(2, len(cursors))

        # Evaluate that new rectangle with different color is present
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_BACKGROUND,
                x=0,
                y=0,
                width=min(cursors) - 1,
            )
        )
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_RANGE,
                x=min(cursors) + 1,
                y=0,
                width=max(cursors),
            )
        )
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_BACKGROUND,
                x=max(cursors) + 1,
                y=0,
            )
        )

        # Move Cursors
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["move_cursor_left_1x"]))
        self.processEvents(timeout=0.01)
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["move_cursor_left_20x"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.pg.grab()
        self.assertFalse(Pixmap.is_black(range_pixmap))

        # Get X position of Cursors
        new_cursors = Pixmap.cursors_x(range_pixmap)
        # Evaluate that two cursors are available
        self.assertEqual(2, len(new_cursors))
        self.assertEqual(cursors[0], new_cursors[0], "First cursor have new position after manipulation")
        self.assertNotEqual(cursors[1], new_cursors[1], "Second cursors have same position after manipulation")

        # Evaluate that new rectangle with different color is present
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_BACKGROUND,
                x=0,
                y=0,
                width=min(new_cursors) - 1,
            )
        )
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_RANGE,
                x=min(new_cursors) + 1,
                y=0,
                width=max(new_cursors),
            )
        )
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_BACKGROUND,
                x=max(new_cursors) + 1,
                y=0,
            )
        )

        cursors = new_cursors
        # Press Key 'R' for range selection
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["toggle_range"]))
        self.processEvents(timeout=0.01)

        # Move Cursors
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["move_cursor_right_1x"]))
        self.processEvents(timeout=0.01)
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["move_cursor_left_20x"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.pg.grab()
        self.assertFalse(Pixmap.is_black(range_pixmap))

        # Get X position of Cursors
        new_cursors = Pixmap.cursors_x(range_pixmap)
        # Evaluate that two cursors are available
        self.assertEqual(2, len(new_cursors))
        for c in cursors:
            self.assertNotIn(c, new_cursors, f"cursor {c} is the same")

        # Evaluate that new rectangle with different color is present
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_BACKGROUND,
                x=0,
                y=0,
                width=min(new_cursors) - 1,
            )
        )
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_RANGE,
                x=min(new_cursors) + 1,
                y=0,
                width=max(new_cursors),
            )
        )
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_BACKGROUND,
                x=max(new_cursors) + 1,
                y=0,
            )
        )

        # Press Key 'R' for range selection
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["range"]))
        self.processEvents(timeout=0.01)
        # Save PixMap of clear plot
        clean_pixmap = self.pg.grab()
        self.assertTrue(Pixmap.is_black(clean_pixmap))

    def test_zoom_to_range_shortcut(self):
        """
        Test Scope:
            Check if fitting between cursors is released after pressing key "X".
        Events:
            - Display 1 signal on plot
            - Maximize window
            - Mouse click in the middle of plot
            - Press "R"
            - Press "X"
        Evaluate:
            - Evaluate that intersection of signal and midd line is exactly how much it intersects between cursors
            - Evaluate X range, it must be almost equal as range value after pressing key R
        """
        # Setup
        self.add_channels([35])
        channel_color = self.channels[0].color.name()

        # Count intersections between middle line and signal
        initial_intersections = Pixmap.color_map(self.pg.grab(QRect(0, int(self.pg.height() / 2), self.pg.width(), 1)))[
            0
        ].count(channel_color)
        self.assertTrue(initial_intersections)

        # Setup for cursor
        self.widget.set_cursor_options(False, False, 1, Pixmap.COLOR_CURSOR)
        # Mouse click on a center of plot
        QTest.mouseClick(
            self.pg.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self.pg.rect().center(),
        )
        # Press R
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["range"]))
        self.processEvents()

        x_range = self.pg.region.getRegion()
        self.assertNotIn(x_range[0], self.pg.x_range)
        self.assertNotIn(x_range[1], self.pg.x_range)
        # Get X position of Cursor
        cursors = Pixmap.cursors_x(self.pg.grab())
        # Ensure that both cursors were found
        self.assertEqual(len(cursors), 2)

        # Get a set of colors founded between cursors
        colors = Pixmap.color_names_exclude_defaults(
            self.pg.grab(QRect(cursors[0], 0, cursors[1] - cursors[0], self.pg.height()))
        )
        # Exclude channel original color
        if channel_color in colors:
            colors.remove(channel_color)
        # caught ya
        color = colors.pop()
        # Evaluate if color was found, in set must remain only the new channel color situated between cursors
        self.assertTrue(color)

        # Count intersection of midd line and signal between cursors
        expected_intersections = Pixmap.color_map(
            self.pg.grab(QRect(cursors[0], int(self.pg.height() / 2), cursors[1] - cursors[0], 1))
        )[0].count(color)
        self.assertTrue(expected_intersections)

        # Press key "X"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["zoom_to_range"]))
        self.processEvents()

        # Evaluate how much times signal intersect midd line
        actual_intersections = Pixmap.color_map(self.pg.grab(QRect(0, int(self.pg.height() / 2), self.pg.width(), 1)))[
            0
        ].count(channel_color)
        self.assertEqual(actual_intersections, expected_intersections)
        self.assertLess(actual_intersections, initial_intersections)

        # Evaluate ranges of signal
        self.assertAlmostEqual(self.pg.x_range[0], x_range[0], delta=0.001)
        self.assertAlmostEqual(self.pg.x_range[1], x_range[1], delta=0.001)

    def test_fit__stack_shortcuts(self):
        """
        Test Scope:
            Check if:
              > all signals is stack after pressing key "S"
              > only selected signal is fitted after pressing combination "Sift + F"
              > only selected signal is stacked after pressing combination "Shift + S"
              > all signals is fitted after pressing key "F"

        Events:
            - Display 3 channels to plot
            - Press Key "S"
            - Press combination "Shift + F"
            - Press combination "Shift + S"
            - press key "F"

        Evaluate:
            - Evaluate that signals are separated in top, midd and bottom third of plot after pressing key "S"
            - Evaluate that only selected signal is fitted after pressing combination "Shift + F"
            - Evaluate that only selected signal is stacked after pressing combination "Shift + S"
            - Evaluate that all signals are fitted after pressing key "F"

        Additional Evaluation
            - Evaluate that all signals are continuous on plot
        """

        def continuous(ch):
            pixmap = self.pg.grab()
            image = pixmap.toImage()
            signal_color = ch.color.name()
            start, stop = Pixmap.search_signal_extremes_by_ax(pixmap, signal_color=signal_color, ax="x")
            for x in range(start, stop + 1):
                for j in range(self.pg.height()):
                    if image.pixelColor(x, j).name() == signal_color:
                        break

                else:
                    raise Exception(f"column {x} doesn't have color of channel {ch.name} from {start=} to {stop=}")

        self.pg.cursor1.color = "#000000"
        settings = QSettings()
        settings.setValue("zoom_x_center_on_cursor", True)

        self.add_channels([35, 36, 37])

        channel_35 = self.channels[0]
        channel_36 = self.channels[1]
        channel_37 = self.channels[2]
        color_35 = channel_35.color.name()
        color_36 = channel_36.color.name()
        color_37 = channel_37.color.name()

        # Press "S"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["stack_all"]))
        self.processEvents()
        # Evaluate
        with self.subTest("test_stack_all_shortcut"):
            # First 2 lines
            self.assertTrue(Pixmap.is_black(self.pg.grab(QRect(0, 0, self.pg.width(), 1))))
            # Top
            pixmap = self.pg.grab(QRect(0, 0, self.pg.width(), int(self.pg.height() / 3)))
            self.assertTrue(Pixmap.has_color(pixmap, color_35))
            self.assertFalse(Pixmap.has_color(pixmap, color_36))
            self.assertFalse(Pixmap.has_color(pixmap, color_37))
            # Midd
            pixmap = self.pg.grab(
                QRect(
                    0,
                    int(self.pg.height() / 3),
                    self.pg.width(),
                    int(self.pg.height() / 3),
                )
            )
            self.assertFalse(Pixmap.has_color(pixmap, color_35))
            self.assertTrue(Pixmap.has_color(pixmap, color_36))
            self.assertFalse(Pixmap.has_color(pixmap, color_37))
            # Bottom
            pixmap = self.pg.grab(
                QRect(
                    0,
                    int(self.pg.height() / 3) * 2,
                    self.pg.width(),
                    int(self.pg.height() / 3),
                )
            )
            self.assertFalse(Pixmap.has_color(pixmap, color_35))
            self.assertFalse(Pixmap.has_color(pixmap, color_36))
            self.assertTrue(Pixmap.has_color(pixmap, color_37))
            # Last 2 lines

            pixmap = self.pg.grab(QRect(0, self.pg.height() - 3, self.pg.width(), 2))
            self.assertTrue(Pixmap.is_black(pixmap))

        # select the first channel
        self.mouseClick_WidgetItem(channel_35)
        # Press "Shift+F"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["fit_selected"]))
        self.is_not_blinking(self.pg, {color_35, color_36, color_37})

        # Evaluate
        with self.subTest("test_fit_selected_shortcut"):
            # First line
            self.assertTrue(Pixmap.is_black(self.pg.grab(QRect(0, 0, self.pg.width(), 1))))
            # Top
            pixmap = self.pg.grab(QRect(0, 0, self.pg.width(), int(self.pg.height() / 3)))
            self.assertTrue(Pixmap.has_color(pixmap, color_35))
            self.assertFalse(Pixmap.has_color(pixmap, color_36))
            self.assertFalse(Pixmap.has_color(pixmap, color_37))
            # Midd
            pixmap = self.pg.grab(
                QRect(
                    0,
                    int(self.pg.height() / 3),
                    self.pg.width(),
                    int(self.pg.height() / 3),
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, color_35))
            self.assertTrue(Pixmap.has_color(pixmap, color_36))
            self.assertFalse(Pixmap.has_color(pixmap, color_37))
            # Bottom
            pixmap = self.pg.grab(
                QRect(
                    0,
                    int(self.pg.height() / 3) * 2,
                    self.pg.width(),
                    int(self.pg.height() / 3),
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, color_35))
            self.assertFalse(Pixmap.has_color(pixmap, color_36))
            self.assertTrue(Pixmap.has_color(pixmap, color_37))
            # Last line
            self.assertTrue(Pixmap.is_black(self.pg.grab(QRect(0, self.pg.height() - 2, self.pg.width(), 1))))

        # select second channel
        self.mouseClick_WidgetItem(channel_36)
        # Press "Shift+F"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["stack_selected"]))
        self.is_not_blinking(self.pg, {color_35, color_36, color_37})
        # Evaluate
        with self.subTest("test_stack_selected_shortcut"):
            # First line
            self.assertTrue(Pixmap.is_black(self.pg.grab(QRect(0, 0, self.pg.width(), 1))))
            # Top
            pixmap = self.pg.grab(QRect(0, 0, self.pg.width(), int(self.pg.height() / 3)))
            self.assertTrue(Pixmap.has_color(pixmap, color_35))
            self.assertTrue(Pixmap.has_color(pixmap, color_36))
            self.assertFalse(Pixmap.has_color(pixmap, color_37))
            # Midd
            pixmap = self.pg.grab(
                QRect(
                    0,
                    int(self.pg.height() / 3),
                    self.pg.width(),
                    int(self.pg.height() / 3),
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, color_35))
            self.assertTrue(Pixmap.has_color(pixmap, color_36))
            self.assertFalse(Pixmap.has_color(pixmap, color_37))
            # Bottom
            pixmap = self.pg.grab(
                QRect(
                    0,
                    int(self.pg.height() / 3) * 2,
                    self.pg.width(),
                    int(self.pg.height() / 3),
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, color_35))
            self.assertTrue(Pixmap.has_color(pixmap, color_36))
            self.assertTrue(Pixmap.has_color(pixmap, color_37))
            # Last line
            self.assertTrue(Pixmap.is_black(self.pg.grab(QRect(0, self.pg.height() - 1, self.pg.width(), 1))))

            # Press "F"
            QTest.keySequence(self.pg, QKeySequence(self.shortcuts["fit_all"]))
            self.is_not_blinking(self.pg, {color_35, color_36, color_37})
            # Evaluate
        with self.subTest("test_fit_all_shortcut"):
            # First line
            self.assertTrue(Pixmap.is_black(self.pg.grab(QRect(0, 0, self.pg.width(), 1))))
            # Top
            pixmap = self.pg.grab(QRect(0, 0, self.pg.width(), int(self.pg.height() / 3)))
            self.assertTrue(Pixmap.has_color(pixmap, color_35))
            self.assertTrue(Pixmap.has_color(pixmap, color_36))
            self.assertTrue(Pixmap.has_color(pixmap, color_37))
            # Midd
            pixmap = self.pg.grab(
                QRect(
                    0,
                    int(self.pg.height() / 3),
                    self.pg.width(),
                    int(self.pg.height() / 3),
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, color_35))
            self.assertTrue(Pixmap.has_color(pixmap, color_36))
            self.assertTrue(Pixmap.has_color(pixmap, color_37))
            # Bottom
            pixmap = self.pg.grab(
                QRect(
                    0,
                    int(self.pg.height() / 3) * 2,
                    self.pg.width(),
                    int(self.pg.height() / 3),
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, color_35))
            self.assertTrue(Pixmap.has_color(pixmap, color_36))
            self.assertTrue(Pixmap.has_color(pixmap, color_37))
            # Last line
            self.assertTrue(Pixmap.is_black(self.pg.grab(QRect(0, self.pg.height() - 1, self.pg.width(), 1))))

            # deselect all channels
            for channel in self.channels:
                self.mouseDClick_WidgetItem(channel)

            # search if all channels are fitted into extremes
            self.mouseDClick_WidgetItem(channel_35)
            continuous(channel_35)

            self.mouseDClick_WidgetItem(channel_35)
            self.mouseDClick_WidgetItem(channel_36)
            continuous(channel_36)

            self.mouseDClick_WidgetItem(channel_36)
            self.mouseDClick_WidgetItem(channel_37)
            continuous(channel_37)

    def test_grid_shortcut(self):
        """
        Test Scope:
            Check if grid is created properly after pressing key "G".

        Events:
            - If axes are hidden - press "Show axes" button
            - Press Key "G" 20 times

        Evaluate:
            - Evaluate that grid is displayed in order after pressing key "G":
                1. Is only X axis grid.
                2. Is X and Y axes grids.
                3. Is only Y axis grid
                4. There is no grid.
        """
        # Check if the grid is available

        if self.plot.hide_axes_btn.isFlat():
            QTest.mouseClick(self.plot.hide_axes_btn, Qt.MouseButton.LeftButton)

        next_grid = {
            (False, False): (True, False),
            (True, False): (True, True),
            (True, True): (False, True),
            (False, True): (False, False),
        }

        current_grid = self.pg.x_axis.grid, self.pg.y_axis.grid
        for i in range(20):
            # press key "G"
            QTest.keySequence(self.pg, QKeySequence(self.shortcuts["grid"]))
            self.processEvents()
            # Evaluate
            current_grid = next_grid[current_grid]
            self.assertEqual(current_grid, (self.pg.x_axis.grid, self.pg.y_axis.grid))

    def test_go_to_timestamp_shortcut(self):
        """
        Test scope:
            Ensure that Shift+G will switchto selected timestamp

        Events:
            - Display one item to the widget.
            - Press Shift+G -> input some value -> "Ok".

        Evaluate:
             - Evaluate that there is added one item to the widget.
             - Evaluate that getDouble object was called.
             - Evaluate that timestamp value is almost equal with the inputted value.
        Returns
        -------

        """
        # Setup
        # Add one item to the widget
        self.assertIsNotNone(self.add_channels([10]))

        # Event
        with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QInputDialog.getDouble") as mo_getDouble:
            expected_pos = 3.0
            mo_getDouble.return_value = expected_pos, True
            # Press Shift+G
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["go_to_timestamp"]))
            self.processEvents(0.1)
        self.processEvents(0.01)
        # Evaluate
        mo_getDouble.assert_called()
        ci = self.plot.cursor_info
        pos = float(ci.text().split()[2].split(ci.unit)[0])

        self.assertAlmostEqual(expected_pos, pos, delta=0.01)

    def test_zoom_in__out_shortcuts(self):
        """
        Test Scope:
            Check if zooming is released after pressing keys "I", "Shift+I", "O", "Shift+O".

        Events:
            - Display 1 signal on plot
            - Select signal, click in the middle of plot
            - Press "I".
            - Press "O".
            - Press "Shift+I".
            - Press "Shift+O".

        Evaluate:
            - Evaluate ranges of the "X" and the "Y" asis.
        """

        def get_expected_result(step, is_x_axis: bool):
            if is_x_axis:
                delta = self.pg.x_range[1] - self.pg.x_range[0]
                val = self.pg.cursor1.value()
                step = delta * step
                return val - delta / 2 - step, val + delta / 2 + step
            else:
                val, bottom, top = self.pg.value_at_cursor()
                delta = (top - bottom) * step
                return val - delta / 2, val + delta / 2

        # Setup
        if self.plot.lock_btn.isFlat():
            QTest.mouseClick(self.plot.lock_btn, Qt.MouseButton.LeftButton)

        y_step = 0.165
        x_step = 0.25

        self.assertIsNotNone(self.add_channels([35]))
        self.mouseClick_WidgetItem(self.channels[0])
        self.processEvents()

        self.pg.viewbox.menu.set_x_zoom_mode()
        self.pg.viewbox.menu.set_y_zoom_mode()

        # click con center
        QTest.mouseClick(
            self.plot.plot.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(int(self.pg.width() / 2), int(self.pg.height() / 2)),
        )
        self.processEvents()

        # Events Without Pressed Shift
        expected_x_zoom_in_range = get_expected_result(-x_step, True)
        # Press "I"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["x_zoom_in"]))
        self.processEvents()
        x_zoom_in_range = self.pg.x_range

        expected_x_zoom_out_range = get_expected_result(x_step * 2, True)
        # Press "O"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["x_zoom_out"]))
        self.processEvents()
        x_zoom_out_range = self.pg.x_range

        # Events with pressed Shift
        expected_y_zoom_in_range = get_expected_result(1 / (1 + y_step), False)
        # Press "I"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["y_zoom_in"]))
        self.processEvents()
        y_zoom_in_range = self.pg.signals[0].y_range

        expected_y_zoom_out_range = get_expected_result(1 + y_step, False)
        # Press "O"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["y_zoom_out"]))
        self.processEvents()

        y_zoom_out_range = self.pg.signals[0].y_range
        # Evaluate
        delta = pow(10, -4)
        # Key Shift wasn't pressed
        self.assertAlmostEqual(x_zoom_in_range[0], expected_x_zoom_in_range[0], delta=delta)
        self.assertAlmostEqual(x_zoom_in_range[1], expected_x_zoom_in_range[1], delta=delta)
        self.assertAlmostEqual(x_zoom_out_range[0], expected_x_zoom_out_range[0], delta=delta)
        self.assertAlmostEqual(x_zoom_out_range[1], expected_x_zoom_out_range[1], delta=delta)
        # Key Shift was pressed
        self.assertAlmostEqual(y_zoom_in_range[0], expected_y_zoom_in_range[0], delta=delta)
        self.assertAlmostEqual(y_zoom_in_range[1], expected_y_zoom_in_range[1], delta=delta)
        self.assertAlmostEqual(y_zoom_out_range[0], expected_y_zoom_out_range[0], delta=delta)
        self.assertAlmostEqual(y_zoom_out_range[1], expected_y_zoom_out_range[1], delta=delta)

    def test_range_shortcut(self):
        """
        Test Scope:
            Check if Range Selection rectangle is painted over the plot.
        Events:
            - Press Key R for range selection
            - Move Cursors
            - Press Key R for range selection
        Evaluate:
            - Evaluate that two cursors are available
            - Evaluate that new rectangle with different color is present
            - Evaluate that sum of rectangle areas is same with rectangle of full black plot.
            - Evaluate that range selection disappears after pressing key R second time.
        """
        self.widget.set_cursor_options(False, False, 1, Pixmap.COLOR_CURSOR)
        # Save PixMap of clear plot
        clear_pixmap = self.pg.grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

        # Get X position of Cursor
        cursors = Pixmap.cursors_x(clear_pixmap)
        # Evaluate that there is only one cursor
        self.assertEqual(1, len(cursors))

        # Press Key 'R' for range selection
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["range"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.pg.grab()
        self.assertFalse(Pixmap.is_black(range_pixmap))

        # Get X position of Cursors
        cursors = Pixmap.cursors_x(range_pixmap)
        # Evaluate that two cursors are available
        self.assertEqual(2, len(cursors))

        # Evaluate that new rectangle with different color is present
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_BACKGROUND,
                x=0,
                y=0,
                width=min(cursors) - 1,
            )
        )
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_RANGE,
                x=min(cursors) + 1,
                y=0,
                width=max(cursors),
            )
        )
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_BACKGROUND,
                x=max(cursors) + 1,
                y=0,
            )
        )

        # Move Cursors
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["move_cursor_right_1x"]))
        self.processEvents(timeout=0.01)
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["move_cursor_left_20x"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.pg.grab()
        self.assertFalse(Pixmap.is_black(range_pixmap))

        # Get X position of Cursors
        new_cursors = Pixmap.cursors_x(range_pixmap)
        # Evaluate that two cursors are available
        self.assertEqual(2, len(cursors))
        for c in cursors:
            self.assertNotIn(c, new_cursors, f"cursor {c} is the same")

        # Evaluate that new rectangle with different color is present
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_BACKGROUND,
                x=0,
                y=0,
                width=min(new_cursors) - 1,
            )
        )
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_RANGE,
                x=min(new_cursors) + 1,
                y=0,
                width=max(new_cursors),
            )
        )
        self.assertTrue(
            Pixmap.is_colored(
                pixmap=range_pixmap,
                color_name=Pixmap.COLOR_BACKGROUND,
                x=max(new_cursors) + 1,
                y=0,
            )
        )
        # Press Key 'R' for range selection
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["range"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of clear plot
        clear_pixmap = self.pg.grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

    def test_save_active_subplot_channels_shortcut(self):
        """
        Test Scope:
            Check if by pressing "Ctrl+S" is saved in the new measurement file only active channels.

        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a new plot
            - Deselect first channel from first plot
            - Press Key "Ctrl+S"
            - Open recently created measurement file in a new window.

        Evaluate:
            - Evaluate that object getSaveFileName() was called after pressing combination "Ctrl+S"
            - Evaluate that in measurement file is saved only active channels
        """
        # Setup
        file_path = os.path.join(self.test_workspace, "test_file.mf4")
        self.create_window(window_type="Plot", channels_indexes=(20, 21, 22))
        self.processEvents()
        second_plot_items = []
        iterator = QTreeWidgetItemIterator(self.widget.mdi_area.subWindowList()[1].widget().channel_selection)
        while item := iterator.value():
            second_plot_items.append(item.name)
            iterator += 1
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]))
        expected_channels = [channel.name for channel in self.channels]
        expected_channels.append("time")
        self.processEvents()

        # select all channels excluding the first one
        for _ in range(1, len(self.channels)):
            self.channels[_].setSelected(True)
        expected_channels.remove(self.channels[0].name)
        self.processEvents()

        with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = (file_path, "")
            # Press Ctrl+S
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["save_active_subplot_channels"]))
        # Evaluate
        mo_getSaveFileName.assert_called()

        # get waved file as MDF
        process_bus_logging = ("process_bus_logging", True)
        mdf_file = mdf.MDF(file_path, process_bus_logging=process_bus_logging)

        # Evaluate
        for name in expected_channels:
            self.assertIn(name, mdf_file.channels_db.keys())
        for name in second_plot_items:
            self.assertNotIn(name, mdf_file.channels_db.keys())
        mdf_file.close()

    def test_move_cursor_left__right_shortcuts(self):
        """
        Test Scope:
            Check that Arrow Keys: Left & Right ensure navigation on channels evolution.
            Ensure that navigation is working.
        Events:
            - Display one signal to plot
            - Click Right key specific number of times.
            - Click Left key specific number of times.
            - Click Ctrl+Right key specific number of times.
            - Click Ctrl+Left key specific number of times.
        Evaluate:
            - Evaluate values from selected channel value and cursor info, it must be equal to the expected values.
        """
        # Setup
        if self.plot.selected_channel_value_btn.isFlat():
            QTest.mouseClick(self.plot.selected_channel_value_btn, Qt.MouseButton.LeftButton)
        self.add_channels([37])
        ch = self.channels[0]
        # Number of times that specific key will be pressed
        right_clicks = 50
        ctrl_right_clicks = 20
        left_clicks = 30
        ctrl_left_clicks = 15
        ci = self.plot.cursor_info
        # Select channel
        self.mouseClick_WidgetItem(ch)
        self.pg.setFocus()
        self.processEvents(0.1)
        pos = 0
        cursor_prev = Pixmap.cursors_x(self.pg.grab())[0]
        c1_pos = self.pg.cursor1.getXPos()

        # Evaluate
        self.assertEqual(f"{ch.signal.samples[pos]} {ch.unit}", self.plot.selected_channel_value.text())
        self.assertEqual(f"{ci.name} = {round(ch.signal.timestamps[pos], ci.precision)}{ci.unit}", ci.text())
        self.assertEqual(ch.signal.timestamps[pos], self.pg.cursor1.getXPos())

        # Event
        for _ in range(right_clicks):
            QTest.keySequence(self.pg, QKeySequence(self.shortcuts["move_cursor_right_1x"]))
            self.processEvents(0.01)
        self.processEvents(0.5)
        pos += right_clicks
        cursor_now = Pixmap.cursors_x(self.pg.grab())[0]

        # Evaluate
        self.assertGreater(cursor_now, cursor_prev)
        self.assertEqual(f"{ch.signal.samples[pos]} {ch.unit}", self.plot.selected_channel_value.text())
        self.assertEqual(f"{ci.name} = {round(ch.signal.timestamps[pos], ci.precision)}{ci.unit}", ci.text())
        self.assertEqual(ch.signal.timestamps[pos], self.pg.cursor1.getXPos())

        # New setup
        pos -= left_clicks
        # Send Key strokes
        for _ in range(left_clicks):
            QTest.keySequence(self.pg, QKeySequence(self.shortcuts["move_cursor_left_1x"]))
            self.processEvents(0.01)
        self.processEvents(0.5)
        cursor_prev = cursor_now
        cursor_now = Pixmap.cursors_x(self.pg.grab())[0]

        # Evaluate
        self.assertLess(cursor_now, cursor_prev)
        self.assertEqual(f"{ch.signal.samples[pos]} {ch.unit}", self.plot.selected_channel_value.text())
        self.assertEqual(f"{ci.name} = {round(ch.signal.timestamps[pos], ci.precision)}{ci.unit}", ci.text())
        self.assertEqual(ch.signal.timestamps[pos], self.pg.cursor1.getXPos())

        # Send Key strokes
        for _ in range(ctrl_right_clicks):
            QTest.keySequence(self.pg, QKeySequence(self.shortcuts["move_cursor_right_20x"]))
            self.processEvents(0.01)
        self.processEvents(0.1)
        pos += 20 * ctrl_right_clicks
        cursor_prev = cursor_now
        cursor_now = Pixmap.cursors_x(self.pg.grab())[0]

        # Evaluate
        self.assertGreater(cursor_now, cursor_prev)
        self.assertEqual(f"{ch.signal.samples[pos]} {ch.unit}", self.plot.selected_channel_value.text())
        self.assertEqual(f"{ci.name} = {round(ch.signal.timestamps[pos], ci.precision)}{ci.unit}", ci.text())
        self.assertEqual(ch.signal.timestamps[pos], self.pg.cursor1.getXPos())

        # Send Key strokes
        for _ in range(ctrl_left_clicks):
            QTest.keySequence(self.pg, QKeySequence(self.shortcuts["move_cursor_left_20x"]))
            self.processEvents(0.01)
        self.processEvents(0.1)
        pos -= 20 * ctrl_left_clicks
        cursor_prev = cursor_now
        cursor_now = Pixmap.cursors_x(self.pg.grab())[0]

        # Evaluate
        self.assertLess(cursor_now, cursor_prev)
        self.assertEqual(f"{ch.signal.samples[pos]} {ch.unit}", self.plot.selected_channel_value.text())
        self.assertEqual(f"{ci.name} = {round(ch.signal.timestamps[pos], ci.precision)}{ci.unit}", ci.text())
        self.assertEqual(ch.signal.timestamps[pos], self.pg.cursor1.getXPos())

    def test_shift_channels_shortcut(self):
        """
        Test Scope:
            Check that Shift + Arrow Keys ensure moving of selected channels.
        Events:
            - Create plot with 2 channels
            - Press key "S" to separate signals for better evaluation
            - Click on first channel
            - Press "Shift" key + arrow "Down" & "Left"
            - Click on second channel
            - Press "Shift" key + arrow "Up" & "Right"
        Evaluate:
            - Evaluate that first signal is shifted down & left after pressing combination "Shift+Down" & "Shift+Left"
            - Evaluate that second signal is shifted up & right after pressing combination "Shift+Up" & "Shift+Right"
        """
        if self.plot.lock_btn.isFlat():
            QTest.mouseClick(self.plot.lock_btn, Qt.MouseButton.LeftButton)
        self.add_channels([36, 37])
        channel_36 = self.channels[0]
        channel_37 = self.channels[1]

        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["stack_all"]))
        self.processEvents(0.01)

        # Zoom out
        x = round(self.plot.plot.width() / 2)
        y = round(self.plot.plot.height() / 2)
        QTest.mouseClick(self.pg.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x, y))
        self.wheel_action(self.pg.viewport(), x, y, -1)
        self.processEvents(0.1)

        # Find extremes of signals
        old_from_to_y_channel_36 = Pixmap.search_signal_extremes_by_ax(self.pg.grab(), channel_36.color.name(), "y")
        old_from_to_x_channel_36 = Pixmap.search_signal_extremes_by_ax(self.pg.grab(), channel_36.color.name(), "x")
        old_from_to_y_channel_37 = Pixmap.search_signal_extremes_by_ax(self.pg.grab(), channel_37.color.name(), "y")
        old_from_to_x_channel_37 = Pixmap.search_signal_extremes_by_ax(self.pg.grab(), channel_37.color.name(), "x")

        # Select first channel and move signal using commands Shift + PgDown/Down/Left
        self.mouseClick_WidgetItem(channel_36)
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["shift_channels_down_10x"]))
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["shift_channels_down_1x"]))
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["shift_channels_left"]))
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["shift_channels_left"]))
        self.is_not_blinking(self.pg, {channel_36.color.name(), channel_37.color.name()})

        # Select second channel and move signal using commands Shift + PgUp/Up/Right
        self.mouseClick_WidgetItem(channel_37)
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["shift_channels_up_10x"]))
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["shift_channels_up_1x"]))
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["shift_channels_right"]))
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["shift_channels_right"]))
        self.is_not_blinking(self.pg, {channel_36.color.name(), channel_37.color.name()})

        # Find new extremes
        new_from_to_y_channel_36 = Pixmap.search_signal_extremes_by_ax(self.pg.grab(), channel_36.color.name(), "y")
        new_from_to_y_channel_37 = Pixmap.search_signal_extremes_by_ax(self.pg.grab(), channel_37.color.name(), "y")
        new_from_to_x_channel_36 = Pixmap.search_signal_extremes_by_ax(self.pg.grab(), channel_36.color.name(), "x")
        new_from_to_x_channel_37 = Pixmap.search_signal_extremes_by_ax(self.pg.grab(), channel_37.color.name(), "x")

        # Evaluate
        self.assertLess(old_from_to_y_channel_36[0], new_from_to_y_channel_36[0])
        self.assertLess(old_from_to_y_channel_36[1], new_from_to_y_channel_36[1])
        self.assertGreater(old_from_to_x_channel_36[0], new_from_to_x_channel_36[0])
        self.assertGreater(old_from_to_x_channel_36[1], new_from_to_x_channel_36[1])

        self.assertGreater(old_from_to_y_channel_37[0], new_from_to_y_channel_37[0])
        self.assertGreater(old_from_to_y_channel_37[1], new_from_to_y_channel_37[1])
        self.assertLess(old_from_to_x_channel_37[0], new_from_to_x_channel_37[0])
        self.assertLess(old_from_to_x_channel_37[1], new_from_to_x_channel_37[1])

    def test_test_honeywell_shortcut(self):
        """
        Test Scope:
            Check if honeywell function is applied to signal after pressing key "H"

        Events:

            - Display 1 signal on plot
            - Select signal
            - Press "H"
            - Set window mode to full screen
            - Pres "H"

        Evaluate:
            - Evaluate the range of x-axis after pressing key "H", "honey range" must be respected
            - Evaluate the range of x-axis is same for maximized window
            - Evaluate the range of x-axis after pressing key "H" second time, "honey range" must be respected
        """

        def find_honey_range(plot):
            rect = plot.plotItem.vb.sceneBoundingRect()
            dpi = QGuiApplication.primaryScreen().physicalDotsPerInchX()
            dpc = dpi / 2.54  # from inch to cm
            physical_viewbox_width = (rect.width() - 5) / dpc  # cm
            return physical_viewbox_width * 0.1

        # Setup
        self.add_channels([35])
        expected_normal_screen_honey_range = find_honey_range(self.pg)
        # Press "H"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["honeywell"]))
        self.processEvents(0.01)
        delta_normal_screen_x_range = self.pg.x_range[1] - self.pg.x_range[0]
        # Evaluate
        self.assertAlmostEqual(delta_normal_screen_x_range, expected_normal_screen_honey_range, delta=0.0001)

        # Minimize widget
        self.widget.setFixedSize(int(self.widget.width() * 0.9), int(self.widget.height() * 0.9))
        self.processEvents()
        # Evaluate
        self.assertAlmostEqual(self.pg.x_range[1] - self.pg.x_range[0], delta_normal_screen_x_range, delta=0.0001)

        expected_full_screen_honey_range = find_honey_range(self.pg)
        # Press "H"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["honeywell"]))
        self.processEvents(0.01)
        delta_full_screen_x_range = self.pg.x_range[1] - self.pg.x_range[0]

        # Evaluate
        self.assertNotEqual(delta_full_screen_x_range, delta_normal_screen_x_range)
        self.assertAlmostEqual(delta_full_screen_x_range, expected_full_screen_honey_range, delta=0.0001)

    def test_home_shortcuts(self):
        """
        Check if the signal is fitted properly after pressing key "W".
        Events:
            - Create a plot window with 2 signals
            - Press key "I"
            - Press key "W"
        Evaluate:
            - Evaluate that there is at least one column with first signal color
            - Evaluate first and last columns where is first signal:
                > first column after pressing "I" is full black => signal colors are not there
                > signal is zoomed => is extended to left side => last column contain signal color
            - Evaluate that after pressing key "W", signal is displayed from first to last column
        """
        settings = QSettings()
        settings.setValue("zoom_x_center_on_cursor", True)

        self.assertIsNotNone(self.add_channels([35]))
        channel_35 = self.channels[0]

        # check if the grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QTest.mouseClick(self.plot.hide_axes_btn, Qt.MouseButton.LeftButton)

        # Press "W"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["home"]))
        self.processEvents(0.01)

        # search first and last column where is displayed first signal
        extremes_of_channel_35 = Pixmap.search_signal_extremes_by_ax(self.pg.grab(), channel_35.color.name(), ax="x")
        # Evaluate that there are extremes of first signal
        self.assertTrue(extremes_of_channel_35)
        # Press "I"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["x_zoom_in"]))
        self.processEvents(0.5)

        # save left and right pixel column
        x_left_column = self.pg.grab(QRect(extremes_of_channel_35[0], 0, 1, self.pg.height()))
        x_right_column = self.pg.grab(QRect(extremes_of_channel_35[1], 0, 1, self.pg.height()))
        self.assertTrue(Pixmap.is_black(x_left_column))
        self.assertTrue(Pixmap.has_color(x_right_column, channel_35.color.name()))

        # press "F"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["fit_all"]))
        # Press "W"
        QTest.keySequence(self.pg, QKeySequence(self.shortcuts["home"]))
        self.processEvents()
        # Select all columns from left to right
        for x in range(self.pg.height() - 1):
            column = self.pg.grab(QRect(x, 0, 1, self.pg.height()))
            if x < extremes_of_channel_35[0] - 1:
                self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")
            elif extremes_of_channel_35[0] <= x <= extremes_of_channel_35[1]:
                self.assertTrue(
                    Pixmap.has_color(column, channel_35.color.name()),
                    f"column {x} doesn't have {channel_35.name} color",
                )
            else:
                self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")

    def test_insert_computation_shortcut(self):
        """
        Test Scope:
            Check Insert key shortcut action

        Events:
            - Press Insert with preconditions:
                > There is no user defined functon
                > There is user defined functon, button Cancel or "X" was pressed
                > There is user defined function, button Apply was pressed

        Evaluate:
            - Evaluate that channel selection area is empty
        Evaluate (0):
            - Evaluate that warning message box was triggered after pressing key insert
            - Evaluate displayed warning message
            - Evaluate that channel selection area is empty
        Evaluate (1):
            - Evaluate that channel selection area is empty
            - Evaluate that DefineChannel object was called
        Evaluate (2):
            - Evaluate that there is one channel in channel selection area
            - Evaluate that the name of this channel is correct
            - Evaluate that DefineChannel object was called
        """
        # Evaluate precondition
        self.assertEqual(0, self.plot.channel_selection.topLevelItemCount())

        with self.subTest("_0_test_warning_no_user_function_defined"):
            warnings_msgs = [
                "Cannot add computed channel",
                "There is no user defined function. Create new function using the Functions Manger (F6)",
            ]
            # mock for warning message box
            with mock.patch("asammdf.gui.widgets.plot.MessageBox.warning") as mo_waring:
                # Press key Insert
                QTest.keySequence(self.plot.channel_selection, QKeySequence(self.shortcuts["insert_computation"]))
            # Evaluate
            self.assertEqual(0, self.plot.channel_selection.topLevelItemCount())
            mo_waring.assert_called()
            for w in warnings_msgs:
                self.assertIn(w, mo_waring.call_args.args)

        with self.subTest("_1_test_cancel_dlg_with_user_function_defined"):
            file_name = "test_insert_cfg.dspf"
            file_path = os.path.join(self.resource, file_name)
            self.load_display_file(file_path)
            self.plot = self.widget.mdi_area.subWindowList()[0].widget()
            with mock.patch("asammdf.gui.widgets.plot.DefineChannel") as mo_DefineChannel:
                # Press key Insert
                QTest.keySequence(self.plot.channel_selection, QKeySequence(self.shortcuts["insert_computation"]))

            # Evaluate
            self.assertEqual(1, self.plot.channel_selection.topLevelItemCount())
            mo_DefineChannel.assert_called()

        with self.subTest("_2_test_apply_dlg_with_user_function_defined"):
            file_name = "test_insert_cfg.dspf"
            file_path = os.path.join(self.resource, file_name)
            self.load_display_file(file_path)
            self.plot = self.widget.mdi_area.subWindowList()[0].widget()
            computed_channel = {
                "type": "channel",
                "common_axis": False,
                "individual_axis": False,
                "enabled": True,
                "mode": "phys",
                "fmt": "{:.3f}",
                "format": "phys",
                "precision": 3,
                "flags": 0,
                "ranges": [],
                "unit": "",
                "computed": True,
                "color": "#994380",
                "uuid": "525ad72a531a",
                "origin_uuid": "812d7b792168",
                "group_index": -1,
                "channel_index": -1,
                "name": self.id(),
                "computation": {
                    "args": {},
                    "type": "python_function",
                    "definition": "",
                    "channel_name": "Function_728d4a149b44",
                    "function": "Function1",
                    "channel_unit": "",
                    "channel_comment": "",
                    "triggering": "triggering_on_all",
                    "triggering_value": "all",
                    "computation_mode": "sample_by_sample",
                },
            }
            with mock.patch("asammdf.gui.widgets.plot.DefineChannel") as mo_DefineChannel:
                mo_DefineChannel.return_value.result = computed_channel
                # Press key Insert
                QTest.keySequence(self.plot.channel_selection, QKeySequence(self.shortcuts["insert_computation"]))

            # Evaluate
            self.assertEqual(2, self.plot.channel_selection.topLevelItemCount())
            self.assertEqual(self.plot.channel_selection.topLevelItem(1).name, self.id())
            mo_DefineChannel.assert_called()

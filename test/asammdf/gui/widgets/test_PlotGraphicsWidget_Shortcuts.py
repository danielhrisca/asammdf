#!/usr/bin/env python\

import os
from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget
from unittest import mock

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtTest import QTest


class TestPlotGraphicsShortcuts(TestPlotWidget):
    def setUp(self):
        """
        Events:
            Open test measurement file
            Set sort method: Natural sort
            Create new plot widget
            Add one channel to plot
        Evaluate:
            - Evaluate that one widget was created
            - Evaluate that one channel was added to plot
        """
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        # Plot widget object
        self.plot = self.widget.mdi_area.subWindowList()[0].widget().plot
        self.assertIsNotNone(self.add_channels(channels_list=[35]))
        self.processEvents()

        # get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.plot))
        self.processEvents()

    def test_Plot_PlotGraphics_All_Shortcuts(self):
        """
        Unittests for PlotGraphics widget shortcuts
        """
        # Setup
        min_ = self.plot.signals[0].samples.min()
        max_ = self.plot.signals[0].samples.max()
        delta = 0.01 * (max_ - min_)
        min_, max_ = min_ - delta, max_ + delta

        with self.subTest("test_Y"):
            self.plot.region_lock = None
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["toggle_range"]))
            self.assertIsNotNone(self.plot.region_lock)
            self.assertFalse(self.plot.region.movable)
            self.assertFalse(self.plot.region.lines[0].movable)
            self.assertTrue(self.plot.region.lines[0].locked)

            self.plot.region_lock = 0.01
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["toggle_range"]))
            self.assertIsNone(self.plot.region_lock)
            self.assertTrue(self.plot.region.movable)
            self.assertTrue(self.plot.region.lines[0].movable)
            self.assertFalse(self.plot.region.lines[0].locked)

        with self.subTest("test_X_R"):
            with mock.patch.object(self.plot.viewbox, "setXRange") as mo_setXRange:
                self.plot.region = None
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["zoom_to_range"]))
                mo_setXRange.assert_not_called()

                self.assertIsNone(self.plot.region)
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["range"]))
                self.assertIsNotNone(self.plot.region)

                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["zoom_to_range"]))
                mo_setXRange.assert_called()
                self.assertIsNone(self.plot.region)

        with self.subTest("test_F"):
            with mock.patch.object(self.plot.viewbox, "setYRange") as mo_setYRange:
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["fit_all"]))
                mo_setYRange.assert_called_with(min_, max_, padding=0)

        with self.subTest("test_Shift_F"):
            with mock.patch.object(self.plot.viewbox, "setYRange") as mo_setYRange:
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["fit_selected"]))
                mo_setYRange.assert_not_called()
                self.mouseClick_WidgetItem(self.channels[0])
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["fit_selected"]))
                mo_setYRange.assert_called_with(min_, max_, padding=0)

        with self.subTest("test_G_Shift_G"):
            self.plot.y_axis.grid = True
            self.plot.x_axis.grid = True
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["grid"]))
            self.assertFalse(self.plot.y_axis.grid or self.plot.x_axis.grid)

            self.plot.x_axis.grid = True
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["grid"]))
            self.assertTrue(self.plot.y_axis.grid and self.plot.x_axis.grid)

            self.plot.y_axis.grid = True
            self.plot.x_axis.grid = False
            QTest.keySequence(self.plot, QKeySequence(self.shortcuts["grid"]))
            self.assertTrue(self.plot.x_axis.grid)
            self.assertFalse(self.plot.y_axis.grid)
            with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QInputDialog.getDouble") as mo_getDouble:
                expected_pos = 0.5
                mo_getDouble.return_value = expected_pos, True
                QTest.keySequence(self.plot, QKeySequence(self.shortcuts["go_to_timestamp"]))
                mo_getDouble.assert_called()
                self.assertAlmostEqual(self.plot.cursor1.getPos()[0], expected_pos, delta=0.001)


class TestPlotGraphicsShortcutsFunctionality(TestPlotWidget):

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
        # PlotGraphics object
        self.plot_graphics = self.plot.plot

        # Setup for PlotGraphics
        if self.plot_graphics.with_dots:
            self.plot_graphics.set_dots(False)
        if not self.plot.hide_axes_btn.isFlat():
            QTest.mouseClick(self.plot.hide_axes_btn, Qt.MouseButton.LeftButton)
        if self.plot.show_bookmarks:
            self.plot.toggle_bookmarks(hide=True)
        if self.plot_graphics.cursor1.show_circle:
            self.plot_graphics.cursor1.show_circle = False

        self.processEvents(0.01)
        # Evaluate that plot is black
        self.assertTrue(Pixmap.is_black(self.plot_graphics.grab()))

        # get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.plot_graphics))
        self.processEvents()

    def tearDown(self):
        """
        Destroy widget if it still exists
        """
        if self.widget:
            self.widget.destroy()

    def test_Plot_PlotGraphics_Shortcut_Key_Y(self):
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
        clean_pixmap = self.plot_graphics.grab()
        # Evaluate that plot is black
        self.assertTrue(Pixmap.is_black(clean_pixmap))

        # Get position of Cursor
        cursors = Pixmap.cursors_x(clean_pixmap)
        # Evaluate that there is only one cursor
        self.assertEqual(1, len(cursors))

        # Press Key 'Y' for range selection
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["toggle_range"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.plot_graphics.grab()
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
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["move_cursor_left_1x"]))
        self.processEvents(timeout=0.01)
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["move_cursor_left_10x"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.plot_graphics.grab()
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
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["toggle_range"]))
        self.processEvents(timeout=0.01)

        # Move Cursors
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["move_cursor_right_1x"]))
        self.processEvents(timeout=0.01)
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["move_cursor_left_10x"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.plot_graphics.grab()
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
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["range"]))
        self.processEvents(timeout=0.01)
        # Save PixMap of clear plot
        clean_pixmap = self.plot_graphics.grab()
        self.assertTrue(Pixmap.is_black(clean_pixmap))

    def test_Plot_PlotGraphics_Shortcut_Key_X(self):
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
            - Evaluate that intersection of signal and midd line is exactly how much it intersect between cursors
            - Evaluate X range, it must be almost equal as range value after pressing key R
        """
        # Setup
        self.add_channels([35])
        channel_color = self.channels[0].color.name()

        self.widget.showMaximized()
        self.processEvents()

        # Count intersections between middle line and signal
        initial_intersections = Pixmap.color_map(
            self.plot_graphics.grab(QRect(0, int(self.plot_graphics.height() / 2), self.plot_graphics.width(), 1))
        )[0].count(channel_color)
        self.assertTrue(initial_intersections)

        # Setup for cursor
        self.widget.set_cursor_options(False, False, 1, Pixmap.COLOR_CURSOR)
        # Mouse click on a center of plot
        QTest.mouseClick(
            self.plot_graphics.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifiers(),
            self.plot_graphics.rect().center(),
        )
        # Press R
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["range"]))
        self.processEvents()

        x_range = self.plot_graphics.region.getRegion()
        self.assertNotIn(x_range[0], self.plot_graphics.x_range)
        self.assertNotIn(x_range[1], self.plot_graphics.x_range)
        # Get X position of Cursor
        cursors = Pixmap.cursors_x(self.plot_graphics.grab())
        # Ensure that both cursors was found
        self.assertEqual(len(cursors), 2)

        # Get a set of colors founded between cursors
        colors = Pixmap.color_names_exclude_defaults(
            self.plot_graphics.grab(QRect(cursors[0], 0, cursors[1] - cursors[0], self.plot_graphics.height()))
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
            self.plot_graphics.grab(QRect(cursors[0], int(self.plot_graphics.height() / 2), cursors[1] - cursors[0], 1))
        )[0].count(color)
        self.assertTrue(expected_intersections)

        # Press key "X"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["zoom_to_range"]))
        self.processEvents()

        # Evaluate how much times signal intersect midd line
        actual_intersections = Pixmap.color_map(
            self.plot_graphics.grab(QRect(0, int(self.plot_graphics.height() / 2), self.plot_graphics.width(), 1))
        )[0].count(channel_color)
        self.assertEqual(actual_intersections, expected_intersections)
        self.assertLess(actual_intersections, initial_intersections)

        # Evaluate ranges of signal
        self.assertAlmostEqual(self.plot_graphics.x_range[0], x_range[0], delta=0.001)
        self.assertAlmostEqual(self.plot_graphics.x_range[1], x_range[1], delta=0.001)

    def test_Plot_PlotGraphics_Shortcut_Key_S_ShiftS_ShiftF_F(self):
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
            - Evaluate that all signals is fitted after pressing key "F"

        Additional Evaluation
            - Evaluate that all signals is continuous on plot
        """
        self.plot_graphics.cursor1.color = "#000000"

        self.add_channels([35, 36, 37])

        channel_35 = self.channels[0]
        channel_36 = self.channels[1]
        channel_37 = self.channels[2]
        # Press "S"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["stack_all"]))
        self.processEvents()
        # Evaluate
        with self.subTest("test_shortcut_S"):
            # First 2 lines
            self.assertTrue(Pixmap.is_black(self.plot_graphics.grab(QRect(0, 0, self.plot_graphics.width(), 2))))
            # Top
            pixmap = self.plot_graphics.grab(
                QRect(0, 0, self.plot_graphics.width(), int(self.plot_graphics.height() / 3))
            )
            self.assertTrue(Pixmap.has_color(pixmap, channel_35.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, channel_37.color.name()))
            # Midd
            pixmap = self.plot_graphics.grab(
                QRect(
                    0,
                    int(self.plot_graphics.height() / 3),
                    self.plot_graphics.width(),
                    int(self.plot_graphics.height() / 3),
                )
            )
            self.assertFalse(Pixmap.has_color(pixmap, channel_35.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, channel_37.color.name()))
            # Bottom
            pixmap = self.plot_graphics.grab(
                QRect(
                    0,
                    int(self.plot_graphics.height() / 3) * 2,
                    self.plot_graphics.width(),
                    int(self.plot_graphics.height() / 3),
                )
            )
            self.assertFalse(Pixmap.has_color(pixmap, channel_35.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, channel_36.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, channel_37.color.name()))
            # Last 2 lines

            pixmap = self.plot_graphics.grab(QRect(0, self.plot_graphics.height() - 3, self.plot_graphics.width(), 2))
            cn = Pixmap.color_names_exclude_defaults(pixmap)
            self.assertEqual(len(cn), 0)

        # select the first channel
        self.mouseClick_WidgetItem(channel_35)
        # Press "Shift+F"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["fit_selected"]))
        self.avoid_blinking_issue(self.plot.channel_selection)
        for _ in range(50):
            self.processEvents()
        # Evaluate
        with self.subTest("test_shortcut_Shift_F"):
            # First line
            self.assertTrue(Pixmap.is_black(self.plot_graphics.grab(QRect(0, 0, self.plot_graphics.width(), 1))))
            # Top
            pixmap = self.plot_graphics.grab(
                QRect(0, 0, self.plot_graphics.width(), int(self.plot_graphics.height() / 3))
            )
            self.assertTrue(Pixmap.has_color(pixmap, channel_35.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, channel_37.color.name()))
            # Midd
            pixmap = self.plot_graphics.grab(
                QRect(
                    0,
                    int(self.plot_graphics.height() / 3),
                    self.plot_graphics.width(),
                    int(self.plot_graphics.height() / 3),
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, channel_35.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, channel_37.color.name()))
            # Bottom
            pixmap = self.plot_graphics.grab(
                QRect(
                    0,
                    int(self.plot_graphics.height() / 3) * 2,
                    self.plot_graphics.width(),
                    int(self.plot_graphics.height() / 3),
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, channel_35.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, channel_36.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, channel_37.color.name()))
            # Last line
            self.assertTrue(
                Pixmap.is_black(
                    self.plot_graphics.grab(QRect(0, self.plot_graphics.height() - 2, self.plot_graphics.width(), 1))
                )
            )

        # select second channel
        self.mouseClick_WidgetItem(channel_36)
        # Press "Shift+F"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["stack_selected"]))
        self.avoid_blinking_issue(self.plot.channel_selection)
        # Evaluate
        with self.subTest("test_shortcut_Shift_S"):
            # First line
            self.assertTrue(Pixmap.is_black(self.plot_graphics.grab(QRect(0, 0, self.plot_graphics.width(), 1))))
            # Top
            pixmap = self.plot_graphics.grab(
                QRect(0, 0, self.plot_graphics.width(), int(self.plot_graphics.height() / 3))
            )
            self.assertTrue(Pixmap.has_color(pixmap, channel_35.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, channel_37.color.name()))
            # Midd
            pixmap = self.plot_graphics.grab(
                QRect(
                    0,
                    int(self.plot_graphics.height() / 3),
                    self.plot_graphics.width(),
                    int(self.plot_graphics.height() / 3),
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, channel_35.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, channel_37.color.name()))
            # Bottom
            pixmap = self.plot_graphics.grab(
                QRect(
                    0,
                    int(self.plot_graphics.height() / 3) * 2,
                    self.plot_graphics.width(),
                    int(self.plot_graphics.height() / 3),
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, channel_35.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, channel_36.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, channel_37.color.name()))
            # Last line
            self.assertTrue(
                Pixmap.is_black(
                    self.plot_graphics.grab(QRect(0, self.plot_graphics.height() - 1, self.plot_graphics.width(), 1))
                )
            )

            # Press "F"
            QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["fit_all"]))
            self.avoid_blinking_issue(self.plot.channel_selection)
            # Evaluate
            with self.subTest("test_shortcut_F"):
                # First line
                self.assertTrue(Pixmap.is_black(self.plot_graphics.grab(QRect(0, 0, self.plot_graphics.width(), 1))))
                # Top
                pixmap = self.plot_graphics.grab(
                    QRect(0, 0, self.plot_graphics.width(), int(self.plot_graphics.height() / 3))
                )
                self.assertTrue(Pixmap.has_color(pixmap, channel_35.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, channel_36.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, channel_37.color.name()))
                # Midd
                pixmap = self.plot_graphics.grab(
                    QRect(
                        0,
                        int(self.plot_graphics.height() / 3),
                        self.plot_graphics.width(),
                        int(self.plot_graphics.height() / 3),
                    )
                )
                self.assertTrue(Pixmap.has_color(pixmap, channel_35.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, channel_36.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, channel_37.color.name()))
                # Bottom
                pixmap = self.plot_graphics.grab(
                    QRect(
                        0,
                        int(self.plot_graphics.height() / 3) * 2,
                        self.plot_graphics.width(),
                        int(self.plot_graphics.height() / 3),
                    )
                )
                self.assertTrue(Pixmap.has_color(pixmap, channel_35.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, channel_36.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, channel_37.color.name()))
                # Last line
                self.assertTrue(
                    Pixmap.is_black(
                        self.plot_graphics.grab(
                            QRect(0, self.plot_graphics.height() - 1, self.plot_graphics.width(), 1)
                        )
                    )
                )
                # deselect all channels
                for channel in self.channels:
                    self.mouseDClick_WidgetItem(channel)

                # search if all channels is fitted into extremes
                self.mouseDClick_WidgetItem(channel_35)
                extremes = Pixmap.search_signal_extremes_by_ax(
                    self.plot_graphics.grab(), signal_color=channel_35.color.name(), ax="x"
                )
                for x in range(self.plot_graphics.height() - 1):
                    column = self.plot_graphics.grab(QRect(x, 0, 1, self.plot_graphics.height()))
                    if x < extremes[0] - 1:
                        self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")
                    elif extremes[0] <= x <= extremes[1]:
                        self.assertTrue(
                            Pixmap.has_color(column, channel_35.color.name()),
                            f"column {x} doesn't have color of channel 35",
                        )
                    else:
                        self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")

                self.mouseDClick_WidgetItem(channel_35)
                self.mouseDClick_WidgetItem(channel_36)
                for x in range(self.plot_graphics.height() - 1):
                    column = self.plot_graphics.grab(QRect(x, 0, 1, self.plot_graphics.height()))
                    if x < extremes[0] - 1:
                        self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")
                    elif extremes[0] <= x <= extremes[1]:
                        self.assertTrue(
                            Pixmap.has_color(column, channel_36.color.name()),
                            f"column {x} doesn't have color of channel 36",
                        )
                    else:
                        self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")

                self.mouseDClick_WidgetItem(channel_37)
                self.mouseDClick_WidgetItem(channel_36)
                for x in range(self.plot_graphics.height() - 1):
                    column = self.plot_graphics.grab(QRect(x, 0, 1, self.plot_graphics.height()))
                    if x < extremes[0] - 1:
                        self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")
                    elif extremes[0] <= x <= extremes[1]:
                        self.assertTrue(
                            Pixmap.has_color(column, channel_37.color.name()),
                            f"column {x} doesn't have color of channel 37",
                        )
                    else:
                        self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")

    def test_Plot_PlotGraphics_Shortcut_Key_G(self):
        """
        Test Scope:
            Check if grid is created properly after pressing key "G".

        Events:
            - If axes is hidden - press "Show axes" button
            - Press Key "G" 3 times

        Evaluate:
            - Evaluate that grid is displayed in order after pressing key "G":
                1. Is only X axes grid
                2. Is X and Y axes grid
                3. There is no grid
        """
        # check if grid is available
        if self.plot.hide_axes_btn.isFlat():
            QTest.mouseClick(self.plot.hide_axes_btn, Qt.MouseButton.LeftButton)
        # case 1: X and Y axes is hidden
        if not self.plot_graphics.x_axis.grid and not self.plot_graphics.y_axis.grid:
            with self.subTest("test_shortcut_key_G_no_grid_displayed"):
                # press key "G"
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["grid"]))
                self.processEvents()
                # Evaluate
                self.assertTrue(self.plot_graphics.x_axis.grid)
                self.assertFalse(self.plot_graphics.y_axis.grid)
                # press key "G"
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["grid"]))
                self.processEvents()
                # Evaluate
                self.assertTrue(self.plot_graphics.x_axis.grid)
                self.assertTrue(self.plot_graphics.y_axis.grid)
                # press key "G"
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["grid"]))
                self.processEvents()
                # Evaluate
                self.assertFalse(self.plot_graphics.x_axis.grid)
                self.assertFalse(self.plot_graphics.y_axis.grid)
        # case 2: X is visible, Y is hidden
        elif self.plot_graphics.x_axis.grid and not self.plot_graphics.y_axis.grid:
            with self.subTest("test_shortcut_key_G_X_grid_already_displayed"):
                # press key "G"
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["grid"]))
                self.processEvents()
                # Evaluate
                self.assertTrue(self.plot_graphics.x_axis.grid)
                self.assertTrue(self.plot_graphics.y_axis.grid)
                # press key "G"
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["grid"]))
                self.processEvents()
                # Evaluate
                self.assertFalse(self.plot_graphics.x_axis.grid)
                self.assertFalse(self.plot_graphics.y_axis.grid)
                # press key "G"
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["grid"]))
                self.processEvents()
                # Evaluate
                self.assertTrue(self.plot_graphics.x_axis.grid)
                self.assertFalse(self.plot_graphics.y_axis.grid)
        # case 3: X and Y axes is visible
        else:
            with self.subTest("test_shortcut_key_G_XU_grid_already_displayed"):
                # press key "G"
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["grid"]))
                self.processEvents()
                # Evaluate
                self.assertTrue(self.plot_graphics.x_axis.grid)
                self.assertFalse(self.plot_graphics.y_axis.grid)
                # press key "G"
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["grid"]))
                self.processEvents()
                # Evaluate
                self.assertTrue(self.plot_graphics.x_axis.grid)
                self.assertTrue(self.plot_graphics.y_axis.grid)
                # press key "G"
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["grid"]))
                self.processEvents()
                # Evaluate
                self.assertFalse(self.plot_graphics.x_axis.grid)
                self.assertFalse(self.plot_graphics.y_axis.grid)

    def test_Plot_PlotGraphics_Shortcut_Keys_I__Shift_I__O__Shift_O(self):
        """
        Test Scope:
            Check if zooming is released after pressing keys "I", "Shift+I", "O", "Shift+O".

        Events:
            - Display 1 signal on plot
            - Select signal, click in the middle of plot
            - Press "I"
            - Press "O"
            - Press "Shift+I"
            - Press "Shift+O"

        Evaluate:
            - Evaluate ranges of X and Y asis.
        """

        def get_expected_result(step, is_x_axis: bool):
            if is_x_axis:
                delta = self.plot_graphics.x_range[1] - self.plot_graphics.x_range[0]
                cursor_pos = self.plot_graphics.cursor1.value()
                step = delta * step
                return cursor_pos - delta / 2 - step, cursor_pos + delta / 2 + step
            else:
                bottom = self.plot_graphics.signals[0].y_range[1]  # 0
                top = self.plot_graphics.signals[0].y_range[0]  # 255
                delta = top - bottom
                cursor_pos = self.plot_graphics.cursor1.value()
                y_value = self.plot_graphics.signals[0].value_at_timestamp(cursor_pos, numeric=True)[0]

                dp = (y_value - (top + bottom) / 2) / (top - bottom)  # -0.468627

                shift = dp * delta  # -119.5
                top, bottom = shift + top, shift + bottom  # 135.5, 119.5
                delta = top - bottom
                return top - delta * step, bottom + delta * step

        # Setup
        y_step = 0.165
        x_step = 0.25

        self.assertIsNotNone(self.add_channels([35]))
        # click con center
        QTest.mouseClick(
            self.plot.plot.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifiers(),
            QPoint(int(self.plot_graphics.width() / 2), int(self.plot_graphics.height() / 2)),
        )
        self.processEvents()

        # Events Without Pressed Shift
        expected_x_zoom_in_range = get_expected_result(-x_step, True)
        # Press "I"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["x_zoom_in"]))
        self.processEvents()
        x_zoom_in_range = self.plot_graphics.x_range

        expected_x_zoom_out_range = get_expected_result(x_step * 2, True)
        # Press "O"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["x_zoom_out"]))
        self.processEvents()
        x_zoom_out_range = self.plot_graphics.x_range

        # Events with pressed Shift
        expected_y_zoom_in_range = get_expected_result(y_step, False)
        # Press "I"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["y_zoom_in"]))
        self.processEvents()
        y_zoom_in_range = self.plot_graphics.signals[0].y_range

        expected_y_zoom_out_range = get_expected_result(-y_step, False)
        # Press "O"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["y_zoom_out"]))
        self.processEvents()

        y_zoom_out_range = self.plot_graphics.signals[0].y_range
        # Evaluate
        # Key Shift wasn't pressed
        self.assertAlmostEqual(x_zoom_in_range[0], expected_x_zoom_in_range[0], delta=0.0001)
        self.assertAlmostEqual(x_zoom_in_range[1], expected_x_zoom_in_range[1], delta=0.0001)
        self.assertAlmostEqual(x_zoom_out_range[0], expected_x_zoom_out_range[0], delta=0.0001)
        self.assertAlmostEqual(x_zoom_out_range[1], expected_x_zoom_out_range[1], delta=0.0001)
        # Key Shift was pressed
        self.assertAlmostEqual(y_zoom_in_range[0], expected_y_zoom_in_range[0], delta=0.0001)
        self.assertAlmostEqual(y_zoom_in_range[1], expected_y_zoom_in_range[1], delta=0.0001)
        self.assertAlmostEqual(y_zoom_out_range[0], expected_y_zoom_out_range[0], delta=0.0001)
        self.assertAlmostEqual(y_zoom_out_range[1], expected_y_zoom_out_range[1], delta=0.0001)

    def test_Plot_PlotGraphics_Shortcut_Key_R(self):
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
            - Evaluate that range selection disappear after pressing key R second time.
        """
        self.widget.set_cursor_options(False, False, 1, Pixmap.COLOR_CURSOR)
        # Save PixMap of clear plot
        clear_pixmap = self.plot_graphics.grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

        # Get X position of Cursor
        cursors = Pixmap.cursors_x(clear_pixmap)
        # Evaluate that there is only one cursor
        self.assertEqual(1, len(cursors))

        # Press Key 'R' for range selection
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["range"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.plot_graphics.grab()
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
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["move_cursor_right_1x"]))
        self.processEvents(timeout=0.01)
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["move_cursor_left_10x"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.plot_graphics.grab()
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
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["range"]))
        self.processEvents(timeout=0.01)

        # Save PixMap of clear plot
        clear_pixmap = self.plot_graphics.grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

    def test_Plot_PlotGraphics_Shortcut_Key_LeftRight(self):
        """
        Test Scope:
            Check that Arrow Keys: Left & Right ensure navigation on channels evolution.
            Ensure that navigation is working.
        Events:
            - Display 2 signals to plot
            - Send KeyClick Right 5 times
            - Send KeyClick Left 4 times
        Evaluate:
            - Evaluate values from `Value` column on self.plot.channels_selection
            - Evaluate timestamp label
        """
        self.add_channels([36, 37])
        channel_36 = self.channels[0]
        channel_37 = self.channels[1]

        # Case 0:
        with self.subTest("test_Plot_Plot_Shortcut_Key_LeftRight_0"):
            # Select channel: ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            self.mouseClick_WidgetItem(channel_37)
            self.plot_graphics.setFocus()
            self.processEvents(0.1)

            self.assertEqual("25", channel_36.text(self.Column.VALUE))
            self.assertEqual("244", channel_37.text(self.Column.VALUE))

            # Send Key strokes
            for _ in range(6):
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["move_cursor_right_1x"]))
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("8", channel_36.text(self.Column.VALUE))
            self.assertEqual("6", channel_37.text(self.Column.VALUE))
            self.assertEqual("t = 0.082657s", self.plot.cursor_info.text())

            # Send Key strokes
            for _ in range(5):
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["move_cursor_left_1x"]))
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("21", channel_36.text(self.Column.VALUE))
            self.assertEqual("247", channel_37.text(self.Column.VALUE))
            self.assertEqual("t = 0.032657s", self.plot.cursor_info.text())

        # Case 1:
        with self.subTest("test_Plot_Plot_Shortcut_Key_LeftRight_1"):
            # Select channel: ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            self.mouseClick_WidgetItem(channel_37)
            self.plot_graphics.setFocus()
            self.processEvents(0.1)

            # Send Key strokes
            for _ in range(6):
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["move_cursor_right_1x"]))
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("5", channel_36.text(self.Column.VALUE))
            self.assertEqual("9", channel_37.text(self.Column.VALUE))
            self.assertEqual("t = 0.092657s", self.plot.cursor_info.text())

            # Send Key strokes
            for _ in range(5):
                QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["move_cursor_left_1x"]))
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("18", channel_36.text(self.Column.VALUE))
            self.assertEqual("250", channel_37.text(self.Column.VALUE))
            self.assertEqual("t = 0.042657s", self.plot.cursor_info.text())

    def test_Plot_PlotGraphics_Shortcut_Key_Shift_Arrows(self):
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
        self.add_channels([36, 37])
        channel_36 = self.channels[0]
        channel_37 = self.channels[1]

        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["stack_all"]))
        self.processEvents(0.01)
        old_from_to_y_channel_36 = Pixmap.search_signal_extremes_by_ax(
            self.plot_graphics.grab(), channel_36.color.name(), "y"
        )
        old_from_to_y_channel_37 = Pixmap.search_signal_extremes_by_ax(
            self.plot_graphics.grab(), channel_37.color.name(), "y"
        )
        old_from_to_x_channel_36 = Pixmap.search_signal_extremes_by_ax(
            self.plot_graphics.grab(), channel_36.color.name(), "x"
        )
        old_from_to_x_channel_37 = Pixmap.search_signal_extremes_by_ax(
            self.plot_graphics.grab(), channel_37.color.name(), "x"
        )

        self.mouseClick_WidgetItem(channel_36)
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["shift_channels_down_1x"]))
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["shift_channels_left"]))
        self.mouseClick_WidgetItem(channel_37)
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["shift_channels_up_1x"]))
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["shift_channels_right"]))

        self.avoid_blinking_issue(self.plot.channel_selection)

        new_from_to_y_channel_36 = Pixmap.search_signal_extremes_by_ax(
            self.plot_graphics.grab(), channel_36.color.name(), "y"
        )
        new_from_to_y_channel_37 = Pixmap.search_signal_extremes_by_ax(
            self.plot_graphics.grab(), channel_37.color.name(), "y"
        )
        new_from_to_x_channel_36 = Pixmap.search_signal_extremes_by_ax(
            self.plot_graphics.grab(), channel_36.color.name(), "x"
        )
        new_from_to_x_channel_37 = Pixmap.search_signal_extremes_by_ax(
            self.plot_graphics.grab(), channel_37.color.name(), "x"
        )

        # Evaluate
        self.assertLess(old_from_to_y_channel_36[0], new_from_to_y_channel_36[0])
        self.assertLess(old_from_to_y_channel_36[1], new_from_to_y_channel_36[1])
        self.assertGreater(old_from_to_x_channel_36[0], new_from_to_x_channel_36[0])
        self.assertGreater(old_from_to_x_channel_36[1], new_from_to_x_channel_36[1])

        self.assertGreater(old_from_to_y_channel_37[0], new_from_to_y_channel_37[0])
        self.assertGreater(old_from_to_y_channel_37[1], new_from_to_y_channel_37[1])
        self.assertLess(old_from_to_x_channel_37[0], new_from_to_x_channel_37[0])
        self.assertLess(old_from_to_x_channel_37[1], new_from_to_x_channel_37[1])

    def test_Plot_PlotGraphics_Shortcut_Key_H(self):
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
        expected_normal_screen_honey_range = find_honey_range(self.plot_graphics)
        # Press "H"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["honeywell"]))
        self.avoid_blinking_issue(self.plot.channel_selection)
        delta_normal_screen_x_range = self.plot_graphics.x_range[1] - self.plot_graphics.x_range[0]
        # Evaluate
        self.assertAlmostEqual(delta_normal_screen_x_range, expected_normal_screen_honey_range, delta=0.0001)

        # New Full screen test
        self.widget.showMaximized()
        self.processEvents()
        # Evaluate
        self.assertAlmostEqual(
            self.plot_graphics.x_range[1] - self.plot_graphics.x_range[0], delta_normal_screen_x_range, delta=0.0001
        )

        expected_full_screen_honey_range = find_honey_range(self.plot_graphics)
        # Press "H"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["honeywell"]))
        self.avoid_blinking_issue(self.plot.channel_selection)
        delta_full_screen_x_range = self.plot_graphics.x_range[1] - self.plot_graphics.x_range[0]

        # Evaluate
        self.assertNotEqual(delta_full_screen_x_range, delta_normal_screen_x_range)
        self.assertAlmostEqual(delta_full_screen_x_range, expected_full_screen_honey_range, delta=0.0001)

    def test_Plot_PlotGraphics_Shortcut_Key_W(self):
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
        self.assertIsNotNone(self.add_channels([35]))
        channel_35 = self.channels[0]

        # check if the grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QTest.mouseClick(self.plot.hide_axes_btn, Qt.MouseButton.LeftButton)

        # Press "W"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["home"]))
        self.processEvents(0.01)

        # search first and last column where is displayed first signal
        extremes_of_channel_35 = Pixmap.search_signal_extremes_by_ax(
            self.plot_graphics.grab(), channel_35.color.name(), ax="x"
        )
        # Evaluate that there are extremes of first signal
        self.assertTrue(extremes_of_channel_35)
        # Press "I"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["x_zoom_in"]))
        self.processEvents()

        # save left and right pixel column
        x_left_column = self.plot_graphics.grab(QRect(extremes_of_channel_35[0], 0, 1, self.plot_graphics.height()))
        x_right_column = self.plot_graphics.grab(QRect(extremes_of_channel_35[1], 0, 1, self.plot_graphics.height()))
        self.assertTrue(Pixmap.is_black(x_left_column))
        self.assertTrue(Pixmap.has_color(x_right_column, channel_35.color.name()))

        # press "F"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["fit_all"]))
        # Press "W"
        QTest.keySequence(self.plot_graphics, QKeySequence(self.shortcuts["home"]))
        self.processEvents()
        # Select all columns from left to right
        for x in range(self.plot_graphics.height() - 1):
            column = self.plot_graphics.grab(QRect(x, 0, 1, self.plot_graphics.height()))
            if x < extremes_of_channel_35[0] - 1:
                self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")
            elif extremes_of_channel_35[0] <= x <= extremes_of_channel_35[1]:
                column.save("D:\\x.png")
                self.assertTrue(
                    Pixmap.has_color(column, channel_35.color.name()),
                    f"column {x} doesn't have {channel_35.name} color",
                )
            else:
                self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")

    def test_Plot_PlotGraphics_Shortcut_Key_Insert(self):
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

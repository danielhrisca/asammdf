#!/usr/bin/env python
from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget

from PySide6 import QtCore, QtGui, QtTest


class TestShortcuts(TestPlotWidget):
    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot", channels_indexes=(36, 37))
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()
        # Settings for cursor
        self.widget.set_cursor_options(False, False, 1, "#000000")
        # channels
        self.channel_36 = self.plot.channel_selection.topLevelItem(0)
        self.channel_37 = self.plot.channel_selection.topLevelItem(1)
        self.assertEqual(2, self.plot.channel_selection.topLevelItemCount())
        # Double-click on channels
        self.mouseDClick_WidgetItem(self.channel_36)
        self.mouseDClick_WidgetItem(self.channel_37)
        self.processEvents()
        # Remove dots
        if self.plot.plot.with_dots:
            self.plot.plot.set_dots(False)
        self.processEvents()
        # pixmap is not black
        self.assertFalse(Pixmap.is_black(self.plot.plot.viewport().grab()))

    def test_Plot_Plot_Shortcut_Key_LeftRight(self):
        """
        Test Scope:
            Check that Arrow Keys: Left & Right ensure navigation on channels evolution.
            Ensure that navigation is working.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
            - Drag and Drop channels from FileWidget.channels_tree to self.plot.channels_selection:
                # First
                - ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
                # Second
                - ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            - Send KeyClick Right 5 times
            - Send KeyClick Left 4 times
        Evaluate:
            - Evaluate values from `Value` column on self.plot.channels_selection
            - Evaluate timestamp label
        """
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        # Case 0:
        with self.subTest("test_Plot_Plot_Shortcut_Key_LeftRight_0"):
            # Select channel: ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            self.mouseClick_WidgetItem(self.channel_37)
            self.plot.plot.setFocus()
            self.processEvents(0.1)

            self.assertEqual("25", self.channel_36.text(self.Column.VALUE))
            self.assertEqual("244", self.channel_37.text(self.Column.VALUE))

            # Send Key strokes
            for _ in range(6):
                QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key_Right)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("8", self.channel_36.text(self.Column.VALUE))
            self.assertEqual("6", self.channel_37.text(self.Column.VALUE))
            self.assertEqual("t = 0.082657s", self.plot.cursor_info.text())

            # Send Key strokes
            for _ in range(5):
                QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key_Left)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("21", self.channel_36.text(self.Column.VALUE))
            self.assertEqual("247", self.channel_37.text(self.Column.VALUE))
            self.assertEqual("t = 0.032657s", self.plot.cursor_info.text())

        # Case 1:
        with self.subTest("test_Plot_Plot_Shortcut_Key_LeftRight_1"):
            # Select channel: ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            self.mouseClick_WidgetItem(self.channel_37)
            self.plot.plot.setFocus()
            self.processEvents(0.1)

            # Send Key strokes
            for _ in range(6):
                QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key_Right)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("5", self.channel_36.text(self.Column.VALUE))
            self.assertEqual("9", self.channel_37.text(self.Column.VALUE))
            self.assertEqual("t = 0.092657s", self.plot.cursor_info.text())

            # Send Key strokes
            for _ in range(5):
                QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key_Left)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("18", self.channel_36.text(self.Column.VALUE))
            self.assertEqual("250", self.channel_37.text(self.Column.VALUE))
            self.assertEqual("t = 0.042657s", self.plot.cursor_info.text())

    def test_Plot_Plot_Shortcut_Key_R(self):
        """
        Test Scope:
            Check if Range Selection rectangle is painted over the self.plot.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Press PushButton "Create Window"
            - Press PushButton HideAxis (easy for evaluation)
            - Press Key R for range selection
            - Move Cursors
            - Press Key R for range selection
        Evaluate:
            - Evaluate that two cursors are available
            - Evaluate that new rectangle with different color is present
            - Evaluate that sum of rectangle areas is same with the one when plot is full black.
            - Evaluate that range selection disappear.
        """
        # Delete channels
        self.mouseClick_WidgetItem(self.channel_36)
        QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key_Delete)
        self.mouseClick_WidgetItem(self.channel_37)
        QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key_Delete)
        # Press PushButton "Hide axis"
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.LeftButton)

        # Save PixMap of clear plot
        clear_pixmap = self.plot.plot.viewport().grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

        # Get X position of Cursor
        cursors = Pixmap.cursors_x(clear_pixmap)
        # Evaluate that there is only one cursor
        self.assertEqual(1, len(cursors))

        # Press Key 'R' for range selection
        QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key_R)
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.plot.plot.viewport().grab()
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
        QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key_Right)
        self.processEvents(timeout=0.01)
        QtTest.QTest.keySequence(self.plot.plot, QtGui.QKeySequence("Ctrl+Left"))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = self.plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(range_pixmap))

        # Get X position of Cursors
        new_cursors = Pixmap.cursors_x(range_pixmap)
        # Evaluate that two cursors are available
        self.assertEqual(2, len(cursors))
        for c in cursors:
            self.assertNotIn(c, new_cursors)

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
        QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key_R)
        self.processEvents(timeout=0.01)

        # Save PixMap of clear plot
        clear_pixmap = self.plot.plot.viewport().grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

    def test_Plot_Plot_Shortcut_Key_F(self):
        """
        Test Scope:
            Check if 2 plotted signals is fitted on Y ax after pressing key "F".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 2 signals and create a plot
            - Press Key "S" to separate signals (precondition)
            - Press Key "F"
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black
            - Evaluate that after pressing key "S":
                > first signal is not displayed in top half of plot
                > second signal is not displayed in bottom half of plot
            - Evaluate that both colors of signals is in from top to bottom lines of plot after pressing key "F"
        """
        # check if grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)

        # Press "S"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        self.processEvents()
        # Create a color map for plot
        colorMap = Pixmap.color_map(self.plot.plot.viewport().grab())
        self.assertTrue(colorMap, "Failed to create color map")
        # Evaluate if signal was separated
        for y in range(self.plot.plot.height()):
            if y < self.plot.plot.height() / 2:
                self.assertFalse(
                    colorMap[y].count(self.channel_37.color.name()),
                    f"Line {y} contain {self.channel_37.name} color",
                )
            else:
                self.assertFalse(
                    colorMap[y].count(self.channel_36.color.name()),
                    f"Line {y} contain{self.channel_36.name} color",
                )

        # Press "F"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_F)
        self.processEvents()
        # Create a color map for plot
        colorMap = Pixmap.color_map(self.plot.plot.viewport().grab())
        self.assertTrue(colorMap, "Failed to create color map")
        # First and last line is black
        for y in range(1, self.plot.plot.height() - 1):
            self.assertTrue(
                colorMap[y].count(self.channel_36.color.name()),
                f"Line {y} doesn't contain {self.channel_36.name} color",
            )
            self.assertTrue(
                colorMap[y].count(self.channel_37.color.name()),
                f"Line {y} doesn't contain {self.channel_37.name} color",
            )

    def test_Plot_Plot_Shortcut_Shift_F(self):
        """
        Check if selected signal is fitted on Y ax after pressing "Shift+F".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 2 measurement and create a plot
            - Press Key "S"
            - Select first channel
            - Press "Shift+F"
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black
            - Evaluate that signals is separated after pressing key "S"
            - Evaluate that only first signal is fitted by Y ax of plot after pressing "Shift+F"
        """
        # check if grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)

        # Press "S"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        self.processEvents()
        # Create a color map for plot
        colorMap = Pixmap.color_map(self.plot.plot.viewport().grab())
        self.assertTrue(colorMap, "Failed to create color map")
        # Evaluate if signal was separated
        for y in range(self.plot.plot.height()):
            if y < self.plot.plot.height() / 2:
                self.assertFalse(
                    colorMap[y].count(self.channel_37.color.name()),
                    f"Line {y} contain {self.channel_37.name} color",
                )
            else:
                self.assertFalse(
                    colorMap[y].count(self.channel_36.color.name()),
                    f"Line {y} contain{self.channel_36.name} color",
                )
        # Select first signal
        self.mouseClick_WidgetItem(self.channel_36)
        # Press "Shift+F"
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Shift+F"))
        for i in range(100):
            self.processEvents()
        # Create a color map for plot
        colorMap = Pixmap.color_map(self.plot.plot.viewport().grab())
        self.assertTrue(colorMap, "Failed to create color map")
        for y in range(1, self.plot.plot.height() - 1):
            self.assertTrue(
                colorMap[y].count(self.channel_36.color.name()),
                f"Line {y} doesn't contain {self.channel_36.name} color",
            )
            if y < self.plot.plot.height() / 2:
                self.assertFalse(
                    colorMap[y].count(self.channel_37.color.name()),
                    f"Line {y} doesn't contain {self.channel_37.name} color",
                )

    def test_Plot_Plot_Shortcut_Key_G(self):
        """
        Check if grid is created properly after pressing key "G".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Create a plot window
            - If axes is hidden - press "Show axes" button
            - Press Key "G" 3 times
        Evaluate:
            - Evaluate that window is created
            - Evaluate that grid is displayed in order after pressing key "G":
                1. Is only X axes grid
                2. Is X and Y axes grid
                3. There is no grid
        """
        # check if grid is available
        if self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        # case 1: X and Y axes is hidden
        if not self.plot.plot.x_axis.grid and not self.plot.plot.y_axis.grid:
            with self.subTest("test_shortcut_key_G_no_grid_displayed"):
                # press key "G"
                QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_G)
                self.processEvents()
                self.assertTrue(self.plot.plot.x_axis.grid)
                self.assertFalse(self.plot.plot.y_axis.grid)
                # press key "G"
                QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_G)
                self.processEvents()
                self.assertTrue(self.plot.plot.x_axis.grid)
                self.assertTrue(self.plot.plot.y_axis.grid)
                # press key "G"
                QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_G)
                self.processEvents()
                self.assertFalse(self.plot.plot.x_axis.grid)
                self.assertFalse(self.plot.plot.y_axis.grid)
        # case 2: X is visible, Y is hidden
        elif self.plot.plot.x_axis.grid and not self.plot.plot.y_axis.grid:
            with self.subTest("test_shortcut_key_G_X_grid_already_displayed"):
                # press key "G"
                QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_G)
                self.processEvents()
                self.assertTrue(self.plot.plot.x_axis.grid)
                self.assertTrue(self.plot.plot.y_axis.grid)
                # press key "G"
                QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_G)
                self.processEvents()
                self.assertFalse(self.plot.plot.x_axis.grid)
                self.assertFalse(self.plot.plot.y_axis.grid)
                # press key "G"
                QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_G)
                self.processEvents()
                self.assertTrue(self.plot.plot.x_axis.grid)
                self.assertFalse(self.plot.plot.y_axis.grid)
        # case 3: X and Y axes is visible
        else:
            with self.subTest("test_shortcut_key_G_XU_grid_already_displayed"):
                # press key "G"
                QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_G)
                self.processEvents()
                self.assertTrue(self.plot.plot.x_axis.grid)
                self.assertFalse(self.plot.plot.y_axis.grid)
                # press key "G"
                QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_G)
                self.processEvents()
                self.assertTrue(self.plot.plot.x_axis.grid)
                self.assertTrue(self.plot.plot.y_axis.grid)
                # press key "G"
                QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_G)
                self.processEvents()
                self.assertFalse(self.plot.plot.x_axis.grid)
                self.assertFalse(self.plot.plot.y_axis.grid)

    def test_Plot_Plot_Shortcut_Key_W(self):
        """
        Check if signal is fitted properly after pressing key "W".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Create a plot window and load 2 signals
            - Press key "S" and "I"
            - Press key "W"
        Evaluate:
            - Evaluate that window is created
            - Evaluate that there are at least one column with first signal color
            - Evaluate first and last columns where is first signal:
                > first column after pressing "S" and "I" is full black => signal colors are not there
                > signal is zoomed => is extended to left side => in last column is both signals colors
            - Evaluate that after pressing key "W" in first and last column is displayed both signals
        """
        # check if grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)

        # search first and last column where is displayed first signal
        firstColoredColumn = None
        lastColoredColumn = None
        for x in range(self.plot.plot.width()):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(x, 0, 1, self.plot.plot.viewport().height())),
                self.channel_36.color.name(),
            ):
                firstColoredColumn = x
                break
        # Evaluate that there are at least one column with signal color
        self.assertTrue(firstColoredColumn)
        for x in range(self.plot.plot.width(), firstColoredColumn, -1):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(x, 0, 1, self.plot.plot.viewport().height())),
                self.channel_36.color.name(),
            ):
                lastColoredColumn = x
                break
        # Press "S" and "I"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_I)
        self.processEvents()

        # save left and right pixel column
        xLeftColumn = self.plot.plot.viewport().grab(QtCore.QRect(firstColoredColumn, 0, 1, self.plot.plot.height()))
        xRightColumn = self.plot.plot.viewport().grab(QtCore.QRect(lastColoredColumn, 0, 1, self.plot.plot.height()))
        self.assertFalse(Pixmap.has_color(xLeftColumn, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(xLeftColumn, self.channel_37.color.name()))
        self.assertTrue(Pixmap.has_color(xRightColumn, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(xRightColumn, self.channel_37.color.name()))
        # Press "W"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_W)
        self.processEvents()
        # Save left and right pixel column
        xLeftColumn = self.plot.plot.viewport().grab(QtCore.QRect(firstColoredColumn, 0, 1, self.plot.plot.height()))
        xRightColumn = self.plot.plot.viewport().grab(QtCore.QRect(lastColoredColumn, 0, 1, self.plot.plot.height()))
        self.assertTrue(Pixmap.has_color(xLeftColumn, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(xLeftColumn, self.channel_37.color.name()))
        self.assertTrue(Pixmap.has_color(xRightColumn, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(xRightColumn, self.channel_37.color.name()))

    def test_Plot_Plot_Shortcut_Key_S_2_Signals(self):
        """
        Test Scope:
            Check if 2 plotted signals is separated on Y ax after pressing key "S".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 2 signals and create a plot
            - Press Key "F" to fit signals (precondition)
            - Press Key "S"
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black
            - Evaluate that the color of both signals is displayed on midd line of plot after pressing key "F"
            - Evaluate that signals are separated in top and bottom half of plot after pressing key "S"
                > top, midd and bottom lines is empty
                > top half contain only first signal
                > bottom half contain only second signal
        """
        # check if grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)

        # Press "F"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_F)
        self.processEvents()
        # Create a color map for plot
        colorMap = Pixmap.color_map(self.plot.plot.viewport().grab())
        self.assertTrue(colorMap, "Failed to create color map")
        # Evaluate if signal was fitted (first and last line is black)
        for y in range(1, self.plot.plot.height() - 1):
            self.assertTrue(
                colorMap[y].count(self.channel_37.color.name()),
                f"Line {y} doesn't contain {self.channel_37.name} color",
            )
            self.assertTrue(
                colorMap[y].count(self.channel_36.color.name()),
                f"Line {y} doesn't contain {self.channel_36.name} color",
            )

        # Press "S"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        self.processEvents()
        # Create a color map for plot
        colorMap = Pixmap.color_map(self.plot.plot.viewport().grab())
        self.assertTrue(colorMap, "Failed to create color map")
        # Search top and bottom
        topOfChannel_36 = None
        for y in range(self.plot.plot.height()):
            if colorMap[y].count(self.channel_36.color.name()):
                topOfChannel_36 = y
                break
        self.assertTrue(topOfChannel_36)
        bottomOfChannel_36 = None
        for y in range(int(self.plot.plot.height() / 2), topOfChannel_36, -1):
            if colorMap[y].count(self.channel_36.color.name()):
                bottomOfChannel_36 = y
                break
        topOfChannel_37 = None
        for y in range(int(self.plot.plot.height() / 2), self.plot.plot.height(), 1):
            if colorMap[y].count(self.channel_37.color.name()):
                topOfChannel_37 = y
                break
        self.assertTrue(topOfChannel_37)
        bottomOfChannel_37 = None
        for y in range(self.plot.plot.height() - 1, topOfChannel_37, -1):
            if colorMap[y].count(self.channel_37.color.name()):
                bottomOfChannel_37 = y
                break
        # Evaluate plot
        self.assertNotEqual(bottomOfChannel_36, topOfChannel_37, "The signals is intersected")
        for key in colorMap.keys():
            if key < topOfChannel_36:
                self.assertEqual(
                    colorMap[key].count("#000000"), self.plot.plot.width(), f"Top line {key} of plot is not only black"
                )
            elif topOfChannel_36 <= key <= bottomOfChannel_36:
                self.assertTrue(
                    colorMap[key].count(self.channel_36.color.name()),
                    f"In line {key} of plot was not found color of {self.channel_36.name}",
                )
                self.assertFalse(
                    colorMap[key].count(self.channel_37.color.name()),
                    f"In line {key} of plot was found color of {self.channel_37.name}",
                )
            elif bottomOfChannel_36 < key < topOfChannel_37:
                self.assertEqual(
                    colorMap[key].count("#000000"), self.plot.plot.width(), f"Midd line {key} of plot is not only black"
                )
            elif topOfChannel_37 <= key <= bottomOfChannel_37:
                self.assertFalse(
                    colorMap[key].count(self.channel_36.color.name()),
                    f"In line {key} of plot was found color of {self.channel_36.name}",
                )
                self.assertTrue(
                    colorMap[key].count(self.channel_37.color.name()),
                    f"In line {key} of plot was not found color of {self.channel_37.name}",
                )
            else:
                self.assertEqual(
                    colorMap[key].count("#000000"),
                    self.plot.plot.width(),
                    f"Bottom line {key} of plot is not only black",
                )

    def test_Plot_Plot_Shortcut_Key_S_3_Signals(self):
        """
        Test Scope:
            Check if 3 plotted signals is separated on Y ax after pressing key "S".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a plot
            - Press Key "F" to fit signals (precondition)
            - Press Key "S"
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black
            - Evaluate that the color of both signals is displayed on midd line of plot after pressing key "F"
            - Evaluate that signals are separated in top and bottom third of plot after pressing key "S"
                > midd line is empty
                > top third contain only first signal
                > midd third contain only second signal
                > bottom third contain only third signal
        """

        # close all subWindows
        self.widget.mdi_area.closeAllSubWindows()
        # Open plot with 3 channels
        self.create_window(window_type="Plot", channels_indexes=(35, 36, 37))
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()
        channel_35 = self.plot.channel_selection.topLevelItem(0)
        self.channel_36 = self.plot.channel_selection.topLevelItem(1)
        self.channel_37 = self.plot.channel_selection.topLevelItem(2)
        self.assertEqual(3, self.plot.channel_selection.topLevelItemCount())
        # Double-click on channels
        self.mouseDClick_WidgetItem(channel_35)
        self.mouseDClick_WidgetItem(self.channel_36)
        self.mouseDClick_WidgetItem(self.channel_37)
        # check if grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        # Remove dots
        if self.plot.plot.with_dots:
            self.plot.plot.set_dots(False)
        self.processEvents()
        # pixmap is not black
        self.assertFalse(Pixmap.is_black(self.plot.plot.viewport().grab()))
        # Press "F"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_F)
        self.processEvents()
        # Create a color map for plot
        colorMap = Pixmap.color_map(self.plot.plot.viewport().grab())
        self.assertTrue(colorMap, "Failed to create color map")
        # Evaluate if signal was fitted (first and last line is black)
        for y in range(1, self.plot.plot.height() - 1):
            self.assertTrue(
                colorMap[y].count(self.channel_37.color.name()),
                f"Line {y} doesn't contain {self.channel_37.name} color",
            )
            self.assertTrue(
                colorMap[y].count(self.channel_36.color.name()),
                f"Line {y} doesn't contain {self.channel_36.name} color",
            )
            self.assertTrue(
                colorMap[y].count(channel_35.color.name()),
                f"Line {y} doesn't contain {channel_35.name} color",
            )

        # Press "S"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        self.processEvents()
        # Create a color map for plot
        colorMap = Pixmap.color_map(self.plot.plot.viewport().grab())
        self.assertTrue(colorMap, "Failed to create color map")
        # Search top and bottom of channels on plot
        topOfChannel_35 = None
        for y in range(self.plot.plot.height()):
            if colorMap[y].count(channel_35.color.name()):
                topOfChannel_35 = y
                break
        self.assertTrue(topOfChannel_35)
        bottomOfChannel_35 = None
        for y in range(int(self.plot.plot.height() / 3), topOfChannel_35, -1):
            if colorMap[y].count(channel_35.color.name()):
                bottomOfChannel_35 = y
                break
        topOfChannel_36 = None
        for y in range(int(self.plot.plot.height() / 3), int(self.plot.plot.height() / 3) * 2):
            if colorMap[y].count(self.channel_36.color.name()):
                topOfChannel_36 = y
                break
        self.assertTrue(topOfChannel_36)
        bottomOfChannel_36 = None
        for y in range(int(self.plot.plot.height() / 3) * 2, topOfChannel_36, -1):
            if colorMap[y].count(self.channel_36.color.name()):
                bottomOfChannel_36 = y
                break
        topOfChannel_37 = None
        for y in range(int(self.plot.plot.height() / 3) * 2, self.plot.plot.height(), 1):
            if colorMap[y].count(self.channel_37.color.name()):
                topOfChannel_37 = y
                break
        self.assertTrue(topOfChannel_37)
        bottomOfChannel_37 = None
        for y in range(self.plot.plot.height() - 1, topOfChannel_37, -1):
            if colorMap[y].count(self.channel_37.color.name()):
                bottomOfChannel_37 = y
                break
        # Evaluate plot
        self.assertNotEqual(bottomOfChannel_36, topOfChannel_37, "The signals is intersected")
        for key in colorMap.keys():
            if key < topOfChannel_35:
                self.assertEqual(
                    colorMap[key].count("#000000"), self.plot.plot.width(), f"Top line {key} of plot is not only black"
                )
            elif topOfChannel_35 <= key <= bottomOfChannel_35:
                self.assertTrue(
                    colorMap[key].count(channel_35.color.name()),
                    f"In line {key} of plot was not found color of {channel_35.name}",
                )
                self.assertFalse(
                    colorMap[key].count(self.channel_36.color.name()),
                    f"In line {key} of plot was found color of {self.channel_36.name}",
                )
                self.assertFalse(
                    colorMap[key].count(self.channel_37.color.name()),
                    f"In line {key} of plot was found color of {self.channel_37.name}",
                )
            elif bottomOfChannel_35 < key < topOfChannel_36:
                self.assertEqual(
                    colorMap[key].count("#000000"), self.plot.plot.width(), f"Top line {key} of plot is not only black"
                )
            elif topOfChannel_36 <= key <= bottomOfChannel_36:
                self.assertFalse(
                    colorMap[key].count(channel_35.color.name()),
                    f"In line {key} of plot was found color of {channel_35.name}",
                )
                self.assertTrue(
                    colorMap[key].count(self.channel_36.color.name()),
                    f"In line {key} of plot was not found color of {self.channel_36.name}",
                )
                self.assertFalse(
                    colorMap[key].count(self.channel_37.color.name()),
                    f"In line {key} of plot was found color of {self.channel_37.name}",
                )
            elif bottomOfChannel_36 < key < topOfChannel_37:
                self.assertEqual(
                    colorMap[key].count("#000000"), self.plot.plot.width(), f"Midd line {key} of plot is not only black"
                )
            elif topOfChannel_37 <= key <= bottomOfChannel_37:
                self.assertFalse(
                    colorMap[key].count(channel_35.color.name()),
                    f"In line {key} of plot was found color of {channel_35.name}",
                )
                self.assertFalse(
                    colorMap[key].count(self.channel_36.color.name()),
                    f"In line {key} of plot was found color of {self.channel_36.name}",
                )
                self.assertTrue(
                    colorMap[key].count(self.channel_37.color.name()),
                    f"In line {key} of plot was not found color of {self.channel_37.name}",
                )
            else:
                self.assertEqual(
                    colorMap[key].count("#000000"),
                    self.plot.plot.width(),
                    f"Bottom line {key} of plot is not only black",
                )

    def test_Plot_Plot_Shortcut_Shift_S(self):
        """
        Test Scope:
            Check if 2 plotted signals is separated on Y ax after pressing "Shift+S".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 2 signals and create a plot
            - Press Key "F" to fit signals (precondition)
            - Press Key "S" to separate signals (precondition)
            - Select first signal
            - Press combination "Shift+S"
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black
            - Evaluate that the color of signals is displayed on top and bottom line of plot after pressing key "F"
            - Evaluate that signals are separated in top and bottom half of plot after pressing key "S"
            - Evaluate that first signal is in both half's of plot and second is ony in bottom half of plot
        """
        # Press "F"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_F)
        self.processEvents()
        # Select midd line
        halfOfY = int(self.plot.plot.height() / 2)  # halfOfY of Y ax
        yMiddLine = self.plot.plot.viewport().grab(QtCore.QRect(0, halfOfY, self.plot.plot.viewport().width(), 1))
        self.assertTrue(Pixmap.has_color(yMiddLine, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(yMiddLine, self.channel_37.color.name()))

        # Press "S"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        self.processEvents()
        # Select midd line
        yMiddLine = self.plot.plot.viewport().grab(QtCore.QRect(0, halfOfY, self.plot.plot.viewport().width(), 1))
        self.assertFalse(Pixmap.has_color(yMiddLine, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yMiddLine, self.channel_37.color.name()))
        # Select first signal and pres "Shift+S"
        self.mouseClick_WidgetItem(self.channel_36)
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Shift+S"))
        for i in range(52):
            self.processEvents()
        # Select midd line
        yTopLine = self.plot.plot.viewport().grab(
            QtCore.QRect(0, halfOfY - int(halfOfY / 2), self.plot.plot.viewport().width(), 1)
        )
        yMiddLine = self.plot.plot.viewport().grab(QtCore.QRect(0, halfOfY, self.plot.plot.viewport().width(), 1))
        yBottomLine = self.plot.plot.viewport().grab(
            QtCore.QRect(0, halfOfY + int(halfOfY / 2), self.plot.plot.viewport().width(), 1)
        )
        self.assertTrue(Pixmap.has_color(yTopLine, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yTopLine, self.channel_37.color.name()))
        self.assertTrue(Pixmap.has_color(yMiddLine, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yMiddLine, self.channel_37.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomLine, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomLine, self.channel_37.color.name()))

    def test_Plot_Plot_Shortcut_I(self):
        """
        Test Scope:
            Check if zooming is released after pressing key "I".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Press combination "Shift+S"
        Evaluate:
            - Evaluate that plot is not black
            - Evaluate that distance between first and second transition of signal in the same line is increased
                after pressing key "I"
        """
        self.mouseDClick_WidgetItem(self.channel_37)
        self.processEvents()
        # Select line
        yMiddLine = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        colorMap = Pixmap.color_map(yMiddLine)
        distanceInPixels_1 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(colorMap[0]):
            if x == self.channel_36.color.name():
                distanceInPixels_1 = i - distanceInPixels_1
                if distanceInPixels_1 != i:
                    break
        self.assertTrue(distanceInPixels_1)
        # Press "I"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_I)
        self.processEvents()
        # Select line
        yMiddLine = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        colorMap = Pixmap.color_map(yMiddLine)
        distanceInPixels_2 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(colorMap[0]):
            if x == self.channel_36.color.name():
                distanceInPixels_2 = i - distanceInPixels_2
                if distanceInPixels_2 != i:
                    break
        self.assertLess(distanceInPixels_1, distanceInPixels_2)

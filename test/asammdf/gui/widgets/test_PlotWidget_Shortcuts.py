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
        cls.PlotOffset = 5

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
        # Test if plot is not only black
        pixmap = self.plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap))

        # Press "S"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        for y in range(self.plot.plot.height()):
            if y < self.plot.plot.height() / 2:
                self.assertFalse(
                    Pixmap.has_color(
                        self.plot.plot.viewport().grab(QtCore.QRect(0, y, self.plot.plot.viewport().width(), 1)),
                        self.channel_37.color.name(),
                    ),
                    f"Line {y} doesn't contain {self.channel_37.name} color",
                )
            else:
                self.assertFalse(
                    Pixmap.has_color(
                        self.plot.plot.viewport().grab(QtCore.QRect(0, y, self.plot.plot.viewport().width(), 1)),
                        self.channel_36.color.name(),
                    ),
                    f"Line {y} doesn't contain{self.channel_36.name} color",
                )

        # Press "F"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_F)
        self.processEvents()
        for y in range(1, self.plot.plot.height() - 1):
            self.assertTrue(
                Pixmap.has_color(
                    self.plot.plot.viewport().grab(QtCore.QRect(0, y, self.plot.plot.viewport().width(), 1)),
                    self.channel_36.color.name(),
                ),
                f"Line {y} contain {self.channel_36.name} color",
            )
            self.assertTrue(
                Pixmap.has_color(
                    self.plot.plot.viewport().grab(QtCore.QRect(0, y, self.plot.plot.viewport().width(), 1)),
                    self.channel_37.color.name(),
                ),
                f"Line {y} contain {self.channel_37.name} color",
            )

    def test_Plot_Plot_Shortcut_Shift_F(self):
        """
        Check if selected signal is fitted on Y ax after pressing "Shift+F".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 2 measurement and create a plot
            - Press Key "S"
            - Press "Shift+F"
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black
            - Evaluate that the color of signals is not displayed on top and bottom line of plot after pressing key "S"
            - Evaluate that only color of first signal is displayed on top and bottom line of plot
                    after pressing "Shift+F"
        """
        # check if grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)

        # Evaluate that plot is not only black
        pixmap = self.plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap))

        # Press "S"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        for y in range(self.plot.plot.height()):
            if y < self.plot.plot.height() / 2:
                self.assertFalse(
                    Pixmap.has_color(
                        self.plot.plot.viewport().grab(QtCore.QRect(0, y, self.plot.plot.viewport().width(), 1)),
                        self.channel_37.color.name(),
                    ),
                    f"Line {y} doesn't contain {self.channel_37.name} color",
                )
            else:
                self.assertFalse(
                    Pixmap.has_color(
                        self.plot.plot.viewport().grab(QtCore.QRect(0, y, self.plot.plot.viewport().width(), 1)),
                        self.channel_36.color.name(),
                    ),
                    f"Line {y} doesn't contain{self.channel_36.name} color",
                )
        # Select first signal
        self.mouseClick_WidgetItem(self.channel_36)
        # Press "Shift+F"
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Shift+F"))
        for i in range(100):
            self.processEvents()
        for y in range(1, self.plot.plot.height() - 1):
            self.assertTrue(
                Pixmap.has_color(
                    self.plot.plot.viewport().grab(QtCore.QRect(0, y, self.plot.plot.viewport().width(), 1)),
                    self.channel_36.color.name(),
                ),
                f"Line {y} contain {self.channel_36.name} color",
            )
            if y < self.plot.plot.height() / 2:
                self.assertFalse(
                    Pixmap.has_color(
                        self.plot.plot.viewport().grab(QtCore.QRect(0, y, self.plot.plot.viewport().width(), 1)),
                        self.channel_37.color.name(),
                    ),
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
        for x in range(self.plot.plot.viewport().width()):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(x, 0, 1, self.plot.plot.viewport().height())),
                self.channel_36.color.name(),
            ):
                firstColoredColumn = x
                break
        # Evaluate that there are at least one column with signal color
        self.assertTrue(firstColoredColumn)
        for x in range(self.plot.plot.viewport().width(), firstColoredColumn, -1):
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
        # save pixmap
        pixmap = self.plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap))
        # save left and right pixel column
        xLeftColumn = self.plot.plot.viewport().grab(
            QtCore.QRect(firstColoredColumn, 0, 1, self.plot.plot.viewport().height())
        )
        xRightColumn = self.plot.plot.viewport().grab(
            QtCore.QRect(lastColoredColumn, 0, 1, self.plot.plot.viewport().height())
        )
        self.assertFalse(Pixmap.has_color(xLeftColumn, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(xLeftColumn, self.channel_37.color.name()))
        self.assertTrue(Pixmap.has_color(xRightColumn, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(xRightColumn, self.channel_37.color.name()))
        # Press "W"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_W)
        self.processEvents()
        # Save left and right pixel column
        xLeftColumn = self.plot.plot.viewport().grab(
            QtCore.QRect(firstColoredColumn, 0, 1, self.plot.plot.viewport().height())
        )
        xRightColumn = self.plot.plot.viewport().grab(
            QtCore.QRect(lastColoredColumn, 0, 1, self.plot.plot.viewport().height())
        )
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
                > midd line iis empty
                > top half contain only first signal
                > bottom half contain only second signal
        """
        # check if grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)

        # save pixmap
        pixmap = self.plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap))
        # Press "F"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_F)
        self.processEvents()
        # Save Top and Bottom pixel line of plot
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
        # Select First and Last line for first signal
        firstColoredLine_channel_36 = None
        lastColoredLine_channel_36 = None
        for column in range(halfOfY):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(0, column, self.plot.plot.viewport().width(), 1)),
                self.channel_36.color.name(),
            ):
                firstColoredLine_channel_36 = column
                break
        # Evaluate that there are at least one column with signal color
        self.assertTrue(firstColoredLine_channel_36)
        for column in range(halfOfY, firstColoredLine_channel_36, -1):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(0, column, self.plot.plot.viewport().width(), 1)),
                self.channel_36.color.name(),
            ):
                lastColoredLine_channel_36 = column
                break
        yTopLine_channel_36 = self.plot.plot.viewport().grab(
            QtCore.QRect(0, firstColoredLine_channel_36, self.plot.plot.viewport().width(), 1)
        )
        yBottomLine_channel_36 = self.plot.plot.viewport().grab(
            QtCore.QRect(0, lastColoredLine_channel_36, self.plot.plot.viewport().width(), 1)
        )
        self.assertTrue(Pixmap.has_color(yTopLine_channel_36, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yTopLine_channel_36, self.channel_37.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomLine_channel_36, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine_channel_36, self.channel_37.color.name()))
        # Select First and Last line for second signal
        firstColoredLine_channel_37 = None
        lastColoredLine_channel_37 = None
        for column in range(halfOfY, self.plot.plot.viewport().height(), 1):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(0, column, self.plot.plot.viewport().width(), 1)),
                self.channel_37.color.name(),
            ):
                firstColoredLine_channel_37 = column
                break
        # Evaluate that there are at least one column with signal color
        self.assertTrue(firstColoredLine_channel_37)
        for column in range(firstColoredLine_channel_37, halfOfY, -1):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(0, column, self.plot.plot.viewport().width(), 1)),
                self.channel_37.color.name(),
            ):
                lastColoredLine_channel_37 = column
                break
        yTopLine_channel_37 = self.plot.plot.viewport().grab(
            QtCore.QRect(0, firstColoredLine_channel_37, self.plot.plot.viewport().width(), 1)
        )
        yBottomLine_channel_37 = self.plot.plot.viewport().grab(
            QtCore.QRect(0, lastColoredLine_channel_37, self.plot.plot.viewport().width(), 1)
        )
        self.assertFalse(Pixmap.has_color(yTopLine_channel_37, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(yTopLine_channel_37, self.channel_37.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine_channel_37, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomLine_channel_37, self.channel_37.color.name()))

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

        # save pixmap
        pixmap = self.plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap))
        # Press "F"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_F)
        self.processEvents()
        # Save Top and Bottom pixel line of plot
        thirdOfY = int(self.plot.plot.height() / 3)  # third of Y ax
        yTopThirdLine = self.plot.plot.viewport().grab(QtCore.QRect(0, thirdOfY, self.plot.plot.viewport().width(), 1))
        yBottomThirdLine = self.plot.plot.viewport().grab(
            QtCore.QRect(0, thirdOfY * 2, self.plot.plot.viewport().width(), 1)
        )
        self.assertTrue(Pixmap.has_color(yTopThirdLine, channel_35.color.name()))
        self.assertTrue(Pixmap.has_color(yTopThirdLine, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(yTopThirdLine, self.channel_37.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomThirdLine, channel_35.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomThirdLine, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomThirdLine, self.channel_37.color.name()))

        # Press "S"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        self.processEvents()
        # Select third lines
        yTopThirdLine = self.plot.plot.viewport().grab(QtCore.QRect(0, thirdOfY, self.plot.plot.viewport().width(), 1))
        yBottomThirdLine = self.plot.plot.viewport().grab(
            QtCore.QRect(0, thirdOfY * 2, self.plot.plot.viewport().width(), 1)
        )
        self.assertFalse(Pixmap.has_color(yTopThirdLine, channel_35.color.name()))
        self.assertFalse(Pixmap.has_color(yTopThirdLine, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yTopThirdLine, self.channel_37.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomThirdLine, channel_35.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomThirdLine, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomThirdLine, self.channel_37.color.name()))

        # Select First and Last line for first signal
        firstColoredLine_channel_35 = None
        lastColoredLine_channel_35 = None
        for column in range(thirdOfY):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(0, column, self.plot.plot.viewport().width(), 1)),
                channel_35.color.name(),
            ):
                firstColoredLine_channel_35 = column
                break
        # Evaluate that there are at least one column with signal color
        self.assertTrue(firstColoredLine_channel_35)
        for column in range(thirdOfY, firstColoredLine_channel_35, -1):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(0, column, self.plot.plot.viewport().width(), 1)),
                channel_35.color.name(),
            ):
                lastColoredLine_channel_35 = column
                break
        yTopLine_channel_35 = self.plot.plot.viewport().grab(
            QtCore.QRect(0, firstColoredLine_channel_35, self.plot.plot.viewport().width(), 1)
        )
        yBottomLine_channel_35 = self.plot.plot.viewport().grab(
            QtCore.QRect(0, lastColoredLine_channel_35, self.plot.plot.viewport().width(), 1)
        )
        self.assertTrue(Pixmap.has_color(yTopLine_channel_35, channel_35.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomLine_channel_35, channel_35.color.name()))
        self.assertFalse(Pixmap.has_color(yTopLine_channel_35, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine_channel_35, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yTopLine_channel_35, self.channel_37.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine_channel_35, self.channel_37.color.name()))

        # Select First and Last line for second signal
        firstColoredLine_channel_36 = None
        lastColoredLine_channel_36 = None
        for column in range(thirdOfY, thirdOfY * 2):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(0, column, self.plot.plot.viewport().width(), 1)),
                self.channel_36.color.name(),
            ):
                firstColoredLine_channel_36 = column
                break
        # Evaluate that there are at least one column with signal color
        self.assertTrue(firstColoredLine_channel_36)
        for column in range(thirdOfY * 2, thirdOfY, -1):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(0, column, self.plot.plot.viewport().width(), 1)),
                self.channel_36.color.name(),
            ):
                lastColoredLine_channel_36 = column
                break
        yTopLine_channel_36 = self.plot.plot.viewport().grab(
            QtCore.QRect(0, firstColoredLine_channel_36, self.plot.plot.viewport().width(), 1)
        )
        yBottomLine_channel_36 = self.plot.plot.viewport().grab(
            QtCore.QRect(0, lastColoredLine_channel_36, self.plot.plot.viewport().width(), 1)
        )
        self.assertFalse(Pixmap.has_color(yTopLine_channel_36, channel_35.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine_channel_36, channel_35.color.name()))
        self.assertTrue(Pixmap.has_color(yTopLine_channel_36, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomLine_channel_36, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yTopLine_channel_36, self.channel_37.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine_channel_36, self.channel_37.color.name()))

        # Select First and Last line for third signal
        firstColoredLine_channel_37 = None
        lastColoredLine_channel_37 = None
        for column in range(thirdOfY * 2, self.plot.plot.viewport().height()):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(0, column, self.plot.plot.viewport().width(), 1)),
                self.channel_37.color.name(),
            ):
                firstColoredLine_channel_37 = column
                break
        # Evaluate that there are at least one column with signal color
        self.assertTrue(firstColoredLine_channel_37)
        for column in range(self.plot.plot.viewport().height(), firstColoredLine_channel_37, -1):
            if Pixmap.has_color(
                self.plot.plot.viewport().grab(QtCore.QRect(0, column, self.plot.plot.viewport().width(), 1)),
                self.channel_37.color.name(),
            ):
                lastColoredLine_channel_37 = column
                break
        yTopLine_channel_37 = self.plot.plot.viewport().grab(
            QtCore.QRect(0, firstColoredLine_channel_37, self.plot.plot.viewport().width(), 1)
        )
        yBottomLine_channel_37 = self.plot.plot.viewport().grab(
            QtCore.QRect(0, lastColoredLine_channel_37, self.plot.plot.viewport().width(), 1)
        )
        self.assertFalse(Pixmap.has_color(yBottomLine_channel_37, channel_35.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine_channel_37, channel_35.color.name()))
        self.assertFalse(Pixmap.has_color(yTopLine_channel_37, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(yTopLine_channel_37, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(yTopLine_channel_37, self.channel_37.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomLine_channel_37, self.channel_37.color.name()))

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
        # save pixmap
        pixmap = self.plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap))
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

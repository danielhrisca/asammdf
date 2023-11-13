#!/usr/bin/env python
import os.path
from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget
from unittest import mock

from PySide6 import QtCore, QtGui, QtTest


class TestShortcutsWOChannels(TestPlotWidget):
    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()
        # check if grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        # hide bookmarks if it's available
        if self.plot.show_bookmarks:
            self.plot.toggle_bookmarks(hide=True)
        self.processEvents()
        # pixmap is black
        self.assertTrue(Pixmap.is_black(self.plot.plot.viewport().grab()))

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
        # uncheck channels

        # Press PushButton "Hide axis"
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.LeftButton)
        self.widget.set_cursor_options(False, False, 1, Pixmap.COLOR_CURSOR)
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
        QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key_R)
        self.processEvents(timeout=0.01)

        # Save PixMap of clear plot
        clear_pixmap = self.plot.plot.viewport().grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

    def test_Plot_Plot_Shortcut_Key_G(self):
        """
        Test Scope:
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

    def test_Plot_Plot_Shortcut_Key_M_WO_Ch_0(self):
        """
        Test Scope:
            Check if error will not raise after pressing key M on empty plot.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Press PushButton "Create Window"
            - Mock Error Dialog object
            - Press Key M
        Evaluate:
            - Evaluate that Error Dialog object was not called
        """
        # mock for ErrorDialog object
        with mock.patch("asammdf.gui.utils.ErrorDialog") as mo_ErrorDialog:
            QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_M)
        mo_ErrorDialog.assert_not_called()

    def test_Plot_Plot_Shortcut_Key_M_WO_Ch_1(self):
        """
        Test Scope:
            Check if widget "info" will change its visibility after pressing key M.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Press PushButton "Create Window"
            - Press Key M 2 times
        Evaluate:
            - Evaluate that "Info" widget is not visible by default
            - Evaluate that "Info" widget is visible after pressing key "M" first time
            - Evaluate that "Info" widget is not visible after pressing key "M" second time
        """
        self.assertFalse(self.plot.info.isVisible())
        # press key "M"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_M)
        self.assertTrue(self.plot.info.isVisible())
        # press key "M"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_M)
        self.assertFalse(self.plot.info.isVisible())

    def test_Plot_Plot_Shortcut_Key_Shift_L(self):
        """
        Test Scope:
            Check if by pressing the combination of keys "Shift+L", visibility of channel list is changed
        Events:
            - Open 'FileWidget' with valid measurement.
            - Press "Shift+L"
            - Press "Shift+L"
        Evaluate:
            - Evaluate that channel list is visible by default
            - Evaluate that channel list is hidden after pressing first time "Shift+L"
            - Evaluate that channel list is visible after pressing second time "Shift+L"
        """
        self.assertTrue(self.widget.channel_view.isVisible())
        # Press "Shift+L"
        QtTest.QTest.keySequence(self.widget, QtGui.QKeySequence("Shift+L"))
        self.assertFalse(self.widget.channel_view.isVisible())
        # Press "Shift+L"
        QtTest.QTest.keySequence(self.widget, QtGui.QKeySequence("Shift+L"))
        self.assertTrue(self.widget.channel_view.isVisible())

    def test_Plot_Plot_Shortcut_Shift_Key_C(self):
        """
        Test Scope:
            Check if only object cascadeSubWindows() from C module of PySide6.QtWidgets was called
                after pressing the combination of keys "Shift+C"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Mock objects from QMdiArea module
            - Press "Shift+C"
        Evaluate (0):
            - Evaluate that only cascadeSubWindows() object was called"
        """
        # mocks
        with (
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.cascadeSubWindows") as mo_cascadeSubWindows,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tileSubWindows") as mo_tileSubWindows,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tile_vertically") as mo_tile_vertically,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tile_horizontally") as mo_tile_horizontally,
        ):
            # Press Shift+C
            QtTest.QTest.keySequence(self.widget, QtGui.QKeySequence("Shift+C"))
        # Evaluate
        mo_cascadeSubWindows.assert_called()
        mo_tileSubWindows.assert_not_called()
        mo_tile_vertically.assert_not_called()
        mo_tile_horizontally.assert_not_called()

    def test_Plot_Plot_Shortcut_Shift_Key_T(self):
        """
        Test Scope:
            Check if only object tileSubWindows() from QMdiArea module of PySide6.QtWidgets was called
                after pressing the combination of keys "Shift+T"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Mock objects from QMdiArea module
            - Press "Shift+T"
        Evaluate (0):
            - Evaluate that only tileSubWindows() object was called"
        """
        # mocks
        with (
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.cascadeSubWindows") as mo_cascadeSubWindows,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tileSubWindows") as mo_tileSubWindows,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tile_vertically") as mo_tile_vertically,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tile_horizontally") as mo_tile_horizontally,
        ):
            # Press Shift+T
            QtTest.QTest.keySequence(self.widget, QtGui.QKeySequence("Shift+T"))
        # Evaluate
        mo_cascadeSubWindows.assert_not_called()
        mo_tileSubWindows.assert_called()
        mo_tile_vertically.assert_not_called()
        mo_tile_horizontally.assert_not_called()

    def test_Plot_Plot_Shortcut_Shift_Key_V(self):
        """
        Test Scope:
            Check if only object tile_vertically() from QMdiArea module of PySide6.QtWidgets was called
                after pressing the combination of keys "Shift+V"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Mock objects from QMdiArea module
            - Press "Shift+V"
        Evaluate (0):
            - Evaluate that only tile_vertically() object was called"
        """
        # mocks
        with (
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.cascadeSubWindows") as mo_cascadeSubWindows,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tileSubWindows") as mo_tileSubWindows,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tile_vertically") as mo_tile_vertically,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tile_horizontally") as mo_tile_horizontally,
        ):
            # Press Shift+V
            QtTest.QTest.keySequence(self.widget, QtGui.QKeySequence("Shift+V"))
        # Evaluate
        mo_cascadeSubWindows.assert_not_called()
        mo_tileSubWindows.assert_not_called()
        mo_tile_vertically.assert_called()
        mo_tile_horizontally.assert_not_called()

    def test_Plot_Plot_Shortcut_Shift_Key_H(self):
        """
        Test Scope:
            Check if only object tile_horizontally() from QMdiArea module of PySide6.QtWidgets was called
                after pressing the combination of keys "Shift+H"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Mock objects from QMdiArea module
            - Press "Shift+H"
        Evaluate (0):
            - Evaluate that only tile_horizontally() object was called"
        """
        # mocks
        with (
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.cascadeSubWindows") as mo_cascadeSubWindows,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tileSubWindows") as mo_tileSubWindows,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tile_vertically") as mo_tile_vertically,
            mock.patch("asammdf.gui.widgets.file.MdiAreaWidget.tile_horizontally") as mo_tile_horizontally,
        ):
            # Press Shift+H
            QtTest.QTest.keySequence(self.widget, QtGui.QKeySequence("Shift+H"))
        # Evaluate
        mo_cascadeSubWindows.assert_not_called()
        mo_tileSubWindows.assert_not_called()
        mo_tile_vertically.assert_not_called()
        mo_tile_horizontally.assert_called()

    def test_Plot_Plot_Shortcut_Shift_Key_Alt_F(self):
        """
        Test Scope:
            Check if sub-windows frame was toggled after pressing keys "Shift+Alt+F"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Press twice "Shift+Alt+F"
        Evaluate (0):
            - Evaluate that by default sub-windows are not frameless
            - Evaluate that sub-windows is frameless after pressing "Shift+Alt+F" first time
            - Evaluate that sub-windows is not frameless after pressing "Shift+Alt+F" second time
        """
        # Evaluate
        self.assertFalse(self.widget._frameless_windows)
        # Press Shift+Alt+F
        QtTest.QTest.keySequence(self.widget, QtGui.QKeySequence("Shift+Alt+F"))
        # Evaluate
        self.assertTrue(self.widget._frameless_windows)
        # Press Shift+Alt+F
        QtTest.QTest.keySequence(self.widget, QtGui.QKeySequence("Shift+Alt+F"))
        # Evaluate
        self.assertFalse(self.widget._frameless_windows)


class TestShortcutsWith_1_Channel(TestPlotWidget):
    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot", channels_indexes=(35,))
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()
        # Settings for cursor
        self.widget.set_cursor_options(False, False, 1, Pixmap.COLOR_BACKGROUND)
        # channels
        self.channel_35 = self.plot.channel_selection.topLevelItem(0)
        self.assertEqual(1, self.plot.channel_selection.topLevelItemCount())
        # Double-click on channels
        self.mouseDClick_WidgetItem(self.channel_35)
        self.processEvents()
        # Remove dots
        if self.plot.plot.with_dots:
            self.plot.plot.set_dots(False)
        self.processEvents()
        # check if grid is available -> hide grid
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        # hide bookmarks if it's available
        if self.plot.show_bookmarks:
            self.plot.toggle_bookmarks(hide=True)
        self.processEvents()
        # pixmap is not black
        self.assertFalse(Pixmap.is_black(self.plot.plot.viewport().grab()))

    def test_Plot_Plot_Shortcut_Key_W(self):
        """
        Check if signal is fitted properly after pressing key "W".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Create a plot window and load 2 signals
            - Press key "I"
            - Press key "W"
        Evaluate:
            - Evaluate that window is created
            - Evaluate that there are at least one column with first signal color
            - Evaluate first and last columns where is first signal:
                > first column after pressing "I" is full black => signal colors are not there
                > signal is zoomed => is extended to left side => last column contain signal color
            - Evaluate that after pressing key "W" from first to last column is displayed signal
        """
        # check if the grid is available
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)

        # search first and last column where is displayed first signal
        extremesOfChannel_35 = Pixmap.search_signal_from_to_x(
            self.plot.plot.viewport().grab(), self.channel_35.color.name()
        )
        # Evaluate that there are extremes of first signal
        self.assertTrue(extremesOfChannel_35)
        # Press "I"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_I)
        self.processEvents()

        # save left and right pixel column
        xLeftColumn = self.plot.plot.viewport().grab(
            QtCore.QRect(extremesOfChannel_35[0], 0, 1, self.plot.plot.height())
        )
        xRightColumn = self.plot.plot.viewport().grab(
            QtCore.QRect(extremesOfChannel_35[1], 0, 1, self.plot.plot.height())
        )
        self.assertTrue(Pixmap.is_black(xLeftColumn))
        self.assertTrue(Pixmap.has_color(xRightColumn, self.channel_35.color.name()))
        # Press "W"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_W)
        self.processEvents()
        # Select all columns from left to right
        for x in range(self.plot.plot.height() - 1):
            column = self.plot.plot.viewport().grab(QtCore.QRect(x, 0, 1, self.plot.plot.height()))
            if x < extremesOfChannel_35[0]:
                self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")
            elif extremesOfChannel_35[0] <= x <= extremesOfChannel_35[1]:
                self.assertTrue(
                    Pixmap.has_color(column, self.channel_35.color.name()),
                    f"column {x} doesn't have {self.channel_35.name} color",
                )
            else:
                self.assertTrue(Pixmap.is_black(column), f"column {x} is not black")

    def test_Plot_Plot_Shortcut_Key_I(self):
        """
        Test Scope:
            Check if zooming is released after pressing key "I".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Select signal
            - Press "I""
        Evaluate:
            - Evaluate that plot is not black
            - Evaluate that distance between first and second transition of signal in the same line is increased
                after pressing key "I"
        """
        # Select line
        yMiddLine = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        colorMap = Pixmap.color_map(yMiddLine)
        distanceInPixels_1 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(colorMap[0]):
            if x == self.channel_35.color.name():
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
            if x == self.channel_35.color.name():
                distanceInPixels_2 = i - distanceInPixels_2
                if distanceInPixels_2 != i:
                    break
        self.assertLess(distanceInPixels_1, distanceInPixels_2)

    def test_Plot_Plot_Shortcut_Key_O(self):
        """
        Test Scope:
            Check if zooming is released after pressing key "O".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Select signal
            - Press "O"
        Evaluate:
            - Evaluate that plot is not black
            - Evaluate that the distance between first and second transition of signal in the same line is decreased
                after pressing key "O"
        """
        # Select line
        yMiddLine = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        colorMap = Pixmap.color_map(yMiddLine)
        distanceInPixels_1 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(colorMap[0]):
            if x == self.channel_35.color.name():
                distanceInPixels_1 = i - distanceInPixels_1
                if distanceInPixels_1 != i:
                    break
        self.assertTrue(distanceInPixels_1)
        # Press "I"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_O)
        self.processEvents()
        # Select line
        yMiddLine = self.plot.plot.viewport().grab(
            QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.viewport().width(), 1)
        )
        colorMap = Pixmap.color_map(yMiddLine)
        distanceInPixels_2 = 0
        # Find distance between first and second signal transit trough midd line
        for i, x in enumerate(colorMap[0]):
            if x == self.channel_35.color.name():
                distanceInPixels_2 = i - distanceInPixels_2
                if distanceInPixels_2 != i:
                    break
        self.assertGreater(distanceInPixels_1, distanceInPixels_2)

    def test_Plot_Plot_Shortcut_Key_X(self):
        """
        Test Scope:
            Check if fitting between cursors is released after pressing key "X".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Maximize window
            - Count how many times signal intersects midd line of plot
            - Mouse click in the middle of plot
            - Press "R"
            - Get color of signal between cursors
            - Get Y coordinates of signal where it intersects cursors
            - Count how many times signal intersect midd line of plot between cursors
            - Press "X"
            - Get Y coordinates of start and end point of signal
            - Count how many times signal intersect midd line of plot
        Evaluate:
            > Precondition
                - Evaluate that plot is not black
                - Evaluate that plot intersects midd line of plot
                - Evaluate that new color of signal was found
                - Evaluate that one row after first cursor and one row before second cursor signal exist
                        or signal was found between cursors
                - Evaluate that signal intersect midd line of plot between cursors
            > Final tests
                - Evaluate that start and end points of signal was found
                - Evaluate lines where is situated start and end points of signal by this logic:
                    • After zooming to range - signal slope decrease
                    • After calculus - points can differ, but difference must be minimal, a few pixels maximum
                    • By dividing expected values and actual values if the values not differ too much,
                        expected result must be near 1, precision can be adjusted
                - Evaluate that the number of intersections between signal and the midd line inner cursors are equal
                    with number of intersections between signal and the midd line after pressing key "X"
                - Evaluate that the signal intersect the midd line less time after pressing key "X"
        """
        # Setup
        self.widget.showMaximized()
        self.processEvents()
        # Count intersections between middle line and signal
        firstIntersections = Pixmap.color_map(
            self.plot.plot.viewport().grab(QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.width(), 1))
        )[0].count(self.channel_35.color.name())
        self.assertTrue(firstIntersections)

        # Setup for cursor
        self.widget.set_cursor_options(False, False, 1, Pixmap.COLOR_CURSOR)
        # Mouse click on a center of plot
        QtTest.QTest.mouseClick(
            self.plot.plot.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            self.plot.plot.viewport().rect().center(),
        )
        QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key_R)
        self.processEvents()
        # Get X position of Cursor
        cursors = Pixmap.cursors_x(self.plot.plot.viewport().grab())
        self.assertEqual(len(cursors), 2)

        # Get a set of colors founded between cursors
        colors = Pixmap.color_names_exclude_defaults(
            self.plot.plot.viewport().grab(
                QtCore.QRect(cursors[0], 0, cursors[1] - cursors[0], self.plot.plot.height())
            )
        )
        # Exclude channel original color
        if self.channel_35.color.name() in colors:
            colors.remove(self.channel_35.color.name())
        # caught ya
        color = colors.pop()
        # Evaluate if color was found
        self.assertTrue(color)

        # Count intersection of midd line and signal between cursors
        interCursorsIntersectionsR = Pixmap.color_map(
            self.plot.plot.viewport().grab(
                QtCore.QRect(cursors[0], int(self.plot.plot.height() / 2), cursors[1] - cursors[0], 1)
            )
        )[0].count(color)
        self.assertTrue(interCursorsIntersectionsR)
        # Search lines where signal intersects cursors
        expectedSignalStartYPoint = Pixmap.search_y_of_signal_in_x(
            self.plot.plot.viewport().grab(QtCore.QRect(cursors[0] + 1, 0, 1, self.plot.plot.height())), color
        )
        expectedSignalEndYPoint = Pixmap.search_y_of_signal_in_x(
            self.plot.plot.viewport().grab(QtCore.QRect(cursors[1] - 1, 0, 1, self.plot.plot.height())), color
        )
        self.assertTrue(expectedSignalStartYPoint)
        self.assertTrue(expectedSignalEndYPoint)

        # Press key "X"
        QtTest.QTest.keyClick(self.plot.plot, QtCore.Qt.Key_X)
        self.processEvents()

        # Find where signal start and end on X axes
        from_to_x = Pixmap.search_signal_from_to_x(self.plot.plot.viewport().grab(), self.channel_35.color.name())
        self.assertEqual(len(from_to_x), 2)
        # Search lines where signal start and end
        signalStartOnLine = Pixmap.search_y_of_signal_in_x(
            self.plot.plot.viewport().grab(QtCore.QRect(from_to_x[0], 0, 1, self.plot.plot.height())),
            self.channel_35.color.name(),
        )
        signalEndOnLine = Pixmap.search_y_of_signal_in_x(
            self.plot.plot.viewport().grab(QtCore.QRect(from_to_x[1], 0, 1, self.plot.plot.height())),
            self.channel_35.color.name(),
        )
        # Evaluate
        precission = 0.05
        self.assertAlmostEqual(
            expectedSignalStartYPoint / signalStartOnLine,
            1,
            None,
            f"Difference is too big: {expectedSignalStartYPoint / signalStartOnLine}",
            precission,
        )
        self.assertAlmostEqual(
            expectedSignalEndYPoint / signalEndOnLine,
            1,
            None,
            f"Difference is too big: {expectedSignalEndYPoint / signalEndOnLine}",
            precission,
        )
        # The Number of intersections between signal and midd line must be the same as in the first case
        interCursorsIntersectionsX = Pixmap.color_map(
            self.plot.plot.viewport().grab(QtCore.QRect(0, int(self.plot.plot.height() / 2), self.plot.plot.width(), 1))
        )[0].count(self.channel_35.color.name())
        self.assertEqual(interCursorsIntersectionsR, interCursorsIntersectionsX)
        self.assertLess(interCursorsIntersectionsX, firstIntersections)

    def test_Plot_Plot_Shortcut_Ctrl_H_Ctrl_B_Ctrl_P(self):
        """
        Test Scope:
            Check if values is converted to int, hex, bin after pressing combination of key "Ctrl+<H>|<B>|<P>"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Press "Ctrl+H"
            - Press "Ctrl+B"
            - Press "Ctrl+P"
        Evaluate:
            - Evaluate that plot is not black
            - Evaluate that unit is changed to Hex after pressing key "Ctrl+H"
            - Evaluate that unit is changed to Bin after pressing key "Ctrl+B"
            - Evaluate that unit is changed to Int after pressing key "Ctrl+P"
        """
        # Setup
        Physical = self.plot.selected_channel_value.text()
        physicalHours = int(Physical.split(" ")[0])
        # Press "Ctrl+H"
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Ctrl+H"))

        Hex = self.plot.selected_channel_value.text()
        # Convert Hex value to Int
        hexHours = int(Hex.split(" ")[0], 16)
        # Evaluate
        self.assertNotEqual(Physical, Hex)
        self.assertIn("0x", Hex)
        self.assertEqual(physicalHours, hexHours)

        # Press "Ctrl+B"
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Ctrl+B"))
        Bin = self.plot.selected_channel_value.text()
        binHours = int(Bin.split(" ")[0].replace(".", ""), 2)
        self.assertNotEqual(Physical, Bin)
        self.assertEqual(physicalHours, binHours)

        self.assertAlmostEqual(..., ...)

        # Press "Ctrl+P"
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Ctrl+P"))
        newPhysical = self.plot.selected_channel_value.text()
        newPhysicalHours = int(newPhysical.split(" ")[0])
        self.assertEqual(Physical, newPhysical)
        self.assertEqual(physicalHours, newPhysicalHours)

    def test_Plot_Plot_Shortcut_Key_Period(self):
        """
        @BUG
        Sometime dots are designed close to the original graphic, not on them.
        The distance sometimes is too big.
        It's happened after drag and drop new signal on plot and pres key "Period" <.>
        @@@@@ @@@@ @@@ @@ @
        Test Scope:
            Check if variable "with_dots" is created, and it's value is modifiable by pressing key "Period"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Press 3 times key "Period" <<.>>
        Evaluate:
            - Evaluate that plot is not black
            - Evaluate that object "self.plot" doesn't have attribute "with_dots"
            - Evaluate that object "self.plot" have attribute "with_dots" with t's value "False"
                    after pressing key "Period"
            - Evaluate that attribute "with_dots" is "True" after pressing second time key "Period"
            - Evaluate that attribute "with_dots" is "False" after pressing third time key "Period"
        """
        self.assertFalse(hasattr(self.plot, "with_dots"))
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_Period)
        self.assertFalse(self.plot.with_dots)
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_Period)
        self.processEvents()
        self.assertTrue(self.plot.with_dots)
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_Period)
        self.assertFalse(self.plot.with_dots)

    def test_Plot_Plot_Shortcut_Key_Ctrl_I(self):
        """
        Test Scope:
            Check if bookmark is created after pressing key "Ctrl+I"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Mock QInputDialog object
            - Press "Ctrl+I"
        Evaluate:
            - Evaluate that bookmarks are not displayed before pressing "Ctrl+I"
            - Evaluate that bookmarks are displayed after pressing "Ctrl+I"
                and the message of the last bookmark is the first element of the returned list of mock object
        """
        # mock for bookmark
        with mock.patch("PySide6.QtWidgets.QInputDialog") as mo_QInputDialog:
            mo_QInputDialog.getMultiLineText.return_value = [self.id(), True]
            # Press "Ctrl+I"
            QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Ctrl+I"))
        # Evaluate
        self.assertTrue(self.plot.show_bookmarks)
        self.assertEqual(self.plot.plot.bookmarks[len(self.plot.plot.bookmarks) - 1].message, self.id())

        # question message box is called
        ...

    def test_Plot_Plot_Shortcut_Key_Alt_I(self):
        """
        Test Scope:
            Check functionality of key "Ctrl+I"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Mock QInputDialog object
            - Press 3 times "Alt+I"
        Evaluate:
            - Evaluate that bookmarks are not displayed before pressing "Alt+I"
            - Evaluate that bookmarks are displayed after pressing "Alt+I" first time
            - Evaluate that bookmarks are not displayed after pressing "Alt+I" second time
            - Evaluate that bookmarks are displayed after pressing "Alt+I" third time
        """
        # Evaluate
        self.assertFalse(self.plot.show_bookmarks)
        # Press "Al+I"
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Alt+I"))
        # Evaluate
        self.assertTrue(self.plot.show_bookmarks)
        # Press "Al+I"
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Alt+I"))
        # Evaluate
        self.assertFalse(self.plot.show_bookmarks)
        # Press "Al+I"
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Alt+I"))
        # Evaluate
        self.assertTrue(self.plot.show_bookmarks)

    def test_Plot_Plot_Shortcut_Key_Alt_R_Alt_S(self):
        """
        Test Scope:
            Check functionality of key "Alt+I" and "Alt+S"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Press "Alt+R"
            - Press "Alt+S"
        Evaluate:
            - Evaluate that signal mode is raw and line style is DashLine after pressing key "Alt+R"
            - Evaluate that signal mode is phys and line style is SolidLine after pressing key "Alt+S"
        """
        # Press "Alt+R"
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Alt+R"))
        # Evaluate
        # Signal mode = raw
        self.assertEqual(self.plot.plot.signals[0].mode, "raw")
        # Signal line style = Dash line
        self.assertEqual(self.plot.plot.signals[0].pen.style(), QtCore.Qt.PenStyle.DashLine)

        # Press "Alt+S"
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Alt+S"))
        # Evaluate
        # Signal mode = raw
        self.assertEqual(self.plot.plot.signals[0].mode, "phys")
        # Signal line style = Dash line
        self.assertEqual(self.plot.plot.signals[0].pen.style(), QtCore.Qt.PenStyle.SolidLine)


class TestShortcutsWith_2_Channels(TestPlotWidget):
    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot", channels_indexes=(36, 37))
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()
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
        # check if grid is available -> hide grid
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        # hide bookmarks if it's available
        if self.plot.show_bookmarks:
            self.plot.toggle_bookmarks(hide=True)
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
        # Case 0:
        with self.subTest("test_Plot_Plot_Shortcut_Key_LeftRight_0"):
            # Select channel: ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            self.mouseClick_WidgetItem(self.channel_37)
            self.plot.plot.setFocus()
            self.processEvents(0.1)

            self.assertEqual("23", self.channel_36.text(self.Column.VALUE))
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

    def test_Plot_Plot_Shortcut_Key_Shift_Arrows(self):
        """
        Test Scope:
            Check that Shift + Arrow Keys ensure moving of selected channels.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
            - Create plot with 2 channels
            - Press key "S" to separate signals for better evaluation
            - Click on first channel
            - Press "Shift" key + arrow "Down" & "Left"
            - Click on second channel
            - Press "Shift" key + arrow "Up" & "Right"
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black and contain colors of all 3 channels
            - Evaluate that first signal is shifted down & left after pressing combination "Shift+Down" & "Shift+Left"
            - Evaluate that second signal is shifted up & right after pressing combination "Shift+Up" & "Shift+Right"
        """
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        self.processEvents()
        old_from_to_y_channel_36 = Pixmap.search_signal_from_to_y(
            self.plot.plot.viewport().grab(), self.channel_36.color.name()
        )
        old_from_to_y_channel_37 = Pixmap.search_signal_from_to_y(
            self.plot.plot.viewport().grab(), self.channel_37.color.name()
        )
        old_from_to_x_channel_36 = Pixmap.search_signal_from_to_x(
            self.plot.plot.viewport().grab(), self.channel_36.color.name()
        )
        old_from_to_x_channel_37 = Pixmap.search_signal_from_to_x(
            self.plot.plot.viewport().grab(), self.channel_37.color.name()
        )

        self.mouseClick_WidgetItem(self.channel_36)
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Shift+Down"))
        for i in range(10):
            self.processEvents()
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Shift+Left"))
        for i in range(10):
            self.processEvents()
        self.mouseClick_WidgetItem(self.channel_37)
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Shift+Up"))
        for i in range(10):
            self.processEvents()
        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Shift+Right"))
        for i in range(10):
            self.processEvents()

        new_from_to_y_channel_36 = Pixmap.search_signal_from_to_y(
            self.plot.plot.viewport().grab(), self.channel_36.color.name()
        )
        new_from_to_y_channel_37 = Pixmap.search_signal_from_to_y(
            self.plot.plot.viewport().grab(), self.channel_37.color.name()
        )
        new_from_to_x_channel_36 = Pixmap.search_signal_from_to_x(
            self.plot.plot.viewport().grab(), self.channel_36.color.name()
        )
        new_from_to_x_channel_37 = Pixmap.search_signal_from_to_x(
            self.plot.plot.viewport().grab(), self.channel_37.color.name()
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


class TestShortcutsWith_3_Channels(TestPlotWidget):
    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot", channels_indexes=(35, 36, 37))
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()
        # Settings for cursor
        # self.widget.set_cursor_options(False, False, 1, Pixmap.COLOR_BACKGROUND)
        # channels
        self.channel_35 = self.plot.channel_selection.topLevelItem(0)
        self.channel_36 = self.plot.channel_selection.topLevelItem(1)
        self.channel_37 = self.plot.channel_selection.topLevelItem(2)
        self.assertEqual(3, self.plot.channel_selection.topLevelItemCount())
        # Double-click on channels
        self.mouseDClick_WidgetItem(self.channel_35)
        self.mouseDClick_WidgetItem(self.channel_36)
        self.mouseDClick_WidgetItem(self.channel_37)
        self.processEvents()
        # Remove dots
        if self.plot.plot.with_dots:
            self.plot.plot.set_dots(False)
        self.processEvents()
        # check if grid is available -> hide grid
        if not self.plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(self.plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        # hide bookmarks if it's available
        if self.plot.show_bookmarks:
            self.plot.toggle_bookmarks(hide=True)
        self.processEvents()
        # pixmap is not black
        self.assertFalse(Pixmap.is_black(self.plot.plot.viewport().grab()))

    def test_Plot_Plot_Shortcut_Key_S_ShiftS_ShiftF_F(self):
        """
        Test Scope:
            To check if:
              > all signals is stack after pressing key "S"
              > only selected signal is fitted after pressing combination "Sift + F"
              > only selected signal is stacked after pressing combination "Shift + S"
              > all signals is fitted after pressing key "F"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a plot
            - Press Key "S"
            - Press combination "Shift + F"
            - Press combination "Shift + S"
            - press key "F"
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black
            - Evaluate that signals are separated in top, midd and bottom third of plot after pressing key "S"
            - Evaluate that only selected signal is fitted after pressing combination "Shift + F"
            - Evaluate that only selected signal is stacked after pressing combination "Shift + S"
            - Evaluate that all signals is fitted after pressing key "F"
        """
        # Press "S"
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_S)
        self.processEvents()
        # Evaluate
        with self.subTest("test_shortcut_S"):
            # First 2 lines
            self.assertTrue(
                Pixmap.is_black(self.plot.plot.viewport().grab(QtCore.QRect(0, 0, self.plot.plot.width(), 2)))
            )
            # Top
            pixmap = self.plot.plot.viewport().grab(
                QtCore.QRect(0, 0, self.plot.plot.width(), int(self.plot.plot.height() / 3))
            )
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_37.color.name()))
            # Midd
            pixmap = self.plot.plot.viewport().grab(
                QtCore.QRect(
                    0, int(self.plot.plot.height() / 3), self.plot.plot.width(), int(self.plot.plot.height() / 3)
                )
            )
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_35.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_37.color.name()))
            # Bottom
            pixmap = self.plot.plot.viewport().grab(
                QtCore.QRect(
                    0, int(self.plot.plot.height() / 3) * 2, self.plot.plot.width(), int(self.plot.plot.height() / 3)
                )
            )
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_35.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_36.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_37.color.name()))
            # Last 2 lines
            self.assertTrue(
                Pixmap.is_black(
                    self.plot.plot.viewport().grab(
                        QtCore.QRect(0, self.plot.plot.height() - 2, self.plot.plot.width(), 2)
                    )
                )
            )

        # select the first channel
        self.mouseClick_WidgetItem(self.channel_35)
        # Press "Shift+F"
        QtTest.QTest.keySequence(self.plot.plot, QtGui.QKeySequence("Shift+F"))
        for i in range(100):
            self.processEvents()
        # Evaluate
        with self.subTest("test_shortcut_Shift_F"):
            # First line
            self.assertTrue(
                Pixmap.is_black(self.plot.plot.viewport().grab(QtCore.QRect(0, 0, self.plot.plot.width(), 1)))
            )
            # Second line
            self.assertTrue(
                Pixmap.has_color(
                    self.plot.plot.viewport().grab(QtCore.QRect(0, 1, self.plot.plot.width(), 1)),
                    self.channel_35.color.name(),
                )
            )
            # Top
            pixmap = self.plot.plot.viewport().grab(
                QtCore.QRect(0, 0, self.plot.plot.width(), int(self.plot.plot.height() / 3))
            )
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_37.color.name()))
            # Midd
            pixmap = self.plot.plot.viewport().grab(
                QtCore.QRect(
                    0, int(self.plot.plot.height() / 3), self.plot.plot.width(), int(self.plot.plot.height() / 3)
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_37.color.name()))
            # Bottom
            pixmap = self.plot.plot.viewport().grab(
                QtCore.QRect(
                    0, int(self.plot.plot.height() / 3) * 2, self.plot.plot.width(), int(self.plot.plot.height() / 3)
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_36.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_37.color.name()))
            # Last 2 lines
            self.assertTrue(
                Pixmap.has_color(
                    self.plot.plot.viewport().grab(
                        QtCore.QRect(0, self.plot.plot.height() - 2, self.plot.plot.width(), 1)
                    ),
                    self.channel_35.color.name(),
                )
            )
            self.assertTrue(
                Pixmap.is_black(
                    self.plot.plot.viewport().grab(
                        QtCore.QRect(0, self.plot.plot.height() - 1, self.plot.plot.width(), 1)
                    )
                )
            )

        # select second channel
        self.mouseClick_WidgetItem(self.channel_36)
        # Press "Shift+F"
        QtTest.QTest.keySequence(self.plot.plot, QtGui.QKeySequence("Shift+S"))
        for i in range(100):
            self.processEvents()
        # Evaluate
        with self.subTest("test_shortcut_Shift_S"):
            # First line
            self.assertTrue(
                Pixmap.is_black(self.plot.plot.viewport().grab(QtCore.QRect(0, 0, self.plot.plot.width(), 1)))
            )
            # Second line
            self.assertTrue(
                Pixmap.has_color(
                    self.plot.plot.viewport().grab(QtCore.QRect(0, 1, self.plot.plot.width(), 1)),
                    self.channel_35.color.name(),
                )
            )
            # Top
            pixmap = self.plot.plot.viewport().grab(
                QtCore.QRect(0, 0, self.plot.plot.width(), int(self.plot.plot.height() / 3))
            )
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_37.color.name()))
            # Midd
            pixmap = self.plot.plot.viewport().grab(
                QtCore.QRect(
                    0, int(self.plot.plot.height() / 3), self.plot.plot.width(), int(self.plot.plot.height() / 3)
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
            self.assertFalse(Pixmap.has_color(pixmap, self.channel_37.color.name()))
            # Bottom
            pixmap = self.plot.plot.viewport().grab(
                QtCore.QRect(
                    0, int(self.plot.plot.height() / 3) * 2, self.plot.plot.width(), int(self.plot.plot.height() / 3)
                )
            )
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
            self.assertTrue(Pixmap.has_color(pixmap, self.channel_37.color.name()))
            # Last 2 lines
            self.assertTrue(
                Pixmap.has_color(
                    self.plot.plot.viewport().grab(
                        QtCore.QRect(0, self.plot.plot.height() - 2, self.plot.plot.width(), 1)
                    ),
                    self.channel_35.color.name(),
                )
            )
            self.assertTrue(
                Pixmap.is_black(
                    self.plot.plot.viewport().grab(
                        QtCore.QRect(0, self.plot.plot.height() - 1, self.plot.plot.width(), 1)
                    )
                )
            )

            # Press "F"
            QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_F)
            self.processEvents()
            # Evaluate
            with self.subTest("test_shortcut_S"):
                # First line
                self.assertTrue(
                    Pixmap.is_black(self.plot.plot.viewport().grab(QtCore.QRect(0, 0, self.plot.plot.width(), 1)))
                )
                # Top
                pixmap = self.plot.plot.viewport().grab(
                    QtCore.QRect(0, 0, self.plot.plot.width(), int(self.plot.plot.height() / 3))
                )
                self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, self.channel_37.color.name()))
                # Midd
                pixmap = self.plot.plot.viewport().grab(
                    QtCore.QRect(
                        0, int(self.plot.plot.height() / 3), self.plot.plot.width(), int(self.plot.plot.height() / 3)
                    )
                )
                self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, self.channel_37.color.name()))
                # Bottom
                pixmap = self.plot.plot.viewport().grab(
                    QtCore.QRect(
                        0,
                        int(self.plot.plot.height() / 3) * 2,
                        self.plot.plot.width(),
                        int(self.plot.plot.height() / 3),
                    )
                )
                self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
                self.assertTrue(Pixmap.has_color(pixmap, self.channel_37.color.name()))
                # Last line
                self.assertTrue(
                    Pixmap.is_black(
                        self.plot.plot.viewport().grab(
                            QtCore.QRect(0, self.plot.plot.height() - 1, self.plot.plot.width(), 1)
                        )
                    )
                )

    def test_Plot_Plot_Shortcut_Key_2(self):
        """
        Test Scope:
            To check if is displayed only selected channel after pressing key "2"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a plot
            - Press Key "2"
            - Select first channel
            - Pres key "Down"
            - Select third channel
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black and contain colors of all 3 channels
            - Evaluate that plot is black after pressing key "2"
            - Evaluate that plot contains only color of first channel after clicked on first channel
            - Evaluate that plot contains only color of second channel after pressing key "Down"
            - Evaluate that plot contains only color of third channel after clicked on third channel
            - Evaluate that plot contains colors of all 3 channels after pressing key "2"
        """
        pixmap = self.plot.plot.viewport().grab()
        # Evaluate
        self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channel_37.color.name()))

        # case 0
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_2)
        for i in range(100):
            self.processEvents()
        # Evaluate
        self.assertTrue(Pixmap.is_black(self.plot.plot.viewport().grab()))

        # case 1
        self.mouseClick_WidgetItem(self.channel_35)
        for i in range(200):
            self.processEvents()
        pixmap = self.plot.plot.viewport().grab()
        # Evaluate
        self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
        self.assertFalse(Pixmap.has_color(pixmap, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(pixmap, self.channel_37.color.name()))

        # case 2
        QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key_Down)
        for i in range(200):
            self.processEvents()
        pixmap = self.plot.plot.viewport().grab()
        # Evaluate
        self.assertFalse(Pixmap.has_color(pixmap, self.channel_35.color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
        self.assertFalse(Pixmap.has_color(pixmap, self.channel_37.color.name()))

        # case 3
        self.mouseClick_WidgetItem(self.channel_37)
        self.processEvents()
        pixmap = self.plot.plot.viewport().grab()
        # Evaluate
        self.assertFalse(Pixmap.has_color(pixmap, self.channel_35.color.name()))
        self.assertFalse(Pixmap.has_color(pixmap, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channel_37.color.name()))

        # case 4
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_2)
        self.processEvents()
        pixmap = self.plot.plot.viewport().grab()
        # Evaluate
        self.assertTrue(Pixmap.has_color(pixmap, self.channel_35.color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channel_36.color.name()))
        self.assertTrue(Pixmap.has_color(pixmap, self.channel_37.color.name()))

    def test_Plot_Plot_Shortcut_Key_M_3_CH(self):
        """
        Test Scope:
            To check if is displayed info about selected channel and after deleting all channels buffer is clear
                by pressing key "M"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a plot
            - Press Key "M"
            - Select third channel
            - Select first channel
            - Pres key "Down"
        Evaluate:
            - Evaluate that displayed info is related to third channel
            - Evaluate that displayed info is related to first channel
            - Evaluate that displayed info is related to second channel
            - Evaluate that buffer is clear
        """
        QtTest.QTest.keyClick(self.plot.plot.viewport(), QtCore.Qt.Key_M)
        self.processEvents()
        with self.subTest("test_key_M_last_channel"):
            self.assertEqual(self.plot.info._name, self.channel_37.name)
            self.assertEqual(self.plot.info.color, self.channel_37.color.name())

        # click on a first channel
        with self.subTest("test_key_M_click_on_first_channel"):
            self.mouseClick_WidgetItem(self.channel_35)
            self.assertEqual(self.plot.info._name, self.channel_35.name)
            self.assertEqual(self.plot.info.color, self.channel_35.color.name())

        with self.subTest("test_key_M_press_key_Down_on_channel_selection"):
            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key_Down)
            self.assertEqual(self.plot.info._name, self.channel_36.name)
            self.assertEqual(self.plot.info.color, self.channel_36.color.name())

        # delete all channels
        with self.subTest("test_key_M_delete_all_channels"):
            self.mouseClick_WidgetItem(self.channel_35)
            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key_Delete)
            self.mouseClick_WidgetItem(self.channel_36)
            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key_Delete)
            self.mouseClick_WidgetItem(self.channel_37)
            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key_Delete)

            # Not save value of the last selected channel
            self.assertNotEqual(self.plot.info._name, self.channel_37.name)

    def test_Plot_Plot_Shortcut_Ctrl_Key_S(self):
        """
        Test Scope:
            Check if by pressing "Ctrl+S" is saved in the new measurement file only active channels
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a plot
            _ Deselect last channel
            - Mock getSaveFileName() object and set return value of this object a file path of the new measurement file
            - Press Key "Ctrl+S"
            - Open recently created measurement file in a new window
        Evaluate:
            - Evaluate that object getSaveFileName() was called after pressing combination "Ctrl+S"
            - Evaluate that in measurement file is saved only active channels
        """
        self.mouseDClick_WidgetItem(self.channel_37)
        # mock for getSaveFileName object
        with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            file_path = os.path.join(self.test_workspace, "file.mf4")
            mo_getSaveFileName.return_value = (file_path, "")
            # Press Ctrl+S
            QtTest.QTest.keySequence(self.plot.plot, QtGui.QKeySequence("Ctrl+S"))
        # Evaluate
        mo_getSaveFileName.assert_called()

        # Open recently saved measurement file
        self.setUpFileWidget(measurement_file=file_path, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Evaluate
        self.assertIn(self.channel_35.name, self.widget.channels_db_items)
        self.assertIn(self.channel_36.name, self.widget.channels_db_items)
        self.assertNotIn(self.channel_37.name, self.widget.channels_db_items)

    def test_Plot_Plot_Shortcut_Ctrl_Key_Shift_S(self):
        """
        Test Scope:
            Check if by pressing "Ctrl+Shift+S" is saved in the new measurement file all channels from mdi area
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 3 signals and create a plot
            - Mock getSaveFileName() object and set return value of this object a file path of the new measurement file
            - Press Key "Ctrl+Shift+S"
            - Open recently created measurement file in a new window
        Evaluate:
            - Evaluate that object getSaveFileName() was called after pressing combination "Ctrl+Shift+S"
            - Evaluate that in measurement file is saved all channels from mdi area
        """
        # mock for getSaveFileName object
        with mock.patch("asammdf.gui.widgets.mdi_area.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            file_path = os.path.join(self.test_workspace, "file.mf4")
            mo_getSaveFileName.return_value = (file_path, "")
            # Press Ctrl+S
            QtTest.QTest.keySequence(self.plot.plot, QtGui.QKeySequence("Ctrl+Shift+S"))
        # Evaluate
        mo_getSaveFileName.assert_called()

        # Open recently saved measurement file
        self.setUpFileWidget(measurement_file=file_path, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Evaluate
        self.assertIn(self.channel_35.name, self.widget.channels_db_items)
        self.assertIn(self.channel_36.name, self.widget.channels_db_items)
        self.assertIn(self.channel_37.name, self.widget.channels_db_items)

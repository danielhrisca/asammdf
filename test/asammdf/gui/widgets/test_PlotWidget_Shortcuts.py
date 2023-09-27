#!/usr/bin/env python
from test.asammdf.gui.test_base import Pixmap
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget

from PySide6 import QtCore, QtGui, QtTest


class TestShortcuts(TestPlotWidget):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.PlotOffset = 5

    def test_Plot_Plot_Shortcut_Key_LeftRight(self):
        """
        Test Scope:
            Check that Arrow Keys: Left & Right ensure navigation on channels evolution.
            Ensure that navigation is working.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Switch ComboBox to "Natural sort"
            - Press PushButton "Create Window"
            - Drag and Drop channels from FileWidget.channels_tree to Plot.channels_selection:
                # First
                - ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
                # Second
                - ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            - Send KeyClick Right 5 times
            - Send KeyClick Left 4 times
        Evaluate:
            - Evaluate values from `Value` column on Plot.channels_selection
            - Evaluate timestamp label
        """
        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)

        plot = self.widget.mdi_area.subWindowList()[0].widget()
        channel_selection = plot.channel_selection
        channel_14 = self.add_channel_to_plot(plot=plot, channel_name="ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL")
        channel_15 = self.add_channel_to_plot(plot=plot, channel_name="ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL")
        self.assertEqual(2, plot.channel_selection.topLevelItemCount())

        # Case 0:
        with self.subTest("test_Plot_Plot_Shortcut_Key_LeftRight_0"):
            # Select channel: ASAM_[15].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            self.mouseClick_WidgetItem(channel_15)
            plot.plot.setFocus()
            self.processEvents(0.1)

            self.assertEqual("25", channel_14.text(self.Column.VALUE))
            self.assertEqual("244", channel_15.text(self.Column.VALUE))

            # Send Key strokes
            for _ in range(6):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("8", channel_14.text(self.Column.VALUE))
            self.assertEqual("6", channel_15.text(self.Column.VALUE))
            self.assertEqual("t = 0.082657s", plot.cursor_info.text())

            # Send Key strokes
            for _ in range(5):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Left)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("21", channel_14.text(self.Column.VALUE))
            self.assertEqual("247", channel_15.text(self.Column.VALUE))
            self.assertEqual("t = 0.032657s", plot.cursor_info.text())

        # Case 1:
        with self.subTest("test_Plot_Plot_Shortcut_Key_LeftRight_1"):
            # Select channel: ASAM_[14].M.MATRIX_DIM_16.UBYTE.IDENTICAL
            self.mouseClick_WidgetItem(channel_15)
            plot.plot.setFocus()
            self.processEvents(0.1)

            # Send Key strokes
            for _ in range(6):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("5", channel_14.text(self.Column.VALUE))
            self.assertEqual("9", channel_15.text(self.Column.VALUE))
            self.assertEqual("t = 0.092657s", plot.cursor_info.text())

            # Send Key strokes
            for _ in range(5):
                QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Left)
                self.processEvents(0.1)
            self.processEvents(0.1)

            # Evaluate
            self.assertEqual("18", channel_14.text(self.Column.VALUE))
            self.assertEqual("250", channel_15.text(self.Column.VALUE))
            self.assertEqual("t = 0.042657s", plot.cursor_info.text())

    def test_Plot_Plot_Shortcut_Key_R(self):
        """
        Test Scope:
            Check if Range Selection rectangle is painted over the plot.
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
        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Press PushButton "Create Window"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Press PushButton "Hide axis"
        if not plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(plot.hide_axes_btn, QtCore.Qt.LeftButton)

        # Save PixMap of clear plot
        clear_pixmap = plot.plot.viewport().grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

        # Get X position of Cursor
        cursors = Pixmap.cursors_x(clear_pixmap)
        # Evaluate that there is only one cursor
        self.assertEqual(1, len(cursors))

        # Press Key 'R' for range selection
        QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_R)
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = plot.plot.viewport().grab()
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
        QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_Right)
        self.processEvents(timeout=0.01)
        QtTest.QTest.keySequence(plot.plot, QtGui.QKeySequence("Ctrl+Left"))
        self.processEvents(timeout=0.01)

        # Save PixMap of Range plot
        range_pixmap = plot.plot.viewport().grab()
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
        QtTest.QTest.keyClick(plot.plot, QtCore.Qt.Key_R)
        self.processEvents(timeout=0.01)

        # Save PixMap of clear plot
        clear_pixmap = plot.plot.viewport().grab()
        self.assertTrue(Pixmap.is_black(clear_pixmap))

    def test_Plot_Plot_Shortcut_Key_F(self):
        """
        Test Scope:
            Check if all plotted signals is fitted on Y ax after pressing key "F".
        Events:
            - Open 'FileWidget' with valid measurement.
            - Select 2 measurement and create a plot
            - Press Key "S"
            - Press Key "F"
        Evaluate:
            - Evaluate that two signals are available
            - Evaluate that plot is not black
            - Evaluate that the color of signals is not displayed on top and bottom line of plot after pressing key "S"
            - Evaluate that are both colors of signals in top and bottom line of plot after pressing key "F"
        """
        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Press PushButton "Create Window"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        channel_0 = self.add_channel_to_plot(plot=plot, channel_index=7)
        channel_1 = self.add_channel_to_plot(plot=plot, channel_name="ASAM.M.SCALAR.UBYTE.HYPERBOLIC")
        self.assertEqual(2, plot.channel_selection.topLevelItemCount())
        # Press "S"
        QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_S)
        # save pixmap
        pixmap = plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap))
        # Select first and last effective pixel line of plot
        yTopLine = plot.plot.viewport().grab(
            QtCore.QRect(0, plot.plot.height() - self.PlotOffset, plot.plot.viewport().width(), 1)
            )
        yBottomLine = plot.plot.viewport().grab(
            QtCore.QRect(0, self.PlotOffset, plot.plot.viewport().width(), 1)
        )
        self.assertFalse(Pixmap.has_color(yTopLine, channel_0.color.name()))
        self.assertFalse(Pixmap.has_color(yTopLine, channel_1.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine, channel_0.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine, channel_1.color.name()))
        # Press "F"
        QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_F)
        self.processEvents()
        # Save Top and Bottom pixel line of plot
        yTopLine = plot.plot.viewport().grab(
            QtCore.QRect(0, plot.plot.height() - self.PlotOffset, plot.plot.viewport().width(), 1)
            )
        yBottomLine = plot.plot.viewport().grab(
            QtCore.QRect(0, self.PlotOffset, plot.plot.viewport().width(), 1)
        )
        self.assertTrue(Pixmap.has_color(yTopLine, channel_0.color.name()))
        self.assertTrue(Pixmap.has_color(yTopLine, channel_1.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomLine, channel_0.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomLine, channel_1.color.name()))

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
        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Press PushButton "Create Window"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        channel_0 = self.add_channel_to_plot(plot=plot, channel_index=7)
        channel_1 = self.add_channel_to_plot(plot=plot, channel_name="ASAM.M.SCALAR.UBYTE.HYPERBOLIC")
        self.assertEqual(2, plot.channel_selection.topLevelItemCount())
        # Press "S"
        QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_S)
        # save pixmap
        pixmap = plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap))
        # Select top and bottom pixel line of plot
        yTopLine = plot.plot.viewport().grab(
            QtCore.QRect(0, plot.plot.height() - self.PlotOffset, plot.plot.viewport().width(), 1)
        )
        yBottomLine = plot.plot.viewport().grab(
            QtCore.QRect(0, self.PlotOffset, plot.plot.viewport().width(), 1)
        )
        self.assertFalse(Pixmap.has_color(yTopLine, channel_0.color.name()))
        self.assertFalse(Pixmap.has_color(yTopLine, channel_1.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine, channel_0.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine, channel_1.color.name()))
        # Select first signal
        self.mouseClick_WidgetItem(channel_0)
        # Press "Shift+F"
        QtTest.QTest.keySequence(plot.plot.viewport(), QtGui.QKeySequence("Shift+F"))
        self.processEvents()
        # save top and bottom pixel line of plot
        yTopLine = plot.plot.viewport().grab(
            QtCore.QRect(0, plot.plot.height() - self.PlotOffset, plot.plot.viewport().width(), 1)
        )
        yBottomLine = plot.plot.viewport().grab(
            QtCore.QRect(0, self.PlotOffset, plot.plot.viewport().width(), 1)
        )
        self.assertTrue(Pixmap.has_color(yTopLine, channel_0.color.name()))
        self.assertFalse(Pixmap.has_color(yTopLine, channel_1.color.name()))
        self.assertTrue(Pixmap.has_color(yBottomLine, channel_0.color.name()))
        self.assertFalse(Pixmap.has_color(yBottomLine, channel_1.color.name()))

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
        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Press PushButton "Create Window"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        # check if grid is available
        if plot.hide_axes_btn.isFlat():
            QtTest.QTest.mouseClick(plot.hide_axes_btn, QtCore.Qt.MouseButton.LeftButton)
        # case 1: check if X and Y axes is hidden
        if not plot.plot.x_axis.grid and not plot.plot.y_axis.grid:
            # press key "G"
            QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_G)
            self.processEvents()
            self.assertTrue(plot.plot.x_axis.grid)
            self.assertFalse(plot.plot.y_axis.grid)
            # press key "G"
            QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_G)
            self.processEvents()
            self.assertTrue(plot.plot.x_axis.grid)
            self.assertTrue(plot.plot.y_axis.grid)
            # press key "G"
            QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_G)
            self.processEvents()
            self.assertFalse(plot.plot.x_axis.grid)
            self.assertFalse(plot.plot.y_axis.grid)
        elif plot.plot.x_axis.grid and  not plot.plot.y_axis.grid:
            # press key "G"
            QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_G)
            self.processEvents()
            self.assertTrue(plot.plot.x_axis.grid)
            self.assertTrue(plot.plot.y_axis.grid)
            # press key "G"
            QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_G)
            self.processEvents()
            self.assertFalse(plot.plot.x_axis.grid)
            self.assertFalse(plot.plot.y_axis.grid)
            # press key "G"
            QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_G)
            self.processEvents()
            self.assertTrue(plot.plot.x_axis.grid)
            self.assertFalse(plot.plot.y_axis.grid)
        else:
            # press key "G"
            QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_G)
            self.processEvents()
            self.assertTrue(plot.plot.x_axis.grid)
            self.assertFalse(plot.plot.y_axis.grid)
            # press key "G"
            QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_G)
            self.processEvents()
            self.assertTrue(plot.plot.x_axis.grid)
            self.assertTrue(plot.plot.y_axis.grid)
            # press key "G"
            QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_G)
            self.processEvents()
            self.assertFalse(plot.plot.x_axis.grid)
            self.assertFalse(plot.plot.y_axis.grid)

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
                * first column after pressing "S" and "I" is full black => signal colors are not there
                * signal is zoomed => is extended to left side => in last column is both signals colors
            - Evaluate that after pressing key "W" in first and last column is displayed both signals
        """
        # Event
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Press PushButton "Create Window"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        channel_0 = self.add_channel_to_plot(plot=plot, channel_index=7)
        channel_1 = self.add_channel_to_plot(plot=plot, channel_name="ASAM.M.SCALAR.UBYTE.HYPERBOLIC")
        self.assertEqual(2, plot.channel_selection.topLevelItemCount())
        # search first and last column where is displayed first signal
        firstColoredColumn = None
        lastColoredColumn = None
        for line in range(plot.plot.viewport().height()):
            if Pixmap.has_color(plot.plot.viewport().grab(
                QtCore.QRect(line, 0, 1, plot.plot.viewport().rect().height())
            ), channel_0.color.name()):
                firstColoredColumn = line
                break
        # Evaluate that there are at least one column with signal color
        self.assertTrue(firstColoredColumn)
        for line in range(plot.plot.viewport().height(), firstColoredColumn, -1):
            if Pixmap.has_color(plot.plot.viewport().grab(
                QtCore.QRect(line, 0, 1, plot.plot.viewport().rect().height())
            ), channel_0.color.name()):
                lastColoredColumn = line
                break
        # Press "S" and "I"
        QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_S)
        QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_I)
        self.processEvents()
        # save pixmap
        pixmap = plot.plot.viewport().grab()
        self.assertFalse(Pixmap.is_black(pixmap))
        # save left and right pixel column
        xLeftColumn = plot.plot.viewport().grab(
            QtCore.QRect(firstColoredColumn, 0, 1, plot.plot.viewport().rect().height())
        )
        xRightColumn = plot.plot.viewport().grab(
            QtCore.QRect(lastColoredColumn, 0, 1, plot.plot.viewport().rect().height())
        )
        self.assertFalse(Pixmap.has_color(xLeftColumn, channel_0.color.name()))
        self.assertFalse(Pixmap.has_color(xLeftColumn, channel_1.color.name()))
        self.assertTrue(Pixmap.has_color(xRightColumn, channel_0.color.name()))
        self.assertTrue(Pixmap.has_color(xRightColumn, channel_1.color.name()))
        # Press "W"
        QtTest.QTest.keyClick(plot.plot.viewport(), QtCore.Qt.Key_W)
        self.processEvents()
        # Save left and right pixel column
        xLeftColumn = plot.plot.viewport().grab(
            QtCore.QRect(firstColoredColumn, 0, 1, plot.plot.viewport().rect().height())
        )
        xRightColumn = plot.plot.viewport().grab(
            QtCore.QRect(lastColoredColumn, 0, 1, plot.plot.viewport().rect().height())
        )
        self.assertTrue(Pixmap.has_color(xLeftColumn, channel_0.color.name()))
        self.assertTrue(Pixmap.has_color(xLeftColumn, channel_1.color.name()))
        self.assertTrue(Pixmap.has_color(xRightColumn, channel_0.color.name()))
        self.assertTrue(Pixmap.has_color(xRightColumn, channel_1.color.name()))

#!/usr/bin/env python

from asammdf.gui.widgets.formated_axis import FormatedAxis as FA
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget
from unittest import mock

from PySide6 import QtCore, QtGui, QtTest


class TestPlotShortcuts(TestPlotWidget):
    def setUp(self):
        # Open measurement file
        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

    def test_Plot_Plot_Shortcut_Key_M(self):
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
            - Evaluate that setSizes() method was called 2 times
        """
        expected_mo_call_count = 2
        # At start Statistics is hidden
        self.assertFalse(self.plot.info.isVisible())
        with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QSplitter.setSizes") as mo_setSizes:
            # press key "M"
            QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_M)
            self.assertTrue(self.plot.info.isVisible())
            # press key "M"
            QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_M)
            self.assertFalse(self.plot.info.isVisible())
        self.assertEqual(mo_setSizes.call_count, expected_mo_call_count)

    def test_Plot_Plot_Shortcut_Key_2(self):
        """
        Test Scope:
            ...
        Events:
            - Open 'FileWidget' with valid measurement.
            - Press PushButton "Create Window"
            - Press Key 2
        Evaluate:
            -
        """
        # Setup
        previous_focused_mode_btn = self.plot.focused_mode_btn.isFlat()
        # Press 2
        QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_2)
        self.assertNotEqual(previous_focused_mode_btn, self.plot.focused_mode_btn.isFlat())

        # Update
        previous_focused_mode_btn = self.plot.focused_mode_btn.isFlat()
        # Press 2
        QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_2)
        self.assertNotEqual(previous_focused_mode_btn, self.plot.focused_mode_btn.isFlat())

    def test_Plot_Plot_Shortcut_Keys_Ctrl_B__Ctrl_H__Ctrl_P__Ctrl_T(self):
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
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()
        # mock...
        with mock.patch("asammdf.gui.widgets.plot.Plot.item_by_uuid"):
            with mock.patch("asammdf.gui.widgets.plot.PlotGraphics.get_axis") as mo_get_axis:
                mo_get_axis.return_value.mock_add_spec(FA)
                with self.subTest("test_Ctrl_B"):
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+B"))
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "bin")
                    self.assertEqual(mo_get_axis.return_value.format, "bin")
                with self.subTest("test_Ctrl_H"):
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+H"))
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "hex")
                    self.assertEqual(mo_get_axis.return_value.format, "hex")
                with self.subTest("test_Ctrl_P"):
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+P"))
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "phys")
                    self.assertEqual(mo_get_axis.return_value.format, "phys")
                with self.subTest("test_Ctrl_T"):
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+T"))
                    self.assertEqual(self.plot.channel_selection.topLevelItem(0).format, "ascii")
                    self.assertEqual(mo_get_axis.return_value.format, "ascii")

    def test_Plot_Plot_Shortcut_Key_R(self):
        """
        Test Scope:
            Check R
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Select signal
            - Press "R"
        Evaluate:
            -
        """
        # Setup
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.tree.ChannelsTreeItem.set_prefix") as mo_set_prefix:
            with mock.patch("asammdf.gui.widgets.tree.ChannelsTreeItem.set_value") as mo_set_value:
                QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_R)
        mo_set_prefix.assert_called()
        mo_set_value.assert_called()

    def test_Plot_Plot_Shortcut_Key_Ctrl_R(self):
        """
        Test Scope:
            Check Ctrl+R
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Select signal
            - Press "Ctrl+R"
        Evaluate:
            -
        """
        # Setup
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.tree.ChannelsTreeWidget.keyPressEvent") as mo_keyPressEvent:
            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+R"))
        mo_keyPressEvent.assert_called()

    def test_Plot_Plot_Shortcut_Keys_Alt_R__Alt_S(self):
        """
        Test Scope:
            Check functionality of key "Alt+I" and "Alt+S"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot
            - Press "Alt+R"
            - Press "Alt+S"
        Evaluate:
            - Evaluate that signal mode is raw and after pressing key "Alt+R"
            - Evaluate that signal mode is phys after pressing key "Alt+S"
        """
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()

        with mock.patch.object(self.plot.plot, "viewbox"):
            with mock.patch.object(self.plot.plot, "update"):
                with mock.patch.object(self.plot.plot, "get_axis") as mo_get_axis:
                    # Event
                    # Press "Alt+R"
                    QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Alt+R"))
                    # Evaluate
                    with self.subTest("test_key_Alt_R"):
                        # Signal mode = raw
                        self.assertEqual(mo_get_axis.return_value.mode, "raw")
                        self.assertIsNone(mo_get_axis.return_value.picture)
                        mo_get_axis.return_value.update.assert_called()

                    # Event
                    # Press "Alt+S"
                    QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Alt+S"))
                    with self.subTest("test_key_Alt_S"):
                        # Signal mode = phys
                        self.assertEqual(mo_get_axis.return_value.mode, "phys")
                        self.assertIsNone(mo_get_axis.return_value.picture)
                        mo_get_axis.return_value.update.assert_called()

        # Final evaluation
        self.assertEqual(mo_get_axis.return_value.update.call_count, 2)
        self.assertEqual(mo_get_axis.call_count, 6)

    def test_Plot_Plot_Shortcut_Key_Ctrl_I(self):
        """
        ...
        """
        with mock.patch("asammdf.gui.widgets.plot.QtWidgets.QInputDialog.getMultiLineText") as mo_getMultiLineText:
            mo_getMultiLineText.return_value = self.id(), True
            with mock.patch.object(self.plot.plot, "bookmarks") as mo_bookmarks:
                with mock.patch("asammdf.gui.widgets.plot.Bookmark") as mo_Bookmark:
                    with mock.patch.object(self.plot.plot, "viewbox") as mo_viewbox:
                        QtTest.QTest.keySequence(self.plot.plot.viewport(), QtGui.QKeySequence("Ctrl+I"))

        mo_bookmarks.append.assert_called_with(mo_Bookmark.return_value)
        mo_viewbox.addItem.assert_called_with(mo_bookmarks.__getitem__())
        self.assertEqual(mo_Bookmark.call_args[1]["message"], self.id())
        mo_getMultiLineText.assert_called()

    def test_Plot_Plot_Shortcut_Key_Alt_I(self):
        """
        ...
        """
        bookmark_btn_previous_state = self.plot.bookmark_btn.isFlat()
        QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Alt+I"))  # ToDo is unittest???
        self.assertEqual(self.plot.bookmark_btn.isFlat(), not bookmark_btn_previous_state)

    def test_Plot_Plot_Shortcut_Key_Ctrl_G(self):
        """
        ...
        """
        offset = 10.0
        scale = 100.0
        y_bottom = -offset * scale / 100
        expected_y_range = (y_bottom, y_bottom + scale)
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()
        self.plot.channel_selection.topLevelItem(0).setSelected(True)
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.plot.ScaleDialog") as mo_ScaleDialog:
            mo_ScaleDialog.return_value.exec.return_value = True
            mo_ScaleDialog.return_value.offset.value.return_value = offset
            mo_ScaleDialog.return_value.scaling.value.return_value = scale

            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+G"))

        x = 5
        mo_ScaleDialog.return_value.exec.assert_called()
        self.assertTupleEqual(self.plot.plot.signals[0].y_range, expected_y_range)

    def test_Plot_Plot_Shortcut_Keys_C__Ctrl_C__Ctrl_Shift_C(self):
        """
        Test Scope:
            -
        Events:
            -
        Evaluate:
            -
        """

        with mock.patch("asammdf.gui.widgets.plot.ChannelsTreeWidget.keyPressEvent") as mo_keyPressEvent:
            with self.subTest("test_Key_C"):
                # Event
                QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_C)
                # Evaluate
                mo_keyPressEvent.assert_called()

            with self.subTest("test_Ctrl_C"):
                # Event
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+C"))
                # Evaluate
                mo_keyPressEvent.assert_called()

            with self.subTest("test_Ctrl_Shift_C"):
                # Event
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+Shift+C"))
                # Evaluate
                mo_keyPressEvent.assert_called()

    def test_Plot_Plot_Shortcut_Keys_Ctrl_V__Ctrl_Shift_V(self):
        """
        ...
        """
        # Setup
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()
        with mock.patch("asammdf.gui.widgets.plot.ChannelsTreeWidget.keyPressEvent") as mo_keyPressEvent:
            with self.subTest("test_Ctrl_V"):
                # Event
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+V"))
                # Evaluate
                mo_keyPressEvent.assert_called()

            with self.subTest("test_Ctrl_Shift_V"):
                # Event
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+Shift+V"))
                # Evaluate
                mo_keyPressEvent.assert_called()

    def test_Plot_Plot_Shortcut_Key_Ctrl_BracketLeft(self):
        """
        ...
        """
        # Setup
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()
        initial_size = self.plot.font().pointSize()
        with mock.patch("asammdf.gui.widgets.plot.ChannelsTreeWidget.set_font_size") as mo_set_font_size:
            with mock.patch.object(self.plot.plot, "y_axis"):
                with mock.patch.object(self.plot.plot, "x_axis"):
                    # Event
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+["))
        # Evaluate
        self.assertEqual(initial_size, self.plot.font().pointSize() + 1)

    def test_Plot_Plot_Shortcut_Key_Ctrl_BracketRight(self):
        """
        ...
        """
        # Setup
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()
        initial_size = self.plot.font().pointSize()
        with mock.patch("asammdf.gui.widgets.plot.ChannelsTreeWidget.set_font_size") as mo_set_font_size:
            with mock.patch.object(self.plot.plot, "y_axis"):
                with mock.patch.object(self.plot.plot, "x_axis"):
                    # Event
                    QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence("Ctrl+]"))
        # Evaluate
        self.assertEqual(initial_size, self.plot.font().pointSize() - 1)

    def test_Plot_Plot_Shortcut_Key_Backspace(self):
        """
        ...
        """
        # Setup
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()
        with mock.patch.object(self.plot, "undo_zoom") as mo_undo_zoom:
            # Event
            QtTest.QTest.keyClick(self.plot, QtCore.Qt.Key_Backspace)
        # Evaluate
        mo_undo_zoom.assert_called()

    def test_Plot_Plot_Shortcut_Key_Shift_Backspace(self):
        """
        ...
        """
        # Setup
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()
        with mock.patch.object(self.plot, "redo_zoom") as mo_redo_zoom:
            # Event
            QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence(QtGui.Qt.SHIFT + QtGui.Qt.Key_Backspace))
        # Evaluate
        mo_redo_zoom.assert_called()

    def test_Plot_Plot_Shortcut_Key_Shift_W(self):
        """
        ...
        """
        # Setup
        # Adds one channel to plot
        self.widget.add_new_channels([self.widget.channels_tree.topLevelItem(35).name], self.plot)
        self.processEvents()
        with mock.patch.object(self.plot.plot, "set_y_range") as mo_set_y_range:
            with mock.patch.object(self.plot.plot, "viewbox") as mo_viewbox:
                # Event
                QtTest.QTest.keySequence(self.plot, QtGui.QKeySequence(QtGui.Qt.SHIFT + QtGui.Qt.Key_W))
        # Evaluate
        mo_set_y_range.assert_called()
        mo_viewbox.setXRange.assert_called()
        self.assertFalse(self.plot.undo_btn.isEnabled())

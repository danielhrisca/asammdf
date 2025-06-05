#!/usr/bin/env python\

import pathlib
from unittest import mock

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QKeySequence, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QTreeWidgetItemIterator

from test.asammdf.gui.test_base import OpenMDF, Pixmap
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget


class TestTreeWidgetShortcuts(TestFileWidget):
    def setUp(self):
        """
        Events:
         - Open measurement file

        Returns
        -------

        """
        super().setUp()
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        self.tw = self.widget.channels_tree  # TreeWidget

        # get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.tw))

    def test_toggle_select_channel_shortcut(self):
        """
        Test scope:
            Check if by pressing key Space, check status of tree item is changed

        Events:
            - Select all even channels and press key Space
            - Select the same channels and press again key Space

        Evaluate:
            - Evaluate that check state of all channels is "Unchecked" at start
            - Evaluate that check state of even channels is "Checked" after
                even channels became selected and key Space is pressed
            - Evaluate that check state of all channels is "Unchecked" after
                even channels became selected second time and key Space is pressed
        Returns
        -------

        """
        items_count = self.tw.topLevelItemCount()
        for _ in range(items_count - 1):
            # Evaluate that all items is ...
            self.assertEqual(self.tw.topLevelItem(_).checkState(0), Qt.CheckState.Unchecked)

            # Select all impar items and hit the Space key
            if _ % 2 == 1:
                self.tw.topLevelItem(_).setSelected(True)
                QTest.keySequence(self.tw, QKeySequence(self.shortcuts["toggle_check_state"]))
                self.tw.topLevelItem(_).setSelected(False)

        self.processEvents()

        # Evaluate that all impar items are checked
        for _ in range(items_count - 1):
            if _ % 2 == 1:
                self.assertEqual(self.tw.topLevelItem(_).checkState(0), Qt.CheckState.Checked)

                # Select item and hit again the Space key
                self.tw.topLevelItem(_).setSelected(True)
                QTest.keySequence(self.tw, QKeySequence(self.shortcuts["toggle_check_state"]))
                self.tw.topLevelItem(_).setSelected(False)
            else:
                self.assertEqual(self.tw.topLevelItem(_).checkState(0), Qt.CheckState.Unchecked)

        # Evaluate that all items are unchecked
        for _ in range(items_count - 1):
            self.assertEqual(self.tw.topLevelItem(_).checkState(0), Qt.CheckState.Unchecked)


class TestChannelsTreeWidgetShortcuts(TestPlotWidget):
    """
    Tests for Ctrl+C, Ctrl+V was tested in test_PlotWidget_Shortcuts
    """

    def setUp(self):
        """
        Events:
            - Open measurement file
            - Create a plot window
        Returns
        -------

        """
        super().setUp()
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))
        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")

        self.ctw = self.widget.mdi_area.subWindowList()[0].widget().channel_selection  # ChannelsTreeWidget

        # get shortcuts
        self.assertIsNotNone(self.load_shortcuts_from_json_file(self.ctw))

    def test_delete_shortcut(self):
        """
        Test Scope:
            - Ensure that key Delete will remove selected channels.

        Events:
            - Add some channels to plot.
            - Select first item and press key Delete.
            - Select all items and press key delete.

        Evaluate:
            - Evaluate that there are all items added to plot.
            - Evaluate that first item is removed from a list after it became selected and key Delete is pressed.
            - Evaluate that all items after they became selected are removed by pressing key delete.
        Returns
        -------

        """

        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]))
        channel_count = self.ctw.topLevelItemCount()
        # Click on last channel
        channel_0 = self.channels.pop()
        self.mouseClick_WidgetItem(channel_0)
        # Press key Delete
        QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["delete_items"]))

        # Evaluate
        self.assertEqual(self.ctw.topLevelItemCount(), channel_count - 1)
        iterator = QTreeWidgetItemIterator(self.ctw)
        while item := iterator.value():
            self.assertNotEqual(channel_0, item)
            iterator += 1

        # select all items
        QTest.keySequence(self.ctw, QKeySequence("Ctrl+A"))
        # Press key Delete
        QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["delete_items"]))

        # Evaluate
        self.assertEqual(self.ctw.topLevelItemCount(), 0)

    def test_add_pattern_based_channel_group_shortcut(self):
        """
        Test Scope:
            Check if a new pattern-based channel group can be created by pressing shortcut "Ctrl+Insert"

        Events:
            - Press "Ctrl+Insert"
            - Simulate dialog window fill with:
                    "pattern": f"*{Filter}*",
                    "match_type": "Wildcard",
                    "case_sensitive": False,
                    "filter_type": "Unspecified",
                    "filter_value": 0.0,
                    "raw": False,
                    "ranges": [],
                    "name": "Matrix",
                    "integer_format": "phys".

        Evaluate:
            - Evaluate that method exec_() of AdvancedSearch object was called.
            - Evaluate that in plot channel selection exists a new group with specific name and pattern dict.
            - Evaluate that group contains all channels from measurement with a specific sequence of string
                in their name.
        """
        group_name = "Matrix"
        result = {
            "pattern": f"*{group_name}*",
            "match_type": "Wildcard",
            "case_sensitive": False,
            "filter_type": "Unspecified",
            "filter_value": 0.0,
            "raw": False,
            "ranges": [],
            "name": f"{group_name}",
            "integer_format": "phys",
        }
        # mock for QInputDialog object
        with mock.patch("asammdf.gui.widgets.tree.AdvancedSearch") as mo_AdvancedSearch:
            mo_AdvancedSearch.return_value.result = result
            # Press Ctrl+Insert
            QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["add_pattern_based_group"]))
        # Evaluate
        mo_AdvancedSearch.return_value.exec_.assert_called()
        self.assertEqual(1, self.ctw.topLevelItemCount())
        group = self.ctw.topLevelItem(0)
        self.assertEqual(group.name, result["name"])
        self.assertDictEqual(group.pattern, result)
        # Store all channels names from a group in a list
        group_channels_name = [channel.name for channel in group.get_all_channel_items()]
        # Store all channels with specific pattern in their names in a list
        items = []
        with OpenMDF(self.measurement_file) as mdf:
            items.extend(ch.name for ch in mdf.iter_channels() if group_name.upper() in ch.name.upper())
        self.assertEqual(len(group_channels_name), len(items))
        for channel_name in group_channels_name:
            self.assertIn(channel_name, items)
            # To avoid duplicates
            items.remove(channel_name)

    def test_add_channel_group(self):
        """
        Test Scope:
            Check if a new channel group can be created by pressing shortcut "Shift+Insert"
        Events:
            - Press "Shift+Insert"
            - Type some text and press "Ok"
        Evaluate:
            - Evaluate that method getText() of QInputDialog object was called
            - Evaluate that in plot channel selection exist new group with specific name
        """
        # mock for QInputDialog object
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog") as mo_QInputDialog:
            mo_QInputDialog.getText.return_value = (self.id(), True)
            # Press Shift+Insert
            QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["add_channel_group"]))
        # Evaluate
        mo_QInputDialog.getText.assert_called()
        self.assertEqual(self.ctw.topLevelItem(0).name, self.id())

    def test_select_channel_shortcut(self):
        """
        Test scope:
            Ensure that key Space change checks state for selected item.
        Events:
            - Add few channels to plot
            - Click on first channel -> press key Space.
            - Click on rest channels -> press key Space.
            - Click in first channel -> press key Space.

        Evaluate
            - Evaluate that all items are checked by default.
            - Evaluate that by selecting one item from a list and pressing key space, it states became "Unchecked".
            - Evaluate that by selecting all checked items and pressing key space, its states became "Unchecked".
            - Evaluate that by selecting one unchecked item and pressing key space, it states became "Checked".
        Returns
        -------

        """
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]))

        iterator = QTreeWidgetItemIterator(self.ctw)
        while item := iterator.value():
            self.assertEqual(item.checkState(0), Qt.CheckState.Checked)
            iterator += 1
        # Select click on first item -> press key space
        self.mouseClick_WidgetItem(self.ctw.topLevelItem(0))
        QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["toggle_check_state"]))

        self.assertEqual(self.ctw.topLevelItem(0).checkState(0), Qt.CheckState.Unchecked)

        for _ in range(1, self.ctw.topLevelItemCount()):
            self.mouseClick_WidgetItem(self.ctw.topLevelItem(_))
            QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["toggle_check_state"]))

        iterator = QTreeWidgetItemIterator(self.ctw)
        while item := iterator.value():
            self.assertEqual(item.checkState(0), Qt.CheckState.Unchecked)
            iterator += 1

        # Select click on first item -> press key space
        self.mouseClick_WidgetItem(self.ctw.topLevelItem(0))
        QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["toggle_check_state"]))
        self.assertEqual(self.ctw.topLevelItem(0).checkState(0), Qt.CheckState.Checked)

    def test_color_shortcut(self):
        """
        Test Scope:
            - Ensure that channel color is changed.
        Events:
            - Add 2 channels to Plot.
            - Press C
            - Select 1 Channel -> Press C
            - Select 2 Channels -> Press C
        Evaluate:
            - Evaluate that there is successful added channels to plot.
            - Evaluate that color dialog is not open if channel is not selected.
            - Evaluate that channel color is changed only for a selected channel.
        """
        self.assertIsNotNone(self.add_channels([10, 11, 12]))
        with self.subTest("test_WOSelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["set_color"]))
                mo_getColor.assert_not_called()

        with self.subTest("test_1SelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                # Setup
                self.mouseClick_WidgetItem(self.channels[0])
                previous_color = self.channels[0].color.name()
                color = QColor("magenta")
                mo_getColor.return_value = color
                # Event
                QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["set_color"]))
                # Evaluate
                mo_getColor.assert_called()
                self.assertNotEqual(previous_color, self.channels[0].color.name())
                self.assertEqual(color.name(), self.channels[0].color.name())

        with self.subTest("test_allSelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                # Setup
                # Set selected both channels
                QTest.keySequence(self.ctw, QKeySequence("Ctrl+A"))
                # store previous colors of channels
                previous_ch_colors = [channel.color.name() for channel in self.channels]
                color = QColor("cyan") if "#00FFFF" not in previous_ch_colors else QColor("black")
                mo_getColor.return_value = color
                # Event
                QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["set_color"]))
                # Evaluate
                mo_getColor.assert_called()
                for channel in self.channels:
                    self.assertNotIn(channel.color.name(), previous_ch_colors)
                    self.assertEqual(channel.color.name(), color.name())

    def test_copy_display_properties_shortcut(self):
        """
        Test Scope:
            - Ensure that key "Ctrl+Shift+C" can copy display properties of selected item

        Events:
            - Open Plot with one channel
            - Select this channel
            - Press Ctrl+Shift+C

        Evaluate:
            - Evaluate that there is added one item to plot
            - Evaluate that in clipboard is display properties of selected item
        """
        self.assertIsNotNone(self.add_channels([10]))
        # Evaluate precondition
        ch_0_display_properties = self.ctw.topLevelItem(0).get_display_properties()
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QApplication.instance") as mo_instance:
            # Click on a first channel
            self.mouseClick_WidgetItem(self.channels[0])
            # Press Ctrl+Shift+C
            QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["copy_display_properties"]))
        # Evaluate
        mo_instance.return_value.clipboard.return_value.setText.assert_called_with(ch_0_display_properties)

    def test_paste_display_properties_shortcut(self):
        """
        Test Scope:
            - Ensure that key "Ctrl+Shift+V" can paste from clipboard display properties to selected item.

        Events:
            - Open Plot with two channels.
            - Copy display properties of the first item.
            - Select second item -> press Ctrl+Shift+C.

        Evaluate:
            - Evaluate that to plot are added two items.
            - Evaluate that display properties of channels are not identical.
            - Evaluate that after selecting second channel and hit keys "Ctrl+Shift+V",
                the second channel has the same display properties with the first channel.
        """
        self.assertIsNotNone(self.add_channels([10, 11]))
        # Evaluate precondition
        ch_0_display_properties = self.ctw.topLevelItem(0).get_display_properties()
        self.assertNotEqual(ch_0_display_properties, self.ctw.topLevelItem(1).get_display_properties())
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QApplication.instance") as mo_instance:
            mo_instance.return_value.clipboard.return_value.text.return_value = ch_0_display_properties

            # Click on a first channel
            self.mouseClick_WidgetItem(self.channels[1])
            # Press Ctrl+Shift+C
            QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["paste_display_properties"]))
        # Evaluate
        mo_instance.return_value.clipboard.return_value.text.assert_called()
        self.assertEqual(
            self.ctw.topLevelItem(0).get_display_properties(), self.ctw.topLevelItem(1).get_display_properties()
        )

    def test_copy_names_shortcut(self):
        """
        Test Scope:
            - Ensure that key "Ctrl+Shift+V" can paste from clipboard display properties to selected item.

        Events:
            - Open Plot with few channels.
            - Select first item -> press Ctrl+N.
            - Select all items -> press Ctrl+N.

        Evaluate:
            - Evaluate that to plot are added few items.
            - Evaluate that after selecting one channel and pressing key Ctrl+N, in clipboard is added it name.
            - Evaluate that after selecting all channels and pressing key Ctrl+N, in clipboard is added items names.
        Returns
        -------

        """
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]))
        expected_cb_all_items_call = ""
        iterator = QTreeWidgetItemIterator(self.ctw)
        while item := iterator.value():
            expected_cb_all_items_call += "\n" + item.name
            iterator += 1
        expected_cb_all_items_call = expected_cb_all_items_call.split("\n", 1)[1]

        channel_0 = self.ctw.topLevelItem(0)
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QApplication.instance") as mo_instance:
            # Select click on first item -> press key Ctrl+N
            self.mouseClick_WidgetItem(channel_0)
            QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["copy_names"]))
            # Evaluate
            mo_instance.return_value.clipboard.return_value.setText.assert_called_with(self.ctw.topLevelItem(0).name)

            # Select all items
            QTest.keySequence(self.ctw, QKeySequence("Ctrl+A"))
            #  Press key Ctrl+N
            QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["copy_names"]))
            # Evaluate
            mo_instance.return_value.clipboard.return_value.setText.assert_called_with(expected_cb_all_items_call)

    def test_set_color_range_shortcut(self):
        """
        Test Scope:
            Check if color range is triggered after pressing key Ctrl+R.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Display 1 signal on plot.
            - Select signal.
            - Press "Ctrl+R" -> set ranges from 0 to 40% of y value and colors green and red -> apply.
            - Click on unchanged color part of signal on plot.
            - Click on changed color part of signal on plot.
            - Click Ctrl+G to shift plot from and to 40% of y range.
        Evaluate:
            - Evaluate that plot is not black.
            - Evaluate that plot selected channel value has channel color.
            - Evaluate RangeEditor object was called.
            - Evaluate that plot selected channel value area doesn't have red and green colors,
                only the original one, when the cursor does not intersect the red part of the signal.
            - Evaluate that after clicking on part that enter in selected ranges,
                plot selected channel value area has only red and green colors.
            - Evaluate using Y axis scaling if plot is correctly painted:
                from 0 to 40% it is red, from 40% is not affected.
        """
        # Setup
        # add channels to plot
        self.assertIsNotNone(self.add_channels([35]))
        plot = self.widget.mdi_area.subWindowList()[0].widget()
        self.processEvents(0.01)
        if plot.selected_channel_value_btn.isFlat():
            QTest.mouseClick(plot.selected_channel_value_btn, Qt.MouseButton.LeftButton)
        if not plot.bookmark_btn.isFlat():
            QTest.mouseClick(plot.bookmark_btn, Qt.MouseButton.LeftButton)
        self.mouseClick_WidgetItem(self.channels[0])

        plot.plot.set_dots(False)

        # Evaluate
        self.assertTrue(Pixmap.has_color(plot.selected_channel_value.grab(), self.channels[0].color.name()))
        y_range = plot.plot.y_axis.range[1] - plot.plot.y_axis.range[0]
        offset = 40
        red_range = y_range * offset / 100
        green = QColor.fromRgbF(0.000000, 1.000000, 0.000000, 1.000000)
        red = QColor.fromRgbF(1.000000, 0.000000, 0.000000, 1.000000)

        range_editor_result = [
            {
                "background_color": green,
                "font_color": red,
                "op1": "<=",
                "op2": "<=",
                "value1": 0.0,
                "value2": red_range,
            }
        ]

        # Click on channel
        self.mouseClick_WidgetItem(self.channels[0])
        # Sometimes, because small display resolution, mouse click on qItem isn't performed properly on linux
        if not plot.channel_selection.topLevelItem(0).isSelected():
            plot.channel_selection.topLevelItem(0).setSelected(True)
        self.processEvents(0.01)

        with mock.patch("asammdf.gui.widgets.tree.RangeEditor") as mo_RangeEditor:
            mo_RangeEditor.return_value.result = range_editor_result
            mo_RangeEditor.return_value.pressed_button = "apply"
            # Press "Alt+R"
            QTest.keySequence(self.ctw, QKeySequence(self.shortcuts["set_color_range"]))

        # Evaluate
        mo_RangeEditor.assert_called()
        self.assertEqual(self.channels[0].ranges, range_editor_result)

        self.mouseClick_WidgetItem(self.channels[0])
        for _ in range(100):
            self.processEvents(None)

        # Displayed signal
        sig = self.channels[0].signal
        # Coordinates for viewbox, second dot in the center of visual
        min_ = med_ = max_ = None
        for _ in range(len(sig)):
            if sig.samples[_] == 0 and (sig.samples[_ + 10] - sig.samples[_ - 10]) < (sig.samples.max() / 10):
                min_ = _
                break
        for _ in range(min_, len(sig)):
            if sig.samples[_] == red_range:
                med_ = _
                break
        for _ in range(med_, len(sig)):
            if sig.samples[_] == sig.samples.max():
                max_ = _
                break
        x, y, w, h = sig.timestamps[min_], sig.samples[min_], sig.timestamps[med_], sig.samples[med_]
        # Set X and Y ranges for viewbox
        plot.plot.viewbox.setXRange(x, w, padding=0)
        plot.plot.viewbox.setYRange(y, h, padding=0)
        self.processEvents(None)

        # Click in the middle of the plot
        QTest.mouseClick(
            plot.plot.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(int(plot.plot.width() / 2), int(plot.plot.height() / 2)),
        )
        self.processEvents(0.1)
        for _ in range(100):
            self.processEvents(0.01)
        selected_channel_value = plot.selected_channel_value.grab()
        plot_graphics = plot.plot.grab()

        # Evaluate
        self.assertTrue(Pixmap.has_color(selected_channel_value, red))
        self.assertTrue(Pixmap.has_color(selected_channel_value, green))
        self.assertFalse(Pixmap.has_color(selected_channel_value, self.channels[0]))
        self.assertTrue(Pixmap.has_color(plot_graphics, red))
        self.assertFalse(Pixmap.has_color(plot_graphics, self.channels[0]))

        x, y, w, h = sig.timestamps[med_], sig.samples[med_], sig.timestamps[max_], sig.samples[max_]
        # Set X and Y ranges for viewbox
        plot.plot.viewbox.setXRange(x, w, padding=0)
        plot.plot.viewbox.setYRange(y, h, padding=0)
        self.processEvents(0.1)

        # Click in the middle of the plot
        QTest.mouseClick(
            plot.plot.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(plot.plot.width() // 2, plot.plot.height() // 2),
        )
        self.processEvents(1)
        selected_channel_value = plot.selected_channel_value.grab()
        plot_graphics = plot.plot.grab()

        # Evaluate
        self.assertFalse(Pixmap.has_color(selected_channel_value, red))
        self.assertFalse(Pixmap.has_color(selected_channel_value, green))
        self.assertFalse(Pixmap.has_color(plot_graphics, red))
        self.assertTrue(Pixmap.has_color(selected_channel_value, self.channels[0]))
        self.assertTrue(Pixmap.has_color(plot_graphics, self.channels[0]))

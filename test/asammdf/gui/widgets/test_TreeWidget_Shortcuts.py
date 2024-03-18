#!/usr/bin/env python\
import json
import pathlib
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget
from unittest import mock

from PySide6 import QtCore, QtGui, QtTest, QtWidgets


class TestTreeWidgetShortcuts(TestFileWidget):
    def setUp(self):
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))

        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        self.tw = self.widget.channels_tree  # TreeWidget

    def test_TreeWidgetShortcut_Key_Space(self):
        """

        Returns
        -------

        """
        items_count = self.tw.topLevelItemCount()
        for _ in range(items_count - 1):
            # Evaluate that all items is ...
            self.assertEqual(self.tw.topLevelItem(_).checkState(0), QtCore.Qt.CheckState.Unchecked)

            # Select all impar items and hit the Space key
            if _ % 2 == 1:
                self.tw.topLevelItem(_).setSelected(True)
                QtTest.QTest.keyClick(self.tw, QtGui.Qt.Key_Space)
                self.tw.topLevelItem(_).setSelected(False)

        self.processEvents()

        # Evaluate that all impar items are checked
        for _ in range(items_count - 1):
            if _ % 2 == 1:
                self.assertEqual(self.tw.topLevelItem(_).checkState(0), QtCore.Qt.CheckState.Checked)

                # Select item and hit again the Space key
                self.tw.topLevelItem(_).setSelected(True)
                QtTest.QTest.keyClick(self.tw, QtGui.Qt.Key_Space)
                self.tw.topLevelItem(_).setSelected(False)
            else:
                self.assertEqual(self.tw.topLevelItem(_).checkState(0), QtCore.Qt.CheckState.Unchecked)

        # Evaluate that all items are unchecked
        for _ in range(items_count - 1):
            self.assertEqual(self.tw.topLevelItem(_).checkState(0), QtCore.Qt.CheckState.Unchecked)


class TestChannelsTreeWidgetShortcuts(TestPlotWidget):
    def setUp(self):
        # Open measurement file
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))
        # Open measurement file
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        # Select channels -> Press PushButton "Create Window" -> "Plot"
        self.create_window(window_type="Plot")

        self.ctw = self.widget.mdi_area.subWindowList()[0].widget().channel_selection  # ChannelsTreeWidget

    def tearDown(self):
        if self.widget:
            self.widget.destroy()

    def test_ChannelsTreeWidget_Shortcut_Key_Shift_Delete(self):
        """

        Returns
        -------

        """

        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]))
        channel_count = self.ctw.topLevelItemCount()
        # Click on last channel
        channel_0 = self.channels.pop()
        self.mouseClick_WidgetItem(channel_0)
        # Press key Delete
        QtTest.QTest.keyClick(self.ctw, QtCore.Qt.Key_Delete)

        # Evaluate
        self.assertEqual(self.ctw.topLevelItemCount(), channel_count - 1)
        iterator = QtWidgets.QTreeWidgetItemIterator(self.ctw)
        while item := iterator.value():
            self.assertNotEqual(channel_0, item)
            iterator += 1

        # select all items
        QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+A"))
        # Press key Delete
        QtTest.QTest.keyClick(self.ctw, QtCore.Qt.Key_Delete)

        # Evaluate
        self.assertEqual(self.ctw.topLevelItemCount(), 0)

    def test_ChannelsTreeWidget_Shortcut_Key_Ctrl_Insert(self):
        """
        Test Scope:
            Check if a new pattern based channel group can be created by pressing shortcut "Ctrl+Insert"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Mock AdvancedSearch object
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
                    "integer_format": "phys",
        Evaluate:
            - Evaluate that method exec_() of AdvancedSearch object was called
            - Evaluate that in plot channel selection exist new group with specific name and pattern dict
            - Evaluate that group contains all channels from measurement with specific sequence of string in their
            name
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
            # Press Shift+Insert
            QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+Ins"))
        # Evaluate
        mo_AdvancedSearch.return_value.exec_.assert_called()
        self.assertEqual(1, self.ctw.topLevelItemCount())
        group = self.ctw.topLevelItem(0)
        self.assertEqual(group.name, result["name"])
        self.assertDictEqual(group.pattern, result)
        # Store all channels names from a group in a list
        group_channels_name = [channel.name for channel in group.get_all_channel_items()]
        # Store all channels with specific pattern in their names in a list
        ct = self.widget.channels_tree
        items = [
            ct.topLevelItem(_).name
            for _ in range(ct.topLevelItemCount() - 1)
            if group_name.upper() in ct.topLevelItem(_).name.upper()
        ]
        self.assertEqual(len(group_channels_name), len(items))
        for channel_name in group_channels_name:
            self.assertIn(channel_name, items)
            # To avoid duplicates
            items.remove(channel_name)

    def test_ChannelsTreeWidget_Shortcut_Key_Shift_Insert(self):
        """
        Test Scope:
            Check if a new channel group can be created by pressing shortcut "Shift+Insert"
        Events:
            - Open 'FileWidget' with valid measurement.
            - Mock QInputDialog object
            - Press "Shift+Insert"
        Evaluate (0):
            - Evaluate that method getText() of QInputDialog object was called
            - Evaluate that in plot channel selection exist new group with specific name
        """
        # mock for QInputDialog object
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog") as mo_QInputDialog:
            mo_QInputDialog.getText.return_value = (self.id(), True)
            # Press Shift+Insert
            QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Shift+Ins"))
        # Evaluate
        mo_QInputDialog.getText.assert_called()
        self.assertEqual(self.ctw.topLevelItem(0).name, self.id())

    def test_ChannelsTreeWidget_Shortcut_Key_Shift_Space(self):
        """
            - Add few channels to plot
            - Click on channels -> press key Space

            Evaluate
                - Click on space must change check state of selected channel
        Returns
        -------

        """
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]))

        iterator = QtWidgets.QTreeWidgetItemIterator(self.ctw)
        while item := iterator.value():
            self.assertEqual(item.checkState(0), QtCore.Qt.CheckState.Checked)
            iterator += 1
        # Select click on first item -> press key space
        self.mouseClick_WidgetItem(self.ctw.topLevelItem(0))
        QtTest.QTest.keyClick(self.ctw, QtCore.Qt.Key_Space)

        self.assertEqual(self.ctw.topLevelItem(0).checkState(0), QtCore.Qt.CheckState.Unchecked)

        for _ in range(1, self.ctw.topLevelItemCount()):
            self.mouseClick_WidgetItem(self.ctw.topLevelItem(_))
            QtTest.QTest.keyClick(self.ctw, QtCore.Qt.Key_Space)

        iterator = QtWidgets.QTreeWidgetItemIterator(self.ctw)
        while item := iterator.value():
            self.assertEqual(item.checkState(0), QtCore.Qt.CheckState.Unchecked)
            iterator += 1

        # Select click on first item -> press key space
        self.mouseClick_WidgetItem(self.ctw.topLevelItem(0))
        QtTest.QTest.keyClick(self.ctw, QtCore.Qt.Key_Space)
        self.assertEqual(self.ctw.topLevelItem(0).checkState(0), QtCore.Qt.CheckState.Checked)

    def test_Plot_Channel_Selection_Shortcut_Key_C(self):
        """
        Test Scope:
            - Ensure that channel color is changed.
        Events:
            - Open Plot with 2 channels
            - Mock getColor() object
            - Press C
            - Select 1 Channel
            - Press C
            - Select 2 Channels
            - Press C
        Evaluate:
            - Evaluate that color dialog is not open if channel is not selected.
            - Evaluate that channel color is changed only for selected channel
        """
        self.assertIsNotNone(self.add_channels([10, 11, 12]))
        with self.subTest("test_WOSelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                QtTest.QTest.keyClick(self.ctw, QtCore.Qt.Key_C)
                mo_getColor.assert_not_called()

        with self.subTest("test_1SelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                # Setup
                self.mouseClick_WidgetItem(self.channels[0])
                previous_color = self.channels[0].color.name()
                color = QtGui.QColor("magenta")
                mo_getColor.return_value = color
                # Event
                QtTest.QTest.keyClick(self.ctw, QtCore.Qt.Key_C)
                # Evaluate
                mo_getColor.assert_called()
                self.assertNotEqual(previous_color, self.channels[0].color.name())
                self.assertEqual(color.name(), self.channels[0].color.name())

        with self.subTest("test_allSelectedChannel"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                # Setup
                # Set selected both channels
                QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+A"))
                # store previous colors of channels
                previous_ch_colors = [channel.color.name() for channel in self.channels]
                color = QtGui.QColor("cyan") if "#00FFFF" not in previous_ch_colors else QtGui.QColor("black")
                mo_getColor.return_value = color
                # Event
                QtTest.QTest.keyClick(self.ctw, QtCore.Qt.Key_C)
                # Evaluate
                mo_getColor.assert_called()
                for channel in self.channels:
                    self.assertNotIn(channel.color.name(), previous_ch_colors)
                    self.assertEqual(channel.color.name(), color.name())

    def test_Plot_Channel_Selection_Shortcut_Key_Ctrl_C(self):
        """
        Test Scope:
            - Ensure that copied to clipboard channel is pasted into a plot.
        Events:
            - Open Plot with 3 channels
            - Select all channel
            - Press Ctrl+C
        Evaluate:
            - Evaluate in clipboard exist names of all selected channels
        """
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]))
        # Set selected all channels
        QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+A"))
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QApplication.instance") as mo_instance:
            # Press Ctrl+C -> Ctrl+V
            QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+C"))
        self.processEvents()
        # Evaluate that now is three channels available
        mo_set_text = mo_instance.return_value.clipboard.return_value.setText
        mo_set_text.assert_called()
        for channel in self.channels:
            self.assertIn(channel.name, mo_set_text.call_args.args[0])

    def test_Plot_Channel_Selection_Shortcut_Key_Ctrl_V(self):
        """
        Test Scope:
            - Ensure that selected channel is copied to clipboard.
        Events:
            - Open Plot
            - Copy proprieties of one channel
            - Press Ctrl+V
        Evaluate:
            - Evaluate that channel was pasted into a plot
        """
        color = "#00ff00"
        d = {
            "type": "channel",
            "name": self.id(),
            "unit": "V",
            "flags": 0,
            "enabled": True,
            "individual_axis": False,
            "common_axis": False,
            "color": color,
            "computed": False,
            "ranges": [],
            "precision": 3,
            "fmt": "{:.3f}",
            "format": "bin",
            "mode": "raw",
            "y_range": [0.0, 255.0],
            "origin_uuid": None,
            "uuid": "696c55c0e78b",
            "group_index": 1,
            "channel_index": 2,
        }
        clipboard_data = json.dumps([d])
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QApplication.instance") as mo_instance:
            mo_instance.return_value.clipboard.return_value.text.return_value = clipboard_data
            # Press Ctrl+C -> Ctrl+V
            QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+V"))
            mo_instance.return_value.clipboard.return_value.text.assert_called()
        self.processEvents()
        # Evaluate that now is one channel available
        self.assertEqual(1, self.ctw.topLevelItemCount())
        self.assertEqual(self.ctw.topLevelItem(0).name, self.id())
        self.assertEqual(self.ctw.topLevelItem(0).color.name(), color)

    def test_Plot_Channel_Selection_Shortcut_Key_Ctrl_Shift_C(self):
        """
        ...
        """
        self.assertIsNotNone(self.add_channels([10]))
        # Evaluate precondition
        ch_0_display_properties = self.ctw.topLevelItem(0).get_display_properties()
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QApplication.instance") as mo_instance:
            # Click on first channel
            self.mouseClick_WidgetItem(self.channels[0])
            # Press Ctrl+Shift+C
            QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+Shift+C"))
        # Evaluate
        mo_instance.return_value.clipboard.return_value.setText.assert_called_with(ch_0_display_properties)

    def test_Plot_Channel_Selection_Shortcut_Key_Ctrl_Shift_V(self):
        """
        - Add 2 channels to plot
        - Set clipboard text = display_properties of first channel
        - Click on item -> Press Ctrl+V

        Evaluate
            - display_properties of both channels must be equal
        """
        self.assertIsNotNone(self.add_channels([10, 11]))
        # Evaluate precondition
        ch_0_display_properties = self.ctw.topLevelItem(0).get_display_properties()
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QApplication.instance") as mo_instance:
            mo_instance.return_value.clipboard.return_value.text.return_value = self.ctw.topLevelItem(
                0
            ).get_display_properties()

            # Click on first channel
            self.mouseClick_WidgetItem(self.channels[1])
            # Press Ctrl+Shift+C
            QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+Shift+V"))
        # Evaluate
        mo_instance.return_value.clipboard.return_value.text.assert_called()
        self.assertEqual(
            self.ctw.topLevelItem(0).get_display_properties(), self.ctw.topLevelItem(1).get_display_properties()
        )

    def test_ChannelsTreeWidget_Shortcut_Key_Ctrl_N(self):
        """

        Returns
        -------

        """
        self.assertIsNotNone(self.add_channels([10, 11, 12, 13]))
        expected_cb_all_items_call = ""
        iterator = QtWidgets.QTreeWidgetItemIterator(self.ctw)
        while item := iterator.value():
            expected_cb_all_items_call += "\n" + item.name
            iterator += 1
        expected_cb_all_items_call = expected_cb_all_items_call.split("\n", 1)[1]

        channel_0 = self.ctw.topLevelItem(0)
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QApplication.instance") as mo_instance:
            # Select click on first item -> press key Ctrl+N
            self.mouseClick_WidgetItem(channel_0)
            QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+N"))
            # Evaluate
            mo_instance.return_value.clipboard.return_value.setText.assert_called_with(self.ctw.topLevelItem(0).name)

            # Select all items
            QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+A"))
            #  Press key Ctrl+N
            QtTest.QTest.keySequence(self.ctw, QtGui.QKeySequence("Ctrl+N"))
            # Evaluate
            mo_instance.return_value.clipboard.return_value.setText.assert_called_with(expected_cb_all_items_call)

    # def test_ChannelsTreeWidget_Shortcut_Ctrl_Key_R(self) tested in <test_PlotWidget_Shortcut.py>

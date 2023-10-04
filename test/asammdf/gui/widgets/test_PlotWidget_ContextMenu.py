#!/usr/bin/env python
import json
from json import JSONDecodeError
import re
import sys
from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget
import unittest
from unittest import mock
from unittest.mock import ANY

from PySide6 import QtCore, QtTest, QtWidgets


class TestContextMenu(TestPlotWidget):
    # Note: Test Plot Widget through FileWidget.

    def setUp(self):
        super().setUp()

        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        self.create_window(window_type="Plot", channels_indexes=[10, 11, 12, 13, 15])
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)

        # Drag and Drop channel from FileWidget.channel_tree to Plot
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()
        self.plot_channel_a = self.plot.channel_selection.topLevelItem(0)
        self.plot_channel_a.setCheckState(self.Column.NAME, QtCore.Qt.CheckState.Checked)
        self.plot_channel_b = self.plot.channel_selection.topLevelItem(1)
        self.plot_channel_b.setCheckState(self.Column.NAME, QtCore.Qt.CheckState.Checked)
        self.plot_channel_c = self.plot.channel_selection.topLevelItem(2)
        self.plot_channel_c.setCheckState(self.Column.NAME, QtCore.Qt.CheckState.Checked)
        self.plot_channel_d = self.plot.channel_selection.topLevelItem(3)
        self.plot_channel_d.setCheckState(self.Column.NAME, QtCore.Qt.CheckState.Checked)
        self.plot_channel_e = self.plot.channel_selection.topLevelItem(4)
        self.plot_channel_e.setCheckState(self.Column.NAME, QtCore.Qt.CheckState.Checked)

        self.processEvents()

    def test_Action_SearchItem_Cancel(self):
        """
        Test Scope:
            Ensure that action 'Search Item' call 'QtWidgets.QInputDialog.getText' in order to get a channel name.
            The Dialog is canceled and nothing happens.
        Events:
            - Open Context Menu
            - Trigger 'Search Item' action
                - Simulate that the dialog is canceled.
        Evaluate:
            - Ensure that there is no selected channel because
            performing Right Click for ContextMenu will clear selection of the tree.
        """
        QtTest.QTest.keyClick(self.plot.channel_selection.viewport(), QtCore.Qt.Key_Down)
        self.assertEqual(1, len(self.plot.channel_selection.selectedItems()))

        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = None, None
            self.context_menu(action_text="Search item")

        self.assertEqual(0, len(self.plot.channel_selection.selectedItems()))

    def test_Action_SearchItem_NonexistentChannel(self):
        """
        Test Scope:
            Ensure that action 'Search Item' call 'QtWidgets.QInputDialog.getText' in order to get a channel name.
            The Dialog is filled with non-existing channel name.
        Events:
            - Open Context Menu
            - Trigger 'Search Item' action
                - Simulate that the dialog is filled with non-existing channel name.
        Evaluate:
            - Ensure that selection is cleared.
            - Ensure that MessageBox informs user.
        """
        QtTest.QTest.keyClick(self.plot.channel_selection.viewport(), QtCore.Qt.Key_Down)
        self.assertEqual(1, len(self.plot.channel_selection.selectedItems()))

        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText, mock.patch(
            "asammdf.gui.widgets.tree.MessageBox.warning"
        ) as mo_warning:
            mo_getText.return_value = self.id(), True
            self.context_menu(action_text="Search item")

        self.assertEqual(0, len(self.plot.channel_selection.selectedItems()))
        mo_warning.assert_called_with(self.plot.channel_selection, "No matches found", ANY)

    def test_Action_SearchItem_ExistentChannel(self):
        """
        Test Scope:
            Ensure that action 'Search Item' call 'QtWidgets.QInputDialog.getText' in order to get a channel name.
            The Dialog is filled with existing channel name.
            Found channel is selected on tree
        Events:
            - Open Context Menu
            - Trigger 'Search Item' action
                - Simulate that the dialog is filled with existing channel name.
        Evaluate:
            - Ensure that selection is set on the found channel.
        """
        QtTest.QTest.keyClick(self.plot.channel_selection.viewport(), QtCore.Qt.Key_Down)
        self.assertEqual(1, len(self.plot.channel_selection.selectedItems()))

        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText, mock.patch(
            "asammdf.gui.widgets.tree.MessageBox.warning"
        ) as mo_warning:
            mo_getText.return_value = self.plot_channel_b.text(self.Column.NAME), True
            self.context_menu(action_text="Search item")

        self.assertEqual(1, len(self.plot.channel_selection.selectedItems()))
        mo_warning.assert_not_called()

    def test_Action_AddChannelGroup_Cancel(self):
        """
        Test Scope:
            - Ensure that action 'Add channel group' will add new item (type Group) on tree.
        Events:
            - Open Context Menu
            - Trigger 'Add channel group' action
                - Simulate that the dialog is canceled.
        """
        channels_nr = self.plot.channel_selection.topLevelItemCount()

        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "FirstGroup", None
            self.context_menu(action_text="Add channel group [Shift+Insert]")

        self.assertEqual(channels_nr, self.plot.channel_selection.topLevelItemCount())

    def test_Action_AddChannelGroup(self):
        """
        Test Scope:
            - Ensure that action 'Add channel group' will add new item (type Group) on tree.
        Events:
            - Open Context Menu
            - Trigger 'Add channel group' action
                - Simulate that the dialog is confirmed.
        Evaluate:
            - Check if there is one extra element in tree.
        """
        channels_nr = self.plot.channel_selection.topLevelItemCount()

        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "FirstGroup", True
            self.context_menu(action_text="Add channel group [Shift+Insert]")

        self.assertEqual(channels_nr + 1, self.plot.channel_selection.topLevelItemCount())

    def test_Action_CopyNames(self):
        """
        Test Scope:
            - Ensure that action 'Copy Names' will place channel names in clipboard.
        Events:
            - Select one channel
            - Open Context Menu
            - Trigger 'Copy Names' action
            - Select two channels
            - Open Context Menu
            - Trigger 'Copy Names' action
        Evaluate:
            - Evaluate that channels names are placed in clipboard.
        """

        with self.subTest("1Channel"):
            position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
            self.context_menu(action_text="Copy names [Ctrl+N]", position=position)

            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            self.assertEqual(self.plot_channel_a.text(self.Column.NAME), clipboard)

        with self.subTest("2Channels"):
            self.plot_channel_a.setSelected(True)
            self.plot_channel_b.setSelected(True)
            position_1 = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()
            self.context_menu(action_text="Copy names [Ctrl+N]", position=position_1)

            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            channels = (self.plot_channel_a.text(self.Column.NAME), self.plot_channel_b.text(self.Column.NAME))
            self.assertEqual("\n".join(channels), clipboard)

    def test_Action_CopyNamesAndValues(self):
        """
        Test Scope:
            - Ensure that action 'Copy names and value' will place channel names in clipboard.
        Events:
            - Select one channel
            - Open Context Menu
            - Trigger 'Copy names and value' action
            - Select two channels
            - Open Context Menu
            - Trigger 'Copy names and value' action
        Evaluate:
            - Evaluate that channels names are placed in clipboard.
        """

        with self.subTest("1Channel"):
            position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
            self.context_menu(action_text="Copy names and values", position=position)

            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            pattern_name = re.escape(self.plot_channel_a.text(self.Column.NAME))
            pattern_unit = re.escape(self.plot_channel_a.text(self.Column.UNIT))
            self.assertRegex(clipboard, expected_regex=f"{pattern_name}, t = \d+[.]?\d+s, \d+[.]?\d+{pattern_unit}")

        with self.subTest("2Channels"):
            self.plot_channel_a.setSelected(True)
            self.plot_channel_b.setSelected(True)
            position_1 = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()
            self.context_menu(action_text="Copy names and values", position=position_1)

            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            expected_regex = []
            for channel in (self.plot_channel_a, self.plot_channel_b):
                pattern_name = re.escape(channel.text(self.Column.NAME))
                pattern_unit = re.escape(channel.text(self.Column.UNIT))
                expected_regex.append(f"{pattern_name}, t = \d+[.]?\d+s, \d+[.]?\d+{pattern_unit}")

            self.assertRegex(clipboard, "\\n".join(expected_regex))

    def test_Action_CopyDisplayProperties_Channel(self):
        """
        Test Scope:
            - Ensure that channel display properties are copied in clipboard.
        Events:
            - Select one channel
            - Open Context Menu
            - Trigger 'Copy display properties' action
        Evaluate:
            - Evaluate that channel display properties are stored in clipboard in json format.
        """
        position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
        self.context_menu(action_text="Copy display properties [Ctrl+Shift+C]", position=position)

        clipboard = QtWidgets.QApplication.instance().clipboard().text()
        try:
            content = json.loads(clipboard)
        except JSONDecodeError:
            self.fail("Clipboard Content cannot be decoded as JSON content.")
        else:
            self.assertIsInstance(content, dict)

    def test_Action_PasteDisplayProperties_Channel(self):
        """
        Test Scope:
            - Ensure that channel display properties can be pasted over different channel.
        Events:
            - Select one channel
            - Open Context Menu
            - Trigger 'Copy display properties' action
            - Select a different channel
            - Open Context Menu
            - Trigger 'Paste display properties' action
            - Select dst channel
            - Open Context Menu
            - Trigger 'Copy display properties' action
        Evaluate:
            - Evaluate that display properties are transferred from one channel to another
        """
        action_copy = "Copy display properties [Ctrl+Shift+C]"
        action_paste = "Paste display properties [Ctrl+Shift+V]"

        position_src = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
        self.context_menu(action_text=action_copy, position=position_src)

        channel_a_properties = QtWidgets.QApplication.instance().clipboard().text()

        # Paste
        position_dst = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()
        self.context_menu(action_text=action_paste, position=position_dst)

        # Copy
        position_src = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()
        self.context_menu(action_text=action_copy, position=position_src)

        channel_b_properties = QtWidgets.QApplication.instance().clipboard().text()

        self.assertEqual(channel_a_properties, channel_b_properties)

    def test_Action_CopyChannelStructure_Channel(self):
        """
        Test Scope:
            - Ensure that channel structure is copied in clipboard.
        Events:
            - Select one channel
            - Open Context Menu
            - Trigger 'Copy channel structure' action
        Evaluate:
            - Evaluate that channel structure is stored in clipboard in json format.
        """
        position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
        self.context_menu(action_text="Copy channel structure [Ctrl+C]", position=position)

        clipboard = QtWidgets.QApplication.instance().clipboard().text()
        try:
            content = json.loads(clipboard)
        except JSONDecodeError:
            self.fail("Clipboard Content cannot be decoded as JSON content.")
        else:
            self.assertIsInstance(content, list)
            for channel_properties in content:
                self.assertIsInstance(channel_properties, dict)

    def test_Action_PasteChannelStructure_Channel(self):
        """
        Test Scope:
            - Ensure that channel and structure is duplicated and structure is kept.
        Events:
            - Select one channel
            - Open Context Menu
            - Trigger 'Copy display properties' action
            - Trigger 'Copy channel structure' action
            - Open Context Menu
            - Trigger 'Paste channel structure' action
            - Select dst channel
            - Open Context Menu
            - Trigger 'Copy display properties' action
        Evaluate:
            - Evaluate that channel is duplicated and structure is kept.
        """
        action_copy_dsp_properties = "Copy display properties [Ctrl+Shift+C]"
        action_copy = "Copy channel structure [Ctrl+C]"
        action_paste = "Paste channel structure [Ctrl+V]"

        position_src = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
        # Copy Channel Structure
        self.context_menu(action_text=action_copy, position=position_src)

        channels_count = self.plot.channel_selection.topLevelItemCount()

        # Paste Channel Structure
        self.context_menu(action_text=action_paste)

        self.assertEqual(channels_count + 1, self.plot.channel_selection.topLevelItemCount())

        channels = self.plot.channel_selection.findItems(
            self.plot_channel_a.text(self.Column.NAME), QtCore.Qt.MatchFlags()
        )
        self.assertEqual(2, len(channels))

        # Copy DSP Properties
        channel_properties = []
        for channel in channels:
            position_src = self.plot.channel_selection.visualItemRect(channel).center()
            self.context_menu(action_text=action_copy_dsp_properties, position=position_src)
            channel_properties.append(QtWidgets.QApplication.instance().clipboard().text())

        self.assertEqual(channel_properties[0], channel_properties[1])

        self.processEvents(0.2)

    @unittest.skipIf(sys.platform != "win32", "Timers cannot be started/stopped from another thread.")
    def test_Action_CopyChannelStructure_Group(self):
        """
        Test Scope:
            - Ensure that channel structure is copied in clipboard.
        Events:
            - Select one group
            - Open Context Menu
            - Trigger 'Copy channel structure' action
        Evaluate:
            - Evaluate that group channel structure is stored in clipboard in json format.
        """
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "FirstGroup", True
            self.context_menu(action_text="Add channel group [Shift+Insert]")

        # Add Channels to Group
        group_channel = self.plot.channel_selection.findItems("FirstGroup", QtCore.Qt.MatchFlags())[0]
        self.move_channel_to_group(src=self.plot_channel_a, dst=group_channel)

        position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
        self.context_menu(action_text="Copy channel structure [Ctrl+C]", position=position)

        clipboard = QtWidgets.QApplication.instance().clipboard().text()
        try:
            content = json.loads(clipboard)
        except JSONDecodeError:
            self.fail("Clipboard Content cannot be decoded as JSON content.")
        else:
            self.assertIsInstance(content, list)
            for channel_properties in content:
                self.assertIsInstance(channel_properties, dict)

    @unittest.skipIf(sys.platform != "win32", "Timers cannot be started/stopped from another thread.")
    def test_Action_PasteChannelStructure_Group(self):
        """
        Test Scope:
            - Ensure that channel and structure is duplicated and structure is kept.
        Events:
            - Create group
            - Add channels to group
            - Open Context Menu
            - Trigger 'Copy channel structure' action
            - Trigger 'Paste channel structure' action over new group
        Evaluate:
            - Evaluate that group channel is duplicated and structure is kept.
        """
        action_copy = "Copy channel structure [Ctrl+C]"
        action_paste = "Paste channel structure [Ctrl+V]"

        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "FirstGroup", True
            self.context_menu(action_text="Add channel group [Shift+Insert]")

        # Add Channels to Group
        group_channel = self.plot.channel_selection.findItems("FirstGroup", QtCore.Qt.MatchFlags())[0]
        group_channel.setExpanded(True)
        self.move_channel_to_group(src=self.plot_channel_c, dst=group_channel)
        self.move_channel_to_group(src=self.plot_channel_d, dst=group_channel)

        # Copy Channel Structure
        position_src = self.plot.channel_selection.visualItemRect(group_channel).center()
        self.context_menu(action_text=action_copy, position=position_src)

        channels_count = self.plot.channel_selection.topLevelItemCount()

        # Paste Channel Structure
        position_src = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
        self.context_menu(action_text=action_paste, position=position_src)
        self.processEvents(0.1)

        self.assertEqual(channels_count + 1, self.plot.channel_selection.topLevelItemCount())
        self.assertEqual(2, len(self.plot.channel_selection.findItems("FirstGroup", QtCore.Qt.MatchFlags())))

    def test_Action_EnableDisable_DeactivateGroups_Channel(self):
        """
        Test Scope:
            - Ensure that group items can be disabled from Context Menu.
        Events:
            - Select 1 channel
            - Open Context Menu
            - Trigger "Enable/Disable -> Deactivate Groups" action
        Expected:
            - Individual channels that are not part of group should not be affected
        """
        # Events
        position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
        self.context_menu(action_text="Deactivate groups", position=position)

        # Evaluate
        self.assertEqual(False, self.plot_channel_a.isDisabled())

    @unittest.skipIf(sys.platform != "win32", "Timers cannot be started/stopped from another thread.")
    def test_Action_EnableDisable_DeactivateGroups_Group(self):
        """
        Test Scope:
            - Ensure that group items can be disabled from Context Menu.
        Events:
            - Add two groups
            - Add channels to group
            - Add group in group
            - Select 1 group
            - Open Context Menu
            - Trigger "Enable/Disable -> Deactivate Groups" action
        Expected:
            - Group Items should be disabled
            - Sub-groups should be disabled
        """
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "Group_A", True
            self.context_menu(action_text="Add channel group [Shift+Insert]")
            mo_getText.return_value = "Group_B", True
            self.context_menu(action_text="Add channel group [Shift+Insert]")

        # Add Channels to Group
        group_a_channel = self.plot.channel_selection.findItems("Group_A", QtCore.Qt.MatchFlags())[0]
        group_a_channel.setExpanded(True)
        self.move_channel_to_group(src=self.plot_channel_a, dst=group_a_channel)
        self.move_channel_to_group(src=self.plot_channel_b, dst=group_a_channel)

        # Add Channels to Group
        group_b_channel = self.plot.channel_selection.findItems("Group_B", QtCore.Qt.MatchFlags())[0]
        group_b_channel.setExpanded(True)
        self.move_channel_to_group(src=self.plot_channel_c, dst=group_b_channel)
        self.move_channel_to_group(src=self.plot_channel_d, dst=group_b_channel)

        self.move_channel_to_group(src=group_b_channel, dst=group_a_channel)
        group_a_channel.setExpanded(True)
        group_b_channel.setExpanded(True)

        position = self.plot.channel_selection.visualItemRect(group_b_channel).center()
        self.context_menu(action_text="Deactivate groups", position=position)

        # Evaluate
        self.assertEqual(False, group_a_channel.isDisabled())
        self.assertEqual(True, group_b_channel.isDisabled())
        for child_index in range(group_b_channel.childCount()):
            child = group_b_channel.child(child_index)
            self.assertEqual(True, child.isDisabled())
        for child_index in range(group_a_channel.childCount()):
            child = group_a_channel.child(child_index)
            if child.type() == child.Channel:
                self.assertEqual(False, child.isDisabled())

        position = self.plot.channel_selection.visualItemRect(group_a_channel).center()
        self.context_menu(action_text="Deactivate groups", position=position)

        # Evaluate
        self.assertEqual(True, group_a_channel.isDisabled())
        self.assertEqual(True, group_b_channel.isDisabled())
        for child_index in range(group_b_channel.childCount()):
            child = group_b_channel.child(child_index)
            self.assertEqual(True, child.isDisabled())
        for child_index in range(group_a_channel.childCount()):
            child = group_a_channel.child(child_index)
            self.assertEqual(True, child.isDisabled())

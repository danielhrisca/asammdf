#!/usr/bin/env python
import json
from json import JSONDecodeError
import pathlib
import re
from unittest import mock
from unittest.mock import ANY

from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget


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
        QtTest.QTest.keyClick(self.plot.channel_selection.viewport(), QtCore.Qt.Key.Key_Down)
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
        QtTest.QTest.keyClick(self.plot.channel_selection.viewport(), QtCore.Qt.Key.Key_Down)
        self.assertEqual(1, len(self.plot.channel_selection.selectedItems()))

        with (
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText,
            mock.patch("asammdf.gui.widgets.tree.MessageBox.warning") as mo_warning,
        ):
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
        QtTest.QTest.keyClick(self.plot.channel_selection.viewport(), QtCore.Qt.Key.Key_Down)
        self.assertEqual(1, len(self.plot.channel_selection.selectedItems()))

        with (
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText,
            mock.patch("asammdf.gui.widgets.tree.MessageBox.warning") as mo_warning,
        ):
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
            self.context_menu(action_text="Add channel group")

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
            self.context_menu(action_text="Add channel group")

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
            self.context_menu(action_text="Copy names", position=position)

            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            self.assertEqual(self.plot_channel_a.text(self.Column.NAME), clipboard)

        with self.subTest("2Channels"):
            self.plot_channel_a.setSelected(True)
            self.plot_channel_b.setSelected(True)
            position_1 = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()
            self.context_menu(action_text="Copy names", position=position_1)

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
            self.assertRegex(clipboard, expected_regex=f"{pattern_name}, t = \\d+[.]?\\d+s, \\d+[.]?\\d+{pattern_unit}")

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
                expected_regex.append(f"{pattern_name}, t = \\d+[.]?\\d+s, \\d+[.]?\\d+{pattern_unit}")

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
        self.context_menu(action_text="Copy display properties", position=position)

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
        action_copy = "Copy display properties"
        action_paste = "Paste display properties"

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
        self.context_menu(action_text="Copy channel structure", position=position)

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
        action_copy_dsp_properties = "Copy display properties"
        action_copy = "Copy channel structure"
        action_paste = "Paste channel structure"

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

    def test_Menu_EnableDisable_Action_DeactivateGroups_Channel(self):
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

    def test_Menu_EnableDisable_Action_DisableAllButThis(self):
        """
        Test Scope:
            - Ensure that all channels are disabled except the item selected.
        Events:
            - Open Context Menu
            - Select action "Disable all but this" from sub-menu "Enable/disable"
        Expected:
            - Evaluate that all channels are disabled except selected item.
        """
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "A", True
            self.context_menu(action_text="Add channel group")

        positions_src = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()
        self.context_menu(action_text="Disable all but this", position=positions_src)

        # Evaluate
        self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_b.checkState(self.Column.NAME))
        count = self.plot.channel_selection.topLevelItemCount()
        for i in range(count):
            item = self.plot.channel_selection.topLevelItem(i)
            if item.type() != item.Info and item != self.plot_channel_b:
                self.assertEqual(QtCore.Qt.CheckState.Unchecked, item.checkState(self.Column.NAME))

    def test_Menu_ShowHide_Action_HideDisabledItems(self):
        """
        Test Scope:
            - Ensure that item is hidden from channel selection when is disabled.
        Events:
            - Disable 1 channel by key Space
            - Disable 1 channel by mouseClick on item CheckBox
        Evaluate:
            - Evaluate that items that are unchecked are not present anymore on channel selection
        """
        self.context_menu(action_text="Hide disabled items")

        with self.subTest("DisableBySpace"):
            # Select one channel
            self.mouseClick_WidgetItem(self.plot_channel_a)
            # Event
            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Space)
            # Evaluate
            self.assertTrue(self.plot_channel_a.isHidden())

        with self.subTest("DisableByClick"):
            pos = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()
            # Magic Number to detect center of checkbox
            pos = QtCore.QPoint(28, pos.y())
            # Event
            QtTest.QTest.mouseClick(
                self.plot.channel_selection.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                pos,
            )
            # Evaluate
            self.assertTrue(self.plot_channel_b.isHidden())

    def test_Menu_ShowHide_Action_ShowDisabledItems(self):
        """
        Test Scope:
            - Ensure that item is showed on channel selection when is disabled.
        Events:
            - Disable 1 channel by key Space
            - Disable 1 channel by mouseClick on item CheckBox
        Evaluate:
            - Evaluate that items that are unchecked are not present anymore on channel selection
        """

        with self.subTest("DisableBySpace"):
            self.context_menu(action_text="Hide disabled items")
            # Select one channel
            self.mouseClick_WidgetItem(self.plot_channel_a)
            # Event
            QtTest.QTest.keyClick(self.plot.channel_selection, QtCore.Qt.Key.Key_Space)
            # Evaluate
            self.assertTrue(self.plot_channel_a.isHidden())
            self.context_menu(action_text="Hide disabled items")
            self.assertFalse(self.plot_channel_a.isHidden())

        with self.subTest("DisableByClick"):
            self.context_menu(action_text="Hide disabled items")
            pos = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()
            # Magic Number to detect center of checkbox
            pos = QtCore.QPoint(28, pos.y())
            # Event
            QtTest.QTest.mouseClick(
                self.plot.channel_selection.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                pos,
            )
            # Evaluate
            self.assertTrue(self.plot_channel_b.isHidden())
            self.context_menu(action_text="Hide disabled items")
            self.assertFalse(self.plot_channel_b.isHidden())

    def test_Menu_ShowHide_Action_HideMissingItems(self):
        """
        Test Scope:
            - Ensure that missing item is hidden from channel selection.
        Events:
            - Disable 1 channel by key Space
            - Disable 1 channel by mouseClick on item CheckBox
        Evaluate:
            - Evaluate that missing items are not present anymore on channel selection
        """
        dspf_filepath = pathlib.Path(self.resource, "missingItems.dspf")
        self.load_display_file(display_file=dspf_filepath)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        plot_channel = self.find_channel(channel_tree=self.plot.channel_selection, channel_name="1stMissingItem")
        self.processEvents()

        self.assertFalse(plot_channel.isHidden())

        # Events
        self.context_menu(action_text="Hide missing items")

        # Evaluate
        self.assertTrue(plot_channel.isHidden())
        self.processEvents(timeout=0.01)

    def test_Menu_ShowHide_Action_ShowMissingItems(self):
        """
        Test Scope:
            - Ensure that missing item is visible from channel selection.
        Events:
            - Open Context Menu
            - Select action: Hide missing items
            - Disable 1 channel by key Space
            - Disable 1 channel by mouseClick on item CheckBox
        Evaluate:
            - Evaluate that missing items are present anymore on channel selection
        """
        dspf_filepath = pathlib.Path(self.resource, "missingItems.dspf")
        self.load_display_file(display_file=dspf_filepath)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        self.context_menu(action_text="Hide missing items")

        plot_channel = self.find_channel(channel_tree=self.plot.channel_selection, channel_name="1stMissingItem")
        self.processEvents()

        self.assertTrue(plot_channel.isHidden())

        # Events
        self.context_menu(action_text="Hide missing items")

        # Evaluate
        self.assertFalse(plot_channel.isHidden())
        self.processEvents(timeout=0.01)

    def test_Menu_ShowHide_Action_FilterOnlyComputedChannels(self):
        """
        Test Scope:
            - Ensure that all channels are hidden from channel selection except VirtualChannels.
        Events:
            - Open Context Menu
            - Select Filter Only Computed Channels
        Evaluate:
            - Evaluate that channels items are not present anymore on channel selection except VirtualChannels.
        """
        dspf_filepath = pathlib.Path(self.resource, "missingItems.dspf")
        self.load_display_file(display_file=dspf_filepath)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        iterator = QtWidgets.QTreeWidgetItemIterator(self.plot.channel_selection)
        while iterator.value():
            item = iterator.value()
            if item:
                self.assertFalse(item.isHidden())
            iterator += 1

        # Events
        self.context_menu(action_text="Filter only computed channels")

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.plot.channel_selection)
        while iterator.value():
            item = iterator.value()
            if item and item.text(0) == "FirstVirtualChannel":
                self.assertFalse(item.isHidden())
            else:
                self.assertTrue(item.isHidden())
            iterator += 1

    def test_Menu_ShowHide_Action_UnfilterComputedChannels(self):
        """
        Test Scope:
            - Ensure that all channels are hidden from channel selection except VirtualChannels.
        Events:
            - Open Context Menu
            - Select Filter Only Computed Channels
        Evaluate:
            - Evaluate that channels items are not present anymore on channel selection except VirtualChannels.
        """
        dspf_filepath = pathlib.Path(self.resource, "missingItems.dspf")
        self.load_display_file(display_file=dspf_filepath)
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Events
        self.context_menu(action_text="Filter only computed channels")
        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.plot.channel_selection)
        while iterator.value():
            item = iterator.value()
            if item and item.text(0) == "FirstVirtualChannel":
                self.assertFalse(item.isHidden())
            else:
                self.assertTrue(item.isHidden())
            iterator += 1

        # Events
        self.context_menu(action_text="Filter only computed channels")

        # Evaluate
        iterator = QtWidgets.QTreeWidgetItemIterator(self.plot.channel_selection)
        while iterator.value():
            item = iterator.value()
            if item:
                self.assertFalse(item.isHidden())
            iterator += 1

    def test_Action_EditYAxisScaling(self):
        """
        Test Scope:
            - Ensure that action forwards the request to plot.
        Event:
            - Select channel
            - Open Context Menu
            - Select 'Edit Y axis scaling'
        Evaluate:
            - Evaluate that event is forwarded to plot
        """
        # Setup
        position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
        with mock.patch.object(self.plot, "keyPressEvent") as mo_keyPressEvent:
            self.context_menu(action_text="Edit Y axis scaling", position=position)
            mo_keyPressEvent.assert_called()
            event = mo_keyPressEvent.call_args.args[0]
            self.assertEqual(QtCore.QEvent.Type.KeyPress, event.type())
            self.assertEqual(QtCore.Qt.Key.Key_G, event.key())
            self.assertEqual(QtCore.Qt.KeyboardModifier.ControlModifier, event.modifiers())

    def test_Action_AddToCommonYAxis(self):
        """
        Test Scope:
            - Ensure that action will mark as checked checkbox for Y Axis.
        Event:
            - Select one channel
            - Open Context Menu
            - Select 'Add to common Y axis'
            - Select two channels
            - Open Context Menu
            - Select 'Add to common Y axis'
        Evaluate:
            - Evaluate that checkbox on column COMMON_AXIS is checked.
        """
        with self.subTest("1Channel"):
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_a.checkState(self.Column.COMMON_AXIS))

            position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
            self.context_menu(action_text="Add to common Y axis", position=position)

            self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_a.checkState(self.Column.COMMON_AXIS))

            self.context_menu(action_text="Add to common Y axis", position=position)

            self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_a.checkState(self.Column.COMMON_AXIS))

        with self.subTest("2Channels"):
            self.plot_channel_b.setSelected(True)
            self.plot_channel_c.setSelected(True)
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_b.checkState(self.Column.COMMON_AXIS))
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_c.checkState(self.Column.COMMON_AXIS))

            position = self.plot.channel_selection.visualItemRect(self.plot_channel_c).center()
            self.context_menu(action_text="Add to common Y axis", position=position)

            self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_b.checkState(self.Column.COMMON_AXIS))
            self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_c.checkState(self.Column.COMMON_AXIS))

            self.context_menu(action_text="Add to common Y axis", position=position)

            self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_b.checkState(self.Column.COMMON_AXIS))
            self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_c.checkState(self.Column.COMMON_AXIS))

    def test_Action_RemoveFromCommonYAxis(self):
        """
        Test Scope:
            - Ensure that action will mark as unchecked checkbox for Y Axis.
        Event:
            - Select one channel
            - Open Context Menu
            - Select 'Remove from common Y axis'
            - Select two channels
            - Open Context Menu
            - Select 'Remove from common Y axis'
        Evaluate:
            - Evaluate that checkbox on column COMMON_AXIS is unchecked.
        """
        with self.subTest("1Channel"):
            # Setup
            position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()

            # Pre-evaluation
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_a.checkState(self.Column.COMMON_AXIS))

            # Event
            self.context_menu(action_text="Remove from common Y axis", position=position)
            # Evaluate
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_a.checkState(self.Column.COMMON_AXIS))

            # Event
            self.context_menu(action_text="Add to common Y axis", position=position)
            self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_a.checkState(self.Column.COMMON_AXIS))
            self.context_menu(action_text="Remove from common Y axis", position=position)
            # Evaluate
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_a.checkState(self.Column.COMMON_AXIS))

            # Event
            self.context_menu(action_text="Remove from common Y axis", position=position)
            # Evaluate
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_a.checkState(self.Column.COMMON_AXIS))

        with self.subTest("2Channels"):
            # Setup
            self.plot_channel_b.setSelected(True)
            self.plot_channel_c.setSelected(True)
            position_c = self.plot.channel_selection.visualItemRect(self.plot_channel_c).center()

            # Pre-evaluation
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_b.checkState(self.Column.COMMON_AXIS))
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_c.checkState(self.Column.COMMON_AXIS))

            # Event

            self.context_menu(action_text="Remove from common Y axis", position=position_c)
            # Evaluate
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_b.checkState(self.Column.COMMON_AXIS))
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_c.checkState(self.Column.COMMON_AXIS))

            # Event
            self.context_menu(action_text="Add to common Y axis", position=position_c)
            self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_b.checkState(self.Column.COMMON_AXIS))
            self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_c.checkState(self.Column.COMMON_AXIS))
            self.context_menu(action_text="Remove from common Y axis", position=position)
            # Evaluate
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_b.checkState(self.Column.COMMON_AXIS))
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_c.checkState(self.Column.COMMON_AXIS))

            # Event
            self.context_menu(action_text="Remove from common Y axis", position=position)
            # Evaluate
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_b.checkState(self.Column.COMMON_AXIS))
            self.assertNotEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_c.checkState(self.Column.COMMON_AXIS))

    def test_Action_SetColor(self):
        """
        Test Scope:
            - Ensure that channel color is changed.
        Events:
            - Open Context Menu
            - Select 'Set color [C]'
            - Select 1 Channel
            - Open Context Menu
            - Select 'Set color [C]'
            - Select 2 Channels
            - Open Context Menu
            - Select 'Set color'
        Evaluate:
            - Evaluate that color dialog is not open if channel is not selected.
            - Evaluate that channel color is changed.
        """
        action_text = "Set color"

        # Event
        with self.subTest("NoChannelSelected"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                self.context_menu(action_text=action_text)
                mo_getColor.assert_not_called()

        with self.subTest("1ChannelSelected"):
            position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                # Setup
                previous_color = self.plot_channel_a.color.name()
                color = QtGui.QColor("red")
                color_name = color.name()
                mo_getColor.return_value = color
                # Event
                self.context_menu(action_text=action_text, position=position)
                # Evaluate
                current_color = self.plot_channel_a.color.name()
                mo_getColor.assert_called()
                self.assertNotEqual(previous_color, current_color)
                self.assertEqual(color_name, current_color)

        with self.subTest("2ChannelsSelected"):
            self.mouseClick_WidgetItem(self.plot_channel_b)
            self.plot_channel_b.setSelected(True)
            self.plot_channel_c.setSelected(True)
            position = self.plot.channel_selection.visualItemRect(self.plot_channel_c).center()
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                # Setup
                previous_b_color = self.plot_channel_b.color.name()
                previous_c_color = self.plot_channel_c.color.name()
                color = QtGui.QColor("blue")
                color_name = color.name()
                mo_getColor.return_value = color
                # Event
                self.context_menu(action_text=action_text, position=position)
                # Evaluate
                current_b_color = self.plot_channel_b.color.name()
                current_c_color = self.plot_channel_c.color.name()
                mo_getColor.assert_called()
                self.assertNotEqual(previous_b_color, current_b_color)
                self.assertNotEqual(previous_c_color, current_c_color)
                self.assertEqual(color_name, current_b_color)
                self.assertEqual(color_name, current_c_color)

    def test_Action_SetRandomColor(self):
        """
        Test Scope:
            - Ensure that channel color is changed.
        Events:
            - Open Context Menu
            - Select 'Set random color'
            - Select 1 Channel
            - Open Context Menu
            - Select 'Set random color'
            - Select 2 Channels
            - Open Context Menu
            - Select 'Set random color'
        Evaluate:
            - Evaluate that color dialog is not open if channel is not selected.
            - Evaluate that channel color is changed.
        """
        action_text = "Set random color"

        # Event
        with self.subTest("NoChannelSelected"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QColorDialog.getColor") as mo_getColor:
                self.context_menu(action_text=action_text)
                mo_getColor.assert_not_called()

        with self.subTest("1ChannelSelected"):
            position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
            # Setup
            previous_color = self.plot_channel_a.color.name()
            # Event
            self.context_menu(action_text=action_text, position=position)
            # Evaluate
            current_color = self.plot_channel_a.color.name()
            self.assertNotEqual(previous_color, current_color)

        with self.subTest("2ChannelsSelected"):
            self.mouseClick_WidgetItem(self.plot_channel_b)
            self.plot_channel_b.setSelected(True)
            self.plot_channel_c.setSelected(True)
            position = self.plot.channel_selection.visualItemRect(self.plot_channel_c).center()
            # Setup
            previous_b_color = self.plot_channel_b.color.name()
            previous_c_color = self.plot_channel_c.color.name()
            # Event
            self.context_menu(action_text=action_text, position=position)
            # Evaluate
            current_b_color = self.plot_channel_b.color.name()
            current_c_color = self.plot_channel_c.color.name()
            self.assertNotEqual(previous_b_color, current_b_color)
            self.assertNotEqual(previous_c_color, current_c_color)
            self.assertNotEqual(current_b_color, current_c_color)

    # @unittest.skipIf(sys.platform == "win32", "times out on Windows")
    def test_Action_CopyDisplayProperties_Group(self):
        """
        Test Scope:
            - Ensure that first channel display properties are copied in clipboard if item selected is group item.
        Events:
            - Insert Empty Group
            - Select Group
            - Trigger 'Copy display properties' action
            - Add items to group
            - Open Context Menu
            - Trigger 'Copy display properties' action
            - Remove channel from group
            - Open Context Menu
            - Trigger 'Copy display properties' action
        Evaluate:
            - Evaluate that channel display properties are stored in clipboard in json format.
        """
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "A", True
            self.context_menu(action_text="Add channel group")
            mo_getText.return_value = "B", True
            self.context_menu(action_text="Add channel group")
            mo_getText.return_value = "C", True
            self.context_menu(action_text="Add channel group")

        group_channel_a = self.plot.channel_selection.findItems("A", QtCore.Qt.MatchFlags())[0]
        group_channel_b = self.plot.channel_selection.findItems("B", QtCore.Qt.MatchFlags())[0]
        group_channel_c = self.plot.channel_selection.findItems("C", QtCore.Qt.MatchFlags())[0]

        QtWidgets.QApplication.instance().clipboard().clear()
        with self.subTest("EmptyGroup_0"):
            position = self.plot.channel_selection.visualItemRect(group_channel_a).center()
            self.context_menu(action_text="Copy display properties", position=position)
            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            try:
                content = json.loads(clipboard)
            except JSONDecodeError:
                self.fail("Clipboard Content cannot be decoded as JSON content.")
            else:
                self.assertIsInstance(content, dict)
                self.assertTrue(content["type"] == "group")

        with self.subTest("PopulatedGroup"):
            # Add Channels to Group
            self.move_item_inside_channels_tree_widget(src=self.plot_channel_a, dst=group_channel_a)

            position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
            self.context_menu(action_text="Copy display properties", position=position)

            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            try:
                content = json.loads(clipboard)
            except JSONDecodeError:
                self.fail("Clipboard Content cannot be decoded as JSON content.")
            else:
                self.assertIsInstance(content, dict)
                self.assertTrue(content["type"] == "channel")

        with self.subTest("EmptyGroup_1"):
            group_channel_a.removeChild(self.plot_channel_a)
            position = self.plot.channel_selection.visualItemRect(group_channel_a).center()
            self.context_menu(action_text="Copy display properties", position=position)
            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            try:
                content = json.loads(clipboard)
            except JSONDecodeError:
                self.fail("Clipboard Content cannot be decoded as JSON content.")
            else:
                self.assertIsInstance(content, dict)
                self.assertTrue(content["type"] == "group")

        with self.subTest("EmptyGroup_2"):
            # Add Channels to Group
            self.move_item_inside_channels_tree_widget(src=group_channel_c, dst=group_channel_b)
            group_channel_a.setExpanded(True)
            group_channel_b.setExpanded(True)
            group_channel_c.setExpanded(True)
            self.move_item_inside_channels_tree_widget(src=group_channel_b, dst=group_channel_a)
            group_channel_a.setExpanded(True)
            group_channel_b.setExpanded(True)
            group_channel_c.setExpanded(True)
            position = self.plot.channel_selection.visualItemRect(group_channel_a).center()
            self.context_menu(action_text="Copy display properties", position=position)
            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            try:
                content = json.loads(clipboard)
            except JSONDecodeError:
                self.fail("Clipboard Content cannot be decoded as JSON content.")
            else:
                self.assertIsInstance(content, dict)
                self.assertTrue(content["type"] == "group")

    def test_Action_PasteDisplayProperties_Group(self):
        """
        Test Scope:
            - Ensure that channel display properties can be pasted over different channel.
        Events:
            - Select one channel
            - Open Context Menu
            - Trigger 'Copy display properties' action
            - Select a group channel
            - Open Context Menu
            - Trigger 'Paste display properties' action
            - Select group channel
            - Open Context Menu
            - Trigger 'Copy display properties' action
            - Select a channel
            - Open Context Menu
            - Trigger 'Paste display properties' action
        Evaluate:
            - Evaluate that display properties are transferred from one channel to another
        """
        action_copy = "Copy display properties"
        action_paste = "Paste display properties"

        # Insert Group
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "A", True
            self.context_menu(action_text="Add channel group")
            mo_getText.return_value = "B", True
            self.context_menu(action_text="Add channel group")
            mo_getText.return_value = "C", True
            self.context_menu(action_text="Add channel group")

        # Add Channels to Group
        group_channel_a = self.plot.channel_selection.findItems("A", QtCore.Qt.MatchFlags())[0]
        group_channel_a.setExpanded(True)
        group_channel_c = self.plot.channel_selection.findItems("C", QtCore.Qt.MatchFlags())[0]
        group_channel_c.setExpanded(True)
        self.move_item_inside_channels_tree_widget(src=group_channel_c, dst=group_channel_a)
        group_channel_a.setExpanded(True)
        group_channel_c.setExpanded(True)
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_b, dst=group_channel_c)

        # Add Channels to Group
        group_channel_b = self.plot.channel_selection.findItems("B", QtCore.Qt.MatchFlags())[0]
        group_channel_b.setExpanded(True)
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_d, dst=group_channel_b)

        with self.subTest("FromChannel_ToGroup"):
            position_src = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
            self.context_menu(action_text=action_copy, position=position_src)

            channel_a_properties = QtWidgets.QApplication.instance().clipboard().text()
            channel_a_properties = json.loads(channel_a_properties)

            # Paste
            position_dst = self.plot.channel_selection.visualItemRect(group_channel_a).center()
            self.context_menu(action_text=action_paste, position=position_dst)

            # Copy
            position_src = self.plot.channel_selection.visualItemRect(group_channel_a).center()
            self.context_menu(action_text=action_copy, position=position_src)

            group_channel_properties = QtWidgets.QApplication.instance().clipboard().text()
            group_channel_properties = json.loads(group_channel_properties)

            # Evaluate
            self.assertEqual(channel_a_properties["ranges"], group_channel_properties["ranges"])

        with self.subTest("FromGroup_ToChannel"):
            position_src = self.plot.channel_selection.visualItemRect(group_channel_a).center()
            self.context_menu(action_text=action_copy, position=position_src)

            # Paste
            position_dst = self.plot.channel_selection.visualItemRect(self.plot_channel_c).center()
            self.context_menu(action_text=action_paste, position=position_dst)

            # Copy
            position_src = self.plot.channel_selection.visualItemRect(self.plot_channel_c).center()
            self.context_menu(action_text=action_copy, position=position_src)

            channel_c_properties = QtWidgets.QApplication.instance().clipboard().text()
            channel_c_properties = json.loads(channel_c_properties)

            # Evaluate
            self.assertEqual(channel_c_properties["ranges"], group_channel_properties["ranges"])

        with self.subTest("FromGroup_ToGroup"):
            self.move_item_inside_channels_tree_widget(src=self.plot_channel_a, dst=group_channel_a)

            position_src = self.plot.channel_selection.visualItemRect(group_channel_b).center()
            self.context_menu(action_text=action_copy, position=position_src)
            group_channel_b_properties = QtWidgets.QApplication.instance().clipboard().text()
            group_channel_b_properties = json.loads(group_channel_b_properties)

            # Paste
            position_dst = self.plot.channel_selection.visualItemRect(group_channel_a).center()
            self.context_menu(action_text=action_paste, position=position_dst)

            # Copy
            position_src = self.plot.channel_selection.visualItemRect(group_channel_a).center()
            self.context_menu(action_text=action_copy, position=position_src)

            group_channel_a_properties = QtWidgets.QApplication.instance().clipboard().text()
            group_channel_a_properties = json.loads(group_channel_a_properties)

            # Evaluate
            self.assertEqual(group_channel_a_properties, group_channel_b_properties)

    # @unittest.skipIf(sys.platform == "win32", "times out on Windows")
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
            self.context_menu(action_text="Add channel group")

        # Add Channels to Group
        group_channel = self.plot.channel_selection.findItems("FirstGroup", QtCore.Qt.MatchFlags())[0]
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_a, dst=group_channel)

        position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
        self.context_menu(action_text="Copy channel structure", position=position)

        clipboard = QtWidgets.QApplication.instance().clipboard().text()
        try:
            content = json.loads(clipboard)
        except JSONDecodeError:
            self.fail("Clipboard Content cannot be decoded as JSON content.")
        else:
            self.assertIsInstance(content, list)
            for channel_properties in content:
                self.assertIsInstance(channel_properties, dict)

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
        action_copy = "Copy channel structure"
        action_paste = "Paste channel structure"

        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "FirstGroup", True
            self.context_menu(action_text="Add channel group")

        # Add Channels to Group
        group_channel = self.plot.channel_selection.findItems("FirstGroup", QtCore.Qt.MatchFlags())[0]
        group_channel.setExpanded(True)
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_c, dst=group_channel)
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_d, dst=group_channel)

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

    def test_Menu_EnableDisable_Action_DisableSelected(self):
        """
        Test Scope:
            - Ensure that channel selected is disabled.
        Events:
            - Disable all
            - Open Context Menu
            - Select action "Disable selected" from sub-menu "Enable/disable"
            - Select two channel items
            - Open Context Menu
            - Select action "Disable selected" from sub-menu "Enable/disable"
            - Select channel group
            - Open Context Menu
            - Select action "Disable selected" from sub-menu "Enable/disable"
        Expected:
            - Evaluate that selected items were disabled.
        """
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "A", True
            self.context_menu(action_text="Add channel group")

        # Add Channels to Group
        group_channel = self.plot.channel_selection.findItems("A", QtCore.Qt.MatchFlags())[0]
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_c, dst=group_channel)

        self.context_menu(action_text="Disable selected")

        # Ensure that all items are disabled
        count = self.plot.channel_selection.topLevelItemCount()
        for i in range(count):
            item = self.plot.channel_selection.topLevelItem(i)
            if item.type() != item.Info:
                self.assertEqual(QtCore.Qt.CheckState.Checked, item.checkState(self.Column.NAME))

        # Select two channels
        self.plot_channel_a.setSelected(True)
        self.plot_channel_b.setSelected(True)
        position_src = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()
        self.context_menu(action_text="Disable selected", position=position_src)

        self.assertEqual(QtCore.Qt.CheckState.Unchecked, self.plot_channel_a.checkState(self.Column.NAME))
        self.assertEqual(QtCore.Qt.CheckState.Unchecked, self.plot_channel_b.checkState(self.Column.NAME))

        # Select group
        position_src = self.plot.channel_selection.visualItemRect(group_channel).center()
        self.context_menu(action_text="Disable selected", position=position_src)

        self.assertEqual(QtCore.Qt.CheckState.Unchecked, group_channel.checkState(self.Column.NAME))

    def test_Menu_EnableDisable_Action_DeactivateGroups_Group(self):
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
            self.context_menu(action_text="Add channel group")
            mo_getText.return_value = "Group_B", True
            self.context_menu(action_text="Add channel group")

        # Add Channels to Group
        group_a_channel = self.plot.channel_selection.findItems("Group_A", QtCore.Qt.MatchFlags())[0]
        group_a_channel.setExpanded(True)
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_a, dst=group_a_channel)
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_b, dst=group_a_channel)

        # Add Channels to Group
        group_b_channel = self.plot.channel_selection.findItems("Group_B", QtCore.Qt.MatchFlags())[0]
        group_b_channel.setExpanded(True)
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_c, dst=group_b_channel)
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_d, dst=group_b_channel)

        self.move_item_inside_channels_tree_widget(src=group_b_channel, dst=group_a_channel)
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

    def test_Menu_EnableDisable_Action_DisableAll(self):
        """
        Test Scope:
            - Ensure that all channels are disabled.
        Events:
            - Open Context Menu
            - Select action "Disable All" from sub-menu "Enable/disable"
            - Disable elements
            - Open Context Menu
            - Select action "Disable All" from sub-menu "Enable/disable"
        Expected:
            - Evaluate that all items are disabled.
        """
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "A", True
            self.context_menu(action_text="Add channel group")

        # Add Channels to Group
        group_channel = self.plot.channel_selection.findItems("A", QtCore.Qt.MatchFlags())[0]
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_a, dst=group_channel)
        self.processEvents(1)

        # Ensure that all items are enabled
        count = self.plot.channel_selection.topLevelItemCount()
        for i in range(count):
            item = self.plot.channel_selection.topLevelItem(i)
            if item.type() != item.Info:
                self.assertEqual(QtCore.Qt.CheckState.Checked, item.checkState(self.Column.NAME))

        self.context_menu(action_text="Disable all")

        # Ensure that all items are disabled
        count = self.plot.channel_selection.topLevelItemCount()
        for i in range(count):
            item = self.plot.channel_selection.topLevelItem(i)
            if item.type() != item.Info:
                self.assertEqual(QtCore.Qt.CheckState.Unchecked, item.checkState(self.Column.NAME))
                # Disable items
                item.setCheckState(self.Column.NAME, QtCore.Qt.CheckState.Checked)

        self.context_menu(action_text="Disable all")

        for i in range(count):
            item = self.plot.channel_selection.topLevelItem(i)
            if item.type() != item.Info:
                self.assertEqual(QtCore.Qt.CheckState.Unchecked, item.checkState(self.Column.NAME))

    def test_Menu_EnableDisable_Action_EnableSelected(self):
        """
        Test Scope:
            - Ensure that channel selected is enabled.
        Events:
            - Disable all
            - Open Context Menu
            - Select action "Enable selected" from sub-menu "Enable/disable"
            - Select two channel items
            - Open Context Menu
            - Select action "Enable selected" from sub-menu "Enable/disable"
            - Select channel group
            - Open Context Menu
            - Select action "Enable selected" from sub-menu "Enable/disable"
        Expected:
            - Evaluate that selected items were enabled.
        """
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "A", True
            self.context_menu(action_text="Add channel group")

        # Add Channels to Group
        group_channel = self.plot.channel_selection.findItems("A", QtCore.Qt.MatchFlags())[0]
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_c, dst=group_channel)

        # Disable all items
        count = self.plot.channel_selection.topLevelItemCount()
        for i in range(count):
            item = self.plot.channel_selection.topLevelItem(i)
            if item.type() != item.Info:
                item.setCheckState(self.Column.NAME, QtCore.Qt.CheckState.Unchecked)

        self.context_menu(action_text="Enable selected")

        # Ensure that all items are disabled
        count = self.plot.channel_selection.topLevelItemCount()
        for i in range(count):
            item = self.plot.channel_selection.topLevelItem(i)
            if item.type() != item.Info:
                self.assertEqual(QtCore.Qt.CheckState.Unchecked, item.checkState(self.Column.NAME))

        # Select two channels
        self.plot_channel_a.setSelected(True)
        self.plot_channel_b.setSelected(True)
        position_src = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()
        self.context_menu(action_text="Enable selected", position=position_src)

        self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_a.checkState(self.Column.NAME))
        self.assertEqual(QtCore.Qt.CheckState.Checked, self.plot_channel_b.checkState(self.Column.NAME))

        # Select group
        position_src = self.plot.channel_selection.visualItemRect(group_channel).center()
        self.context_menu(action_text="Enable selected", position=position_src)

        self.assertEqual(QtCore.Qt.CheckState.Checked, group_channel.checkState(self.Column.NAME))

    def test_Menu_EnableDisable_Action_EnableAll(self):
        """
        Test Scope:
            - Ensure that all channels are enabled.
        Events:
            - Open Context Menu
            - Select action "Enable All" from sub-menu "Enable/disable"
            - Disable elements
            - Open Context Menu
            - Select action "Enable All" from sub-menu "Enable/disable"
        Expected:
            - Evaluate that all items are enabled.
        """
        with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText:
            mo_getText.return_value = "A", True
            self.context_menu(action_text="Add channel group")

        # Add Channels to Group
        group_channel = self.plot.channel_selection.findItems("A", QtCore.Qt.MatchFlags())[0]
        self.move_item_inside_channels_tree_widget(src=self.plot_channel_a, dst=group_channel)

        # Ensure that all items are enabled:
        count = self.plot.channel_selection.topLevelItemCount()
        for i in range(count):
            item = self.plot.channel_selection.topLevelItem(i)
            if item.type() != item.Info:
                self.assertEqual(QtCore.Qt.CheckState.Checked, item.checkState(self.Column.NAME))

        self.context_menu(action_text="Enable all")

        # Ensure that all items are enabled:
        count = self.plot.channel_selection.topLevelItemCount()
        for i in range(count):
            item = self.plot.channel_selection.topLevelItem(i)
            if item.type() != item.Info:
                self.assertEqual(QtCore.Qt.CheckState.Checked, item.checkState(self.Column.NAME))
                # Disable items
                item.setCheckState(self.Column.NAME, QtCore.Qt.CheckState.Unchecked)

        self.context_menu(action_text="Enable all")

        for i in range(count):
            item = self.plot.channel_selection.topLevelItem(i)
            if item.type() != item.Info:
                self.assertEqual(QtCore.Qt.CheckState.Checked, item.checkState(self.Column.NAME))

#!/usr/bin/env python
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets

from test.asammdf.gui.widgets.test_BasePlotWidget import TestPlotWidget


class QMenuWrap(QtWidgets.QMenu):
    return_action = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def exec(self, *args, **kwargs):
        if not self.return_action:
            return super().exec_(*args, **kwargs)
        return self.return_action


class TestContextMenu(TestPlotWidget):
    # Note: Test Plot Widget through FileWidget.

    def setUp(self):
        super().setUp()

        self.setUpFileWidget(measurement_file=self.measurement_file, default=True)
        # Switch ComboBox to "Natural sort"
        self.widget.channel_view.setCurrentText("Natural sort")

        self.create_window(window_type="Plot")
        self.assertEqual(len(self.widget.mdi_area.subWindowList()), 1)

        # Drag and Drop channel from FileWidget.channel_tree to Plot
        self.plot = self.widget.mdi_area.subWindowList()[0].widget()

        # Add channels
        self.plot_channel_a = self.add_channel_to_plot(channel_index=10)
        self.plot_channel_b = self.add_channel_to_plot(channel_index=11)
        self.plot_channel_c = self.add_channel_to_plot(channel_index=12)
        self.plot_channel_d = self.add_channel_to_plot(channel_index=13)
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

        with (
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QMenu", wraps=QMenuWrap),
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText
        ):
            mo_getText.return_value = None, None

            mo_action = mock.MagicMock()
            mo_action.text.return_value = "Search item"
            QMenuWrap.return_action = mo_action

            QtTest.QTest.mouseClick(self.plot.channel_selection.viewport(), QtCore.Qt.MouseButton.RightButton)
            self.processEvents(0.01)

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

        with (
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QMenu", wraps=QMenuWrap),
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText,
            mock.patch("asammdf.gui.widgets.tree.MessageBox.warning") as mo_warning,
        ):
            mo_action = mock.MagicMock()
            mo_action.text.return_value = "Search item"
            QMenuWrap.return_action = mo_action

            mo_getText.return_value = self.id(), True

            QtTest.QTest.mouseClick(self.plot.channel_selection.viewport(), QtCore.Qt.MouseButton.RightButton)
            while not mo_action.text.called:
                self.processEvents(0.02)

        self.assertEqual(0, len(self.plot.channel_selection.selectedItems()))
        mo_warning.assert_called_with(
            self.plot.channel_selection,
            "No matches found",
            "No item matches the pattern\n"
            "test_PlotWidget_ContextMenu.TestContextMenu.test_Action_SearchItem_NonexistentChannel"
        )

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

        with (
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QMenu", wraps=QMenuWrap),
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText,
            mock.patch("asammdf.gui.widgets.tree.MessageBox.warning") as mo_warning,
        ):
            mo_action = mock.MagicMock()
            mo_action.text.return_value = "Search item"
            QMenuWrap.return_action = mo_action

            mo_getText.return_value = self.plot_channel_b.text(self.Column.NAME), True

            QtTest.QTest.mouseClick(self.plot.channel_selection.viewport(), QtCore.Qt.MouseButton.RightButton)
            while not mo_action.text.called:
                self.processEvents(0.02)

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

        with (
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QMenu", wraps=QMenuWrap),
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText,
        ):
            mo_action = mock.MagicMock()
            mo_action.text.return_value = "Add channel group [Shift+Insert]"
            QMenuWrap.return_action = mo_action

            mo_getText.return_value = "FirstGroup", None

            QtTest.QTest.mouseClick(self.plot.channel_selection.viewport(), QtCore.Qt.MouseButton.RightButton)
            while not mo_action.text.called:
                self.processEvents(0.02)

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

        with (
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QMenu", wraps=QMenuWrap),
            mock.patch("asammdf.gui.widgets.tree.QtWidgets.QInputDialog.getText") as mo_getText,
        ):
            mo_action = mock.MagicMock()
            mo_action.text.return_value = "Add channel group [Shift+Insert]"
            QMenuWrap.return_action = mo_action

            mo_getText.return_value = "FirstGroup", True

            QtTest.QTest.mouseClick(self.plot.channel_selection.viewport(), QtCore.Qt.MouseButton.RightButton)
            while not mo_action.text.called:
                self.processEvents(0.02)

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
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QMenu", wraps=QMenuWrap):
                mo_action = mock.MagicMock()
                mo_action.text.return_value = "Copy names [Ctrl+N]"
                QMenuWrap.return_action = mo_action

                position = self.plot.channel_selection.visualItemRect(self.plot_channel_a).center()
                QtTest.QTest.mouseClick(
                    self.plot.channel_selection.viewport(),
                    QtCore.Qt.MouseButton.RightButton,
                    QtCore.Qt.KeyboardModifiers(),
                    position
                )
                while not mo_action.text.called:
                    self.processEvents(0.02)

            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            self.assertEqual(self.plot_channel_a.text(self.Column.NAME), clipboard)

        with self.subTest("2Channels"):
            with mock.patch("asammdf.gui.widgets.tree.QtWidgets.QMenu", wraps=QMenuWrap):
                mo_action = mock.MagicMock()
                mo_action.text.return_value = "Copy names [Ctrl+N]"
                QMenuWrap.return_action = mo_action

                self.plot_channel_a.setSelected(True)
                self.plot_channel_b.setSelected(True)
                position_1 = self.plot.channel_selection.visualItemRect(self.plot_channel_b).center()

                QtTest.QTest.mouseClick(
                    self.plot.channel_selection.viewport(),
                    QtCore.Qt.MouseButton.RightButton,
                    QtCore.Qt.KeyboardModifiers(),
                    position_1
                )
                while not mo_action.text.called:
                    self.processEvents(0.02)

            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            channels = (
                self.plot_channel_a.text(self.Column.NAME),
                self.plot_channel_b.text(self.Column.NAME)
            )
            self.assertEqual("\n".join(channels), clipboard)
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets

from asammdf.gui.widgets.file import FileWidget
from test.asammdf.gui.test_base import TestBase


class TestFileWidget(TestBase):
    def setUp(self):
        super().setUp()
        self.widget = None
        self.plot = None

        patcher = mock.patch("asammdf.gui.widgets.file.ErrorDialog")
        self.mc_widget_ed = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        if self.widget:
            self.widget.close()
            self.widget.destroy()
            self.widget.deleteLater()
        self.mc_ErrorDialog.reset_mock()
        super().tearDown()

    def setUpFileWidget(self, *args, measurement_file, default):
        """
        Created because for a lot of testcases,
        we do not need other parameters for FileWidget initialization.
        """
        if default:
            self.widget = FileWidget(
                measurement_file,
                True,  # with_dots
                True,  # subplots
                True,  # subplots_link
                False,  # ignore_value2text_conversions
                False,  # display_cg_name
                "line",  # line_interconnect
                1,  # password
                None,  # hide_missing_channels
                None,  # hide_disabled_channels
            )
        else:
            self.widget = FileWidget(measurement_file, *args)
        self.widget.showMaximized()
        self.widget.activateWindow()
        self.processEvents()

    def create_window(self, window_type, channels_names=(), channels_indexes=()):
        channel_tree = self.widget.channels_tree
        channel_tree.clearSelection()
        for channel in channels_names:
            channel = self.find_channel(channel_tree, channel_name=channel)
            channel.setCheckState(0, QtCore.Qt.CheckState.Checked)
        for channel in channels_indexes:
            channel = self.find_channel(channel_tree, channel_index=channel)
            channel.setCheckState(0, QtCore.Qt.CheckState.Checked)

        with mock.patch("asammdf.gui.widgets.file.WindowSelectionDialog") as mc_WindowSelectionDialog:
            mc_WindowSelectionDialog.return_value.result.return_value = True
            mc_WindowSelectionDialog.return_value.selected_type.return_value = window_type
            # - Press PushButton "Create Window"
            QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.MouseButton.LeftButton)
            widget_types = self.get_subwindows()
            self.assertIn(window_type, widget_types)

    @staticmethod
    def find_channel(channel_tree, channel_name=None, channel_index=None):
        selected_channel = None
        if not channel_name and not channel_index:
            selected_channel = channel_tree.topLevelItem(0)
        elif channel_index:
            selected_channel = channel_tree.topLevelItem(channel_index)
        elif channel_name:
            iterator = QtWidgets.QTreeWidgetItemIterator(channel_tree)
            while iterator.value():
                item = iterator.value()
                if item and item.text(0) == channel_name:
                    selected_channel = item
                    break
                iterator += 1
        return selected_channel

    def get_subwindows(self):
        widget_types = sorted(w.widget().__class__.__name__ for w in self.widget.mdi_area.subWindowList())
        return widget_types

    def load_display_file(self, display_file):
        with (
            mock.patch.object(self.widget, "load_window", wraps=self.widget.load_window) as mo_load_window,
            mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileName") as mo_getOpenFileName,
        ):
            mo_getOpenFileName.return_value = display_file, None
            QtTest.QTest.mouseClick(
                self.widget.load_channel_list_btn,
                QtCore.Qt.MouseButton.LeftButton,
            )
            # Pre-Evaluate
            mo_load_window.assert_called()

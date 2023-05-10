import pathlib
from test.asammdf.gui.test_base import DragAndDrop
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets


class TestPlotWidget(TestFileWidget):
    measurement_file = str(pathlib.Path(TestFileWidget.resource, "ASAP2_Demo_V171.mf4"))

    def create_window(self, window_type):
        with mock.patch(
            "asammdf.gui.widgets.file.WindowSelectionDialog"
        ) as mc_WindowSelectionDialog:
            mc_WindowSelectionDialog.return_value.result.return_value = True
            mc_WindowSelectionDialog.return_value.selected_type.return_value = (
                window_type
            )
            # - Press PushButton "Create Window"
            QtTest.QTest.mouseClick(self.widget.create_window_btn, QtCore.Qt.LeftButton)
            widget_types = self.get_subwindows()
            self.assertIn(window_type, widget_types)

    def add_channel_to_plot(self, plot, channel_name=None, channel_index=None):
        # Select channel
        selected_channel = None
        channel_tree = self.widget.channels_tree
        channel_tree.clearSelection()

        if not channel_name and not channel_index:
            selected_channel = channel_tree.topLevelItem(0)
            channel_name = selected_channel.text(0)
        elif channel_index:
            selected_channel = channel_tree.topLevelItem(channel_index)
            channel_name = selected_channel.text(0)
        elif channel_name:
            iterator = QtWidgets.QTreeWidgetItemIterator(channel_tree)
            while iterator.value():
                item = iterator.value()
                if item and item.text(0) == channel_name:
                    item.setSelected(True)
                    selected_channel = item
                iterator += 1

        drag_position = channel_tree.visualItemRect(selected_channel).center()
        drop_position = plot.channel_selection.viewport().rect().center()

        # PreEvaluation
        DragAndDrop(
            source_widget=channel_tree,
            destination_widget=plot.channel_selection,
            source_pos=drag_position,
            destination_pos=drop_position,
        )
        plot_channel = None
        iterator = QtWidgets.QTreeWidgetItemIterator(plot.channel_selection)
        while iterator.value():
            item = iterator.value()
            if item and item.text(0) == channel_name:
                plot_channel = item
            iterator += 1

        return plot_channel

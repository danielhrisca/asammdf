from collections.abc import Sequence
import os
from pathlib import Path
import shutil
from unittest import mock

from PySide6 import QtCore
from PySide6.QtWidgets import QTreeWidgetItemIterator

from asammdf.gui.widgets.batch import BatchWidget
from test.asammdf.gui.test_base import TestBase


class TestBatchWidget(TestBase):
    testResult = None
    concatenate_aspect = 0
    modify_aspect = 1
    stack_aspect = 3
    bus_aspect = 4
    default_test_file = "ASAP2_Demo_V171.mf4"

    def setUp(self):
        super().setUp()
        self.widget = None
        self.plot = None

        patcher = mock.patch("asammdf.gui.widgets.file.ErrorDialog")
        self.mc_widget_ed = patcher.start()
        self.addCleanup(patcher.stop)

        self.processEvents()

    def setUpBatchWidget(self, *args, measurement_files: Sequence[str] | None):
        """
        Created because a lot of testcases,
        we do not need other parameters for BatchWidget initialization.
        """
        if measurement_files is None:
            self.widget = BatchWidget(*args)
            self.processEvents()
            self.widget.files_list.addItems([str(Path(self.test_workspace, self.default_test_file))])

        else:
            self.widget = BatchWidget(*args)
            self.processEvents()
            for file in measurement_files:
                self.assertTrue(Path(file).exists())
            self.widget.files_list.addItems(measurement_files)

            # Evaluate that all files was opened
            self.assertEqual(self.widget.files_list.count(), len(measurement_files))

        self.widget.showNormal()

    def copy_mdf_files_to_workspace(self):
        # copy mf4 files from resources to test workspace
        for file in os.listdir(self.resource):
            if file.endswith((".mf4", ".mdf")):
                shutil.copyfile(Path(self.resource, file), Path(self.test_workspace, file))

    def select_channels(self, start=0, end=0) -> list:
        """
        Set selected channels from start to end index.

        Parameters
        ----------
        start: first selected channel index
        end: last selected channel index

        Returns
        -------
        channels names list
        """
        channels = []
        iterator = QTreeWidgetItemIterator(self.widget.filter_tree)

        count = 0
        while iterator.value() and not (count == end):
            if count >= start:
                item = iterator.value()
                item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                self.assertTrue(item.checkState(0) == QtCore.Qt.CheckState.Checked)
                channels.append(item.text(0))
            iterator += 1
            count += 1

        # Evaluate that channels were added to "selected_filter_channels"
        for index in range(self.widget.selected_filter_channels.count()):
            item = self.widget.selected_filter_channels.item(index)
            self.assertIn(item.text(), channels)

        return channels

    def get_selected_groups(self, channels: list) -> dict:
        self.widget.filter_view.setCurrentText("Internal file structure")
        self.processEvents(1)

        groups = {}
        iterator = QTreeWidgetItemIterator(self.widget.filter_tree)
        count = 0

        while iterator.value():
            item = iterator.value()
            if item.text(0) in channels and item.text(0) != "time":
                if item.parent().text(0) not in groups.keys():
                    groups[item.parent().text(0)] = [item.text(0)]
                else:
                    groups[item.parent().text(0)].append(item.text(0))

            iterator += 1
            count += 1
        return groups

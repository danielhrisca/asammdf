from collections.abc import Iterable, Sequence
import itertools
import os
from pathlib import Path
import shutil
from unittest import mock

from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton, QTreeWidgetItemIterator, QWidget

from asammdf.gui.widgets.batch import BatchWidget
from test.asammdf.gui.test_base import TestBase


class TestBatchWidget(TestBase):
    testResult = None
    concatenate_aspect = 0
    modify_aspect = 1
    stack_aspect = 2
    bus_aspect = 3
    default_test_file = "ASAP2_Demo_V171.mf4"

    def setUp(self):
        super().setUp()
        self.widget = None
        self.plot = None

        patcher = mock.patch("asammdf.gui.widgets.file.ErrorDialog")
        self.mc_widget_ed = patcher.start()
        self.addCleanup(patcher.stop)

        patcher_settings = mock.patch("asammdf.gui.widgets.batch.QtCore.QSettings", spec=QtCore.QSettings)
        self.mc_settings = patcher_settings.start()
        self.mc_settings.return_value.value.side_effect = itertools.chain(["Natural sort"], itertools.repeat(None))
        self.mc_settings.return_value.setValue.return_value = None
        self.addCleanup(patcher_settings.stop)

        patcher_restore = mock.patch.object(BatchWidget, "restore_export_settings")
        self.mc_restore = patcher_restore.start()
        self.addCleanup(patcher_restore.stop)

        self.processEvents()

    def setUpBatchWidget(self, *args, measurement_files: Sequence[str] | None):
        """
        Created because a lot of testcases,
        we do not need other parameters for BatchWidget initialization.
        """
        if measurement_files is None:
            self.widget = BatchWidget(*args)
            self.processEvents()

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

    def select_channels(self, channels_list: Iterable[str | int]) -> list:
        """
        Select channels from a list of names or indexes.

        Parameters
        ----------
        channels_list: a list of channel names or indexes

        Returns
        -------
        channels names list
        """
        self.selected_channels = []
        iterator = QTreeWidgetItemIterator(self.widget.filter_tree)

        count = 0
        while iterator.value():
            item = iterator.value()
            if (
                (item.name in channels_list or count in channels_list)
                and "time" not in item.name.lower()
                and "$" not in item.name
            ):  # by name or index, exclde time and calibration channels
                item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                self.assertTrue(item.checkState(0) == QtCore.Qt.CheckState.Checked)
                self.selected_channels.append(item.name)
                self.processEvents()

            iterator += 1
            count += 1

        # Evaluate that channels were added to "selected_filter_channels"
        for index in range(self.widget.selected_filter_channels.count()):
            item = self.widget.selected_filter_channels.item(index)
            self.assertIn(item.text(), self.selected_channels)

        return self.selected_channels

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

    def mouse_click_on_btn_with_progress(self, btn: QPushButton):
        """
        Click on the button with progress bar and wait until progress bar will be closed.

        Parameters
        ----------
        btn := QPushButton
        """
        # Mouse click on button
        QTest.mouseClick(btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for progress bar thread to finish
        while self.widget._progress:
            self.processEvents(0.01)
        self.processEvents()

    def toggle_checkboxes(self, widget: QWidget, check=True):
        """
        Iterate given widget to find QCheckBox items.
        All founded checkboxes will have 'check' check state

        Parameters
        ----------
        widget := widget to iterate
        check := check state (True/False)

        """
        # set checkboxes check state
        for checkbox in widget.findChildren(QtWidgets.QCheckBox):
            if checkbox.isChecked() != check:
                self.mouseClick_CheckboxButton(checkbox)

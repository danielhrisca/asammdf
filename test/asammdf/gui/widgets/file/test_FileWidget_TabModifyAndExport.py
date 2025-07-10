#!/usr/bin/env python
import pathlib
import shutil
import time
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets

from test.asammdf.gui.test_base import OpenMDF
from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestTabModifyAndExport(TestFileWidget):
    def test_PushButton_ScrambleTexts(self):
        """
        Events:
            - Open 'FileWidget' with valid measurement.
            - Go to Tab: "Modify & Export": Index 1
            - Press PushButton "Scramble texts"
        Evaluate:
            - New file is created
            - No channel from first file is found in 2nd file (scrambled file)
        """
        # Setup
        measurement_file = str(pathlib.Path(self.test_workspace, "ASAP2_Demo_V171.mf4"))
        shutil.copy(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"), measurement_file)
        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)
        # Press PushButton ScrambleTexts
        QtTest.QTest.mouseClick(self.widget.scramble_btn, QtCore.Qt.MouseButton.LeftButton)

        channels = list(self.widget.mdf.channels_db)

        # Evaluate
        scrambled_filepath = pathlib.Path(self.test_workspace, "ASAP2_Demo_V171.scrambled.mf4")
        # Wait for Thread to finish
        time.sleep(0.1)

        with OpenMDF(scrambled_filepath) as mdf_file:
            result = filter(lambda c: c in mdf_file.channels_db, channels)
            self.assertFalse(any(result))

    def test_ExportMDF(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Open 'FileWidget' with valid measurement.
            - Go to Tab: "Modify & Export": Index 1
            - Set "channel_view" to "Natural sort"
            - Select two channels
            - Ensure that output format is MDF
            - Case 0:
                - Press PushButton Apply.
                    - Simulate that no valid path is provided.
            - Case 1:
                - Press PushButton Apply.
                    - Simulate that no valid path is provided.
        Evaluate:
            - Evaluate that file was created.
            - Open File and check that there are only two channels.
        """
        # Setup
        measurement_file = str(pathlib.Path(self.resource, "ASAP2_Demo_V171.mf4"))
        # Event
        self.setUpFileWidget(measurement_file=measurement_file, default=True)
        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)
        self.widget.filter_view.setCurrentText("Natural sort")

        count = 2
        selected_channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.filter_tree)
        while iterator.value() and count:
            item = iterator.value()
            item.setCheckState(0, QtCore.Qt.CheckState.Checked)
            self.assertTrue(item.checkState(0) == QtCore.Qt.CheckState.Checked)
            selected_channels.append(item.text(0))
            iterator += 1
            count -= 1
        # Evaluate that channels were added to "selected_filter_channels"
        for index in range(self.widget.selected_filter_channels.count()):
            item = self.widget.selected_filter_channels.item(index)
            self.assertIn(item.text(), selected_channels)

        self.widget.output_format.setCurrentText("MDF")

        # Case 0:
        self.processEvents()
        with self.subTest("test_ExportMDF_0"):
            with (
                mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getSaveFileName") as mc_getSaveFileName,
                mock.patch("asammdf.gui.widgets.file.setup_progress") as mo_setup_progress,
            ):
                mc_getSaveFileName.return_value = None, None
                QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
                self.processEvents()
            # Evaluate
            # Progress is not created
            mo_setup_progress.assert_not_called()

        # Case 1:
        self.processEvents()
        with self.subTest("test_ExportMDF_1"):
            saved_file = pathlib.Path(self.test_workspace, f"{self.id()}.mf4")
            with mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getSaveFileName") as mc_getSaveFileName:
                mc_getSaveFileName.return_value = str(saved_file), None
                QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
                self.processEvents(1)
        # Wait for thread to finish
        self.processEvents(1)

        # Evaluate
        self.assertTrue(saved_file.exists())

        # TearDown Widget
        with OpenMDF(saved_file) as mdf_file:
            for sig in mdf_file.iter_channels():
                self.assertIn(sig.name, selected_channels)

#!/usr/bin/env python
import pathlib
import shutil
import time
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets

from asammdf import mdf
from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestTabModifyAndExport(TestBatchWidget):
    def test_PushButton_ScrambleTexts(self):
        """
        Events:
            - Open 'BatchWidget' with valid measurement.
            - Go to Tab: "Modify & Export": Index 1
            - Press PushButton "Scramble texts"
        Evaluate:
            - New file is created
            - No channel from first file is found in 2nd file (scrambled file)
        """
        file = "ASAP2_Demo_V171.mf4"
        # Setup
        measurement_file = str(pathlib.Path(self.resource, file))
        shutil.copy(measurement_file, self.test_workspace)
        filepath = str(pathlib.Path(self.test_workspace, file))
        # Event
        self.setUpBatchWidget(measurement_files=[filepath], default=None)

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)
        # Press PushButton ScrambleTexts
        QtTest.QTest.mouseClick(self.widget.scramble_btn, QtCore.Qt.MouseButton.LeftButton)

        channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.filter_tree)
        while item := iterator.value():
            iterator += 1
            channels.append(item.name)

        self.processEvents(1)
        # Evaluate
        scrambled_filepath = pathlib.Path(self.test_workspace, file.replace(".", ".scrambled."))
        self.assertTrue(scrambled_filepath.exists())
        # Wait for Thread to finish
        time.sleep(0.1)

        # get saved file as MDF
        process_bus_logging = ("process_bus_logging", True)
        mdf_file = mdf.MDF(scrambled_filepath, process_bus_logging=process_bus_logging)

        # Evaluate
        for name in channels:
            self.assertNotIn(name, mdf_file.channels_db.keys())
        mdf_file.close()
        self.processEvents(1)

    def test_ExportMDF(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Open 'BatchWidget' with valid measurement.
            - Go to Tab: "Modify & Export": Index 1
            - Set "channel_view" to "Natural sort"
            - Select two channels
            - Ensure that output format is MDF

            - Case 1:
                - Press PushButton Apply.
                    - Simulate that valid path is provided.
        Evaluate:
            - Evaluate that file was created.
            - Open File and check that there are only two channels.
        """
        # Setup
        file = "ASAP2_Demo_V171.mf4"
        measurement_file = pathlib.Path(self.resource, file)
        shutil.copy(measurement_file, self.test_workspace)
        filepath = str(pathlib.Path(self.test_workspace, file))
        # Event
        self.setUpBatchWidget(measurement_files=[filepath], default=None)
        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)
        self.widget.filter_view.setCurrentText("Natural sort")

        count = 3
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

        self.processEvents()
        saved_file = pathlib.Path(self.test_workspace, "ASAP2_Demo_V171.modified.mf4")
        with mock.patch("asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getSaveFileName") as mc_getSaveFileName:
            mc_getSaveFileName.return_value = str(saved_file), None
            QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
            self.processEvents(1)
        # Wait for thread to finish
        self.processEvents(1)

        # Evaluate
        self.assertTrue(saved_file.exists())

        channels = []
        # get saved file as MDF
        process_bus_logging = ("process_bus_logging", True)
        mdf_file = mdf.MDF(saved_file, process_bus_logging=process_bus_logging)

        # Evaluate
        for name in channels:
            self.assertIn(name, mdf_file.channels_db.keys())

        mdf_file.close()
        self.processEvents(1)

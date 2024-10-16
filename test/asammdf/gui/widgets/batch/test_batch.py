#!/usr/bin/env python
import pathlib
import shutil
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets

from asammdf import mdf
from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestBatchFilesList(TestBatchWidget):
    def test_PushButtons_Sort(self):
        """
        Events:
            - Open 'BatchWidget' with 2 valid measurements.
            - Press PushButton "Sort by start time"
            - Press PushButton "Sort alphabetically"

        Evaluate:
            - Files are sorted by start time
            - Files are sorted alphabetically
        """
        # Setup
        file_1 = "ASAP2_Demo_V171.mf4"
        file_2 = "test_batch.mf4"
        measurement_file_1 = pathlib.Path(self.resource, file_1)
        measurement_file_2 = pathlib.Path(self.resource, file_2)
        test_file_1 = pathlib.Path(self.test_workspace, file_1)
        test_file_2 = pathlib.Path(self.test_workspace, file_2)
        alpha_sort = [str(test_file_1), str(test_file_2)]
        time_sort = [str(test_file_2), str(test_file_1)]

        # copy files in test workspace directory
        shutil.copyfile(measurement_file_1, test_file_1)
        shutil.copyfile(measurement_file_2, test_file_2)

        # Event
        self.setUpBatchWidget(measurement_files=alpha_sort, default=None)

        with self.subTest("test sort by start time btn"):
            # Press `Sort by start time` button
            QtTest.QTest.mouseClick(self.widget.sort_by_start_time_btn, QtCore.Qt.MouseButton.LeftButton)

            # Evaluate
            self.assertEqual([self.widget.files_list.item(row).text() for row in range(2)], time_sort)

        with self.subTest("test sort alphabetically btn"):
            # Press `Sort by start time` button
            QtTest.QTest.mouseClick(self.widget.sort_alphabetically_btn, QtCore.Qt.MouseButton.LeftButton)

            # Evaluate
            self.assertEqual([self.widget.files_list.item(row).text() for row in range(2)], alpha_sort)


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
        # Setup
        file = "ASAP2_Demo_V171.mf4"
        measurement_file = pathlib.Path(self.resource, file)
        test_file = pathlib.Path(self.test_workspace, file)
        scrambled_filepath = pathlib.Path(self.test_workspace, file.replace(".", ".scrambled."))

        shutil.copyfile(measurement_file, test_file)
        self.setUpBatchWidget(measurement_files=[str(test_file)], default=None)

        # Events
        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)
        # Press PushButton ScrambleTexts
        QtTest.QTest.mouseClick(self.widget.scramble_btn, QtCore.Qt.MouseButton.LeftButton)

        channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.filter_tree)
        while item := iterator.value():
            iterator += 1
            channels.append(item.name)
        # Wait for Thread to finish
        self.processEvents(1)

        # Evaluate
        self.assertTrue(scrambled_filepath.exists())

        # get saved file as MDF
        process_bus_logging = ("process_bus_logging", True)
        mdf_file = mdf.MDF(scrambled_filepath, process_bus_logging=process_bus_logging)

        # Evaluate file
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
            - Select 5 channels
            - Check `Cut` checkbox and set `Start` and `End` times for cut function
            - Ensure that output format is MDF
            - Set output folder to `test_workspace`
            - Press PushButton Apply.

        Evaluate:
            - Evaluate that file was created.
            - Ensure that output file has only selected channels
            - For each channel Start time and End time is 0.5s and 1.5s
        """
        # Setup
        min_time = 0.5
        max_time = 1.5
        file = "ASAP2_Demo_V171.mf4"
        measurement_file = str(pathlib.Path(self.resource, file))
        saved_file = pathlib.Path(self.test_workspace, file)

        # Event
        self.setUpBatchWidget(measurement_files=[measurement_file], default=None)
        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)
        self.widget.filter_view.setCurrentText("Natural sort")

        count = 5
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

        # Setup for Cut
        self.widget.cut_group.setChecked(True)
        self.widget.cut_start.setValue(min_time)
        self.widget.cut_stop.setValue(max_time)

        # Ensure output format
        self.widget.output_format.setCurrentText("MDF")
        # Ensure output folder
        self.widget.modify_output_folder.setText(str(self.test_workspace))

        self.processEvents()
        # Mouse click on Apply button
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(1)

        # Evaluate
        self.assertTrue(saved_file.exists())

        # get saved file as MDF
        process_bus_logging = ("process_bus_logging", True)
        mdf_file = mdf.MDF(saved_file, process_bus_logging=process_bus_logging)

        # Evaluate saved file
        for channel in mdf_file.iter_channels():
            self.assertIn(channel.name, selected_channels)
            selected_channels.remove(channel.name)  # remove if ok

            self.assertEqual(channel.timestamps.min(), min_time)
            self.assertEqual(channel.timestamps.max(), max_time)

        self.assertEqual(len(selected_channels), 0)
        # tearDown
        mdf_file.close()
        self.processEvents(1)


class TestTabStack(TestBatchWidget):
    def test_PushButton_Stack(self):
        """
        Events:
            - Open 'BatchWidget' with 2 valid measurement.
            - Go to Tab: "Stack": Index 2
            - Press PushButton "Stack"

        Evaluate:
            - New file is created
            - All channels from both files is available in new created file
        """
        # Setup
        file_1 = "ASAP2_Demo_V171.mf4"
        file_2 = "test_batch.mf4"
        measurement_file_1 = pathlib.Path(self.resource, file_1)
        measurement_file_2 = pathlib.Path(self.resource, file_2)

        saved_file = pathlib.Path(self.test_workspace, self.id().split(".")[-1] + ".mf4")

        self.setUpBatchWidget(measurement_files=[str(measurement_file_1), str(measurement_file_2)], default=None)

        # Event
        with mock.patch("asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = str(saved_file), ""
            QtTest.QTest.mouseClick(self.widget.stack_btn, QtCore.Qt.MouseButton.LeftButton)
        # Let progress bar finish
        self.processEvents(1)
        # Evaluate that file was created
        self.assertTrue(saved_file.exists())

        # get files as MDF
        process_bus_logging = ("process_bus_logging", True)
        mdf_saved_file = mdf.MDF(saved_file, process_bus_logging=process_bus_logging)
        mdf_1_file = mdf.MDF(measurement_file_1, process_bus_logging=process_bus_logging)
        mdf_2_file = mdf.MDF(measurement_file_2, process_bus_logging=process_bus_logging)

        all_channels = [channel.name for channel in mdf_1_file.iter_channels()]
        for channel in mdf_2_file.iter_channels():
            all_channels.append(channel.name)

        # Evaluate saved file
        for channel in mdf_saved_file.iter_channels():
            self.assertIn(channel.name, all_channels)

        for file in mdf_1_file, mdf_2_file, mdf_saved_file:
            file.close()
        self.processEvents()

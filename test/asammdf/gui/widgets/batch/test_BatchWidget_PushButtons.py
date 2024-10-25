#!/usr/bin/env python
import pathlib
from random import randint
import shutil
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets

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


class TestTabConcatenate(TestBatchWidget):
    default_test_file = "ASAP2_Demo_V171.mf4"
    test_file_0 = "test_file_0.mf4"
    test_file_1 = "test_file_1.mf4"
    output_file = "output.mf4"
    # Start and end timestamps to cut from default test file to generate 2 test files
    start_time_0 = randint(0, 20) / 10
    start_time_1 = randint(60, 90) / 10
    end_time_0 = randint(20, 40) / 10
    end_time_1 = randint(90, 120) / 10

    def setUp(self):
        """
        Create 2 measurement files from default test file for feature tests use
        """
        super().setUp()
        # Setup
        self.measurement_file = pathlib.Path(self.resource, self.default_test_file)
        test_file = pathlib.Path(self.test_workspace, self.default_test_file)

        shutil.copyfile(self.measurement_file, test_file)
        self.setUpBatchWidget(measurement_files=[str(self.measurement_file)], default=None)
        self.processEvents()

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)
        self.widget.filter_view.setCurrentText("Natural sort")
        self.processEvents(0.1)

        count = 5
        self.channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.filter_tree)
        while iterator.value() and count:
            item = iterator.value()
            item.setCheckState(0, QtCore.Qt.CheckState.Checked)
            self.assertTrue(item.checkState(0) == QtCore.Qt.CheckState.Checked)
            self.channels.append(item.text(0))
            iterator += 1
            count -= 1
        # Evaluate that channels were added to "selected_filter_channels"
        for index in range(self.widget.selected_filter_channels.count()):
            item = self.widget.selected_filter_channels.item(index)
            self.assertIn(item.text(), self.channels)

        # Export selected channels in 2 files
        self.test_file_0 = pathlib.Path(self.test_workspace, self.test_file_0)
        self.test_file_1 = pathlib.Path(self.test_workspace, self.test_file_1)
        self.output_file = pathlib.Path(self.test_workspace, self.output_file)

        self.cut_file(self.start_time_0, self.end_time_0)
        test_file.rename(self.test_file_0)
        self.assertTrue(self.test_file_0.exists())

        self.cut_file(self.start_time_1, self.end_time_1)
        test_file.rename(self.test_file_1)
        self.assertTrue(self.test_file_1.exists())
        self.mouseClick_WidgetItem(self.widget.files_list.item(0))
        QtTest.QTest.keyClick(self.widget.files_list, QtCore.Qt.Key.Key_Delete)
        self.processEvents(0.1)

    def test_PushButton_Concatenate(self):
        """

        Returns
        -------

        """
        # Go to Tab: Concatenate
        self.widget.aspects.setCurrentIndex(self.concatenate_aspect)
        # Add test files
        self.widget.files_list.addItems([str(self.test_file_0), str(self.test_file_1)])
        self.processEvents(0.1)

        # Event
        with mock.patch("asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = self.output_file, ""
            QtTest.QTest.mouseClick(self.widget.concatenate_btn, QtCore.Qt.MouseButton.LeftButton)
        # Allow progress bar to close
        self.processEvents(2)
        self.assertTrue(self.output_file.exists())

        # Evaluate
        with self.OpenMDF(self.output_file) as mdf_file:
            # Evaluate saved file
            for name in self.channels:
                self.assertIn(name, mdf_file.channels_db.keys())

            channel = next(mdf_file.iter_channels())
            self.assertAlmostEqual(channel.timestamps.min(), self.start_time_0)
            self.assertAlmostEqual(channel.timestamps.max(), self.end_time_1)

        self.widget.files_list.clear()


class TestTabModifyAndExport(TestBatchWidget):
    default_test_file = "ASAP2_Demo_V171.mf4"

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
        measurement_file = str(pathlib.Path(self.resource, self.default_test_file))
        test_file = pathlib.Path(self.test_workspace, self.default_test_file)
        scrambled_filepath = pathlib.Path(self.test_workspace, self.default_test_file.replace(".", ".scrambled."))

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
        with self.OpenMDF(scrambled_filepath) as mdf_file:
            # Evaluate file
            for name in channels:
                self.assertNotIn(name, mdf_file.channels_db.keys())

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
        measurement_file = str(pathlib.Path(self.resource, self.default_test_file))
        saved_file = pathlib.Path(self.test_workspace, self.default_test_file)

        # Event
        self.setUpBatchWidget(measurement_files=[measurement_file], default=None)
        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)
        self.widget.filter_view.setCurrentText("Natural sort")
        self.processEvents(0.1)

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

        self.cut_file(min_time, max_time)

        # Evaluate
        self.assertTrue(saved_file.exists())

        # get saved file as MDF
        with self.OpenMDF(saved_file) as mdf_file:
            # Evaluate saved file
            for channel in mdf_file.iter_channels():
                self.assertIn(channel.name, selected_channels)
                selected_channels.remove(channel.name)  # remove if ok

                self.assertEqual(channel.timestamps.max(), max_time)

            self.assertEqual(len(selected_channels), 0)
        # tearDown
        self.processEvents(1)


class TestTabStack(TestBatchWidget):
    default_test_file = "ASAP2_Demo_V171.mf4"
    class_test_file = "test_batch.mf4"

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
        measurement_file_1 = pathlib.Path(self.resource, self.default_test_file)
        measurement_file_2 = pathlib.Path(self.resource, self.class_test_file)
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

        channels = []
        # get files as MDF
        with self.OpenMDF(measurement_file_1) as mdf_file:
            channels.extend(mdf_file.iter_channels())

        with self.OpenMDF(measurement_file_2) as mdf_file:
            channels.extend(mdf_file.iter_channels())

        # Evaluate saved file
        with self.OpenMDF(saved_file) as mdf_file:
            for channel in mdf_file.iter_channels():
                self.assertIn(channel.name, channels)

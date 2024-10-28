#!/usr/bin/env python
from pathlib import Path
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets

from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestBatchFilesList(TestBatchWidget):
    test_file = "test_batch.mf4"

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
        file_0 = Path(self.test_workspace, self.default_test_file)
        file_1 = Path(self.test_workspace, self.test_file)

        alpha_sort = [str(file_0), str(file_1)]
        time_sort = [str(file_1), str(file_0)]

        self.setUpBatchWidget(measurement_files=alpha_sort)

        # Event
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
    test_file_0_name = "test_batch_cut_0.mf4"
    test_file_1_name = "test_batch_cut_1.mf4"
    output_file_name = "output.mf4"

    def test_PushButton_Concatenate(self):
        """
        Events:
            - Open 'BatchWidget' with created measurement.
            - Go to Tab: "Concatenate": Index 0
            - Press PushButton "Concatenate"

        Evaluate:
            - New file is created
            - No channel from first file is found in 2nd file (scrambled file)
        """
        output_file = Path(self.test_workspace, self.output_file_name)
        test_file_0 = Path(self.test_workspace, self.test_file_0_name)
        test_file_1 = Path(self.test_workspace, self.test_file_1_name)

        # Get evaluation data
        channels = set()
        with self.OpenMDF(test_file_0) as mdf_file:
            for channel in mdf_file.iter_channels():
                channels.add(channel.name)
            _min = channel.timestamps.min()

        with self.OpenMDF(test_file_1) as mdf_file:
            channel = next(mdf_file.iter_channels())
            _max = channel.timestamps.max()

        # Setup
        self.setUpBatchWidget(measurement_files=[str(test_file_0), str(test_file_1)])
        # Go to Tab: Concatenate
        self.widget.aspects.setCurrentIndex(self.concatenate_aspect)
        self.processEvents(0.1)

        # Event
        with mock.patch("asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = output_file, ""
            QtTest.QTest.mouseClick(self.widget.concatenate_btn, QtCore.Qt.MouseButton.LeftButton)
        # Allow progress bar to close
        self.processEvents(2)

        # Evaluate that file exist
        self.assertTrue(output_file.exists())

        # Evaluate
        with self.OpenMDF(output_file) as mdf_file:
            # Evaluate saved file
            for name in channels:
                self.assertIn(name, mdf_file.channels_db.keys())

            channel = next(mdf_file.iter_channels())
            self.assertAlmostEqual(channel.timestamps.min(), _min)
            self.assertAlmostEqual(channel.timestamps.max(), _max)


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
        test_file = Path(self.test_workspace, self.default_test_file)
        scrambled_filepath = Path(self.test_workspace, self.default_test_file.replace(".", ".scrambled."))

        self.setUpBatchWidget(measurement_files=None)

        channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.filter_tree)
        while item := iterator.value():
            iterator += 1
            channels.append(item.name)

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)

        # Events
        QtTest.QTest.mouseClick(self.widget.scramble_btn, QtCore.Qt.MouseButton.LeftButton)

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
            - Select some channels
            - Ensure that output format is MDF
            - Press PushButton Apply.

        Evaluate:
            - Evaluate that file was created.
            - Ensure that output file has only selected channels
        """
        # Setup
        measurement_file = str(Path(self.test_workspace, self.default_test_file))
        saved_file = Path(self.test_workspace, self.default_test_file.replace(".", ".modified."))

        self.setUpBatchWidget(measurement_files=[measurement_file])

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)

        self.widget.filter_view.setCurrentText("Natural sort")
        self.processEvents(0.1)

        selected_channels = self.select_channels(5, 15)

        # Ensure output format
        self.widget.output_format.setCurrentText("MDF")

        # Mouse click on Apply button
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(2)

        # Evaluate
        self.assertTrue(saved_file.exists())

        # get saved file as MDF
        with self.OpenMDF(saved_file) as mdf_file:
            # Evaluate saved file
            for channel in selected_channels:
                self.assertIn(channel, mdf_file.channels_db.keys())


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
        measurement_file_1 = Path(self.test_workspace, self.default_test_file)
        measurement_file_2 = Path(self.test_workspace, self.class_test_file)
        saved_file = Path(self.test_workspace, self.id().split(".")[-1] + ".mf4")

        self.setUpBatchWidget(measurement_files=[str(measurement_file_1), str(measurement_file_2)])

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

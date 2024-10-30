#!/usr/bin/env python
import datetime
from pathlib import Path
from random import randint
from unittest import mock

from PySide6 import QtCore, QtTest, QtWidgets

from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtons(TestBatchWidget):

    def setUp(self):
        self.copy_mdf_files_to_workspace()
        super().setUp()

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

        self.setUpBatchWidget(measurement_files=[str(test_file)])

        channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.filter_tree)
        while item := iterator.value():
            iterator += 1
            channels.append(item.name)

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(self.modify_aspect)

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
            - File was created.
            - Ensure that output file has only selected channels
        """
        # Setup
        measurement_file = str(Path(self.test_workspace, self.default_test_file))
        saved_file = Path(self.test_workspace, self.default_test_file.replace(".", ".modified."))

        self.setUpBatchWidget(measurement_files=[measurement_file])

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(self.modify_aspect)

        self.widget.filter_view.setCurrentText("Natural sort")
        self.processEvents(0.1)

        count = self.widget.filter_tree.topLevelItemCount()
        selected_channels = self.select_channels(randint(0, int(count / 2) - 1), randint(int(count / 2) + 1, count - 1))

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


class TestModifyAndExport(TestBatchWidget):
    def setUp(self):
        """
        Events
            - Open 'BatchWidget' with valid measurement.
            - Select random channels
            - Set test workspace folder as output folder

        Evaluate
            - Modify output folder line text is equal to test workspace folder
        """
        super().setUp()
        file = Path(self.resource, self.default_test_file)

        self.setUpBatchWidget(measurement_files=[str(file)])

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(self.modify_aspect)
        self.widget.filter_view.setCurrentText("Natural sort")
        self.processEvents(0.1)

        count = self.widget.filter_tree.topLevelItemCount()
        self.channels = self.select_channels(randint(0, int(count / 2) - 1), randint(int(count / 2) + 1, count - 1))
        self.output_file = Path(self.test_workspace, self.default_test_file)

        with self.OpenMDF(file) as mdf_file:
            self.start_time = mdf_file.start_time.astimezone().replace(tzinfo=None)
            self.ts_min = min(
                [
                    ch.timestamps.min()
                    for ch in mdf_file.iter_channels()
                    if ch.timestamps.min() != 0 and ch.name in self.channels
                ]
            )

        # set test_workspace folder as output folder
        with mock.patch(
            "asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getExistingDirectory"
        ) as mo_getExistingDirectory:
            mo_getExistingDirectory.return_value = self.test_workspace
            QtTest.QTest.mouseClick(self.widget.modify_output_folder_btn, QtCore.Qt.MouseButton.LeftButton)

        self.assertEqual(self.widget.modify_output_folder.text().strip(), self.test_workspace)

    def test_cut_group(self):
        """
        Events
            - Check cut group checkbox
            - Set cut start and stop time
            - Check whence checkbox
            - Check cut time from zero checkbox
            - Press Apply bush button

        Evaluate
            - In test workspace folder was created file with same name as original measurement file
            - Start time for mdf file is updated
            - Timestamps start for all channels is equal to 0
            - Timestamps stop for all channels is equal to difference between stop and start time
        """
        # Setup
        start_cut = randint(1, 42) / 10
        stop_cut = randint(60, 90) / 10
        expected_start_time = self.start_time + datetime.timedelta(milliseconds=(start_cut + self.ts_min) * 1000)

        QtTest.QTest.keyClick(self.widget.cut_group, QtCore.Qt.Key.Key_Space)
        # self.widget.cut_group.setChecked(True)
        self.widget.cut_start.setValue(start_cut)
        self.widget.cut_stop.setValue(stop_cut)
        self.mouseClick_CheckboxButton(self.widget.whence)
        self.mouseClick_CheckboxButton(self.widget.cut_time_from_zero)
        self.processEvents(0.01)

        # Event
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents(2)

        # Evaluate
        self.output_file.exists()
        with self.OpenMDF(self.output_file) as mdf_file:
            time_dif = expected_start_time - mdf_file.start_time
            self.assertEqual(time_dif.microseconds, 0)
            self.assertEqual(time_dif.seconds, 0)
            self.assertEqual(time_dif.days, 0)
            for channel in mdf_file.iter_channels():
                self.assertEqual(channel.timestamps.min(), 0)
                self.assertEqual(channel.timestamps.max(), stop_cut - start_cut)

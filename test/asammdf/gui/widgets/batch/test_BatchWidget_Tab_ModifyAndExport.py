#!/usr/bin/env python
import datetime
from pathlib import Path
from random import randint
from unittest import mock

import scipy
from h5py import File as HDF5
import pandas as pd
from PySide6 import QtCore, QtTest, QtWidgets

from test.asammdf.gui.test_base import OpenMDF
from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget


# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtonScrambleTexts(TestBatchWidget):
    def test_ScrambleTexts(self):
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
        self.copy_mdf_files_to_workspace()

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
        with OpenMDF(scrambled_filepath) as mdf_file:
            # Evaluate file
            for name in channels:
                self.assertNotIn(name, mdf_file.channels_db)


class TestPushButtonApply(TestBatchWidget):
    def setUp(self):
        """
        Events
            - Open 'BatchWidget' with valid measurement.
            - Go to Tab: "Modify & Export": Index 1
            - Set "channel_view" to "Natural sort"
            - Select some channels
            - Set test workspace directory as output folder

        Evaluate
            Output folder text is changed to test workspace directory

        """
        super().setUp()
        self.measurement_file = str(Path(self.resource, self.default_test_file))

        self.setUpBatchWidget(measurement_files=[self.measurement_file])

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(self.modify_aspect)

        self.widget.filter_view.setCurrentText("Natural sort")
        self.processEvents(0.1)

        count = self.widget.filter_tree.topLevelItemCount()
        self.selected_channels = self.select_channels(
            randint(0, int(count / 2) - 1), randint(int(count / 2) + 1, count - 1)
        )

        # set test_workspace folder as output folder
        with mock.patch(
            "asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getExistingDirectory"
        ) as mo_getExistingDirectory:
            mo_getExistingDirectory.return_value = self.test_workspace
            QtTest.QTest.mouseClick(self.widget.modify_output_folder_btn, QtCore.Qt.MouseButton.LeftButton)

        self.assertEqual(self.widget.modify_output_folder.text().strip(), self.test_workspace)

    def test_output_format_MDF(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is MDF
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
        """
        # Expected result
        saved_file = Path(self.test_workspace, self.default_test_file)

        # Ensure output format
        self.widget.output_format.setCurrentText("MDF")

        # Event
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(2)

        # Evaluate
        self.assertTrue(saved_file.exists())

        # get saved file as MDF
        with OpenMDF(saved_file) as mdf_file:
            # Evaluate saved file
            for channel in self.selected_channels:
                self.assertIn(channel, mdf_file.channels_db)

    def test_output_format_ASC(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is ASC
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
        """
        # Ensure output format
        self.widget.output_format.setCurrentText("ASC")

        # Expected results
        saved_file = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".asc"))
        with OpenMDF(self.measurement_file) as mdf_file:
            start = mdf_file.start_time.strftime("%a %b %d %I:%M:%S.%f %p %Y")

        expected_text = f"date {start}\nbase hex  timestamps absolute\nno internal events logged\n"

        # Event
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(2)

        # Evaluate
        self.assertTrue(saved_file.exists())
        with open(saved_file) as asc_file:
            self.assertEqual(asc_file.read(), expected_text)

        # cls.tempdir_obd = tempfile.TemporaryDirectory()
        #
        # url = "https://github.com/danielhrisca/asammdf/files/4328945/OBD2-DBC-MDF4.zip"
        # urllib.request.urlretrieve(url, "test.zip")
        # ZipFile(r"test.zip").extractall(cls.tempdir_obd.name)
        # Path("test.zip").unlink()

    def test_output_format_CSV(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is CSV
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
        """
        # Ensure output format
        self.widget.output_format.setCurrentText("CSV")
        # uncheck all checkboxes
        for checkbox in self.widget.CSV.findChildren(QtWidgets.QCheckBox):
            if checkbox.isChecked():
                self.mouseClick_CheckboxButton(checkbox)

        self.processEvents()
        # Expected results
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Event
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(3)

        # Evaluate
        for index, (group_name, channels_list) in enumerate(groups.items()):
            csv_file = Path(
                self.test_workspace,
                self.default_test_file.replace(
                    ".mf4", f"{group_name.replace(group_name[:4], f".ChannelGroup_{index}").replace(" ", "_")}.csv"
                ),
            )
            self.assertTrue(csv_file.exists(), csv_file)
            pandas_tab = pd.read_csv(csv_file)
            for channel in channels_list:
                self.assertIn(channel, pandas_tab.columns)

        # ToDo is necessary to evaluate dataframe values (for each signal, min, max, len)

    def test_output_format_HDF5(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is MDF
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
        """
        # Ensure output format
        self.widget.output_format.setCurrentText("HDF5")

        # Expected results
        hdf5_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".hdf"))
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Mouse click on Apply button
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish

        # Evaluate
        self.processEvents(2)
        self.assertTrue(hdf5_path.exists())

        hdf5_file = HDF5(hdf5_path)
        self.assertEqual(len(hdf5_file.items()) - 1, len(groups))  # 5th item is file path
        for value, group in zip(hdf5_file.values(), groups.values()):
            for channel in group:
                self.assertIn(channel, value)
            # todo timestamps

    def test_output_format_MAT(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is MAT
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
        """
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Ensure output format
        self.widget.output_format.setCurrentText("MAT")

        # Expected results
        mat_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".mat"))
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Mouse click on Apply button
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish

        self.processEvents(2)

        # Evaluate
        self.assertTrue(mat_path.exists())

        mat_file = scipy.io.loadmat(str(mat_path))
        to_replace = {" ", ".", "[", "]"}
        for index, channels_list in enumerate(groups.values()):
            for channel in channels_list:
                for character in to_replace:
                    channel = channel.replace(character, "_")
                self.assertIn(f"DG{index}_{channel}", mat_file)

    def test_output_format_Parquet(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is Parquet
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
        """
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Ensure output format
        self.widget.output_format.setCurrentText("Parquet")

        # Expected results
        parquet_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".parquet"))
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Mouse click on Apply button
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish

        self.processEvents(2)

        # Evaluate
        self.assertTrue(parquet_path.exists())
        pandas_tab = pd.read_parquet(parquet_path)
        from_parquet = set(pandas_tab.columns)
        self.assertSetEqual(set(self.selected_channels), from_parquet)

    def test_cut_checkbox_0(self):
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
        # Expected values

        output_file = Path(self.test_workspace, self.default_test_file)

        with OpenMDF(self.measurement_file) as mdf_file:
            self.start_time = mdf_file.start_time.astimezone().replace(tzinfo=None)
            self.ts_min = min(
                [
                    ch.timestamps.min()
                    for ch in mdf_file.iter_channels()
                    if ch.timestamps.min() != 0 and ch.name in self.selected_channels
                ]
            )

        start_cut = randint(1, 42) / 10
        stop_cut = randint(60, 90) / 10
        expected_start_time = self.start_time + datetime.timedelta(milliseconds=(start_cut + self.ts_min) * 1000)

        # Setup
        QtTest.QTest.keyClick(self.widget.cut_group, QtCore.Qt.Key.Key_Space)
        self.widget.cut_start.setValue(start_cut)
        self.widget.cut_stop.setValue(stop_cut)
        self.mouseClick_CheckboxButton(self.widget.whence)
        self.mouseClick_CheckboxButton(self.widget.cut_time_from_zero)
        self.processEvents(0.01)

        # Event
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents(2)

        # Evaluate
        output_file.exists()
        with OpenMDF(output_file) as mdf_file:
            time_dif = expected_start_time - mdf_file.start_time
            self.assertEqual(time_dif.microseconds, 0)
            self.assertEqual(time_dif.seconds, 0)
            self.assertEqual(time_dif.days, 0)
            for channel in mdf_file.iter_channels():
                self.assertEqual(channel.timestamps.min(), 0)
                self.assertEqual(channel.timestamps.max(), stop_cut - start_cut)

    def test_cut_checkbox_1(self):
        """
        Events
            - Check cut group checkbox
            - Set cut start and stop time
            - Press Apply bush button

        Evaluate
            - In test workspace folder was created file with same name as original measurement file
            - Start time for mdf file is updated
            - Timestamps start for all channels is equal to 0
            - Timestamps stop for all channels is equal to difference between stop and start time
        """
        # Expected values
        output_file = Path(self.test_workspace, self.default_test_file)

        with OpenMDF(self.measurement_file) as mdf_file:
            self.start_time = mdf_file.start_time
            self.ts_min = min(
                [
                    ch.timestamps.min()
                    for ch in mdf_file.iter_channels()
                    if ch.timestamps.min() != 0 and ch.name in self.selected_channels
                ]
            )

        start_cut = randint(1, 42) / 10
        stop_cut = randint(60, 90) / 10

        # Setup
        QtTest.QTest.keyClick(self.widget.cut_group, QtCore.Qt.Key.Key_Space)
        self.widget.cut_start.setValue(start_cut)
        self.widget.cut_stop.setValue(stop_cut)
        self.processEvents(0.01)

        # Event
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        self.processEvents(2)

        # Evaluate
        output_file.exists()
        with OpenMDF(output_file) as mdf_file:
            self.assertEqual(self.start_time, mdf_file.start_time)

            for channel in mdf_file.iter_channels():
                self.assertEqual(channel.timestamps.min(), start_cut)
                self.assertEqual(channel.timestamps.max(), stop_cut)

#!/usr/bin/env python
import datetime
from pathlib import Path
from random import randint
from unittest import mock

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtTest, QtWidgets
import scipy

from test.asammdf.gui.test_base import OpenHDF5, OpenMDF
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
        time_channels = len(
            [
                self.widget.filter_tree.topLevelItem(_).text(0)
                for _ in range(count)
                if self.widget.filter_tree.topLevelItem(_).text(0) == "time"
            ]
        )
        count -= time_channels
        self.selected_channels = self.select_channels(
            randint(1, int(count / 2) - 1), randint(int(count / 2) + 1, count - 1)
        )

        # set test_workspace folder as output folder
        with mock.patch(
            "asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getExistingDirectory"
        ) as mo_getExistingDirectory:
            mo_getExistingDirectory.return_value = self.test_workspace
            QtTest.QTest.mouseClick(self.widget.modify_output_folder_btn, QtCore.Qt.MouseButton.LeftButton)

        self.assertEqual(self.widget.modify_output_folder.text().strip(), self.test_workspace)

    def set_up_qWidget(self, name: str, check: bool = True) -> bool:
        for q_widget in self.widget.output_options.children():
            if name in q_widget.objectName():
                break
        else:
            return False
        # uncheck all checkboxes
        for checkbox in q_widget.findChildren(QtWidgets.QCheckBox):
            if checkbox.isChecked() != check:
                self.mouseClick_CheckboxButton(checkbox)
        self.processEvents(0.01)
        return True

    def test_output_format_MDF(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Cut and resample features will be tested in separate tests.
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
        self.processEvents(3)

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
        name = "ASC"
        # Ensure output format
        self.widget.output_format.setCurrentText(name)

        # Expected results
        saved_file = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".asc"))
        with OpenMDF(self.measurement_file) as mdf_file:
            start = mdf_file.start_time.strftime("%a %b %d %I:%M:%S.%f %p %Y")

        expected_text = f"date {start}\nbase hex  timestamps absolute\nno internal events logged\n"

        # Event
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(3)

        # Evaluate
        self.assertTrue(saved_file.exists())
        with open(saved_file) as asc_file:
            self.assertEqual(asc_file.read(), expected_text)

        # ToDo use this file instead of default test file
        # cls.tempdir_obd = tempfile.TemporaryDirectory()
        #
        # url = "https://github.com/danielhrisca/asammdf/files/4328945/OBD2-DBC-MDF4.zip"
        # urllib.request.urlretrieve(url, "test.zip")
        # ZipFile(r"test.zip").extractall(cls.tempdir_obd.name)
        # Path("test.zip").unlink()

    def test_output_format_CSV_0(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is CSV, no checked checkboxes
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
            - Evaluate CSV files
        """
        name = "CSV"
        # Ensure output format
        self.widget.output_format.setCurrentText(name)
        self.set_up_qWidget(name, check=False)

        # Expected results
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Event
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(3)

        # Evaluate
        with OpenMDF(self.measurement_file) as mdf_file:
            for index, (group_name, channels_list) in enumerate(groups.items()):
                # prepare expected results
                suffix = group_name.replace(group_name[:4], f".ChannelGroup_{index}").replace(" ", "_") + ".csv"
                csv_file = Path(self.test_workspace, self.default_test_file.replace(".mf4", suffix))
                mdf_channels = mdf_file.select(channels_list)

                # Evaluate if file exist
                self.assertTrue(csv_file.exists(), csv_file)
                # Read file as pandas
                pandas_tab = pd.read_csv(csv_file)
                for channel in mdf_channels:
                    self.assertIn(channel.name, pandas_tab.columns)
                    if np.issubdtype(channel.samples.dtype, np.number):  # problematic conversion
                        self.assertEqual(channel.samples.max(), pandas_tab[channel.name].values.max())
                        self.assertEqual(channel.samples.min(), pandas_tab[channel.name].values.min())
                    self.assertEqual(channel.samples.size, pandas_tab[channel.name].values.size)
                    self.assertAlmostEqual(channel.timestamps.max(), pandas_tab.timestamps.max(), places=15)
                    self.assertAlmostEqual(channel.timestamps.min(), pandas_tab.timestamps.min(), places=15)

    def test_output_format_CSV_1(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is CSV, all checkboxes are checked
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
            - Evaluate CSV files
        """
        # Expected output file name
        csv_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".csv"))
        # Expected units
        units = [""]

        name = "CSV"
        # Ensure output format
        self.widget.output_format.setCurrentText(name)
        self.set_up_qWidget(name)

        # Event
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(3)

        # Evaluate
        self.assertTrue(csv_path.exists(), csv_path)

        with OpenMDF(self.measurement_file) as mdf_file:
            mdf_channels = mdf_file.select(self.selected_channels, raw=True)
            units.extend(channel.unit for channel in mdf_channels)

            # Get channels timestamps max, min, difference between extremes
            min_ts = min(channel.timestamps.min() for channel in mdf_channels)
            max_ts = max(channel.timestamps.max() for channel in mdf_channels)
            seconds = int(max_ts - min_ts)
            microseconds = np.floor((max_ts - min_ts - seconds) * pow(10, 6))

            # Read file as pandas
            pandas_tab = pd.read_csv(csv_path, header=[0, 1])
            pandas_tab.timestamps = pd.DatetimeIndex(
                pandas_tab.timestamps.values, yearfirst=True
            ).values - np.datetime64(mdf_file.start_time)

            # Evaluate timestamps min
            self.assertEqual(np.timedelta64(pandas_tab.timestamps.values.min(), "us").item().microseconds, 0)
            self.assertEqual(np.timedelta64(pandas_tab.timestamps.values.min(), "us").item().seconds, 0)
            # Evaluate timestamps max
            self.assertEqual(np.timedelta64(pandas_tab.timestamps.values.max(), "us").item().microseconds, microseconds)
            self.assertEqual(np.timedelta64(pandas_tab.timestamps.values.max(), "us").item().seconds, seconds)

            # Evaluate channels names and units
            for channel, unit in zip(mdf_channels, units):
                if channel.display_names:  # DI.<channel name>
                    display_name, _ = zip(*channel.display_names.items())
                    channel_name = display_name[0]
                else:
                    channel_name = channel.name
                self.assertIn(channel_name, pandas_tab.columns.get_level_values(0))
                if channel.unit:
                    self.assertIn(channel.unit, pandas_tab.columns.get_level_values(1))

    def test_output_format_MAT_0(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is MAT, no checkboxes checked
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
        """
        name = "MAT"
        # Ensure output format
        self.widget.output_format.setCurrentText(name)
        self.set_up_qWidget(name, check=False)

        # Expected results
        mat_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".mat"))
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Mouse click on Apply button
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(3)

        # Evaluate
        self.assertTrue(mat_path.exists())

        mat_file = scipy.io.loadmat(str(mat_path))
        to_replace = {" ", ".", "[", "]"}
        for index, channels_list in enumerate(groups.values()):
            for channel in channels_list:
                for character in to_replace:
                    channel = channel.replace(character, "_")
                self.assertIn(f"DG{index}_{channel}", mat_file)

    def test_output_format_HDF5_0(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is HDF5, no checked checkboxes
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
        """
        name = "HDF5"
        # Ensure output format
        self.widget.output_format.setCurrentText(name)
        self.set_up_qWidget(name, check=False)

        # Expected results
        hdf5_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".hdf"))
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Mouse click on Apply button
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(3)

        # Evaluate
        self.assertTrue(hdf5_path.exists())

        with OpenHDF5(hdf5_path) as hdf5_file, OpenMDF(self.measurement_file) as mdf_file:
            self.assertEqual(len(hdf5_file.items()) - 1, len(groups))  # 5th item is file path

            for mdf_group, hdf5_group in zip(groups.values(), hdf5_file.values()):
                for name in hdf5_group:
                    if name != "time":  # Evaluate channels
                        self.assertIn(name, mdf_group)
                        mdf_channel = mdf_file.select([name])[0]
                        hdf5_channel = hdf5_group.get(name)
                        # Evaluate values from extremes if samples are numbers
                        if np.issubdtype(mdf_channel.samples.dtype, np.number):
                            self.assertEqual(mdf_channel.samples.max(), max(hdf5_channel))
                            self.assertEqual(mdf_channel.samples.min(), min(hdf5_channel))
                        # Evaluate samples size
                        self.assertEqual(mdf_channel.samples.size, hdf5_channel.size)
                    else:  # evaluate timestamps
                        hdf5_channel = hdf5_group.get(name)  # for evaluation will be used latest mdf channel from group
                        self.assertEqual(mdf_channel.timestamps.max(), max(hdf5_channel))
                        self.assertEqual(mdf_channel.timestamps.min(), min(hdf5_channel))
                        self.assertEqual(mdf_channel.timestamps.size, hdf5_channel.size)

    def test_output_format_HDF5_1(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is HDF5, all checkboxes are checked
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
        """
        name = "HDF5"
        # Ensure output format
        self.widget.output_format.setCurrentText(name)
        self.set_up_qWidget(name)

        # Expected results
        hdf5_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".hdf"))

        # Mouse click on Apply button
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(3)

        # Evaluate
        self.assertTrue(hdf5_path.exists())

        with OpenHDF5(hdf5_path) as hdf5_file, OpenMDF(self.measurement_file) as mdf_file:
            self.assertEqual(len(hdf5_file.items()), 1)  # 1 item

            # Prepare results
            size = 0
            differences = []
            hdf5_channels = hdf5_file[str(hdf5_path)]

            for channel in self.selected_channels:  # Evaluate channels
                _channel = mdf_file.select([channel])[0]  # nu intreba
                if _channel.display_names:  # DI.<channel name>...
                    display_name, _ = zip(*_channel.display_names.items())
                    channel = display_name[0]

                # Because exist 2 channels with identical "DISPLAY" name -_-
                mdf_channel = mdf_file.select([(channel, _channel.group_index, _channel.channel_index)], raw=True)[0]
                hdf5_channel = hdf5_channels.get(channel)

                self.assertIn(channel, hdf5_channels)

                # Evaluate extremes
                self.assertEqual(mdf_channel.samples.max(), max(hdf5_channel))
                self.assertEqual(mdf_channel.samples.min(), min(hdf5_channel))

                # for feature timestamps evaluation
                ceva = mdf_channel.timestamps.max() - mdf_channel.timestamps.min()
                if ceva not in differences:
                    differences.append(ceva)
                    size += mdf_channel.timestamps.size

            # Evaluate size of timestamps
            self.assertEqual(hdf5_channel.size, size)

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
        name = "Parquet"
        # Ensure output format
        self.widget.output_format.setCurrentText(name)

        # Expected results
        parquet_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".parquet"))
        # groups = self.get_selected_groups(channels=self.selected_channels)

        # Mouse click on Apply button
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(3)

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
        # Wait for thread to finish
        self.processEvents(3)

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
        # Wait for thread to finish
        self.processEvents(3)

        # Evaluate
        output_file.exists()
        with OpenMDF(output_file) as mdf_file:
            self.assertEqual(self.start_time, mdf_file.start_time)

            for channel in mdf_file.iter_channels():
                self.assertEqual(channel.timestamps.min(), start_cut)
                self.assertEqual(channel.timestamps.max(), stop_cut)

#!/usr/bin/env python
import datetime
from math import ceil
import os
from pathlib import Path
from random import randint
import unittest
from unittest import mock
import urllib
import urllib.request
from zipfile import ZipFile

from can.io import ASCReader
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
        # Prepare workspace
        self.copy_mdf_files_to_workspace()

        test_file = Path(self.test_workspace, self.default_test_file)
        scrambled_filepath = Path(self.test_workspace, self.default_test_file.replace(".", ".scrambled."))

        # Evaluate
        self.assertFalse(scrambled_filepath.exists())

        # Setup
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

        self.setUpBatchWidget(measurement_files=None)

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(self.modify_aspect)

        self.widget.filter_view.setCurrentText("Natural sort")
        self.processEvents(0.1)

        # set test_workspace folder as output folder
        with mock.patch(
            "asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getExistingDirectory"
        ) as mo_getExistingDirectory:
            mo_getExistingDirectory.return_value = self.test_workspace
            QtTest.QTest.mouseClick(self.widget.modify_output_folder_btn, QtCore.Qt.MouseButton.LeftButton)

        self.assertEqual(self.widget.modify_output_folder.text().strip(), self.test_workspace)

        self.tested_btn = self.widget.apply_btn

    def generic_setup(self, name: str = "MDF", check: bool = True):  # Parquet is removed?
        """
        Add default test file to files list
        Check checkboxes for random channels.
        Set check state for output options widget checkboxes .

        Parameters
        ----------
        name: name of output options widget
        check: check state for output options widget checkboxes
        """
        self.widget.files_list.addItems([str(Path(self.resource, self.default_test_file))])
        self.select_random_channels()

        for q_widget in self.widget.output_format.children():
            if name in q_widget.objectName():
                break
        else:
            unittest.skip(f"{name} output option widget is not defined")

        self.toggle_checkboxes(widget=self.widget.output_options, check=check)

        self.processEvents(0.01)

    def select_random_channels(self):
        count = self.widget.filter_tree.topLevelItemCount()
        self.select_channels(set(np.random.randint(0, count, size=randint(int(count / 4), int(count / 2)))))
        self.processEvents(0.01)

    def get_mdf_from_git(self) -> Path:
        url = "https://github.com/danielhrisca/asammdf/files/4328945/OBD2-DBC-MDF4.zip"
        urllib.request.urlretrieve(url, "test.zip")
        ZipFile(r"test.zip").extractall(self.test_workspace)
        Path("test.zip").unlink()
        temp_dir = Path(self.test_workspace)

        # Get mdf file path
        return [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]

    @staticmethod
    def interpolated_timestamps(mdf, channels: set):
        all_ts = []
        for group_index, gp in enumerate(mdf.groups):
            if gp.channel_group.cycles_nr and (channels - {ch.name for ch in gp.channels}) != channels:
                all_ts.append(mdf.get_master(group_index))
        if all_ts:
            return np.unique(np.concatenate(all_ts))
        else:
            return np.array([], dtype="f8")

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
        name = "MDF"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(name, check=False)

        # Event
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate
        self.assertTrue(saved_file.exists())

        # get saved file as MDF
        with OpenMDF(saved_file) as mdf_file:
            # Evaluate saved file
            for channel in self.selected_channels:
                self.assertIn(channel, mdf_file.channels_db)

    @unittest.skipIf(os.getenv("NO_NET_ACCESS"), "Test requires Internet access")
    def test_output_format_ASC(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is ASC
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels data
        """
        # Get test file from git archive
        mdf_path = self.get_mdf_from_git()
        # Add extracted file to files list
        self.widget.files_list.addItems([str(mdf_path)])

        # Expected results
        asc_path = Path.with_suffix(Path(self.test_workspace, mdf_path.stem), ".asc")

        # Ensure output format
        self.widget.output_format.setCurrentText("ASC")

        # Event
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate
        self.assertTrue(asc_path.exists())

        with OpenMDF(mdf_path) as mdf_file:
            start_datetime = mdf_file.start_time.strftime("%a %b %d %I:%M:%S.%f %p %Y")

            for index, group in enumerate(mdf_file.groups):
                names = [ch.name for ch in group.channels if "time" not in ch.name.lower()]
                if not names:
                    continue

                for channel_name in names:
                    if "." not in channel_name:
                        names.remove(channel_name)
                        break

                if channel_name:
                    channel = mdf_file.get(channel_name, index)
                    if not channel.samples.size:
                        continue

                    with ASCReader(asc_path) as asc_file:
                        for row, msg in enumerate(asc_file):
                            if "Error" in channel_name:
                                self.assertTrue(msg.is_error_frame)
                                continue
                            elif "Remote" in channel_name:
                                self.assertTrue(msg.is_remote_frame)

                            # Evaluate timestamps
                            self.assertAlmostEqual(msg.timestamp, channel.timestamps[row], places=10)

                            # Evaluate other fields
                            for name in names:
                                if name.endswith(".DataBytes"):
                                    self.assertEqual(msg.data, channel[name][row][: msg.dlc].tobytes())
                                else:
                                    data = channel[name].astype("<u4").tolist()
                                    if name.endswith(".ID"):
                                        self.assertEqual(msg.arbitration_id, data[row])
                                    elif name.endswith(".DLC"):
                                        self.assertEqual(msg.dlc, data[row])
                                    elif name.endswith(".Dir"):
                                        self.assertEqual(msg.is_rx, not bool(data[row]))
                                    elif name.endswith(".ESI"):
                                        self.assertEqual(msg.error_state_indicator, bool(data[row]))
                                        self.assertTrue(msg.is_fd)
                                    elif name.endswith(".BRS"):
                                        self.assertEqual(msg.bitrate_switch, bool(data[row]))
                else:
                    continue

            self.assertEqual(asc_file.base, "hex")
            self.assertEqual(asc_file.timestamps_format, "absolute")
            self.assertFalse(asc_file.internal_events_logged)
            self.assertEqual(
                datetime.datetime.strptime(start_datetime, "%a %b %d %I:%M:%S.%f %p %Y"),
                datetime.datetime.strptime(asc_file.date, "%b %d %I:%M:%S.%f %p %Y"),
            )

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
        # Setup
        name = "CSV"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(name, check=False)

        # Expected results
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Event
        self.mouse_click_on_btn_with_progress(self.tested_btn)

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

                    np.testing.assert_almost_equal(channel.timestamps, pandas_tab.timestamps.values, decimal=9)
                    if np.issubdtype(channel.samples.dtype, np.number):  # problematic conversion
                        np.testing.assert_almost_equal(channel.samples, pandas_tab[channel.name].values, decimal=9)

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

        # Ensure output format
        name = "CSV"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(name)

        # Event
        self.mouse_click_on_btn_with_progress(self.tested_btn)

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
            for channel, unit in zip(mdf_channels, units, strict=False):
                if channel.display_names:  # DI.<channel name>
                    display_name, _ = zip(*channel.display_names.items(), strict=False)
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
        # Setup
        name = "MAT"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(name, check=False)

        # Expected results
        mat_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".mat"))
        groups = self.get_selected_groups(channels=self.selected_channels)

        # Mouse click on Apply button
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate
        self.assertTrue(mat_path.exists())

        mat_file = scipy.io.loadmat(str(mat_path))
        to_replace = {" ", ".", "[", "]"}

        with OpenMDF(Path(self.resource, self.default_test_file)) as mdf_file:
            for index, channels_list in enumerate(groups.values()):
                for name in channels_list:
                    signal = mdf_file.get(name)
                    for character in to_replace:
                        name = name.replace(character, "_")
                    name = f"DG{index}_{name}"

                    self.assertIn(name, mat_file)
                    if np.issubdtype(signal.samples.dtype, np.number):
                        self.assertTrue(np.array_equal(signal.samples, mat_file[name][0]))

    def test_output_format_MAT_1(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is MAT, no checkboxes checked
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels
        """
        # Setup
        name = "MAT"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(name)

        # Expected results
        mat_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".mat"))

        # Mouse click on Apply button
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate
        self.assertTrue(mat_path.exists())

        mat_file = scipy.io.loadmat(str(mat_path))
        to_replace = {" ", ".", "[", "]", ":"}

        with OpenMDF(Path(self.resource, self.default_test_file)) as mdf_file:
            # get common timestamps
            common_timestamps = self.interpolated_timestamps(mdf_file, set(self.selected_channels))

            for channel in self.selected_channels:
                mdf_signal = mdf_file.get(channel, raw=True).interp(
                    common_timestamps,
                    integer_interpolation_mode=self.widget.integer_interpolation,
                    float_interpolation_mode=self.widget.float_interpolation,
                )

                if mdf_signal.display_names:
                    display_name, _ = zip(*mdf_signal.display_names.items(), strict=False)
                    channel = display_name[0]
                for character in to_replace:
                    channel = channel.replace(character, "_")
                if channel + "_0" in mat_file.keys() and "RAT" in mdf_signal.name:
                    channel += "_0"

                self.assertIn(channel[:60], mat_file.keys())  # limit of maximum 60 ch for channel name
                np.testing.assert_almost_equal(
                    mdf_signal.samples, mat_file[channel[:60]][0], decimal=3, err_msg=mdf_signal.name
                )

            common_timestamps -= common_timestamps[0]  # start from 0
            np.testing.assert_almost_equal(mat_file["timestamps"][0], common_timestamps, decimal=3)

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
        # Setup
        name = "HDF5"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(name, check=False)

        # Expected results
        hdf5_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".hdf"))
        groups = self.get_selected_groups(channels=self.selected_channels)
        groups = list(groups.values())

        # Mouse click on Apply button
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate
        self.assertTrue(hdf5_path.exists())

        with OpenHDF5(hdf5_path) as hdf5_file, OpenMDF(self.measurement_file) as mdf_file:
            self.assertEqual(len(hdf5_file.items()) - 1, len(groups))  # 1th item is file path

            for hdf5_group in hdf5_file:
                hdf5_group = hdf5_file[hdf5_group]
                if hdf5_group.name.startswith("ChannelGroup"):
                    index = int(hdf5_group.name.split("_")[1])
                    mdf_group = groups[index]

                    for name in hdf5_group:
                        if name != "time":  # Evaluate channels
                            self.assertIn(name, mdf_group)
                            mdf_channel = mdf_file.select([name])[0]
                            hdf5_channel = hdf5_group.get(name)

                            if np.issubdtype(mdf_channel.samples.dtype, np.number):  # samples are numbers
                                np.testing.assert_almost_equal(mdf_channel.samples, hdf5_channel, decimal=3)
                            else:
                                # Evaluate samples shape
                                self.assertEqual(mdf_channel.samples.size, hdf5_channel.size)

                        else:  # evaluate timestamps
                            hdf5_channel = hdf5_group.get(
                                name
                            )  # for evaluation will be used latest mdf channel from group
                            np.testing.assert_almost_equal(mdf_channel.timestamps, hdf5_channel, decimal=3)

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
        # Expected results
        hdf5_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".hdf"))

        # Ensure output format
        name = "HDF5"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(name)

        # Mouse click on Apply button
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate
        self.assertTrue(hdf5_path.exists())

        with OpenHDF5(hdf5_path) as hdf5_file, OpenMDF(self.measurement_file) as mdf_file:
            self.assertEqual(len(hdf5_file.items()), 1)  # 1 item

            # Prepare results
            common_timestamps = self.interpolated_timestamps(mdf_file, set(self.selected_channels))
            hdf5_channels = hdf5_file[str(hdf5_path)]

            for channel in self.selected_channels:
                mdf_channel = mdf_file.get(channel, raw=True).interp(
                    common_timestamps,
                    integer_interpolation_mode=self.widget.integer_interpolation,
                    float_interpolation_mode=self.widget.float_interpolation,
                )

                if mdf_channel.display_names:
                    display_name, _ = zip(*mdf_channel.display_names.items(), strict=False)
                    channel = display_name[0]
                if channel + "_0" in hdf5_channels.keys() and "RAT" in mdf_channel.name:
                    channel += "_0"

                hdf5_channel = hdf5_channels.get(channel)

                # Evaluate
                self.assertIn(channel, hdf5_channels)
                np.testing.assert_almost_equal(mdf_channel.samples, hdf5_channel, decimal=3)

    @unittest.skipIf(os.getenv("NO_NET_ACCESS"), "Test requires Internet access")
    def test_output_format_Parquet_0(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is Parquet
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels with expected values
        """
        # Expected results
        parquet_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".parquet"))

        # Setup
        name = "Parquet"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(check=False)

        # Mouse click on Apply button
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate if parquet file exist
        self.assertTrue(parquet_path.exists())

        with OpenMDF(self.measurement_file) as mdf_file:
            pandas_tab = pd.read_parquet(parquet_path)  # open parquet file as pandas tab
            # Get common timestamps
            common_timestamps = self.interpolated_timestamps(mdf_file, set(self.selected_channels))

            # Evaluate
            np.testing.assert_almost_equal(common_timestamps, pandas_tab.index.values, decimal=9)

            for name in self.selected_channels:
                channel = mdf_file.get(name).interp(
                    common_timestamps,
                    integer_interpolation_mode=self.widget.integer_interpolation,
                    float_interpolation_mode=self.widget.float_interpolation,
                )
                if np.issubdtype(channel.samples.dtype, np.number):  # problematic conversion
                    np.testing.assert_almost_equal(channel.samples, pandas_tab[name].values, decimal=9)

    @unittest.skipIf(os.getenv("NO_NET_ACCESS"), "Test requires Internet access")
    def test_output_format_Parquet_1(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Ensure that output format is Parquet
            - Press PushButton Apply.

        Evaluate:
            - File was created.
            - Ensure that output file has only selected channels with expected values
        """
        # Expected results
        parquet_path = Path(self.test_workspace, self.default_test_file.replace(".mf4", ".parquet"))

        # Setup
        name = "Parquet"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup()

        # Mouse click on Apply button
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate if parquet file exist
        self.assertTrue(parquet_path.exists())

        # Evaluate
        with OpenMDF(self.measurement_file) as mdf_file:
            pandas_tab = pd.read_parquet(parquet_path)  # open parquet file as pandas tab
            # get common timestamp
            common_timestamps = self.interpolated_timestamps(mdf_file, set(self.selected_channels))

            # Evaluate
            np.testing.assert_almost_equal(common_timestamps - common_timestamps[0], pandas_tab.index.values, decimal=9)
            for name in self.selected_channels:
                channel = mdf_file.get(name, raw=True).interp(
                    common_timestamps,
                    integer_interpolation_mode=self.widget.integer_interpolation,
                    float_interpolation_mode=self.widget.float_interpolation,
                )
                if channel.display_names:
                    display_name, _ = zip(*channel.display_names.items(), strict=False)
                    name = display_name[0]
                if name + "_0" in pandas_tab.columns and "RAT" in channel.name:
                    name += "_0"

                if np.issubdtype(channel.samples.dtype, np.number):  # problematic conversion
                    np.testing.assert_almost_equal(channel.samples, pandas_tab[name].values, decimal=3)

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
        # Setup
        name = "MDF"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(check=False)

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
        for checkbox in self.widget.cut_group.findChildren(QtWidgets.QCheckBox):
            if not checkbox.isChecked():
                self.mouseClick_CheckboxButton(checkbox)
        self.processEvents(0.01)

        # Event
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate
        self.assertTrue(output_file.exists())
        with OpenMDF(output_file) as mdf_file:
            time_dif = expected_start_time - mdf_file.start_time
            self.assertEqual(time_dif.microseconds, 0)
            self.assertEqual(time_dif.seconds, 0)
            self.assertEqual(time_dif.days, 0)
            for channel in mdf_file.iter_channels():
                self.assertEqual(channel.timestamps.min(), 0)
                self.assertAlmostEqual(channel.timestamps.max(), stop_cut - start_cut, 3)

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
        # Setup
        name = "MDF"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(check=False)

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
        for checkbox in self.widget.cut_group.findChildren(QtWidgets.QCheckBox):
            if checkbox.isChecked():
                self.mouseClick_CheckboxButton(checkbox)
        self.processEvents(0.01)

        # Event
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate
        self.assertTrue(output_file.exists())
        with OpenMDF(output_file) as mdf_file:
            self.assertEqual(self.start_time, mdf_file.start_time)

            for channel in mdf_file.iter_channels():
                self.assertEqual(channel.timestamps.min(), start_cut)
                self.assertEqual(channel.timestamps.max(), stop_cut)

    # @unittest.skip("FIXME: test keeps failing in CI")
    def test_resample_by_step_0(self):
        """
        Events
            - Check resample group checkbox
            - Set step time
            - Press Apply bush button

        Evaluate
            - In test workspace folder was created file with same name as original measurement file
            - Start time for timestamps is not updated
            - Timestamps step for all channels is the same as step time value
        """
        # Expected values
        output_file = Path(self.test_workspace, self.default_test_file)
        step = randint(1, 100) / 10

        # Setup
        name = "MDF"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(check=False)

        QtTest.QTest.keyClick(self.widget.resample_group, QtCore.Qt.Key.Key_Space)
        self.widget.raster_type_step.setChecked(True)
        self.widget.raster.setValue(step)
        for checkbox in self.widget.resample_group.findChildren(QtWidgets.QCheckBox):
            if checkbox.isChecked():
                self.mouseClick_CheckboxButton(checkbox)
        self.processEvents(0.01)

        # Event
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate
        self.assertTrue(output_file.exists())
        with OpenMDF(output_file) as mdf_file:
            for channel in mdf_file.iter_channels():
                size = ceil((channel.timestamps.max() - channel.timestamps.min()) / step) + 1  # + min
                self.assertEqual(size, channel.timestamps.size)
                for i in range(size):
                    self.assertAlmostEqual(channel.timestamps[i], channel.timestamps.min() + step * i, places=12)

    # @unittest.skip("FIXME: test keeps failing in CI")
    def test_resample_by_step_1(self):
        """
        Events
            - Check resample group checkbox
            - Set step time
            - Check Time from zero checkbox
            - Press Apply bush button

        Evaluate
            - In test workspace folder was created file with same name as original measurement file
            - Start time for timestamps is 0
            - Timestamps step for all channels is the same as step time value
        """
        # Expected values
        output_file = Path(self.test_workspace, self.default_test_file)
        step = randint(1, 100) / 10

        # Setup
        name = "MDF"
        self.widget.output_format.setCurrentText(name)
        self.generic_setup(check=False)

        QtTest.QTest.keyClick(self.widget.resample_group, QtCore.Qt.Key.Key_Space)
        self.widget.raster_type_step.setChecked(True)
        self.widget.raster.setValue(step)
        for checkbox in self.widget.resample_group.findChildren(QtWidgets.QCheckBox):
            if not checkbox.isChecked():
                self.mouseClick_CheckboxButton(checkbox)
        self.processEvents(0.01)

        # Event
        self.mouse_click_on_btn_with_progress(self.tested_btn)

        # Evaluate
        self.assertTrue(output_file.exists())
        with OpenMDF(output_file) as mdf_file:
            for channel in mdf_file.iter_channels():
                size = ceil((channel.timestamps.max() - channel.timestamps.min()) / step) + 1  # + min
                self.assertEqual(size, channel.timestamps.size)
                for i in range(size):
                    self.assertAlmostEqual(channel.timestamps[i], step * i, places=12)

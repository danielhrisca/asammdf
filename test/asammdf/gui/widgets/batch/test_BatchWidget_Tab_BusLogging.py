#!/usr/bin/env python
import codecs
import os
from pathlib import Path
import sys
import unittest
from unittest import mock
import urllib
import urllib.request
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtTest

from test.asammdf.gui.test_base import DBC, OpenMDF
from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


@unittest.skipIf(os.getenv("NO_NET_ACCESS"), "Test requires Internet access")
class TestPushButtons(TestBatchWidget):
    def setUp(self):
        super().setUp()
        url = "https://github.com/danielhrisca/asammdf/files/4328945/OBD2-DBC-MDF4.zip"
        urllib.request.urlretrieve(url, "test.zip")
        ZipFile(r"test.zip").extractall(self.test_workspace)
        Path("test.zip").unlink()

        temp_dir = Path(self.test_workspace)

        # Get test files path
        self.mdf_path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]
        self.dbc_path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]

        self.setUpBatchWidget(measurement_files=[str(self.mdf_path)])
        # Go to Tab: "Bus logging": Index 3
        self.widget.aspects.setCurrentIndex(self.bus_aspect)

        # Ensure that CAN & LIN lists are empty
        self.assertEqual(self.widget.can_database_list.count(), 0)
        self.assertEqual(self.widget.lin_database_list.count(), 0)

    def load_database(self, path: Path | str | None = None, is_can=True):
        """
        Load database to CAN/LIN database list
        Parameters
        ----------
        path := path to .dbc file; if None: default .dbc path will be used
        is_can := CAN or LIN
        """
        if not path:
            path = self.dbc_path
        with mock.patch("asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getOpenFileNames") as mo_getOpenFileNames:
            mo_getOpenFileNames.return_value = [str(path)], None
            if is_can:
                QtTest.QTest.mouseClick(self.widget.load_can_database_btn, QtCore.Qt.MouseButton.LeftButton)
            else:
                QtTest.QTest.mouseClick(self.widget.load_lin_database_btn, QtCore.Qt.MouseButton.LeftButton)

    def get_prefix(self):
        """
        Get prefix for channels names.
        Example:
            - 'CAN1.OBD2.'
        Returns
        -------
        prefix: str
        """
        # Prepare expected results
        # Linux cannot decode using 'utf-8' standard codec
        with OpenMDF(self.mdf_path) as mdf_file, codecs.open(str(self.dbc_path), encoding="ISO-8859-1") as dbc_file:
            for key, value in mdf_file.bus_logging_map.items():
                if value:
                    prefix = key
                    for new_key, new_value in value.items():
                        if value:
                            prefix += str(new_key)
            self.assertTrue(prefix)  # there is a suffix for feature channels
            prefix += "." + DBC.BO(dbc_file.readlines()).name + "."
            return prefix

    def test_load_can_database_btn(self):
        """
        Events:
            - Press Load CAN database button.

        Evaluate:
            - There is one item in can database list
            - The item's text is .dbc path
        """
        # Event
        self.load_database()
        # Evaluate
        self.assertEqual(self.widget.can_database_list.count(), 1)
        self.assertEqual(
            self.widget.can_database_list.itemWidget(self.widget.can_database_list.item(0)).database.text(),
            str(self.dbc_path),
        )

    def test_load_lin_database_btn(self):
        """
        Events:
            - Press Load LIN database button.

        Evaluate:
            - There is one item in lin database list
            - The item's text is .dbc path
        """
        # Event
        self.load_database(is_can=False)
        # Evaluate
        self.assertEqual(self.widget.lin_database_list.count(), 1)
        self.assertEqual(
            self.widget.lin_database_list.itemWidget(self.widget.lin_database_list.item(0)).database.text(),
            str(self.dbc_path),
        )

    def test_extract_bus_btn(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Set prefix text
            - Press PushButton Extract Bus signals.

        Evaluate:
            - File was created.
            - New channels was created from .dbc and .mf4 files.
            - There is no channel from original measurement file, only from .dbc file
        """
        # Expected result
        output_file = Path.with_suffix(self.mdf_path, f".bus_logging{self.mdf_path.suffix}")

        # Precondition
        self.load_database()
        self.assertFalse(output_file.exists())
        self.assertEqual(self.widget.output_info_bus.toPlainText(), "")  # bus output info tab is clean

        # Set Prefix
        prefix = self.id().split(".")[-1]
        self.widget.prefix.setText(prefix)

        # Event
        self.mouse_click_on_btn_with_progress(self.widget.extract_bus_btn)

        self.assertTrue(output_file.exists())
        # Evaluate bus output info
        self.assertIn(str(self.dbc_path), self.widget.output_info_bus.toPlainText())
        self.assertIn(str(self.mdf_path), self.widget.output_info_bus.toPlainText())

        # because linux UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb0 in position 943: invalid start byte
        with (
            OpenMDF(output_file) as bus_file,
            OpenMDF(self.mdf_path) as mdf_file,
            codecs.open(str(self.dbc_path), encoding="ISO-8859-1") as dbc_file,
        ):
            dbc_lines = dbc_file.readlines()
            source = DBC.BO(dbc_lines)

            self.assertNotEqual(len(mdf_file.groups), len(bus_file.groups))
            for channel in bus_file.channels_db:
                self.assertNotIn(channel.replace(prefix, ""), mdf_file.channels_db)

            # Evaluate Source:
            for name, group_index in mdf_file.channels_db.items():
                group, index = group_index[0]
                if name.endswith("ID"):
                    self.assertEqual(mdf_file.get(name, group, index).samples[0], int(source.id))
                if name.endswith("DataLength"):
                    self.assertEqual(mdf_file.get(name, group, index).samples[0], int(source.data_length))
                if "time" in name.lower():
                    max_from_mdf = mdf_file.get(name, group, index).samples[-1]
                    min_from_mdf = mdf_file.get(name, group, index).samples[0]
                    size_from_mdf = mdf_file.get(name, group, index).samples.size

            timestamps_size = set()
            timestamps_min = set()
            timestamps_max = set()
            for name, group_index in bus_file.channels_db.items():
                group, index = group_index[0]
                if not name.endswith("time"):
                    self.assertIn(prefix, name)

                    channel = bus_file.get(name, group, index, raw=True)
                    signal = DBC.SG(name.replace(self.id().split(".")[-1], "").split(".")[-1], dbc_lines)

                    if not ("service" in name or "response" in name):
                        self.assertEqual(channel.bit_count, signal.bit_count, name)

                    self.assertEqual(channel.unit, signal.unit)

                    if hasattr(channel.conversion, "a"):
                        self.assertEqual(channel.conversion.a, signal.conversion_a, name)
                        self.assertEqual(channel.conversion.b, signal.conversion_b, name)

                    timestamps_size.add(channel.timestamps.size)
                    timestamps_min.add(channel.timestamps.min())
                    timestamps_max.add(channel.timestamps.max())

                for group in bus_file.groups:
                    self.assertEqual(self.id().split(".")[-1], group.channel_group.acq_name.split(":")[0])

                for group in mdf_file.groups:
                    self.assertNotIn(self.id().split(".")[-1], group.channel_group.acq_name)

            self.assertEqual(max(timestamps_size), size_from_mdf)
            self.assertEqual(min(timestamps_min), min_from_mdf)
            self.assertEqual(max(timestamps_max), max_from_mdf)

    def test_extract_bus_csv_btn_0(self):
        """
        When QThreads are running, event-loops needs to be processed.
        This test will use mf4 generated file for feature evaluation.
        If `test_extract_bus_btn` is failed: this test must be ignored.

        Events:
            - Set prefix text
            - Ensure that all checkboxes are unchecked
            - Press PushButton Export to csv.

        Evaluate:
            - CSV files was created.
            - New channels was created from .dbc and .mf4 files
            - CSV files is mf4 channels groups
            - Channels names and groups names contain prefix
            - Signals samples are added to CSV
        """
        # Prepare expected results
        prefix = self.get_prefix()
        to_replace = [" ", '"', ":"]

        # Precondition
        self.load_database()
        for file in Path(self.test_workspace).iterdir():
            self.assertNotEqual(file.suffix, ".csv")
        self.assertEqual(self.widget.output_info_bus.toPlainText(), "")
        self.toggle_checkboxes(widget=self.widget.extract_bus_tab, check=False)  # uncheck all checkboxes

        # Expected results
        compare_to_file_path = Path.with_suffix(self.mdf_path, ".bus_logging.mf4")

        # Set Prefix
        self.widget.prefix.setText(self.id().split(".")[-1])

        # Get new mdf file
        self.mouse_click_on_btn_with_progress(self.widget.extract_bus_btn)

        # Event
        self.mouse_click_on_btn_with_progress(self.widget.extract_bus_csv_btn)

        csv_tables = [
            (file, pd.read_csv(file)) for file in sorted(Path(self.test_workspace).iterdir()) if file.suffix == ".csv"
        ]
        with OpenMDF(compare_to_file_path) as mdf_file:
            for group, (path, table) in zip(mdf_file.groups, csv_tables, strict=False):
                comment = group.channel_group.comment
                for char in to_replace:
                    comment = comment.replace(char, "_")

                # Evaluate CSV name
                self.assertIn(self.id().split(".")[-1], str(path))
                self.assertTrue(str(path).endswith(comment + ".csv"))
                self.assertTrue(str(path.stem).startswith(compare_to_file_path.stem))

                # Evaluate channels
                for channel in group.channels:
                    if channel.name != "time":
                        name = prefix + channel.name

                        self.assertIn(name, table.columns)
                        ch = mdf_file.get(channel.name)

                        # Evaluate samples
                        self.assertEqual(ch.samples.size, table[name].size)
                        if np.issubdtype(ch.samples.dtype, np.number):
                            self.assertAlmostEqual(ch.samples.min(), table[name].min(), places=10)
                            self.assertAlmostEqual(ch.samples.max(), table[name].max(), places=10)
                        else:
                            self.assertEqual(
                                str(min(ch.samples)).replace(" ", ""),
                                str(table[name].min()).replace(" ", ""),
                            )
                            self.assertEqual(
                                str(max(ch.samples)).replace(" ", ""),
                                str(table[name].max()).replace(" ", ""),
                            )

                        # Evaluate timestamps
                        self.assertEqual(ch.timestamps.size, table.timestamps.size)
                        self.assertAlmostEqual(ch.timestamps.min(), table.timestamps.min(), places=10)
                        self.assertAlmostEqual(ch.timestamps.max(), table.timestamps.max(), places=10)

    def test_extract_bus_csv_btn_1(self):
        """
        When QThreads are running, event-loops needs to be processed.
        This test will use mf4 generated file for feature evaluation.
        If `test_extract_bus_btn` is failed: this test must be ignored.

        Events:
            - Set prefix text
            - Ensure that all checkboxes are checked
            - Press PushButton Export to csv.

        Evaluate:
            - CSV file is created
            - Timestamps start from 0 + measurement start time
            - CSV file contain new channels from .mf4 file
            - Channels names contain prefix
            - Signals samples are added to CSV
        """
        prefix = self.get_prefix()

        # Precondition
        self.load_database()
        for file in Path(self.test_workspace).iterdir():
            self.assertNotEqual(file.suffix, ".csv")
        self.assertEqual(self.widget.output_info_bus.toPlainText(), "")
        self.toggle_checkboxes(widget=self.widget.extract_bus_tab, check=True)  # uncheck all checkboxes

        # Expected results
        csv_path = Path.with_suffix(self.mdf_path, ".bus_logging.csv")
        compare_to_file_path = Path.with_suffix(self.mdf_path, ".bus_logging.mf4")

        # Set Prefix
        self.widget.prefix.setText(self.id().split(".")[-1])

        # Get new mdf file
        self.mouse_click_on_btn_with_progress(self.widget.extract_bus_btn)

        # Event
        self.mouse_click_on_btn_with_progress(self.widget.extract_bus_csv_btn)

        csv_table = pd.read_csv(csv_path, header=[0, 1], engine="python", encoding="ISO-8859-1")
        with OpenMDF(compare_to_file_path) as mdf_file:
            # Get channels timestamps max, min, difference between extremes
            min_ts = min(channel.timestamps.min() for channel in mdf_file.iter_channels())
            max_ts = max(channel.timestamps.max() for channel in mdf_file.iter_channels())
            seconds = int(max_ts - min_ts)
            microseconds = np.floor((max_ts - min_ts - seconds) * pow(10, 6))

            delay = np.datetime64(mdf_file.start_time)  # - np.timedelta64(3, "h")   # idk from why, local fail...
            csv_table.timestamps = pd.DatetimeIndex(csv_table.timestamps.values, yearfirst=True).values - delay

            # Evaluate timestamps min
            self.assertEqual(np.timedelta64(csv_table.timestamps.values.min(), "us").item().microseconds, 0)
            self.assertEqual(np.timedelta64(csv_table.timestamps.values.min(), "us").item().seconds, 0)
            # Evaluate timestamps max
            self.assertEqual(
                np.timedelta64(csv_table.timestamps.values.max(), "us").item().microseconds, int(microseconds)
            )
            self.assertEqual(np.timedelta64(csv_table.timestamps.values.max(), "us").item().seconds, seconds)

            for ch in mdf_file.iter_channels():
                name = prefix + ch.name
                self.assertIn(name, csv_table.columns)
                column = csv_table[name]
                if ch.unit and (sys.platform == "win32"):
                    self.assertEqual(ch.unit, column.columns[0])

                # Evaluate channel samples
                if np.issubdtype(ch.samples.dtype, np.number):
                    self.assertAlmostEqual(ch.samples.min(), column.values.min(), places=10)
                    self.assertAlmostEqual(ch.samples.max(), column.values.max(), places=10)
                else:
                    self.assertEqual(
                        str(min(ch.samples)).replace(" ", ""),
                        str(column.values.min()).replace(" ", ""),
                    )
                    self.assertEqual(
                        str(max(ch.samples)).replace(" ", ""),
                        str(column.values.max()).replace(" ", ""),
                    )

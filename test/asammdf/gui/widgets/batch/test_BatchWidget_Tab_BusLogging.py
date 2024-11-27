#!/usr/bin/env python
import os
import shutil
import urllib
import urllib.request
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

from PySide6 import QtCore, QtTest

from asammdf.blocks.utils import load_can_database
from test.asammdf.gui.test_base import OpenMDF, DBC
from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtons(TestBatchWidget):
    def setUp(self):
        super().setUp()
        # url = "https://github.com/danielhrisca/asammdf/files/4328945/OBD2-DBC-MDF4.zip"
        # urllib.request.urlretrieve(url, "test.zip")
        # ZipFile(r"test.zip").extractall(self.test_workspace)
        # Path("test.zip").unlink()
        #
        tmp_path = Path("D:\\GHP\\tmp\\asammdf\\BUS")

        for file in tmp_path.iterdir():
            shutil.copy(file, self.test_workspace)
        temp_dir = Path(self.test_workspace)

        # Get test files path
        self.mdf_path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]
        self.dbc_path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]
        npy_path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]

        self.setUpBatchWidget(measurement_files=[str(self.mdf_path)])
        # Go to Tab: "Bus logging": Index 3
        self.widget.aspects.setCurrentIndex(self.bus_aspect)

        self.widget.can_database_list.clear()
        self.widget.lin_database_list.clear()

        # Load DBC
        with mock.patch("asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getOpenFileNames") as mo_getOpenFileNames:
            mo_getOpenFileNames.return_value = [str(self.dbc_path)], None
            QtTest.QTest.mouseClick(self.widget.load_can_database_btn, QtCore.Qt.MouseButton.LeftButton)

        self.assertEqual(self.widget.can_database_list.count(), 1)
        self.can_database_matrix = load_can_database(self.dbc_path)

    def test_extract_bus_btn(self):
        """

        Returns
        -------

        """
        # Expected result
        output_file = Path.with_suffix(self.mdf_path, f".bus_logging{self.mdf_path.suffix}")

        # Precondition
        self.assertFalse(output_file.exists())
        self.assertEqual(self.widget.output_info_bus.toPlainText(), "")

        # Set Prefix
        self.widget.prefix.setText(self.id().split(".")[-1])

        # Event
        self.mouse_click_on_btn_with_progress(self.widget.extract_bus_btn)

        self.assertTrue(output_file.exists())
        # Evaluate bus output info
        self.assertIn(str(self.dbc_path), self.widget.output_info_bus.toPlainText())
        self.assertIn(str(self.mdf_path), self.widget.output_info_bus.toPlainText())

        with OpenMDF(output_file) as bus_file, OpenMDF(self.mdf_path) as mdf_file, open(self.dbc_path) as dbc_file:
            dbc_lines = dbc_file.readlines()
            source = DBC.BO(dbc_lines)

            self.assertNotEqual(len(mdf_file.groups), len(bus_file.groups))
            for channel in bus_file.channels_db:
                self.assertNotIn(channel.replace(self.id().split(".")[-1], ""), mdf_file.channels_db)

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
                if name != "time":
                    self.assertIn(self.id().split(".")[-1], name)

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

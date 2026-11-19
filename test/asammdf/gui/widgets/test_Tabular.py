from datetime import datetime, UTC
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from asammdf import MDF, Signal
from asammdf.gui.widgets.file import FileWidget
from asammdf.gui.widgets.tabular import Tabular


class TestTabularChannelAlignment(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_table(self, frame):
        table = Tabular(frame, start=datetime(2025, 1, 1, tzinfo=UTC))
        self.addCleanup(table.close)
        return table

    def test_incremental_channels_match_single_selection(self):
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "channels.mf4"
            with MDF(version="4.10") as measurement:
                measurement.append(
                    Signal(np.array([0x80000123, 0x80000124], dtype="u4"), np.array([2.0, 4.0]), name="Frame.ID")
                )
                measurement.append(Signal(np.array([10, 22, 23, 50]), np.array([1.0, 2.0, 2.0, 5.0]), name="added"))
                measurement.append(Signal(np.array([30]), np.array([3.0]), name="middle"))
                measurement.save(filename)
            file = FileWidget(filename, with_dots=False, subplots=True)
            try:
                entries = []
                for name in ("Frame.ID", "added", "middle"):
                    group, channel = file.mdf.whereis(name)[0]
                    entries.append(
                        {
                            "name": name,
                            "group_index": group,
                            "channel_index": channel,
                            "origin_uuid": file.uuid,
                            "type": "channel",
                            "ranges": [],
                        }
                    )
                settings = QtCore.QSettings()
                setting = "tabular/interpolation"
                old_value = settings.value(setting)
                if old_value is None:
                    self.addCleanup(settings.remove, setting)
                else:
                    self.addCleanup(settings.setValue, setting, old_value)
                for interpolate in (False, True):
                    with self.subTest(interpolate=interpolate):
                        settings.setValue(setting, interpolate)
                        file.add_window(("Tabular", entries))
                        expected = file.mdi_area.subWindowList()[-1].widget()
                        file.add_window(("Tabular", entries[:1]))
                        actual = file.mdi_area.subWindowList()[-1].widget()
                        for entry in entries[1:]:
                            file.add_new_channels([entry], actual)
                        pd.testing.assert_frame_equal(actual.tree.pgdf.df, expected.tree.pgdf.df)
                        if not interpolate:
                            channel = actual.tree.pgdf.df["Frame.ID"]
                            self.assertEqual(channel.dropna().tolist(), [0x123, 0x124])
                            self.assertEqual(channel.isna().sum(), 3)

                        restored, _ = file._load_tabular_window(
                            {"type": "Tabular", "title": "Saved", "configuration": actual.to_config()}
                        )
                        pd.testing.assert_frame_equal(restored.widget().tree.pgdf.df, expected.tree.pgdf.df)
                        storage = actual.tree.pgdf
                        storage.delete_column(storage.df.columns.get_loc("added"))
                        file.add_new_channels(entries[-1:], actual)
                        self.assertNotIn("added", storage.df.columns)
            finally:
                file.close()

    def test_sparse_payload_uses_its_length_and_keeps_missing_samples(self):
        data = "Frame.DataBytes"
        length = "Frame.DataLength"
        payloads = pd.DataFrame(
            {
                data: [np.array([16, 17, 18], dtype="u1"), None, np.array([32, 33, 34], dtype="u1")],
                length: [2, np.nan, 3],
            },
            index=[1.0, 2.0, 3.0],
        )
        table = self.make_table(pd.DataFrame({"other": [5, 6]}, index=[2.0, 3.0]))
        table.add_new_channels(payloads)
        whole = self.make_table(payloads.copy())
        for widget in (table, whole):
            column = widget.tree.pgdf.df[data]
            self.assertEqual(column.iloc[0], "10 11")
            self.assertTrue(pd.isna(column.iloc[1]))
            self.assertEqual(column.iloc[2], "20 21 22")

    def test_filter_and_date_mode_after_adding_earlier_and_later_samples(self):
        table = self.make_table(pd.DataFrame({"original": [2.0, 4.0]}, index=[2.0, 4.0]))
        table.add_filter()
        rule = table.filters.itemWidget(table.filters.item(0))
        rule.column.setCurrentText("original")
        rule.op.setCurrentText(">")
        rule.target.setText("0")
        rule.validate_target()
        table.apply_filters()
        table.time_as_date.setCheckState(QtCore.Qt.CheckState.Checked)

        table.add_new_channels(pd.DataFrame({"added": [10, 40, 50]}, index=[1.0, 4.0, 5.0]))
        storage = table.tree.pgdf
        expected = table.start + pd.to_timedelta([1, 2, 4, 5], unit="s")
        self.assertEqual(list(storage.df_unfiltered["timestamps"]), list(expected))
        self.assertEqual(list(storage.df["timestamps"]), list(expected[[1, 2]]))
        self.assertEqual(list(storage.df["original"]), [2, 4])

        table.time_as_date.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.assertEqual(list(storage.df_unfiltered["timestamps"]), [1, 2, 4, 5])
        self.assertEqual(list(storage.df["timestamps"]), [2, 4])
        table.set_timestamp(4)
        self.assertEqual(table.tree.dataView.currentIndex().row(), 1)

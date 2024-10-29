#!/usr/bin/env python
from pathlib import Path
from unittest import mock

from PySide6 import QtCore, QtTest

from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtons(TestBatchWidget):
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

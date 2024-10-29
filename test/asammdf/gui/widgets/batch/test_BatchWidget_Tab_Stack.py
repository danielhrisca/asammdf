#!/usr/bin/env python
from pathlib import Path
from unittest import mock

from PySide6 import QtCore, QtTest

from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtons(TestBatchWidget):
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

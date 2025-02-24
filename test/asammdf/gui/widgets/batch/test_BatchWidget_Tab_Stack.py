#!/usr/bin/env python
from pathlib import Path
from unittest import mock

import numpy as np

from test.asammdf.gui.test_base import OpenMDF
from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtons(TestBatchWidget):
    default_test_file = "ASAP2_Demo_V171.mf4"
    class_test_file = "test_batch.mf4"

    def setUp(self):
        """
        Events
            - Open 'BatchWidget' with 2 valid measurement.
            - Go to Tab: "Stack"
        """
        super().setUp()
        self.copy_mdf_files_to_workspace()

        # Setup
        self.measurement_file_1 = Path(self.test_workspace, self.default_test_file)
        self.measurement_file_2 = Path(self.test_workspace, self.class_test_file)
        self.saved_file = Path(self.test_workspace, self.id().split(".")[-1] + ".mf4")

        self.setUpBatchWidget(measurement_files=[str(self.measurement_file_1), str(self.measurement_file_2)])
        self.widget.aspects.setCurrentIndex(self.stack_aspect)

    def test_PushButton_Stack(self):
        """
        Events:
            - Press PushButton "Stack"
            - Save measurement file

        Evaluate:
            - New file is created
            - All channels samples and timestamps from both input files can be found in selected output file
        """
        # Event
        with mock.patch("asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = str(self.saved_file), ""
            self.mouse_click_on_btn_with_progress(self.widget.stack_btn)

        # Evaluate
        self.assertTrue(self.saved_file.exists())

        # Evaluate saved file
        with (
            OpenMDF(self.saved_file) as new_mdf_file,
            OpenMDF(self.measurement_file_1) as mdf_test_file_0,
            OpenMDF(self.measurement_file_2) as mdf_test_file_1,
        ):
            for channel in mdf_test_file_1.iter_channels():
                new_file_channel = new_mdf_file.get(channel.name)
                self.assertTrue(np.array_equal(channel.samples, new_file_channel.samples))

            for channel in mdf_test_file_0.iter_channels():
                new_file_channel = new_mdf_file.get(channel.name)
                self.assertTrue(np.array_equal(channel.samples, new_file_channel.samples))

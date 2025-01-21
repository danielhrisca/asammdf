#!/usr/bin/env python
from pathlib import Path
from unittest import mock

import numpy as np

from test.asammdf.gui.test_base import OpenMDF
from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtons(TestBatchWidget):
    test_file_0_name = "test_batch_cut_0.mf4"
    test_file_1_name = "test_batch_cut_1.mf4"
    output_file_name = "output.mf4"

    def setUp(self):
        super().setUp()

        self.test_file_0 = Path(self.test_workspace, self.test_file_0_name)
        self.test_file_1 = Path(self.test_workspace, self.test_file_1_name)

        self.copy_mdf_files_to_workspace()
        self.setUpBatchWidget(measurement_files=[str(self.test_file_0), str(self.test_file_1)])

        # Go to Tab: Concatenate
        self.widget.aspects.setCurrentIndex(self.concatenate_aspect)
        self.processEvents(0.1)

    def test_PushButton_Concatenate(self):
        """
        Events:
            - Open 'BatchWidget' with created measurement.
            - Go to Tab: "Concatenate": Index 0
            - Press PushButton "Concatenate"

        Evaluate:
            - New file is created
            - All channels samples and timestamps from test files are concatenated intro selected output file
        """
        # Get evaluation data
        output_file = Path(self.test_workspace, self.output_file_name)

        # Event
        with mock.patch("asammdf.gui.widgets.batch.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName:
            mo_getSaveFileName.return_value = output_file, ""
            self.mouse_click_on_btn_with_progress(self.widget.concatenate_btn)

        # Evaluate that file exist
        self.assertTrue(output_file.exists())

        # Evaluate
        with (
            OpenMDF(output_file) as new_mdf_file,
            OpenMDF(self.test_file_0) as original_mdf_0,
            OpenMDF(self.test_file_1) as original_mdf_1,
        ):
            # Evaluate saved file
            for ch in new_mdf_file.iter_channels():
                if ch.name in original_mdf_0.channels_db:  # avoid `__samples_origin` channel
                    ch_0 = original_mdf_0.get(ch.name)
                    ch_1 = original_mdf_1.get(ch.name)

                    self.assertTrue(np.array_equal(np.concatenate([ch_0.samples, ch_1.samples]), ch.samples))
                    self.assertTrue(np.array_equal(np.concatenate([ch_0.timestamps, ch_1.timestamps]), ch.timestamps))

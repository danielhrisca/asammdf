#!/usr/bin/env python
from pathlib import Path

from PySide6 import QtCore, QtTest

from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtons(TestBatchWidget):
    test_file = "test_batch.mf4"

    def test_PushButtons_Sort(self):
        """
        Events:
            - Open 'BatchWidget' with 2 valid measurements.
            - Press PushButton "Sort by start time"
            - Press PushButton "Sort alphabetically"

        Evaluate:
            - Files are sorted by start time
            - Files are sorted alphabetically
        """
        # Setup
        file_0 = Path(self.test_workspace, self.default_test_file)
        file_1 = Path(self.test_workspace, self.test_file)

        alpha_sort = [str(file_0), str(file_1)]
        time_sort = [str(file_1), str(file_0)]

        self.copy_mdf_files_to_workspace()
        self.setUpBatchWidget(measurement_files=alpha_sort)

        # Event
        with self.subTest("test sort by start time btn"):
            # Press `Sort by start time` button
            QtTest.QTest.mouseClick(self.widget.sort_by_start_time_btn, QtCore.Qt.MouseButton.LeftButton)

            # Evaluate
            self.assertEqual([self.widget.files_list.item(row).text() for row in range(2)], time_sort)

        with self.subTest("test sort alphabetically btn"):
            # Press `Sort by start time` button
            QtTest.QTest.mouseClick(self.widget.sort_alphabetically_btn, QtCore.Qt.MouseButton.LeftButton)

            # Evaluate
            self.assertEqual([self.widget.files_list.item(row).text() for row in range(2)], alpha_sort)

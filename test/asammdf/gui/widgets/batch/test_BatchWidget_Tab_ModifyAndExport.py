#!/usr/bin/env python
from pathlib import Path

from PySide6 import QtCore, QtTest, QtWidgets

from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtons(TestBatchWidget):

    def test_PushButton_ScrambleTexts(self):
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
        test_file = Path(self.test_workspace, self.default_test_file)
        scrambled_filepath = Path(self.test_workspace, self.default_test_file.replace(".", ".scrambled."))

        self.setUpBatchWidget(measurement_files=None)

        channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.widget.filter_tree)
        while item := iterator.value():
            iterator += 1
            channels.append(item.name)

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)

        # Events
        QtTest.QTest.mouseClick(self.widget.scramble_btn, QtCore.Qt.MouseButton.LeftButton)

        # Wait for Thread to finish
        self.processEvents(1)

        # Evaluate
        self.assertTrue(scrambled_filepath.exists())

        # get saved file as MDF
        with self.OpenMDF(scrambled_filepath) as mdf_file:
            # Evaluate file
            for name in channels:
                self.assertNotIn(name, mdf_file.channels_db.keys())

    def test_ExportMDF(self):
        """
        When QThreads are running, event-loops needs to be processed.
        Events:
            - Open 'BatchWidget' with valid measurement.
            - Go to Tab: "Modify & Export": Index 1
            - Set "channel_view" to "Natural sort"
            - Select some channels
            - Ensure that output format is MDF
            - Press PushButton Apply.

        Evaluate:
            - Evaluate that file was created.
            - Ensure that output file has only selected channels
        """
        # Setup
        measurement_file = str(Path(self.test_workspace, self.default_test_file))
        saved_file = Path(self.test_workspace, self.default_test_file.replace(".", ".modified."))

        self.setUpBatchWidget(measurement_files=[measurement_file])

        # Go to Tab: "Modify & Export": Index 1
        self.widget.aspects.setCurrentIndex(1)

        self.widget.filter_view.setCurrentText("Natural sort")
        self.processEvents(0.1)

        selected_channels = self.select_channels(5, 15)

        # Ensure output format
        self.widget.output_format.setCurrentText("MDF")

        # Mouse click on Apply button
        QtTest.QTest.mouseClick(self.widget.apply_btn, QtCore.Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(2)

        # Evaluate
        self.assertTrue(saved_file.exists())

        # get saved file as MDF
        with self.OpenMDF(saved_file) as mdf_file:
            # Evaluate saved file
            for channel in selected_channels:
                self.assertIn(channel, mdf_file.channels_db.keys())

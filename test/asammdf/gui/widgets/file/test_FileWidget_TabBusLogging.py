#!/usr/bin/env python
from pathlib import Path
from unittest import mock

from PySide6 import QtCore, QtTest

from test.asammdf.gui.widgets.test_BaseFileWidget import TestFileWidget
from test.test_CAN_bus_logging import can_logging


class TestBusSummary(TestFileWidget):
    def test_csv_summary(self):
        """
        Events:
            - Extract the bus signals of a log with the frames 0x100 and 0x300 to CSV, the DBC only has 0x100.

        Evaluate:
            - The summary lists 0x100 as matched and 0x300 as unknown standard CAN ID.
        """
        dbc_path = Path(self.test_workspace, "known.dbc")
        dbc_path.write_text(
            'VERSION ""\n\nBO_ 256 Known: 8 Vector__XXX\n SG_ Value : 0|8@1+ (1,0) [0|255] "" Vector__XXX\n'
        )
        mdf_path = Path(self.test_workspace, "log.mf4")
        with can_logging([0x100, 0x300], [0, 0], [0.0, 0.0]) as mdf:
            mdf.save(mdf_path)

        self.setUpFileWidget(measurement_file=mdf_path, default=True)
        with (
            mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getOpenFileNames") as mo_getOpenFileNames,
            mock.patch("asammdf.gui.widgets.file.QtWidgets.QFileDialog.getSaveFileName") as mo_getSaveFileName,
        ):
            mo_getOpenFileNames.return_value = [str(dbc_path)], None
            mo_getSaveFileName.return_value = str(Path(self.test_workspace, "bus.csv")), None
            QtTest.QTest.mouseClick(self.widget.load_can_database_btn, QtCore.Qt.MouseButton.LeftButton)
            QtTest.QTest.mouseClick(self.widget.extract_bus_csv_btn, QtCore.Qt.MouseButton.LeftButton)
            while self.widget._progress:
                self.processEvents(0.01)

        lines = self.widget.output_info_bus.toPlainText().splitlines()
        self.assertIn(f"- 0x100 extended=False --> Known in <{dbc_path}>", lines)
        self.assertIn("- 0x300 extended=False", lines)

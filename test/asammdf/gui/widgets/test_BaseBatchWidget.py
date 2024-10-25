from collections.abc import Sequence
import os
import pathlib
from unittest import mock

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from asammdf import mdf
from asammdf.gui.widgets.batch import BatchWidget
from test.asammdf.gui.test_base import TestBase


class TestBatchWidget(TestBase):
    testResult = None
    concatenate_aspect = 0
    modify_aspect = 1
    stack_aspect = 3
    bus_aspect = 4

    def setUp(self):
        super().setUp()
        self.widget = None
        self.plot = None

        patcher = mock.patch("asammdf.gui.widgets.file.ErrorDialog")
        self.mc_widget_ed = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        path_ = os.path.join(self.screenshots, self.__module__)
        if not os.path.exists(path_):
            os.makedirs(path_)

        self.widget.grab().save(os.path.join(path_, f"{self.id().split('.')[-1]}.png"))

        if self.widget:
            self.widget.close()
            self.widget.destroy()
            self.widget.deleteLater()
        self.mc_ErrorDialog.reset_mock()
        super().tearDown()

    def setUpBatchWidget(self, *args, measurement_files: Sequence[str], default):
        """
        Created because a lot of testcases,
        we do not need other parameters for BatchWidget initialization.
        """
        if default:
            self.widget = BatchWidget(
                *args,
            )
        else:
            self.widget = BatchWidget(*args)
        self.processEvents()
        for file in measurement_files:
            self.assertTrue(pathlib.Path(file).exists())
        self.widget.files_list.addItems(measurement_files)

        # Evaluate that all files was opened
        self.assertEqual(self.widget.files_list.count(), len(measurement_files))

        self.widget.showNormal()

    def cut_file(self, start_time, end_time):
        # Setup for Cut
        self.widget.cut_group.setChecked(True)
        self.widget.cut_start.setValue(start_time)
        self.widget.cut_stop.setValue(end_time)
        # Ensure output format
        self.widget.output_format.setCurrentText("MDF")
        # Ensure output folder
        self.widget.modify_output_folder.setText(str(self.test_workspace))

        # Mouse click on Apply button
        QTest.mouseClick(self.widget.apply_btn, Qt.MouseButton.LeftButton)
        # Wait for thread to finish
        self.processEvents(2)

    class OpenMDF:
        def __init__(self, file_path):
            self.mdf = None
            self._file_path = file_path
            self._process_bus_logging = ("process_bus_logging", True)

        def __enter__(self):
            self.mdf = mdf.MDF(self._file_path, process_bus_logging=self._process_bus_logging)
            return self.mdf

        def __exit__(self, exc_type, exc_val, exc_tb):
            for exc in (exc_type, exc_val, exc_tb):
                if exc is not None:
                    raise exc
            self.mdf.close()

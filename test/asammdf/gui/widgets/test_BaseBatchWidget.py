import os
from unittest import mock

from natsort import natsorted

from asammdf.gui.widgets.batch import BatchWidget
from test.asammdf.gui.test_base import TestBase


class TestBatchWidget(TestBase):
    testResult = None

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

    def setUpBatchWidget(self, *args, measurement_files: list, default):
        """
        Created because for a lot of testcases,
        we do not need other parameters for FileWidget initialization.
        """
        if default:
            self.widget = BatchWidget(
                *args,
            )
        else:
            self.widget = BatchWidget(*args)
        self.processEvents()
        self.widget.files_list.addItems(natsorted(measurement_files))
        self.widget.showNormal()
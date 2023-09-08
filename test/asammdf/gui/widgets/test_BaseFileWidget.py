from test.asammdf.gui.test_base import TestBase
from unittest import mock

from asammdf.gui.widgets.file import FileWidget


class TestFileWidget(TestBase):
    def setUp(self):
        super().setUp()
        self.widget = None

        patcher = mock.patch("asammdf.gui.widgets.file.ErrorDialog")
        self.mc_widget_ed = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        if self.widget:
            self.widget.close()
            self.widget.destroy()
            self.widget.deleteLater()
        self.mc_ErrorDialog.reset_mock()
        super().tearDown()

    def setUpFileWidget(self, *args, measurement_file, default):
        """
        Created because for a lot of testcases,
        we do not need other parameters for FileWidget initialization.
        """
        if default:
            self.widget = FileWidget(
                measurement_file,
                True,  # with_dots
                True,  # subplots
                True,  # subplots_link
                False,  # ignore_value2text_conversions
                False,  # display_cg_name
                "line",  # line_interconnect
                1,  # password
                None,  # hide_missing_channels
                None,  # hide_disabled_channels
            )
        else:
            self.widget = FileWidget(measurement_file, *args)
        self.widget.showNormal()

    def get_subwindows(self):
        widget_types = sorted(
            map(
                lambda w: w.widget().__class__.__name__,
                self.widget.mdi_area.subWindowList(),
            )
        )
        return widget_types

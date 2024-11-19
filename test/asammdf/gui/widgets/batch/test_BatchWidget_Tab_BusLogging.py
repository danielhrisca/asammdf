#!/usr/bin/env python
import urllib
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtons(TestBatchWidget):
    def setUp(self):
        url = "https://github.com/danielhrisca/asammdf/files/4328945/OBD2-DBC-MDF4.zip"
        urllib.request.urlretrieve(url, "test.zip")
        ZipFile(r"test.zip").extractall(self.test_workspace)
        Path("test.zip").unlink()
        temp_dir = Path(self.test_workspace)

        # Get test files path
        mdf_path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".mf4"][0]
        dbc_path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".dbc"][0]
        npy_path = [input_file for input_file in temp_dir.iterdir() if input_file.suffix == ".npy"]

        print(mdf_path, dbc_path, npy_path)

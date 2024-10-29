#!/usr/bin/env python

from test.asammdf.gui.widgets.test_BaseBatchWidget import TestBatchWidget

# Note: If it's possible and make sense, use self.subTests
# to avoid initializing widgets multiple times and consume time.


class TestPushButtons(TestBatchWidget):
    default_test_file = "ASAP2_Demo_V171.mf4"
    class_test_file = "test_batch.mf4"

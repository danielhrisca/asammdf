#!/usr/bin/env python
"""
Base Module for Testing GUI
 - responsible to set up Qt in order to run on multiple platforms
 - responsible to set up Application
class TestBase
    - responsible to set up and tearDown test workspace
    - responsible to create easy access to 'resource' directory
    - responsible to set up ErrorDialog for easy evaluation of raised exceptions
    - provide method `manual_use` created to help test development process
class DragAndDrop
    - responsible to perform Drag and Drop operations
     from source widget - specific point, to destination widget - specific point
"""
import os
import pathlib
import shutil
import sys
import time
import unittest
from unittest import mock

import pyqtgraph
from PySide6 import QtCore, QtTest

from asammdf.gui.utils import excepthook

if sys.platform == "win32":
    os.environ["QT_QPA_PLATFORM"] = "windows"
elif sys.platform == "linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
elif sys.platform == "darwin":
    os.environ["QT_QPA_PLATFORM"] = "cocoa"
else:
    os.environ["QT_QPA_PLATFORM"] = "windows"

app = pyqtgraph.mkQApp()
app.setOrganizationName("py-asammdf")
app.setOrganizationDomain("py-asammdf")
app.setApplicationName("py-asammdf")


@unittest.skipIf(
    sys.platform == "darwin", "Test Development on MacOS was not done yet."
)
class TestBase(unittest.TestCase):
    """
    - setUp and tearDown test workspace
    - provide method to execute widget
    - setUp ErrorDialog for evaluation raised exceptions
    """

    longMessage = False

    resource = os.path.normpath(os.path.join(os.path.dirname(__file__), "resources"))
    test_workspace = os.path.join(
        os.path.join(os.path.dirname(__file__), "test_workspace")
    )
    patchers = []
    # MockClass ErrorDialog
    mc_ErrorDialog = None

    def shortDescription(self):
        return self._testMethodDoc

    @staticmethod
    def manual_use(widget):
        """
        Execute Widget for debug/development purpose.
        """
        widget.showNormal()
        app.exec()

    @staticmethod
    def processEvents(timeout=0.001):
        QtCore.QCoreApplication.processEvents()
        if timeout:
            time.sleep(timeout)
            QtCore.QCoreApplication.processEvents()

    def setUp(self) -> None:
        if os.path.exists(self.test_workspace):
            shutil.rmtree(self.test_workspace)
        os.makedirs(self.test_workspace)
        self.mc_ErrorDialog.reset_mock()
        self.processEvents()

    @classmethod
    def setUpClass(cls):
        sys.excepthook = excepthook
        for patcher in (mock.patch("asammdf.gui.utils.ErrorDialog"),):
            _ = patcher.start()
            cls.patchers.append(patcher)
        cls.mc_ErrorDialog = _

    @classmethod
    def tearDownClass(cls):
        for patcher in cls.patchers:
            patcher.stop()

    def tearDown(self):
        self.processEvents()
        if self.test_workspace and pathlib.Path(self.test_workspace).exists():
            shutil.rmtree(self.test_workspace)

    def mouseClick_QTreeWidgetItem(self, qitem):
        treeWidget = qitem.treeWidget()
        QtTest.QTest.mouseClick(
            treeWidget.viewport(),
            QtCore.Qt.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            treeWidget.visualItemRect(qitem).center(),
        )
        self.processEvents(0.5)

    def mouseDClick_QTreeWidgetItem(self, qitem):
        treeWidget = qitem.treeWidget()
        QtTest.QTest.mouseDClick(
            treeWidget.viewport(),
            QtCore.Qt.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            treeWidget.visualItemRect(qitem).center(),
        )
        self.processEvents(0.5)


class DragAndDrop:
    _previous_position = None

    class MoveThread(QtCore.QThread):
        def __init__(self, widget, position=None, step=None):
            super().__init__()
            self.widget = widget
            self.position = position
            self.step = step

        def run(self):
            time.sleep(0.1)
            if not self.step:
                QtTest.QTest.mouseMove(self.widget, self.position)
            else:
                for step in range(self.step):
                    QtTest.QTest.mouseMove(
                        self.widget, self.position + QtCore.QPoint(step, step)
                    )
                    QtTest.QTest.qWait(2)
            QtTest.QTest.qWait(10)
            # Release
            QtTest.QTest.mouseRelease(
                self.widget,
                QtCore.Qt.LeftButton,
                QtCore.Qt.KeyboardModifiers(),
                self.position,
            )
            QtTest.QTest.qWait(10)

    def __init__(self, source_widget, destination_widget, source_pos, destination_pos):
        # Ensure that previous drop was not in the same place because mouse needs to be moved.
        if self._previous_position and self._previous_position == destination_pos:
            move_thread = DragAndDrop.MoveThread(
                widget=source_widget, position=QtCore.QPoint(101, 101)
            )
            move_thread.start()
            move_thread.wait()
            move_thread.quit()
        DragAndDrop._previous_position = destination_pos

        QtCore.QCoreApplication.processEvents()
        if hasattr(source_widget, "viewport"):
            source_viewport = source_widget.viewport()
        else:
            source_viewport = source_widget
        # Press on Source Widget
        QtTest.QTest.mousePress(
            source_viewport,
            QtCore.Qt.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            source_pos,
        )
        # Drag few pixels in order to detect startDrag event
        # drag_thread = DragAndDrop.MoveThread(widget=source_widget, position=source_pos, step=50)
        # drag_thread.start()
        # Move to Destination Widget
        if hasattr(destination_widget, "viewport") and sys.platform == "win32":
            destination_viewport = destination_widget.viewport()
        else:
            destination_viewport = destination_widget
        move_thread = DragAndDrop.MoveThread(
            widget=destination_viewport, position=destination_pos
        )
        move_thread.start()

        source_widget.startDrag(QtCore.Qt.MoveAction)
        QtTest.QTest.qWait(50)

        # drag_thread.wait()
        move_thread.wait()
        move_thread.quit()

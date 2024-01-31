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
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

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


@unittest.skipIf(sys.platform == "darwin", "Test Development on MacOS was not done yet.")
class TestBase(unittest.TestCase):
    """
    - setUp and tearDown test workspace
    - provide method to execute widget
    - setUp ErrorDialog for evaluation raised exceptions
    """

    longMessage = False

    resource = os.path.normpath(os.path.join(os.path.dirname(__file__), "resources"))
    test_workspace = os.path.join(os.path.join(os.path.dirname(__file__), "test_workspace"))
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
        QtCore.QCoreApplication.sendPostedEvents()
        if timeout:
            time.sleep(timeout)
            QtCore.QCoreApplication.processEvents()
            QtCore.QCoreApplication.sendPostedEvents()

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

    def mouseClick_RadioButton(self, qitem):
        QtTest.QTest.mouseClick(
            qitem,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            QtCore.QPoint(2, qitem.height() / 2),
        )
        self.processEvents()

    def mouseClick_CheckboxButton(self, qitem):
        # Same function
        self.mouseClick_RadioButton(qitem)

    def mouseClick_WidgetItem(self, qitem):
        if isinstance(qitem, QtWidgets.QTreeWidgetItem):
            widget = qitem.treeWidget()
        elif isinstance(qitem, QtWidgets.QListWidgetItem):
            widget = qitem.listWidget()
        else:
            raise NotImplementedError
        QtTest.QTest.mouseClick(
            widget.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            widget.visualItemRect(qitem).center(),
        )
        self.processEvents(0.5)

    def mouseDClick_WidgetItem(self, qitem):
        if isinstance(qitem, QtWidgets.QTreeWidgetItem):
            widget = qitem.treeWidget()
        elif isinstance(qitem, QtWidgets.QListWidgetItem):
            widget = qitem.listWidget()
        else:
            raise NotImplementedError
        QtTest.QTest.mouseDClick(
            widget.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifiers(),
            widget.visualItemRect(qitem).center(),
        )
        self.processEvents(0.5)

    def avoid_blinking_issue(self, w):
        self.processEvents(0.01)
        # To avoid blinking issue, click on a center of widget
        QtTest.QTest.mouseClick(w, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifiers(), w.rect().center())


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
                    QtTest.QTest.mouseMove(self.widget, self.position + QtCore.QPoint(step, step))
                    QtTest.QTest.qWait(2)
            QtTest.QTest.qWait(10)
            # Release
            QtTest.QTest.mouseRelease(
                self.widget,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifiers(),
                self.position,
            )
            QtTest.QTest.qWait(10)

    def __init__(self, src_widget, dst_widget, src_pos, dst_pos):
        # Ensure that the previous drop was not in the same place because the mouse needs to be moved.
        if self._previous_position and self._previous_position == dst_pos:
            move_thread = DragAndDrop.MoveThread(widget=src_widget, position=QtCore.QPoint(101, 101))
            move_thread.start()
            move_thread.wait()
            move_thread.quit()
        DragAndDrop._previous_position = dst_pos

        QtCore.QCoreApplication.processEvents()
        if hasattr(src_widget, "viewport"):
            source_viewport = src_widget.viewport()
        else:
            source_viewport = src_widget
        # Move to Destination Widget
        if hasattr(dst_widget, "viewport"):
            destination_viewport = dst_widget.viewport()
        else:
            destination_viewport = dst_widget

        # Press on Source Widget
        QtTest.QTest.mousePress(
            source_viewport, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifiers(), src_pos
        )

        move_thread = DragAndDrop.MoveThread(widget=destination_viewport, position=dst_pos)
        move_thread.start()

        src_widget.startDrag(QtCore.Qt.DropAction.MoveAction)
        QtTest.QTest.qWait(50)

        move_thread.wait()
        move_thread.quit()
        QtCore.QCoreApplication.processEvents()


class Pixmap:
    COLOR_BACKGROUND = "#000000"
    COLOR_RANGE = "#000032"
    COLOR_CURSOR = "#e69138"

    @staticmethod
    def is_black(pixmap):
        """
        Excepting cursor
        """
        cursor_x = None
        cursor_y = None
        cursor_color = None
        image = pixmap.toImage()

        for y in range(image.height()):
            for x in range(image.width()):
                color = QtGui.QColor(image.pixel(x, y))
                if color.name() != Pixmap.COLOR_BACKGROUND:
                    if not cursor_x and not cursor_y and not cursor_color:
                        cursor_x = x
                        cursor_y = y + 1
                        cursor_color = color
                        continue
                    elif cursor_x == x and cursor_y == y and cursor_color == color:
                        cursor_y += 1
                        continue
                    else:
                        return False
        return True

    @staticmethod
    def is_colored(pixmap, color_name, x, y, width=None, height=None):
        image = pixmap.toImage()

        offset = 1
        y = y + offset

        if not width:
            width = image.width()
        if not height:
            height = image.height()

        for _y in range(offset, image.height()):
            for _x in range(image.width()):
                color = QtGui.QColor(image.pixel(_x, _y))
                if _x < x or _y < y:
                    continue
                # De unde 2?
                elif (_x > width - x) or (_y > height - y - 2):
                    break
                if color.name() != color_name:
                    print(x, y, width, height)
                    print(_x, _y, color.name())
                    return False
        return True

    @staticmethod
    def has_color(pixmap, color_name):
        """
        Return True if Pixmap has selected color
        """
        image = pixmap.toImage()
        if not isinstance(color_name, str):
            if hasattr(color_name, "color"):
                color_name = color_name.color.name()
            elif hasattr(color_name, "name"):
                color_name = color_name.name()
            else:
                raise SyntaxError(f"Object {color_name} doesn't have the attribute <<color>> or <<name()>>")
        for y in range(image.height()):
            for x in range(image.width()):
                color = QtGui.QColor(image.pixel(x, y))
                if color.name() == color_name:
                    return True
        return False

    @staticmethod
    def color_names(pixmap):
        color_names = set()

        image = pixmap.toImage()
        for y in range(image.height()):
            for x in range(image.width()):
                color = QtGui.QColor(image.pixel(x, y))
                color_names.add(color.name())
        return color_names

    @staticmethod
    def color_names_exclude_defaults(pixmap):
        color_names = set()
        defaults = (Pixmap.COLOR_BACKGROUND, Pixmap.COLOR_CURSOR, Pixmap.COLOR_RANGE)
        image = pixmap.toImage()
        for y in range(image.height()):
            for x in range(image.width()):
                color = QtGui.QColor(image.pixel(x, y))
                if color not in defaults:
                    color_names.add(color.name())
        return color_names

    @staticmethod
    def color_map(pixmap):
        """
        return dict, where:
            > keys are line of pixmap
            > values is a list of color names ordered by columns of pixmap
        """
        color_dict = {}
        line = []
        image = pixmap.toImage()
        for y in range(image.height()):
            for x in range(image.width()):
                line.append(QtGui.QColor(image.pixel(x, y)).name())
            color_dict[y] = line
            line = []
        return color_dict

    @staticmethod
    def cursors_x(pixmap):
        image = pixmap.toImage()

        cursors = []

        for x in range(image.width()):
            count = 0
            for y in range(image.height()):
                color = QtGui.QColor(image.pixel(x, y))
                # Skip Black
                if color.name() == Pixmap.COLOR_BACKGROUND:
                    continue
                if color.name() == Pixmap.COLOR_CURSOR:
                    count += 1
            if count >= image.height() / 2 - 1:  # For Y shortcut tests, one cursor is a discontinuous line
                cursors.append(x)

        return cursors

    @staticmethod
    def search_signal_extremes_by_ax(pixmap, signal_color, ax: str):
        """
        Return column where signal start and end
        If ax = Y: Return a list with extremes of signal by 0Y axes
        If ax = X: Return a list with extremes of signal by 0X axes
        """
        if not isinstance(signal_color, str):
            if hasattr(signal_color, "color"):
                signal_color = signal_color.color.name()
            elif hasattr(signal_color, "name"):
                signal_color = signal_color.name()
            else:
                raise SyntaxError(f"Object {signal_color} doesn't have the attribute <<color>> or <<name()>>")
        from_to = []
        image = pixmap.toImage()
        if ax == "x" or ax == "X":
            for x in range(image.width()):
                for y in range(image.height()):
                    if QtGui.QColor(image.pixel(x, y)).name() == signal_color:
                        from_to.append(x)
                        break
                if from_to:
                    break
            if not from_to:
                return
            for x in range(image.width(), from_to[0], -1):
                for y in range(image.height()):
                    if QtGui.QColor(image.pixel(x, y)).name() == signal_color:
                        from_to.append(x)
                        break
                if len(from_to) == 2:
                    break
            return from_to

        elif ax == "y" or ax == "Y":
            for y in range(image.height()):
                for x in range(image.width()):
                    if QtGui.QColor(image.pixel(x, y)).name() == signal_color:
                        from_to.append(y)
                        break
                if from_to:
                    break
            if not from_to:
                return

            for y in range(image.height(), from_to[0], -1):
                for x in range(image.width()):
                    if QtGui.QColor(image.pixel(x, y)).name() == signal_color:
                        from_to.append(y)
                        break
                if len(from_to) == 2:
                    break
            return from_to

    @staticmethod
    def search_y_of_signal_in_column(pixmap_column, signal_color):
        """
        Return the first pixel line number where the signal color was found.
        """
        image = pixmap_column.toImage()
        if image.width() > 1:
            raise TypeError(f"<<{image.width()} != 1>>. Please check pixmap width!")
        if not isinstance(signal_color, str):
            if hasattr(signal_color, "color"):
                signal_color = signal_color.color.name()
            elif hasattr(signal_color, "name"):
                signal_color = signal_color.name()
            else:
                raise SyntaxError(f"Object {signal_color} doesn't have the attribute <<color>> or <<name()>>")

        line = None
        for y in range(image.height()):
            if QtGui.QColor(image.pixel(0, y)).name() == signal_color:
                line = y
                break
        return line

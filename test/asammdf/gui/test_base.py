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

from collections.abc import Iterable
import os
import pathlib
import shutil
import sys
import time
import unittest
from unittest import mock

from h5py import File as HDF5
import pyqtgraph
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from asammdf import mdf
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

    pyqtgraph.setConfigOption("background", "k")
    pyqtgraph.setConfigOption("foreground", "w")

    settings = QtCore.QSettings()
    settings.setValue("zoom_x_center_on_cursor", True)
    settings.setValue("plot_cursor_precision", 6)

    longMessage = False

    resource = os.path.normpath(os.path.join(os.path.dirname(__file__), "resources"))
    test_workspace = os.path.join(os.path.dirname(__file__), "test_workspace")
    screenshots = os.path.join(os.path.dirname(__file__), "screenshots")

    patchers = []
    # MockClass ErrorDialog
    mc_ErrorDialog = None

    def shortDescription(self):
        return self._testMethodDoc

    @staticmethod
    def manual_use(w, duration=None):
        """
        Execute Widget for debug/development purpose.

        Parameters
        ----------
        duration : float | None
            duration in seconds
        """
        if duration is None:
            duration = 3600
        else:
            duration = abs(duration)

        w.showMaximized()

        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(int(duration * 1000), loop.quit)
        loop.exec_()

        w.showMaximized()

    @staticmethod
    def processEvents(timeout=0.001):
        QtCore.QCoreApplication.processEvents()
        QtCore.QCoreApplication.sendPostedEvents()
        QtCore.QEventLoop.processEvents(QtCore.QEventLoop())
        if timeout:
            time.sleep(timeout)
            QtCore.QCoreApplication.processEvents()
            QtCore.QCoreApplication.sendPostedEvents()

    def setUp(self) -> None:
        if os.path.exists(self.test_workspace):
            try:
                shutil.rmtree(self.test_workspace)
            except PermissionError as e:
                print(e)

        os.makedirs(self.screenshots, exist_ok=True)
        os.makedirs(self.test_workspace, exist_ok=True)

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
        w = getattr(self, "widget", None)
        if w:
            self.destroy(w)

        self.mc_ErrorDialog.reset_mock()

        if self.test_workspace and pathlib.Path(self.test_workspace).exists():
            try:
                shutil.rmtree(self.test_workspace)
            except PermissionError as e:
                self.destroy(w)
                print(e)

    @staticmethod
    def destroy(w):
        w.close()
        w.destroy()
        w.deleteLater()

    def mouseClick_RadioButton(self, qitem):
        QtTest.QTest.mouseClick(
            qitem,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
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
            QtCore.Qt.KeyboardModifier.NoModifier,
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
            QtCore.Qt.KeyboardModifier.NoModifier,
            widget.visualItemRect(qitem).center(),
        )
        self.processEvents(0.5)

    def is_not_blinking(self, to_grab, colors: set[str], timeout=5):
        """
        Parameters
        ----------
        - to_grab: widget ex: self.plot
        - colors: a set of colors names ex: {"#123456", "#ffffff"}
        Returns
        -------
        - True: if not timeout and all colors exist on pixmap
        - False: if timeout and no colors exist on pixmap
        """
        if not hasattr(to_grab, "grab"):
            raise Warning(f"object {to_grab} has no attribute grab")

        now = time.perf_counter()
        all_colors = Pixmap.color_names_exclude_defaults(to_grab.grab())
        while not colors.issubset(all_colors):
            self.processEvents(0.01)
            all_colors = Pixmap.color_names_exclude_defaults(to_grab.grab())
            if time.perf_counter() - now > timeout:
                return False
        return True


class DragAndDrop:
    def __init__(self, src_widget, dst_widget, src_pos, dst_pos):
        QtCore.QCoreApplication.processEvents()
        # hack QDrag object
        with mock.patch(f"{src_widget.__module__}.QtGui.QDrag") as mo_QDrag:
            src_widget.startDrag(QtCore.Qt.DropAction.MoveAction)
            mo_QDrag.assert_called()
            mime_data = mo_QDrag.return_value.setMimeData.call_args.args[0]

            event = QtGui.QDragEnterEvent(
                dst_pos,
                QtCore.Qt.DropAction.MoveAction,
                mime_data,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
            with mock.patch.object(event, "source", return_value=src_widget):
                dst_widget.dragEnterEvent(event)

            event = QtGui.QDropEvent(
                dst_pos,
                QtCore.Qt.DropAction.MoveAction,
                mime_data,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
            with mock.patch.object(event, "source", return_value=src_widget):
                dst_widget.dropEvent(event)
        QtCore.QCoreApplication.processEvents()


class Pixmap:
    COLOR_BACKGROUND = "#000000"
    COLOR_RANGE = "#000032"
    COLOR_CURSOR = "#e69138"

    @staticmethod
    def is_black(pixmap) -> bool:
        """
        Excepting cursor
        """
        image = pixmap.toImage()

        for y in range(image.height()):
            for x in range(image.width()):
                color = QtGui.QColor(image.pixel(x, y))
                if color.name() not in (Pixmap.COLOR_BACKGROUND, Pixmap.COLOR_CURSOR):
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
        Return True if "pixmap" has "color_name" color
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
        """

        Parameters
        ----------
        pixmap: QPixmap object of PlotGraphics object

        Returns
        -------
        All colors from pixmap including default colors
        """
        color_names = set()

        image = pixmap.toImage()
        for y in range(image.height()):
            for x in range(image.width()):
                color = QtGui.QColor(image.pixel(x, y))
                color_names.add(color.name())
        return color_names

    @staticmethod
    def color_names_exclude_defaults(pixmap):
        """
        Parameters
        ----------
        pixmap: QPixmap object of PlotGraphics object

        Returns
        -------
        All colors from pixmap excluding default colors
        """
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
        """

        Parameters
        ----------
        pixmap: QPixmap object of PlotGraphics object

        Returns
        -------
        list of cursors line from pixmap
        """
        image = pixmap.toImage()

        cursors = []
        possible_cursor = None

        for x in range(image.width()):
            count = 0
            for y in range(image.height()):
                color = QtGui.QColor(image.pixel(x, y))
                # Count straight vertical line pixels with COLOR_CURSOR color
                if color.name() == Pixmap.COLOR_CURSOR:
                    count += 1
            if count >= (image.height() - 1) / 2 - 1:  # For Y shortcut tests, one cursor is a discontinuous line
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
        if ax in ("x", "X"):
            for x in range(image.width()):
                for y in range(image.height()):
                    if image.pixelColor(x, y).name() == signal_color:
                        from_to.append(x)
                        break
                if from_to:
                    break
            if not from_to:
                return
            for x in range(image.width() - 1, from_to[0], -1):
                for y in range(image.height()):
                    if image.pixelColor(x, y).name() == signal_color:
                        from_to.append(x)
                        break
                if len(from_to) == 2:
                    break
            return from_to

        elif ax in ("y", "Y"):
            for y in range(image.height()):
                for x in range(image.width()):
                    if image.pixelColor(x, y).name() == signal_color:
                        from_to.append(y)
                        break
                if from_to:
                    break
            if not from_to:
                return

            for y in range(image.height() - 1, from_to[0], -1):
                for x in range(image.width()):
                    if image.pixelColor(x, y).name() == signal_color:
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


class OpenFileContextManager:
    """
    Generic class for opening a file using context manager.
    Methods:
        __enter__: return opened file object.
        __exit__: close file object. If exc_type, exc_val, exc_tb, raise exception.
    """

    def __init__(self, file_path: str | pathlib.Path):
        """
        Parameters
        ----------
        file_path: file path as str or pathlib.Path object
        """
        self.file = None
        if isinstance(file_path, str):
            self._file_path = pathlib.Path(file_path)
        else:
            self._file_path = file_path

        assert self._file_path.exists(), "Provided file does not exist"

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        for exc in (exc_type, exc_val, exc_tb):
            if exc is not None:
                print(exc)


class OpenMDF(OpenFileContextManager):
    """
    Open MDF file using context manager.
    """

    def __enter__(self):
        self.file = mdf.MDF(self._file_path, process_bus_logging=("process_bus_logging", True))
        return self.file


class OpenHDF5(OpenFileContextManager):
    """
    Open HDF5 file using context manager.
    """

    def __enter__(self):
        self.file = HDF5(self._file_path)
        return self.file


class DBC:
    class BO:
        def __init__(self, lines: Iterable[str]):
            for line in lines:
                if line.startswith("BO_ "):
                    self.data = line

        def __repr__(self):
            return self.data

        @property
        def name(self) -> str:
            if self.data:
                return self.data.split()[2].strip(":")

        @property
        def id(self) -> str:
            if self.data:
                return self.data.split()[1]

        @property
        def data_length(self) -> int:
            if self.data:
                return int(self.data.split()[3])

    class SG:
        def __init__(self, name: str, lines: Iterable[str]):
            self.name = name
            self.data = None
            for line in lines:
                if "SG_ " in line and name in line:
                    self.data = line.split("SG_ ")[1]

        def __repr__(self):
            return self.data

        @property
        def unit(self) -> str:
            return self.data.split('"')[1]  # middle value, ex: "blah blah "unit" blah

        @property
        def bit_count(self) -> int:
            return int(self.data.split("|")[1].split("@")[0])  # is complicated this CAN :)

        @property
        def conversion_a(self) -> float:
            return float(self.data.split("(")[1].split(")")[0].split(",")[0])

        @property
        def conversion_b(self) -> float:
            return float(self.data.split("(")[1].split(")")[0].split(",")[1])

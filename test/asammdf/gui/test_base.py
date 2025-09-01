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
import functools
import os
import pathlib
import shutil
import sys
import time
from typing import Literal
import unittest
from unittest import mock

from h5py import File as HDF5
import numpy as np
from numpy.typing import NDArray
import pyqtgraph
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from asammdf import mdf
from asammdf.gui.utils import excepthook
from asammdf.gui.widgets.tree import ChannelsTreeItem

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


def safe_setup(func):
    """Decorator for a TestCase setUp method.
    Ensures that if setUp raises an exception, tearDown is still called.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except:
            if hasattr(self, "tearDown"):
                self.tearDown()
            raise

    return wrapper


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
    settings.setValue("plot/zoom/x_center_on_cursor", True)
    settings.setValue("plot/cursor/display_precision", 6)

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
        loop.exec()

        w.showMaximized()

    @staticmethod
    def processEvents(timeout=0.001):
        t_end = time.perf_counter() + timeout
        while True:
            time.sleep(0.001)
            QtCore.QCoreApplication.processEvents(flags=QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
            if time.perf_counter() > t_end:
                break

    @safe_setup
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
            self.take_screenshot(w)
            self.destroy(w)

        self.mc_ErrorDialog.reset_mock()

        if self.test_workspace and pathlib.Path(self.test_workspace).exists():
            shutil.rmtree(self.test_workspace)

    def take_screenshot(self, widget):
        path = self.screenshots
        for name in self.id().split(".")[:-1]:
            _path = os.path.join(path, name)
            if not os.path.exists(_path):
                os.makedirs(_path)
            path = _path
        widget.grab().save(os.path.join(path, f"{self.id().split('.')[-1]}.png"))

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

    def is_not_blinking(self, to_grab: QtWidgets.QWidget, colors: set[str], timeout: float = 5.0) -> bool:
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
        while True:
            self.processEvents(0.01)
            all_colors = Pixmap.color_names_exclude_defaults(to_grab.grab())
            if colors.issubset(all_colors):
                break
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
    def to_numpy(pixmap: QtGui.QPixmap) -> NDArray[np.uint32]:
        """
        Convert a QPixmap to a NumPy array of 32-bit ARGB values.

        This function returns a new NumPy array with a copy of the pixel data, so
        modifications to the array do not affect the original QPixmap.

        Parameters
        ----------
        pixmap : QtGui.QPixmap
            The QPixmap to convert.

        Returns
        -------
        NDArray[np.uint32]
            A 2D NumPy array of shape (height, width), where each element is a
            32-bit unsigned integer representing a pixel in ARGB format.
        """
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32)
        ptr = image.bits()
        arr = np.frombuffer(ptr, np.uint32).reshape((pixmap.height(), pixmap.width()))
        return arr.copy()

    @staticmethod
    def is_black(pixmap: QtGui.QPixmap) -> bool:
        """
        Check if a QPixmap is entirely black, ignoring background and cursor colors.

        The function computes all unique colors in the pixmap, excludes the
        predefined background and cursor colors, and determines whether any
        remaining colors exist.

        Parameters
        ----------
        pixmap : QtGui.QPixmap
            The QPixmap to check.

        Returns
        -------
        bool
            True if all visible pixels are black (excluding background and cursor),
            False otherwise.
        """
        ignored_colors = {Pixmap.COLOR_BACKGROUND, Pixmap.COLOR_CURSOR}
        colors = Pixmap.color_names(pixmap) - ignored_colors
        return len(colors) == 0

    @staticmethod
    def is_colored(
        pixmap: QtGui.QPixmap,
        color_name: str,
        x: int,
        y: int,
        width: int | None = None,
        height: int | None = None,
    ) -> bool:
        """
        Check whether a rectangular region of a QPixmap is entirely filled with a given color.

        Parameters
        ----------
        pixmap : QtGui.QPixmap
            The QPixmap to inspect.
        color_name : str
            The target color, expressed as a hex string (e.g., "#ff0000").
        x : int
            The x-coordinate of the top-left corner of the rectangle.
        y : int
            The y-coordinate of the top-left corner of the rectangle.
        width : int | None, optional
            The width of the rectangle. If None, defaults to the width of the pixmap from x.
        height : int | None, optional
            The height of the rectangle. If None, defaults to the height of the pixmap from y.

        Returns
        -------
        bool
            True if all pixels in the specified rectangle have the given color, False otherwise.

        """
        width = width or (pixmap.width() - x)
        height = height or (pixmap.height() - x)

        arr = Pixmap.to_numpy(pixmap)
        arr_rect = arr[y : y + height, x : x + width]
        colors_in_rect = {QtGui.QColor.fromRgba(x).name() for x in np.unique_values(arr_rect)}
        return colors_in_rect == {color_name}

    @staticmethod
    def has_color(pixmap: QtGui.QPixmap, color_name: str | QtGui.QColor | ChannelsTreeItem, tolerance: int = 0) -> bool:
        """
        Return True if "pixmap" contains a color close to "color_name", allowing some tolerance.

        This function accounts for anti-aliasing and smoothing by checking
        if any pixel's color is within a specified tolerance of the target color.

        Parameters
        ----------
        pixmap : QtGui.QPixmap
            The pixmap to check.
        color_name : QtGui.QColor or str
            The color to detect, either as a QColor or hex string (e.g., "#ff0000").
        tolerance : int
            Maximum allowed Euclidean distance in RGB space to consider a pixel as matching.

        Returns
        -------
        bool
            True if at least one pixel in the pixmap is close enough to the target color.
        """
        if isinstance(color_name, str):
            color_name = QtGui.QColor(color_name)
        elif isinstance(color_name, ChannelsTreeItem):
            color_name = QtGui.QColor(color_name.color)

        arr = Pixmap.to_numpy(pixmap)
        # Extract RGB components
        r = (arr >> 16) & 0xFF
        g = (arr >> 8) & 0xFF
        b = arr & 0xFF

        # Compute squared distance to target color in RGB space
        dr = r.astype(np.int64) - color_name.red()
        dg = g.astype(np.int64) - color_name.green()
        db = b.astype(np.int64) - color_name.blue()
        dist_sq = np.square(dr) + np.square(dg) + np.square(db)

        return bool(np.any(dist_sq <= tolerance**2))

    @staticmethod
    def color_names(pixmap: QtGui.QPixmap) -> set[str]:
        """
        Return a set of all unique colors in a QPixmap as hex strings.

        The function converts the QPixmap to a NumPy array of 32-bit ARGB values,
        finds all unique pixel values, and converts each to a standard hex color
        string (e.g., "#ff0000").

        Parameters
        ----------
        pixmap : QtGui.QPixmap
            The QPixmap from which to extract colors.

        Returns
        -------
        set[str]
            A set of hex color strings representing all unique colors present in the pixmap.
        """
        arr = Pixmap.to_numpy(pixmap)
        argb32_set = np.unique_values(arr)
        color_names = {QtGui.QColor.fromRgba(x).name() for x in argb32_set}
        return color_names

    @staticmethod
    def color_names_exclude_defaults(pixmap: QtGui.QPixmap) -> set[str]:
        """
        Return a set of all unique colors in a QPixmap, excluding default colors.

        This function extracts all unique colors from the pixmap and removes
        predefined default colors such as background, cursor, and range colors.

        Parameters
        ----------
        pixmap : QtGui.QPixmap
            The QPixmap from which to extract colors.

        Returns
        -------
        set[str]
            A set of hex color strings representing all unique colors in the pixmap,
            excluding the default background, cursor, and range colors.
        """
        defaults = {Pixmap.COLOR_BACKGROUND, Pixmap.COLOR_CURSOR, Pixmap.COLOR_RANGE}
        return Pixmap.color_names(pixmap) - defaults

    @staticmethod
    def color_map(pixmap: QtGui.QPixmap) -> dict[int, list[str]]:
        """
        Convert a QPixmap into a per-line mapping of pixel colors as hex strings.

        This function efficiently maps each pixel in the pixmap to its corresponding
        hex color string using NumPy. It avoids Python loops over individual pixels
        by vectorizing the conversion from ARGB32 values to color strings.

        Parameters
        ----------
        pixmap : QtGui.QPixmap
            The QPixmap to convert.

        Returns
        -------
        dict[int, list[str]]
            A dictionary where:
            - Keys are row indices (0-based).
            - Values are lists of hex color strings representing the colors of
            pixels in that row, ordered by column.
        """
        arr = Pixmap.to_numpy(pixmap)  # shape (H, W), dtype uint32

        # 1. Map all unique ARGB values to hex strings
        unique_vals = np.unique(arr)  # returns sortes values
        lookup = np.array([QtGui.QColor.fromRgba(int(val)).name() for val in unique_vals])

        # 2. Create an index array
        # np.searchsorted requires sorted unique_vals
        indices = np.searchsorted(unique_vals, arr)

        # 3. Map indices to hex strings
        hex_array = lookup[indices]  # shape (H, W) (hex strings)

        return {y: list(row) for y, row in enumerate(hex_array)}

    @staticmethod
    def cursors_x(pixmap: QtGui.QPixmap, threshold: float = 0.3) -> list[int]:
        """
        Find the x-coordinates of vertical cursors in a QPixmap, including dashed cursors.

        A column is considered a vertical cursor if the fraction of pixels matching
        the cursor color exceeds the given threshold.

        Parameters
        ----------
        pixmap : QtGui.QPixmap
            The pixmap to scan for vertical cursor pixels.
        threshold : float
            Minimum fraction of pixels in a column that must match the cursor color
            to consider it a vertical cursor (between 0 and 1).

        Returns
        -------
        list[int]
            List of x-coordinates corresponding to detected vertical cursors.
        """
        arr = Pixmap.to_numpy(pixmap)
        cursor_color = QtGui.QColor(Pixmap.COLOR_CURSOR).rgba()

        # Count pixels matching the cursor color per column
        match_counts = np.sum(arr == cursor_color, axis=0)

        # Determine fraction of pixels in each column that match
        fraction = match_counts / arr.shape[0]

        # Columns exceeding the threshold are considered vertical cursors
        cols_with_cursor = np.nonzero(fraction >= threshold)[0]

        return [int(x) for x in cols_with_cursor]

    @staticmethod
    def search_signal_extremes_by_ax(
        pixmap: QtGui.QPixmap, signal_color: str | QtGui.QColor, ax: Literal["X", "Y"]
    ) -> tuple[int, int] | None:
        """
        Find the start and end indices of a colored signal in a QPixmap along a specified axis.

        This function returns the first and last row or column where the signal color appears,
        using NumPy for fast pixel scanning.

        Parameters
        ----------
        pixmap : QtGui.QPixmap
            The pixmap containing the signal.
        signal_color : str or QtGui.QColor
            The color of the signal to search for, as a hex string or QColor.
        ax : {"X", "Y"}
            The axis along which to find extremes:
            - "X": returns [start_column, end_column]
            - "Y": returns [start_row, end_row]

        Returns
        -------
        tuple[int, int] | None
            Two-element tuple with the start and end index along the chosen axis, or None if color not found.
        """
        # Normalize signal color to hex string
        if isinstance(signal_color, str):
            signal_color = QtGui.QColor(signal_color)

        arr = Pixmap.to_numpy(pixmap)  # shape (H, W), dtype uint32
        # Convert signal color to ARGB uint32
        signal_val = QtGui.QColor(signal_color).rgba()

        if ax.upper() == "X":
            mask = np.any(arr == signal_val, axis=0)  # columns containing signal
            indices = np.flatnonzero(mask)
        elif ax.upper() == "Y":
            mask = np.any(arr == signal_val, axis=1)  # rows containing signal
            indices = np.flatnonzero(mask)
        else:
            raise ValueError("ax must be 'X' or 'Y'")

        if len(indices) == 0:
            return None

        return (int(indices[0]), int(indices[-1]))


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

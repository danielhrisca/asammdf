#
# this file contains substantial amount of code from https://github.com/adamerose/PandasGUI which is licensed as MIT:
#
# MIT License
#
# Copyright (c) 2018 Adam Rose
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
#

import bisect
import datetime
import logging
import threading
from traceback import format_exc

import numpy as np
import numpy.core.defchararray as npchar
import pandas as pd
import pyqtgraph.functions as fn
from PySide6 import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt

import asammdf.mdf as mdf_module

from ...blocks.utils import (
    csv_bytearray2hex,
    extract_mime_names,
    pandas_query_compatible,
)
from ..dialogs.range_editor import RangeEditor
from ..ui.tabular import Ui_TabularDisplay
from ..utils import (
    copy_ranges,
    FONT_SIZE,
    get_colors_using_ranges,
    run_thread_with_progress,
    value_as_str,
)
from .tabular_filter import TabularFilter

logger = logging.getLogger("asammdf.gui")
LOCAL_TIMEZONE = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo


MONOSPACE_FONT = None


class TabularTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, column_types, int_format, ranges=None, *args, **kwargs):
        self.column_types = column_types
        self.int_format = int_format
        self.ranges = ranges
        super().__init__(*args, **kwargs)

        self._back_ground_color = self.background(0)
        self._font_color = self.foreground(0)

        self._current_background_color = self._back_ground_color
        self._current_font_color = self._font_color

        self.check_signal_range()

    def __lt__(self, other):
        column = self.treeWidget().sortColumn()

        dtype = self.column_types[column]

        if dtype in "ui":
            if self.int_format == "hex":
                return int(self.text(column), 16) < int(other.text(column), 16)
            elif self.int_format == "bin":
                return int(self.text(column), 2) < int(other.text(column), 2)
            else:
                return int(self.text(column)) < int(other.text(column))

        elif dtype == "f":
            return float(self.text(column)) < float(other.text(column))

        else:
            return self.text(column) < other.text(column)

    def check_signal_range(self):
        if not self.ranges:
            return

        for column, channel_ranges in enumerate(self.ranges, 1):
            value = self.text(column)
            dtype = self.column_types[column]

            if dtype in "ui":
                if self.int_format == "hex":
                    value = int(value, 16)
                elif self.int_format == "bin":
                    value = int(value, 2)
                else:
                    value = int(value)
                value = float(value)

            elif dtype == "f":
                value = float(value)

            new_background_color, new_font_color = get_colors_using_ranges(
                value,
                ranges=channel_ranges,
                default_background_color=self._current_background_color,
                default_font_color=self._current_font_color,
            )

            self.setBackground(column, new_background_color)
            self.setForeground(column, new_font_color)


class DataFrameStorage:
    """
    All methods that modify the data should modify self.df_unfiltered, then self.df gets computed from that
    """

    def __init__(self, df, tabular):
        super().__init__()

        self.df = df
        self.df_unfiltered = df
        self.tabular = tabular

        self.sorted_column_name = None
        self.sorted_index_level = None
        self.sort_state = "None"
        self.dataframe_viewer = None
        self.filter_viewer = None

        self.filters = []
        self.filtered_index_map = df.reset_index().index

        self.data_changed()

    @property
    def sorted_column_ix(self):
        try:
            return list(self.df_unfiltered.columns).index(self.sorted_column_name)
        except ValueError:
            return None

    ###################################
    # Changing columns
    def delete_column(self, ix):
        col_name = self.df_unfiltered.columns[ix]
        self.df_unfiltered = self.df_unfiltered.drop(col_name, axis=1)

        # Need to inform the PyQt model too so column widths properly shift
        self.dataframe_viewer._remove_column(ix)

        self.parent.apply_filters()

    def move_column(self, src, dest):
        cols = list(self.df_unfiltered.columns)
        cols.insert(dest, cols.pop(src))
        self.df_unfiltered = self.df_unfiltered.reindex(cols, axis=1)

        self.dataframe_viewer.setUpdatesEnabled(False)
        # Need to inform the PyQt model too so column widths properly shift
        self.dataframe_viewer._move_column(src, dest)
        self.apply_filters()
        self.dataframe_viewer.setUpdatesEnabled(True)

    ###################################
    # Sorting

    def sort_column(self, ix, next_sort_state=None):
        col_name = self.df_unfiltered.columns[ix]

        # Determine next sorting state by current state
        if next_sort_state is None:
            # Clicked an unsorted column
            if ix != self.sorted_column_ix:
                next_sort_state = "Asc"
            # Clicked a sorted column
            elif ix == self.sorted_column_ix and self.sort_state == "Asc":
                next_sort_state = "Desc"
            # Clicked a reverse sorted column - reset to sorted by index
            elif ix == self.sorted_column_ix:
                next_sort_state = "None"

        if next_sort_state == "Asc":
            self.df_unfiltered = self.df_unfiltered.sort_values(col_name, ascending=True, kind="mergesort")
            self.sorted_column_name = self.df_unfiltered.columns[ix]
            self.sort_state = "Asc"

        elif next_sort_state == "Desc":
            self.df_unfiltered = self.df_unfiltered.sort_values(col_name, ascending=False, kind="mergesort")
            self.sorted_column_name = self.df_unfiltered.columns[ix]
            self.sort_state = "Desc"

        elif next_sort_state == "None":
            self.df_unfiltered = self.df_unfiltered.sort_index(ascending=True, kind="mergesort")
            self.sorted_column_name = None
            self.sort_state = "None"

        self.sorted_index_level = None
        self.tabular.apply_filters()

    def sort_index(self, ix: int):
        # Clicked an unsorted index level
        if ix != self.sorted_index_level:
            self.df_unfiltered = self.df_unfiltered.sort_index(level=ix, ascending=True, kind="mergesort")
            self.sorted_index_level = ix
            self.sort_state = "Asc"

        # Clicked a sorted index level
        elif ix == self.sorted_index_level and self.sort_state == "Asc":
            self.df_unfiltered = self.df_unfiltered.sort_index(level=ix, ascending=False, kind="mergesort")
            self.sorted_index_level = ix
            self.sort_state = "Desc"

        # Clicked a reverse sorted index level - reset to sorted by full index
        elif ix == self.sorted_index_level:
            self.df_unfiltered = self.df_unfiltered.sort_index(ascending=True, kind="mergesort")

            self.sorted_index_level = None
            self.sort_state = "None"

        self.sorted_column = None
        self.tabular.apply_filters()

    def change_column_type(self, ix: int, type):
        name = self.df_unfiltered.columns[ix]
        self.df_unfiltered[name] = self.df_unfiltered[name].astype(type)
        self.tabular.apply_filters()

    ###################################
    # Other

    def data_changed(self):
        self.refresh_ui()

    # Refresh PyQt models when the underlying pgdf is changed in anyway that needs to be reflected in the GUI
    def refresh_ui(self):
        self.models = []

        if self.filter_viewer is not None:
            self.models += [self.filter_viewer.list_model]

        for model in self.models:
            model.beginResetModel()
            model.endResetModel()

        if self.dataframe_viewer is not None:
            self.dataframe_viewer.refresh_ui()


class DataTableModel(QtCore.QAbstractTableModel):
    """
    Model for DataTableView to connect for DataFrame data
    """

    def __init__(self, parent, background_color, font_color):
        super().__init__(parent)
        self.dataframe_viewer = parent
        self.pgdf = parent.pgdf
        self.format = "phys"
        self.float_precision = -1
        self.background_color = background_color
        self.font_color = font_color

    def headerData(self, section, orientation, role=None):
        # Headers for DataTableView are hidden. Header data is shown in HeaderView
        pass

    def columnCount(self, parent=None):
        return self.pgdf.df.columns.shape[0]

    def rowCount(self, parent=None):
        return len(self.pgdf.df)

    # Returns the data from the DataFrame
    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        row = index.row()
        col = index.column()

        cell = self.pgdf.df.iloc[row, col]

        name = self.pgdf.df_unfiltered.columns[col]

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            # Need to check type since a cell might contain a list or Series, then .isna returns a Series not a bool
            cell_is_na = pd.isna(cell)

            if type(cell_is_na) == bool and cell_is_na:
                return "NaN"
            elif isinstance(cell, (bytes, np.bytes_)):
                return cell.decode("utf-8", "replace")
            else:
                return value_as_str(cell, self.format, None, self.float_precision)

        elif role == QtCore.Qt.ItemDataRole.BackgroundRole:
            channel_ranges = self.pgdf.tabular.ranges[name]

            try:
                value = float(cell)
            except:
                value = str(cell)

            new_background_color, new_font_color = get_colors_using_ranges(
                value,
                ranges=channel_ranges,
                default_background_color=self.background_color,
                default_font_color=self.font_color,
            )

            return new_background_color if new_background_color != self.background_color else None

        elif role == QtCore.Qt.ItemDataRole.ForegroundRole:
            channel_ranges = self.pgdf.tabular.ranges[name]

            try:
                value = float(cell)
            except:
                value = str(cell)

            new_background_color, new_font_color = get_colors_using_ranges(
                value,
                ranges=channel_ranges,
                default_background_color=self.background_color,
                default_font_color=self.font_color,
            )

            return new_font_color if new_font_color != self.font_color else None

        elif role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(cell, str):
                return int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            elif isinstance(cell, pd.Timestamp):
                return int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            else:
                if self.float_precision == -1 and isinstance(cell, (float, np.floating)):
                    return int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
                else:
                    return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

    def flags(self, index):
        return QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable

    def setData(self, index, value, role=None):
        pass


class DataTableView(QtWidgets.QTableView):
    add_channels_request = QtCore.Signal(list)

    def __init__(self, parent):
        super().__init__(parent)
        self.dataframe_viewer = parent
        self.pgdf = parent.pgdf

        self._backgrund_color = self.palette().color(QtGui.QPalette.ColorRole.Window)
        self._font_color = self.palette().color(QtGui.QPalette.ColorRole.WindowText)

        # Create and set model
        model = DataTableModel(parent, self._backgrund_color, self._font_color)
        self.setModel(model)

        # Hide the headers. The DataFrame headers (index & columns) will be displayed in the DataFrameHeaderViews
        self.horizontalHeader().hide()
        self.verticalHeader().hide()

        # Link selection to headers
        self.selectionModel().selectionChanged.connect(self.on_selectionChanged)

        # Settings
        # self.setWordWrap(True)
        # self.resizeRowsToContents()
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)

        font = QtGui.QFont()
        font.fromString(MONOSPACE_FONT)
        self.setFont(font)

        self.setAcceptDrops(False)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.NoDragDrop)
        self.setDropIndicatorShown(False)

    def on_selectionChanged(self):
        """
        Runs when cells are selected in the main table. This logic highlights the correct cells in the vertical and
        horizontal headers when a data cell is selected
        """
        columnHeader = self.dataframe_viewer.columnHeader
        indexHeader = self.dataframe_viewer.indexHeader

        # The two blocks below check what columns or rows are selected in the data table and highlights the
        # corresponding ones in the two headers. The if statements check for focus on headers, because if the user
        # clicks a header that will auto-select all cells in that row or column which will trigger this function
        # and cause and infinite loop

        if not columnHeader.hasFocus():
            selection = self.selectionModel().selection()
            columnHeader.selectionModel().select(
                selection,
                QtCore.QItemSelectionModel.SelectionFlag.Columns
                | QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )

        if not indexHeader.hasFocus():
            selection = self.selectionModel().selection()
            indexHeader.selectionModel().select(
                selection,
                QtCore.QItemSelectionModel.SelectionFlag.Rows | QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )

    def sizeHint(self):
        # Set width and height based on number of columns in model
        # Width
        width = 2 * self.frameWidth()  # Account for border & padding
        # width += self.verticalScrollBar().width()  # Dark theme has scrollbars always shown
        for i in range(self.model().columnCount()):
            width += self.columnWidth(i)

        # Height
        height = 2 * self.frameWidth()  # Account for border & padding
        # height += self.horizontalScrollBar().height()  # Dark theme has scrollbars always shown
        height += 24 * self.model().rowCount()
        # for i in range(self.model().rowCount()):
        #     height += self.rowHeight(i)

        return QtCore.QSize(width, height)

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        if e.source() is self:
            return
        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
            else:
                return

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key.Key_R and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.accept()
            selected_items = {index.column() for index in self.selectedIndexes() if index.isValid()}

            if selected_items:
                ranges = []
                for index in selected_items:
                    original_name = self.pgdf.df_unfiltered.columns[index]
                    ranges.update(self.pgdf.tabular.ranges[original_name])

                dlg = RangeEditor("<selected signals>", "", ranges=ranges, parent=self, brush=True)
                dlg.exec_()
                if dlg.pressed_button == "apply":
                    ranges = dlg.result

                    for index in selected_items:
                        original_name = self.pgdf.df_unfiltered.columns[index]
                        self.pgdf.tabular.ranges[original_name] = copy_ranges(ranges)

        else:
            super().keyPressEvent(event)


class HeaderModel(QtCore.QAbstractTableModel):
    def __init__(self, parent, orientation):
        super().__init__(parent)
        self.orientation = orientation
        self.pgdf = parent.pgdf
        self.prefix = ""

    def columnCount(self, parent=None):
        if self.orientation == Qt.Orientation.Horizontal:
            return self.pgdf.df.columns.shape[0]
        else:  # Vertical
            return self.pgdf.df.index.nlevels

    def rowCount(self, parent=None):
        if self.orientation == Qt.Orientation.Horizontal:
            return self.pgdf.df.columns.nlevels
        elif self.orientation == Qt.Orientation.Vertical:
            return self.pgdf.df.index.shape[0]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        row = index.row()
        col = index.column()

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if self.orientation == Qt.Orientation.Horizontal:
                if isinstance(self.pgdf.df.columns, pd.MultiIndex):
                    val = str(self.pgdf.df.columns[col][row])
                else:
                    val = str(self.pgdf.df.columns[col])

                val = val[len(self.prefix) :] if val.startswith(self.prefix) else val
                return val

            elif self.orientation == Qt.Orientation.Vertical:
                if isinstance(self.pgdf.df.index, pd.MultiIndex):
                    return str(self.pgdf.df.index[row][col])
                else:
                    return str(self.pgdf.df.index[row])

        elif role == QtCore.Qt.ItemDataRole.DecorationRole:
            if self.pgdf.sort_state == "Asc":
                icon = QtGui.QIcon(":/sort-ascending.png")
            elif self.pgdf.sort_state == "Desc":
                icon = QtGui.QIcon(":/sort-descending.png")
            else:
                return

            if (
                col == self.pgdf.sorted_column_ix
                and row == self.rowCount() - 1
                and self.orientation == Qt.Orientation.Horizontal
            ):
                return icon

        elif role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            if self.orientation == Qt.Orientation.Horizontal:
                name = self.pgdf.df_unfiltered.columns[col]
                dtype = self.pgdf.df_unfiltered[name].values.dtype

                float_precision = self.pgdf.dataframe_viewer.dataView.model().float_precision

                if np.issubdtype(dtype, np.integer):
                    return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                elif float_precision != -1 and np.issubdtype(dtype, np.floating):
                    return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                else:
                    return int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            else:
                return QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter

    # The headers of this table will show the level names of the MultiIndex
    def headerData(self, section, orientation, role=None):
        # This was moved to HeaderNamesModel
        pass


class HeaderView(QtWidgets.QTableView):
    """
    Displays the DataFrame index or columns depending on orientation
    """

    def __init__(self, parent, orientation):
        super().__init__(parent)
        self.dataframe_viewer = parent
        self.pgdf = parent.pgdf
        self.setProperty("orientation", "horizontal" if orientation == 1 else "vertical")  # Used in stylesheet

        # Setup
        self.orientation = orientation
        self.table = parent.dataView
        self.setModel(HeaderModel(parent, orientation))
        self.padding = 90

        ###############
        # These are used in self.manage_resizing

        # Holds the index of the cell being resized, or None if resize isn't happening
        self.header_cell_being_resized = None
        # Boolean indicating whether the header itself is currently being resized
        self.header_being_resized = False
        ###############

        # Handled by self.eventFilter()
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().installEventFilter(self)

        # Settings
        self.setIconSize(QtCore.QSize(16, 16))
        self.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        )
        self.setWordWrap(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        font = QtGui.QFont()
        font.fromString(MONOSPACE_FONT)
        font.setBold(True)
        self.setFont(font)

        # Link selection to DataTable
        self.selectionModel().selectionChanged.connect(lambda x: self.on_selectionChanged())
        # self.set_spans()

        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Automatically stretch rows/columns as widget is resized
        if self.orientation == Qt.Orientation.Vertical:
            self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        # Set initial size
        self.resize(self.sizeHint())

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        super().showEvent(a0)
        self.initial_size = self.size()

    def mouseDoubleClickEvent(self, event):
        point = event.pos()
        ix = self.indexAt(point)
        col = ix.column()
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # When a header is clicked, sort the DataFrame by that column
            if self.orientation == Qt.Orientation.Horizontal:
                self.pgdf.sort_column(col)
            else:
                self.on_selectionChanged()
        else:
            super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        point = event.pos()
        ix = self.indexAt(point)
        col = ix.column()
        if event.button() == QtCore.Qt.MouseButton.RightButton and self.orientation == Qt.Orientation.Horizontal:
            self.dataframe_viewer.show_column_menu(col)
        else:
            super().mousePressEvent(event)

    # Header
    def on_selectionChanged(self, force=False):
        """
        Runs when cells are selected in the Header. This selects columns in the data table when the header is clicked,
        and then calls selectAbove()
        """
        # Check focus so we don't get recursive loop, since headers trigger selection of data cells and vice versa
        if self.hasFocus() or force:
            dataView = self.dataframe_viewer.dataView

            # Set selection mode so selecting one row or column at a time adds to selection each time
            if self.orientation == Qt.Orientation.Horizontal:  # This case is for the horizontal header
                # Get the header's selected columns
                selection = self.selectionModel().selection()

                # Removes the higher levels so that only the lowest level of the header affects the data table selection
                last_row_ix = self.pgdf.df.columns.nlevels - 1
                last_col_ix = self.model().columnCount() - 1
                higher_levels = QtCore.QItemSelection(
                    self.model().index(0, 0),
                    self.model().index(last_row_ix - 1, last_col_ix),
                )
                selection.merge(higher_levels, QtCore.QItemSelectionModel.SelectionFlag.Deselect)

                # Select the cells in the data view
                dataView.selectionModel().select(
                    selection,
                    QtCore.QItemSelectionModel.SelectionFlag.Columns
                    | QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
                )
            if self.orientation == Qt.Orientation.Vertical:
                selection = self.selectionModel().selection()

                last_row_ix = self.model().rowCount() - 1
                last_col_ix = self.pgdf.df.index.nlevels - 1
                higher_levels = QtCore.QItemSelection(
                    self.model().index(0, 0),
                    self.model().index(last_row_ix, last_col_ix - 1),
                )
                selection.merge(higher_levels, QtCore.QItemSelectionModel.SelectionFlag.Deselect)

                dataView.selectionModel().select(
                    selection,
                    QtCore.QItemSelectionModel.SelectionFlag.Rows
                    | QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
                )

        self.selectAbove()

    # Take the current set of selected cells and make it so that any spanning cell above a selected cell is selected too
    # This should happen after every selection change
    def selectAbove(self):
        # Disabling this to allow selecting specific cells in headers
        return

        if self.orientation == Qt.Orientation.Horizontal:
            if self.pgdf.df.columns.nlevels == 1:
                return
        else:
            if self.pgdf.df.index.nlevels == 1:
                return

        for ix in self.selectedIndexes():
            if self.orientation == Qt.Orientation.Horizontal:
                # Loop over the rows above this one
                for row in range(ix.row()):
                    ix2 = self.model().index(row, ix.column())
                    self.setSelection(self.visualRect(ix2), QtCore.QItemSelectionModel.SelectionFlag.Select)
            else:
                # Loop over the columns left of this one
                for col in range(ix.column()):
                    ix2 = self.model().index(ix.row(), col)
                    self.setSelection(self.visualRect(ix2), QtCore.QItemSelectionModel.SelectionFlag.Select)

    # This sets spans to group together adjacent cells with the same values
    def set_spans(self):
        df = self.pgdf.df
        self.clearSpans()
        # Find spans for horizontal HeaderView
        if self.orientation == Qt.Orientation.Horizontal:
            # Find how many levels the MultiIndex has
            if isinstance(df.columns, pd.MultiIndex):
                N = len(df.columns[0])
            else:
                N = 1

            for level in range(N):  # Iterates over the levels
                # Find how many segments the MultiIndex has
                if isinstance(df.columns, pd.MultiIndex):
                    arr = [df.columns[i][level] for i in range(len(df.columns))]
                else:
                    arr = df.columns

                # Holds the starting index of a range of equal values.
                # None means it is not currently in a range of equal values.
                match_start = None

                for col in range(1, len(arr)):  # Iterates over cells in row
                    # Check if cell matches cell to its left
                    if arr[col] == arr[col - 1]:
                        if match_start is None:
                            match_start = col - 1
                        # If this is the last cell, need to end it
                        if col == len(arr) - 1:
                            match_end = col
                            span_size = match_end - match_start + 1
                            self.setSpan(level, match_start, 1, span_size)
                    else:
                        if match_start is not None:
                            match_end = col - 1
                            span_size = match_end - match_start + 1
                            self.setSpan(level, match_start, 1, span_size)
                            match_start = None

        # Find spans for vertical HeaderView
        else:
            # Find how many levels the MultiIndex has
            if isinstance(df.index, pd.MultiIndex):
                N = len(df.index[0])
            else:
                N = 1

            for level in range(N):  # Iterates over the levels
                # Find how many segments the MultiIndex has
                if isinstance(df.index, pd.MultiIndex):
                    arr = [df.index[i][level] for i in range(len(df.index))]
                else:
                    arr = df.index

                # Holds the starting index of a range of equal values.
                # None means it is not currently in a range of equal values.
                match_start = None

                for row in range(1, len(arr)):  # Iterates over cells in column
                    # Check if cell matches cell above
                    if arr[row] == arr[row - 1]:
                        if match_start is None:
                            match_start = row - 1
                        # If this is the last cell, need to end it
                        if row == len(arr) - 1:
                            match_end = row
                            span_size = match_end - match_start + 1
                            self.setSpan(match_start, level, span_size, 1)
                    else:
                        if match_start is not None:
                            match_end = row - 1
                            span_size = match_end - match_start + 1
                            self.setSpan(match_start, level, span_size, 1)
                            match_start = None

    def eventFilter(self, object: QtCore.QObject, event: QtCore.QEvent):
        if event.type() in [
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QEvent.Type.MouseButtonDblClick,
            QtCore.QEvent.Type.MouseMove,
        ]:
            return self.manage_resizing(object, event)

        return False

    # This method handles all the resizing of headers including column width, row height, and header width/height
    def manage_resizing(self, object: QtCore.QObject, event: QtCore.QEvent):
        # This is used for resizing column widths and row heights
        # For the horizontal header, return the column edge the mouse is over
        # For the vertical header, return the row edge the mouse is over
        # mouse_position is the position along the relevant axis, ie. horizontal x position for the top header
        def over_header_cell_edge(mouse_position, margin=3):
            # Return the index of the column this x position is on the right edge of
            if self.orientation == Qt.Orientation.Horizontal:
                x = mouse_position
                if self.columnAt(x - margin) != self.columnAt(x + margin):
                    if self.columnAt(x + margin) == 0:
                        # We're at the left edge of the first column
                        return None
                    else:
                        return self.columnAt(x - margin)
                else:
                    return None

            # Return the index of the row this y position is on the top edge of
            elif self.orientation == Qt.Orientation.Vertical:
                y = mouse_position
                if self.rowAt(y - margin) != self.rowAt(y + margin):
                    if self.rowAt(y + margin) == 0:
                        # We're at the top edge of the first row
                        return None
                    else:
                        return self.rowAt(y - margin)
                else:
                    return None

        # This is used for resizing the left header width or the top header height
        # Returns a boolean indicating whether the mouse is over the header edge to allow resizing
        def over_header_edge(mouse_position: QtCore.QPoint(), margin=7) -> bool:
            if self.orientation == Qt.Orientation.Horizontal:
                return abs(mouse_position - self.height()) < margin
            elif self.orientation == Qt.Orientation.Vertical:
                return abs(mouse_position - self.width()) < margin

        # mouse_position is the position along the axis of the header. X pos for top header, Y pos for side header
        if self.orientation == Qt.Orientation.Horizontal:
            mouse_position = event.pos().x()
            orthogonal_mouse_position = event.pos().y()
        else:
            mouse_position = event.pos().y()
            orthogonal_mouse_position = event.pos().x()

        # Set the cursor shape
        if over_header_cell_edge(mouse_position) is not None:
            if self.orientation == Qt.Orientation.Horizontal:
                self.viewport().setCursor(QtGui.QCursor(Qt.CursorShape.SplitHCursor))
            elif self.orientation == Qt.Orientation.Vertical:
                self.viewport().setCursor(QtGui.QCursor(Qt.CursorShape.SplitVCursor))

        elif over_header_edge(orthogonal_mouse_position):
            if self.orientation == Qt.Orientation.Horizontal:
                # Disabling vertical resizing of top header for now
                pass
                # self.viewport().setCursor(QtGui.QCursor(Qt.CursorShape.SplitVCursor))
            elif self.orientation == Qt.Orientation.Vertical:
                self.viewport().setCursor(QtGui.QCursor(Qt.CursorShape.SplitHCursor))

        else:
            self.viewport().setCursor(QtGui.QCursor(Qt.CursorShape.ArrowCursor))

        # If mouse is on an edge, start the drag resize process
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if over_header_cell_edge(mouse_position) is not None:
                self.header_cell_being_resized = over_header_cell_edge(mouse_position)
                return True
            # Disabling vertical resizing of top header for now
            elif over_header_edge(orthogonal_mouse_position) and self.orientation == Qt.Orientation.Vertical:
                self.header_being_resized = True
                return True
            else:
                self.header_cell_being_resized = None

        # End the drag process
        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            self.header_cell_being_resized = None
            self.header_being_resized = False

        # Auto size the column that was double clicked
        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            # Find which column or row edge the mouse was over and auto size it
            if over_header_cell_edge(mouse_position) is not None:
                header_index = over_header_cell_edge(mouse_position)
                if self.orientation == Qt.Orientation.Horizontal:
                    self.dataframe_viewer.auto_size_column(header_index)
                elif self.orientation == Qt.Orientation.Vertical:
                    self.dataframe_viewer.auto_size_row(header_index)
                return True

        # Handle drag resizing
        if event.type() == QtCore.QEvent.Type.MouseMove:
            # If this is None, there is no drag resize happening
            if self.header_cell_being_resized is not None:
                size = mouse_position - self.columnViewportPosition(self.header_cell_being_resized)
                if size > 10:
                    if self.orientation == Qt.Orientation.Horizontal:
                        self.setColumnWidth(self.header_cell_being_resized, size)
                        self.dataframe_viewer.dataView.setColumnWidth(self.header_cell_being_resized, size)
                    if self.orientation == Qt.Orientation.Vertical:
                        self.setRowHeight(self.header_cell_being_resized, size)
                        self.dataframe_viewer.dataView.setRowHeight(self.header_cell_being_resized, size)

                    self.updateGeometry()
                    self.dataframe_viewer.dataView.updateGeometry()
                return True
            elif self.header_being_resized:
                if self.orientation == Qt.Orientation.Horizontal:
                    size = orthogonal_mouse_position - self.geometry().top()
                    self.setFixedHeight(max(size, self.initial_size.height()))
                if self.orientation == Qt.Orientation.Vertical:
                    size = orthogonal_mouse_position - self.geometry().left()
                    self.setFixedWidth(max(size, self.initial_size.width()))

                self.updateGeometry()
                self.dataframe_viewer.dataView.updateGeometry()
                return True
        return False

    # Return the size of the header needed to match the corresponding DataTableView
    def sizeHint(self):
        # Columm headers
        if self.orientation == Qt.Orientation.Horizontal:
            # Width of DataTableView
            width = self.table.sizeHint().width() + self.verticalHeader().width()
            # Height
            # height = 2 * self.frameWidth()  # Account for border & padding
            # for i in range(self.model().rowCount()):
            #     height += self.rowHeight
            height = 24 * self.model().rowCount()

        # Index header
        else:
            # Height of DataTableView
            height = self.table.sizeHint().height() + self.horizontalHeader().height()
            # Width
            width = 2 * self.frameWidth()  # Account for border & padding
            for i in range(self.model().columnCount()):
                width += max(self.columnWidth(i), 100)
        return QtCore.QSize(width, height)

    # This is needed because otherwise when the horizontal header is a single row it will add whitespace to be bigger
    def minimumSizeHint(self):
        if self.orientation == Qt.Orientation.Horizontal:
            return QtCore.QSize(0, self.sizeHint().height())
        else:
            return QtCore.QSize(self.sizeHint().width(), 0)


class HeaderNamesModel(QtCore.QAbstractTableModel):
    def __init__(self, parent, orientation):
        super().__init__(parent)
        self.orientation = orientation
        self.pgdf = parent.pgdf
        self.prefix = ""

    def columnCount(self, parent=None):
        if self.orientation == Qt.Orientation.Horizontal:
            return 1
        elif self.orientation == Qt.Orientation.Vertical:
            return self.pgdf.df.index.nlevels

    def rowCount(self, parent=None):
        if self.orientation == Qt.Orientation.Horizontal:
            return self.pgdf.df.columns.nlevels
        elif self.orientation == Qt.Orientation.Vertical:
            return 1

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        row = index.row()
        col = index.column()

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if self.orientation == Qt.Orientation.Horizontal:
                val = self.pgdf.df.columns.names[row]
                if val is None:
                    val = ""

                val = val[len(self.prefix) :] if val.startswith(self.prefix) else val
                return str(val)

            elif self.orientation == Qt.Orientation.Vertical:
                val = self.pgdf.df.index.names[col]
                if val is None:
                    val = "Index"
                return str(val)

        elif role == QtCore.Qt.ItemDataRole.DecorationRole:
            if self.pgdf.sort_state == "Asc":
                icon = QtGui.QIcon(":/sort-ascending.png")
            elif self.pgdf.sort_state == "Desc":
                icon = QtGui.QIcon(":/sort-descending.png")
            else:
                return

            if col == self.pgdf.sorted_index_level and self.orientation == Qt.Orientation.Vertical:
                return icon

        elif role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)


class HeaderNamesView(QtWidgets.QTableView):
    def __init__(self, parent, orientation):
        super().__init__(parent)
        self.dataframe_viewer = parent
        self.pgdf = parent.pgdf

        self.setProperty("orientation", "horizontal" if orientation == 1 else "vertical")  # Used in stylesheet

        # Setup
        self.orientation = orientation
        self.setModel(HeaderNamesModel(parent, orientation))

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)

        self.horizontalHeader().hide()
        self.verticalHeader().hide()

        self.setSelectionMode(self.SelectionMode.NoSelection)

        # Automatically stretch rows/columns as widget is resized
        if self.orientation == Qt.Orientation.Horizontal:
            self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        else:
            self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        font = QtGui.QFont()
        font.fromString(MONOSPACE_FONT)
        font.setBold(True)
        self.setFont(font)
        self.init_size()

    def mouseDoubleClickEvent(self, event):
        point = event.pos()
        ix = self.indexAt(point)
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.orientation == Qt.Orientation.Vertical:
                self.pgdf.sort_index(ix.column())
        else:
            super().mouseDoubleClickEvent(event)

    def init_size(self):
        # Match vertical header name widths to vertical header
        if self.orientation == Qt.Orientation.Vertical:
            for ix in range(self.model().columnCount()):
                self.setColumnWidth(ix, self.columnWidth(ix))

    def sizeHint(self):
        if self.orientation == Qt.Orientation.Horizontal:
            width = self.columnWidth(0)
            height = 24
        else:  # Vertical
            width = self.dataframe_viewer.indexHeader.sizeHint().width()
            height = 24

        return QtCore.QSize(width, height)

    def minimumSizeHint(self):
        return self.sizeHint()

    def rowHeight(self, row: int) -> int:
        return self.dataframe_viewer.columnHeader.rowHeight(row)

    def columnWidth(self, column: int) -> int:
        if self.orientation == Qt.Orientation.Horizontal:
            if all(name is None for name in self.pgdf.df.columns.names):
                return 0
            else:
                return super().columnWidth(column)
        else:
            return self.dataframe_viewer.indexHeader.columnWidth(column)


class ColumnMenu(QtWidgets.QMenu):
    def __init__(self, pgdf, column_ix, parent=None):
        super().__init__(parent)

        self.pgdf = pgdf
        self.column_ix = column_ix

        ########################
        # Info
        idx = self.pgdf.dataframe_viewer.columnHeader.model().index(0, column_ix)
        self.name = self.pgdf.dataframe_viewer.columnHeader.model().data(idx, role=QtCore.Qt.ItemDataRole.DisplayRole)
        label = QtWidgets.QLabel(self.name)
        font = QtGui.QFont()
        font.setBold(True)
        label.setFont(font)
        self.add_widget(label)

        self.addSeparator()

        ########################
        # Sorting

        self.add_widget(QtWidgets.QLabel("Set sorting"))

        def select_button():
            self.sort_b1.setDown(self.pgdf.sort_state == "Asc" and self.pgdf.sorted_column_ix == column_ix)
            self.sort_b2.setDown(self.pgdf.sort_state == "Desc" and self.pgdf.sorted_column_ix == column_ix)
            self.sort_b3.setDown(self.pgdf.sort_state == "None" or self.pgdf.sorted_column_ix != column_ix)

        self.sort_b1 = QtWidgets.QPushButton("Asc")
        self.sort_b1.clicked.connect(lambda: [self.pgdf.sort_column(self.column_ix, "Asc"), select_button()])

        self.sort_b2 = QtWidgets.QPushButton("Desc")
        self.sort_b2.clicked.connect(lambda: [self.pgdf.sort_column(self.column_ix, "Desc"), select_button()])

        self.sort_b3 = QtWidgets.QPushButton("None")
        self.sort_b3.clicked.connect(lambda: [self.pgdf.sort_column(self.column_ix, "None"), select_button()])

        select_button()

        sort_control = QtWidgets.QWidget()
        sort_control_layout = QtWidgets.QHBoxLayout()
        sort_control_layout.setSpacing(0)
        sort_control_layout.setContentsMargins(0, 0, 0, 0)
        sort_control.setLayout(sort_control_layout)
        [sort_control_layout.addWidget(w) for w in [self.sort_b1, self.sort_b2, self.sort_b3]]

        self.add_widget(sort_control)

        self.addSeparator()

        button = QtWidgets.QPushButton("Edit ranges")
        button.clicked.connect(self.edit_ranges)
        self.add_widget(button)

        self.addSeparator()

        action = self.addAction("Automatic set columns width")
        action.triggered.connect(self.automatic_columns_width)

        ########################
        # Move
        #
        # col_name = self.pgdf.df.columns[column_ix]
        # self.move_b1 = QtWidgets.QPushButton("<<")
        # self.move_b1.clicked.connect(lambda: [self.pgdf.move_column(column_ix, 0),
        #                                       self.close(), self.pgdf.dataframe_viewer.show_column_menu(col_name)])
        # self.move_b2 = QtWidgets.QPushButton("<")
        # self.move_b2.clicked.connect(lambda: [self.pgdf.move_column(column_ix, column_ix - 1),
        #                                       self.close(), self.pgdf.dataframe_viewer.show_column_menu(col_name)])
        # self.move_b3 = QtWidgets.QPushButton(">")
        # self.move_b3.clicked.connect(lambda: [self.pgdf.move_column(column_ix, column_ix + 1),
        #                                       self.close(), self.pgdf.dataframe_viewer.show_column_menu(col_name)])
        # self.move_b4 = QtWidgets.QPushButton(">>")
        # self.move_b4.clicked.connect(lambda: [self.pgdf.move_column(column_ix, len(df.columns)),
        #                                       self.close(), self.pgdf.dataframe_viewer.show_column_menu(col_name)])
        #
        # move_control = QtWidgets.QWidget()
        # move_control_layout = QtWidgets.QHBoxLayout()
        # move_control_layout.setSpacing(0)
        # move_control_layout.setContentsMargins(0, 0, 0, 0)
        # move_control.setLayout(move_control_layout)
        # [move_control_layout.addWidget(w) for w in [self.move_b1, self.move_b2, self.move_b3, self.move_b4]]
        #
        # self.add_widget(move_control)

        ########################
        # Delete

        # button = QtWidgets.QPushButton("Delete Column")
        # button.clicked.connect(lambda: [self.pgdf.delete_column(column_ix), self.close()])
        # self.add_widget(button)

        # ########################
        # # Parse Date
        #
        # button = QtWidgets.QPushButton("Parse Date")
        # button.clicked.connect(lambda: [self.pgdf.parse_date(column_ix), self.close()])
        # self.add_widget(button)

        # ########################
        # # Data Type
        # col_type = self.pgdf.df_unfiltered.dtypes[column_ix]
        # types = list(dict.fromkeys([str(col_type)] + ['float', 'bool', 'category', 'str']))
        # combo = QtWidgets.QComboBox()
        # combo.addItems(types)
        # combo.setCurrentText(str(col_type))
        # combo.currentTextChanged.connect(lambda x: [self.pgdf.change_column_type(column_ix, x)])
        # self.add_widget(combo)

        # ########################
        # # Coloring
        # self.add_action("Color by None",
        #                 lambda: [setattr(self.pgdf.dataframe_viewer, 'color_mode', None),
        #                          self.pgdf.dataframe_viewer.refresh_ui()]
        #                 )
        #
        # self.add_action("Color by columns",
        #                 lambda: [setattr(self.pgdf.dataframe_viewer, 'color_mode', 'column'),
        #                          self.pgdf.dataframe_viewer.refresh_ui()]
        #                 )
        #
        # self.add_action("Color by rows",
        #                 lambda: [setattr(self.pgdf.dataframe_viewer, 'color_mode', 'row'),
        #                          self.pgdf.dataframe_viewer.refresh_ui()]
        #                 )
        #
        # self.add_action("Color by all",
        #                 lambda: [setattr(self.pgdf.dataframe_viewer, 'color_mode', 'all'),
        #                          self.pgdf.dataframe_viewer.refresh_ui()]
        #                 )

    def edit_ranges(self, *args):
        self.hide()
        self.pgdf.tabular.edit_ranges(self.column_ix, self.name)

        self.close()

    def add_action(self, text, function):
        action = QtGui.QAction(text, self)
        action.triggered.connect(function)
        self.addAction(action)

    def add_widget(self, widget):
        # https://stackoverflow.com/questions/55086498/highlighting-custom-qwidgetaction-on-hover
        widget.setMouseTracking(True)

        custom_action = QtWidgets.QWidgetAction(self)
        widget.setStyleSheet(widget.styleSheet() + "margin: 5px;")
        custom_action.setDefaultWidget(widget)
        self.addAction(custom_action)

    def show_menu(self, point):
        self.move(point)
        self.show()

    def automatic_columns_width(self, *args):
        self.pgdf.dataframe_viewer.auto_size_header()


class TabularBase(Ui_TabularDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.Signal(list)
    timestamp_changed_signal = QtCore.Signal(object, float)

    def __init__(self, df, ranges=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        if not ranges:
            self.ranges = {name: [] for name in df.columns}
        else:
            self.ranges = {}

            for name, ranges_ in ranges.items():
                for range_info in ranges_:
                    range_info["font_color"] = fn.mkBrush(range_info["font_color"])
                    range_info["background_color"] = fn.mkBrush(range_info["background_color"])
                self.ranges[name] = ranges_

        df = DataFrameStorage(df, self)

        self.tree = DataFrameViewer(df)

        self.tree.dataView.selectionModel().currentChanged.connect(self.current_changed)
        self.tree.dataView.add_channels_request.connect(self.add_channels_request)

        self.horizontalLayout.insertWidget(0, self.tree)

        self.add_filter_btn.clicked.connect(self.add_filter)
        self.apply_filters_btn.clicked.connect(self.apply_filters)
        self.time_as_date.stateChanged.connect(self.time_as_date_changed)
        self.remove_prefix.stateChanged.connect(self.remove_prefix_changed)
        self.format_selection.currentTextChanged.connect(self.set_format)

        self.toggle_filters_btn.clicked.connect(self.toggle_filters)
        self.filters_group.setHidden(True)

        self.float_precision.addItems(["Full float precision"] + [f"{i} float decimals" for i in range(16)])
        self.float_precision.setCurrentIndex(0)
        self.float_precision.currentIndexChanged.connect(self.float_precision_changed)

        self._timestamps = None

        # self.show()

        self.tree.auto_size_header()

    def float_precision_changed(self, index):
        self.tree.dataView.model().float_precision = index - 1
        self.tree.pgdf.data_changed()

    def current_changed(self, current, previous):
        if current.isValid():
            row = current.row()
            self._filtered_ts_series = self._original_ts_series.reindex(self.tree.pgdf.df.index)
            ts = float(self._filtered_ts_series.iloc[row])
            self.timestamp_changed_signal.emit(self, ts)

    def toggle_filters(self, event=None):
        if self.toggle_filters_btn.text() == "Show filters":
            self.toggle_filters_btn.setText("Hide filters")
            self.filters_group.setHidden(False)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/up.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            self.toggle_filters_btn.setIcon(icon)
        else:
            self.toggle_filters_btn.setText("Show filters")
            self.filters_group.setHidden(True)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/down.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            self.toggle_filters_btn.setIcon(icon)

    def add_filter(self, event=None):
        filter_widget = TabularFilter(
            [
                (
                    self.tree.pgdf.df_unfiltered.index.name,
                    self.tree.pgdf.df_unfiltered.index.values.dtype.kind,
                    0,
                    False,
                )
            ]
            + [
                (
                    name,
                    self.tree.pgdf.df_unfiltered[name].values.dtype.kind,
                    self.signals_descr[name],
                )
                for name in self.tree.pgdf.df_unfiltered.columns
            ],
            self.format_selection.currentText(),
        )

        item = QtWidgets.QListWidgetItem(self.filters)
        item.setSizeHint(filter_widget.sizeHint())
        self.filters.addItem(item)
        self.filters.setItemWidget(item, filter_widget)

    def apply_filters(self, event=None):
        df = self.tree.pgdf.df_unfiltered.copy()

        friendly_names = {name: pandas_query_compatible(name) for name in df.columns}

        original_names = {val: key for key, val in friendly_names.items()}

        df.rename(columns=friendly_names, inplace=True)

        filters = []
        count = self.filters.count()

        for i in range(count):
            filter = self.filters.itemWidget(self.filters.item(i))
            if filter.enabled.checkState() == QtCore.Qt.CheckState.Unchecked:
                continue

            target = filter._target
            if target is None:
                continue

            if filters:
                filters.append(filter.relation.currentText().lower())

            column_name = filter.column.currentText()
            if column_name == df.index.name:
                is_byte_array = False
            else:
                is_byte_array = self.signals_descr[column_name]
            column_name = pandas_query_compatible(column_name)
            op = filter.op.currentText()

            if target != target:  # noqa: PLR0124
                # target is NaN
                _nan = np.nan  # used in pandas query
                if op in (">", ">=", "<", "<="):
                    filters.extend((column_name, op, "@_nan"))
                elif op == "!=":
                    filters.extend((column_name, "==", column_name))
                elif op == "==":
                    filters.extend((column_name, "!=", column_name))
            else:
                if column_name == "timestamps" and df["timestamps"].dtype.kind == "M":
                    ts = pd.Timestamp(target, tz=LOCAL_TIMEZONE)
                    _ts = ts.tz_convert("UTC").to_datetime64()  # used in pandas query

                    filters.extend((column_name, op, "@_ts"))

                elif is_byte_array:
                    target = str(target).replace(" ", "").strip('"')

                    if f"{column_name}__as__bytes" not in df.columns:
                        df[f"{column_name}__as__bytes"] = pd.Series([bytes(s) for s in df[column_name]], index=df.index)
                    _val = bytes.fromhex(target)  # used in pandas query

                    filters.extend((f"{column_name}__as__bytes", op, "@_val"))

                else:
                    filters.extend((column_name, op, str(target)))

        if filters:
            try:
                new_df = df.query(" ".join(filters))
            except:
                logger.exception(f'Failed to apply filter for tabular window: {" ".join(filters)}')
                self.query.setText(format_exc())
            else:
                to_drop = [name for name in df.columns if name.endswith("__as__bytes")]
                if to_drop:
                    df.drop(columns=to_drop, inplace=True)
                    new_df.drop(columns=to_drop, inplace=True)
                self.query.setText(" ".join(filters))
                new_df.rename(columns=original_names, inplace=True)
                self.tree.pgdf.df = new_df
                self.tree.pgdf.data_changed()
        else:
            self.query.setText("")
            df.rename(columns=original_names, inplace=True)
            self.tree.pgdf.df = df
            self.tree.pgdf.data_changed()

        self.tree.pgdf.df_unfiltered.rename(columns=original_names, inplace=True)

    def add_new_channels(self, signals, mime_data=None):
        if len(self.tree.pgdf.df_unfiltered) != len(self.tree.pgdf.df):
            filtered = True
        else:
            filtered = False

        index = pd.Series(np.arange(len(signals), dtype="u8"), index=signals.index)
        signals["Index"] = index

        signals.set_index(index, inplace=True)
        dropped = {}

        ranges = dict(zip(self.tree.pgdf.df_unfiltered.columns, self.ranges.values()))

        for name_ in signals.columns:
            col = signals[name_]
            if col.dtype.kind == "O":
                if name_.endswith("DataBytes"):
                    try:
                        sizes = signals[name_.replace("DataBytes", "DataLength")]
                    except:
                        sizes = None
                    dropped[name_] = pd.Series(
                        csv_bytearray2hex(
                            col,
                            sizes,
                        ),
                        index=signals.index,
                    )

                elif name_.endswith("Data Bytes"):
                    try:
                        sizes = signals[name_.replace("Data Bytes", "Data Length")]
                    except:
                        sizes = None
                    dropped[name_] = pd.Series(
                        csv_bytearray2hex(
                            col,
                            sizes,
                        ),
                        index=signals.index,
                    )

                elif col.dtype.name != "category":
                    try:
                        dropped[name_] = pd.Series(csv_bytearray2hex(col), index=signals.index)
                    except:
                        pass

                self.signals_descr[name_] = 0

            elif col.dtype.kind == "S":
                try:
                    dropped[name_] = pd.Series(npchar.decode(col, "utf-8"), index=signals.index)
                except:
                    dropped[name_] = pd.Series(npchar.decode(col, "latin-1"), index=signals.index)
                self.signals_descr[name_] = 0
            else:
                self.signals_descr[name_] = 0

            ranges[name_] = []

        signals = signals.drop(columns=["Index", *list(dropped)])
        for name, s in dropped.items():
            signals[name] = s

        names = list(signals.columns)
        names = [
            *[name for name in names if name.endswith((".ID", ".DataBytes"))],
            *[name for name in names if name != "timestamps" and not name.endswith((".ID", ".DataBytes"))],
        ]
        signals = signals[names]

        self.tree.pgdf.df_unfiltered = self.tree.pgdf.df = pd.concat([self.tree.pgdf.df_unfiltered, signals], axis=1)
        self.ranges = ranges

        if filtered:
            self.apply_filters()
        else:
            self.tree.pgdf.data_changed()

        self.tree.auto_size_header()
        self.tree.update_horizontal_scroll()

    def to_config(self):
        count = self.filters.count()

        pattern = self.pattern
        if pattern:
            ranges = copy_ranges(pattern["ranges"])

            for range_info in ranges:
                range_info["font_color"] = range_info["font_color"].color().name()
                range_info["background_color"] = range_info["background_color"].color().name()

            pattern["ranges"] = ranges

        ranges = {}
        for name, channel_ranges in self.ranges.items():
            channel_ranges = copy_ranges(channel_ranges)

            for range_info in channel_ranges:
                range_info["font_color"] = range_info["font_color"].color().name()
                range_info["background_color"] = range_info["background_color"].color().name()

            ranges[name] = channel_ranges

        config = {
            "sorted": True,
            "channels": list(self.tree.pgdf.df_unfiltered.columns) if not self.pattern else [],
            "filtered": bool(self.query.toPlainText()),
            "filters": (
                [self.filters.itemWidget(self.filters.item(i)).to_config() for i in range(count)]
                if not self.pattern
                else []
            ),
            "time_as_date": self.time_as_date.checkState() == QtCore.Qt.CheckState.Checked,
            "pattern": pattern,
            "format": self.format,
            "ranges": ranges,
            "header_sections_width": [
                self.tree.columnHeader.horizontalHeader().sectionSize(i)
                for i in range(self.tree.columnHeader.horizontalHeader().count())
            ],
            "font_size": self.tree.dataView.font().pointSize(),
        }

        return config

    def time_as_date_changed(self, state):
        s = self.start
        count = self.filters.count()

        if state == QtCore.Qt.CheckState.Checked:
            for i in range(count):
                filter = self.filters.itemWidget(self.filters.item(i))
                filter.dtype_kind[0] = "M"

                if filter.column.currentIndex() == 0:
                    filter.column_changed(0)
                else:
                    filter.validate_target()

            delta = pd.to_timedelta(self.tree.pgdf.df_unfiltered["timestamps"], unit="s")

            timestamps = self.start + delta
            self.tree.pgdf.df_unfiltered["timestamps"] = timestamps
        else:
            for i in range(count):
                filter = self.filters.itemWidget(self.filters.item(i))
                filter.dtype_kind[0] = "f"

                if filter.column.currentIndex() == 0:
                    filter.column_changed(0)
                else:
                    filter.validate_target()
            self.tree.pgdf.df_unfiltered["timestamps"] = self._original_timestamps

        if self.query.toPlainText():
            self.apply_filters()

        self.tree.pgdf.data_changed()
        self.tree.auto_size_column(0)

    def remove_prefix_changed(self, state):
        if state == QtCore.Qt.CheckState.Checked:
            self.prefix.setEnabled(True)

            self.tree.columnHeaderNames.model().prefix = self.prefix.currentText()
            self.tree.columnHeader.model().prefix = self.prefix.currentText()
            self.tree.prefix = self.prefix.currentText()

        else:
            self.prefix.setEnabled(False)

            self.tree.columnHeaderNames.model().prefix = ""
            self.tree.columnHeader.model().prefix = ""
            self.tree.prefix = ""

        self.tree.auto_size_header()

    def prefix_changed(self, index):
        self.remove_prefix_changed(QtCore.Qt.CheckState.Checked)

    def open_menu(self, position):
        menu = QtWidgets.QMenu()

        menu.addAction(self.tr("Export to CSV"))

        action = menu.exec_(self.tree.viewport().mapToGlobal(position))

        if action is None:
            return

        if action.text() == "Export to CSV":
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select output CSV file",
                "",
                "CSV (*.csv)",
            )

            if file_name:
                self.progress = 0, 0
                progress = QtWidgets.QProgressDialog(f'Data export to CSV file "{file_name}"', "", 0, 0, self.parent())

                progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
                progress.setCancelButton(None)
                progress.setAutoClose(True)
                progress.setWindowTitle("Export tabular window to CSV")
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(":/csv.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
                progress.setWindowIcon(icon)
                progress.show()

                target = self.tree.pgdf.df_unfiltered.to_csv
                kwargs = {
                    "path_or_buf": file_name,
                    "index_label": "Index",
                    "date_format": "%Y-%m-%d %H:%M:%S.%f%z",
                }

                result = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=0,
                    offset=0,
                    progress=progress,
                )

                progress.cancel()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if (
            key in (QtCore.Qt.Key.Key_H, QtCore.Qt.Key.Key_B, QtCore.Qt.Key.Key_P)
            and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            event.accept()
            if key == QtCore.Qt.Key.Key_H:
                self.format_selection.setCurrentText("hex")
            elif key == QtCore.Qt.Key.Key_B:
                self.format_selection.setCurrentText("bin")
            else:
                self.format_selection.setCurrentText("phys")

        elif key == QtCore.Qt.Key.Key_S and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.accept()
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save as measurement file",
                "",
                "MDF version 4 files (*.mf4)",
            )

            if file_name:
                with mdf_module.MDF() as mdf:
                    mdf.append(self.tree.pgdf.df_unfiltered)
                    mdf.save(file_name, overwrite=True)

        elif key == QtCore.Qt.Key.Key_BracketLeft and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.accept()
            self.decrease_font()

        elif key == QtCore.Qt.Key.Key_BracketRight and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.accept()
            self.increase_font()

        elif key == QtCore.Qt.Key.Key_G and modifiers == QtCore.Qt.KeyboardModifier.ShiftModifier:
            event.accept()
            value, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Go to time stamp",
                "Time stamp",
                decimals=9,
            )

            if ok:
                self.set_timestamp(value)

        else:
            self.tree.dataView.keyPressEvent(event)

    def set_format(self, fmt):
        self.format = fmt

        self.tree.dataView.model().format = fmt
        self.tree.pgdf.data_changed()
        self._settings.setValue("tabular_format", fmt)

        for row in range(self.filters.count()):
            filter = self.filters.itemWidget(self.filters.item(row))
            filter.int_format = fmt
            filter.validate_target()

        if self.query.toPlainText():
            self.apply_filters()

    def set_timestamp(self, stamp):
        self._filtered_ts_series = self._original_ts_series.reindex(self.tree.pgdf.df.index)

        if not len(self._filtered_ts_series):
            return

        if not (self._filtered_ts_series.iloc[0] <= stamp <= self._filtered_ts_series.iloc[-1]):
            return

        idx = self._filtered_ts_series.searchsorted(stamp, side="right") - 1
        if idx < 0:
            idx = 0

        self.tree.dataView.selectRow(idx)

    def edit_ranges(self, index, name):
        if index >= 0:
            original_name = self.tree.pgdf.df_unfiltered.columns[index]

            dlg = RangeEditor(name, "", self.ranges[original_name], parent=self, brush=True)
            dlg.exec_()
            if dlg.pressed_button == "apply":
                ranges = dlg.result
                self.ranges[original_name] = ranges

    def decrease_font(self):
        font = self.tree.dataView.font()
        size = font.pointSize()
        pos = bisect.bisect_left(FONT_SIZE, size) - 1
        if pos < 0:
            pos = 0
        new_size = FONT_SIZE[pos]

        self.set_font_size(new_size)

    def increase_font(self):
        font = self.tree.dataView.font()
        size = font.pointSize()
        pos = bisect.bisect_right(FONT_SIZE, size)
        if pos == len(FONT_SIZE):
            pos -= 1
        new_size = FONT_SIZE[pos]

        self.set_font_size(new_size)

    def set_font_size(self, size):
        self.hide()
        font = self.tree.dataView.font()
        font.setPointSize(size)
        self.tree.dataView.setFont(font)
        font.setBold(True)
        self.tree.indexHeader.setFont(font)
        self.tree.indexHeaderNames.setFont(font)
        self.tree.columnHeader.setFont(font)
        self.tree.columnHeaderNames.setFont(font)
        self.show()

        self.tree.default_row_height = 12 + size
        self.tree.set_styles()


class DataFrameViewer(QtWidgets.QWidget):
    def __init__(self, pgdf):
        super().__init__()

        global MONOSPACE_FONT

        if MONOSPACE_FONT is None:
            families = QtGui.QFontDatabase().families()
            for family in (
                "Consolas",
                "Liberation Mono",
                "DejaVu Sans Mono",
                "Droid Sans Mono",
                "Liberation Mono",
                "Roboto Mono",
                "Monaco",
                "Courier",
            ):
                if family in families:
                    MONOSPACE_FONT = f"{family},9,-1,5,400,0,0,0,0,0,0,0,0,0,0,1,Regular"
                    break

        pgdf.dataframe_viewer = self
        self.pgdf = pgdf
        self.prefix = ""

        # Local state
        # How to color cells
        self.color_mode = None

        # Set up DataFrame TableView and Model
        self.dataView = DataTableView(parent=self)

        # Create headers
        self.columnHeader = HeaderView(parent=self, orientation=Qt.Orientation.Horizontal)
        self.indexHeader = HeaderView(parent=self, orientation=Qt.Orientation.Vertical)

        self.columnHeaderNames = HeaderNamesView(parent=self, orientation=Qt.Orientation.Horizontal)
        self.indexHeaderNames = HeaderNamesView(parent=self, orientation=Qt.Orientation.Vertical)

        # Set up layout
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.setLayout(self.gridLayout)

        # Linking scrollbars
        # Scrolling in data table also scrolls the headers
        self.dataView.horizontalScrollBar().valueChanged.connect(self.columnHeader.horizontalScrollBar().setValue)
        self.dataView.horizontalScrollBar().valueChanged.connect(self.columnHeaderNames.horizontalScrollBar().setValue)
        self.dataView.verticalScrollBar().valueChanged.connect(self.indexHeader.verticalScrollBar().setValue)
        # Scrolling in headers also scrolls the data table
        self.columnHeader.horizontalScrollBar().valueChanged.connect(self.dataView.horizontalScrollBar().setValue)
        self.indexHeader.verticalScrollBar().valueChanged.connect(self.dataView.verticalScrollBar().setValue)
        # Turn off default scrollbars
        self.dataView.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.dataView.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Disable scrolling on the headers. Even though the scrollbars are hidden, scrolling by dragging desyncs them
        self.indexHeader.horizontalScrollBar().valueChanged.connect(lambda: None)

        class CornerWidget(QtWidgets.QWidget):
            def __init__(self):
                super().__init__()
                # https://stackoverflow.com/questions/32313469/stylesheet-in-pyside-not-working
                self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)

        self.corner_widget = CornerWidget()
        self.corner_widget.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        )
        # Add items to grid layout
        # self.gridLayout.addWidget(self.corner_widget, 0, 0)
        self.gridLayout.addWidget(self.columnHeader, 0, 1, 2, 2, Qt.AlignmentFlag.AlignTop)
        self.gridLayout.addWidget(self.columnHeaderNames, 0, 3, 2, 1)
        self.gridLayout.addWidget(self.indexHeader, 2, 0, 2, 2, Qt.AlignmentFlag.AlignLeft)
        self.gridLayout.addWidget(self.indexHeaderNames, 1, 0, 1, 1, Qt.AlignmentFlag.AlignBottom)
        self.gridLayout.addWidget(self.dataView, 3, 2, 1, 1)
        self.gridLayout.addWidget(self.dataView.horizontalScrollBar(), 4, 2, 1, 1)
        self.gridLayout.addWidget(self.dataView.verticalScrollBar(), 3, 3, 1, 1)

        # Fix scrollbars forcing a minimum height of the dataView which breaks layout for small number of rows
        self.dataView.verticalScrollBar().setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Ignored)
        )
        self.dataView.horizontalScrollBar().setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        )

        # These expand when the window is enlarged instead of having the grid squares spread out
        # self.gridLayout.setColumnStretch(4, 1)
        # self.gridLayout.setRowStretch(5, 1)

        self.gridLayout.setColumnStretch(2, 1)
        self.gridLayout.setRowStretch(5, 1)

        #
        # self.gridLayout.addItem(QtWidgets.QSpacerItem(0, 0,
        #                                               QtWidgets.QSizePolicy.Policy.Expanding,
        #                                               QtWidgets.QSizePolicy.Policy.Expanding), 0, 0, 1, 1, )

        self.default_row_height = 24
        self.set_styles()
        self.indexHeader.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Maximum
        )
        self.columnHeader.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.MinimumExpanding
        )

        # Set column widths
        for column_index in range(self.columnHeader.model().columnCount()):
            self.auto_size_column(column_index)

        self.columnHeader.horizontalHeader().setStretchLastSection(True)
        self.columnHeaderNames.horizontalHeader().setStretchLastSection(True)

        self.columnHeader.horizontalHeader().sectionResized.connect(self.update_horizontal_scroll)

        self.columnHeader.horizontalHeader().setMinimumSectionSize(1)
        self.dataView.horizontalHeader().setMinimumSectionSize(1)

        # self.show()

        self.auto_size_header()

    def set_styles(self):
        for item in [
            self.dataView,
            self.columnHeader,
            self.indexHeader,
            self.indexHeaderNames,
            self.columnHeaderNames,
        ]:
            item.setContentsMargins(0, 0, 0, 0)
            # item.setItemDelegate(NoFocusDelegate())

        self.indexHeaderNames.verticalHeader().setDefaultSectionSize(self.default_row_height)
        self.indexHeaderNames.verticalHeader().setMinimumSectionSize(self.default_row_height)
        self.indexHeaderNames.verticalHeader().setMaximumSectionSize(self.default_row_height)
        self.indexHeaderNames.verticalHeader().sectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.indexHeader.verticalHeader().setDefaultSectionSize(self.default_row_height)
        self.indexHeader.verticalHeader().setMinimumSectionSize(self.default_row_height)
        self.indexHeader.verticalHeader().setMaximumSectionSize(self.default_row_height)
        self.indexHeader.verticalHeader().sectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.dataView.verticalHeader().setDefaultSectionSize(self.default_row_height)
        self.dataView.verticalHeader().setMinimumSectionSize(self.default_row_height)
        self.dataView.verticalHeader().setMaximumSectionSize(self.default_row_height)
        self.dataView.verticalHeader().sectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.columnHeader.verticalHeader().setDefaultSectionSize(self.default_row_height)
        self.columnHeader.verticalHeader().setMinimumSectionSize(self.default_row_height)
        self.columnHeader.verticalHeader().setMaximumSectionSize(self.default_row_height)
        self.columnHeader.verticalHeader().sectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)

    def __reduce__(self):
        # This is so dataclasses.asdict doesn't complain about this being unpicklable
        return "DataFrameViewer"

    def auto_size_header(self):
        s = 0
        for i in range(self.columnHeader.model().columnCount()):
            s += self.auto_size_column(i)

        delta = int((self.dataView.size().width() - s) // len(self.pgdf.df.columns))

        if delta > 0:
            for i in range(self.columnHeader.model().columnCount()):
                self.auto_size_column(i, extra_padding=delta)
            self.dataView.horizontalScrollBar().hide()
        else:
            self.dataView.horizontalScrollBar().show()

    def update_horizontal_scroll(self, *args):
        s = 0
        for i in range(self.columnHeader.model().columnCount()):
            s += self.columnHeader.columnWidth(i) + self.columnHeader.frameWidth()

        if self.dataView.size().width() < s:
            self.dataView.horizontalScrollBar().show()
        else:
            self.dataView.horizontalScrollBar().hide()

    def auto_size_column(self, column_index, extra_padding=0):
        """
        Set the size of column at column_index to fit its contents
        """

        width = 0

        # Iterate over the data view rows and check the width of each to determine the max width for the column
        # Only check the first N rows for performance. If there is larger content in cells below it will be cut off
        N = 100
        for i in range(self.dataView.model().rowCount())[:N]:
            mi = self.dataView.model().index(i, column_index)
            text = self.dataView.model().data(mi)
            w = self.dataView.fontMetrics().boundingRect(text).width()
            width = max(width, w)

        # Repeat for header cells
        for i in range(self.columnHeader.model().rowCount()):
            mi = self.columnHeader.model().index(i, column_index)
            text = self.columnHeader.model().data(mi)
            text = text[len(self.prefix) :] if text.startswith(self.prefix) else text
            w = self.columnHeader.fontMetrics().boundingRect(text).width()
            width = max(width, w)

        padding = 20
        width += padding + extra_padding

        self.columnHeader.setColumnWidth(column_index, width)
        self.dataView.setColumnWidth(column_index, width)

        self.dataView.updateGeometry()
        self.columnHeader.updateGeometry()

        return width

    def auto_size_row(self, row_index):
        """
        Set the size of row at row_index to fix its contents
        """
        height = 24

        self.indexHeader.setRowHeight(row_index, height)
        self.dataView.setRowHeight(row_index, height)

        self.dataView.updateGeometry()
        self.indexHeader.updateGeometry()

    def scroll_to_column(self, column=0):
        index = self.dataView.model().index(0, column)
        self.dataView.scrollTo(index)
        self.columnHeader.selectColumn(column)
        self.columnHeader.on_selectionChanged(force=True)

    def keyPressEvent(self, event):
        QtWidgets.QWidget.keyPressEvent(self, event)
        mods = event.modifiers()

        if event.key() == Qt.Key.Key_C and (mods & Qt.KeyboardModifier.ControlModifier):
            event.accept()
            self.copy(header=True)

        elif (
            event.key() == Qt.Key.Key_C
            and (mods & Qt.KeyboardModifier.ShiftModifier)
            and (mods & Qt.KeyboardModifier.ControlModifier)
        ):
            event.accept()
            self.copy(header=True)

        else:
            self.dataView.keyPressEvent(event)

    def copy(self, header=False):
        """
        Copy the selected cells to clipboard in an Excel-pasteable format
        """
        # Get the bounds using the top left and bottom right selected cells

        fmt = self.dataView.model().format

        # Copy from data, columns, or index depending which has focus
        if header or self.dataView.hasFocus():
            indexes = self.dataView.selectionModel().selection().indexes()
            rows = [ix.row() for ix in indexes]
            cols = [ix.column() for ix in indexes]

            temp_df = self.pgdf.df
            df = temp_df.iloc[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1]

        elif self.indexHeader.hasFocus():
            indexes = self.indexHeader.selectionModel().selection().indexes()
            rows = [ix.row() for ix in indexes]
            cols = [ix.column() for ix in indexes]

            temp_df = self.pgdf.df.index.to_frame()
            df = temp_df.iloc[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1]

        elif self.columnHeader.hasFocus():
            indexes = self.columnHeader.selectionModel().selection().indexes()
            rows = [ix.row() for ix in indexes]
            cols = [ix.column() for ix in indexes]

            # Column header should be horizontal so we transpose
            temp_df = self.pgdf.df.columns.to_frame().transpose()
            df = temp_df.iloc[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1]
        else:
            return

        if fmt in ("hex", "bin") and len(df):
            fmt = "{:X}" if fmt == "hex" else "{:b}"

            for name in df.columns:
                col = df[name]
                if isinstance(col.values[0], np.integer):
                    col = pd.Series([fmt.format(val) for val in col], index=df.index)
                    df[name] = col

        if self.dataView.model().float_precision != -1:
            decimals = self.dataView.model().float_precision
            for name in df.columns:
                col = df[name]
                if isinstance(col.values[0], np.floating):
                    col = col.round(decimals)
                    df[name] = col
            float_format = f"%.{decimals}f"
        else:
            float_format = "%.16f"

        # If I try to use df.to_clipboard without starting new thread, large selections give access denied error
        if df.shape == (1, 1):
            # Special case for single-cell copy, excel=False removes the trailing \n character.
            threading.Thread(
                target=lambda df: df.to_clipboard(
                    index=header,
                    header=header,
                    excel=False,
                    float_format=float_format,
                ),
                args=(df,),
            ).start()
        else:
            threading.Thread(
                target=lambda df: df.to_clipboard(
                    index=header,
                    header=header,
                    float_format=float_format,
                ),
                args=(df,),
            ).start()

    def show_column_menu(self, column_ix_or_name):
        if isinstance(self.pgdf.df.columns, pd.MultiIndex):
            logger.info("Column menu not implemented for MultiIndex")
            return

        if type(column_ix_or_name) == str:
            column_ix = list(self.pgdf.df.columns).index(column_ix_or_name)
        else:
            column_ix = column_ix_or_name

        point = QtCore.QPoint(
            self.columnHeader.columnViewportPosition(column_ix) + self.columnHeader.columnWidth(column_ix) - 15,
            self.columnHeader.geometry().bottom() - 6,
        )

        menu = ColumnMenu(self.pgdf, column_ix, self)
        menu.show_menu(self.columnHeader.mapToGlobal(point))

    def _remove_column(self, ix):
        for model in [self.dataView.model(), self.columnHeader.model()]:
            parent = QtCore.QModelIndex()
            model.beginRemoveColumns(parent, ix, ix)
            model.endRemoveColumns()

    def _move_column(self, ix, new_ix, refresh=True):
        for view in [self.dataView, self.columnHeader]:
            model = view.model()
            column_widths = [view.columnWidth(ix) for ix in range(model.columnCount())]
            column_widths.insert(new_ix, column_widths.pop(ix))

            # Set width of destination column to the width of the source column
            for j in range(len(column_widths)):
                view.setColumnWidth(j, column_widths[j])

        if refresh:
            self.refresh_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_horizontal_scroll()

    def refresh_ui(self):
        self.models = [
            self.dataView.model(),
            self.columnHeader.model(),
            self.indexHeader.model(),
            self.columnHeaderNames.model(),
            self.indexHeaderNames.model(),
        ]

        for model in self.models:
            model.beginResetModel()
            model.endResetModel()

        # Update sizing
        for view in [self.columnHeader, self.indexHeader, self.dataView]:
            view.updateGeometry()

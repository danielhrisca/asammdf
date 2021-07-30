# -*- coding: utf-8 -*-
import datetime
import logging
from traceback import format_exc

import numpy as np
import numpy.core.defchararray as npchar
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from ...blocks.utils import (
    csv_bytearray2hex,
    csv_int2bin,
    csv_int2hex,
    pandas_query_compatible,
)
from ..ui import resource_rc as resource_rc
from ..ui.tabular import Ui_TabularDisplay
from ..utils import run_thread_with_progress
from .tabular_filter import TabularFilter

logger = logging.getLogger("asammdf.gui")
LOCAL_TIMEZONE = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo


class TabularTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, column_types, int_format, *args, **kwargs):
        self.column_types = column_types
        self.int_format = int_format
        super().__init__(*args, **kwargs)

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


class TabularBase(Ui_TabularDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.open_menu)

        self.add_filter_btn.clicked.connect(self.add_filter)
        self.apply_filters_btn.clicked.connect(self.apply_filters)
        self.sort.stateChanged.connect(self.sorting_changed)
        self.time_as_date.stateChanged.connect(self.time_as_date_changed)
        self.remove_prefix.stateChanged.connect(self.remove_prefix_changed)
        self.tree.header().sortIndicatorChanged.connect(self._sort)

        self.tree_scroll.valueChanged.connect(self._display)
        self.tree.verticalScrollBar().valueChanged.connect(self._scroll_tree)
        self.tree.currentItemChanged.connect(self._scroll_tree)
        self.format_selection.currentTextChanged.connect(self.set_format)

    def _scroll_tree(self, selected_item):
        count = self.tree.topLevelItemCount()
        if count <= 1:
            return

        if isinstance(selected_item, int):

            try:
                row = int(
                    selected_item
                    / self.tree.verticalScrollBar().maximum()
                    * (count - 1)
                )
            except:
                row = count - 1
            selected_item = self.tree.topLevelItem(row)

        first = self.tree.topLevelItem(0)
        last = self.tree.topLevelItem(count - 1)

        if (
            selected_item is first
            and self.tree_scroll.value() != self.tree_scroll.minimum()
        ):
            current_index = selected_item.text(0)
            self.tree_scroll.setValue(
                self.tree_scroll.value() - self.tree_scroll.singleStep()
            )

            item = self.tree.findItems(current_index, QtCore.Qt.MatchExactly)[0]
            above = self.tree.itemAbove(item)
            self.tree.scrollToItem(above, QtWidgets.QAbstractItemView.PositionAtTop)
            try:
                item = self.tree.itemBelow(above)
                self.tree.setCurrentItem(item)
            except:
                pass

        elif (
            selected_item is last
            and self.tree_scroll.value() != self.tree_scroll.maximum()
        ):
            current_index = selected_item.text(0)
            self.tree_scroll.setValue(
                self.tree_scroll.value() + self.tree_scroll.singleStep()
            )

            item = self.tree.findItems(current_index, QtCore.Qt.MatchExactly)[0]
            below = self.tree.itemBelow(item)
            self.tree.scrollToItem(below, QtWidgets.QAbstractItemView.PositionAtBottom)
            try:
                item = self.tree.itemAbove(below)
                self.tree.setCurrentItem(item)
            except:
                pass

    def _sort(self, index, mode):
        ascending = mode == QtCore.Qt.AscendingOrder
        names = [self.df.index.name, *self.df.columns]
        name = names[index]

        if index:
            try:
                self.df.sort_values(
                    by=[name, "timestamps"], ascending=ascending, inplace=True
                )
            except:
                pass
        else:
            self.df.sort_index(ascending=ascending, inplace=True)

        self.tree_scroll.setSliderPosition(self.tree_scroll.maximum())
        self.tree_scroll.setSliderPosition(self.tree_scroll.minimum())

    def add_filter(self, event=None):
        filter_widget = TabularFilter(
            [(self.signals.index.name, self.signals.index.values.dtype.kind, 0, False)]
            + [
                (
                    name,
                    self.signals[name].values.dtype.kind,
                    self.signals_descr[name],
                )
                for name in self.signals.columns
            ],
            self.format_selection.currentText(),
        )

        item = QtWidgets.QListWidgetItem(self.filters)
        item.setSizeHint(filter_widget.sizeHint())
        self.filters.addItem(item)
        self.filters.setItemWidget(item, filter_widget)

    def apply_filters(self, event=None):
        df = self.signals

        friendly_names = {name: pandas_query_compatible(name) for name in df.columns}

        original_names = {val: key for key, val in friendly_names.items()}

        df.rename(columns=friendly_names, inplace=True)

        filters = []
        count = self.filters.count()

        for i in range(count):
            filter = self.filters.itemWidget(self.filters.item(i))
            if filter.enabled.checkState() == QtCore.Qt.Unchecked:
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

            if target != target:
                # here we have NaN
                nan = np.nan

                if op in (">", ">=", "<", "<="):
                    filters.append(column_name)
                    filters.append(op)
                    filters.append("@nan")
                elif op == "!=":
                    filters.append(column_name)
                    filters.append("==")
                    filters.append(column_name)
                elif op == "==":
                    filters.append(column_name)
                    filters.append("!=")
                    filters.append(column_name)
            else:
                if column_name == "timestamps" and df["timestamps"].dtype.kind == "M":

                    ts = pd.Timestamp(target, tz=LOCAL_TIMEZONE)
                    ts = ts.tz_convert("UTC").to_datetime64()

                    filters.append(column_name)
                    filters.append(op)
                    filters.append("@ts")

                elif is_byte_array:
                    target = str(target).replace(" ", "").strip('"')

                    if f"{column_name}__as__bytes" not in df.columns:
                        df[f"{column_name}__as__bytes"] = pd.Series(
                            [bytes(s) for s in df[column_name]], index=df.index
                        )
                    val = bytes.fromhex(target)

                    filters.append(f"{column_name}__as__bytes")
                    filters.append(op)
                    filters.append("@val")

                else:
                    filters.append(column_name)
                    filters.append(op)
                    filters.append(str(target))

        if filters:
            try:
                new_df = df.query(" ".join(filters))
            except:
                logger.exception(
                    f'Failed to apply filter for tabular window: {" ".join(filters)}'
                )
                self.query.setText(format_exc())
            else:
                to_drop = [name for name in df.columns if name.endswith("__as__bytes")]
                if to_drop:
                    df.drop(columns=to_drop, inplace=True)
                    new_df.drop(columns=to_drop, inplace=True)
                self.query.setText(" ".join(filters))
                new_df.rename(columns=original_names, inplace=True)
                self.build(new_df)
        else:
            self.query.setText("")
            df.rename(columns=original_names, inplace=True)
            self.build(df)

        self.signals.rename(columns=original_names, inplace=True)

    def items_deleted(self, names):
        for name in names:
            self.signals.pop(name)
            self.signals_descr.pop(name)
        self.build()

    def build(self, df, reset_header_names=False):
        self.tree.setSortingEnabled(False)
        self.tree.clear()

        if self.remove_prefix.checkState() == QtCore.Qt.Checked:
            prefix = self.prefix.currentText()
            dim = len(prefix)

            dropped = {}

            for name_ in df.columns:
                if name_.startswith(prefix):
                    dropped[name_] = pd.Series(df[name_], index=df.index)

            df = df.drop(columns=list(dropped))
            for name, s in dropped.items():
                df[name[dim:]] = s

        names = ["Index", *df.columns]

        if reset_header_names:
            self.header_names = names

        self.tree.setColumnCount(len(names))
        self.tree.setHeaderLabels(names)

        self.df = df
        self.size = len(df.index)
        self.position = 0

        count = max(1, self.size // 10 + 1)

        self.tree_scroll.setMaximum(count)

        self.tree_scroll.setSliderPosition(0)

        self._display(0)

    def _display(self, position):
        self.tree.setSortingEnabled(False)
        self.tree.clear()

        df = self.df.iloc[max(0, position * 10 - 50) : max(0, position * 10 + 100)]

        if df["timestamps"].dtype.kind == "M":
            timestamps = (
                pd.Index(df["timestamps"]).tz_localize("UTC").tz_convert(LOCAL_TIMEZONE)
            )
        else:
            timestamps = df["timestamps"]

        timestamps = timestamps.astype(str)
        items = [df.index.astype(str), timestamps]

        for i, name in enumerate(df.columns[1:], 1):
            column = df[name]
            kind = column.dtype.kind

            if kind in "ui":
                if self.format == "hex":
                    items.append(pd.Series(csv_int2hex(column)).values)
                elif self.format == "bin":
                    items.append(pd.Series(csv_int2bin(column)).values)
                else:
                    items.append(column.astype(str))

            elif kind == "f":
                items.append(column.astype(str))
            elif kind == "S":
                try:
                    items.append(npchar.decode(column, "utf-8"))
                except:
                    items.append(npchar.decode(column, "latin-1"))
            elif kind == "O":
                items.append(column)
            else:
                items.append(column)

        if position == 0:
            self.tree.verticalScrollBar().setSliderPosition(0)
        elif position == self.tree_scroll.maximum():
            self.tree.verticalScrollBar().setSliderPosition(
                self.tree.verticalScrollBar().maximum()
            )

        column_types = ["u", *[df[name].dtype.kind for name in df.columns]]
        int_format = self.format_selection.currentText()

        items = [TabularTreeItem(column_types, int_format, [str(e) for e in row]) for row in zip(*items)]

        self.tree.addTopLevelItems(items)
        self.update_header()

        self.tree.setSortingEnabled(self.sort.checkState() == QtCore.Qt.Checked)

    def add_new_channels(self, signals):
        index = pd.Series(np.arange(len(signals), dtype="u8"), index=signals.index)
        signals["Index"] = index

        signals.set_index(index, inplace=True)
        dropped = {}

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
                        dropped[name_] = pd.Series(
                            csv_bytearray2hex(col), index=signals.index
                        )
                    except:
                        pass

                self.signals_descr[name_] = 0

            elif col.dtype.kind == "S":
                try:
                    dropped[name_] = pd.Series(
                        npchar.decode(col, "utf-8"), index=signals.index
                    )
                except:
                    dropped[name_] = pd.Series(
                        npchar.decode(col, "latin-1"), index=signals.index
                    )
                self.signals_descr[name_] = 0
            else:
                self.signals_descr[name_] = 0

        signals = signals.drop(columns=["Index", *list(dropped)])
        for name, s in dropped.items():
            signals[name] = s

        names = list(signals.columns)
        names = [
            *[name for name in names if name.endswith((".ID", ".DataBytes"))],
            *[
                name
                for name in names
                if name != "timestamps" and not name.endswith((".ID", ".DataBytes"))
            ],
        ]
        signals = signals[names]

        self.signals = pd.concat([self.signals, signals], axis=1)
        
        self.build(self.signals, reset_header_names=True)

    def to_config(self):

        count = self.filters.count()

        config = {
            "sorted": self.sort.checkState() == QtCore.Qt.Checked,
            "channels": list(self.signals.columns) if not self.pattern else [],
            "filtered": bool(self.query.toPlainText()),
            "filters": [
                self.filters.itemWidget(self.filters.item(i)).to_config()
                for i in range(count)
            ]
            if not self.pattern
            else [],
            "time_as_date": self.time_as_date.checkState() == QtCore.Qt.Checked,
            "pattern": self.pattern,
            "format": self.format,
        }

        return config

    def sorting_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.tree.setSortingEnabled(True)
            self.tree.header().setSortIndicator(0, QtCore.Qt.AscendingOrder)
        else:
            self.tree.setSortingEnabled(False)

        self._display(0)

    def time_as_date_changed(self, state):
        count = self.filters.count()

        if state == QtCore.Qt.Checked:
            for i in range(count):
                filter = self.filters.itemWidget(self.filters.item(i))
                filter.dtype_kind[0] = "M"

                if filter.column.currentIndex() == 0:
                    filter.column_changed(0)
                else:
                    filter.validate_target()

            timestamps = pd.to_datetime(
                self.signals["timestamps"] + self.start,
                unit="s",
            )

            self.signals["timestamps"] = timestamps
        else:
            for i in range(count):
                filter = self.filters.itemWidget(self.filters.item(i))
                filter.dtype_kind[0] = "f"

                if filter.column.currentIndex() == 0:
                    filter.column_changed(0)
                else:
                    filter.validate_target()
            self.signals["timestamps"] = self._original_timestamps

        if self.query.toPlainText():
            self.apply_filters()
        else:
            self.build(self.signals)

    def remove_prefix_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.prefix.setEnabled(True)
        else:
            self.prefix.setEnabled(False)
        self.update_header()

    def prefix_changed(self, index):
        self.remove_prefix_changed(QtCore.Qt.Checked)

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
                progress = QtWidgets.QProgressDialog(
                    f'Data export to CSV file "{file_name}"', "", 0, 0, self.parent()
                )

                progress.setWindowModality(QtCore.Qt.ApplicationModal)
                progress.setCancelButton(None)
                progress.setAutoClose(True)
                progress.setWindowTitle("Export tabular window to CSV")
                icon = QtGui.QIcon()
                icon.addPixmap(
                    QtGui.QPixmap(":/csv.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                )
                progress.setWindowIcon(icon)
                progress.show()

                target = self.signals.to_csv
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
        modifier = event.modifiers()

        if (
            key in (QtCore.Qt.Key_H, QtCore.Qt.Key_B, QtCore.Qt.Key_P)
            and modifier == QtCore.Qt.ControlModifier
        ):
            if key == QtCore.Qt.Key_H:
                self.format_selection.setCurrentText("hex")
            elif key == QtCore.Qt.Key_B:
                self.format_selection.setCurrentText("bin")
            else:
                self.format_selection.setCurrentText("phys")

            event.accept()
        else:
            super().keyPressEvent(event)

    def set_format(self, fmt):
        self.format = fmt
        self._settings.setValue("tabular_format", fmt)
        self.build(self.signals)

        for row in range(self.filters.count()):
            filter = self.filters.itemWidget(self.filters.item(row))
            filter.int_format = fmt
            filter.validate_target()

        if self.query.toPlainText():
            self.apply_filters()

    def update_header(self):
        names = ["Index"]
        df = self.df

        state = self.remove_prefix.checkState()
        prefix = self.prefix.currentText()
        dim = len(prefix)

        for name, column_dtype in zip(self.header_names[1:], df.dtypes):

            if state == QtCore.Qt.Checked:
                name = name[dim:] if name.startswith(prefix) else name

            if column_dtype.kind in "ui":
                if self.format == "hex":
                    names.append(f"{name} [Hex]")
                elif self.format == "bin":
                    names.append(f"{name} [Bin]")
                else:
                    names.append(name)

            else:
                names.append(name)

        self.tree.setHeaderLabels(names)

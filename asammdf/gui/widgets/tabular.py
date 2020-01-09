# -*- coding: utf-8 -*-
import datetime
import logging
from traceback import format_exc

from PyQt5 import QtWidgets
from PyQt5 import QtCore
import pandas as pd
import numpy as np
import numpy.core.defchararray as npchar

from ..ui import resource_rc as resource_rc
from ..ui.tabular import Ui_TabularDisplay
from .tabular_filter import TabularFilter
from ...blocks.utils import csv_bytearray2hex, csv_int2hex, pandas_query_compatible


logger = logging.getLogger("asammdf.gui")
LOCAL_TIMEZONE = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo


class Tabular(Ui_TabularDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.pyqtSignal(list)

    def __init__(self, signals=None, start=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.signals_descr = {}
        self.start = start

        if signals is None:
            self.signals = pd.DataFrame()
        else:
            dropped = {}

            for name_ in signals.columns:
                col = signals[name_]
                if col.dtype.kind == "O":
                    #                    dropped[name_] = pd.Series(csv_bytearray2hex(col), index=signals.index)
                    self.signals_descr[name_] = 1
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

            signals = signals.drop(columns=list(dropped))
            for name, s in dropped.items():
                signals[name] = s

            self.signals = signals

        self.as_hex = [
            name.endswith("CAN_DataFrame.ID") for name in self.signals.columns
        ]

        self._original_index = self.signals.index.values

        self.build(self.signals, True)

        self.add_filter_btn.clicked.connect(self.add_filter)
        self.apply_filters_btn.clicked.connect(self.apply_filters)
        self.sort.stateChanged.connect(self.sorting_changed)
        self.time_as_date.stateChanged.connect(self.time_as_date_changed)
        self.remove_prefix.stateChanged.connect(self.remove_prefix_changed)
        self.tree.header().sortIndicatorChanged.connect(self._sort)

        prefixes = set()
        for name in self.signals.columns:
            while "." in name:
                name = name.rsplit(".", 1)[0]
                prefixes.add(f"{name}.")

        self.prefix.insertItems(0, sorted(prefixes))
        self.prefix.setEnabled(False)

        self.prefix.currentIndexChanged.connect(self.prefix_changed)

        self.tree_scroll.valueChanged.connect(self._display)
        self.tree.verticalScrollBar().valueChanged.connect(self._scroll_tree)

    def _scroll_tree(self, value):
        if (
            value == self.tree.verticalScrollBar().minimum()
            and self.tree_scroll.value() != self.tree_scroll.minimum()
        ):
            self.tree_scroll.setValue(
                self.tree_scroll.value() - self.tree_scroll.singleStep()
            )
            self.tree.verticalScrollBar().setValue(
                self.tree.verticalScrollBar().value()
                + self.tree.verticalScrollBar().singleStep()
            )
        elif value == self.tree.verticalScrollBar().maximum():
            self.tree_scroll.setValue(
                self.tree_scroll.value() + self.tree_scroll.singleStep()
            )
            self.tree.verticalScrollBar().setValue(
                self.tree.verticalScrollBar().value()
                - self.tree.verticalScrollBar().singleStep()
            )

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
                (name, self.signals[name].values.dtype.kind, self.signals_descr[name], as_hex)
                for name, as_hex in zip(self.signals.columns, self.as_hex)
            ]
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
                if column_name == df.index.name and df.index.dtype.kind == "M":

                    ts = pd.Timestamp(target, tz=LOCAL_TIMEZONE)
                    ts = ts.tz_convert("UTC").to_datetime64()

                    filters.append(column_name)
                    filters.append(op)
                    filters.append("@ts")

                elif is_byte_array:
                    target = str(target).replace(" ", "").strip('"')

                    if f"{column_name}__as__bytes" not in df.columns:
                        df[f"{column_name}__as__bytes"] = pd.Series(
                            [bytes(s) for s in df[column_name]], index=df.index,
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

        names = [df.index.name, *df.columns]

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

        if df.index.dtype.kind == "M":
            index = df.index.tz_localize("UTC").tz_convert(LOCAL_TIMEZONE)
        else:
            index = df.index
        items = [
            index.astype(str),
        ]

        for i, name in enumerate(df.columns):
            column = df[name]
            kind = column.dtype.kind

            if self.as_hex[i]:
                items.append(pd.Series(csv_int2hex(column.astype("<u4"))).values)
            else:

                if kind in "uif":
                    items.append(column.astype(str))
                elif kind == "S":
                    try:
                        items.append(npchar.decode(column, "utf-8"))
                    except:
                        items.append(npchar.decode(column, "latin-1"))
                elif kind == "O":
                    try:
                        items.append(pd.Series(csv_bytearray2hex(df[name])).values)
                    except:
                        items.append(pd.Series(df[name]).values)
                else:
                    items.append(column)

        if position == 0:
            self.tree.verticalScrollBar().setSliderPosition(0)
        elif position == self.tree_scroll.maximum():
            self.tree.verticalScrollBar().setSliderPosition(
                self.tree.verticalScrollBar().maximum()
            )

        items = [QtWidgets.QTreeWidgetItem(row) for row in zip(*items)]

        self.tree.addTopLevelItems(items)

        self.tree.setSortingEnabled(self.sort.checkState() == QtCore.Qt.Checked)

    def add_new_channels(self, channels):
        for sig in channels:
            if sig:
                self.signals[sig.name] = sig
        self.build(self.signals, reset_header_names=True)

    def to_config(self):

        count = self.filters.count()

        config = {
            "sorted": self.sort.checkState() == QtCore.Qt.Checked,
            "channels": list(self.signals.columns),
            "filtered": bool(self.query.toPlainText()),
            "filters": [
                self.filters.itemWidget(self.filters.item(i)).to_config()
                for i in range(count)
            ],
            "time_as_date": self.time_as_date.checkState() == QtCore.Qt.Checked,
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
                filter._target = None
                if filter.column.currentIndex() == 0:
                    filter.column_changed(0)
            index = pd.to_datetime(self.signals.index + self.start, unit="s")

            self.signals.index = index
        else:
            for i in range(count):
                filter = self.filters.itemWidget(self.filters.item(i))
                filter.dtype_kind[0] = "f"
                filter._target = None
                if filter.column.currentIndex() == 0:
                    filter.column_changed(0)
            self.signals.index = self._original_index
        self.signals.index.name = "timestamps"

        if self.query.toPlainText():
            self.apply_filters()
        else:
            self.build(self.signals)

    def remove_prefix_changed(self, state):

        if state == QtCore.Qt.Checked:
            self.prefix.setEnabled(True)
            prefix = self.prefix.currentText()
            dim = len(prefix)
            names = [
                name[dim:] if name.startswith(prefix) else name
                for name in self.header_names
            ]
            self.tree.setHeaderLabels(names)
        else:
            self.prefix.setEnabled(False)
            self.tree.setHeaderLabels(self.header_names)

    def prefix_changed(self, index):
        self.remove_prefix_changed(QtCore.Qt.Checked)

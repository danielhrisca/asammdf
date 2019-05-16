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


class TreeItem(QtWidgets.QTreeWidgetItem):

    def __lt__(self, otherItem):
        column = self.treeWidget().sortColumn()

        val1 = self.text(column)
        try:
            val1 = float(val1)
        except:
            pass

        val2 = otherItem.text(column)
        try:
            val2 = float(val2)
        except:
            pass

        try:
            return val1 < val2
        except:
            if isinstance(val1, float):
                return True
            else:
                return False


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
                if col.dtype.kind == 'O':
                    dropped[name_] = pd.Series(csv_bytearray2hex(col), index=signals.index)
                    self.signals_descr[name_] = 1
                elif col.dtype.kind == 'S':
                    try:
                        dropped[name_] = pd.Series(npchar.decode(col, 'utf-8'), index=signals.index)
                    except:
                        dropped[name_] = pd.Series(npchar.decode(col, 'latin-1'), index=signals.index)
                    self.signals_descr[name_] = 0
                else:
                    self.signals_descr[name_] = 0

            signals = signals.drop(columns=list(dropped))
            for name, s in dropped.items():
                signals[name] = s

            self.signals = signals

        self._original_index = self.signals.index.values

        self.build(self.signals)

        self.add_filter_btn.clicked.connect(self.add_filter)
        self.apply_filters_btn.clicked.connect(self.apply_filters)
        self.sort.stateChanged.connect(self.sorting_changed)
        self.time_as_date.stateChanged.connect(self.time_as_date_changed)

    def add_filter(self, event=None):
        filter_widget = TabularFilter(
            [(self.signals.index.name, self.signals.index.values.dtype.kind, 0)] +
            [
                (name, self.signals[name].values.dtype.kind, self.signals_descr[name])
                for name in self.signals.columns
            ]
        )

        item = QtWidgets.QListWidgetItem(self.filters)
        item.setSizeHint(filter_widget.sizeHint())
        self.filters.addItem(item)
        self.filters.setItemWidget(item, filter_widget)

    def apply_filters(self, event=None):
        df = self.signals

        friendly_names = {
            name: pandas_query_compatible(name)
            for name in df.columns
        }

        original_names = {
            val: key
            for key, val in friendly_names.items()
        }

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

            column_name = pandas_query_compatible(filter.column.currentText())
            op = filter.op.currentText()

            if target != target:
                # here we have NaN
                nan = np.nan

                if op in ('>', '>=', '<', '<='):
                    filters.append(column_name)
                    filters.append(op)
                    filters.append('@nan')
                elif op == '!=':
                    filters.append(column_name)
                    filters.append('==')
                    filters.append(column_name)
                elif op == '==':
                    filters.append(column_name)
                    filters.append('!=')
                    filters.append(column_name)
            else:
                if column_name == df.index.name and df.index.dtype.kind == 'M':

                    ts = pd.Timestamp(target, tz=LOCAL_TIMEZONE)
                    ts = ts.tz_convert('UTC').to_datetime64()

                    filters.append(column_name)
                    filters.append(op)
                    filters.append('@ts')

                else:
                    filters.append(column_name)
                    filters.append(op)
                    filters.append(str(target))

        if filters:
            try:
                new_df = df.query(' '.join(filters))
            except:
                logger.exception(f'Failed to apply filter for tabular window: {" ".join(filters)}')
                self.query.setText(format_exc())
            else:

                self.query.setText(' '.join(filters))
                new_df.rename(columns=original_names, inplace=True)
                self.build(new_df)
        else:
            self.query.setText('')
            df.rename(columns=original_names, inplace=True)
            self.build(df)

        self.signals.rename(columns=original_names, inplace=True)

    def items_deleted(self, names):
        for name in names:
            self.signals.pop(name)
        self.build()

    def build(self, df):
        self.tree.setSortingEnabled(False)
        self.tree.clear()

        dropped = {}

        for name_ in df.columns:
            if name_.endswith('CAN_DataFrame.ID'):
                dropped[name_] = pd.Series(csv_int2hex(df[name_].astype('<u4')), index=df.index)

        df = df.drop(columns=list(dropped))
        for name, s in dropped.items():
            df[name] = s

        names = [
            df.index.name,
            *df.columns
        ]

        self.tree.setColumnCount(len(names))
        self.tree.setHeaderLabels(names)

        if df.index.dtype.kind == 'M':
            index = df.index.tz_localize('UTC').tz_convert(LOCAL_TIMEZONE)
        else:
            index = df.index
        items = [index.astype(str),]

        for name in df:
            column = df[name]
            kind = column.dtype.kind

            if kind in 'uif':
                items.append(column.astype(str))
            elif kind == 'S':
                try:
                    items.append(npchar.decode(column, 'utf-8'))
                except:
                    items.append(npchar.decode(column, 'latin-1'))
            else:
                items.append(column)

        items = [
            TreeItem(row)
            for row in zip(*items)
        ]

        self.tree.addTopLevelItems(items)

    def add_new_channels(self, channels):
        for sig in channels:
            if sig:
                self.signals[sig.name] = sig
        self.build()

    def to_config(self):

        count = self.filters.count()

        config = {
            'sorted': self.sort.checkState() == QtCore.Qt.Checked,
            'channels': list(self.signals.columns),
            'filtered': bool(self.query.toPlainText()),
            'filters': [
                self.filters.itemWidget(self.filters.item(i)).to_config()
                for i in range(count)
            ],
            'time_as_date': self.time_as_date.checkState() == QtCore.Qt.Checked,
        }

        return config

    def sorting_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.tree.setSortingEnabled(True)
        else:
            self.tree.header().setSortIndicator(0, QtCore.Qt.AscendingOrder)
            self.tree.setSortingEnabled(False)

    def time_as_date_changed(self, state):
        count = self.filters.count()

        if state == QtCore.Qt.Checked:
            for i in range(count):
                filter = self.filters.itemWidget(self.filters.item(i))
                filter.dtype_kind[0] = 'M'
                filter._target = None
                filter.validate_target()
            index = pd.to_datetime(self.signals.index + self.start, unit='s')

            self.signals.index = index
        else:
            for i in range(count):
                filter = self.filters.itemWidget(self.filters.item(i))
                filter.dtype_kind[0] = 'f'
                filter._target = None
                filter.validate_target()
            self.signals.index = self._original_index
        self.signals.index.name = 'time'

        if self.query.toPlainText():
            self.apply_filters()
        else:
            self.build(self.signals)

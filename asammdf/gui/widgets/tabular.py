# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from natsort import natsorted
from numpy import searchsorted
import pandas as pd

from ..ui import resource_rc as resource_rc
from ..ui.tabular import Ui_TabularDisplay
from ...blocks.utils import csv_bytearray2hex, csv_int2hex


class TreeItem(QtWidgets.QTreeWidgetItem):

    def __lt__(self, otherItem):
        column = self.treeWidget().sortColumn()

        if column == 1:
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
        else:
            return self.text(column) < otherItem.text(column)


class Tabular(Ui_TabularDisplay, QtWidgets.QWidget):
    add_channels_request = QtCore.pyqtSignal(list)

    def __init__(self, signals, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        if signals is not None:
            dropped = {}

            for name_ in signals.columns:
                if name_.endswith('CAN_DataFrame.ID'):
                    dropped[name_] = pd.Series(csv_int2hex(signals[name_].astype('<u4')), index=signals.index)

                elif name_.endswith('CAN_DataFrame.DataBytes'):
                    dropped[name_] = pd.Series(csv_bytearray2hex(signals[name_]), index=signals.index)

            signals = signals.drop(columns=list(dropped))
            for name, s in dropped.items():
                signals[name] = s

            self.signals = signals
        else:
            self.signals = pd.DataFrame()

        self.build()

    def items_deleted(self, names):
        for name in names:
            self.signals.pop(name)
        self.build()

    def build(self):
        self.tree.clear()

        df = self.signals

        names = [
            df.index.name,
            *df.columns
        ]

        self.tree.setColumnCount(len(names))
        self.tree.setHeaderLabels(names)

        vals = [df.index.astype(str), *(df[name].astype(str) for name in df)]

        items = [
            TreeItem(row)
            for row in zip(*vals)
        ]

        self.tree.addTopLevelItems(items)

    def add_new_channels(self, channels):
        for sig in channels:
            if sig:
                self.signals[sig.name] = sig
        self.build()

    def to_config(self):

        channels = []
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channels)
        while 1:
            item = iterator.value()
            if not item:
                break
            channels.append(item.text(0))
            iterator += 1

        config = {
            'format': self.format,
            'channels': channels,
        }

        return config

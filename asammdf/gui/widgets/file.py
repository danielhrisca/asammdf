# -*- coding: utf-8 -*-
import os
from copy import deepcopy
from functools import partial
from threading import Thread
from time import sleep

import numpy as np

try:
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5 import uic
    from ..ui import resource_qt5 as resource_rc

    QT = 5

except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    from PyQt4 import uic
    from ..ui import resource_qt4 as resource_rc

    QT = 4

from ...mdf import MDF, SUPPORTED_VERSIONS
from ..utils import TERMINATED, run_thread_with_progress, setup_progress
from .channel_display import ChannelDisplay
from .channel_stats import ChannelStats
from .list import ListWidget
from .plot import Plot
from .search import SearchWidget
from .tree import TreeWidget
from .tree_item import TreeItem

from ..dialogs.advanced_search import AdvancedSearch
from ..dialogs.channel_info import ChannelInfoDialog
from ..dialogs.tabular import TabularValuesDialog

HERE = os.path.dirname(os.path.realpath(__file__))


class FileWidget(QWidget):
    def __init__(self, file_name, memory, step_mode, with_dots, *args, **kwargs):
        super(FileWidget, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "..", "ui", "file_widget.ui"), self)

        self.plot = None

        self.file_name = file_name
        self.progress = None
        self.mdf = None
        self.memory = memory
        self.info = None
        self.info_index = None
        self.step_mode = step_mode
        self.with_dots = with_dots

        progress = QProgressDialog(
            'Opening "{}"'.format(self.file_name), "", 0, 100, self.parent()
        )

        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setAutoClose(True)
        progress.setWindowTitle("Opening measurement")
        icon = QIcon()
        icon.addPixmap(QPixmap(":/open.png"), QIcon.Normal, QIcon.Off)
        progress.setWindowIcon(icon)
        progress.show()

        if file_name.lower().endswith("erg"):
            progress.setLabelText("Converting from erg to mdf")
            try:
                from mfile import ERG

                self.mdf = ERG(file_name).export_mdf()
            except Exception as err:
                print(err)
                return
        else:

            if file_name.lower().endswith("dl3"):
                progress.setLabelText("Converting from dl3 to mdf")
                try:
                    import win32com.client

                    index = 0
                    while True:
                        mdf_name = "{}.{}.mdf".format(file_name, index)
                        if os.path.exists(mdf_name):
                            index += 1
                        else:
                            break

                    datalyser = win32com.client.Dispatch("Datalyser3.Datalyser3_COM")
                    try:
                        datalyser.DCOM_set_datalyser_visibility(False)
                    except:
                        pass
                    ret = datalyser.DCOM_convert_file_mdf_dl3(file_name, mdf_name, 0)
                    datalyser.DCOM_TerminateDAS()
                    file_name = mdf_name
                except Exception as err:
                    print(err)
                    return

            target = MDF
            kwargs = {
                "name": file_name,
                "memory": memory,
                "callback": self.update_progress,
            }

            self.mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=0,
                progress=progress,
            )

            if self.mdf is TERMINATED:
                return

        progress.setLabelText("Loading graphical elements")

        progress.setValue(35)

        self.filter_field = SearchWidget(deepcopy(self.mdf.channels_db), self)

        progress.setValue(37)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)

        channel_and_search = QWidget(splitter)

        self.channels_tree = TreeWidget(channel_and_search)
        self.search_field = SearchWidget(
            deepcopy(self.mdf.channels_db), channel_and_search
        )
        self.filter_tree = TreeWidget()

        self.search_field.selectionChanged.connect(
            partial(
                self.new_search_result,
                tree=self.channels_tree,
                search=self.search_field,
            )
        )
        self.filter_field.selectionChanged.connect(
            partial(
                self.new_search_result, tree=self.filter_tree, search=self.filter_field
            )
        )

        vbox = QVBoxLayout(channel_and_search)
        self.advanced_search_btn = QPushButton("", channel_and_search)
        icon = QIcon()
        icon.addPixmap(QPixmap(":/search.png"), QIcon.Normal, QIcon.Off)
        self.advanced_search_btn.setIcon(icon)
        self.advanced_search_btn.setToolTip("Advanced search and select channels")
        self.advanced_search_btn.clicked.connect(self.search)
        vbox.addWidget(self.search_field)

        vbox.addWidget(self.channels_tree, 1)
        channel_and_search.setLayout(vbox)

        hbox = QHBoxLayout(channel_and_search)

        self.clear_channels_btn = QPushButton("", channel_and_search)
        self.clear_channels_btn.setToolTip("Reset selection")
        icon = QIcon()
        icon.addPixmap(QPixmap(":/erase.png"), QIcon.Normal, QIcon.Off)
        self.clear_channels_btn.setIcon(icon)
        self.clear_channels_btn.setObjectName("clear_channels_btn")

        self.load_channel_list_btn = QPushButton("", channel_and_search)
        self.load_channel_list_btn.setToolTip("Load channel selection list")
        icon1 = QIcon()
        icon1.addPixmap(QPixmap(":/open.png"), QIcon.Normal, QIcon.Off)
        self.load_channel_list_btn.setIcon(icon1)
        self.load_channel_list_btn.setObjectName("load_channel_list_btn")

        self.save_channel_list_btn = QPushButton("", channel_and_search)
        self.save_channel_list_btn.setToolTip("Save channel selection list")
        icon2 = QIcon()
        icon2.addPixmap(QPixmap(":/save.png"), QIcon.Normal, QIcon.Off)
        self.save_channel_list_btn.setIcon(icon2)
        self.save_channel_list_btn.setObjectName("save_channel_list_btn")

        self.select_all_btn = QPushButton("", channel_and_search)
        self.select_all_btn.setToolTip("Select all channels")
        icon1 = QIcon()
        icon1.addPixmap(QPixmap(":/checkmark.png"), QIcon.Normal, QIcon.Off)
        self.select_all_btn.setIcon(icon1)

        hbox.addWidget(self.load_channel_list_btn)
        hbox.addWidget(self.save_channel_list_btn)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        hbox.addWidget(line)
        hbox.addWidget(self.select_all_btn)
        hbox.addWidget(self.clear_channels_btn)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        hbox.addWidget(line)
        hbox.addWidget(self.advanced_search_btn)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        hbox.addWidget(line)
        self.plot_btn = QPushButton("", channel_and_search)
        self.plot_btn.setToolTip("Plot selected channels")
        icon3 = QIcon()
        icon3.addPixmap(QPixmap(":/graph.png"), QIcon.Normal, QIcon.Off)
        self.plot_btn.setIcon(icon3)
        self.plot_btn.setObjectName("plot_btn")
        hbox.addWidget(self.plot_btn)
        hbox.addSpacerItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        vbox.addLayout(hbox)

        selection_list = QWidget(splitter)
        self.channel_selection = ListWidget(selection_list)
        self.channel_selection.setAlternatingRowColors(False)

        vbox = QVBoxLayout(selection_list)

        hbox = QHBoxLayout(selection_list)
        hbox.addWidget(QLabel("Selected channels"))
        self.cursor_info = QLabel("")
        self.cursor_info.setTextFormat(Qt.RichText)
        self.cursor_info.setAlignment(
            Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter
        )
        hbox.addWidget(self.cursor_info)

        vbox.addLayout(hbox)

        vbox.addWidget(self.channel_selection)

        self.filter_layout.addWidget(self.filter_field, 0, 0, 1, 1)

        self.channels_tree.itemDoubleClicked.connect(self.show_channel_info)
        self.filter_tree.itemDoubleClicked.connect(self.show_channel_info)

        self.channels_layout.insertWidget(0, splitter)
        self.filter_layout.addWidget(self.filter_tree, 1, 0, 8, 1)

        groups_nr = len(self.mdf.groups)

        self.channels_tree.setHeaderLabel("Channels")
        self.channels_tree.setToolTip(
            "Double click channel to see extended information"
        )
        self.filter_tree.setHeaderLabel("Channels")
        self.filter_tree.setToolTip("Double click channel to see extended information")

        for i, group in enumerate(self.mdf.groups):
            channel_group = QTreeWidgetItem()
            filter_channel_group = QTreeWidgetItem()
            channel_group.setText(0, "Channel group {}".format(i))
            filter_channel_group.setText(0, "Channel group {}".format(i))
            channel_group.setFlags(
                channel_group.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable
            )
            filter_channel_group.setFlags(
                channel_group.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable
            )

            self.channels_tree.addTopLevelItem(channel_group)
            self.filter_tree.addTopLevelItem(filter_channel_group)

            for j, ch in enumerate(group["channels"]):

                name = self.mdf.get_channel_name(i, j)
                channel = TreeItem((i, j), channel_group)
                channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                channel.setText(0, name)
                channel.setCheckState(0, Qt.Unchecked)

                channel = TreeItem((i, j), filter_channel_group)
                channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                channel.setText(0, name)
                channel.setCheckState(0, Qt.Unchecked)

            if self.mdf.version >= "4.00":
                for j, ch in enumerate(group["logging_channels"], 1):
                    name = ch.name

                    channel = TreeItem((i, -j), channel_group)
                    channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                    channel.setText(0, name)
                    channel.setCheckState(0, Qt.Unchecked)

                    channel = TreeItem((i, -j), filter_channel_group)
                    channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                    channel.setText(0, name)
                    channel.setCheckState(0, Qt.Unchecked)

            progress.setValue(37 + int(53 * (i + 1) / groups_nr))
            QApplication.processEvents()

        progress.setValue(90)

        self.resample_format.insertItems(0, SUPPORTED_VERSIONS)
        index = self.resample_format.findText(self.mdf.version)
        if index >= 0:
            self.resample_format.setCurrentIndex(index)
        self.resample_compression.insertItems(
            0, ("no compression", "deflate", "transposed deflate")
        )
        self.resample_split_size.setValue(10)
        self.resample_btn.clicked.connect(self.resample)

        self.filter_format.insertItems(0, SUPPORTED_VERSIONS)
        index = self.filter_format.findText(self.mdf.version)
        if index >= 0:
            self.filter_format.setCurrentIndex(index)
        self.filter_compression.insertItems(
            0, ("no compression", "deflate", "transposed deflate")
        )
        self.filter_split_size.setValue(10)
        self.filter_btn.clicked.connect(self.filter)

        self.convert_format.insertItems(0, SUPPORTED_VERSIONS)
        self.convert_compression.insertItems(
            0, ("no compression", "deflate", "transposed deflate")
        )
        self.convert_split_size.setValue(10)
        self.convert_btn.clicked.connect(self.convert)

        self.cut_format.insertItems(0, SUPPORTED_VERSIONS)
        index = self.cut_format.findText(self.mdf.version)
        if index >= 0:
            self.cut_format.setCurrentIndex(index)
        self.cut_compression.insertItems(
            0, ("no compression", "deflate", "transposed deflate")
        )
        self.cut_split_size.setValue(10)
        self.cut_btn.clicked.connect(self.cut)

        self.cut_interval.setText("Unknown measurement interval")

        progress.setValue(99)

        self.empty_channels.insertItems(0, ("zeros", "skip"))
        self.mat_format.insertItems(0, ("4", "5", "7.3"))
        self.oned_as.insertItems(0, ("row", "column"))
        self.export_type.insertItems(0, ("csv", "excel", "hdf5", "mat", "parquet"))
        self.export_btn.clicked.connect(self.export)

        # self.channels_tree.itemChanged.connect(self.select)
        self.plot_btn.clicked.connect(self.plot_pyqtgraph)
        self.clear_filter_btn.clicked.connect(self.clear_filter)
        self.clear_channels_btn.clicked.connect(self.clear_channels)

        self.aspects.setCurrentIndex(0)

        progress.setValue(100)

        self.load_channel_list_btn.clicked.connect(self.load_channel_list)
        self.save_channel_list_btn.clicked.connect(self.save_channel_list)
        self.load_filter_list_btn.clicked.connect(self.load_filter_list)
        self.save_filter_list_btn.clicked.connect(self.save_filter_list)

        self.channel_selection.itemsDeleted.connect(self.channel_selection_reduced)
        self.channel_selection.itemSelectionChanged.connect(
            self.channel_selection_modified
        )

    def set_line_style(self, with_dots=None, step_mode=None):
        if (with_dots, step_mode) != (None, None):
            if step_mode is not None:
                self.step_mode = step_mode

            if with_dots is not None:
                self.with_dots = with_dots

            if self.plot:
                self.plot.update_lines(step_mode=step_mode, with_dots=with_dots)

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if key == Qt.Key_M:

            if self.info is None:

                self.info = ChannelStats(parent=self.splitter)
                if self.info_index is None:
                    self.info.clear()
                else:
                    stats = self.plot.get_stats(self.info_index)
                    self.info.set_stats(stats)
            else:
                self.info.setParent(None)
                self.info.hide()
                self.info = None

        elif modifier == Qt.ControlModifier and key in (Qt.Key_B, Qt.Key_H, Qt.Key_P):
            if key == Qt.Key_B:
                fmt = "bin"
            elif key == Qt.Key_H:
                fmt = "hex"
            else:
                fmt = "phys"
            if self.info and self.info_index is not None:
                self.info.fmt = fmt
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

        elif modifier == Qt.ControlModifier and key == Qt.Key_T:
            selected_items = self.channel_selection.selectedItems()
            rows = self.channel_selection.count()

            indexes = [
                i
                for i in range(rows)
                if self.channel_selection.item(i) in selected_items
            ]

            ranges = [
                self.channel_selection.itemWidget(item).ranges
                for item in selected_items
            ]

            signals = [self.plot.signals[i] for i in indexes]

            dlg = TabularValuesDialog(signals, ranges, self)
            dlg.setModal(True)
            dlg.exec_()

        else:
            super(FileWidget, self).keyPressEvent(event)

    def search(self):
        dlg = AdvancedSearch(self.mdf.channels_db, self)
        dlg.setModal(True)
        dlg.exec_()
        result = dlg.result
        if result:
            iterator = QTreeWidgetItemIterator(self.channels_tree)

            dg_cntr = -1
            ch_cntr = 0

            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    dg_cntr += 1
                    ch_cntr = 0
                    continue

                if (dg_cntr, ch_cntr) in result:
                    item.setCheckState(0, Qt.Checked)

                iterator += 1
                ch_cntr += 1

    def channel_selection_reduced(self, deleted):

        for i in sorted(deleted, reverse=True):
            item = self.plot.curves.pop(i)
            item.hide()
            item.setParent(None)

            item = self.plot.axes.pop(i)
            item.hide()
            item.setParent(None)

            item = self.plot.view_boxes.pop(i)
            item.hide()
            item.setParent(None)

            self.plot.signals.pop(i)

        rows = self.channel_selection.count()

        for i in range(rows):
            item = self.channel_selection.item(i)
            wid = self.channel_selection.itemWidget(item)
            wid.index = i

    def channel_selection_modified(self):
        selected_items = self.channel_selection.selectedItems()
        count = len([sig for sig in self.plot.signals if sig.enable])
        rows = self.channel_selection.count()

        for i in range(rows):
            item = self.channel_selection.item(i)
            if count > 1 and item in selected_items:
                if self.plot.signals[i].enable and not self.plot.axes[i].isVisible():
                    self.plot.axes[i].show()
                if self.info:
                    self.info.clear()
            else:
                if self.plot.axes[i].isVisible():
                    self.plot.axes[i].hide()

        if len(selected_items) == 1:
            self.info_index = self.channel_selection.row(selected_items[0])
        else:
            self.info_index = None

        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def save_channel_list(self):
        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select output channel list file", "", "TXT files (*.txt)"
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select output channel list file", "", "TXT files (*.txt)"
            )
            file_name = str(file_name)
        if file_name:
            with open(file_name, "w") as output:
                iterator = QTreeWidgetItemIterator(self.channels_tree)

                signals = []
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue

                    if item.checkState(0) == Qt.Checked:
                        signals.append(item.text(0))

                    iterator += 1

                output.write("\n".join(signals))

    def load_channel_list(self):
        if QT > 4:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Select channel list file", "", "TXT files (*.txt)"
            )
        else:
            file_name = QFileDialog.getOpenFileName(
                self, "Select channel list file", "", "TXT files (*.txt)"
            )
            file_name = str(file_name)

        if file_name:
            with open(file_name, "r") as infile:
                channels = [line.strip() for line in infile.readlines()]
                channels = [name for name in channels if name]

            iterator = QTreeWidgetItemIterator(self.channels_tree)

            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    continue

                channel_name = item.text(0)
                if channel_name in channels:
                    item.setCheckState(0, Qt.Checked)
                    channels.pop(channels.index(channel_name))
                else:
                    item.setCheckState(0, Qt.Unchecked)

                iterator += 1

    def save_filter_list(self):
        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select output filter list file", "", "TXT files (*.txt)"
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select output filter list file", "", "TXT files (*.txt)"
            )
            file_name = str(file_name)

        if file_name:
            with open(file_name, "w") as output:
                iterator = QTreeWidgetItemIterator(self.filter_tree)

                signals = []
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue

                    if item.checkState(0) == Qt.Checked:
                        signals.append(item.text(0))

                    iterator += 1

                output.write("\n".join(signals))

    def load_filter_list(self):
        if QT > 4:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Select filter list file", "", "TXT files (*.txt)"
            )
        else:
            file_name = QFileDialog.getOpenFileName(
                self, "Select filter list file", "", "TXT files (*.txt)"
            )
            file_name = str(file_name)

        if file_name:
            with open(file_name, "r") as infile:
                channels = [line.strip() for line in infile.readlines()]
                channels = [name for name in channels if name]

            iterator = QTreeWidgetItemIterator(self.filter_tree)

            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    continue

                channel_name = item.text(0)
                if channel_name in channels:
                    item.setCheckState(0, Qt.Checked)
                    channels.pop(channels.index(channel_name))
                else:
                    item.setCheckState(0, Qt.Unchecked)

                iterator += 1

    def cursor_move_finished(self):
        x = self.plot.timebase

        if x is not None and len(x):
            dim = len(x)
            position = self.plot.cursor1.value()

            right = np.searchsorted(x, position, side="right")
            if right == 0:
                next_pos = x[0]
            elif right == dim:
                next_pos = x[-1]
            else:
                if position - x[right - 1] < x[right] - position:
                    next_pos = x[right - 1]
                else:
                    next_pos = x[right]
            self.plot.cursor1.setPos(next_pos)

        self.plot.cursor_hint.setData(x=[], y=[])

    def cursor_moved(self):
        position = self.plot.cursor1.value()

        x = self.plot.timebase

        if x is not None and len(x):
            dim = len(x)
            position = self.plot.cursor1.value()

            right = np.searchsorted(x, position, side="right")
            if right == 0:
                next_pos = x[0]
            elif right == dim:
                next_pos = x[-1]
            else:
                if position - x[right - 1] < x[right] - position:
                    next_pos = x[right - 1]
                else:
                    next_pos = x[right]

            y = []

            _, (hint_min, hint_max) = self.plot.viewbox.viewRange()

            for viewbox, sig, curve in zip(
                self.plot.view_boxes, self.plot.signals, self.plot.curves
            ):
                if curve.isVisible():
                    index = np.argwhere(sig.timestamps == next_pos).flatten()
                    if len(index):
                        _, (y_min, y_max) = viewbox.viewRange()

                        sample = sig.samples[index[0]]
                        sample = (sample - y_min) / (y_max - y_min) * (
                            hint_max - hint_min
                        ) + hint_min

                        y.append(sample)

            if self.plot.curve.isVisible():
                timestamps = self.plot.curve.xData
                samples = self.plot.curve.yData
                if len(samples):
                    index = np.argwhere(timestamps == next_pos).flatten()
                    if len(index):
                        _, (y_min, y_max) = self.plot.viewbox.viewRange()

                        sample = samples[index[0]]
                        sample = (sample - y_min) / (y_max - y_min) * (
                            hint_max - hint_min
                        ) + hint_min

                        y.append(sample)

            self.plot.viewbox.setYRange(hint_min, hint_max, padding=0)
            self.plot.cursor_hint.setData(x=[next_pos] * len(y), y=y)
            self.plot.cursor_hint.show()

        if not self.plot.region:
            self.cursor_info.setText("t = {:.6f}s".format(position))
            for i, signal in enumerate(self.plot.signals):
                cut_sig = signal.cut(position, position)
                if signal.texts is None or len(cut_sig) == 0:
                    samples = cut_sig.samples
                    if signal.conversion and "text_0" in signal.conversion:
                        samples = signal.conversion.convert(samples)
                        try:
                            samples = [s.decode("utf-8") for s in samples]
                        except:
                            samples = [s.decode("latin-1") for s in samples]
                else:
                    t = np.argwhere(signal.timestamps == cut_sig.timestamps).flatten()
                    try:
                        samples = [e.decode("utf-8") for e in signal.texts[t]]
                    except:
                        samples = [e.decode("latin-1") for e in signal.texts[t]]

                item = self.channel_selection.item(i)
                item = self.channel_selection.itemWidget(item)

                item.setPrefix("= ")
                item.setFmt(signal.format)

                if len(samples):
                    item.setValue(samples[0])
                else:
                    item.setValue("n.a.")

        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def cursor_removed(self):
        for i, signal in enumerate(self.plot.signals):
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            if not self.plot.region:
                self.cursor_info.setText("")
                item.setPrefix("")
                item.setValue("")
        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def range_modified(self):
        start, stop = self.plot.region.getRegion()
        self.cut_start.setValue(start)
        self.cut_stop.setValue(stop)

        self.cursor_info.setText(
            (
                "< html > < head / > < body >"
                "< p >t1 = {:.6f}s< / p > "
                "< p >t2 = {:.6f}s< / p > "
                "< p >Δt = {:.6f}s< / p > "
                "< / body > < / html >"
            ).format(start, stop, stop - start)
        )

        for i, signal in enumerate(self.plot.signals):
            samples = signal.cut(start, stop).samples
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            item.setPrefix("Δ = ")
            item.setFmt(signal.format)

            if len(samples):
                if samples.dtype.kind in "ui":
                    delta = np.int64(np.float64(samples[-1]) - np.float64(samples[0]))
                else:
                    delta = samples[-1] - samples[0]

                item.setValue(delta)

            else:
                item.setValue("n.a.")

        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def xrange_changed(self):

        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def range_modified_finished(self):
        start, stop = self.plot.region.getRegion()

        if self.plot.timebase is not None and len(self.plot.timebase):
            timebase = self.plot.timebase
            dim = len(timebase)

            right = np.searchsorted(timebase, start, side="right")
            if right == 0:
                next_pos = timebase[0]
            elif right == dim:
                next_pos = timebase[-1]
            else:
                if start - timebase[right - 1] < timebase[right] - start:
                    next_pos = timebase[right - 1]
                else:
                    next_pos = timebase[right]
            start = next_pos

            right = np.searchsorted(timebase, stop, side="right")
            if right == 0:
                next_pos = timebase[0]
            elif right == dim:
                next_pos = timebase[-1]
            else:
                if stop - timebase[right - 1] < timebase[right] - stop:
                    next_pos = timebase[right - 1]
                else:
                    next_pos = timebase[right]
            stop = next_pos

            self.plot.region.setRegion((start, stop))

    def range_removed(self):
        for i, signal in enumerate(self.plot.signals):
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            item.setPrefix("")
            item.setValue("")
            self.cursor_info.setText("")
        if self.plot.cursor1:
            self.plot.cursor_moved.emit()
        if self.info:
            if self.info_index is None:
                self.info.clear()
            else:
                stats = self.plot.get_stats(self.info_index)
                self.info.set_stats(stats)

    def compute_cut_hints(self):
        # TODO : use master channel physical min and max values
        times = []
        groups_nr = len(self.mdf.groups)
        for i in range(groups_nr):
            master = self.mdf.get_master(i)
            if len(master):
                times.append(master[0])
                times.append(master[-1])
            QApplication.processEvents()

        if len(times):
            time_range = min(times), max(times)

            self.cut_start.setRange(*time_range)
            self.cut_stop.setRange(*time_range)

            self.cut_interval.setText(
                "Cut interval ({:.6f}s - {:.6f}s)".format(*time_range)
            )
        else:
            self.cut_start.setRange(0, 0)
            self.cut_stop.setRange(0, 0)

            self.cut_interval.setText("Empty measurement")

    def update_progress(self, current_index, max_index):
        self.progress = current_index, max_index

    def show_channel_info(self, item, column):
        if item and item.parent():
            group, index = item.entry

            channel = self.mdf.get_channel_metadata(group=group, index=index)

            msg = ChannelInfoDialog(channel, self)
            msg.show()

    def clear_filter(self):
        iterator = QTreeWidgetItemIterator(self.filter_tree)

        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, Qt.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def clear_channels(self):
        iterator = QTreeWidgetItemIterator(self.channels_tree)

        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, Qt.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def new_search_result(self, tree, search):
        group_index, channel_index = search.entries[search.current_index]

        grp = self.mdf.groups[group_index]
        channel_count = len(grp["channels"])

        iterator = QTreeWidgetItemIterator(tree)

        group = -1
        index = 0
        while iterator.value():
            item = iterator.value()
            if item.parent() is None:
                iterator += 1
                group += 1
                index = 0
                continue

            if group == group_index:

                if (
                    channel_index >= 0
                    and index == channel_index
                    or channel_index < 0
                    and index == -channel_index - 1 + channel_count
                ):
                    tree.scrollToItem(item, QAbstractItemView.PositionAtTop)
                    item.setSelected(True)

            index += 1
            iterator += 1

    def close(self):
        mdf_name = self.mdf.name
        self.mdf.close()
        if self.file_name.lower().endswith("dl3"):
            os.remove(mdf_name)

    def convert(self, event):
        version = self.convert_format.currentText()

        memory = self.memory

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.convert_split.checkState() == Qt.Checked
        if split:
            split_size = int(self.convert_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.convert_compression.currentIndex()

        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select output measurement file", "", filter
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select output measurement file", "", filter
            )
            file_name = str(file_name)

        if file_name:

            progress = setup_progress(
                parent=self,
                title="Converting measurement",
                message='Converting "{}" from {} to {} '.format(
                    self.file_name, self.mdf.version, version
                ),
                icon_name="convert",
            )

            # convert self.mdf
            target = self.mdf.convert
            kwargs = {"version": version, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=50,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText('Saving converted file "{}"'.format(file_name))

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=50,
                offset=50,
                progress=progress,
            )

    def resample(self, event):
        version = self.resample_format.currentText()
        raster = self.raster.value()
        memory = self.memory

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.resample_split.checkState() == Qt.Checked
        if split:
            split_size = int(self.resample_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.resample_compression.currentIndex()

        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select output measurement file", "", filter
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select output measurement file", "", filter
            )
            file_name = str(file_name)

        if file_name:
            progress = setup_progress(
                parent=self,
                title="Resampling measurement",
                message='Resampling "{}" to {}s raster '.format(self.file_name, raster),
                icon_name="resample",
            )

            # resample self.mdf
            target = self.mdf.resample
            kwargs = {"raster": raster, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            # convert mdf
            progress.setLabelText(
                "Converting from {} to {}".format(mdf.version, version)
            )

            target = mdf.convert
            kwargs = {"to": version, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=33,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText('Saving resampled file "{}"'.format(file_name))

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=34,
                offset=66,
                progress=progress,
            )

    def cut(self, event):
        version = self.cut_format.currentText()
        start = self.cut_start.value()
        stop = self.cut_stop.value()
        memory = self.memory
        if self.whence.checkState() == Qt.Checked:
            whence = 1
        else:
            whence = 0

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.cut_split.checkState() == Qt.Checked
        if split:
            split_size = int(self.cut_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.cut_compression.currentIndex()

        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select output measurement file", "", filter
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select output measurement file", "", filter
            )
            file_name = str(file_name)

        if file_name:
            progress = setup_progress(
                parent=self,
                title="Cutting measurement",
                message='Cutting "{}" from {}s to {}s'.format(
                    self.file_name, start, stop
                ),
                icon_name="cut",
            )

            # cut self.mdf
            target = self.mdf.cut
            kwargs = {"start": start, "stop": stop, "whence": whence}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            # convert mdf
            progress.setLabelText(
                "Converting from {} to {}".format(mdf.version, version)
            )

            target = mdf.convert
            kwargs = {"to": version, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=33,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText('Saving cut file "{}"'.format(file_name))

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=34,
                offset=66,
                progress=progress,
            )

    def export(self, event):
        export_type = self.export_type.currentText()

        single_time_base = self.single_time_base.checkState() == Qt.Checked
        time_from_zero = self.time_from_zero.checkState() == Qt.Checked
        use_display_names = self.use_display_names.checkState() == Qt.Checked
        empty_channels = self.empty_channels.currentText()
        mat_format = self.mat_format.currentText()
        raster = self.export_raster.value()
        oned_as = self.oned_as.currentText()

        filters = {
            "csv": "CSV files (*.csv)",
            "excel": "Excel files (*.xlsx)",
            "hdf5": "HDF5 files (*.hdf)",
            "mat": "Matlab MAT files (*.mat)",
            "parquet": "Apache Parquet files (*.parquet)",
        }

        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select export file", "", filters[export_type]
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select export file", "", filters[export_type]
            )
            file_name = str(file_name)

        if file_name:
            thr = Thread(
                target=self.mdf.export,
                kwargs={
                    "fmt": export_type,
                    "filename": file_name,
                    "single_time_base": single_time_base,
                    "use_display_names": use_display_names,
                    "time_from_zero": time_from_zero,
                    "empty_channels": empty_channels,
                    "format": mat_format,
                    "raster": raster,
                    "oned_as": oned_as,
                },
            )

            progress = QProgressDialog(
                "Exporting to {} ...".format(export_type), "Abort export", 0, 100
            )
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.setAutoClose(True)
            progress.setWindowTitle("Running export")
            icon = QIcon()
            icon.addPixmap(QPixmap(":/export.png"), QIcon.Normal, QIcon.Off)
            progress.setWindowIcon(icon)

            thr.start()

            cntr = 0

            while thr.is_alive():
                cntr += 1
                progress.setValue(cntr % 98)
                sleep(0.1)

            progress.cancel()

    def plot_pyqtgraph(self, event):
        try:
            iter(event)
            signals = event
        except:

            iterator = QTreeWidgetItemIterator(self.channels_tree)

            group = -1
            index = 0
            signals = []
            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    group += 1
                    index = 0
                    continue

                if item.checkState(0) == Qt.Checked:
                    group, index = item.entry
                    signals.append((None, group, index))

                index += 1
                iterator += 1

            signals = self.mdf.select(signals)

            signals = [
                sig
                for sig in signals
                if not sig.samples.dtype.names and len(sig.samples.shape) <= 1
            ]

        count = self.channel_selection.count()
        for i in range(count):
            self.channel_selection.takeItem(0)

        if self.info:
            self.info.setParent(None)
            self.info = None

        self.plot = Plot(signals, self.with_dots, self.step_mode, self)
        self.plot.range_modified.connect(self.range_modified)
        self.plot.range_removed.connect(self.range_removed)
        self.plot.range_modified_finished.connect(self.range_modified_finished)
        self.plot.cursor_removed.connect(self.cursor_removed)
        self.plot.cursor_moved.connect(self.cursor_moved)
        self.plot.cursor_move_finished.connect(self.cursor_move_finished)
        self.plot.xrange_changed.connect(self.xrange_changed)
        self.plot.show()

        for i, sig in enumerate(self.plot.signals):
            if sig.empty:
                name = "{} [has no samples]".format(sig.name)
            else:
                name = "{} ({})".format(sig.name, sig.unit)
            is_float = sig.samples.dtype.kind == "f"
            item = QListWidgetItem(self.channel_selection)
            it = ChannelDisplay(i, sig.unit, self)
            it.setAttribute(Qt.WA_StyledBackground)

            it.setName(name)
            it.setValue("")
            it.setColor(sig.color)
            item.setSizeHint(it.sizeHint())
            self.channel_selection.addItem(item)
            self.channel_selection.setItemWidget(item, it)

            it.color_changed.connect(self.plot.setColor)
            it.enable_changed.connect(self.plot.setSignalEnable)

        if self.splitter.count() > 1:
            old_plot = self.splitter.widget(1)
            old_plot.setParent(None)
            old_plot.hide()
        self.splitter.addWidget(self.plot)

        width = sum(self.splitter.sizes())

        self.splitter.setSizes((0.2 * width, 0.8 * width))
        QApplication.processEvents()

        self.plot.update_lines(force=True)

    def filter(self, event):
        iterator = QTreeWidgetItemIterator(self.filter_tree)
        memory = self.memory

        group = -1
        index = 0
        channels = []
        while iterator.value():
            item = iterator.value()
            if item.parent() is None:
                iterator += 1
                group += 1
                index = 0
                continue

            if item.checkState(0) == Qt.Checked:
                channels.append((None, group, index))

            index += 1
            iterator += 1

        version = self.filter_format.itemText(self.filter_format.currentIndex())

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.filter_split.checkState() == Qt.Checked
        if split:
            split_size = int(self.filter_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.filter_compression.currentIndex()

        if QT > 4:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Select output measurement file", "", filter
            )
        else:
            file_name = QFileDialog.getSaveFileName(
                self, "Select output measurement file", "", filter
            )
            file_name = str(file_name)

        if file_name:
            progress = setup_progress(
                parent=self,
                title="Filtering measurement",
                message='Filtering selected channels from "{}"'.format(self.file_name),
                icon_name="filter",
            )

            # filtering self.mdf
            target = self.mdf.filter
            kwargs = {"channels": channels, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            # convert mdf
            progress.setLabelText(
                "Converting from {} to {}".format(mdf.version, version)
            )

            target = mdf.convert
            kwargs = {"to": version, "memory": memory}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=33,
                offset=33,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText('Saving filtered file "{}"'.format(file_name))

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=34,
                offset=66,
                progress=progress,
            )

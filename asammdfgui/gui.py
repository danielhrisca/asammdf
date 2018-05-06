import os
import sys
import traceback
import re

from copy import deepcopy
from datetime import datetime
from functools import reduce, partial
from io import StringIO
from threading import Thread
from time import sleep

import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from asammdf import MDF, MDF2, MDF3, MDF4, SUPPORTED_VERSIONS
from asammdf import __version__ as libversion

import asammdfgui.main_window as main_window
import asammdfgui.file_widget as file_widget
import asammdfgui.search_dialog as search_dialog
import asammdfgui.search_widget as search_widget
import asammdfgui.channel_info_widget as channel_info_widget
import asammdfgui.channel_display_widget as channel_display_widget

import pyqtgraph as pg


__version__ = '0.1.0'
TERMINATED = object()
COLORS = [
    '#1f77b4',
    '#aec7e8',
    '#ff7f0e',
    '#ffbb78',
    '#2ca02c',
    '#98df8a',
    '#d62728',
    '#ff9896',
    '#9467bd',
    '#c5b0d5',
    '#8c564b',
    '#c49c94',
    '#e377c2',
    '#f7b6d2',
    '#7f7f7f',
    '#c7c7c7',
    '#bcbd22',
    '#dbdb8d',
    '#17becf',
    '#9edae5',
]

COLORS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]


def excepthook(exc_type, exc_value, tracebackobj):
    """
    Global function to catch unhandled exceptions.

    Parameters
    ----------
    exc_type : str
        exception type
    exc_value : int
        exception value
    tracebackobj : traceback
        traceback object
    """
    separator = '-' * 80
    notice = 'The following error was triggered:'

    now = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")

    info = StringIO()
    traceback.print_tb(tracebackobj, None, info)
    info.seek(0)
    info = info.read()

    errmsg = '{}\t \n{}'.format(exc_type, exc_value)
    sections = [now, separator, errmsg, separator, info]
    msg = '\n'.join(sections)

    print(msg)

    QMessageBox.warning(
        None,
        notice,
        msg,
    )


sys.excepthook = excepthook


def run_thread_with_progress(
        widget,
        target,
        kwargs,
        factor,
        offset,
        progress):
    termination_request = False

    thr = WorkerThread(
        target=target,
        kwargs=kwargs,
    )

    thr.start()

    while widget.progress is None:
        sleep(0.1)

    while thr.is_alive():
        QApplication.processEvents()
        termination_request = progress.wasCanceled()
        if termination_request:
            MDF._terminate = True
            MDF2._terminate = True
            MDF3._terminate = True
            MDF4._terminate = True
        else:
            progress.setValue(
                int(widget.progress[0] / widget.progress[1] * factor) + offset
            )
        sleep(0.1)

    if termination_request:
        MDF._terminate = False
        MDF2._terminate = False
        MDF3._terminate = False
        MDF4._terminate = False

    progress.setValue(factor + offset)

    if thr.error:
        widget.progress = None
        progress.cancel()
        raise Exception(thr.error)

    widget.progress = None

    if termination_request:
        return TERMINATED
    else:
        return thr.output


def setup_progress(parent, title, message, icon_name):
    progress = QProgressDialog(
        message,
        "",
        0,
        100,
        parent,
    )

    progress.setWindowModality(Qt.ApplicationModal)
    progress.setCancelButton(None)
    progress.setAutoClose(True)
    progress.setWindowTitle(title)
    icon = QIcon()
    icon.addPixmap(
        QPixmap(":/{}.png".format(icon_name)),
        QIcon.Normal,
        QIcon.Off,
    )
    progress.setWindowIcon(icon)
    progress.show()

    return progress


class WorkerThread(Thread):
    def __init__(self, output=None, *args, **kargs):
        super(WorkerThread, self).__init__(*args, **kargs)
        self.output = None
        self.error = ''

    def run(self):
        try:
            self.output = self._target(*self._args, **self._kwargs)
        except Exception as err:
            self.error = err


class TreeItem(QTreeWidgetItem):

    def __init__(self, entry, *args, **kwargs):

        super(TreeItem, self).__init__(*args, **kwargs)

        self.entry = entry


class Cursor(pg.InfiniteLine):

    def __init__(self, *args, **kwargs):

        super(Cursor, self).__init__(*args, label='{value:.6f}s', labelOpts={'position': 0.04}, **kwargs)

        self.addMarker('^', 0)
        self.addMarker('v', 1)
        self.label.show()

class FormatedAxis(pg.AxisItem):

    def __init__(self, *args, **kwargs):

        super(FormatedAxis, self).__init__(*args, **kwargs)

        self.format = 'phys'

    def tickStrings(self, values, scale, spacing):
        strns = []

        if self.format == 'phys':
            strns = super(FormatedAxis, self).tickStrings(values, scale, spacing)

        elif self.format == 'hex':
            for val in values:
                val = float(val)
                if val.is_integer():
                    val = hex(int(val))
                else:
                    val = ''
                strns.append(val)

        elif self.format == 'bin':
            for val in values:
                val = float(val)
                if val.is_integer():
                    val = bin(int(val))
                else:
                    val = ''
                strns.append(val)

        return strns


class Plot(pg.PlotWidget):
    cursor_moved = pyqtSignal()
    cursor_removed = pyqtSignal()
    range_removed = pyqtSignal()
    range_modified = pyqtSignal()
    range_modified_finished = pyqtSignal()
    cursor_move_finished = pyqtSignal()



    def __init__(self, signals, *args, **kwargs):

        super(Plot, self).__init__(*args, **kwargs)

        self.disableAutoRange()

        self.curvetype = pg.PlotCurveItem

        self.singleton = None
        self.region = None
        self.cursor1 = None
        self.cursor2 = None
        self.signals = signals
        for sig in self.signals:
            if sig.samples.dtype.kind == 'f':
                sig.format = '{:.6f}'
            else:
                sig.format = 'phys'
            sig.enable = True

        if self.signals:
            self.all_timebase = self.timebase = reduce(
                np.union1d,
                (
                    sig.timestamps
                    for sig in self.signals
                ),
            )
        else:
            self.all_timebase = self.timebase = None

        self.showGrid(x=True, y=True)

        self.plot_item = self.plotItem
        self.plot_item.hideAxis('left')
        self.layout = self.plot_item.layout
        self.scene_ = self.plot_item.scene()
        self.viewbox = self.plot_item.vb
        self.viewbox.sigResized.connect(self.update_views)
        # self.viewbox.enableAutoRange(
        #     axis=pg.ViewBox.XYAxes,
        #     enable=True,
        # )

        self.curve = self.curvetype(
            [],
            [],
        )

        axis = self.layout.itemAt(2, 0)
        axis.setParent(None)
        self.axis = FormatedAxis('left')
        self.layout.removeItem(axis)
        self.layout.addItem(self.axis, 2, 0)
        self.axis.linkToView(axis.linkedView())
        self.plot_item.axes['left']['item'] = self.axis
        self.plot_item.hideAxis('left')

        self.viewbox.addItem(
            self.curve
        )

        self.cursor_hint = pg.PlotDataItem(
            [],
            [],
            pen='#000000',
            symbolBrush='#000000',
            symbolPen='w',
            symbol='s',
            symbolSize=8,
        )
        self.viewbox.addItem(
            self.cursor_hint
        )

        self.view_boxes = []
        self.curves = []
        self.axes = []

        for i, sig in enumerate(self.signals):
            color = COLORS[i % 10]
            sig.color = color

            if len(sig.samples):
                sig.min = np.amin(sig.samples)
                sig.max = np.amax(sig.samples)
                sig.empty = False
            else:
                sig.empty = True

            axis = FormatedAxis("right")

            view_box = pg.ViewBox()
            # view_box.enableAutoRange(
            #     axis=pg.ViewBox.XYAxes,
            #     enable=True,
            # )

            axis.linkToView(view_box)
            axis.setLabel(sig.name, sig.unit, color=color)

            self.layout.addItem(axis, 2, i + 2)

            self.scene_.addItem(view_box)

            # curve = pg.PlotDataItem(
            curve = self.curvetype(
                sig.timestamps,
                sig.samples,
                pen=color,
                symbolBrush=color,
                symbolPen=color,
                symbol='o',
                symbolSize=4,
                # connect='pairs'
            )
            # curve.setDownsampling(ds=100, auto=True, method='peak')

            view_box.addItem(
                curve
            )

            view_box.setXLink(self.viewbox)
            view_box.enableAutoRange(
                axis=pg.ViewBox.XYAxes,
                enable=True,
            )
            # view_box.sigResized.connect(self.update_views)

            self.view_boxes.append(view_box)
            self.curves.append(curve)
            self.axes.append(axis)
            axis.hide()

        self.update_views()

        self.keyPressEvent(
            QKeyEvent(
                QEvent.KeyPress,
                Qt.Key_H,
                Qt.NoModifier,
            )
        )

    def setColor(self, index, color):
        self.signals[index].color = color
        self.curves[index].setPen(color)
        if self.curvetype == pg.PlotDataItem:
            self.curves[index].setSymbolPen(color)
            self.curves[index].setSymbolBrush(color)

        self.axes[index].setPen(color)

    def setSignalEnable(self, index, state):
        if state in (Qt.Checked, True, 1):
            self.signals[index].enable = True
        else:
            self.signals[index].enable = False

        selected_items = [
            index
            for index, sig in enumerate(self.signals)
            if sig.enable
        ]

        if len(selected_items) == 1:
            row = selected_items[0]
            if self.singleton != row:
                self.singleton = row
                sig = self.signals[row]
                color = sig.color

                self.plotItem.showAxis('left')
                for axis, viewbox, curve in zip(
                        self.axes,
                        self.view_boxes,
                        self.curves):
                    if curve.isVisible():
                        axis.hide()
                        viewbox.setYLink(None)
                        viewbox.setXLink(None)
                        curve.hide()

                sig_axis = self.axes[row]
                viewbox = self.view_boxes[row]
                viewbox.setXLink(self.viewbox)
                axis = self.axis

                axis.setRange(*sig_axis.range)
                axis.linkedView().setYRange(*sig_axis.range)

                viewbox.setYLink(axis.linkedView())

                self.curve.updateData(
                    sig.timestamps,
                    sig.samples,
                    pen=color,
                    symbolBrush=color,
                    symbolPen=color,
                    symbol='o',
                    symbolSize=4,
                )

                axis.setLabel(sig.name, sig.unit, color=color)

            self.curve.show()
            self.plotItem.showAxis('left')
            self.timebase = self.curve.xData

        else:
            self.plotItem.hideAxis('left')
            self.curve.hide()

            if len(selected_items):
                self.singleton = None

                for i, sig in enumerate(self.signals):
                    if sig.enable and not self.curves[i].isVisible():
                        # self.axes[i].show()
                        self.view_boxes[i].setYLink(None)
                        self.view_boxes[i].setXLink(self.viewbox)
                        self.curves[i].show()
                    elif not sig.enable and self.curves[i].isVisible():
                        self.view_boxes[i].setXLink(None)
                        self.axes[i].hide()
                        self.curves[i].hide()

            self.timebase = self.all_timebase
        if self.cursor1:
            self.cursor_move_finished.emit()

    def update_views(self):
        self.viewbox.linkedViewChanged(self.viewbox, self.viewbox.XAxis)
        for view_box in self.view_boxes:
            if view_box.isVisible():
                view_box.setGeometry(self.viewbox.sceneBoundingRect())
                view_box.linkedViewChanged(self.viewbox, view_box.XAxis)

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if key == Qt.Key_C:
            if self.cursor1 is None:
                start, stop = self.viewbox.viewRange()[0]
                self.cursor1 = Cursor(
                    pos=0,
                    angle=90,
                    movable=True,
                )
                self.plotItem.addItem(self.cursor1, ignoreBounds=True)
                self.cursor1.sigPositionChanged.connect(self.cursor_moved.emit)
                self.cursor1.sigPositionChangeFinished.connect(self.cursor_move_finished.emit)
                self.cursor1.setPos((start + stop) / 2)
                self.cursor_move_finished.emit()

            else:
                self.cursor_removed.emit()
                self.plotItem.removeItem(self.cursor1)
                self.cursor1.setParent(None)
                self.cursor1 = None

        elif key == Qt.Key_F:
            x_range, _ = self.viewbox.viewRange()
            for viewbox in self.view_boxes:
                viewbox.autoRange(padding=0)
            self.viewbox.autoRange(padding=0)
            self.viewbox.setXRange(*x_range, padding=0)
            if self.cursor1:
                self.cursor_moved.emit()

        elif key == Qt.Key_G:
            if self.plotItem.ctrl.xGridCheck.isChecked():
                self.showGrid(x=False, y=False)
            else:
                self.showGrid(x=True, y=True)

        elif key in (Qt.Key_I, Qt.Key_O):
            x_range, _ = self.viewbox.viewRange()
            delta = x_range[1] - x_range[0]
            step = delta * 0.05
            if key == Qt.Key_I:
                step = -step
            if self.cursor1:
                pos = self.cursor1.value()
                x_range = pos - delta / 2, pos + delta / 2
            self.viewbox.setXRange(
                x_range[0] - step,
                x_range[1] + step,
                padding=0,
            )

        elif key == Qt.Key_R:
            if self.region is None:

                self.region = pg.LinearRegionItem((0, 0))
                self.region.setZValue(-10)
                self.plotItem.addItem(self.region)
                self.region.sigRegionChanged.connect(self.range_modified.emit)
                self.region.sigRegionChangeFinished.connect(self.range_modified_finished.emit)
                start, stop = self.viewbox.viewRange()[0]
                start, stop = start + 0.1*(stop-start), stop - 0.1*(stop-start)
                self.region.setRegion((start, stop))

            else:
                self.region.setParent(None)
                self.region.hide()
                self.region = None
                self.range_removed.emit()

        elif key == Qt.Key_S:
            count = len(
                [
                    sig
                    for (sig, curve) in zip(self.signals, self.curves)
                    if not sig.empty and curve.isVisible()
                ]
            )

            if count:
                position = 0
                for i, signal in enumerate(self.signals):
                    if not signal.empty and self.curves[i].isVisible():
                        viewbox = self.view_boxes[i]
                        min_ = signal.min
                        max_ = signal.max
                        if min_ == max_:
                            min_, max_ = min_ - 1, max_ + 1

                        dim = (max_ - min_) * 1.1

                        max_ = min_ + dim * count
                        min_, max_ = min_ - dim * position, max_ - dim * position

                        viewbox.setYRange(min_, max_, padding=0)

                        position += 1
            else:
                xrange, _ = self.viewbox.viewRange()
                self.viewbox.autoRange(padding=0)
                self.viewbox.setXRange(*xrange, padding=0)
            if self.cursor1:
                self.cursor_moved.emit()

        elif key == Qt.Key_H and modifier == Qt.ControlModifier:
            for axis, signal in zip(self.axes, self.signals):
                if axis.isVisible() and signal.samples.dtype.kind in 'ui':
                    axis.format = 'hex'
                    signal.format = 'hex'
                    axis.hide()
                    axis.show()
            if self.axis.isVisible() and self.signals[self.singleton].samples.dtype.kind in 'ui':
                self.axis.format = 'hex'
                self.signals[self.singleton].format = 'hex'
                self.axis.hide()
                self.axis.show()
            if self.cursor1:
                self.cursor_moved.emit()

        elif key == Qt.Key_B and modifier == Qt.ControlModifier:
            for axis, signal in zip(self.axes, self.signals):
                if axis.isVisible() and signal.samples.dtype.kind in 'ui':
                    axis.format = 'bin'
                    signal.format = 'bin'
                    axis.hide()
                    axis.show()
            if self.axis.isVisible() and self.signals[self.singleton].samples.dtype.kind in 'ui':
                self.axis.format = 'bin'
                self.signals[self.singleton].format = 'bin'
                self.axis.hide()
                self.axis.show()
            if self.cursor1:
                self.cursor_moved.emit()

        elif key == Qt.Key_P and modifier == Qt.ControlModifier:
            for axis, signal in zip(self.axes, self.signals):
                if axis.isVisible() and signal.samples.dtype.kind in 'ui':
                    axis.format = 'phys'
                    signal.format = 'phys'
                    axis.hide()
                    axis.show()
            if self.axis.isVisible() and self.signals[self.singleton].samples.dtype.kind in 'ui':
                self.axis.format = 'phys'
                self.signals[self.singleton].format = 'phys'
                self.axis.hide()
                self.axis.show()
            if self.cursor1:
                self.cursor_moved.emit()

        elif key in (Qt.Key_Left, Qt.Key_Right):
            if self.cursor1:
                pos = self.cursor1.value()
                dim = len(self.timebase)
                if dim:
                    pos = np.searchsorted(self.timebase, pos)
                    if key == Qt.Key_Right:
                        pos += 1
                    else:
                        pos -= 1
                    pos = np.clip(pos, 0, dim-1)
                    self.cursor1.setValue(self.timebase[pos])
                else:
                    if key == Qt.Key_Right:
                        pos += 1
                    else:
                        pos -= 1

                    self.cursor1.setValue(pos)

        elif key == Qt.Key_H:
            for viewbox in self.view_boxes:
                viewbox.autoRange(padding=0)
            self.viewbox.autoRange(padding=0)
            if self.cursor1:
                self.cursor_moved.emit()

        else:
            super(Plot, self).keyPressEvent(event)


class TreeWidget(QTreeWidget):

    def __init__(self, *args, **kwargs):

        super(TreeWidget, self).__init__(*args, **kwargs)

        self.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            elif len(selected_items) == 1:
                item = selected_items[0]
                checked = item.checkState(0)
                if checked == Qt.Checked:
                    item.setCheckState(0, Qt.Unchecked)
                else:
                    item.setCheckState(0, Qt.Checked)
            else:
                if any(item.checkState(0) == Qt.Unchecked for item in selected_items):
                    checked = Qt.Checked
                else:
                    checked = Qt.Unchecked
                for item in selected_items:
                    item.setCheckState(0, checked)
        else:
            super(TreeWidget, self).keyPressEvent(event)


class ListWidget(QListWidget):

    itemsDeleted = pyqtSignal(list)

    def __init__(self, *args, **kwargs):

        super(ListWidget, self).__init__(*args, **kwargs)

        self.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )

        self.setAlternatingRowColors(True)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Delete:
            selected_items = self.selectedItems()
            deleted = []
            for item in selected_items:
                row = self.row(item)
                deleted.append(row)
                self.takeItem(row)
            if deleted:
                self.itemsDeleted.emit(deleted)
        elif key == Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            try:
                states = [
                    self.itemWidget(item).display.checkState()
                    for item in selected_items
                ]

                if any(state == Qt.Unchecked for state in states):
                    state = Qt.Checked
                else:
                    state = Qt.Unchecked
                for item in selected_items:
                    wid = self.itemWidget(item)
                    wid.display.setCheckState(state)
            except:
                pass
        else:
            super(ListWidget, self).keyPressEvent(event)


class ChannelInfoWidget(QWidget, channel_info_widget.Ui_ChannelInfo):
    def __init__(self, channel, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.channel_label.setText(
            channel.metadata()
        )

        if channel.conversion:
            self.conversion_label.setText(
                channel.conversion.metadata()
            )

        if channel.source:
            self.source_label.setText(
                channel.source.metadata()
            )


class AdvancedSearch(QDialog, search_dialog.Ui_SearchDialog):
    def __init__(self, channels_db, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.result = []
        self.channels_db = sorted(set(channels_db))

        self.apply_btn.clicked.connect(self._apply)
        self.apply_all_btn.clicked.connect(self._apply_all)
        self.cancel_btn.clicked.connect(self._cancel)

        self.search_box.textChanged.connect(self.search_text_changed)

    def search_text_changed(self, text):
        if len(text) >= 2:
            pattern = text.replace('*', '.*')
            pattern = re.compile('(?i){}'.format(pattern))
            matches = [
                name
                for name in self.channels_db
                if pattern.search(name)
            ]
            self.matches.clear()
            self.matches.addItems(matches)

    def _apply(self, event):
        self.result = [
            item.text()
            for item in self.matches.selectedItems()
        ]
        print('in', self.result)
        self.close()

    def _apply_all(self, event):
        count = self.matches.count()
        self.result = [
            self.matches.item(i).text()
            for i in range(count)
        ]
        print('in', self.result)
        self.close()

    def _cancel(self, event):
        self.result = []
        self.close()


class ChannelInfoDialog(QDialog):
    def __init__(self, channel, *args, **kwargs):
        super(QDialog, self).__init__(*args, **kwargs)

        self.setWindowFlags(Qt.Window)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle(channel.name)

        layout.addWidget(ChannelInfoWidget(channel, self))

        self.setStyleSheet("font: 8pt \"Consolas\";}")

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/info.png"),
            QIcon.Normal,
            QIcon.Off,
        )

        self.setWindowIcon(icon)
        self.setGeometry(
            240,
            60,
            1200,
            600,
        )

        screen = QApplication.desktop().screenGeometry()
        self.move(
            (screen.width() - 1200) // 2,
            (screen.height() - 600) // 2,
        )


class ChannelDisplay(QWidget, channel_display_widget.Ui_ChannelDiplay):

    color_changed = pyqtSignal(int, str)
    enable_changed = pyqtSignal(int, int)

    def __init__(self, index, *args, **kargs):
        super(ChannelDisplay, self).__init__(*args, **kargs)
        self.setupUi(self)

        self.color = '#ff0000'
        self._value_prefix = ''
        self._value = ''
        self._name = ''
        self.fmt = '{}'
        self.index = index

        self.color_btn.clicked.connect(self.select_color)
        self.display.stateChanged.connect(self.display_changed)

    def display_changed(self, state):
        state = self.display.checkState()
        self.enable_changed.emit(self.index, state)

    def select_color(self):
        color = QColorDialog.getColor(QColor(self.color)).name()
        self.setColor(color)

        self.color_changed.emit(self.index, color)

    def setFmt(self, fmt):
        if fmt == 'hex':
            self.fmt = '0x{:X}'
        elif fmt == 'bin':
            self.fmt = '0b{:b}'
        elif fmt == 'phys':
            self.fmt = '{}'
        else:
            self.fmt = fmt

    def setColor(self, color):
        self.color = color
        self.setName(self._name)
        self.setValue(self._value)
        self.color_btn.setStyleSheet("background-color: {};".format(color))

    def setName(self, text=''):
        self._name = text
        self.name.setText(
            "<html><head/><body><p><span style=\" color:{};\">{}</span></p></body></html>".format(
                self.color,
                self._name,
            )
        )

    def setPrefix(self, text=''):
        self._value_prefix = text

    def setValue(self, value):
        self._value = value
        template = "<html><head/><body><p><span style=\" color:{{}};\">{{}}{}</span></p></body></html>"
        if value not in ('', 'n.a.'):
            template = template.format(
                self.fmt
            )
        else:
            template = template.format(
                '{}'
            )
        self.value.setText(
            template.format(
                self.color,
                self._value_prefix,
                value,
            )
        )


class SearchWidget(QWidget, search_widget.Ui_SearchWidget):

    selectionChanged = pyqtSignal()

    def __init__(self, channels_db, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.channels_db = channels_db

        self.matches = 0
        self.current_index = 1
        self.entries = []

        completer = QCompleter(
            sorted(self.channels_db, key=lambda x: x.lower()),
            self,
        )
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setModelSorting(QCompleter.CaseInsensitivelySortedModel)
        completer.setFilterMode(Qt.MatchContains)
        self.search.setCompleter(completer)

        self.search.textChanged.connect(self.display_results)

        self.up_btn.clicked.connect(self.up)
        self.down_btn.clicked.connect(self.down)

    def down(self, event):
        if self.matches:
            self.current_index += 1
            if self.current_index >= self.matches:
                self.current_index = 0
            self.label.setText('{} of {}'.format(self.current_index + 1, self.matches))
            self.selectionChanged.emit()

    def up(self, event):
        if self.matches:
            self.current_index -= 1
            if self.current_index < 0:
                self.current_index = self.matches - 1
            self.label.setText('{} of {}'.format(self.current_index + 1, self.matches))
            self.selectionChanged.emit()

    def set_search_option(self, option):
        if option == 'Match start':
            self.search.completer().setFilterMode(Qt.MatchStartsWith)
        elif option == 'Match contains':
            self.search.completer().setFilterMode(Qt.MatchContains)

    def display_results(self, text):
        channel_name = text.strip()
        if channel_name in self.channels_db:
            self.entries = self.channels_db[channel_name]
            self.matches = len(self.entries)
            self.label.setText('1 of {}'.format(self.matches))
            self.current_index = 0
            self.selectionChanged.emit()

        else:
            self.label.setText('No match')
            self.matches = 0
            self.current_index = 0
            self.entries = []


class FileWidget(QWidget, file_widget.Ui_file_widget):
    def __init__(self, file_name, memory, parent=None):
        super().__init__(parent)
        self.plot = None
        self.setupUi(self)
        self.file_name = file_name
        self.progress = None
        self.mdf = None
        self.memory = memory

        progress = QProgressDialog(
            'Opening "{}"'.format(self.file_name),
            "",
            0,
            100,
            self.parent(),
        )

        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setAutoClose(True)
        progress.setWindowTitle('Opening measurement')
        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/open.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        progress.setWindowIcon(icon)
        progress.show()

        if file_name.lower().endswith('dl3'):
            progress.setLabelText('Converting from dl3 to mdf')
            try:
                import win32com.client

                index = 0
                while True:
                    mdf_name = '{}.{}.mdf'.format(
                        file_name,
                        index,
                    )
                    if os.path.exists(mdf_name):
                        index += 1
                    else:
                        break

                datalyser = win32com.client.Dispatch('Datalyser3.Datalyser3_COM')
                datalyser.DCOM_set_datalyser_visibility(False)
                datalyser.DCOM_convert_file_mdf_dl3(
                    file_name,
                    mdf_name,
                    0
                )
                datalyser.DCOM_TerminateDAS()
                file_name = mdf_name
            except:
                return

        target = MDF
        kwargs = {
            'name': file_name,
            'memory': memory,
            'callback': self.update_progress,
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

        progress.setLabelText('Loading graphical elements')

        progress.setValue(35)

        self.filter_field = SearchWidget(
            deepcopy(self.mdf.channels_db),
            self,
        )

        progress.setValue(37)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)

        channel_and_search = QWidget(splitter)

        self.channels_tree = TreeWidget(channel_and_search)
        self.search_field = SearchWidget(
            deepcopy(self.mdf.channels_db),
            channel_and_search,
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
                self.new_search_result,
                tree=self.filter_tree,
                search=self.filter_field,
            )
        )

        vbox = QVBoxLayout(channel_and_search)
        self.advanced_search_btn = QPushButton('Advanced search & select')
        self.advanced_search_btn.clicked.connect(self.search)
        vbox.addWidget(self.search_field)
        vbox.addWidget(self.advanced_search_btn)
        vbox.addWidget(self.channels_tree, 1)
        channel_and_search.setLayout(vbox)

        self.clear_channels_btn = QPushButton('Reset selection', channel_and_search)
        icon = QIcon()
        icon.addPixmap(QPixmap(":/erase.png"), QIcon.Normal, QIcon.Off)
        self.clear_channels_btn.setIcon(icon)
        self.clear_channels_btn.setObjectName("clear_channels_btn")
        vbox.addWidget(self.clear_channels_btn)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.load_channel_list_btn = QPushButton('Load channel list', channel_and_search)
        icon1 = QIcon()
        icon1.addPixmap(QPixmap(":/open.png"), QIcon.Normal, QIcon.Off)
        self.load_channel_list_btn.setIcon(icon1)
        self.load_channel_list_btn.setObjectName("load_channel_list_btn")
        self.horizontalLayout.addWidget(self.load_channel_list_btn)
        self.save_channel_list_btn = QPushButton('Save channel list', channel_and_search)
        icon2 = QIcon()
        icon2.addPixmap(QPixmap(":/save.png"), QIcon.Normal, QIcon.Off)
        self.save_channel_list_btn.setIcon(icon2)
        self.save_channel_list_btn.setObjectName("save_channel_list_btn")
        self.horizontalLayout.addWidget(self.save_channel_list_btn)
        vbox.addLayout(self.horizontalLayout)
        self.line_12 = QFrame(self.verticalLayoutWidget)
        self.line_12.setFrameShape(QFrame.HLine)
        self.line_12.setFrameShadow(QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        vbox.addWidget(self.line_12)
        self.plot_btn = QPushButton('Plot', channel_and_search)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_btn.sizePolicy().hasHeightForWidth())
        self.plot_btn.setSizePolicy(sizePolicy)
        self.plot_btn.setMaximumSize(QSize(16777215, 100))
        icon3 = QIcon()
        icon3.addPixmap(QPixmap(":/graph.png"), QIcon.Normal, QIcon.Off)
        self.plot_btn.setIcon(icon3)
        self.plot_btn.setObjectName("plot_btn")
        vbox.addWidget(self.plot_btn)

        selection_list = QWidget(splitter)
        self.channel_selection = ListWidget(selection_list)
        self.channel_selection.setAlternatingRowColors(False)

        vbox = QVBoxLayout(selection_list)

        hbox = QHBoxLayout(selection_list)
        hbox.addWidget(QLabel('Selected channels'))
        self.cursor_info = QLabel('')
        self.cursor_info.setTextFormat(Qt.RichText)
        self.cursor_info.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        hbox.addWidget(self.cursor_info)

        vbox.addLayout(hbox)

        vbox.addWidget(self.channel_selection)

        self.filter_layout.addWidget(self.filter_field, 0, 0, 1, 1)

        self.channels_tree.itemDoubleClicked.connect(self.show_channel_info)
        self.filter_tree.itemDoubleClicked.connect(self.show_channel_info)

        self.channels_layout.insertWidget(0, splitter)
        self.filter_layout.addWidget(self.filter_tree, 1, 0, 8, 1)

        groups_nr = len(self.mdf.groups)

        self.channels_tree.setHeaderLabel('Channels')
        self.channels_tree.setToolTip(
            'Double click channel to see extended information'
        )
        self.filter_tree.setHeaderLabel('Channels')
        self.filter_tree.setToolTip(
            'Double click channel to see extended information'
        )

        for i, group in enumerate(self.mdf.groups):
            channel_group = QTreeWidgetItem()
            filter_channel_group = QTreeWidgetItem()
            channel_group.setText(
                0,
                "Channel group {}".format(i)
            )
            filter_channel_group.setText(
                0,
                "Channel group {}".format(i)
            )
            channel_group.setFlags(
                channel_group.flags()
                | Qt.ItemIsTristate
                | Qt.ItemIsUserCheckable
            )
            filter_channel_group.setFlags(
                channel_group.flags()
                | Qt.ItemIsTristate
                | Qt.ItemIsUserCheckable
            )

            self.channels_tree.addTopLevelItem(channel_group)
            self.filter_tree.addTopLevelItem(filter_channel_group)

            for j, ch in enumerate(group['channels']):

                name = self.mdf.get_channel_name(group=i, index=j)
                channel = TreeItem((i, j), channel_group)
                channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                channel.setText(0, name)
                channel.setCheckState(0, Qt.Unchecked)

                channel = TreeItem((i, j), filter_channel_group)
                channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                channel.setText(0, name)
                channel.setCheckState(0, Qt.Unchecked)

            if self.mdf.version >= '4.00':
                for j, ch in enumerate(group['logging_channels'], 1):
                    name = ch.name

                    channel = TreeItem((i, -j), channel_group)
                    channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                    channel.setText(0, name)
                    channel.setCheckState(0, Qt.Unchecked)

                    channel = TreeItem((i, -j), filter_channel_group)
                    channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                    channel.setText(0, name)
                    channel.setCheckState(0, Qt.Unchecked)

            progress.setValue(37 + int(53 * (i+1) / groups_nr))
            QApplication.processEvents()

        progress.setValue(90)

        self.resample_format.insertItems(
            0,
            SUPPORTED_VERSIONS,
        )
        index = self.resample_format.findText(self.mdf.version)
        if index >= 0:
            self.resample_format.setCurrentIndex(index)
        self.resample_compression.insertItems(
            0,
            (
                'no compression',
                'deflate',
                'transposed deflate',
            ),
        )
        self.resample_split_size.setValue(10)
        self.resample_btn.clicked.connect(self.resample)

        self.filter_format.insertItems(
            0,
            SUPPORTED_VERSIONS,
        )
        index = self.filter_format.findText(self.mdf.version)
        if index >= 0:
            self.filter_format.setCurrentIndex(index)
        self.filter_compression.insertItems(
            0,
            (
                'no compression',
                'deflate',
                'transposed deflate',
            ),
        )
        self.filter_split_size.setValue(10)
        self.filter_btn.clicked.connect(self.filter)

        self.convert_format.insertItems(
            0,
            SUPPORTED_VERSIONS,
        )
        self.convert_compression.insertItems(
            0,
            (
                'no compression',
                'deflate',
                'transposed deflate',
            ),
        )
        self.convert_split_size.setValue(10)
        self.convert_btn.clicked.connect(self.convert)

        self.cut_format.insertItems(
            0,
            SUPPORTED_VERSIONS,
        )
        index = self.cut_format.findText(self.mdf.version)
        if index >= 0:
            self.cut_format.setCurrentIndex(index)
        self.cut_compression.insertItems(
            0,
            (
                'no compression',
                'deflate',
                'transposed deflate',
            ),
        )
        self.cut_split_size.setValue(10)
        self.cut_btn.clicked.connect(self.cut)

        self.cut_interval.setText(
            'Unknown measurement interval'
        )

        progress.setValue(99)

        self.empty_channels.insertItems(
            0,
            (
                'zeros',
                'skip',
            ),
        )
        self.mat_format.insertItems(
            0,
            (
                '4',
                '5',
                '7.3',
            ),
        )
        self.export_type.insertItems(
            0,
            (
                'csv',
                'excel',
                'hdf5',
                'mat',
            ),
        )
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
        self.channel_selection.itemSelectionChanged.connect(self.channel_selection_modified)

    def search(self):
        dlg = AdvancedSearch(self.mdf.channels_db, self)
        dlg.setModal(True)
        dlg.exec_()
        result = dlg.result
        if result:
            iterator = QTreeWidgetItemIterator(
                self.channels_tree,
            )

            while iterator.value():
                item = iterator.value()
                if item.parent() is None:
                    iterator += 1
                    continue

                channel_name = item.text(0)
                if channel_name in result:
                    item.setCheckState(0, Qt.Checked)
                    result.pop(result.index(channel_name))

                iterator += 1

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
        count = len(
            [
                sig
                for sig in self.plot.signals
                if sig.enable
            ]
        )
        rows = self.channel_selection.count()

        for i in range(rows):
            item = self.channel_selection.item(i)
            if count > 1 and item in selected_items:
                if self.plot.signals[i].enable and not self.plot.axes[i].isVisible():
                    self.plot.axes[i].show()
            else:
                if self.plot.axes[i].isVisible():
                    self.plot.axes[i].hide()

    def save_channel_list(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output channel list file",
            '',
            "TXT files (*.txt)",
        )
        if file_name:
            with open(file_name, 'w') as output:
                iterator = QTreeWidgetItemIterator(
                    self.channels_tree,
                )

                signals = []
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue

                    if item.checkState(0) == Qt.Checked:
                        signals.append(item.text(0))

                    iterator += 1

                output.write('\n'.join(signals))

    def load_channel_list(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select channel list file",
            '',
            "TXT files (*.txt)",
        )
        if file_name:
            with open(file_name, 'r') as infile:
                channels = [
                    line.strip()
                    for line in infile.readlines()
                ]
                channels = [
                    name
                    for name in channels
                    if name
                ]

            iterator = QTreeWidgetItemIterator(
                self.channels_tree,
            )

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
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output filter list file",
            '',
            "TXT files (*.txt)",
        )
        if file_name:
            with open(file_name, 'w') as output:
                iterator = QTreeWidgetItemIterator(
                    self.filter_tree,
                )

                signals = []
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue

                    if item.checkState(0) == Qt.Checked:
                        signals.append(item.text(0))

                    iterator += 1

                output.write('\n'.join(signals))

    def load_filter_list(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select filter list file",
            '',
            "TXT files (*.txt)",
        )
        if file_name:
            with open(file_name, 'r') as infile:
                channels = [
                    line.strip()
                    for line in infile.readlines()
                ]
                channels = [
                    name
                    for name in channels
                    if name
                ]

            iterator = QTreeWidgetItemIterator(
                self.filter_tree,
            )

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

            right = np.searchsorted(x, position, side='right')
            if right == 0:
                next_pos = x[0]
            elif right == dim:
                next_pos = x[-1]
            else:
                if position - x[right-1] < x[right] - position:
                    next_pos = x[right-1]
                else:
                    next_pos = x[right]
            self.plot.cursor1.setPos(next_pos)

        self.plot.cursor_hint.setData(
            x=[],
            y=[],
        )

    def cursor_moved(self):
        position = self.plot.cursor1.value()

        x = self.plot.timebase

        if x is not None and len(x):
            dim = len(x)
            position = self.plot.cursor1.value()

            right = np.searchsorted(x, position, side='right')
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
                    self.plot.view_boxes,
                    self.plot.signals,
                    self.plot.curves):
                if curve.isVisible():
                    index = np.argwhere(
                        sig.timestamps == next_pos
                    ).flatten()
                    if len(index):
                        _, (y_min, y_max) = viewbox.viewRange()

                        sample = sig.samples[index[0]]
                        sample = (sample - y_min) / (y_max - y_min) * (hint_max - hint_min) + hint_min

                        y.append(sample)

            if self.plot.curve.isVisible():
                timestamps = self.plot.curve.xData
                samples = self.plot.curve.yData
                index = np.argwhere(
                    timestamps == next_pos
                ).flatten()
                if len(index):
                    _, (y_min, y_max) = self.plot.viewbox.viewRange()

                    sample = samples[index[0]]
                    sample = (sample - y_min) / (y_max - y_min) * (hint_max - hint_min) + hint_min

                    y.append(sample)

            self.plot.viewbox.setYRange(hint_min, hint_max, padding=0)
            self.plot.cursor_hint.setData(
                x=[next_pos, ] * len(y),
                y=y,
            )
            self.plot.cursor_hint.show()

        if not self.plot.region:
            self.cursor_info.setText('t = {:.6f}s'.format(position))
            for i, signal in enumerate(self.plot.signals):
                samples = signal.cut(position, position).samples
                item = self.channel_selection.item(i)
                item = self.channel_selection.itemWidget(item)

                item.setPrefix('= ')
                item.setFmt(signal.format)

                if len(samples):
                    item.setValue(samples[0])
                else:
                    item.setValue('n.a.')

    def cursor_removed(self):
        for i, signal in enumerate(self.plot.signals):
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            if not self.plot.region:
                self.cursor_info.setText('')
                item.setPrefix('')
                item.setValue('')

    def range_modified(self):
        start, stop = self.plot.region.getRegion()
        self.cut_start.setValue(start)
        self.cut_stop.setValue(stop)

        self.cursor_info.setText(
            (
                '< html > < head / > < body >'
                '< p >t1 = {:.6f}s< / p > '
                '< p >t2 = {:.6f}s< / p > '
                '< p >Δt = {:.6f}s< / p > '
                '< / body > < / html >'
            ).format(
                start,
                stop,
                stop-start,
            )
        )

        for i, signal in enumerate(self.plot.signals):
            samples = signal.cut(start, stop).samples
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            item.setPrefix('Δ = ')
            item.setFmt(signal.format)

            if len(samples):
                if samples.dtype.kind in 'ui':
                    delta = np.int64(np.float64(samples[-1]) - np.float64(samples[0]))
                else:
                    delta = samples[-1] - samples[0]

                item.setValue(delta)

            else:
                item.setValue('n.a.')

    def range_modified_finished(self):
        start, stop = self.plot.region.getRegion()

        if self.plot.timebase is not None and len(self.plot.timebase):
            timebase = self.plot.timebase
            dim = len(timebase)

            right = np.searchsorted(
                timebase,
                start,
                side='right',
            )
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

            right = np.searchsorted(
                timebase,
                stop,
                side='right',
            )
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

            self.plot.region.setRegion(
                (start, stop)
            )


    def range_removed(self):
        for i, signal in enumerate(self.plot.signals):
            item = self.channel_selection.item(i)
            item = self.channel_selection.itemWidget(item)

            item.setPrefix('')
            item.setValue('')
            self.cursor_info.setText('')
        if self.plot.cursor1:
            self.plot.cursor_moved.emit()

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
                'Cut interval ({:.6f}s - {:.6f}s)'.format(
                    *time_range
                )
            )
        else:
            self.cut_start.setRange(0, 0)
            self.cut_stop.setRange(0, 0)

            self.cut_interval.setText('Empty measurement')

    def update_progress(self, current_index, max_index):
        self.progress = current_index, max_index

    def show_channel_info(self, item, column):
        if item and item.parent():
            group, index = item.entry

            channel = self.mdf.get_channel_metadata(
                group=group,
                index=index,
            )

            msg = ChannelInfoDialog(channel, self)
            msg.show()

    def clear_filter(self):
        iterator = QTreeWidgetItemIterator(
            self.filter_tree,
        )

        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, Qt.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def clear_channels(self):
        iterator = QTreeWidgetItemIterator(
            self.channels_tree,
        )

        while iterator.value():
            item = iterator.value()
            item.setCheckState(0, Qt.Unchecked)

            if item.parent() is None:
                item.setExpanded(False)

            iterator += 1

    def new_search_result(self, tree, search):
        group_index, channel_index = search.entries[search.current_index]

        grp = self.mdf.groups[group_index]
        channel_count = len(grp['channels'])

        iterator = QTreeWidgetItemIterator(
            tree,
        )

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

                if channel_index >= 0 and index == channel_index \
                        or channel_index < 0 and index == -channel_index -1 + channel_count:
                    tree.scrollToItem(
                        item,
                        QAbstractItemView.PositionAtTop,
                    )
                    item.setSelected(True)

            index += 1
            iterator += 1

    def close(self):
        mdf_name = self.mdf.name
        self.mdf.close()
        if self.file_name.lower().endswith('dl3'):
            os.remove(mdf_name)

    def convert(self, event):
        version = self.convert_format.currentText()

        memory = self.memory

        if version < '4.00':
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

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            '',
            filter,
        )

        if file_name:

            progress = setup_progress(
                parent=self,
                title='Converting measurement',
                message='Converting "{}" from {} to {} '.format(
                    self.file_name,
                    self.mdf.version,
                    version,
                ),
                icon_name='convert',
            )

            # convert self.mdf
            target = self.mdf.convert
            kwargs = {
                'to': version,
                'memory': memory,
            }

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
            progress.setLabelText(
                'Saving converted file "{}"'.format(
                    file_name,
                )
            )

            target = mdf.save
            kwargs = {
                'dst': file_name,
                'compression': compression,
                'overwrite': True,
            }

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

        if version < '4.00':
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

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            '',
            filter,
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title='Resampling measurement',
                message='Resampling "{}" to {}s raster '.format(
                    self.file_name,
                    raster,
                ),
                icon_name='resample',
            )

            # resample self.mdf
            target = self.mdf.resample
            kwargs = {
                'raster': raster,
                'memory': memory,
            }

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
                'Converting from {} to {}'.format(
                    mdf.version,
                    version,
                )
            )

            target = mdf.convert
            kwargs = {
                'to': version,
                'memory': memory,
            }

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
            progress.setLabelText(
                'Saving resampled file "{}"'.format(
                    file_name,
                )
            )

            target = mdf.save
            kwargs = {
                'dst': file_name,
                'compression': compression,
                'overwrite': True,
            }

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

        if version < '4.00':
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

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            '',
            filter,
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title='Cutting measurement',
                message='Cutting "{}" from {}s to {}s'.format(
                    self.file_name,
                    start,
                    stop,
                ),
                icon_name='cut',
            )

            # cut self.mdf
            target = self.mdf.cut
            kwargs = {
                'start': start,
                'stop': stop,
            }

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
                'Converting from {} to {}'.format(
                    mdf.version,
                    version,
                )
            )

            target = mdf.convert
            kwargs = {
                'to': version,
                'memory': memory,
            }

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
            progress.setLabelText(
                'Saving cut file "{}"'.format(
                    file_name,
                )
            )

            target = mdf.save
            kwargs = {
                'dst': file_name,
                'compression': compression,
                'overwrite': True,
            }

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

        filters = {
            'csv': "CSV files (*.csv)",
            'excel': "Excel files (*.xlsx)",
            'hdf5': "HDF5 files (*.hdf)",
            'mat': "Matlab MAT files (*.mat)",
        }

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select export file",
            '',
            filters[export_type],
        )

        if file_name:
            thr = Thread(
                target=self.mdf.export,
                kwargs={
                    'fmt': export_type,
                    'filename': file_name,
                    'single_time_base': single_time_base,
                    'use_display_names': use_display_names,
                    'time_from_zero': time_from_zero,
                    'empty_channels': empty_channels,
                    'format': mat_format,
                    'raster': raster,
                },
            )

            progress = QProgressDialog(
                "Exporting to {} ...".format(export_type),
                "Abort export",
                0,
                100)
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.setAutoClose(True)
            progress.setWindowTitle('Running export')
            icon = QIcon()
            icon.addPixmap(
                QPixmap(":/export.png"),
                QIcon.Normal,
                QIcon.Off,
            )
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

            iterator = QTreeWidgetItemIterator(
                self.channels_tree,
            )

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
                if not sig.samples.dtype.names
                and sig.samples.dtype.kind not in 'SV'
                and len(sig.samples.shape) <= 1
            ]

        count = self.channel_selection.count()
        for i in range(count):
            self.channel_selection.takeItem(0)

        self.plot = Plot(signals, self)
        self.plot.range_modified.connect(self.range_modified)
        self.plot.range_removed.connect(self.range_removed)
        self.plot.range_modified_finished.connect(self.range_modified_finished)
        self.plot.cursor_removed.connect(self.cursor_removed)
        self.plot.cursor_moved.connect(self.cursor_moved)
        self.plot.cursor_move_finished.connect(self.cursor_move_finished)
        self.plot.show()

        for i, sig in enumerate(self.plot.signals):
            if sig.empty:
                name = '{} [has no samples]'.format(sig.name)
            else:
                name = '{} ({})'.format(sig.name, sig.unit)
            # self.channel_selection.addItem(name)
            # self.channel_selection.item(i).setForeground(
            #     QColor(sig.color)
            # )
            item = QListWidgetItem(self.channel_selection)
            it = ChannelDisplay(self)
            it.setAttribute(Qt.WA_StyledBackground)
            it.index = i

            it.setName(name)
            it.setValue('')
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

        self.splitter.setSizes(
            (0.2 * width, 0.8*width)
        )

    def filter(self, event):
        iterator = QTreeWidgetItemIterator(
            self.filter_tree,
        )
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

        version = self.filter_format.itemText(
            self.filter_format.currentIndex())

        if version < '4.00':
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

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            '',
            filter,
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title='Filtering measurement',
                message='Filtering selected channels from "{}"'.format(
                    self.file_name,
                ),
                icon_name='filter',
            )

            # filtering self.mdf
            target = self.mdf.filter
            kwargs = {
                'channels': channels,
                'memory': memory,
            }

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
                'Converting from {} to {}'.format(
                    mdf.version,
                    version,
                )
            )

            target = mdf.convert
            kwargs = {
                'to': version,
                'memory': memory,
            }

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
            progress.setLabelText(
                'Saving filtered file "{}"'.format(
                    file_name,
                )
            )

            target = mdf.save
            kwargs = {
                'dst': file_name,
                'compression': compression,
                'overwrite': True,
            }

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=34,
                offset=66,
                progress=progress,
            )


class MainWindow(QMainWindow, main_window.Ui_PyMDFMainWindow):
    def __init__(self, parent=None):

        super().__init__(parent)

        self.progress = None

        self.setupUi(self)

        self.files.tabCloseRequested.connect(self.close_file)

        self.concatenate.toggled.connect(self.function_select)
        self.cs_btn.clicked.connect(self.cs_clicked)
        self.cs_format.insertItems(
            0,
            SUPPORTED_VERSIONS,
        )
        self.cs_compression.insertItems(
            0,
            (
                'no compression',
                'deflate',
                'transposed deflate',
            ),
        )
        self.cs_split_size.setValue(10)

        self.files_list = ListWidget(self)
        self.files_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.files_layout.addWidget(self.files_list, 0, 0, 1, 2)
        self.files_list.itemDoubleClicked.connect(self.delete_item)

        self.statusbar.addPermanentWidget(
            QLabel(
                'asammdf {}'.format(
                    libversion,
                )
            )
        )

        menu = self.menubar.addMenu('File')
        open_group = QActionGroup(self)
        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/open.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, 'Open')
        action.triggered.connect(
            self.open
        )
        open_group.addAction(action)
        menu.addActions(open_group.actions())

        menu = QMenu('Settings', self.menubar)
        self.menubar.addMenu(menu)

        # memory option menu
        memory_option = QActionGroup(self)

        for option in (
                'full',
                'low',
                'minimum'):

            action = QAction(option)
            action.setCheckable(True)
            memory_option.addAction(action)
            action.triggered.connect(
                partial(self.set_memory_option, option)
            )

            if option == 'minimum':
                action.setChecked(True)

        submenu = QMenu('Memory', self.menubar)
        submenu.addActions(memory_option.actions())
        menu.addMenu(submenu)

        # search mode menu
        search_option = QActionGroup(self)

        for option in (
                'Match start',
                'Match contains'):

            action = QAction(option)
            action.setCheckable(True)
            search_option.addAction(action)
            action.triggered.connect(
                partial(self.set_search_option, option)
            )

            if option == 'Match start':
                action.setChecked(True)

        submenu = QMenu('Search', self.menubar)
        submenu.addActions(search_option.actions())
        menu.addMenu(submenu)

        # plot option menu
        plot_actions = QActionGroup(self)

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/fit.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, '{: <20}\tF'.format('Fit trace'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_F)
        )
        action.setShortcut(Qt.Key_F)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/grid.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, '{: <20}\tG'.format('Grid'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_G)
        )
        action.setShortcut(Qt.Key_G)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/home.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, '{: <20}\tH'.format('Home'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_H)
        )
        action.setShortcut(Qt.Key_H)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/list2.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, '{: <20}\tS'.format('Stack'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_S)
        )
        action.setShortcut(Qt.Key_S)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/zoom-in.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, '{: <20}\tI'.format('Zoom in'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_I)
        )
        action.setShortcut(Qt.Key_I)
        plot_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/zoom-out.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, '{: <20}\tO'.format('Zoom out'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_O)
        )
        action.setShortcut(Qt.Key_O)
        plot_actions.addAction(action)

        # values display

        display_format_actions = QActionGroup(self)

        action = QAction('{: <20}\tCtrl+H'.format('Hex'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_H, modifier=Qt.ControlModifier)
        )
        action.setShortcut(QKeySequence('Ctrl+H'))
        display_format_actions.addAction(action)

        action = QAction('{: <20}\tCtrl+B'.format('Bin'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_B, modifier=Qt.ControlModifier)
        )
        action.setShortcut(QKeySequence('Ctrl+B'))
        display_format_actions.addAction(action)

        action = QAction('{: <20}\tCtrl+P'.format('Physical'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_P, modifier=Qt.ControlModifier)
        )
        action.setShortcut(QKeySequence('Ctrl+P'))
        display_format_actions.addAction(action)

        # cursors
        cursors_actions = QActionGroup(self)

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/cursor.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, '{: <20}\tC'.format('Cursor'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_C)
        )
        action.setShortcut(Qt.Key_C)
        cursors_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/left.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, '{: <20}\t←'.format('Move cursor left'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_Left)
        )
        action.setShortcut(Qt.Key_Left)
        cursors_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/right.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, '{: <20}\t→'.format('Move cursor right'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_Right)
        )
        action.setShortcut(Qt.Key_Right)
        cursors_actions.addAction(action)

        icon = QIcon()
        icon.addPixmap(
            QPixmap(":/range.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        action = QAction(icon, '{: <20}\tR'.format('Range'))
        action.triggered.connect(
            partial(self.plot_action, key=Qt.Key_R)
        )
        action.setShortcut(Qt.Key_R)
        cursors_actions.addAction(action)

        menu = QMenu('Plot', self.menubar)
        menu.addActions(plot_actions.actions())
        menu.addSeparator()
        menu.addActions(cursors_actions.actions())
        menu.addSeparator()
        menu.addActions(display_format_actions.actions())
        self.menubar.addMenu(menu)

        # self.menubar.addAction(menu.menuAction())

        self.memory = 'minimum'
        self.match = 'Match start'
        self.toolBox.setCurrentIndex(0)

        self.show()

    def plot_action(self, key, modifier=Qt.NoModifier):
        event = QKeyEvent(QEvent.KeyPress, key, modifier)
        widget = self.files.currentWidget()
        if widget and widget.plot:
            widget.plot.keyPressEvent(event)

    def set_memory_option(self, option):
        self.memory = option

    def set_search_option(self, option):
        self.match = option
        count = self.files.count()
        for i in range(count):
            self.files.widget(i).search_field.set_search_option(option)
            self.files.widget(i).filter_field.set_search_option(option)

    def update_progress(self, current_index, max_index):
        self.progress = current_index, max_index

    def delete_item(self, item):
        index = self.files_list.row(item)
        self.files_list.takeItem(index)

    def function_select(self, val):
        if self.concatenate.isChecked():
            self.cs_btn.setText('Concatenate')
        else:
            self.cs_btn.setText('Stack')

    def cs_clicked(self, event):
        if self.concatenate.isChecked():
            func = MDF.concatenate
            operation = 'Concatenating'
        else:
            func = MDF.stack
            operation = 'Stacking'

        version = self.cs_format.currentText()

        memory = self.memory

        if version < '4.00':
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.cs_split.checkState() == Qt.Checked
        if split:
            split_size = int(self.cs_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.cs_compression.currentIndex()

        count = self.files_list.count()

        files = [
            self.files_list.item(row).text()
            for row in range(count)
        ]

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            '',
            filter,
        )

        if file_name:

            progress = setup_progress(
                parent=self,
                title='{} measurements'.format(operation),
                message='{} files and saving to {} format'.format(
                    operation,
                    version,
                ),
                icon_name='stack',
            )

            target = func
            kwargs = {
                'files': files,
                'outversion': version,
                'memory': memory,
                'callback': self.update_progress,
            }

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

            # save it
            progress.setLabelText(
                'Saving output file "{}"'.format(
                    file_name,
                )
            )

            target = mdf.save
            kwargs = {
                'dst': file_name,
                'compression': compression,
                'overwrite': True,
            }

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=50,
                offset=50,
                progress=progress,
            )

            progress.cancel()

    def open_multiple_files(self, event):
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Select measurement file",
            '',
            "MDF files (*.dat *.mdf *.mf4)",
        )
        if file_names:
            self.files_list.addItems(file_names)
            count = self.files_list.count()

            icon = QIcon()
            icon.addPixmap(
                QPixmap(":/file.png"),
                QIcon.Normal,
                QIcon.Off,
            )

            for row in range(count):
                self.files_list.item(row).setIcon(icon)

    def open(self, event):
        if self.toolBox.currentIndex() == 0:
            self.open_file(event)
        else:
            self.open_multiple_files(event)

    def open_file(self, event):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select measurement file",
            '',
            "MDF/DL3 files (*.dat *.mdf *.mf4 *.dl3)",
        )
        if file_name:
            index = self.files.count()
            try:
                widget = FileWidget(file_name, self.memory)
                widget.search_field.set_search_option(self.match)
                widget.filter_field.set_search_option(self.match)
            except:
                raise
            else:
                self.files.addTab(widget, os.path.basename(file_name))
                self.files.setTabToolTip(index, file_name)
                self.files.setCurrentIndex(index)

    def close_file(self, index):
        widget = self.files.widget(index)
        if widget:
            widget.close()
            widget.setParent(None)

        self.files.removeTab(index)
        if self.files.count():
            self.files.setCurrentIndex(0)

    def closeEvent(self, event):
        count = self.files.count()
        for i in range(count):
            self.files.widget(i).close()
        event.accept()


def main():

    app = QApplication(sys.argv)
    main = MainWindow()
    app.exec_()


if __name__ == '__main__':
    main()

import os
import sys
import traceback

from datetime import datetime
from functools import reduce
from io import StringIO
from threading import Thread
from time import sleep

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QMessageBox,
    QWidget,
    QMainWindow,
    QFileDialog,
    QProgressDialog,
    QApplication,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QLabel,
)

from PyQt5.QtCore import Qt

from PyQt5.QtGui import QIcon, QPixmap

from asammdf import MDF, SUPPORTED_VERSIONS
from asammdf import __version__ as libversion

import asammdfgui.main_window as main_window
import asammdfgui.file_widget as file_widget


__version__ = '0.1.0'


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

    QMessageBox.warning(
        None,
        notice,
        msg,
    )

    print(msg)

sys.excepthook = excepthook


class FileWidget(QWidget, file_widget.Ui_file_widget):
    def __init__(self, file_name, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.file_name = file_name
        self.mdf = MDF(file_name, memory='minimum')

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

            for j, channel in enumerate(group['channels']):
                name = self.mdf.get_channel_name(group=i, index=j)
                channel = QTreeWidgetItem(channel_group)
                channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                channel.setText(0, name)
                channel.setCheckState(0, Qt.Unchecked)

                channel = QTreeWidgetItem(filter_channel_group)
                channel.setFlags(channel.flags() | Qt.ItemIsUserCheckable)
                channel.setText(0, name)
                channel.setCheckState(0, Qt.Unchecked)

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

        times = [
            self.mdf.get_master(i)
            for i, _ in enumerate(self.mdf.groups)
        ]

        times = reduce(np.union1d, times)
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
        self.plot_btn.clicked.connect(self.plot)

    def close(self):
        self.mdf.close()

    def convert(self, event):
        version = self.convert_format.currentText()

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
            self.mdf.convert(version).save(
                file_name,
                compression=compression,
                overwrite=True,
            )

    def resample(self, event):
        version = self.resample_format.currentText()
        raster = self.raster.value()

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
            self.mdf.convert(version).resample(raster).save(
                file_name,
                compression=compression,
                overwrite=True,
            )

    def cut(self, event):
        version = self.cut_format.currentText()
        start = self.cut_start.value()
        stop = self.cut_stop.value()

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
            self.mdf.convert(version).cut(start=start, stop=stop).save(
                file_name,
                compression=compression,
                overwrite=True,
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
            # self.mdf.export(
            #     fmt=export_type,
            #     filename=file_name,
            #     single_time_base=single_time_base,
            #     use_display_names=use_display_names,
            #     time_from_zero=time_from_zero,
            #     empty_channels=empty_channels,
            #     format=mat_format,
            #     raster=raster,
            # )

            progress = QProgressDialog(
                "Copying files...",
                "Abort Copy",
                0,
                100)
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.setAutoClose(True)
            progress.setWindowTitle('Running check')

            thr.start()

            cntr = 0

            while thr.is_alive():
                cntr += 1
                progress.setValue(cntr % 98)
                sleep(0.1)

            progress.cancel()

    def plot(self, event):

        # QtWidgets.QMainWindow.__init__(self)
        # self.widget = QtWidgets.QWidget()
        # self.setCentralWidget(self.widget)
        # self.widget.setLayout(QtWidgets.QVBoxLayout())
        # self.widget.layout().setContentsMargins(0, 0, 0, 0)
        # self.widget.layout().setSpacing(0)
        #
        # self.fig = fig
        # self.canvas = FigureCanvas(self.fig)
        # self.canvas.draw()
        # self.scroll = QtWidgets.QScrollArea(self.widget)
        # self.scroll.setWidget(self.canvas)
        #
        # self.nav = NavigationToolbar(self.canvas, self.widget)
        # self.widget.layout().addWidget(self.nav)
        # self.widget.layout().addWidget(self.scroll)
        #
        # # self.channels_tree.itemChanged.connect(self.select)
        # self.plot_btn.clicked.connect(self.plot)
        #
        #
        #
        # self.figure = Figure()
        #
        # # this is the Canvas Widget that displays the `figure`
        # # it takes the `figure` instance as a parameter to __init__
        # self.canvas = FigureCanvas(self.figure)
        #
        # # this is the Navigation widget
        # # it takes the Canvas widget and a parent
        # self.toolbar = NavigationToolbar(self.canvas, self)
        #
        # self.ax = self.figure.add_subplot(111)
        #
        # # set the layout
        # layout = QVBoxLayout()
        # layout.addWidget(self.toolbar)
        # layout.addWidget(self.canvas)
        # self.channels_layout.addLayout(layout)
        # self.channels_layout.setStretch(0, 0)
        # self.channels_layout.setStretch(1, 1)
        #
        # # refresh canvas
        # self.canvas.draw()

        while self.scroll_layout.count():
            self.scroll_layout.takeAt(0)

        item = self.channels_grid.itemAtPosition(1, 1)
        if item:
            item.widget().setParent(None)

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
                signals.append((group, index))

            index += 1
            iterator += 1

        rows = len(signals)

        # fig, axes = plt.subplots(ncols=1, nrows=rows, figsize=(5, 3 * rows))
        # for ax, (group, index) in zip(axes.flatten(), signals):
        #     sig = self.mdf.get(group=group, index=index)
        #     ax.plot(sig.timestamps, sig.samples, '.-')
        #     ax.set_title(sig.name)
        #     ax.set_ylabel(sig.unit)
        #
        # canvas = FigureCanvas(fig)
        # canvas.draw()
        #
        # self.plot_scroll.setWidget(canvas)
        #
        # nav = NavigationToolbar(canvas, self)
        # self.channels_grid.addWidget(nav, 1, 1)

        figure = Figure(figsize=(6, 30))


        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__

        for i, (group, index) in enumerate(signals):
            sig = self.mdf.get(group=group, index=index)
            ax = figure.add_subplot(rows, 1, i+1)
            ax.plot(sig.timestamps, sig.samples, '.-')
            ax.set_title(sig.name)
            ax.set_ylabel(sig.unit)

        # refresh canvas
        canvas = FigureCanvas(figure)
        canvas.draw()

        self.scroll_layout.addWidget(canvas)

        nav = NavigationToolbar(canvas, self)
        self.channels_grid.addWidget(nav, 1, 1)

    def filter(self, event):
        iterator = QTreeWidgetItemIterator(
            self.filter_tree,
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
                print(item, item.text(0), item.parent(), group, index)

                signals.append((None, group, index))

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
            self.mdf.filter(signals).convert(version).save(
                file_name,
                compression=compression,
                overwrite=True,
            )


class MainWindow(QMainWindow, main_window.Ui_PyMDFMainWindow):
    def __init__(self, parent=None):

        super().__init__(parent)

        self.last_folder = ''
        self.setupUi(self)

        self.open_file_btn.clicked.connect(self.open_file)
        self.open_multiple_files_btn.clicked.connect(self.open_multiple_files)
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

        self.files_list.itemDoubleClicked.connect(self.delete_item)

        self.statusbar.addPermanentWidget(
            QLabel('asammdfgui {} with asammdf {}'.format(__version__, libversion)))

        self.show()

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
        else:
            func = MDF.stack

        version = self.cs_format.currentText()

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

            mdf = func(
                files,
                outversion=version,
            )
            mdf.configure(write_fragment_size=split_size)

            mdf.save(
                file_name,
                compression=compression,
                overwrite=True,
            )

    def open_multiple_files(self, event):
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Select measurement file",
            self.last_folder,
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

    def open_file(self, event):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select measurement file",
            self.last_folder,
            "MDF files (*.dat *.mdf *.mf4)",
        )
        if file_name:
            index = self.files.count()
            widget = FileWidget(file_name)
            self.files.addTab(widget, os.path.basename(file_name))
            self.files.setTabToolTip(index, file_name)

    def close_file(self, index):
        widget = self.files.widget(index)
        widget.close()
        widget.setParent(None)

        self.files.removeTab(index)
        if self.files.count():
            self.files.setCurrentIndex(0)

    def closeEvent(self, event):
        count = self.files.count()
        for i in range(count):
            self.files.tabCloseRequested.emit(0)
        event.accept()


def main():
    app = QApplication(sys.argv)
    main = MainWindow()
    app.exec_()


if __name__ == '__main__':
    main()


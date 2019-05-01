# -*- coding: utf-8 -*-
from threading import Thread
from time import sleep
from pathlib import Path

from natsort import natsorted

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from ..ui import resource_rc as resource_rc
from ..ui.batch_widget import Ui_batch_widget

from ...mdf import MDF, SUPPORTED_VERSIONS
from ...signal import Signal
from ...blocks.utils import MdfException, extract_cncomment_xml
from ..utils import TERMINATED, run_thread_with_progress, setup_progress
from .search import SearchWidget
from .tree import TreeWidget
from .tree_item import TreeItem


class BatchWidget(Ui_batch_widget, QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.raster_type_channel.toggled.connect(self.set_raster_type)

        for widget in (
              self.concatenate_format,
              self.convert_format,
              self.cut_format,
              self.resample_format,
              self.stack_format,
              self.extract_can_format,
        ):

            widget.insertItems(0, SUPPORTED_VERSIONS)

        for widget in (
              self.concatenate_split_size,
              self.convert_split_size,
              self.cut_split_size,
              self.resample_split_size,
              self.stack_split_size,
        ):

            widget.setValue(4)

        for widget in (
              self.concatenate_compression,
              self.convert_compression,
              self.cut_compression,
              self.resample_compression,
              self.stack_compression,
              self.extract_can_compression,
        ):

            widget.insertItems(
                0, ("no compression", "deflate", "transposed deflate")
            )

        self.add_files_btn.clicked.connect(self.add_files)
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.concatenate_btn.clicked.connect(self.concatenate)
        self.convert_btn.clicked.connect(self.convert)
        self.cut_btn.clicked.connect(self.cut)
        self.resample_btn.clicked.connect(self.resample)
        self.scramble_btn.clicked.connect(self.scramble)
        self.stack_btn.clicked.connect(self.stack)
        self.extract_can_btn.clicked.connect(self.extract_can)

        self.load_can_database_btn.clicked.connect(self.load_can_database)

        self.empty_channels.insertItems(0, ("zeros", "skip"))
        self.mat_format.insertItems(0, ("4", "5", "7.3"))
        self.oned_as.insertItems(0, ("row", "column"))
        self.export_type.insertItems(0, ("csv", "excel", "hdf5", "mat", "parquet"))
        self.export_btn.clicked.connect(self.export)
        self.export_type.currentTextChanged.connect(self.export_changed)
        self.export_type.setCurrentIndex(-1)

        self.aspects.setCurrentIndex(0)

    def add_files(self, event):
        pass

    def add_folder(self, event):
        pass

    def set_raster_type(self, event):
        if self.raster_type_channel.isChecked():
            self.raster_channel.setEnabled(True)
            self.raster.setEnabled(False)
            self.raster.setValue(0)
        else:
            self.raster_channel.setEnabled(False)
            self.raster_channel.setCurrentIndex(0)
            self.raster.setEnabled(True)

    def export_changed(self, name):
        if name == 'parquet':
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(['GZIP', 'SNAPPY'])
            self.export_compression.setCurrentIndex(-1)
        elif name == 'hdf5':
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(["gzip", "lzf", "szip"])
            self.export_compression.setCurrentIndex(-1)
        elif name == 'mat':
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(["enabled", "disabled"])
            self.export_compression.setCurrentIndex(-1)
        else:
            self.export_compression.clear()
            self.export_compression.setEnabled(False)

    def compute_cut_hints(self):
        # TODO : use master channel physical min and max values
        times = []
        groups_nr = len(self.mdf.groups)
        for i in range(groups_nr):
            master = self.mdf.get_master(i)
            if len(master):
                times.append(master[0])
                times.append(master[-1])
            QtWidgets.QApplication.processEvents()

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

    def close(self):
        mdf_name = self.mdf.name
        self.mdf.close()
        if self.file_name.suffix.lower() == ".dl3":
            mdf_name.unlink()

    def convert(self, event):
        version = self.convert_format.currentText()

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.convert_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.convert_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.convert_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:

            progress = setup_progress(
                parent=self,
                title="Converting measurement",
                message=f'Converting "{self.file_name}" from {self.mdf.version} to {version}',
                icon_name="convert",
            )

            # convert self.mdf
            target = self.mdf.convert
            kwargs = {"version": version}

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
            progress.setLabelText(f'Saving converted file "{file_name}"')

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

        if self.raster_type_channel.isChecked():
            raster = self.raster_channel.currentText()
        else:
            raster = self.raster.value()

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.resample_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.resample_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.resample_compression.currentIndex()
        time_from_zero = self.resample_time_from_zero.checkState() == QtCore.Qt.Checked

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title="Resampling measurement",
                message=f'Resampling "{self.file_name}" to {raster}s raster ',
                icon_name="resample",
            )

            # resample self.mdf
            target = self.mdf.resample
            kwargs = {
                "raster": raster,
                "version": version,
                "time_from_zero": time_from_zero,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=66,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText(f'Saving resampled file "{file_name}"')

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
        time_from_zero = self.cut_time_from_zero.checkState() == QtCore.Qt.Checked

        if self.whence.checkState() == QtCore.Qt.Checked:
            whence = 1
        else:
            whence = 0

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.cut_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.cut_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.cut_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title="Cutting measurement",
                message='Cutting "{self.file_name}" from {start}s to {stop}s',
                icon_name="cut",
            )

            # cut self.mdf
            target = self.mdf.cut
            kwargs = {
                "start": start,
                "stop": stop,
                "whence": whence,
                "version": version,
                "time_from_zero": time_from_zero,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=66,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText(f'Saving cut file "{file_name}"')

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

        single_time_base = self.single_time_base.checkState() == QtCore.Qt.Checked
        time_from_zero = self.time_from_zero.checkState() == QtCore.Qt.Checked
        use_display_names = self.use_display_names.checkState() == QtCore.Qt.Checked
        empty_channels = self.empty_channels.currentText()
        mat_format = self.mat_format.currentText()
        raster = self.export_raster.value()
        oned_as = self.oned_as.currentText()
        reduce_memory_usage = self.reduce_memory_usage.checkState() == QtCore.Qt.Checked
        compression = self.export_compression.currentText()

        filters = {
            "csv": "CSV files (*.csv)",
            "excel": "Excel files (*.xlsx)",
            "hdf5": "HDF5 files (*.hdf)",
            "mat": "Matlab MAT files (*.mat)",
            "parquet": "Apache Parquet files (*.parquet)",
        }

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select export file", "", filters[export_type]
        )

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
                    "reduce_memory_usage": reduce_memory_usage,
                    "compression": compression,
                },
            )

            progress = QtWidgets.QProgressDialog(
                f"Exporting to {export_type} ...", "Abort export", 0, 100
            )
            progress.setWindowModality(QtCore.Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.setAutoClose(True)
            progress.setWindowTitle("Running export")
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/export.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            progress.setWindowIcon(icon)

            thr.start()

            cntr = 0

            while thr.is_alive():
                cntr += 1
                progress.setValue(cntr % 98)
                sleep(0.1)

            progress.cancel()

    def filter(self, event):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

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

            if item.checkState(0) == QtCore.Qt.Checked:
                channels.append((None, group, index))

            index += 1
            iterator += 1

        version = self.filter_format.itemText(self.filter_format.currentIndex())

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.filter_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.filter_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        self.mdf.configure(write_fragment_size=split_size)

        compression = self.filter_compression.currentIndex()

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:
            progress = setup_progress(
                parent=self,
                title="Filtering measurement",
                message=f'Filtering selected channels from "{self.file_name}"',
                icon_name="filter",
            )

            # filtering self.mdf
            target = self.mdf.filter
            kwargs = {"channels": channels, "version": version}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=66,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            # then save it
            progress.setLabelText(f'Saving filtered file "{file_name}"')

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

    def scramble(self, event):

        progress = setup_progress(
            parent=self,
            title="Scrambling measurement",
            message=f'Scrambling "{self.file_name}"',
            icon_name="scramble",
        )

        # scrambling self.mdf
        target = MDF.scramble
        kwargs = {"name": self.file_name, "callback": self.update_progress}

        mdf = run_thread_with_progress(
            self,
            target=target,
            kwargs=kwargs,
            factor=100,
            offset=0,
            progress=progress,
        )

        if mdf is TERMINATED:
            progress.cancel()
            return

        self.open_new_file.emit(str(Path(self.file_name).with_suffix(".scrambled.mf4")))

    def extract_can(self, event):
        version = self.extract_can_format.currentText()
        count = self.can_database_list.count()

        dbc_files = []
        for i in range(count):
            item = self.can_database_list.item(i)
            dbc_files.append(item.text())

        compression = self.extract_can_compression.currentIndex()

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:

            progress = setup_progress(
                parent=self,
                title="Extract CAN logging",
                message=f'Extracting CAN signals from "{self.file_name}"',
                icon_name="down",
            )

            # convert self.mdf
            target = self.mdf.extract_can_logging
            kwargs = {
                "dbc_files": dbc_files,
                "version": version,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=70,
                offset=0,
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            # then save it
            progress.setLabelText(f'Saving file to "{file_name}"')

            target = mdf.save
            kwargs = {
                "dst": file_name,
                "compression": compression,
                "overwrite": True,
            }

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=30,
                offset=70,
                progress=progress,
            )

            self.open_new_file.emit(str(file_name))

    def load_can_database(self, event):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select CAN database file",
            "",
            "ARXML or DBC (*.dbc *.axml)",
            "ARXML or DBC (*.dbc *.axml)",
        )

        for file_name in file_names:
            self.can_database_list.addItems(file_names)

    def concatenate(self, event):
        func = MDF.concatenate
        operation = "Concatenating"

        version = self.concatenate_format.currentText()

        sync = self.concatenate_sync.checkState() == QtCore.Qt.Checked
        add_samples_origin = self.concatenate_add_samples_origin.checkState() == QtCore.Qt.Checked

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.concatenate_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.concatenate_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.concatenate_compression.currentIndex()

        count = self.files_list.count()

        files = [self.files_list.item(row).text() for row in range(count)]

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:

            progress = setup_progress(
                parent=self,
                title=f"{operation} measurements",
                message=f"{operation} files and saving to {version} format",
                icon_name="stack",
            )

            target = func
            kwargs = {
                "files": files,
                "outversion": version,
                "callback": self.update_progress,
                "sync": sync,
                "add_samples_origin": add_samples_origin,
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
            progress.setLabelText(f'Saving output file "{file_name}"')

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

            progress.cancel()

    def stack(self, event):
        func = MDF.stack
        operation = "Stacking"

        version = self.stack_format.currentText()

        sync = self.stack_sync.checkState() == QtCore.Qt.Checked
        add_samples_origin = self.stack_add_samples_origin.checkState() == QtCore.Qt.Checked

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
        else:
            filter = "MDF version 4 files (*.mf4)"

        split = self.stack_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.stack_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.stack_compression.currentIndex()

        count = self.files_list.count()

        files = [self.files_list.item(row).text() for row in range(count)]

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output measurement file", "", filter
        )

        if file_name:

            progress = setup_progress(
                parent=self,
                title=f"{operation} measurements",
                message=f"{operation} files and saving to {version} format",
                icon_name="stack",
            )

            target = func
            kwargs = {
                "files": files,
                "outversion": version,
                "callback": self.update_progress,
                "sync": sync,
                "add_samples_origin": add_samples_origin,
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
            progress.setLabelText(f'Saving output file "{file_name}"')

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

            progress.cancel()

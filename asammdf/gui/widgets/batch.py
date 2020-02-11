# -*- coding: utf-8 -*-
from threading import Thread
from time import sleep
from pathlib import Path

from natsort import natsorted
import psutil

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
from .list import MinimalListWidget


class BatchWidget(Ui_batch_widget, QtWidgets.QWidget):
    def __init__(self, ignore_value2text_conversions=False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.ignore_value2text_conversions = ignore_value2text_conversions

        self.progress = None
        self.files_list = MinimalListWidget()
        self.splitter.addWidget(self.files_list)

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

            widget.insertItems(0, ("no compression", "deflate", "transposed deflate"))

        self.concatenate_btn.clicked.connect(self.concatenate)
        self.convert_btn.clicked.connect(self.convert)
        self.cut_btn.clicked.connect(self.cut)
        self.resample_btn.clicked.connect(self.resample)
        self.scramble_btn.clicked.connect(self.scramble)
        self.stack_btn.clicked.connect(self.stack)
        self.extract_can_btn.clicked.connect(self.extract_can)
        self.extract_can_csv_btn.clicked.connect(self.extract_can_csv)

        self.load_can_database_btn.clicked.connect(self.load_can_database)

        self.empty_channels.insertItems(0, ("skip", "zeros"))
        self.empty_channels_can.insertItems(0, ("skip", "zeros"))
        self.mat_format.insertItems(0, ("4", "5", "7.3"))
        self.oned_as.insertItems(0, ("row", "column"))
        self.export_type.insertItems(0, ("csv", "hdf5", "mat", "parquet"))
        self.export_btn.clicked.connect(self.export)
        self.export_type.currentTextChanged.connect(self.export_changed)
        self.export_type.setCurrentIndex(-1)

        self.aspects.setCurrentIndex(0)
        self.setAcceptDrops(True)

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
        if name == "parquet":
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(["GZIP", "SNAPPY"])
            self.export_compression.setCurrentIndex(-1)
        elif name == "hdf5":
            self.export_compression.setEnabled(True)
            self.export_compression.clear()
            self.export_compression.addItems(["gzip", "lzf", "szip"])
            self.export_compression.setCurrentIndex(-1)
        elif name == "mat":
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

        split = self.convert_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.convert_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.convert_compression.currentIndex()

        count = self.files_list.count()

        delta = 100 / count

        progress = setup_progress(
            parent=self,
            title="Converting measurements",
            message=f'Converting "{count}" files to {version}',
            icon_name="convert",
        )

        files = self._prepare_files(progress)
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        for i, (file, source_file) in enumerate(zip(files, source_files)):

            progress.setLabelText(f"Converting file {i+1} of {count} to {version}")

            if not isinstance(file, MDF):

                # open file
                target = MDF
                kwargs = {
                    "name": file,
                    "callback": self.update_progress,
                }

                mdf = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=0,
                    offset=int(i * delta),
                    progress=progress,
                )
            else:
                mdf = file

            target = mdf.convert
            kwargs = {"version": version}

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=int(delta / 2),
                offset=int(i * delta),
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            file_name = source_file.with_suffix(
                ".convert.mdf" if version < "4.00" else ".convert.mf4"
            )

            # then save it
            progress.setLabelText(f'Saving converted file {i+1} to "{file_name}"')

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=int(delta / 2),
                offset=int((i + 1 / 2) * delta),
                progress=progress,
            )

        self.progress = None
        progress.cancel()

    def resample(self, event):
        version = self.resample_format.currentText()

        if self.raster_type_channel.isChecked():
            raster = self.raster_channel.currentText()
        else:
            raster = self.raster.value()

        split = self.resample_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.resample_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.resample_compression.currentIndex()
        time_from_zero = self.resample_time_from_zero.checkState() == QtCore.Qt.Checked

        count = self.files_list.count()

        delta = 100 / count

        progress = setup_progress(
            parent=self,
            title="Resampling measurements",
            message=f'Resampling "{count}" files',
            icon_name="convert",
        )

        files = self._prepare_files(progress)
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        for i, (file, source_file) in enumerate(zip(files, source_files)):

            if isinstance(raster, str):
                progress.setLabelText(
                    f'Resampling file {i+1} of {count} using "{raster}" as raster'
                )
            else:
                progress.setLabelText(
                    f"Resampling file {i+1} of {count} to {raster:.3f}s"
                )

            if not isinstance(file, MDF):

                # open file
                target = MDF
                kwargs = {
                    "name": file,
                    "callback": self.update_progress,
                }

                mdf = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=0,
                    offset=int(i * delta),
                    progress=progress,
                )
            else:
                mdf = file

            target = mdf.resample
            kwargs = {
                "raster": raster,
                "version": version,
                "time_from_zero": time_from_zero,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=int(delta / 2),
                offset=int(i * delta),
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            file_name = source_file.with_suffix(
                ".resample.mdf" if version < "4.00" else ".resample.mf4"
            )

            # then save it
            progress.setLabelText(f'Saving resampled file {i+1} to "{file_name}"')

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=int(delta / 2),
                offset=int((i + 1 / 2) * delta),
                progress=progress,
            )

        self.progress = None
        progress.cancel()

    def cut(self, event):
        version = self.cut_format.currentText()
        start = self.cut_start.value()
        stop = self.cut_stop.value()
        time_from_zero = self.cut_time_from_zero.checkState() == QtCore.Qt.Checked

        if self.whence.checkState() == QtCore.Qt.Checked:
            whence = 1
        else:
            whence = 0

        split = self.cut_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.cut_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.cut_compression.currentIndex()

        count = self.files_list.count()

        delta = 100 / count

        progress = setup_progress(
            parent=self,
            title="Cutting measurements",
            message=f'Cutting "{count}" files from {start}s to {stop}s',
            icon_name="cut",
        )

        files = self._prepare_files(progress)
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        for i, (file, source_file) in enumerate(zip(files, source_files)):

            progress.setLabelText(
                f"Cutting file {i+1} of {count} from {start}s to {stop}s"
            )

            if not isinstance(file, MDF):

                # open file
                target = MDF
                kwargs = {
                    "name": file,
                    "callback": self.update_progress,
                }

                mdf = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=0,
                    offset=int(i * delta),
                    progress=progress,
                )
            else:
                mdf = file

            # cut self.mdf
            target = mdf.cut
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
                factor=int(delta / 2),
                offset=int(i * delta),
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

            mdf.configure(write_fragment_size=split_size)

            file_name = source_file.with_suffix(
                ".cut.mdf" if version < "4.00" else ".cut.mf4"
            )

            # then save it
            progress.setLabelText(f'Saving cut file {i+1} to "{file_name}"')

            target = mdf.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=int(delta / 2),
                offset=int((i + 1 / 2) * delta),
                progress=progress,
            )

        self.progress = None
        progress.cancel()

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
        time_as_date = self.time_as_date.checkState() == QtCore.Qt.Checked

        count = self.files_list.count()

        delta = 100 / count

        progress = setup_progress(
            parent=self,
            title="Exporting measurements",
            message=f'Exporting "{count}" files to {export_type}',
            icon_name="export",
        )

        files = self._prepare_files(progress)
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        if export_type == "csv":
            suffix = ".csv"
        elif export_type == "hdf5":
            suffix = ".hdf"
        elif export_type == "mat":
            suffix = ".mat"
        elif export_type == "parquet":
            suffix = ".parquet"

        for i, (file, source_file) in enumerate(zip(files, source_files)):

            progress.setLabelText(f"Exporting file {i+1} of {count} to {export_type}")

            if not isinstance(file, MDF):

                # open file
                target = MDF
                kwargs = {
                    "name": file,
                    "callback": self.update_progress,
                }

                mdf = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=0,
                    offset=int(i * delta),
                    progress=progress,
                )
            else:
                mdf = file

            file_name = source_file.with_suffix(suffix)

            target = mdf.export
            kwargs = {
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
                "time_as_date": time_as_date,
                "ignore_value2text_conversions": self.ignore_value2text_conversions,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=int(delta / 2),
                offset=int(i * delta),
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

        self.progress = None
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

        count = self.files_list.count()

        delta = 100 / count

        progress = setup_progress(
            parent=self,
            title="Scrambling measurements",
            message=f'Scrambling "{count}" files',
            icon_name="scramble",
        )

        files = self._prepare_files(progress)
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        for i, (file, source_file) in enumerate(zip(files, source_files)):

            progress.setLabelText(f"Scrambling file {i+1} of {count}")

            target = MDF.scramble
            kwargs = {
                "name": file,
                "callback": self.update_progress,
            }

            mdf = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=0,
                offset=int(i * delta),
                progress=progress,
            )

            if mdf is TERMINATED:
                progress.cancel()
                return

        self.progress = None
        progress.cancel()

    def extract_can(self, event):
        self.output_info_can.setPlainText("")
        version = self.extract_can_format.currentText()
        count = self.can_database_list.count()

        dbc_files = []
        for i in range(count):
            item = self.can_database_list.item(i)
            dbc_files.append(item.text())

        compression = self.extract_can_compression.currentIndex()
        ignore_invalid_signals = (
            self.ignore_invalid_signals_mdf.checkState() == QtCore.Qt.Checked
        )

        count = self.files_list.count()

        delta = 100 / count

        progress = setup_progress(
            parent=self,
            title="Extract CAN logging from measurements",
            message=f'Extracting CAN logging from "{count}" files',
            icon_name="down",
        )

        files = self._prepare_files(progress)
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        for i, (file, source_file) in enumerate(zip(files, source_files)):

            progress.setLabelText(f"Extracting CAN logging from file {i+1} of {count}")

            if not isinstance(file, MDF):

                # open file
                target = MDF
                kwargs = {
                    "name": file,
                    "callback": self.update_progress,
                }

                mdf = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=0,
                    offset=int(i * delta),
                    progress=progress,
                )
            else:
                mdf = file

            target = mdf.extract_can_logging
            kwargs = {
                "dbc_files": dbc_files,
                "version": version,
                "ignore_invalid_signals": ignore_invalid_signals,
            }

            mdf_ = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=int(delta / 2),
                offset=int(i * delta),
                progress=progress,
            )

            if mdf_ is TERMINATED:
                progress.cancel()
                return

            call_info = dict(mdf.last_call_info)

            found_id_count = sum(len(e) for e in call_info["found_ids"].values())

            message = [
                "",
                f'Summary of "{mdf.name}":',
                f'- {found_id_count} of {len(call_info["total_unique_ids"])} IDs in the MDF4 file were matched in the DBC and converted',
            ]
            if call_info["unknown_id_count"]:
                message.append(
                    f'- {call_info["unknown_id_count"]} unknown IDs in the MDF4 file'
                )
            else:
                message.append(f"- no unknown IDs inf the MDF4 file")

            message += [
                "",
                "Detailed information:",
                "",
                "The following CAN IDs were in the MDF log file and matched in the DBC:",
            ]
            for dbc_name, found_ids in call_info["found_ids"].items():
                for msg_id, msg_name in sorted(found_ids):
                    message.append(f"- 0x{msg_id:X} --> {msg_name} in <{dbc_name}>")

            message += [
                "",
                "The following CAN IDs were in the MDF log file, but not matched in the DBC:",
            ]
            for msg_id in sorted(call_info["unknown_ids"]):
                message.append(f"- 0x{msg_id:X}")

            self.output_info_can.append("\n".join(message))

            file_name = source_file.with_suffix(
                ".can_logging.mdf" if version < "4.00" else ".can_logging.mf4"
            )

            # then save it
            progress.setLabelText(
                f'Saving extarcted CAN logging file {i+1} to "{file_name}"'
            )

            target = mdf_.save
            kwargs = {"dst": file_name, "compression": compression, "overwrite": True}

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=int(delta / 2),
                offset=int((i + 1 / 2) * delta),
                progress=progress,
            )

        self.progress = None
        progress.cancel()

    def extract_can_csv(self, event):
        self.output_info_can.setPlainText("")
        version = self.extract_can_format.currentText()
        count = self.can_database_list.count()

        dbc_files = []
        for i in range(count):
            item = self.can_database_list.item(i)
            dbc_files.append(item.text())

        ignore_invalid_signals = (
            self.ignore_invalid_signals_csv.checkState() == QtCore.Qt.Checked
        )
        single_time_base = self.single_time_base_can.checkState() == QtCore.Qt.Checked
        time_from_zero = self.time_from_zero_can.checkState() == QtCore.Qt.Checked
        empty_channels = self.empty_channels_can.currentText()
        raster = self.export_raster.value()
        time_as_date = self.can_time_as_date.checkState() == QtCore.Qt.Checked

        count = self.files_list.count()

        delta = 100 / count

        progress = setup_progress(
            parent=self,
            title="Extract CAN logging from measurements to CSV",
            message=f'Extracting CAN logging from "{count}" files',
            icon_name="csv",
        )

        files = self._prepare_files(progress)
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        for i, (file, source_file) in enumerate(zip(files, source_files)):

            progress.setLabelText(f"Extracting CAN logging from file {i+1} of {count}")

            if not isinstance(file, MDF):

                # open file
                target = MDF
                kwargs = {
                    "name": file,
                    "callback": self.update_progress,
                }

                mdf = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=0,
                    offset=int(i * delta),
                    progress=progress,
                )
            else:
                mdf = file

            target = mdf.extract_can_logging
            kwargs = {
                "dbc_files": dbc_files,
                "version": version,
                "ignore_invalid_signals": ignore_invalid_signals,
            }

            mdf_ = run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=int(delta / 2),
                offset=int(i * delta),
                progress=progress,
            )

            if mdf_ is TERMINATED:
                progress.cancel()
                return

            call_info = dict(mdf.last_call_info)

            found_id_count = sum(len(e) for e in call_info["found_ids"].values())

            message = [
                "",
                f'Summary of "{mdf.name}":',
                f'- {found_id_count} of {len(call_info["total_unique_ids"])} IDs in the MDF4 file were matched in the DBC and converted',
            ]
            if call_info["unknown_id_count"]:
                message.append(
                    f'- {call_info["unknown_id_count"]} unknown IDs in the MDF4 file'
                )
            else:
                message.append(f"- no unknown IDs inf the MDF4 file")

            message += [
                "",
                "Detailed information:",
                "",
                "The following CAN IDs were in the MDF log file and matched in the DBC:",
            ]
            for dbc_name, found_ids in call_info["found_ids"].items():
                for msg_id, msg_name in sorted(found_ids):
                    message.append(f"- 0x{msg_id:X} --> {msg_name} in <{dbc_name}>")

            message += [
                "",
                "The following CAN IDs were in the MDF log file, but not matched in the DBC:",
            ]
            for msg_id in sorted(call_info["unknown_ids"]):
                message.append(f"- 0x{msg_id:X}")

            self.output_info_can.append("\n".join(message))

            file_name = source_file.with_suffix(".can_logging.csv")

            # then save it
            progress.setLabelText(
                f'Saving extarcted CAN logging file {i+1} to "{file_name}"'
            )

            target = mdf_.export
            kwargs = {
                "fmt": "csv",
                "filename": file_name,
                "single_time_base": single_time_base,
                "time_from_zero": time_from_zero,
                "empty_channels": empty_channels,
                "raster": raster,
                "time_as_date": time_as_date,
                "ignore_value2text_conversions": self.ignore_value2text_conversions,
            }

            run_thread_with_progress(
                self,
                target=target,
                kwargs=kwargs,
                factor=int(delta / 2),
                offset=int((i + 1 / 2) * delta),
                progress=progress,
            )

        self.progress = None
        progress.cancel()

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
        add_samples_origin = (
            self.concatenate_add_samples_origin.checkState() == QtCore.Qt.Checked
        )

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = ".mdf"
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = ".mf4"

        split = self.concatenate_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.concatenate_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.concatenate_compression.currentIndex()

        output_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            "",
            f"{filter};;All files (*.*)",
            filter,
        )

        if output_file_name:
            output_file_name = Path(output_file_name).with_suffix(suffix)

            progress = setup_progress(
                parent=self,
                title=f"{operation} measurements",
                message=f"{operation} files and saving to {version} format",
                icon_name="stack",
            )

            files = self._prepare_files(progress)

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
            progress.setLabelText(f'Saving output file "{output_file_name}"')

            target = mdf.save
            kwargs = {
                "dst": output_file_name,
                "compression": compression,
                "overwrite": True,
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

    def stack(self, event):
        func = MDF.stack
        operation = "Stacking"

        version = self.stack_format.currentText()

        sync = self.stack_sync.checkState() == QtCore.Qt.Checked
        add_samples_origin = (
            self.stack_add_samples_origin.checkState() == QtCore.Qt.Checked
        )

        if version < "4.00":
            filter = "MDF version 3 files (*.dat *.mdf)"
            suffix = ".mdf"
        else:
            filter = "MDF version 4 files (*.mf4)"
            suffix = ".mf4"

        split = self.stack_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.stack_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.stack_compression.currentIndex()

        output_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output measurement file",
            "",
            f"{filter};;All files (*.*)",
            filter,
        )

        if output_file_name:
            output_file_name = Path(output_file_name).with_suffix(suffix)

            progress = setup_progress(
                parent=self,
                title=f"{operation} measurements",
                message=f"{operation} files and saving to {version} format",
                icon_name="stack",
            )

            files = self._prepare_files(progress)

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
            progress.setLabelText(f'Saving output file "{output_file_name}"')

            target = mdf.save
            kwargs = {
                "dst": output_file_name,
                "compression": compression,
                "overwrite": True,
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

    def _prepare_files(self, progress):
        count = self.files_list.count()

        files = [Path(self.files_list.item(row).text()) for row in range(count)]

        for i, file_name in enumerate(files):
            if file_name.suffix.lower() == ".erg":
                progress.setLabelText(
                    f"Converting file {i+1} of {count} from erg to mdf"
                )
                try:
                    from mfile import ERG

                    files[i] = ERG(file_name).export_mdf()
                except Exception as err:
                    print(err)
                    return
            else:

                if file_name.suffix.lower() == ".dl3":
                    progress.setLabelText(
                        f"Converting file {i+1} of {count} from dl3 to mdf"
                    )
                    datalyser_active = any(
                        proc.name() == "Datalyser3.exe"
                        for proc in psutil.process_iter()
                    )
                    try:
                        import win32com.client

                        index = 0
                        while True:
                            mdf_name = file_name.with_suffix(f".{index}.mdf")
                            if mdf_name.exists():
                                index += 1
                            else:
                                break

                        datalyser = win32com.client.Dispatch(
                            "Datalyser3.Datalyser3_COM"
                        )
                        if not datalyser_active:
                            try:
                                datalyser.DCOM_set_datalyser_visibility(False)
                            except:
                                pass
                        datalyser.DCOM_convert_file_mdf_dl3(file_name, str(mdf_name), 0)
                        if not datalyser_active:
                            datalyser.DCOM_TerminateDAS()
                        files[i] = mdf_name
                    except Exception as err:
                        print(err)
                        return

        return files

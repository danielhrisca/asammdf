# -*- coding: utf-8 -*-
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from tempfile import gettempdir
from traceback import format_exc

from natsort import natsorted
import psutil
from PySide6 import QtCore, QtGui, QtWidgets

from ...blocks.utils import (
    extract_xml_comment,
    load_channel_names_from_file,
    load_lab,
    TERMINATED,
)
from ...blocks.v2_v3_blocks import HeaderBlock as HeaderBlockV3
from ...blocks.v4_blocks import HeaderBlock as HeaderBlockV4
from ...mdf import MDF, SUPPORTED_VERSIONS
from ..dialogs.advanced_search import AdvancedSearch
from ..ui import resource_rc
from ..ui.batch_widget import Ui_batch_widget
from ..utils import HelperChannel, setup_progress, TERMINATED
from .database_item import DatabaseItem
from .list import MinimalListWidget
from .tree import add_children
from .tree_item import TreeItem


class BatchWidget(Ui_batch_widget, QtWidgets.QWidget):
    def __init__(
        self,
        ignore_value2text_conversions=False,
        integer_interpolation=2,
        float_interpolation=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self._settings = QtCore.QSettings()

        self.ignore_value2text_conversions = ignore_value2text_conversions
        self.integer_interpolation = integer_interpolation
        self.float_interpolation = float_interpolation

        self.files_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.progress = None
        self.show()

        sizes = sum(self.splitter.sizes())
        if sizes >= 700:
            self.splitter.setSizes([700, sizes - 700])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        self.raster_type_channel.toggled.connect(self.set_raster_type)

        for widget in (
            self.concatenate_format,
            self.stack_format,
            self.extract_bus_format,
            self.mdf_version,
        ):
            widget.insertItems(0, SUPPORTED_VERSIONS)
            widget.setCurrentText("4.10")

        for widget in (
            self.concatenate_split_size,
            self.stack_split_size,
            self.mdf_split_size,
        ):
            widget.setValue(4)

        for widget in (
            self.concatenate_compression,
            self.stack_compression,
            self.extract_bus_compression,
            self.mdf_compression,
        ):
            widget.insertItems(0, ("no compression", "deflate", "transposed deflate"))
            widget.setCurrentText("transposed deflate")

        self.concatenate_btn.clicked.connect(self.concatenate)
        self.scramble_btn.clicked.connect(self.scramble)
        self.stack_btn.clicked.connect(self.stack)
        self.extract_bus_btn.clicked.connect(self.extract_bus_logging)
        self.extract_bus_csv_btn.clicked.connect(self.extract_bus_csv_logging)
        self.advanced_serch_filter_btn.clicked.connect(self.search)
        self.raster_search_btn.clicked.connect(self.raster_search)
        self.apply_btn.clicked.connect(self.apply_processing)
        self.modify_output_folder_btn.clicked.connect(self.change_modify_output_folder)
        self.output_format.currentTextChanged.connect(self.output_format_changed)
        self.sort_alphabetically_btn.clicked.connect(self.sort_alphabetically)
        self.sort_by_start_time_btn.clicked.connect(self.sort_by_start_time)

        self.filter_view.setCurrentIndex(-1)
        self.filter_view.currentIndexChanged.connect(self.update_channel_tree)
        self.filter_view.currentTextChanged.connect(self.update_channel_tree)
        self.filter_view.setCurrentText(
            self._settings.value("filter_view", "Internal file structure")
        )

        self.filter_tree.itemChanged.connect(self.filter_changed)

        self.load_can_database_btn.clicked.connect(self.load_can_database)
        self.load_lin_database_btn.clicked.connect(self.load_lin_database)

        self.load_filter_list_btn.clicked.connect(self.load_filter_list)
        self.save_filter_list_btn.clicked.connect(self.save_filter_list)

        self.empty_channels_bus.insertItems(0, ("skip", "zeros"))
        self.empty_channels.insertItems(0, ("skip", "zeros"))
        self.empty_channels_bus.insertItems(0, ("skip", "zeros"))
        self.empty_channels_mat.insertItems(0, ("skip", "zeros"))
        self.empty_channels_csv.insertItems(0, ("skip", "zeros"))
        try:
            import scipy

            self.mat_format.insertItems(0, ("4", "5", "7.3"))
        except:
            self.mat_format.insertItems(0, ("7.3",))
        self.oned_as.insertItems(0, ("row", "column"))

        self.aspects.setCurrentIndex(0)
        self.setAcceptDrops(True)

        self.files_list.model().rowsInserted.connect(self.update_channel_tree)
        self.files_list.model().rowsRemoved.connect(self.update_channel_tree)

        self.filter_tree.itemChanged.connect(self.filter_changed)
        self._selected_filter = set()
        self._filter_timer = QtCore.QTimer()
        self._filter_timer.setSingleShot(True)
        self._filter_timer.timeout.connect(self.update_selected_filter_channels)

        databases = {}

        can_databases = self._settings.value("can_databases", [])
        buses = can_databases[::2]
        dbs = can_databases[1::2]

        databases["CAN"] = [(bus, database) for bus, database in zip(buses, dbs)]

        lin_databases = self._settings.value("lin_databases", [])
        buses = lin_databases[::2]
        dbs = lin_databases[1::2]

        databases["LIN"] = [(bus, database) for bus, database in zip(buses, dbs)]

        for bus, database in databases["CAN"]:
            item = QtWidgets.QListWidgetItem()
            widget = DatabaseItem(database, bus_type="CAN")
            widget.bus.setCurrentText(bus)

            self.can_database_list.addItem(item)
            self.can_database_list.setItemWidget(item, widget)
            item.setSizeHint(widget.sizeHint())

        for bus, database in databases["LIN"]:
            item = QtWidgets.QListWidgetItem()
            widget = DatabaseItem(database, bus_type="LIN")
            widget.bus.setCurrentText(bus)

            self.lin_database_list.addItem(item)
            self.lin_database_list.setItemWidget(item, widget)
            item.setSizeHint(widget.sizeHint())

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

    def update_progress(self, current_index, max_index):
        self.progress = current_index, max_index

    def scramble_thread(self, source_files, progress):
        count = len(source_files)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/scramble.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit("Scrambling measurements")

        # scrambling self.mdf
        for i, source_file in enumerate(source_files):
            progress.signals.setLabelText.emit(
                f"Scrambling file {i+1} of {count}\n{source_file}"
            )

            result = MDF.scramble(name=source_file, progress=progress)
            if result is TERMINATED:
                return

    def scramble_finished(self):
        self._progress = None

    def scramble(self, event):
        count = self.files_list.count()
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        if not count:
            return

        self._progress = setup_progress(parent=self, autoclose=False)
        self._progress.finished.connect(self.scramble_finished)

        self._progress.run_thread_with_progress(
            target=self.scramble_thread,
            args=(source_files,),
            kwargs={},
        )

    def extract_bus_logging_finished(self):
        if self._progress.error is None and self._progress.result is not TERMINATED:
            message = self._progress.result

            self.output_info_bus.setPlainText("\n".join(message))

        self._progress = None

    def extract_bus_logging(self, event):
        version = self.extract_bus_format.currentText()

        self.output_info_bus.setPlainText("")

        database_files = {}

        count1 = self.can_database_list.count()
        if count1:
            database_files["CAN"] = []
            for i in range(count1):
                item = self.can_database_list.item(i)
                widget = self.can_database_list.itemWidget(item)
                database_files["CAN"].append(
                    (widget.database.text(), widget.bus.currentIndex())
                )

        count2 = self.lin_database_list.count()
        if count2:
            database_files["LIN"] = []
            for i in range(count2):
                item = self.lin_database_list.item(i)
                widget = self.lin_database_list.itemWidget(item)
                database_files["LIN"].append(
                    (widget.database.text(), widget.bus.currentIndex())
                )

        compression = self.extract_bus_compression.currentIndex()

        count = self.files_list.count()

        if not count or not (count1 + count2):
            return

        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        self._progress = setup_progress(parent=self)
        self._progress.finished.connect(self.extract_bus_logging_finished)

        self._progress.run_thread_with_progress(
            target=self.extract_bus_logging_thread,
            args=(source_files, database_files, count, compression, version),
            kwargs={},
        )

    def extract_bus_logging_thread(
        self, source_files, database_files, count, compression, version, progress
    ):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/down.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit("Extract Bus logging from measurements")
        progress.signals.setLabelText.emit(
            f'Extracting Bus logging from "{count}" files'
        )

        files = self._prepare_files(list(source_files), progress)

        message = []

        progress.signals.setValue.emit(0)
        progress.signals.setMaximum.emit(count)

        for i, (file, source_file) in enumerate(zip(files, source_files)):
            progress.signals.setLabelText.emit(
                f"Extracting Bus logging from file {i+1} of {count}\n{source_file}"
            )

            if not isinstance(file, MDF):
                mdf = MDF(file)
            else:
                mdf = file

            if mdf is TERMINATED:
                return

            mdf.last_call_info = {}

            result = mdf.extract_bus_logging(
                database_files=database_files,
                version=version,
                prefix=self.prefix.text().strip(),
                progress=progress,
            )

            if result is TERMINATED:
                return
            else:
                mdf_ = result

            bus_call_info = dict(mdf.last_call_info)

            for bus, call_info in bus_call_info.items():
                found_id_count = sum(len(e) for e in call_info["found_ids"].values())

                message += [
                    "",
                    f'Summary of "{mdf.name}":',
                    f'- {found_id_count} of {len(call_info["total_unique_ids"])} IDs in the MDF4 file were matched in the DBC and converted',
                ]
                if call_info["unknown_id_count"]:
                    message.append(
                        f'- {call_info["unknown_id_count"]} unknown IDs in the MDF4 file'
                    )
                else:
                    message.append("- no unknown IDs inf the MDF4 file")

                message += [
                    "",
                    "Detailed information:",
                    "",
                    "The following Bus IDs were in the MDF log file and matched in the DBC:",
                ]
                for dbc_name, found_ids in call_info["found_ids"].items():
                    for msg_id, msg_name in sorted(found_ids):
                        try:
                            message.append(
                                f"- 0x{msg_id:X} --> {msg_name} in <{dbc_name}>"
                            )
                        except:
                            pgn, sa = msg_id
                            message.append(
                                f"- PGN=0x{pgn:X} SA=0x{sa:X} --> {msg_name} in <{dbc_name}>"
                            )

                message += [
                    "",
                    "The following Bus IDs were in the MDF log file, but not matched in the DBC:",
                ]
                unknown_standard_can = sorted(
                    [e for e in call_info["unknown_ids"] if isinstance(e, int)]
                )
                unknown_j1939 = sorted(
                    [e for e in call_info["unknown_ids"] if not isinstance(e, int)]
                )
                for msg_id in unknown_standard_can:
                    message.append(f"- 0x{msg_id:X}")

                for pgn, sa in unknown_j1939:
                    message.append(f"- PGN=0x{pgn:X} SA=0x{sa:X}")

            file_name = source_file.with_suffix(
                ".bus_logging.mdf" if version < "4.00" else ".bus_logging.mf4"
            )

            # then save it
            progress.signals.setLabelText.emit(
                f'Saving extracted Bus logging file {i+1} to "{file_name}"'
            )

            result = mdf_.save(
                dst=file_name,
                compression=compression,
                overwrite=True,
                progress=progress,
            )
            if result is TERMINATED:
                return

        return message

    def extract_bus_csv_logging_finished(self):
        if self._progress.error is None and self._progress.result is not TERMINATED:
            message = self._progress.result

            self.output_info_bus.setPlainText("\n".join(message))

        self._progress = None

    def extract_bus_csv_logging(self, event):
        version = self.extract_bus_format.currentText()

        self.output_info_bus.setPlainText("")

        database_files = {}

        count1 = self.can_database_list.count()
        if count1:
            database_files["CAN"] = []
            for i in range(count1):
                item = self.can_database_list.item(i)
                widget = self.can_database_list.itemWidget(item)
                database_files["CAN"].append(
                    (widget.database.text(), widget.bus.currentIndex())
                )

        count2 = self.lin_database_list.count()
        if count2:
            database_files["LIN"] = []
            for i in range(count2):
                item = self.lin_database_list.item(i)
                widget = self.lin_database_list.itemWidget(item)
                database_files["LIN"].append(
                    (widget.database.text(), widget.bus.currentIndex())
                )

        single_time_base = self.single_time_base_bus.checkState() == QtCore.Qt.Checked
        time_from_zero = self.time_from_zero_bus.checkState() == QtCore.Qt.Checked
        empty_channels = self.empty_channels_bus.currentText()
        raster = self.export_raster_bus.value() or None
        time_as_date = self.bus_time_as_date.checkState() == QtCore.Qt.Checked
        delimiter = self.delimiter_bus.text() or ","
        doublequote = self.doublequote_bus.checkState() == QtCore.Qt.Checked
        escapechar = self.escapechar_bus.text() or None
        lineterminator = (
            self.lineterminator_bus.text().replace("\\r", "\r").replace("\\n", "\n")
        )
        quotechar = self.quotechar_bus.text() or '"'
        quoting = self.quoting_bus.currentText()
        add_units = self.add_units_bus.checkState() == QtCore.Qt.Checked

        count = self.files_list.count()

        if not count or not (count1 + count2):
            return

        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        self._progress = setup_progress(parent=self)
        self._progress.finished.connect(self.extract_bus_csv_logging_finished)

        self._progress.run_thread_with_progress(
            target=self.extract_bus_csv_logging_thread,
            args=(
                source_files,
                database_files,
                version,
                single_time_base,
                time_from_zero,
                empty_channels,
                raster,
                time_as_date,
                delimiter,
                doublequote,
                escapechar,
                lineterminator,
                quotechar,
                quoting,
                add_units,
                count,
            ),
            kwargs={},
        )

    def extract_bus_csv_logging_thread(
        self,
        source_files,
        database_files,
        version,
        single_time_base,
        time_from_zero,
        empty_channels,
        raster,
        time_as_date,
        delimiter,
        doublequote,
        escapechar,
        lineterminator,
        quotechar,
        quoting,
        add_units,
        count,
        progress,
    ):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/csv.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit(
            "Extract Bus logging from measurements to CSV"
        )
        progress.signals.setLabelText.emit(
            f'Extracting Bus logging from "{count}" files'
        )

        files = self._prepare_files(list(source_files), progress)

        message = []

        for i, (file, source_file) in enumerate(zip(files, source_files)):
            progress.signals.setLabelText.emit(
                f"Extracting Bus logging from file {i+1} of {count}"
            )

            if not isinstance(file, MDF):
                mdf = MDF(file)
            else:
                mdf = file

            if mdf is TERMINATED:
                return

            mdf.last_call_info = {}

            result = mdf.extract_bus_logging(
                database_files=database_files,
                version=version,
                prefix=self.prefix.text().strip(),
                progress=progress,
            )
            if result is TERMINATED:
                return
            else:
                mdf_ = result

            bus_call_info = dict(mdf.last_call_info)

            for bus, call_info in bus_call_info.items():
                found_id_count = sum(len(e) for e in call_info["found_ids"].values())

                message += [
                    "",
                    f'Summary of "{mdf.name}":',
                    f'- {found_id_count} of {len(call_info["total_unique_ids"])} IDs in the MDF4 file were matched in the DBC and converted',
                ]
                if call_info["unknown_id_count"]:
                    message.append(
                        f'- {call_info["unknown_id_count"]} unknown IDs in the MDF4 file'
                    )
                else:
                    message.append("- no unknown IDs inf the MDF4 file")

                message += [
                    "",
                    "Detailed information:",
                    "",
                    "The following Bus IDs were in the MDF log file and matched in the DBC:",
                ]
                for dbc_name, found_ids in call_info["found_ids"].items():
                    for msg_id, msg_name in sorted(found_ids):
                        message.append(f"- 0x{msg_id:X} --> {msg_name} in <{dbc_name}>")

                message += [
                    "",
                    "The following Bus IDs were in the MDF log file, but not matched in the DBC:",
                ]
                for msg_id in sorted(call_info["unknown_ids"]):
                    message.append(f"- 0x{msg_id:X}")

            file_name = source_file.with_suffix(".bus_logging.csv")

            # then save it
            progress.signals.setLabelText.emit(
                f'Saving extracted Bus logging file {i+1} to "{file_name}"'
            )

            mdf_.configure(
                integer_interpolation=self.integer_interpolation,
                float_interpolation=self.float_interpolation,
            )

            result = mdf_.export(
                **{
                    "fmt": "csv",
                    "filename": file_name,
                    "single_time_base": single_time_base,
                    "time_from_zero": time_from_zero,
                    "empty_channels": empty_channels,
                    "raster": raster or None,
                    "time_as_date": time_as_date,
                    "ignore_value2text_conversions": self.ignore_value2text_conversions,
                    "delimiter": delimiter,
                    "doublequote": doublequote,
                    "escapechar": escapechar,
                    "lineterminator": lineterminator,
                    "quotechar": quotechar,
                    "quoting": quoting,
                    "add_units": add_units,
                    "progress": progress,
                }
            )
            if result is TERMINATED:
                return

        return message

    def load_can_database(self, event):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select CAN database file",
            "",
            "ARXML or DBC (*.dbc *.arxml)",
            "ARXML or DBC (*.dbc *.arxml)",
        )

        if file_names:
            file_names = [
                name
                for name in file_names
                if Path(name).suffix.lower() in (".arxml", ".dbc")
            ]

        if file_names:
            for database in file_names:
                item = QtWidgets.QListWidgetItem()
                widget = DatabaseItem(database, bus_type="CAN")

                self.can_database_list.addItem(item)
                self.can_database_list.setItemWidget(item, widget)
                item.setSizeHint(widget.sizeHint())

    def load_lin_database(self, event):
        file_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select LIN database file",
            "",
            "ARXML or DBC database (*.dbc *.arxml);;LDF database (*.ldf);;All supported formats (*.dbc *.arxml *ldf)",
            "All supported formats (*.dbc *.arxml *ldf)",
        )

        if file_names:
            file_names = [
                name
                for name in file_names
                if Path(name).suffix.lower() in (".arxml", ".dbc", ".ldf")
            ]

        if file_names:
            for database in file_names:
                item = QtWidgets.QListWidgetItem()
                widget = DatabaseItem(database, bus_type="LIN")

                self.lin_database_list.addItem(item)
                self.lin_database_list.setItemWidget(item, widget)
                item.setSizeHint(widget.sizeHint())

    def concatenate_finished(self):
        self._progress = None

    def concatenate(self, event):
        count = self.files_list.count()

        if not count:
            return

        version = self.concatenate_format.currentText()

        sync = self.concatenate_sync.checkState() == QtCore.Qt.Checked
        add_samples_origin = (
            self.concatenate_add_samples_origin.checkState() == QtCore.Qt.Checked
        )

        split = self.concatenate_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.concatenate_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.concatenate_compression.currentIndex()

        if version < "4.00":
            output_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select output measurement file",
                "",
                "MDF version 3 files (*.dat *.mdf);;All files (*.*)",
                "MDF version 3 files (*.dat *.mdf)",
            )
        else:
            output_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select output measurement file",
                "",
                f"MDF version 4 files (*.mf4 *.mf4z);;All files (*.*)",
                "MDF version 4 files (*.mf4 *.mf4z)",
            )

        if not output_file_name:
            return

        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        self._progress = setup_progress(parent=self, autoclose=False)
        self._progress.finished.connect(self.concatenate_finished)

        self._progress.run_thread_with_progress(
            target=self.concatenate_thread,
            args=(
                output_file_name,
                version,
                source_files,
                sync,
                add_samples_origin,
                split_size,
                compression,
            ),
            kwargs={},
        )

    def concatenate_thread(
        self,
        output_file_name,
        version,
        source_files,
        sync,
        add_samples_origin,
        split_size,
        compression,
        progress,
    ):
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/stack.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit(
            f"Stacking files and saving to {version} format"
        )

        output_file_name = Path(output_file_name)

        files = self._prepare_files(source_files, progress)

        result = MDF.concatenate(
            files=files,
            version=version,
            sync=sync,
            add_samples_origin=add_samples_origin,
            progress=progress,
        )

        if result is TERMINATED:
            return
        else:
            mdf = result

        mdf.configure(write_fragment_size=split_size)

        # save it
        progress.signals.setLabelText.emit(f'Saving output file "{output_file_name}"')

        result = mdf.save(
            dst=output_file_name,
            compression=compression,
            overwrite=True,
            progress=progress,
        )

        if result is not TERMINATED:
            return result

    def stack_thread(
        self,
        output_file_name,
        version,
        source_files,
        sync,
        add_samples_origin,
        split_size,
        compression,
        progress,
    ):
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/stack.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit(
            f"Stacking files and saving to {version} format"
        )

        output_file_name = Path(output_file_name)

        files = self._prepare_files(source_files, progress)

        result = MDF.stack(
            files=files,
            version=version,
            sync=sync,
            add_samples_origin=add_samples_origin,
            progress=progress,
        )

        if result is TERMINATED:
            return
        else:
            mdf = result

        mdf.configure(write_fragment_size=split_size)

        # save it
        progress.signals.setLabelText.emit(f'Saving output file "{output_file_name}"')

        result = mdf.save(
            dst=output_file_name,
            compression=compression,
            overwrite=True,
            progress=progress,
        )

        if result is not TERMINATED:
            return result

    def stack_finished(self):
        self._progress = None

    def stack(self, event):
        count = self.files_list.count()

        if not count:
            return

        version = self.stack_format.currentText()

        sync = self.stack_sync.checkState() == QtCore.Qt.Checked
        add_samples_origin = (
            self.stack_add_samples_origin.checkState() == QtCore.Qt.Checked
        )

        split = self.stack_split.checkState() == QtCore.Qt.Checked
        if split:
            split_size = int(self.stack_split_size.value() * 1024 * 1024)
        else:
            split_size = 0

        compression = self.stack_compression.currentIndex()

        if version < "4.00":
            output_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select output measurement file",
                "",
                "MDF version 3 files (*.dat *.mdf);;All files (*.*)",
                "MDF version 3 files (*.dat *.mdf)",
            )
        else:
            output_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select output measurement file",
                "",
                f"MDF version 4 files (*.mf4 *.mf4z);;All files (*.*)",
                "MDF version 4 files (*.mf4 *.mf4z)",
            )

        if not output_file_name:
            return

        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        self._progress = setup_progress(parent=self, autoclose=False)
        self._progress.finished.connect(self.stack_finished)

        self._progress.run_thread_with_progress(
            target=self.stack_thread,
            args=(
                output_file_name,
                version,
                source_files,
                sync,
                add_samples_origin,
                split_size,
                compression,
            ),
            kwargs={},
        )

    def _as_mdf(self, file_name):
        file_name = Path(file_name)
        suffix = file_name.suffix.lower()

        if suffix in (".erg", ".bsig", ".dl3"):
            try:
                from mfile import BSIG, DL3, ERG
            except ImportError:
                from cmerg import BSIG, ERG

            if suffix == ".erg":
                cls = ERG
            elif suffix == ".bsig":
                cls = BSIG
            else:
                cls = DL3

            mdf = cls(file_name).export_mdf()

        elif suffix in (".mdf", ".mf4", ".mf4z"):
            mdf = MDF(file_name)

        return mdf

    def _prepare_files(self, files=None, progress=None):
        count = self.files_list.count()

        if files is None:
            files = [Path(self.files_list.item(row).text()) for row in range(count)]

        progress.signals.setMaximum.emit(count)
        progress.signals.setValue.emit(0)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/list.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit("Preparing measurements")

        for i, file_name in enumerate(files):
            progress.signals.setLabelText.emit(
                f"Preparing the file {i+1} of {count} from {file_name.suffix.lower()} to .mf4\n{file_name}"
            )
            files[i] = self._as_mdf(file_name)
            progress.signals.setValue.emit(i + 1)

        return files

    def _get_filtered_channels(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

        channels = []
        count = 0
        total = 0

        if self.filter_view.currentText() == "Internal file structure":
            while iterator.value():
                item = iterator.value()

                group, index = item.entry
                if index != 0xFFFFFFFFFFFFFFFF:
                    total += 1

                if item.checkState(0) == QtCore.Qt.Checked:
                    if index != 0xFFFFFFFFFFFFFFFF:
                        channels.append((item.name, group, index))
                        count += 1

                iterator += 1
        else:
            while iterator.value():
                item = iterator.value()

                if item.checkState(0) == QtCore.Qt.Checked:
                    group, index = item.entry
                    channels.append((item.name, group, index))
                    count += 1

                total += 1

                iterator += 1

        if not channels:
            return False, channels
        else:
            if total == count:
                return False, channels
            else:
                return True, channels

    def raster_search(self, event):
        if not self.files_list.count():
            return

        with MDF(self.files_list.item(0).text()) as mdf:
            dlg = AdvancedSearch(
                mdf,
                show_add_window=False,
                show_pattern=False,
                parent=self,
                return_names=True,
            )
            dlg.setModal(True)
            dlg.exec_()
            result = dlg.result
            if result:
                name = list(result)[0]
                self.raster_channel.setCurrentText(name)

    def filter_changed(self, item, column):
        name = item.text(0)
        if item.checkState(0) == QtCore.Qt.Checked:
            self._selected_filter.add(name)
        else:
            if name in self._selected_filter:
                self._selected_filter.remove(name)
        self._filter_timer.start(10)

    def update_selected_filter_channels(self, *args):
        self.selected_filter_channels.clear()
        self.selected_filter_channels.addItems(sorted(self._selected_filter))

    def search(self, event=None):
        count = self.files_list.count()
        if not count:
            return

        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        for file_name in source_files:
            if file_name.suffix.lower() in (".mdf", ".mf4"):
                break
        else:
            file_name = source_files[0]

        mdf = self._as_mdf(file_name)

        try:
            widget = self.filter_tree
            view = self.filter_view

            dlg = AdvancedSearch(
                mdf,
                show_add_window=False,
                show_pattern=False,
                parent=self,
                show_apply=True,
                apply_text="Check signals",
            )
            dlg.setModal(True)
            dlg.exec_()
            result = dlg.result

            if result:
                if view.currentText() == "Internal file structure":
                    iterator = QtWidgets.QTreeWidgetItemIterator(widget)

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
                            item.setCheckState(0, QtCore.Qt.Checked)

                        iterator += 1
                        ch_cntr += 1
                elif view.currentText() == "Selected channels only":
                    iterator = QtWidgets.QTreeWidgetItemIterator(widget)

                    signals = set()
                    while iterator.value():
                        item = iterator.value()

                        if item.checkState(0) == QtCore.Qt.Checked:
                            signals.add(item.entry)

                        iterator += 1

                    signals = signals | set(result)

                    widget.clear()

                    items = []
                    for entry in signals:
                        gp_index, ch_index = entry
                        ch = mdf.groups[gp_index].channels[ch_index]
                        channel = TreeItem(entry, ch.name, origin_uuid=self.uuid)
                        channel.setText(0, ch.name)
                        channel.setCheckState(0, QtCore.Qt.Checked)
                        items.append(channel)

                    if len(items) < 30000:
                        items = natsorted(items, key=lambda x: x.name)
                    else:
                        items.sort(key=lambda x: x.name)
                    widget.addTopLevelItems(items)

                else:
                    iterator = QtWidgets.QTreeWidgetItemIterator(widget)
                    while iterator.value():
                        item = iterator.value()

                        if item.entry in result:
                            item.setCheckState(0, QtCore.Qt.Checked)

                        iterator += 1
        except:
            print(format_exc())
        finally:
            mdf.close()

    def update_channel_tree(self, *args):
        if self.filter_view.currentIndex() == -1:
            return

        count = self.files_list.count()
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]
        if not count:
            self.filter_tree.clear()
            return
        else:
            uuid = os.urandom(6).hex()

            for file_name in source_files:
                if file_name.suffix.lower() in (".mdf", ".mf4"):
                    break
            else:
                file_name = source_files[0]

            mdf = self._as_mdf(file_name)
            try:
                widget = self.filter_tree
                view = self.filter_view

                iterator = QtWidgets.QTreeWidgetItemIterator(widget)
                signals = set()

                if widget.mode == "Internal file structure":
                    while iterator.value():
                        item = iterator.value()

                        if item.entry[1] != 0xFFFFFFFFFFFFFFFF:
                            if item.checkState(0) == QtCore.Qt.Checked:
                                signals.add(item.entry)

                        iterator += 1
                else:
                    while iterator.value():
                        item = iterator.value()

                        if item.checkState(0) == QtCore.Qt.Checked:
                            signals.add(item.entry)

                        iterator += 1

                widget.clear()
                widget.mode = view.currentText()

                if widget.mode == "Natural sort":
                    items = []
                    for i, group in enumerate(mdf.groups):
                        for j, ch in enumerate(group.channels):
                            entry = i, j

                            channel = TreeItem(entry, ch.name, origin_uuid=uuid)
                            channel.setText(0, ch.name)
                            if entry in signals:
                                channel.setCheckState(0, QtCore.Qt.Checked)
                            else:
                                channel.setCheckState(0, QtCore.Qt.Unchecked)
                            items.append(channel)

                    if len(items) < 30000:
                        items = natsorted(items, key=lambda x: x.name)
                    else:
                        items.sort(key=lambda x: x.name)
                    widget.addTopLevelItems(items)

                elif widget.mode == "Internal file structure":
                    for i, group in enumerate(mdf.groups):
                        entry = i, 0xFFFFFFFFFFFFFFFF
                        channel_group = TreeItem(entry, origin_uuid=uuid)
                        comment = group.channel_group.comment
                        comment = extract_xml_comment(comment)

                        if comment:
                            channel_group.setText(0, f"Channel group {i} ({comment})")
                        else:
                            channel_group.setText(0, f"Channel group {i}")
                        channel_group.setFlags(
                            channel_group.flags()
                            | QtCore.Qt.ItemIsAutoTristate
                            | QtCore.Qt.ItemIsUserCheckable
                        )

                        widget.addTopLevelItem(channel_group)

                        channels = [
                            HelperChannel(name=ch.name, entry=(i, j))
                            for j, ch in enumerate(group.channels)
                        ]

                        add_children(
                            channel_group,
                            channels,
                            group.channel_dependencies,
                            signals,
                            entries=None,
                            origin_uuid=uuid,
                        )
                else:
                    items = []
                    for entry in signals:
                        gp_index, ch_index = entry
                        ch = mdf.groups[gp_index].channels[ch_index]
                        channel = TreeItem(entry, ch.name, origin_uuid=uuid)
                        channel.setText(0, ch.name)
                        channel.setCheckState(0, QtCore.Qt.Checked)
                        items.append(channel)

                    if len(items) < 30000:
                        items = natsorted(items, key=lambda x: x.name)
                    else:
                        items.sort(key=lambda x: x.name)
                    widget.addTopLevelItems(items)
            except:
                print(format_exc())
            finally:
                mdf.close()

    def _current_options(self):
        options = {
            "needs_cut": self.cut_group.isChecked(),
            "cut_start": self.cut_start.value(),
            "cut_stop": self.cut_stop.value(),
            "cut_time_from_zero": self.cut_time_from_zero.checkState()
            == QtCore.Qt.Checked,
            "whence": int(self.whence.checkState() == QtCore.Qt.Checked),
            "needs_resample": self.resample_group.isChecked(),
            "raster_type_step": self.raster_type_step.isChecked(),
            "raster_type_channel": self.raster_type_channel.isChecked(),
            "raster": self.raster.value(),
            "raster_channel": self.raster_channel.currentText(),
            "resample_time_from_zero": self.resample_time_from_zero.checkState()
            == QtCore.Qt.Checked,
            "output_format": self.output_format.currentText(),
        }

        output_format = self.output_format.currentText()

        if output_format == "MDF":
            new = {
                "mdf_version": self.mdf_version.currentText(),
                "mdf_compression": self.mdf_compression.currentIndex(),
                "mdf_split": self.mdf_split.checkState() == QtCore.Qt.Checked,
                "mdf_split_size": self.mdf_split_size.value() * 1024 * 1024,
            }

        elif output_format == "MAT":
            new = {
                "single_time_base": self.single_time_base_mat.checkState()
                == QtCore.Qt.Checked,
                "time_from_zero": self.time_from_zero_mat.checkState()
                == QtCore.Qt.Checked,
                "time_as_date": self.time_as_date_mat.checkState() == QtCore.Qt.Checked,
                "use_display_names": self.use_display_names_mat.checkState()
                == QtCore.Qt.Checked,
                "reduce_memory_usage": self.reduce_memory_usage_mat.checkState()
                == QtCore.Qt.Checked,
                "compression": self.export_compression_mat.currentText() == "enabled",
                "empty_channels": self.empty_channels_mat.currentText(),
                "mat_format": self.mat_format.currentText(),
                "oned_as": self.oned_as.currentText(),
                "raw": self.raw_mat.checkState() == QtCore.Qt.Checked,
            }

        elif output_format == "CSV":
            new = {
                "single_time_base": self.single_time_base_csv.checkState()
                == QtCore.Qt.Checked,
                "time_from_zero": self.time_from_zero_csv.checkState()
                == QtCore.Qt.Checked,
                "time_as_date": self.time_as_date_csv.checkState() == QtCore.Qt.Checked,
                "use_display_names": self.use_display_names_csv.checkState()
                == QtCore.Qt.Checked,
                "reduce_memory_usage": False,
                "compression": False,
                "empty_channels": self.empty_channels_csv.currentText(),
                "raw": self.raw_csv.checkState() == QtCore.Qt.Checked,
                "delimiter": self.delimiter.text() or ",",
                "doublequote": self.doublequote.checkState() == QtCore.Qt.Checked,
                "escapechar": self.escapechar.text() or None,
                "lineterminator": self.lineterminator.text()
                .replace("\\r", "\r")
                .replace("\\n", "\n"),
                "quotechar": self.quotechar.text() or '"',
                "quoting": self.quoting.currentText(),
                "mat_format": None,
                "oned_as": None,
            }

        else:
            new = {
                "single_time_base": self.single_time_base.checkState()
                == QtCore.Qt.Checked,
                "time_from_zero": self.time_from_zero.checkState() == QtCore.Qt.Checked,
                "time_as_date": self.time_as_date.checkState() == QtCore.Qt.Checked,
                "use_display_names": self.use_display_names.checkState()
                == QtCore.Qt.Checked,
                "reduce_memory_usage": self.reduce_memory_usage.checkState()
                == QtCore.Qt.Checked,
                "compression": self.export_compression.currentText(),
                "empty_channels": self.empty_channels.currentText(),
                "mat_format": None,
                "oned_as": None,
                "raw": self.raw.checkState() == QtCore.Qt.Checked,
            }

        options.update(new)

        class Options:
            def __init__(self, opts):
                self._opts = opts
                for k, v in opts.items():
                    setattr(self, k, v)

        return Options(options)

    def apply_processing_finished(self):
        self._progress = None

    def apply_processing(self, event):
        opts = self._current_options()

        output_format = opts.output_format

        if output_format == "HDF5":
            try:
                from h5py import File as HDF5
            except ImportError:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export to HDF5 unavailale",
                    "h5py package not found; export to HDF5 is unavailable",
                )
                return

        elif output_format == "MAT":
            if opts.mat_format == "7.3":
                try:
                    from hdf5storage import savemat
                except ImportError:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Export to mat v7.3 unavailale",
                        "hdf5storage package not found; export to mat 7.3 is unavailable",
                    )
                    return
            else:
                try:
                    from scipy.io import savemat
                except ImportError:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Export to mat v4 and v5 unavailale",
                        "scipy package not found; export to mat is unavailable",
                    )
                    return

        elif output_format == "Parquet":
            try:
                from fastparquet import write as write_parquet
            except ImportError:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export to parquet unavailale",
                    "fastparquet package not found; export to parquet is unavailable",
                )
                return

        self._progress = setup_progress(parent=self, autoclose=False)
        self._progress.finished.connect(self.apply_processing_finished)

        self._progress.run_thread_with_progress(
            target=self.apply_processing_thread,
            args=(),
            kwargs={},
        )

    def apply_processing_thread(self, progress):
        count = self.files_list.count()
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        if not count:
            return

        opts = self._current_options()

        output_format = opts.output_format

        if output_format == "MDF":
            version = opts.mdf_version

        if output_format == "HDF5":
            suffix = ".hdf"
            from h5py import File as HDF5

        elif output_format == "MAT":
            suffix = ".mat"
            if opts.mat_format == "7.3":
                from hdf5storage import savemat
            else:
                from scipy.io import savemat

        elif output_format == "Parquet":
            suffix = ".parquet"
            from fastparquet import write as write_parquet

        elif output_format == "CSV":
            suffix = ".csv"

        elif output_format == "ASC":
            suffix = ".asc"

        needs_filter, channels = self._get_filtered_channels()

        output_folder = self.modify_output_folder.text().strip()
        if output_folder:
            output_folder = Path(output_folder)
        else:
            output_folder = None

        try:
            root = Path(os.path.commonprefix(source_files)).parent

        except ValueError:
            root = None

        split_size = opts.mdf_split_size if output_format == "MDF" else 0

        integer_interpolation = self.integer_interpolation
        float_interpolation = self.float_interpolation

        files = self._prepare_files(list(source_files), progress)

        for mdf_index, (mdf_file, source_file) in enumerate(zip(files, source_files)):
            mdf_file.configure(
                read_fragment_size=split_size,
                integer_interpolation=self.integer_interpolation,
                float_interpolation=self.float_interpolation,
            )

            mdf = mdf_file

            if needs_filter:
                icon = QtGui.QIcon()
                icon.addPixmap(
                    QtGui.QPixmap(":/filter.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                )
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit(
                    f"Filtering measurement {mdf_index+i} of {count}"
                )
                progress.signals.setLabelText.emit(
                    f'Filtering selected channels from\n"{source_file}"'
                )

                # filtering self.mdf
                result = mdf.filter(
                    channels=channels,
                    version=opts.mdf_version if output_format == "MDF" else "4.10",
                    progress=progress,
                )

                if result is TERMINATED:
                    return
                else:
                    mdf = result

                mdf.configure(
                    read_fragment_size=split_size,
                    write_fragment_size=split_size,
                    integer_interpolation=integer_interpolation,
                    float_interpolation=float_interpolation,
                )

            if opts.needs_cut:
                icon = QtGui.QIcon()
                icon.addPixmap(
                    QtGui.QPixmap(":/cut.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                )
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit(
                    f"Cutting measurement {mdf_index+1} of {count}"
                )
                progress.signals.setLabelText.emit(
                    f"Cutting from {opts.cut_start}s to {opts.cut_stop}s from \n{source_file}"
                )

                # cut self.mdf
                target = mdf.cut
                result = target(
                    start=opts.cut_start,
                    stop=opts.cut_stop,
                    whence=opts.whence,
                    version=opts.mdf_version if output_format == "MDF" else "4.10",
                    time_from_zero=opts.cut_time_from_zero,
                    progress=progress,
                )

                if result is TERMINATED:
                    return
                else:
                    mdf.close()
                    mdf = result

                mdf.configure(
                    read_fragment_size=split_size,
                    write_fragment_size=split_size,
                    integer_interpolation=integer_interpolation,
                    float_interpolation=float_interpolation,
                )

            if opts.needs_resample:
                if opts.raster_type_channel:
                    raster = opts.raster_channel
                    message = f'Resampling using channel "{raster}"\n{source_file}'
                else:
                    raster = opts.raster
                    message = f"Resampling to {raster}s raster\n{source_file}"

                icon = QtGui.QIcon()
                icon.addPixmap(
                    QtGui.QPixmap(":/resample.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                )
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit(
                    f"Resampling measurement {mdf_index+1} of {count}"
                )
                progress.signals.setLabelText.emit(message)

                # resample self.mdf
                target = mdf.resample

                result = target(
                    raster=raster,
                    version=opts.mdf_version if output_format == "MDF" else "4.10",
                    time_from_zero=opts.resample_time_from_zero,
                    progress=progress,
                )

                if result is TERMINATED:
                    return
                else:
                    mdf.close()
                    mdf = result

                mdf.configure(
                    read_fragment_size=split_size,
                    write_fragment_size=split_size,
                    integer_interpolation=integer_interpolation,
                    float_interpolation=float_interpolation,
                )

            if output_format == "MDF":
                if mdf.version != version:
                    icon = QtGui.QIcon()
                    icon.addPixmap(
                        QtGui.QPixmap(":/convert.png"),
                        QtGui.QIcon.Normal,
                        QtGui.QIcon.Off,
                    )
                    progress.signals.setWindowIcon.emit(icon)
                    progress.signals.setWindowTitle.emit(
                        f"Converting measurement {mdf_index+1} of {count}"
                    )
                    progress.signals.setLabelText.emit(
                        f'Converting "{source_file}" from {mdf.version} to {version}'
                    )

                    # convert self.mdf
                    result = mdf.convert(
                        version=version,
                        progress=progress,
                    )

                    if result is TERMINATED:
                        return
                    else:
                        mdf.close()
                        mdf = result

                if version >= "4.00":
                    suffix = ".mf4"
                else:
                    suffix = ".mdf"

                mdf.configure(
                    read_fragment_size=split_size,
                    write_fragment_size=split_size,
                    integer_interpolation=self.integer_interpolation,
                    float_interpolation=self.float_interpolation,
                )

                if output_folder is not None:
                    if root is None:
                        file_name = output_folder / Path(mdf_file.original_name).name
                    else:
                        file_name = output_folder / Path(mdf_file.name).relative_to(
                            root
                        )

                    if not file_name.parent.exists():
                        os.makedirs(file_name.parent, exist_ok=True)
                else:
                    file_name = Path(mdf_file.original_name)
                    file_name = file_name.parent / (
                        file_name.stem + ".modified" + suffix
                    )

                file_name = file_name.with_suffix(suffix)

                # then save it
                icon = QtGui.QIcon()
                icon.addPixmap(
                    QtGui.QPixmap(":/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                )
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit(
                    f"Saving measurement {mdf_index+1} of {count}"
                )
                progress.signals.setLabelText.emit(
                    f"Saving output file {mdf_index+1} of {count}\n{source_file}"
                )

                result = mdf.save(
                    dst=file_name,
                    compression=opts.mdf_compression,
                    overwrite=True,
                    progress=progress,
                )

                if result is TERMINATED:
                    return

            else:
                icon = QtGui.QIcon()
                icon.addPixmap(
                    QtGui.QPixmap(":/export.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                )
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit(
                    f"Export measurement {mdf_index+1} of {count}"
                )
                progress.signals.setLabelText.emit(
                    f"Exporting measurement {mdf_index+1} of {count} to {output_format} (be patient this might take a while)\n{source_file}"
                )

                delimiter = self.delimiter.text() or ","
                doublequote = self.doublequote.checkState() == QtCore.Qt.Checked
                escapechar = self.escapechar.text() or None
                lineterminator = (
                    self.lineterminator.text().replace("\\r", "\r").replace("\\n", "\n")
                )
                quotechar = self.quotechar.text() or '"'
                quoting = self.quoting.currentText()
                add_units = self.add_units.checkState() == QtCore.Qt.Checked

                target = self.mdf.export if mdf is None else mdf.export
                kwargs = {
                    "fmt": opts.output_format.lower(),
                    "filename": file_name,
                    "single_time_base": opts.single_time_base,
                    "use_display_names": opts.use_display_names,
                    "time_from_zero": opts.time_from_zero,
                    "empty_channels": opts.empty_channels,
                    "format": opts.mat_format,
                    "raster": None,
                    "oned_as": opts.oned_as,
                    "reduce_memory_usage": opts.reduce_memory_usage,
                    "compression": opts.compression,
                    "time_as_date": opts.time_as_date,
                    "ignore_value2text_conversions": self.ignore_value2text_conversions,
                    "raw": opts.raw,
                    "delimiter": delimiter,
                    "doublequote": doublequote,
                    "escapechar": escapechar,
                    "lineterminator": lineterminator,
                    "quotechar": quotechar,
                    "quoting": quoting,
                    "add_units": add_units,
                }

                target(**kwargs)

    def change_modify_output_folder(self, event=None):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", ""
        )
        if folder:
            self.modify_output_folder.setText(folder)

    def output_format_changed(self, name):
        if name == "MDF":
            self.output_options.setCurrentIndex(0)
        elif name == "MAT":
            self.output_options.setCurrentIndex(2)

            self.export_compression_mat.clear()
            self.export_compression_mat.addItems(["enabled", "disabled"])
            self.export_compression_mat.setCurrentIndex(0)
        elif name == "CSV":
            self.output_options.setCurrentIndex(3)
        elif name == "ASC":
            self.output_options.setCurrentIndex(4)
        else:
            self.output_options.setCurrentIndex(1)
            if name == "Parquet":
                self.export_compression.setEnabled(True)
                self.export_compression.clear()
                self.export_compression.addItems(["GZIP", "SNAPPY"])
                self.export_compression.setCurrentIndex(0)
            elif name == "HDF5":
                self.export_compression.setEnabled(True)
                self.export_compression.clear()
                self.export_compression.addItems(["gzip", "lzf", "szip"])
                self.export_compression.setCurrentIndex(0)
            elif name:
                self.export_compression.clear()
                self.export_compression.setEnabled(False)

    def sort_alphabetically(self, event=None):
        count = self.files_list.count()

        if not count:
            return

        source_files = natsorted(
            [self.files_list.item(row).text() for row in range(count)]
        )

        self.files_list.clear()
        self.files_list.addItems(source_files)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/file.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        for row in range(count):
            self.files_list.item(row).setIcon(icon)

    def sort_by_start_time(self, event=None):
        count = self.files_list.count()

        if not count:
            return

        source_files = [self.files_list.item(row).text() for row in range(count)]

        start_times = []

        for file_name in source_files:
            with open(file_name, "rb") as f:
                f.seek(64)
                blk_id = f.read(2)
                block_type = HeaderBlockV4 if blk_id == b"##" else HeaderBlockV3
                header = block_type(stream=f, address=64)
                start_times.append((header.start_time, file_name))

        try:
            start_times = sorted(start_times)
        except TypeError:
            start_times = [
                (st.replace(tzinfo=timezone.utc), name) for (st, name) in start_times
            ]
            start_times = sorted(start_times)

        self.files_list.clear()
        self.files_list.addItems([item[1] for item in start_times])

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/file.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        for row in range(count):
            self.files_list.item(row).setIcon(icon)

    def save_filter_list(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select output filter list file",
            "",
            "CANape Lab file (*.lab)",
            "CANape Lab file (*.lab)",
        )

        if file_name:
            file_name = Path(file_name)

            iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

            signals = []
            if self.filter_view.currentText() == "Internal file structure":
                while True:
                    item = iterator.value()
                    if item is None:
                        break

                    iterator += 1

                    if item.parent() is None:
                        continue

                    if item.checkState(0) == QtCore.Qt.Checked:
                        signals.append(item.text(0))
            else:
                while True:
                    item = iterator.value()
                    if item is None:
                        break

                    iterator += 1

                    if item.checkState(0) == QtCore.Qt.Checked:
                        signals.append(item.text(0))

            suffix = file_name.suffix.lower()
            if suffix == ".lab":
                section_name, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Provide .lab file section name",
                    "Section name:",
                )
                if not ok:
                    section_name = "Selected channels"

            with open(file_name, "w") as output:
                if suffix == ".lab":
                    output.write(f"[{section_name}]\n")
                output.write("\n".join(natsorted(signals)))

    def load_filter_list(self, event=None, file_name=None):
        if file_name is None:
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select channel list file",
                "",
                "Config file (*.cfg);;Display files (*.dsp *.dspf);;CANape Lab file (*.lab);;All file types (*.cfg *.dsp *.dspf *.lab)",
                "All file types (*.cfg *.dsp *.dspf *.lab)",
            )

            if file_name is None or Path(file_name).suffix.lower() not in (
                ".cfg",
                ".dsp",
                ".dspf",
                ".lab",
            ):
                return

        if not isinstance(file_name, dict):
            file_name = Path(file_name)

            if file_name.suffix.lower() == ".lab":
                info = load_lab(file_name)
                if info:
                    if len(info) > 1:
                        lab_section, ok = QtWidgets.QInputDialog.getItem(
                            self,
                            "Please select the ASAP section name",
                            "Available sections:",
                            list(info),
                            0,
                            False,
                        )
                        if ok:
                            channels = info[lab_section]
                        else:
                            return
                    else:
                        channels = list(info.values())[0]

            else:
                channels = load_channel_names_from_file(file_name)

        else:
            info = file_name
            channels = info.get("selected_channels", [])

        if channels:
            iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

            if self.filter_view.currentText() == "Internal file structure":
                while iterator.value():
                    item = iterator.value()
                    if item.parent() is None:
                        iterator += 1
                        continue

                    channel_name = item.text(0)
                    if channel_name in channels:
                        item.setCheckState(0, QtCore.Qt.Checked)
                        channels.pop(channels.index(channel_name))
                    else:
                        item.setCheckState(0, QtCore.Qt.Unchecked)

                    iterator += 1
            elif self.filter_view.currentText() == "Natural sort":
                while iterator.value():
                    item = iterator.value()

                    channel_name = item.text(0)
                    if channel_name in channels:
                        item.setCheckState(0, QtCore.Qt.Checked)
                        channels.pop(channels.index(channel_name))
                    else:
                        item.setCheckState(0, QtCore.Qt.Unchecked)

                    iterator += 1

            else:
                items = []
                self.filter_tree.clear()

                for i, gp in enumerate(self.mdf.groups):
                    for j, ch in enumerate(gp.channels):
                        if ch.name in channels:
                            entry = i, j
                            channel = TreeItem(entry, ch.name, origin_uuid=self.uuid)
                            channel.setText(0, ch.name)
                            channel.setCheckState(0, QtCore.Qt.Checked)
                            items.append(channel)

                            channels.pop(channels.index(ch.name))

                if len(items) < 30000:
                    items = natsorted(items, key=lambda x: x.name)
                else:
                    items.sort(key=lambda x: x.name)
                self.filter_tree.addTopLevelItems(items)

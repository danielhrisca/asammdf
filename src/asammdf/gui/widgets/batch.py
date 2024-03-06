from datetime import timezone
import os
from pathlib import Path
from traceback import format_exc

from natsort import natsorted
from PySide6 import QtCore, QtGui, QtWidgets

import asammdf.mdf as mdf_module

from ...blocks.utils import (
    extract_xml_comment,
    load_channel_names_from_file,
    load_lab,
)
from ...blocks.v2_v3_blocks import HeaderBlock as HeaderBlockV3
from ...blocks.v4_blocks import HeaderBlock as HeaderBlockV4
from ...blocks.v4_constants import (
    BUS_TYPE_CAN,
    BUS_TYPE_ETHERNET,
    BUS_TYPE_FLEXRAY,
    BUS_TYPE_LIN,
    BUS_TYPE_USB,
)
from ..dialogs.advanced_search import AdvancedSearch
from ..dialogs.messagebox import MessageBox
from ..ui.batch_widget import Ui_batch_widget
from ..utils import GREEN, HelperChannel, setup_progress, TERMINATED
from .database_item import DatabaseItem
from .tree import add_children
from .tree_item import MinimalTreeItem, TreeItem


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

        self._ignore = False

        self._settings = QtCore.QSettings()

        self.ignore_value2text_conversions = ignore_value2text_conversions
        self.integer_interpolation = integer_interpolation
        self.float_interpolation = float_interpolation

        self.files_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)

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
            widget.insertItems(0, mdf_module.SUPPORTED_VERSIONS)
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
        self.filter_view.setCurrentText(self._settings.value("filter_view", "Internal file structure"))

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
            import scipy  # noqa: F401

            self.mat_format.insertItems(0, ("4", "5", "7.3"))
        except:
            self.mat_format.insertItems(0, ("7.3",))
        self.oned_as.insertItems(0, ("row", "column"))

        self.aspects.setCurrentIndex(0)
        self.setAcceptDrops(True)

        self.files_list.model().rowsInserted.connect(self.update_channel_tree)
        self.files_list.itemsDeleted.connect(self.update_channel_tree)

        self.filter_tree.itemChanged.connect(self.filter_changed)
        self._selected_filter = set()
        self._filter_timer = QtCore.QTimer()
        self._filter_timer.setSingleShot(True)
        self._filter_timer.timeout.connect(self.update_selected_filter_channels)

        databases = {}

        can_databases = self._settings.value("can_databases", [])
        buses = can_databases[::2]
        dbs = can_databases[1::2]

        databases["CAN"] = list(zip(buses, dbs))

        lin_databases = self._settings.value("lin_databases", [])
        buses = lin_databases[::2]
        dbs = lin_databases[1::2]

        databases["LIN"] = list(zip(buses, dbs))

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

        self.restore_export_setttings()
        self.connect_export_updates()

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
        icon.addPixmap(QtGui.QPixmap(":/scramble.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit("Scrambling measurements")

        # scrambling self.mdf
        for i, source_file in enumerate(source_files):
            progress.signals.setLabelText.emit(f"Scrambling file {i+1} of {count}\n{source_file}")

            result = mdf_module.MDF.scramble(name=source_file, progress=progress)
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
        self._progress.qfinished.connect(self.scramble_finished)

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
                database_files["CAN"].append((widget.database.text(), widget.bus.currentIndex()))

        count2 = self.lin_database_list.count()
        if count2:
            database_files["LIN"] = []
            for i in range(count2):
                item = self.lin_database_list.item(i)
                widget = self.lin_database_list.itemWidget(item)
                database_files["LIN"].append((widget.database.text(), widget.bus.currentIndex()))

        compression = self.extract_bus_compression.currentIndex()

        count = self.files_list.count()

        if not count or not (count1 + count2):
            return

        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        self._progress = setup_progress(parent=self)
        self._progress.qfinished.connect(self.extract_bus_logging_finished)

        self._progress.run_thread_with_progress(
            target=self.extract_bus_logging_thread,
            args=(source_files, database_files, count, compression, version),
            kwargs={},
        )

    def extract_bus_logging_thread(self, source_files, database_files, count, compression, version, progress):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/down.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit("Extract Bus logging from measurements")
        progress.signals.setLabelText.emit(f'Extracting Bus logging from "{count}" files')

        files = self._prepare_files(list(source_files), progress)

        message = []

        progress.signals.setValue.emit(0)
        progress.signals.setMaximum.emit(count)

        for i, (file, source_file) in enumerate(zip(files, source_files)):
            progress.signals.setLabelText.emit(f"Extracting Bus logging from file {i+1} of {count}\n{source_file}")

            if not isinstance(file, mdf_module.MDF):
                mdf = mdf_module.MDF(file)
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
                    message.append(f'- {call_info["unknown_id_count"]} unknown IDs in the MDF4 file')
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
                            message.append(f"- 0x{msg_id:X} --> {msg_name} in <{dbc_name}>")
                        except:
                            pgn, sa = msg_id
                            message.append(f"- PGN=0x{pgn:X} SA=0x{sa:X} --> {msg_name} in <{dbc_name}>")

                message += [
                    "",
                    "The following Bus IDs were in the MDF log file, but not matched in the DBC:",
                ]
                unknown_standard_can = sorted([e for e in call_info["unknown_ids"] if isinstance(e, int)])
                unknown_j1939 = sorted([e for e in call_info["unknown_ids"] if not isinstance(e, int)])
                for msg_id in unknown_standard_can:
                    message.append(f"- 0x{msg_id:X}")

                for pgn, sa in unknown_j1939:
                    message.append(f"- PGN=0x{pgn:X} SA=0x{sa:X}")

            file_name = source_file.with_suffix(".bus_logging.mdf" if version < "4.00" else ".bus_logging.mf4")

            # then save it
            progress.signals.setLabelText.emit(f'Saving extracted Bus logging file {i+1} to "{file_name}"')

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
                database_files["CAN"].append((widget.database.text(), widget.bus.currentIndex()))

        count2 = self.lin_database_list.count()
        if count2:
            database_files["LIN"] = []
            for i in range(count2):
                item = self.lin_database_list.item(i)
                widget = self.lin_database_list.itemWidget(item)
                database_files["LIN"].append((widget.database.text(), widget.bus.currentIndex()))

        single_time_base = self.single_time_base_bus.checkState() == QtCore.Qt.CheckState.Checked
        time_from_zero = self.time_from_zero_bus.checkState() == QtCore.Qt.CheckState.Checked
        empty_channels = self.empty_channels_bus.currentText()
        raster = self.export_raster_bus.value() or None
        time_as_date = self.bus_time_as_date.checkState() == QtCore.Qt.CheckState.Checked
        delimiter = self.delimiter_bus.text() or ","
        doublequote = self.doublequote_bus.checkState() == QtCore.Qt.CheckState.Checked
        escapechar = self.escapechar_bus.text() or None
        lineterminator = self.lineterminator_bus.text().replace("\\r", "\r").replace("\\n", "\n")
        quotechar = self.quotechar_bus.text() or '"'
        quoting = self.quoting_bus.currentText()
        add_units = self.add_units_bus.checkState() == QtCore.Qt.CheckState.Checked

        count = self.files_list.count()

        if not count or not (count1 + count2):
            return

        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        self._progress = setup_progress(parent=self)
        self._progress.qfinished.connect(self.extract_bus_csv_logging_finished)

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
        icon.addPixmap(QtGui.QPixmap(":/csv.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit("Extract Bus logging from measurements to CSV")
        progress.signals.setLabelText.emit(f'Extracting Bus logging from "{count}" files')

        files = self._prepare_files(list(source_files), progress)

        message = []

        for i, (file, source_file) in enumerate(zip(files, source_files)):
            progress.signals.setLabelText.emit(f"Extracting Bus logging from file {i+1} of {count}")

            if not isinstance(file, mdf_module.MDF):
                mdf = mdf_module.MDF(file)
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
                    message.append(f'- {call_info["unknown_id_count"]} unknown IDs in the MDF4 file')
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
            progress.signals.setLabelText.emit(f'Saving extracted Bus logging file {i+1} to "{file_name}"')

            mdf_.configure(
                integer_interpolation=self.integer_interpolation,
                float_interpolation=self.float_interpolation,
            )

            result = mdf_.export(
                fmt="csv",
                filename=file_name,
                single_time_base=single_time_base,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                raster=raster or None,
                time_as_date=time_as_date,
                ignore_value2text_conversions=self.ignore_value2text_conversions,
                delimiter=delimiter,
                doublequote=doublequote,
                escapechar=escapechar,
                lineterminator=lineterminator,
                quotechar=quotechar,
                quoting=quoting,
                add_units=add_units,
                progress=progress,
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
            file_names = [name for name in file_names if Path(name).suffix.lower() in (".arxml", ".dbc")]

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
            file_names = [name for name in file_names if Path(name).suffix.lower() in (".arxml", ".dbc", ".ldf")]

        if file_names:
            for database in file_names:
                item = QtWidgets.QListWidgetItem()
                widget = DatabaseItem(database, bus_type="LIN")

                self.lin_database_list.addItem(item)
                self.lin_database_list.setItemWidget(item, widget)
                item.setSizeHint(widget.sizeHint())

    def concatenate_finished(self):
        self._progress = None

    def concatenate(self, event=None):
        count = self.files_list.count()

        if not count:
            return

        version = self.concatenate_format.currentText()

        sync = self.concatenate_sync.checkState() == QtCore.Qt.CheckState.Checked
        add_samples_origin = self.concatenate_add_samples_origin.checkState() == QtCore.Qt.CheckState.Checked

        split = self.concatenate_split.checkState() == QtCore.Qt.CheckState.Checked
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
                "MDF version 4 files (*.mf4 *.mf4z);;All files (*.*)",
                "MDF version 4 files (*.mf4 *.mf4z)",
            )

        if not output_file_name:
            return

        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        self._progress = setup_progress(parent=self, autoclose=False)
        self._progress.qfinished.connect(self.concatenate_finished)

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
        icon.addPixmap(QtGui.QPixmap(":/plus.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit(f"Concatenating files and saving to {version} format")

        output_file_name = Path(output_file_name)

        files = self._prepare_files(source_files, progress)

        result = mdf_module.MDF.concatenate(
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
        icon.addPixmap(QtGui.QPixmap(":/stack.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        progress.signals.setWindowIcon.emit(icon)
        progress.signals.setWindowTitle.emit(f"Stacking files and saving to {version} format")

        output_file_name = Path(output_file_name)

        files = self._prepare_files(source_files, progress)

        result = mdf_module.MDF.stack(
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

        sync = self.stack_sync.checkState() == QtCore.Qt.CheckState.Checked
        add_samples_origin = self.stack_add_samples_origin.checkState() == QtCore.Qt.CheckState.Checked

        split = self.stack_split.checkState() == QtCore.Qt.CheckState.Checked
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
                "MDF version 4 files (*.mf4 *.mf4z);;All files (*.*)",
                "MDF version 4 files (*.mf4 *.mf4z)",
            )

        if not output_file_name:
            return

        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        self._progress = setup_progress(parent=self, autoclose=False)
        self._progress.qfinished.connect(self.stack_finished)

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

        if suffix in (".erg", ".bsig", ".dl3", ".tdms"):
            try:
                from mfile import BSIG, DL3, ERG, TDMS
            except ImportError:
                print(format_exc())
                from cmerg import BSIG, ERG

            if suffix == ".erg":
                cls = ERG
            elif suffix == ".bsig":
                cls = BSIG
            elif suffix == ".tdms":
                cls = TDMS
            else:
                cls = DL3

            mdf = cls(file_name).export_mdf()
            mdf.original_name = file_name

        elif suffix in (".mdf", ".mf4", ".mf4z"):
            mdf = mdf_module.MDF(file_name)

        else:
            raise ValueError(f"Incompatible suffix '{suffix}'")

        return mdf

    def _prepare_files(self, files=None, progress=None):
        if files is None:
            files = [Path(self.files_list.item(row).text()) for row in range(self.files_list.count())]

        count = len(files)
        progress.signals.setMaximum.emit(count)
        progress.signals.setValue.emit(0)
        progress.signals.setWindowTitle.emit("Preparing measurements")

        mdf_files = []
        for i, file_name in enumerate(files):
            progress.signals.setLabelText.emit(f"Preparing the file {i+1} of {count}\n{file_name}")
            try:
                mdf = self._as_mdf(file_name)
            except:
                print(format_exc())
                mdf = None
            mdf_files.append(mdf)
            progress.signals.setValue.emit(i + 1)

        return mdf_files

    def _get_filtered_channels(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

        channels = []
        count = 0
        total = 0

        if self.filter_view.currentText() == "Internal file structure":
            while item := iterator.value():

                group, index = item.entry
                if index != 0xFFFFFFFFFFFFFFFF:
                    total += 1

                if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                    if index != 0xFFFFFFFFFFFFFFFF:
                        channels.append((item.name, group, index))
                        count += 1

                iterator += 1
        else:
            while item := iterator.value():

                if item.checkState(0) == QtCore.Qt.CheckState.Checked:
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

        with mdf_module.MDF(self.files_list.item(0).text()) as mdf:
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

    def filter_changed(self, item, column=0):
        name = item.text(0)
        if item.checkState(0) == QtCore.Qt.CheckState.Checked:
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

                    while item := iterator.value():
                        if item.parent() is None:
                            iterator += 1
                            dg_cntr += 1
                            ch_cntr = 0
                            continue

                        if (dg_cntr, ch_cntr) in result:
                            item.setCheckState(0, QtCore.Qt.CheckState.Checked)

                        iterator += 1
                        ch_cntr += 1

                elif view.currentText() == "Selected channels only":
                    iterator = QtWidgets.QTreeWidgetItemIterator(widget)

                    signals = set()
                    while item := iterator.value():

                        if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                            signals.add(item.entry)

                        iterator += 1

                    signals = signals | set(result)

                    widget.clear()
                    self._selected_filter.clear()

                    uuid = os.urandom(6).hex()

                    items = []
                    for entry in signals:
                        gp_index, ch_index = entry
                        ch = mdf.groups[gp_index].channels[ch_index]
                        channel = MinimalTreeItem(entry, ch.name, strings=[ch.name], origin_uuid=uuid)
                        channel.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        items.append(channel)
                        self._selected_filter.add(ch.name)

                    if len(items) < 30000:
                        items = natsorted(items, key=lambda x: x.name)
                    else:
                        items.sort(key=lambda x: x.name)

                    widget.addTopLevelItems(items)
                    self.update_selected_filter_channels()

                else:
                    iterator = QtWidgets.QTreeWidgetItemIterator(widget)
                    while item := iterator.value():
                        if item.entry in result:
                            item.setCheckState(0, QtCore.Qt.CheckState.Checked)

                        iterator += 1
        except:
            print(format_exc())
        finally:
            mdf.close()

    def update_channel_tree(self, *args):
        if self.filter_view.currentIndex() == -1 or self._ignore:
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
                    while item := iterator.value():
                        if item.entry[1] != 0xFFFFFFFFFFFFFFFF:
                            if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                                signals.add(item.entry)

                        iterator += 1
                else:
                    while item := iterator.value():

                        if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                            signals.add(item.entry)

                        iterator += 1

                widget.clear()
                widget.mode = view.currentText()

                if widget.mode == "Natural sort":
                    items = []
                    for i, group in enumerate(mdf.groups):
                        for j, ch in enumerate(group.channels):
                            entry = i, j

                            channel = MinimalTreeItem(entry, ch.name, strings=[ch.name], origin_uuid=uuid)
                            channel.setToolTip(0, f"{ch.name} @ group {i}, index {j}")

                            if entry in signals:
                                channel.setCheckState(0, QtCore.Qt.CheckState.Checked)
                            else:
                                channel.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

                            items.append(channel)

                    if len(items) < 30000:
                        items = natsorted(items, key=lambda x: x.name)
                    else:
                        items.sort(key=lambda x: x.name)
                    widget.addTopLevelItems(items)

                elif widget.mode == "Internal file structure":
                    items = []

                    for i, group in enumerate(mdf.groups):
                        entry = i, 0xFFFFFFFFFFFFFFFF

                        channel_group = MinimalTreeItem(entry, origin_uuid=uuid)

                        comment = extract_xml_comment(group.channel_group.comment)

                        if mdf.version >= "4.00" and group.channel_group.acq_source:
                            source = group.channel_group.acq_source
                            if source.bus_type == BUS_TYPE_CAN:
                                ico = ":/bus_can.png"
                            elif source.bus_type == BUS_TYPE_LIN:
                                ico = ":/bus_lin.png"
                            elif source.bus_type == BUS_TYPE_ETHERNET:
                                ico = ":/bus_eth.png"
                            elif source.bus_type == BUS_TYPE_USB:
                                ico = ":/bus_usb.png"
                            elif source.bus_type == BUS_TYPE_FLEXRAY:
                                ico = ":/bus_flx.png"
                            else:
                                ico = None

                            if ico is not None:
                                icon = QtGui.QIcon()
                                icon.addPixmap(QtGui.QPixmap(ico), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

                                channel_group.setIcon(0, icon)

                        acq_name = getattr(group.channel_group, "acq_name", "")
                        if acq_name:
                            base_name = f"CG {i} {acq_name}"
                        else:
                            base_name = f"CG {i}"
                        if comment and acq_name != comment:
                            name = base_name + f" ({comment})"
                        else:
                            name = base_name

                        channel_group.setText(0, name)
                        channel_group.setFlags(
                            channel_group.flags()
                            | QtCore.Qt.ItemFlag.ItemIsAutoTristate
                            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                        )

                        if group.channel_group.cycles_nr:
                            channel_group.setForeground(0, QtGui.QBrush(QtGui.QColor(GREEN)))
                        items.append(channel_group)

                        channels = [HelperChannel(name=ch.name, entry=(i, j)) for j, ch in enumerate(group.channels)]

                        add_children(
                            channel_group,
                            channels,
                            group.channel_dependencies,
                            signals,
                            entries=None,
                            origin_uuid=uuid,
                            version=mdf.version,
                        )

                    widget.addTopLevelItems(items)

                else:
                    items = []
                    for entry in signals:
                        gp_index, ch_index = entry
                        ch = mdf.groups[gp_index].channels[ch_index]
                        channel = MinimalTreeItem(entry, ch.name, strings=[ch.name], origin_uuid=uuid)
                        channel.setToolTip(0, f"{ch.name} @ group {gp_index}, index {ch_index}")
                        channel.setCheckState(0, QtCore.Qt.CheckState.Checked)
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
            "cut_time_from_zero": self.cut_time_from_zero.checkState() == QtCore.Qt.CheckState.Checked,
            "whence": int(self.whence.checkState() == QtCore.Qt.CheckState.Checked),
            "needs_resample": self.resample_group.isChecked(),
            "raster_type_step": self.raster_type_step.isChecked(),
            "raster_type_channel": self.raster_type_channel.isChecked(),
            "raster": self.raster.value(),
            "raster_channel": self.raster_channel.currentText(),
            "resample_time_from_zero": self.resample_time_from_zero.checkState() == QtCore.Qt.CheckState.Checked,
            "output_format": self.output_format.currentText(),
        }

        output_format = self.output_format.currentText()

        if output_format == "MDF":
            new = {
                "mdf_version": self.mdf_version.currentText(),
                "mdf_compression": self.mdf_compression.currentIndex(),
                "mdf_split": self.mdf_split.checkState() == QtCore.Qt.CheckState.Checked,
                "mdf_split_size": self.mdf_split_size.value() * 1024 * 1024,
            }

        elif output_format == "MAT":
            new = {
                "single_time_base": self.single_time_base_mat.checkState() == QtCore.Qt.CheckState.Checked,
                "time_from_zero": self.time_from_zero_mat.checkState() == QtCore.Qt.CheckState.Checked,
                "time_as_date": self.time_as_date_mat.checkState() == QtCore.Qt.CheckState.Checked,
                "use_display_names": self.use_display_names_mat.checkState() == QtCore.Qt.CheckState.Checked,
                "reduce_memory_usage": self.reduce_memory_usage_mat.checkState() == QtCore.Qt.CheckState.Checked,
                "compression": self.export_compression_mat.currentText() == "enabled",
                "empty_channels": self.empty_channels_mat.currentText(),
                "mat_format": self.mat_format.currentText(),
                "oned_as": self.oned_as.currentText(),
                "raw": self.raw_mat.checkState() == QtCore.Qt.CheckState.Checked,
            }

        elif output_format == "CSV":
            new = {
                "single_time_base": self.single_time_base_csv.checkState() == QtCore.Qt.CheckState.Checked,
                "time_from_zero": self.time_from_zero_csv.checkState() == QtCore.Qt.CheckState.Checked,
                "time_as_date": self.time_as_date_csv.checkState() == QtCore.Qt.CheckState.Checked,
                "use_display_names": self.use_display_names_csv.checkState() == QtCore.Qt.CheckState.Checked,
                "reduce_memory_usage": False,
                "compression": False,
                "empty_channels": self.empty_channels_csv.currentText(),
                "raw": self.raw_csv.checkState() == QtCore.Qt.CheckState.Checked,
                "delimiter": self.delimiter.text() or ",",
                "doublequote": self.doublequote.checkState() == QtCore.Qt.CheckState.Checked,
                "escapechar": self.escapechar.text() or None,
                "lineterminator": self.lineterminator.text().replace("\\r", "\r").replace("\\n", "\n"),
                "quotechar": self.quotechar.text() or '"',
                "quoting": self.quoting.currentText(),
                "mat_format": None,
                "oned_as": None,
            }

        else:
            new = {
                "single_time_base": self.single_time_base.checkState() == QtCore.Qt.CheckState.Checked,
                "time_from_zero": self.time_from_zero.checkState() == QtCore.Qt.CheckState.Checked,
                "time_as_date": self.time_as_date.checkState() == QtCore.Qt.CheckState.Checked,
                "use_display_names": self.use_display_names.checkState() == QtCore.Qt.CheckState.Checked,
                "reduce_memory_usage": self.reduce_memory_usage.checkState() == QtCore.Qt.CheckState.Checked,
                "compression": self.export_compression.currentText(),
                "empty_channels": self.empty_channels.currentText(),
                "mat_format": None,
                "oned_as": None,
                "raw": self.raw.checkState() == QtCore.Qt.CheckState.Checked,
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
                from h5py import File as HDF5  # noqa: F401
            except ImportError:
                MessageBox.critical(
                    self,
                    "export_batch to HDF5 unavailale",
                    "h5py package not found; export to HDF5 is unavailable",
                )
                return

        elif output_format == "MAT":
            if opts.mat_format == "7.3":
                try:
                    from hdf5storage import savemat
                except ImportError:
                    MessageBox.critical(
                        self,
                        "export_batch to mat v7.3 unavailale",
                        "hdf5storage package not found; export to mat 7.3 is unavailable",
                    )
                    return
            else:
                try:
                    from scipy.io import savemat  # noqa: F401
                except ImportError:
                    MessageBox.critical(
                        self,
                        "export_batch to mat v4 and v5 unavailale",
                        "scipy package not found; export to mat is unavailable",
                    )
                    return

        elif output_format == "Parquet":
            try:
                from fastparquet import write as write_parquet  # noqa: F401
            except ImportError:
                MessageBox.critical(
                    self,
                    "export_batch to parquet unavailale",
                    "fastparquet package not found; export to parquet is unavailable",
                )
                return

        self._progress = setup_progress(parent=self, autoclose=False)
        self._progress.qfinished.connect(self.apply_processing_finished)

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
            from h5py import File as HDF5  # noqa: F401

        elif output_format == "MAT":
            suffix = ".mat"
            if opts.mat_format == "7.3":
                from hdf5storage import savemat
            else:
                from scipy.io import savemat  # noqa: F401

        elif output_format == "Parquet":
            suffix = ".parquet"
            from fastparquet import write as write_parquet  # noqa: F401

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
            root = Path(os.path.commonpath(source_files))

        except ValueError:
            root = None

        split_size = opts.mdf_split_size if output_format == "MDF" else 0

        integer_interpolation = self.integer_interpolation
        float_interpolation = self.float_interpolation

        files = self._prepare_files(list(source_files), progress)

        for mdf_index, (mdf_file, source_file) in enumerate(zip(files, source_files)):
            if mdf_file is None:
                continue

            mdf_file.configure(
                read_fragment_size=split_size,
                integer_interpolation=self.integer_interpolation,
                float_interpolation=self.float_interpolation,
            )

            mdf = mdf_file

            if needs_filter:
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(":/filter.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit(f"Filtering measurement {mdf_index + 1} of {count}")
                progress.signals.setLabelText.emit(f'Filtering selected channels from\n"{source_file}"')

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
                icon.addPixmap(QtGui.QPixmap(":/cut.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit(f"Cutting measurement {mdf_index+1} of {count}")
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
                icon.addPixmap(QtGui.QPixmap(":/resample.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit(f"Resampling measurement {mdf_index+1} of {count}")
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
                        QtGui.QIcon.Mode.Normal,
                        QtGui.QIcon.State.Off,
                    )
                    progress.signals.setWindowIcon.emit(icon)
                    progress.signals.setWindowTitle.emit(f"Converting measurement {mdf_index+1} of {count}")
                    progress.signals.setLabelText.emit(f'Converting "{source_file}" from {mdf.version} to {version}')

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
                        file_name = output_folder / Path(mdf_file.original_name).relative_to(root)

                    if not file_name.parent.exists():
                        os.makedirs(file_name.parent, exist_ok=True)
                else:
                    file_name = Path(mdf_file.original_name)
                    file_name = file_name.parent / (file_name.stem + ".modified" + suffix)

                file_name = file_name.with_suffix(suffix)

                # then save it
                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(":/save.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit(f"Saving measurement {mdf_index+1} of {count}")
                progress.signals.setLabelText.emit(f"Saving output file {mdf_index+1} of {count}\n{source_file}")

                result = mdf.save(
                    dst=file_name,
                    compression=opts.mdf_compression,
                    overwrite=True,
                    progress=progress,
                )

                if result is TERMINATED:
                    return

            else:
                if output_folder is not None:
                    if root is None:
                        file_name = output_folder / Path(mdf_file.original_name).name
                    else:
                        file_name = output_folder / Path(mdf_file.original_name).relative_to(root)

                    if not file_name.parent.exists():
                        os.makedirs(file_name.parent, exist_ok=True)
                else:
                    file_name = Path(mdf_file.original_name)

                file_name = file_name.with_suffix(suffix)

                icon = QtGui.QIcon()
                icon.addPixmap(QtGui.QPixmap(":/export.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
                progress.signals.setWindowIcon.emit(icon)
                progress.signals.setWindowTitle.emit(f"export_batch measurement {mdf_index+1} of {count}")
                progress.signals.setLabelText.emit(
                    f"export_batching measurement {mdf_index+1} of {count} to {output_format} (be patient this might take a while)\n{source_file}"
                )

                delimiter = self.delimiter.text() or ","
                doublequote = self.doublequote.checkState() == QtCore.Qt.CheckState.Checked
                escapechar = self.escapechar.text() or None
                lineterminator = self.lineterminator.text().replace("\\r", "\r").replace("\\n", "\n")
                quotechar = self.quotechar.text() or '"'
                quoting = self.quoting.currentText()
                add_units = self.add_units.checkState() == QtCore.Qt.CheckState.Checked

                target = mdf.export
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
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", "")
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

        source_files = natsorted([self.files_list.item(row).text() for row in range(count)])

        self.files_list.clear()
        self.files_list.addItems(source_files)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/file.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        for row in range(count):
            self.files_list.item(row).setIcon(icon)

    def sort_by_start_time(self, event=None):
        count = self.files_list.count()

        if not count:
            return

        source_files = [self.files_list.item(row).text() for row in range(count)]

        start_times = []

        for file_name in source_files:
            if Path(file_name).suffix.lower() in (".mdf", ".dat", ".mf4"):
                with open(file_name, "rb") as f:
                    f.seek(64)
                    blk_id = f.read(2)
                    block_type = HeaderBlockV4 if blk_id == b"##" else HeaderBlockV3
                    header = block_type(stream=f, address=64)
                    start_times.append((header.start_time, file_name))

            else:
                mdf = self._as_mdf(file_name)
                header = mdf.header
                start_times.append((header.start_time, file_name))

                mdf.close()

        try:
            start_times = sorted(start_times)
        except TypeError:
            start_times = [(st.replace(tzinfo=timezone.utc), name) for (st, name) in start_times]
            start_times = sorted(start_times)

        self.files_list.clear()
        self.files_list.addItems([item[1] for item in start_times])

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/file.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
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
                while item := iterator.value():
                    iterator += 1

                    if item.parent() is None:
                        continue

                    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                        signals.append(item.text(0))
            else:
                while item := iterator.value():
                    iterator += 1

                    if item.checkState(0) == QtCore.Qt.CheckState.Checked:
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
                    output.write(
                        """[SETTINGS]
Version;V1.1
MultiRasterSeparator;&

"""
                    )
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

                    channels = [name.split(";")[0] for name in channels]

            else:
                channels = load_channel_names_from_file(file_name)

        else:
            info = file_name
            channels = info.get("selected_channels", [])

        if channels:
            iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

            if self.filter_view.currentText() == "Internal file structure":
                while item := iterator.value():
                    if item.parent() is None:
                        iterator += 1
                        continue

                    channel_name = item.text(0)
                    if channel_name in channels:
                        item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        channels.pop(channels.index(channel_name))
                    else:
                        item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

                    iterator += 1

            elif self.filter_view.currentText() == "Natural sort":
                while item := iterator.value():

                    channel_name = item.text(0)
                    if channel_name in channels:
                        item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        channels.pop(channels.index(channel_name))
                    else:
                        item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

                    iterator += 1

            else:
                items = []
                self.filter_tree.clear()

                source_files = [Path(self.files_list.item(row).text()) for row in range(self.files_list.count())]

                for file_name in source_files:
                    if file_name.suffix.lower() in (".mdf", ".mf4"):
                        break
                else:
                    file_name = source_files[0]

                mdf = self._as_mdf(file_name)
                origin_uuid = os.urandom(6).hex()

                for i, gp in enumerate(mdf.groups):
                    for j, ch in enumerate(gp.channels):
                        if ch.name in channels:
                            entry = i, j
                            channel = TreeItem(entry, ch.name, origin_uuid=origin_uuid)
                            channel.setText(0, ch.name)
                            channel.setCheckState(0, QtCore.Qt.CheckState.Checked)
                            items.append(channel)

                            channels.pop(channels.index(ch.name))

                if len(items) < 30000:
                    items = natsorted(items, key=lambda x: x.name)
                else:
                    items.sort(key=lambda x: x.name)
                self.filter_tree.addTopLevelItems(items)

    def connect_export_updates(self):
        self.output_format.currentTextChanged.connect(self.store_export_setttings)

        self.mdf_version.currentTextChanged.connect(self.store_export_setttings)
        self.mdf_compression.currentTextChanged.connect(self.store_export_setttings)
        self.mdf_split.stateChanged.connect(self.store_export_setttings)
        self.mdf_split_size.valueChanged.connect(self.store_export_setttings)

        self.single_time_base.stateChanged.connect(self.store_export_setttings)
        self.time_from_zero.stateChanged.connect(self.store_export_setttings)
        self.time_as_date.stateChanged.connect(self.store_export_setttings)
        self.raw.stateChanged.connect(self.store_export_setttings)
        self.use_display_names.stateChanged.connect(self.store_export_setttings)
        self.reduce_memory_usage.stateChanged.connect(self.store_export_setttings)
        self.export_compression.currentTextChanged.connect(self.store_export_setttings)
        self.empty_channels.currentTextChanged.connect(self.store_export_setttings)

        self.single_time_base_csv.stateChanged.connect(self.store_export_setttings)
        self.time_from_zero_csv.stateChanged.connect(self.store_export_setttings)
        self.time_as_date_csv.stateChanged.connect(self.store_export_setttings)
        self.raw_csv.stateChanged.connect(self.store_export_setttings)
        self.add_units.stateChanged.connect(self.store_export_setttings)
        self.doublequote.stateChanged.connect(self.store_export_setttings)
        self.use_display_names_csv.stateChanged.connect(self.store_export_setttings)
        self.empty_channels_csv.currentTextChanged.connect(self.store_export_setttings)
        self.delimiter.editingFinished.connect(self.store_export_setttings)
        self.escapechar.editingFinished.connect(self.store_export_setttings)
        self.lineterminator.editingFinished.connect(self.store_export_setttings)
        self.quotechar.editingFinished.connect(self.store_export_setttings)
        self.quoting.currentTextChanged.connect(self.store_export_setttings)

        self.single_time_base_mat.stateChanged.connect(self.store_export_setttings)
        self.time_from_zero_mat.stateChanged.connect(self.store_export_setttings)
        self.time_as_date_mat.stateChanged.connect(self.store_export_setttings)
        self.raw_mat.stateChanged.connect(self.store_export_setttings)
        self.use_display_names_mat.stateChanged.connect(self.store_export_setttings)
        self.reduce_memory_usage_mat.stateChanged.connect(self.store_export_setttings)
        self.empty_channels_mat.currentTextChanged.connect(self.store_export_setttings)
        self.export_compression_mat.currentTextChanged.connect(self.store_export_setttings)
        self.mat_format.currentTextChanged.connect(self.store_export_setttings)
        self.oned_as.currentTextChanged.connect(self.store_export_setttings)

    def restore_export_setttings(self):
        self.output_format.setCurrentText(self._settings.value("export_batch", "MDF"))

        self.mdf_version.setCurrentText(self._settings.setValue("export_batch/MDF/version", "4.10"))
        self.mdf_compression.setCurrentText(self._settings.value("export_batch/MDF/compression", "transposed deflate"))
        self.mdf_split.setChecked(self._settings.value("export_batch/MDF/split_data_blocks", True, type=bool))
        self.mdf_split_size.setValue(self._settings.value("export_batch/MDF/split_size", 4, type=int))

        self.single_time_base.setChecked(self._settings.value("export_batch/HDF5/single_time_base", False, type=bool))
        self.time_from_zero.setChecked(self._settings.value("export_batch/HDF5/time_from_zero", False, type=bool))
        self.time_as_date.setChecked(self._settings.value("export_batch/HDF5/time_as_date", False, type=bool))
        self.raw.setChecked(self._settings.value("export_batch/HDF5/raw", False, type=bool))
        self.use_display_names.setChecked(self._settings.value("export_batch/HDF5/use_display_names", False, type=bool))
        self.reduce_memory_usage.setChecked(
            self._settings.value("export_batch/HDF5/reduce_memory_usage", False, type=bool)
        )
        self.export_compression.setCurrentText(self._settings.value("export_batch/HDF5/export_compression", "gzip"))
        self.empty_channels.setCurrentText(self._settings.value("export_batch/HDF5/empty_channels", "skip"))

        self.single_time_base_csv.setChecked(
            self._settings.value("export_batch/CSV/single_time_base_csv", False, type=bool)
        )
        self.time_from_zero_csv.setChecked(
            self._settings.value("export_batch/CSV/time_from_zero_csv", False, type=bool)
        )
        self.time_as_date_csv.setChecked(self._settings.value("export_batch/CSV/time_as_date_csv", False, type=bool))
        self.raw_csv.setChecked(self._settings.value("export_batch/CSV/raw_csv", False, type=bool))
        self.add_units.setChecked(self._settings.value("export_batch/CSV/add_units", False, type=bool))
        self.doublequote.setChecked(self._settings.value("export_batch/CSV/doublequote", False, type=bool))
        self.use_display_names_csv.setChecked(
            self._settings.value("export_batch/CSV/use_display_names_csv", False, type=bool)
        )
        self.empty_channels_csv.setCurrentText(self._settings.value("export_batch/CSV/empty_channels_csv", "skip"))
        self.delimiter.setText(self._settings.value("export_batch/CSV/delimiter", ","))
        self.escapechar.setText(self._settings.value("export_batch/CSV/escapechar", ""))
        self.lineterminator.setText(self._settings.value("export_batch/CSV/lineterminator", r"\r\n"))
        self.quotechar.setText(self._settings.value("export_batch/CSV/quotechar", '"'))
        self.quoting.setCurrentText(self._settings.value("export_batch/CSV/quoting", "MINIMAL"))

        self.single_time_base_mat.setChecked(
            self._settings.value("export_batch/MAT/single_time_base_mat", False, type=bool)
        )
        self.time_from_zero_mat.setChecked(
            self._settings.value("export_batch/MAT/time_from_zero_mat", False, type=bool)
        )
        self.time_as_date_mat.setChecked(self._settings.value("export_batch/MAT/time_as_date_mat", False, type=bool))
        self.raw_mat.setChecked(self._settings.value("export_batch/MAT/raw_mat", False, type=bool))
        self.use_display_names_mat.setChecked(
            self._settings.value("export_batch/MAT/use_display_names_mat", False, type=bool)
        )
        self.reduce_memory_usage_mat.setChecked(
            self._settings.value("export_batch/MAT/reduce_memory_usage_mat", False, type=bool)
        )
        self.empty_channels_mat.setCurrentText(self._settings.value("export_batch/MAT/empty_channels_mat", "skip"))
        self.export_compression_mat.setCurrentText(
            self._settings.value("export_batch/MAT/export_compression_mat", "enabled")
        )
        self.mat_format.setCurrentText(self._settings.value("export_batch/MAT/mat_format", "4"))
        self.oned_as.setCurrentText(self._settings.value("export_batch/MAT/oned_as", "row"))

    def store_export_setttings(self, *args):
        self._settings.setValue("export_batch", self.output_format.currentText())

        self._settings.setValue("export_batch/MDF/version", self.mdf_version.currentText())
        self._settings.setValue("export_batch/MDF/compression", self.mdf_compression.currentText())
        self._settings.setValue("export_batch/MDF/split_data_blocks", self.mdf_split.isChecked())
        self._settings.setValue("export_batch/MDF/split_size", self.mdf_split_size.value())

        self._settings.setValue("export_batch/HDF5/single_time_base", self.single_time_base.isChecked())
        self._settings.setValue("export_batch/HDF5/time_from_zero", self.time_from_zero.isChecked())
        self._settings.setValue("export_batch/HDF5/time_as_date", self.time_as_date.isChecked())
        self._settings.setValue("export_batch/HDF5/raw", self.raw.isChecked())
        self._settings.setValue("export_batch/HDF5/use_display_names", self.use_display_names.isChecked())
        self._settings.setValue("export_batch/HDF5/reduce_memory_usage", self.reduce_memory_usage.isChecked())
        self._settings.setValue("export_batch/HDF5/export_compression", self.export_compression.currentText())
        self._settings.setValue("export_batch/HDF5/empty_channels", self.empty_channels.currentText())

        self._settings.setValue("export_batch/CSV/single_time_base_csv", self.single_time_base_csv.isChecked())
        self._settings.setValue("export_batch/CSV/time_from_zero_csv", self.time_from_zero_csv.isChecked())
        self._settings.setValue("export_batch/CSV/time_as_date_csv", self.time_as_date_csv.isChecked())
        self._settings.setValue("export_batch/CSV/raw_csv", self.raw_csv.isChecked())
        self._settings.setValue("export_batch/CSV/add_units", self.add_units.isChecked())
        self._settings.setValue("export_batch/CSV/doublequote", self.doublequote.isChecked())
        self._settings.setValue("export_batch/CSV/use_display_names_csv", self.use_display_names_csv.isChecked())
        self._settings.setValue("export_batch/CSV/empty_channels_csv", self.empty_channels_csv.currentText())
        self._settings.setValue("export_batch/CSV/delimiter", self.delimiter.text())
        self._settings.setValue("export_batch/CSV/escapechar", self.escapechar.text())
        self._settings.setValue("export_batch/CSV/lineterminator", self.lineterminator.text())
        self._settings.setValue("export_batch/CSV/quotechar", self.quotechar.text())
        self._settings.setValue("export_batch/CSV/quoting", self.quoting.currentText())

        self._settings.setValue("export_batch/MAT/single_time_base_mat", self.single_time_base_mat.isChecked())
        self._settings.setValue("export_batch/MAT/time_from_zero_mat", self.time_from_zero_mat.isChecked())
        self._settings.setValue("export_batch/MAT/time_as_date_mat", self.time_as_date_mat.isChecked())
        self._settings.setValue("export_batch/MAT/raw_mat", self.raw_mat.isChecked())
        self._settings.setValue("export_batch/MAT/use_display_names_mat", self.use_display_names_mat.isChecked())
        self._settings.setValue("export_batch/MAT/reduce_memory_usage_mat", self.reduce_memory_usage_mat.isChecked())
        self._settings.setValue("export_batch/MAT/empty_channels_mat", self.empty_channels_mat.currentText())
        self._settings.setValue("export_batch/MAT/export_compression_mat", self.export_compression_mat.currentText())
        self._settings.setValue("export_batch/MAT/mat_format", self.mat_format.currentText())
        self._settings.setValue("export_batch/MAT/oned_as", self.oned_as.currentText())

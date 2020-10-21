# -*- coding: utf-8 -*-
import os
from pathlib import Path
from functools import partial

import psutil
from PyQt5 import QtCore, QtWidgets, QtGui
from natsort import natsorted

from ...blocks.utils import extract_cncomment_xml
from ...mdf import MDF, SUPPORTED_VERSIONS
from ..ui import resource_rc as resource_rc
from ..ui.batch_widget import Ui_batch_widget
from ..dialogs.advanced_search import AdvancedSearch
from .tree_item import TreeItem
from ..utils import (
    add_children,
    HelperChannel,
    load_dsp,
    load_lab,
    run_thread_with_progress,
    setup_progress,
    TERMINATED,
)
from .list import MinimalListWidget


class BatchWidget(Ui_batch_widget, QtWidgets.QWidget):
    def __init__(self, ignore_value2text_conversions=False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self._settings = QtCore.QSettings()

        self.ignore_value2text_conversions = ignore_value2text_conversions

        self.progress = None
        self.files_list = MinimalListWidget()
        self.splitter.addWidget(self.files_list)

        self.raster_type_channel.toggled.connect(self.set_raster_type)

        for widget in (
            self.concatenate_format,
            self.stack_format,
            self.extract_can_format,
            self.mdf_version,
        ):
            widget.insertItems(0, SUPPORTED_VERSIONS)

        for widget in (
            self.concatenate_split_size,
            self.stack_split_size,
            self.mdf_split_size,
        ):
            widget.setValue(4)

        for widget in (
            self.concatenate_compression,
            self.stack_compression,
            self.extract_can_compression,
            self.mdf_compression,
        ):

            widget.insertItems(0, ("no compression", "deflate", "transposed deflate"))

        self.concatenate_btn.clicked.connect(self.concatenate)
        self.scramble_btn.clicked.connect(self.scramble)
        self.stack_btn.clicked.connect(self.stack)
        self.extract_can_btn.clicked.connect(self.extract_can)
        self.extract_can_csv_btn.clicked.connect(self.extract_can_csv)
        self.advanced_serch_filter_btn.clicked.connect(self.search)
        self.raster_search_btn.clicked.connect(self.raster_search)
        self.apply_btn.clicked.connect(self.apply_processing)
        self.modify_output_folder_btn.clicked.connect(self.change_modify_output_folder)
        self.output_format.currentTextChanged.connect(self.output_format_changed)

        self.filter_view.setCurrentIndex(-1)
        self.filter_view.currentIndexChanged.connect(
            self._update_channel_tree
        )
        self.filter_view.currentTextChanged.connect(
            self._update_channel_tree
        )
        self.filter_view.setCurrentText(
            self._settings.value("filter_view", "Internal file structure")
        )

        self.load_can_database_btn.clicked.connect(self.load_can_database)

        self.empty_channels_can.insertItems(0, ("skip", "zeros"))

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

    def update_progress(self, current_index, max_index):
        self.progress = current_index, max_index

    def scramble(self, event):

        count = self.files_list.count()

        if not count:
            return

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
                message.append("- no unknown IDs inf the MDF4 file")

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
                message.append("- no unknown IDs inf the MDF4 file")

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
        count = self.files_list.count()

        if not count:
            return

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
                "version": version,
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
        count = self.files_list.count()

        if not count:
            return

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
                "version": version,
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
            elif file_name.suffix.lower() == ".dl3":
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
            elif file_name.suffix.lower() in (".mdf", ".mf4"):
                files[i] = MDF(file_name)

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

        mdf = MDF(self.files_list.item(0).text())
        channels = self.mdf.channels_db
        mdf.close()
        dlg = AdvancedSearch(
            channels, show_add_window=False, show_pattern=False, parent=self
        )
        dlg.setModal(True)
        dlg.exec_()
        result = dlg.result
        if result:
            dg_cntr, ch_cntr = next(iter(result))

            name = self.mdf.groups[dg_cntr].channels[ch_cntr].name

            self.raster_channel.setCurrentText(name)

    def filter_changed(self, item, column):
        name = item.text(0)
        if item.checkState(0) == QtCore.Qt.Checked:
            self._selected_filter.add(name)
        else:
            if name in self._selected_filter:
                self._selected_filter.remove(name)
        self._filter_timer.start(10)

    def update_selected_filter_channels(self):
        self.selected_filter_channels.clear()
        self.selected_filter_channels.addItems(sorted(self._selected_filter))

    def search(self, event=None):
        if not self.files_list.count():
            return

        mdf = MDF(self.files_list.item(0).text())
        channels = self.mdf.channels_db
        mdf.close()

        widget = self.filter_tree
        view = self.filter_view

        dlg = AdvancedSearch(
            channels, show_add_window=False, show_pattern=False, parent=self
        )
        dlg.setModal(True)
        dlg.exec_()
        result = dlg.result

        if result:
            names = set()
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
                        names.add(item.text(0))

                    iterator += 1
                    ch_cntr += 1
            elif view.currentText() == "Selected channels only":
                iterator = QtWidgets.QTreeWidgetItemIterator(self.filter_tree)

                signals = set()
                while iterator.value():
                    item = iterator.value()

                    if item.checkState(0) == QtCore.Qt.Checked:
                        signals.add(item.entry)

                    iterator += 1

                signals = signals | result

                widget.clear()

                items = []
                for entry in signals:
                    gp_index, ch_index = entry
                    ch = self.mdf.groups[gp_index].channels[ch_index]
                    channel = TreeItem(entry, ch.name, mdf_uuid=self.uuid)
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
                        names.add(item.text(0))

                    iterator += 1

    def _update_channel_tree(self, index=None):
        if self.filter_view.currentIndex() == -1:
            return

        count = self.files_list.count()
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]
        if not count:
            self.filter_tree.clear()
            return
        else:
            uuid = os.urandom(6).hex()
            with MDF(source_files[0]) as mdf:

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

                            channel = TreeItem(entry, ch.name, mdf_uuid=uuid)
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
                        channel_group = TreeItem(entry, mdf_uuid=uuid)
                        comment = group.channel_group.comment
                        comment = extract_cncomment_xml(comment)

                        if comment:
                            channel_group.setText(0, f"Channel group {i} ({comment})")
                        else:
                            channel_group.setText(0, f"Channel group {i}")
                        channel_group.setFlags(
                            channel_group.flags()
                            | QtCore.Qt.ItemIsTristate
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
                            mdf_uuid=uuid,
                        )
                else:
                    items = []
                    for entry in signals:
                        gp_index, ch_index = entry
                        ch = mdf.groups[gp_index].channels[ch_index]
                        channel = TreeItem(entry, ch.name, mdf_uuid=uuid)
                        channel.setText(0, ch.name)
                        channel.setCheckState(0, QtCore.Qt.Checked)
                        items.append(channel)

                    if len(items) < 30000:
                        items = natsorted(items, key=lambda x: x.name)
                    else:
                        items.sort(key=lambda x: x.name)
                    widget.addTopLevelItems(items)

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

    def apply_processing(self, event):

        count = self.files_list.count()
        progress = setup_progress(
            parent=self,
            title="Preparing measurements",
            message="Preparing measurements",
            icon_name="filter",
        )
        files = self._prepare_files(progress)
        progress.cancel()
        source_files = [Path(self.files_list.item(row).text()) for row in range(count)]

        if not count:
            return

        steps = 1
        if self.cut_group.isChecked():
            steps += 1
        if self.resample_group.isChecked():
            steps += 1
        needs_filter, channels = self._get_filtered_channels()
        if needs_filter:
            steps += 1

        opts = self._current_options()

        output_format = opts.output_format

        if output_format == "MDF":
            version = opts.mdf_version

        if output_format == "HDF5":
            suffix = '.hdf'
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
            suffix = '.mat'
            if opts.mat_format == "7.3":
                try:
                    from hdf5storage import savemat
                except ImportError:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Export to mat unavailale",
                        "hdf5storage package not found; export to mat 7.3 is unavailable",
                    )
                    return
            else:
                try:
                    from scipy.io import savemat
                except ImportError:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Export to mat unavailale",
                        "scipy package not found; export to mat is unavailable",
                    )
                    return

        elif output_format == "Parquet":
            suffix = '.parquet'
            try:
                from fastparquet import write as write_parquet
            except ImportError:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export to parquet unavailale",
                    "fastparquet package not found; export to parquet is unavailable",
                )
                return
        elif output_format == "CSV":
            suffix = '.csv'

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

        for i, (mdf_file, source_file) in enumerate(zip(files, source_files)):
            mdf_file.configure(read_fragment_size=split_size)

            mdf = None
            progress = None

            if needs_filter:

                progress = setup_progress(
                    parent=self,
                    title="Filtering measurement",
                    message=f'Filtering selected channels from "{self.file_name}"',
                    icon_name="filter",
                )

                # filtering self.mdf
                target = mdf_file.filter
                kwargs = {
                    "channels": channels,
                    "version": opts.mdf_version if output_format == "MDF" else "4.10",
                }

                result = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=99,
                    offset=0,
                    progress=progress,
                )

                if result is TERMINATED:
                    progress.cancel()
                    return
                else:
                    mdf = result

                mdf.configure(read_fragment_size=split_size, write_fragment_size=split_size)

            if opts.needs_cut:

                if progress is None:
                    progress = setup_progress(
                        parent=self,
                        title="Cutting measurement",
                        message=f"Cutting from {opts.cut_start}s to {opts.cut_stop}s",
                        icon_name="cut",
                    )
                else:
                    icon = QtGui.QIcon()
                    icon.addPixmap(
                        QtGui.QPixmap(":/cut.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                    )
                    progress.setWindowIcon(icon)
                    progress.setWindowTitle("Cutting measurement")
                    progress.setLabelText(
                        f"Cutting from {opts.cut_start}s to {opts.cut_stop}s"
                    )

                # cut mdf_file
                target = mdf_file.cut if mdf is None else mdf.cut
                kwargs = {
                    "start": opts.cut_start,
                    "stop": opts.cut_stop,
                    "whence": opts.whence,
                    "version": opts.mdf_version if output_format == "MDF" else "4.10",
                    "time_from_zero": opts.cut_time_from_zero,
                }

                result = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=99,
                    offset=0,
                    progress=progress,
                )

                if result is TERMINATED:
                    progress.cancel()
                    return
                else:
                    if mdf is None:
                        mdf = result
                    else:
                        mdf.close()
                        mdf = result

                mdf.configure(read_fragment_size=split_size, write_fragment_size=split_size)

            if opts.needs_resample:

                if opts.raster_type_channel:
                    raster = opts.raster_channel
                    message = f'Resampling using channel "{raster}"'
                else:
                    raster = opts.raster
                    message = f"Resampling to {raster}s raster"

                if progress is None:
                    progress = setup_progress(
                        parent=self,
                        title="Resampling measurement",
                        message=message,
                        icon_name="resample",
                    )
                else:
                    icon = QtGui.QIcon()
                    icon.addPixmap(
                        QtGui.QPixmap(":/resample.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                    )
                    progress.setWindowIcon(icon)
                    progress.setWindowTitle("Resampling measurement")
                    progress.setLabelText(message)

                # resample mdf_file
                target = mdf_file.resample if mdf is None else mdf.resample
                kwargs = {
                    "raster": raster,
                    "version": opts.mdf_version if output_format == "MDF" else "4.10",
                    "time_from_zero": opts.resample_time_from_zero,
                }

                result = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=99,
                    offset=0,
                    progress=progress,
                )

                if result is TERMINATED:
                    progress.cancel()
                    return
                else:
                    if mdf is None:
                        mdf = result
                    else:
                        mdf.close()
                        mdf = result

                mdf.configure(read_fragment_size=split_size, write_fragment_size=split_size)

            if output_format == "MDF":
                if mdf is None:
                    if progress is None:
                        progress = setup_progress(
                            parent=self,
                            title="Converting measurement",
                            message=f'Converting from {mdf_file.version} to {version}',
                            icon_name="convert",
                        )
                    else:
                        icon = QtGui.QIcon()
                        icon.addPixmap(
                            QtGui.QPixmap(":/convert.png"),
                            QtGui.QIcon.Normal,
                            QtGui.QIcon.Off,
                        )
                        progress.setWindowIcon(icon)
                        progress.setWindowTitle("Converting measurement")
                        progress.setLabelText(
                            f'Converting from {mdf_file.version} to {version}'
                        )

                    # convert mdf_file
                    target = mdf_file.convert
                    kwargs = {"version": version}

                    result = run_thread_with_progress(
                        self,
                        target=target,
                        kwargs=kwargs,
                        factor=99,
                        offset=0,
                        progress=progress,
                    )

                    if result is TERMINATED:
                        progress.cancel()
                        return
                    else:
                        mdf = result

                if version >= '4.00':
                    suffix = '.mf4'
                else:
                    suffix = '.mdf'

                mdf.configure(read_fragment_size=split_size, write_fragment_size=split_size)

                if output_folder is not None:
                    if root is None:
                        file_name = output_folder / Path(mdf_file.name).name
                    else:
                        file_name = output_folder / Path(mdf_file.name).relative_to(root)

                    if not file_name.parent.exists():
                        os.makedirs(file_name.parent, exist_ok=True)
                else:
                    file_name = Path(mdf_file.name)
                    file_name = file_name.parent / (file_name.stem + '.modified' + suffix)
                file_name = file_name.with_suffix(suffix)

                # then save it
                progress.setLabelText(f'Saving output file "{file_name}"')

                target = mdf.save
                kwargs = {
                    "dst": file_name,
                    "compression": opts.mdf_compression,
                    "overwrite": False,
                }

                run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=99,
                    offset=0,
                    progress=progress,
                )

                self.progress = None
                progress.cancel()

            else:
                if progress is None:
                    progress = setup_progress(
                        parent=self,
                        title="Export measurement",
                        message=f"Exporting to {output_format}",
                        icon_name="export",
                    )
                else:
                    icon = QtGui.QIcon()
                    icon.addPixmap(
                        QtGui.QPixmap(":/export.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
                    )
                    progress.setWindowIcon(icon)
                    progress.setWindowTitle("Export measurement")
                    progress.setLabelText(f"Exporting to {output_format}")

                if mdf is None:
                    mdf = mdf_file

                if output_folder is not None:
                    if root is None:
                        file_name = output_folder / Path(mdf.name).name
                    else:
                        file_name = output_folder / Path(mdf.name).relative_to(root)
                else:
                    file_name = Path(mdf.name)

                file_name = file_name.with_suffix(suffix)

                if not file_name.parent.exists():
                    os.makedirs(file_name.parent, exist_ok=True)

                target = mdf_file.export if mdf is None else mdf.export
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
                }

                result = run_thread_with_progress(
                    self,
                    target=target,
                    kwargs=kwargs,
                    factor=99,
                    offset=0,
                    progress=progress,
                )

                self.progress = None
                progress.cancel()

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

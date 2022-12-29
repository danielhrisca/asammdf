# -*- coding: utf-8 -*-
import re
from traceback import format_exc

from natsort import natsorted
from PySide6 import QtCore, QtWidgets

from ...blocks.utils import extract_xml_comment
from ..ui import resource_rc
from ..ui.search_dialog import Ui_SearchDialog
from .range_editor import RangeEditor


class AdvancedSearch(Ui_SearchDialog, QtWidgets.QDialog):

    NameColumn = 0
    GroupColumn = 1
    ChannelColumn = 2
    UnitColumn = 3
    SourceNameColumn = 4
    SourcePathColumn = 5
    CommentColumn = 6

    columns = 7

    def __init__(
        self,
        mdf,
        return_names=False,
        show_add_window=False,
        show_apply=False,
        show_pattern=True,
        apply_text="Apply",
        add_window_text="Add window",
        show_search=True,
        window_title="Search & select channels",
        pattern=None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.selection.all_texts = True

        self.result = {}
        self.add_window_request = False
        if mdf:
            self.channels_db = mdf.channels_db
            self.mdf = mdf
        else:
            self.mdf = self.channels_db = None

        self.apply_btn.clicked.connect(self._apply)
        self.add_btn.clicked.connect(self._add)
        self.add_window_btn.clicked.connect(self._add_window)
        self.cancel_btn.clicked.connect(self._cancel)

        self.search_box.editingFinished.connect(self.search_text_changed)
        self.match_kind.currentTextChanged.connect(self.search_box.textChanged.emit)
        self.matches.itemDoubleClicked.connect(self._match_double_clicked)
        self.selection.itemDoubleClicked.connect(self._selection_double_clicked)

        self.matches.header().sectionResized.connect(self.section_resized)
        self.selection.header().sectionResized.connect(self.section_resized)

        self.apply_pattern_btn.clicked.connect(self._apply_pattern)
        self.cancel_pattern_btn.clicked.connect(self._cancel_pattern)
        self.define_ranges_btn.clicked.connect(self._define_ranges)

        self.search_box.setFocus()

        self._return_names = return_names
        self.ranges = []

        self.pattern_window = False

        self.apply_btn.setText(apply_text)
        self.add_window_btn.setText(add_window_text)

        self.selection.can_delete_items = True
        self.matches.can_delete_items = False

        if not show_add_window:
            self.add_window_btn.hide()

        if not show_apply:
            self.apply_btn.hide()

        if not show_pattern:
            self.tabs.removeTab(1)

        if not show_search:
            self.tabs.removeTab(0)

        if pattern:
            self.pattern.setText(pattern["pattern"])
            self.filter_type.setCurrentText(pattern["filter_type"])
            self.filter_value.setValue(pattern["filter_value"])
            self.pattern_match_type.setCurrentText(pattern["match_type"])
            self.raw.setCheckState(
                QtCore.Qt.Checked if pattern["raw"] else QtCore.Qt.Unchecked
            )
            self.name.setText(pattern["name"])
            self.ranges = pattern["ranges"]
            self.integer_format.setCurrentText(pattern.get("integer_format", "phys"))

        self.setWindowTitle(window_title)

        self.matches.setColumnWidth(self.NameColumn, 450)
        self.matches.setColumnWidth(self.GroupColumn, 40)
        self.matches.setColumnWidth(self.ChannelColumn, 40)
        self.matches.setColumnWidth(self.UnitColumn, 40)
        self.matches.setColumnWidth(self.SourceNameColumn, 170)
        self.matches.setColumnWidth(self.SourcePathColumn, 170)

        self.showMaximized()

    def search_text_changed(self):
        text = self.search_box.text().strip()
        extened_search = self.extended_search.checkState() == QtCore.Qt.Checked

        if len(text) >= 2:

            self.matches.setSortingEnabled(False)
            self.matches.clear()

            if self.match_kind.currentText() == "Wildcard":
                pattern = text.replace("*", "_WILDCARD_")
                pattern = re.escape(pattern)
                pattern = pattern.replace("_WILDCARD_", ".*")
            else:
                pattern = text

            try:
                pattern = re.compile(f"(?i){pattern}")

                if extened_search:
                    matches = {}

                    for group_index, group in enumerate(self.mdf.groups):
                        cg_source = group.channel_group.acq_source

                        # check channel group source name

                        if cg_source and (
                            pattern.fullmatch(cg_source.name or "")
                            or pattern.fullmatch(cg_source.path or "")
                        ):
                            matches.update(
                                {
                                    (group_index, channel_index): {
                                        "names": [ch.name],
                                        "comment": extract_xml_comment(
                                            ch.comment
                                        ).strip(),
                                        "unit": ch.conversion
                                        and ch.conversion.unit
                                        or ch.unit,
                                        "source_name": cg_source.name,
                                        "source_path": cg_source.path,
                                    }
                                    for channel_index, ch in enumerate(group.channels)
                                }
                            )

                        else:

                            for channel_index, ch in enumerate(group.channels):

                                entry = group_index, channel_index
                                source = ch.source

                                targets = [
                                    ch.name,
                                    *ch.display_names.values(),
                                ]

                                for target in targets:
                                    if pattern.fullmatch(target):

                                        if entry not in matches:
                                            matches[entry] = {
                                                "names": [target],
                                                "comment": extract_xml_comment(
                                                    ch.comment
                                                ).strip(),
                                                "unit": ch.conversion
                                                and ch.conversion.unit
                                                or ch.unit,
                                                "source_name": source.name
                                                if source
                                                else "",
                                                "source_path": source.path
                                                if source
                                                else "",
                                            }
                                        else:
                                            matches[entry]["name"].append(target)

                                if entry not in matches:
                                    targets = [ch.unit, ch.comment]
                                    if source:
                                        targets.append(source.name)
                                        targets.append(source.path)

                                    for target in targets:
                                        if pattern.fullmatch(target):

                                            matches[entry] = {
                                                "names": [ch.name],
                                                "comment": extract_xml_comment(
                                                    ch.comment
                                                ).strip(),
                                                "unit": ch.conversion
                                                and ch.conversion.unit
                                                or ch.unit,
                                                "source_name": source.name
                                                if source
                                                else "",
                                                "source_path": source.path
                                                if source
                                                else "",
                                            }
                                            break

                else:
                    found_names = [
                        name for name in self.channels_db if pattern.fullmatch(name)
                    ]

                    matches = {}
                    for name in found_names:
                        for entry in self.channels_db[name]:

                            if entry not in matches:
                                (group_index, channel_index) = entry
                                ch = self.mdf.groups[group_index].channels[
                                    channel_index
                                ]
                                cg = self.mdf.groups[group_index].channel_group

                                source = ch.source or cg.acq_source

                                matches[entry] = {
                                    "names": [],
                                    "comment": extract_xml_comment(ch.comment).strip(),
                                    "unit": ch.conversion
                                    and ch.conversion.unit
                                    or ch.unit,
                                    "source_name": source.name if source else "",
                                    "source_path": source.path if source else "",
                                }

                            info = matches[entry]

                            if name == ch.name:
                                info["names"].insert(0, name)
                            else:
                                info["names"].append(name)

                matches = [
                    (group_index, channel_index, info)
                    for (group_index, channel_index), info in matches.items()
                ]
                matches.sort(key=lambda x: x[-1]["names"][0])

                self.matches.clear()
                for group_index, channel_index, info in matches:
                    names = info["names"]
                    group_index, channel_index = str(group_index), str(channel_index)
                    item = QtWidgets.QTreeWidgetItem(
                        [
                            names[0],
                            group_index,
                            channel_index,
                            info["unit"],
                            info["source_name"],
                            info["source_path"],
                            info["comment"],
                        ]
                    )
                    self.matches.addTopLevelItem(item)

                    children = [
                        QtWidgets.QTreeWidgetItem(
                            [
                                name,
                                group_index,
                                channel_index,
                                info["unit"],
                                info["source_name"],
                                info["source_path"],
                                info["comment"],
                            ]
                        )
                        for name in names[1:]
                    ]

                    if children:
                        item.addChildren(children)

                if matches:
                    self.status.setText(f"{self.matches.topLevelItemCount()} results")
                else:
                    self.status.setText("No results")

                self.matches.expandAll()

            except Exception as err:
                print(format_exc())
                self.status.setText(str(err))

            self.matches.setSortingEnabled(True)

    def _add(self, event):
        selection = set()

        iterator = QtWidgets.QTreeWidgetItemIterator(self.selection)
        while True:
            item = iterator.value()

            if item is None:
                break

            data = tuple(item.text(i) for i in range(self.columns))
            selection.add(data)

            iterator += 1

        for item in self.matches.selectedItems():
            data = tuple(item.text(i) for i in range(self.columns))
            selection.add(data)

        selection = natsorted(selection)

        items = [QtWidgets.QTreeWidgetItem(texts) for texts in selection]

        self.selection.setSortingEnabled(False)
        self.selection.clear()
        self.selection.addTopLevelItems(items)
        self.selection.setSortingEnabled(True)

    def _apply(self, event=None):
        if self._return_names:
            self.result = set()

            iterator = QtWidgets.QTreeWidgetItemIterator(self.selection)
            while True:
                item = iterator.value()
                if item is None:
                    break
                self.result.add(item.text(self.NameColumn))
                iterator += 1
        else:
            self.result = {}

            iterator = QtWidgets.QTreeWidgetItemIterator(self.selection)
            while True:
                item = iterator.value()
                if item is None:
                    break

                entry = int(item.text(self.GroupColumn)), int(
                    item.text(self.ChannelColumn)
                )
                name = item.text(self.NameColumn)
                self.result[entry] = name
                iterator += 1

        self.close()

    def _apply_pattern(self, event):
        self.result = {
            "pattern": self.pattern.text().strip(),
            "match_type": self.pattern_match_type.currentText(),
            "filter_type": self.filter_type.currentText(),
            "filter_value": self.filter_value.value(),
            "raw": self.raw.checkState() == QtCore.Qt.Checked,
            "ranges": self.ranges,
            "name": self.name.text().strip(),
            "integer_format": self.integer_format.currentText(),
        }

        if not self.result["pattern"]:
            QtWidgets.QMessageBox.warning(
                self, "Cannot apply pattern", "The pattern cannot be empty"
            )
            return

        if not self.result["name"]:
            QtWidgets.QMessageBox.warning(
                self, "Cannot apply pattern", "The name cannot be empty"
            )
            return

        self.pattern_window = True
        self.close()

    def _add_window(self, event=None):
        self.add_window_request = True
        self._apply()

    def _cancel(self, event):
        self.result = {}
        self.close()

    def _cancel_pattern(self, event):
        self.result = {}
        self.close()

    def _define_ranges(self, event=None):
        name = self.pattern.text().strip()
        dlg = RangeEditor(f"Channel of <{name}>", ranges=self.ranges, parent=self)
        dlg.exec_()
        if dlg.pressed_button == "apply":
            self.ranges = dlg.result

    def _match_double_clicked(self, item):
        selection = set()
        new_item = item

        iterator = QtWidgets.QTreeWidgetItemIterator(self.selection)
        while iterator.value():
            item = iterator.value()
            if item is None:
                break

            data = tuple(item.text(i) for i in range(self.columns))
            selection.add(data)

            iterator += 1

        new_data = tuple(new_item.text(i) for i in range(self.columns))

        if new_data not in selection:
            selection.add(new_data)

            selection = natsorted(selection)

            items = [QtWidgets.QTreeWidgetItem(texts) for texts in selection]

            self.selection.clear()
            self.selection.addTopLevelItems(items)

    def _selection_double_clicked(self, item):
        root = self.selection.invisibleRootItem()
        (item.parent() or root).removeChild(item)

    def section_resized(self, index, old_size, new_size):
        self.selection.setColumnWidth(index, new_size)
        self.matches.setColumnWidth(index, new_size)

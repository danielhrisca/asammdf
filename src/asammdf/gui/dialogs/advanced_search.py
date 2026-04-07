import os
import re
from traceback import format_exc

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ...blocks.utils import extract_xml_comment
from ..ui.search_dialog import Ui_SearchDialog
from .messagebox import MessageBox
from .range_editor import RangeEditor

NameColumn = 0
GroupColumn = 1
ChannelColumn = 2
UnitColumn = 3
SourceNameColumn = 4
SourcePathColumn = 5
CommentColumn = 6


class SearchItem:

    def __init__(self, values=(), parent=None):
        self._parent = parent
        self._children = []
        self._values = values

    def __contains__(self, item):
        for child in self._children:
            if item in child:
                return True
            
        return tuple(self) == tuple(item)

    def appendChild(self, item):
        item.parent = self
        self._children.append(item)

    def child(self, row):
        return self._children[row]
    
    def childCount(self):
        return len(self._children)
    
    def clear(self):
        for child in self._children:
            child.clear()

        self._children.clear()

    def copy(self):
        return SearchItem(tuple(self))
    
    def data(self, return_names=False):
        if return_names:
            res = set()
            for child in self._children:
                res |= child.data(return_names)

            if self.parent:
                res.add(self[NameColumn])
        else:
            res = {}
            for child in self._children:
                res.update(child.data(return_names))

            if self.parent:
                res[(self[GroupColumn], self[ChannelColumn])] = self[NameColumn]

        return res
    
    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self, v):
        self._parent = v

    def remove(self, item):
        self._children = [
            child
            for child in self._children
            if tuple(child) != tuple(item)
        ]
        for child in self._children:
            child.remove(item)

    def row(self):
        return self._parent._children.index(self) if self._parent else 0
    
    def sort(self, column, order=QtCore.Qt.SortOrder.AscendingOrder):
        reverse = order == QtCore.Qt.SortOrder.DescendingOrder

        if column != NameColumn:
            self._children.sort(key=lambda x: (x[column], x[NameColumn]), reverse=reverse)
        else:
            self._children.sort(key=lambda x: (x[column], x[GroupColumn], x[ChannelColumn]), reverse=reverse)

        for child in self._children:
            child.sort(column, order)

    def __getitem__(self, idx):
        return self._values[idx]
    
    def __iter__(self):
        yield from self._values


class Model(QtCore.QAbstractItemModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._root = SearchItem()
        self.order = QtCore.Qt.SortOrder.AscendingOrder
        self.column = 0

    def __contains__(self, item):
        return item in self._root
    
    def add(self, data):
        if not data:
            return
        
        if isinstance(data, SearchItem):
            data = [data]

        self.beginResetModel()
        for item in data:
            self._root.appendChild(item.copy())
        self.sort(self.column, self.order)
        self.endResetModel()

    def all_data(self, return_names=False):
        return self._root.data(return_names)

    def clear(self):
        self.beginResetModel()
        self._root.clear()
        self.endResetModel()

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 7

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if index.isValid():

            if role == QtCore.Qt.ItemDataRole.DisplayRole:

                item = index.internalPointer()
                return item[index.column()]


    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return [
                "Name", "Group", "Index", "Unit", "Source name", "Source path", "Comment"
            ][section]

    def index(self, row, column, parent=QtCore.QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()

        if not parent.isValid():
            parentItem = self._root
        else:
            parentItem = parent.internalPointer()

        child = parentItem.child(row)
        if child:
            return self.createIndex(row, column, child)
        else:
            return QtCore.QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()

        child = index.internalPointer()
        parent = child.parent

        if parent is self._root:
            return QtCore.QModelIndex()

        return self.createIndex(parent.row(), 0, parent)
    
    def remove(self, data):
        if isinstance(data, SearchItem):
            data = [data]

        self.beginResetModel()
        for item in data:
            self._root.remove(item)
        self.sort(self.column, self.order)
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):

        if not parent.isValid():
            item = self._root
        else:
            item = parent.internalPointer()

        return item.childCount()

    def set_matches(self, matches):
        self.beginResetModel()
        self._root.clear()
        self._root = matches
        self.sort(self.column, self.order)
        self.endResetModel()

    def sort(self, column, order=QtCore.Qt.SortOrder.AscendingOrder):
        self.beginResetModel()
        self.order = order
        self.column = column
        self._root.sort(column, order)
        self.endResetModel()


class AdvancedSearch(Ui_SearchDialog, QtWidgets.QDialog):

    columns = 7

    def __init__(
        self,
        mdf,
        return_names=False,
        show_add_window=False,
        show_apply=False,
        show_pattern=True,
        apply_text="Apply",
        add_window_text="Add channels",
        show_search=True,
        window_title="Search & select channels",
        pattern=None,
        *args,
        **kwargs,
    ):
        channels_db = kwargs.pop("channels_db", {})
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self._settings = QtCore.QSettings()

        self.match_model = Model()
        self.matches.setModel(self.match_model)

        self.selection_model = Model()
        self.selection.setModel(self.selection_model)

        icon = QtGui.QIcon()
        icon.addFile(":/search.png", QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.setWindowIcon(icon)

        # self.selection.can_delete_items = True

        self.payload = {}
        self.add_window_request = False
        if mdf:
            self.channels_db = mdf.channels_db
            self.mdf = mdf
        else:
            self.mdf = None
            self.channels_db = channels_db

        self.apply_btn.clicked.connect(self._apply)
        self.add_btn.clicked.connect(self._add)
        self.add_window_btn.clicked.connect(self._add_window)
        self.cancel_btn.clicked.connect(self._cancel)

        self.search_box.editingFinished.connect(self.search_text_changed)
        self.match_kind.currentTextChanged.connect(self.search_box.textChanged.emit)
        self.matches.doubleClicked.connect(self._match_double_clicked)
        self.selection.doubleClicked.connect(self._selection_double_clicked)

        self.matches.header().sectionResized.connect(self.section_resized)
        self.selection.header().sectionResized.connect(self.section_resized)
        self.matches.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.selection.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.matches.customContextMenuRequested.connect(self.open_matches_menu)
        self.selection.customContextMenuRequested.connect(self.open_selection_menu)

        self.apply_pattern_btn.clicked.connect(self._apply_pattern)
        self.cancel_pattern_btn.clicked.connect(self._cancel_pattern)
        self.define_ranges_btn.clicked.connect(self._define_ranges)

        self.y_range_min.setMinimum(-np.inf)
        self.y_range_min.setMaximum(np.inf)
        self.y_range_min.setValue(0)
        self.y_range_max.setMinimum(-np.inf)
        self.y_range_max.setMaximum(np.inf)
        self.y_range_max.setValue(100)

        self.search_box.setFocus()

        self._return_names = return_names
        self.ranges = []

        self.pattern_window = False

        self.apply_btn.setText(apply_text)
        self.add_window_btn.setText(add_window_text)

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
            if pattern.get("case_sensitive", False):
                self.case_sensitivity_pattern.setCurrentText("Case sensitive")
            else:
                self.case_sensitivity_pattern.setCurrentText("Case insensitive")
            self.raw.setCheckState(QtCore.Qt.CheckState.Checked if pattern["raw"] else QtCore.Qt.CheckState.Unchecked)
            self.name.setText(pattern["name"])
            self.ranges = pattern["ranges"]
            self.integer_format.setCurrentText(pattern.get("integer_format", "phys"))

            y_range = sorted(pattern.get("y_range", (0, 100)))
            self.y_range_min.setValue(y_range[0])
            self.y_range_max.setValue(y_range[1])

        self.setWindowTitle(window_title)

        self.matches.setColumnWidth(NameColumn, 450)
        self.matches.setColumnWidth(GroupColumn, 40)
        self.matches.setColumnWidth(ChannelColumn, 40)
        self.matches.setColumnWidth(UnitColumn, 40)
        self.matches.setColumnWidth(SourceNameColumn, 170)
        self.matches.setColumnWidth(SourcePathColumn, 170)

        self.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint, True)

        self.pattern.editingFinished.connect(self.update_pattern_matches)
        self.case_sensitivity_pattern.currentIndexChanged.connect(self.update_pattern_matches)
        self.pattern_match_type.currentIndexChanged.connect(self.update_pattern_matches)
        self.filter_type.currentIndexChanged.connect(self.update_pattern_matches)
        self.filter_value.valueChanged.connect(self.update_pattern_matches)
        self.raw.checkStateChanged.connect(self.update_pattern_matches)
        self.show_alias_btn.clicked.connect(self.show_overlapping_alias)
        self.update_pattern_matches()

        self.showMaximized()

    def search_text_changed(self):
        text = self.search_box.text().strip()
        extened_search = self.extended_search.checkState() == QtCore.Qt.CheckState.Checked

        if len(text) >= 2:

            match_kind = self.match_kind.currentText()

            if match_kind == "Wildcard":
                wildcard = f"{os.urandom(6).hex()}_WILDCARD_{os.urandom(6).hex()}"
                if self._settings.value("search/add_wildcards", False, type=bool):
                    text = f'*{text}*'
                pattern = text.replace("*", wildcard)
                pattern = re.escape(pattern)
                pattern = pattern.replace(wildcard, ".*")
            else:
                pattern = text

            try:
                if self.case_sensitivity.currentText() == "Case insensitive":
                    pattern = re.compile(f"(?i){pattern}")
                else:
                    pattern = re.compile(pattern)

                if extened_search:
                    matches = {}

                    for group_index, group in enumerate(self.mdf.groups):
                        cg_source = getattr(group.channel_group, "acq_source", None)

                        # check channel group source name

                        if cg_source and (
                            pattern.fullmatch(cg_source.name or "")
                            or pattern.fullmatch(cg_source.path or "")
                            or pattern.fullmatch(cg_source.name or "")
                            or pattern.fullmatch(cg_source.comment or "")
                            or pattern.fullmatch(group.channel_group.acq_name or "")
                            or pattern.fullmatch(group.channel_group.comment)
                        ):
                            matches.update(
                                {
                                    (group_index, channel_index): {
                                        "names": [ch.name],
                                        "comment": extract_xml_comment(ch.comment).strip(),
                                        "unit": (ch.conversion and ch.conversion.unit) or ch.unit,
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
                                                "comment": extract_xml_comment(ch.comment).strip(),
                                                "unit": (ch.conversion and ch.conversion.unit) or ch.unit,
                                                "source_name": source.name if source else "",
                                                "source_path": source.path if source else "",
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
                                                "comment": extract_xml_comment(ch.comment).strip(),
                                                "unit": (ch.conversion and ch.conversion.unit) or ch.unit,
                                                "source_name": source.name if source else "",
                                                "source_path": source.path if source else "",
                                            }
                                            break

                else:
                    found_names = [name for name in self.channels_db if pattern.fullmatch(name)]

                    matches = {}
                    for name in found_names:
                        for entry in self.channels_db[name]:
                            (group_index, channel_index) = entry
                            ch = self.mdf.groups[group_index].channels[channel_index]

                            if entry not in matches:
                                cg = self.mdf.groups[group_index].channel_group

                                source = ch.source or getattr(cg, "acq_source", None)

                                matches[entry] = {
                                    "names": [],
                                    "comment": extract_xml_comment(ch.comment).strip(),
                                    "unit": (ch.conversion and ch.conversion.unit) or ch.unit,
                                    "source_name": source.name if source else "",
                                    "source_path": source.path if source else "",
                                }

                            info = matches[entry]

                            if name == ch.name:
                                info["names"].insert(0, name)
                            else:
                                info["names"].append(name)

                matches = [(group_index, channel_index, info) for (group_index, channel_index), info in matches.items()]
                
                new_matches = SearchItem()
                for group_index, channel_index, info in matches:
                    names = info["names"]
                    item = SearchItem(
                        [
                            names[0],
                            group_index,
                            channel_index,
                            info["unit"],
                            info["source_name"],
                            info["source_path"],
                            info["comment"],
                        ],
                        parent=new_matches
                    )
                    new_matches.appendChild(item)

                    for name in names[1:]:
                        child = SearchItem(
                                [
                                    name,
                                    group_index,
                                    channel_index,
                                    info["unit"],
                                    info["source_name"],
                                    info["source_path"],
                                    info["comment"],
                                ],
                                parent=item
                            )
                        item.appendChild(child)

                if new_matches:
                    self.match_model.set_matches(new_matches)
                    self.status.setText(f"{len(matches)} results")
                else:
                    self.match_model.clear()
                    self.status.setText("No results")

            except Exception as err:
                print(format_exc())
                self.status.setText(str(err))

            self.matches.collapseAll()

    def _add(self, event):
        indexes = list({index.row(): index for index in self.matches.selectedIndexes() if index.isValid()}.values())
        if not indexes:
            return
        
        data = []
        
        for index in indexes:
            new_data = index.internalPointer()

            if new_data not in self.selection_model:
                data.append(new_data)

        self.selection_model.add(data)

    def _apply(self, event=None):
        self.payload = self.selection_model.all_data(self._return_names)

        self.accept()

    def _apply_pattern(self, event):
        self.payload = {
            "pattern": self.pattern.text().strip(),
            "match_type": self.pattern_match_type.currentText(),
            "case_sensitive": self.case_sensitivity_pattern.currentText() == "Case sensitive",
            "filter_type": self.filter_type.currentText(),
            "filter_value": self.filter_value.value(),
            "raw": self.raw.checkState() == QtCore.Qt.CheckState.Checked,
            "ranges": self.ranges,
            "name": self.name.text().strip(),
            "integer_format": self.integer_format.currentText(),
            "y_range": sorted([self.y_range_min.value(), self.y_range_max.value()]),
        }

        if not self.payload["pattern"]:
            MessageBox.warning(self, "Cannot apply pattern", "The pattern cannot be empty")
            return

        if not self.payload["name"]:
            MessageBox.warning(self, "Cannot apply pattern", "The name cannot be empty")
            return

        self.pattern_window = True
        self.accept()

    def _add_window(self, event=None):
        self.add_window_request = True
        self._apply()

    def _cancel(self, event):
        self.payload = {}
        self.reject()

    def _cancel_pattern(self, event):
        self.payload = {}
        self.reject()

    def _define_ranges(self, event=None):
        name = self.pattern.text().strip()
        dlg = RangeEditor(f"Channel of <{name}>", ranges=self.ranges, parent=self)
        dlg.exec()
        if dlg.pressed_button == "apply":
            self.ranges = dlg.payload

    def _match_double_clicked(self, index):
        if not index.isValid():
            return

        new_data = index.internalPointer()

        if new_data not in self.selection_model:
            self.selection_model.add(new_data)

    def open_matches_menu(self, position):
        count = self.match_model.rowCount()

        self.context_menu = menu = QtWidgets.QMenu()
        menu.addAction(f"{count} items")
        menu.addSeparator()
        action = QtGui.QAction("Expand all", menu)
        action.triggered.connect(self.matches.expandAll)
        menu.addAction(action)
        action = QtGui.QAction("Collapse all", menu)
        action.triggered.connect(self.matches.collapseAll)
        menu.addAction(action)

        menu.exec(self.matches.viewport().mapToGlobal(position))

    def open_selection_menu(self, position):
        count = self.selection_model.rowCount()

        self.context_menu = menu = QtWidgets.QMenu()
        menu.addAction(f"{count} items")
        menu.addSeparator()
        action = QtGui.QAction("Delete", menu)
        action.triggered.connect(self.selection.expandAll)
        menu.addAction(action)

        action = menu.exec(self.selection.viewport().mapToGlobal(position))

        if action is None:
            return

        if action.text() == 'Delete':

            indexes = list({index.row(): index for index in self.selection.selectedIndexes() if index.isValid()}.values())
            if not indexes:
                return
            
            data = []
        
            for index in indexes:
                to_delete = index.internalPointer()

                if to_delete in self.selection_model:
                    data.append(to_delete)

            self.selection_model.remove(data)

    def _selection_double_clicked(self, index):
        if not index.isValid():
            return

        item = index.internalPointer()

        self.selection_model.remove(item)

    @QtCore.Slot(int, int, int, result=None)
    def section_resized(self, index, old_size, new_size):
        if self.selection.columnWidth(index) != new_size:
            self.selection.setColumnWidth(index, new_size)
        if self.matches.columnWidth(index) != new_size:
            self.matches.setColumnWidth(index, new_size)

    def show_overlapping_alias(
        self,
    ):
        indexes = [
            *list({index.row(): index for index in self.matches.selectedIndexes() if index.isValid()}.values()),
            *list({index.row(): index for index in self.selection.selectedIndexes() if index.isValid()}.values()),
        ]
        if not indexes:
            return
        
        for index in indexes:
            item = index.internalPointer()
        
            group_index, channel_index = item[GroupColumn], int(item[ChannelColumn])

        try:
            channel = self.mdf.get_channel_metadata(group=group_index, index=channel_index)
            info = (channel.data_type, channel.byte_offset, channel.bit_count)
            position = (group_index, index)
            alias = {}
            gp = self.mdf.groups[group_index]
            for ch_index, ch in enumerate(gp.channels):
                if ch_index != channel_index and (ch.data_type, ch.byte_offset, ch.bit_count) == info:
                    alias[ch.name] = (group_index, ch_index)

            if alias:
                alias_text = "\n".join(
                    f"{name} - group {group_index} index {ch_index}" for name, (group_index, ch_index) in alias.items()
                )
                MessageBox.information(
                    self,
                    f"{channel.name} - other overlapping alias",
                    f"{channel.name} has the following overlapping alias channels:\n\n{alias_text}",
                )
            else:
                MessageBox.information(
                    self,
                    f"{channel.name} - no other overlapping alias",
                    f"No other overlapping alias channels found for {channel.name}",
                )

        except:
            print(format_exc())

    def update_pattern_matches(self, *args):
        from ..widgets.mdi_area import extract_signals_using_pattern

        self.pattern_matches.clear()

        if not self.channels_db:
            return

        pattern_info = {
            "pattern": self.pattern.text().strip(),
            "case_sensitive": self.case_sensitivity_pattern.currentText() == "Case sensitive",
            "match_type": self.pattern_match_type.currentText(),
            "filter_value": self.filter_value.value(),
            "filter_type": self.filter_type.currentText(),
            "raw": self.raw.isChecked(),
            "integer_format": self.integer_format.currentText(),
            "ranges": [],
        }

        signals = extract_signals_using_pattern(
            mdf=self.mdf,
            channels_db=self.channels_db,
            pattern_info=pattern_info,
            ignore_value2text_conversions=True,
            as_names=True,
        )

        items = [QtWidgets.QTreeWidgetItem([name]) for name in signals]

        self.pattern_matches.addTopLevelItems(items)

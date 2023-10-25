import os
import re
import sys
from traceback import format_exc

from natsort import natsorted
from PySide6 import QtWidgets

from asammdf.gui.utils import excepthook

from ..ui.simple_search_dialog import Ui_SimpleSearchDialog

sys.excepthook = excepthook


class SimpleSearch(Ui_SimpleSearchDialog, QtWidgets.QDialog):
    def __init__(self, channels_db, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.result = set()
        self.channels_db = channels_db

        self.apply_btn.clicked.connect(self._apply)
        self.add_btn.clicked.connect(self._add)
        self.cancel_btn.clicked.connect(self._cancel)

        self.search_box.editingFinished.connect(self.search_text_changed)
        self.match_kind.currentTextChanged.connect(self.search_box.textChanged.emit)

        self.matches.itemDoubleClicked.connect(self._match_double_clicked)
        self.matches.itemSelectionChanged.connect(self.show_match_comment)
        self.selection.itemDoubleClicked.connect(self._selection_double_clicked)

        self.setWindowTitle("Search channels")

        self.search_box.setFocus()

    def show_match_comment(self, items=None):
        items = self.matches.selectedItems()
        if items:
            item = items[0]
            comment = self.channels_db[item.text(0)]
            self.comment.setText(comment)
        else:
            self.comment.setText("")

    def _match_double_clicked(self, new_item):
        selection = set()

        iterator = QtWidgets.QTreeWidgetItemIterator(self.selection)
        while item := iterator.value():
            selection.add(item.text(0))

            iterator += 1

        if new_item.text(0) not in selection:
            selection.add(new_item.text(0))

            selection = natsorted(selection)

            items = [QtWidgets.QTreeWidgetItem([name]) for name in selection]

            self.selection.clear()
            self.selection.addTopLevelItems(items)

    def _selection_double_clicked(self, item):
        root = self.selection.invisibleRootItem()
        (item.parent() or root).removeChild(item)

    def search_text_changed(self):
        self.comment.setText("")
        text = self.search_box.text().strip()
        if len(text) >= 2:
            if self.match_kind.currentText() == "Wildcard":
                wildcard = f"{os.urandom(6).hex()}_WILDCARD_{os.urandom(6).hex()}"
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

                matches = []
                for name in self.channels_db:
                    if pattern.fullmatch(name):
                        matches.append(name)

                items = [QtWidgets.QTreeWidgetItem([name]) for name in matches]

                self.matches.clear()
                self.matches.addTopLevelItems(items)

                if items:
                    self.status.setText("")
                else:
                    self.status.setText("No match found")

                self.matches.setFocus()
            except:
                self.status.setText(format_exc())
                self.matches.clear()

    def _add(self, event):
        items = set()

        iterator = QtWidgets.QTreeWidgetItemIterator(self.selection)
        while item := iterator.value():
            items.add(item.text(0))
            iterator += 1

        for item in self.matches.selectedItems():
            items.add(item.text(0))

        items = natsorted(items)

        items = [QtWidgets.QTreeWidgetItem([name]) for name in items]

        self.selection.clear()
        self.selection.addTopLevelItems(items)

    def _apply(self, event):
        self.result = set()
        iterator = QtWidgets.QTreeWidgetItemIterator(self.selection)
        while item := iterator.value():
            self.result.add(item.text(0))
            iterator += 1
        self.close()

    def _cancel(self, event):
        self.result = set()
        self.close()

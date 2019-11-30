# -*- coding: utf-8 -*-
import re

from natsort import natsorted
from PyQt5 import QtWidgets

from ..ui import resource_rc as resource_rc
from ..ui.search_dialog import Ui_SearchDialog


class AdvancedSearch(Ui_SearchDialog, QtWidgets.QDialog):
    def __init__(
        self, channels_db, return_names=False, show_add_window=False, *args, **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.result = set()
        self.add_window_request = False
        self.channels_db = channels_db

        self.apply_btn.clicked.connect(self._apply)
        self.add_btn.clicked.connect(self._add)
        self.add_window_btn.clicked.connect(self._add_window)
        self.cancel_btn.clicked.connect(self._cancel)

        self.search_box.editingFinished.connect(self.search_text_changed)
        self.match_kind.currentTextChanged.connect(self.search_box.textChanged.emit)

        self._return_names = return_names

        if not show_add_window:
            self.add_window_btn.hide()

        self.setWindowTitle("Search & select channels")

    def search_text_changed(self):
        text = self.search_box.text().strip()
        if len(text) >= 2:
            if self.match_kind.currentText() == "Wildcard":
                pattern = text.replace("*", "_WILDCARD_")
                pattern = re.escape(pattern)
                pattern = pattern.replace("_WILDCARD_", ".*")
            else:
                pattern = text

            try:
                pattern = re.compile(f"(?i){pattern}")
                matches = [name for name in self.channels_db if pattern.match(name)]
                self.matches.clear()
                self.matches.addItems(matches)
                if matches:
                    self.status.setText("")
                else:
                    self.status.setText("No match found")
            except Exception as err:
                self.status.setText(str(err))
                self.matches.clear()

    def _add(self, event):
        count = self.selection.count()
        names = set(self.selection.item(i).text() for i in range(count))

        to_add = set(item.text() for item in self.matches.selectedItems())

        names = natsorted(names | to_add)

        self.selection.clear()
        self.selection.addItems(names)

    def _apply(self, event):
        count = self.selection.count()

        if self._return_names:
            self.result = set(self.selection.item(i).text() for i in range(count))
        else:
            self.result = set()
            for i in range(count):
                for entry in self.channels_db[self.selection.item(i).text()]:
                    self.result.add(entry)
        self.close()

    def _add_window(self, event):
        count = self.selection.count()

        if self._return_names:
            self.result = set(self.selection.item(i).text() for i in range(count))
        else:
            self.result = set()
            for i in range(count):
                for entry in self.channels_db[self.selection.item(i).text()]:
                    self.result.add(entry)

        self.add_window_request = True
        self.close()

    def _cancel(self, event):
        self.result = set()
        self.close()

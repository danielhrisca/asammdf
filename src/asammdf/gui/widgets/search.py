from PySide6 import QtCore, QtWidgets

from ..ui.search_widget import Ui_SearchWidget


class SearchWidget(Ui_SearchWidget, QtWidgets.QWidget):
    selectionChanged = QtCore.Signal()

    def __init__(self, sorted_keys, channels_db, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.channels_db = channels_db

        self.matches = 0
        self.current_index = 1
        self.entries = []

        completer = QtWidgets.QCompleter(sorted_keys, self)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        completer.setModelSorting(QtWidgets.QCompleter.ModelSorting.CaseInsensitivelySortedModel)
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self.search.setCompleter(completer)

        self.search.textChanged.connect(self.display_results)

        self.up_btn.clicked.connect(self.up)
        self.down_btn.clicked.connect(self.down)

    def down(self, event):
        if self.matches:
            self.current_index += 1
            if self.current_index >= self.matches:
                self.current_index = 0
            self.label.setText(f"{self.current_index + 1} of {self.matches}")
            self.selectionChanged.emit()

    def up(self, event):
        if self.matches:
            self.current_index -= 1
            if self.current_index < 0:
                self.current_index = self.matches - 1
            self.label.setText(f"{self.current_index + 1} of {self.matches}")
            self.selectionChanged.emit()

    def set_search_option(self, option):
        if option == "Match start":
            self.search.completer().setFilterMode(QtCore.Qt.MatchFlag.MatchStartsWith)
        elif option == "Match contains":
            self.search.completer().setFilterMode(QtCore.Qt.MatchFlag.MatchContains)

    def display_results(self, text):
        channel_name = text.strip()
        if channel_name in self.channels_db:
            self.entries = self.channels_db[channel_name]
            self.matches = len(self.entries)
            self.label.setText(f"1 of {self.matches}")
            self.current_index = 0
            self.selectionChanged.emit()

        else:
            self.label.setText("No match")
            self.matches = 0
            self.current_index = 0
            self.entries = []

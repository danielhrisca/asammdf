# -*- coding: utf-8 -*-
import os

try:
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5 import uic
    from ..ui import resource_qt5 as resource_rc

    QT = 5

except ImportError:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    from PyQt4 import uic
    from ..ui import resource_qt4 as resource_rc

    QT = 4
    
HERE = os.path.dirname(os.path.realpath(__file__))


class SearchWidget(QWidget):

    selectionChanged = pyqtSignal()

    def __init__(self, channels_db, *args, **kwargs):
        super(SearchWidget, self).__init__(*args, **kwargs)
        uic.loadUi(os.path.join(HERE, "..", "ui", "search_widget.ui"), self)
        self.channels_db = channels_db

        self.matches = 0
        self.current_index = 1
        self.entries = []

        completer = QCompleter(sorted(self.channels_db, key=lambda x: x.lower()), self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setModelSorting(QCompleter.CaseInsensitivelySortedModel)
        if QT == 5:
            completer.setFilterMode(Qt.MatchContains)
        self.search.setCompleter(completer)

        self.search.textChanged.connect(self.display_results)

        self.up_btn.clicked.connect(self.up)
        self.down_btn.clicked.connect(self.down)

    def down(self, event):
        if self.matches:
            self.current_index += 1
            if self.current_index >= self.matches:
                self.current_index = 0
            self.label.setText("{} of {}".format(self.current_index + 1, self.matches))
            self.selectionChanged.emit()

    def up(self, event):
        if self.matches:
            self.current_index -= 1
            if self.current_index < 0:
                self.current_index = self.matches - 1
            self.label.setText("{} of {}".format(self.current_index + 1, self.matches))
            self.selectionChanged.emit()

    def set_search_option(self, option):
        if QT == 5:
            if option == "Match start":
                self.search.completer().setFilterMode(Qt.MatchStartsWith)
            elif option == "Match contains":
                self.search.completer().setFilterMode(Qt.MatchContains)

    def display_results(self, text):
        channel_name = text.strip()
        if channel_name in self.channels_db:
            self.entries = self.channels_db[channel_name]
            self.matches = len(self.entries)
            self.label.setText("1 of {}".format(self.matches))
            self.current_index = 0
            self.selectionChanged.emit()

        else:
            self.label.setText("No match")
            self.matches = 0
            self.current_index = 0
            self.entries = []

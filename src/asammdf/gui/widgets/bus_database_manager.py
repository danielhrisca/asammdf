from pathlib import Path

from PySide6 import QtWidgets

from ..ui.bus_database_manager import Ui_BusDatabaseManager
from .database_item import DatabaseItem


class BusDatabaseManager(Ui_BusDatabaseManager, QtWidgets.QWidget):
    def __init__(self, databases, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

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

        self.load_can_database_btn.clicked.connect(self.load_can_database)
        self.load_lin_database_btn.clicked.connect(self.load_lin_database)

        self.showMaximized()

    def to_config(self):
        dbs = {
            "CAN": [],
            "LIN": [],
        }

        count = self.can_database_list.count()

        for row in range(count):
            item = self.can_database_list.item(row)
            widget = self.can_database_list.itemWidget(item)
            dbs["CAN"].append((widget.bus.currentText(), widget.database.text()))

        count = self.lin_database_list.count()

        for row in range(count):
            item = self.lin_database_list.item(row)
            widget = self.lin_database_list.itemWidget(item)
            dbs["LIN"].append((widget.bus.currentText(), widget.database.text()))

        return dbs

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

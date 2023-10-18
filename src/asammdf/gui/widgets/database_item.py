from PySide6 import QtWidgets

from ..ui.database_item import Ui_DatabaseItemUI


class DatabaseItem(Ui_DatabaseItemUI, QtWidgets.QWidget):
    def __init__(self, database, bus_type="CAN"):
        super().__init__()
        self.setupUi(self)

        items = [f"Any {bus_type} bus"] + [f"{bus_type} {i:>2} only" for i in range(1, 17)]

        self.database.setText(database.strip())
        self.bus.addItems(items)
        self.bus.setCurrentIndex(0)

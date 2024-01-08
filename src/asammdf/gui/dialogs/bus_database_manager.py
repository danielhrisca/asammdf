from PySide6 import QtCore, QtWidgets

from ..widgets.bus_database_manager import BusDatabaseManager


class BusDatabaseManagerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.setObjectName("BusDatabaseManagerDialog")
        self.resize(404, 294)
        self.setSizeGripEnabled(True)
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        self._settings = QtCore.QSettings()

        databases = {}

        can_databases = self._settings.value("can_databases", [])
        buses = can_databases[::2]
        dbs = can_databases[1::2]

        databases["CAN"] = list(zip(buses, dbs))

        lin_databases = self._settings.value("lin_databases", [])
        buses = lin_databases[::2]
        dbs = lin_databases[1::2]

        databases["LIN"] = list(zip(buses, dbs))

        self.widget = BusDatabaseManager(databases)

        self.verticalLayout.addWidget(self.widget)

        self.horLayout = QtWidgets.QHBoxLayout(self)

        spacer = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.horLayout.addSpacerItem(spacer)
        self.horLayout.addWidget(self.apply_btn)
        self.horLayout.addWidget(self.cancel_btn)

        self.verticalLayout.addLayout(self.horLayout)

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.cancel)
        self.pressed_button = "cancel"

        self.setWindowTitle("Bus Database Manager")

        self.showMaximized()

    def apply(self, *args):
        self.pressed_button = "apply"
        self.close()

    def cancel(self, *args):
        self.pressed_button = "cancel"
        self.close()

    def store(self):
        databases = self.widget.to_config()

        dbs = []
        for bus, database in databases["CAN"]:
            dbs.extend((bus, database))

        self._settings.setValue("can_databases", dbs)

        dbs = []
        for bus, database in databases["LIN"]:
            dbs.append(bus)
            dbs.append(database)

        self._settings.setValue("lin_databases", dbs)

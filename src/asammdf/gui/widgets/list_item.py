from PySide6 import QtWidgets


class ListItem(QtWidgets.QListWidgetItem):
    def __init__(self, entry, name="", computation=None, parent=None, origin_uuid=None):
        super().__init__()

        self.entry = entry
        self.name = name
        self.computation = computation
        self.origin_uuid = origin_uuid

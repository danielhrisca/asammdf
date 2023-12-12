from PySide6 import QtCore, QtWidgets


class SearchTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, name, group, index, unit, source_name, source_path, comment):
        super().__init__([name, group, index, unit, source_name, source_path, comment])

        self.name = name
        self.group = group
        self.index = index
        self.unit = unit
        self.source_name = source_name
        self.source_path = source_path
        self.comment = comment

    def __del__(self):
        self.name = None
        self.group = None
        self.index = None
        self.unit = None
        self.source_name = None
        self.source_path = None
        self.comment = None


class SearchTreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, can_delete_items=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.NoDragDrop)
        self.setUniformRowHeights(True)

        self.can_delete_items = can_delete_items

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Delete and self.can_delete_items:
            selected_items = self.selectedItems()

            root = self.invisibleRootItem()
            for item in selected_items:
                (item.parent() or root).removeChild(item)

            event.accept()
        else:
            super().keyPressEvent(event)

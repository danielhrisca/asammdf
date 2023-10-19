# taken from https://stackoverflow.com/a/46096319/11009349

from PySide6 import QtGui
from PySide6.QtWidgets import QMainWindow, QTreeWidget, QTreeWidgetItem


class ViewTree(QTreeWidget):
    def __init__(self, value, parent=None):
        super().__init__(parent)
        self.setHeaderLabel("Computation parameters")

        def fill_item(item, value):
            def new_item(parent, text, val=None):
                child = QTreeWidgetItem([text])
                fill_item(child, val)
                parent.addChild(child)
                child.setExpanded(True)

            if value is None:
                return
            elif isinstance(value, dict):
                for key, val in sorted(value.items()):
                    new_item(item, str(key), val)
            elif isinstance(value, (list, tuple)):
                for val in value:
                    text = str(val) if not isinstance(val, (dict, list, tuple)) else "[%s]" % type(val).__name__
                    new_item(item, text, val)
            else:
                new_item(item, str(value))

        fill_item(self.invisibleRootItem(), value)


class ComputedChannelInfoWindow(QMainWindow):
    def __init__(self, signal, parent=None):
        super().__init__(parent)
        self.setCentralWidget(ViewTree(signal.computation, self))
        self.setWindowTitle(f"Computed channel {signal.name}")
        self.setMinimumSize(600, 400)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)

        self.setWindowIcon(icon)

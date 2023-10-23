import contextlib
from importlib.metadata import distribution, PackageNotFoundError
import re
from typing import Dict, Optional

from packaging.requirements import Requirement
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog, QTreeWidget, QTreeWidgetItem, QVBoxLayout


class DependenciesDlg(QDialog):
    def __init__(self, package_name: str) -> None:
        """Create a dialog to list all dependencies for `package_name`."""

        super().__init__()

        # Variables
        self._package_name = package_name

        # Widgets
        self._tree = QTreeWidget()

        # Setup widgets
        self._setup_widgets()

        # Setup layout
        self._setup_layout()

        # Signals and slots
        self._connect_signals()

    def _setup_widgets(self) -> None:
        self.setWindowTitle(f"Dependency Overview for {self._package_name}")

        self.setMinimumWidth(700)
        self.setMinimumHeight(500)

        icon = QIcon()
        icon.addFile(":/asammdf.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.setWindowIcon(icon)

        headers = ["package", "required", "installed", "summary", "url"]
        self._tree.setHeaderLabels(headers)
        self._populate_tree(self._package_name)
        self._tree.expandAll()
        self._tree.resizeColumnToContents(0)
        self._tree.resizeColumnToContents(1)
        self._tree.resizeColumnToContents(2)

    def _setup_layout(self) -> None:
        vbox = QVBoxLayout()
        self.setLayout(vbox)

        vbox.addWidget(self._tree)

    def _connect_signals(self) -> None:
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)

    def _populate_tree(self, package_name: str) -> None:
        package_dist = distribution(package_name)
        requires = package_dist.requires
        if requires is None:
            return

        root_nodes: Dict[str, QTreeWidgetItem] = {}

        def get_root_node(name: Optional[str] = None) -> QTreeWidgetItem:
            if name is None:
                name = "mandatory"

            if name in root_nodes:
                return root_nodes[name]

            new_root_node = QTreeWidgetItem([name])
            root_nodes[name] = new_root_node

            font = new_root_node.font(0)
            font.setBold(True)
            new_root_node.setFont(0, font)

            self._tree.invisibleRootItem().addChild(new_root_node)
            return new_root_node

        for req_string in requires:
            req = Requirement(req_string)

            parent = get_root_node()

            if req.marker is not None:
                match = re.search(r"extra\s*==\s*['\"](?P<extra>\S+)['\"]", str(req.marker))
                if match:
                    parent = get_root_node(match["extra"])

            item = QTreeWidgetItem()
            item.setText(0, req.name)
            item.setText(1, str(req.specifier))

            with contextlib.suppress(PackageNotFoundError):
                dist = distribution(req.name)
                item.setText(2, str(dist.version))
                item.setText(3, dist.metadata["Summary"])
                item.setText(4, dist.metadata["Home-Page"])

            parent.addChild(item)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        if item.parent() is self._tree.invisibleRootItem():
            return

        package_name = item.text(0)
        DependenciesDlg.show_dependencies(package_name)

    @staticmethod
    def show_dependencies(package_name: str) -> None:
        dlg = DependenciesDlg(package_name)
        dlg.exec_()

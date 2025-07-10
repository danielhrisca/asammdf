from collections import defaultdict
import contextlib
from importlib.metadata import distribution, PackageNotFoundError
import re

from packaging.requirements import Requirement
from PySide6.QtCore import QSize
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)


class DependenciesDlg(QDialog):
    def __init__(self, package_name: str, is_root_package: bool = True) -> None:
        """Create a dialog to list all dependencies for `package_name`."""

        super().__init__()

        # Variables
        self._package_name = package_name
        self._is_root_package = is_root_package

        # Widgets
        self._tree = QTreeWidget()
        self._copy_btn = QPushButton("Copy installed dependencies to clipboard")

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

        # enable copy button for root package only
        self._copy_btn.setVisible(self._is_root_package)

    def _setup_layout(self) -> None:
        vbox = QVBoxLayout()
        self.setLayout(vbox)

        vbox.addWidget(self._tree)
        vbox.addWidget(self._copy_btn)

    def _connect_signals(self) -> None:
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._copy_btn.clicked.connect(self._on_copy_button_clicked)

    def _populate_tree(self, package_name: str) -> None:
        package_dist = distribution(package_name)
        requires = package_dist.requires
        if requires is None:
            return

        root_nodes: dict[str, QTreeWidgetItem] = {}

        def get_root_node(name: str | None = None) -> QTreeWidgetItem:
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

        for group, requirements in grouped_dependencies(package_name).items():
            for req_string in requirements:
                req = Requirement(req_string)
                parent_node = get_root_node(group)

                item = QTreeWidgetItem()
                item.setText(0, req.name)
                item.setText(1, str(req.specifier))

                with contextlib.suppress(PackageNotFoundError):
                    dist = distribution(req.name)
                    item.setText(2, str(dist.version))
                    item.setText(3, dist.metadata["Summary"])
                    item.setText(4, dist.metadata["Home-Page"])

                parent_node.addChild(item)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        if item.parent() is self._tree.invisibleRootItem():
            return

        package_name = item.text(0)
        DependenciesDlg.show_dependencies(package_name, is_root_package=False)

    def _on_copy_button_clicked(self) -> None:
        """Create a list of all dependencies and their versions and write it to
        clipboard.
        """
        lines: list[str] = []
        dependencies = find_all_dependencies(self._package_name)
        max_name_length = max(len(name) for name in dependencies)

        header = f"{'Package':<{max_name_length}} Version"
        lines.append(header)
        lines.append("-" * len(header))
        for name in sorted(dependencies):
            version = distribution(name).version
            lines.append(f"{name:<{max_name_length}} {version}")

        # write to clipboard
        QGuiApplication.clipboard().setText("\n".join(lines))

    @staticmethod
    def show_dependencies(package_name: str, is_root_package: bool = True) -> None:
        dlg = DependenciesDlg(package_name, is_root_package)
        dlg.exec()


def grouped_dependencies(package_name: str) -> dict[str, list[str]]:
    """Retrieve a dictionary grouping the dependencies of a given package into
    mandatory and optional categories.

    This function fetches the dependencies of the specified package and
    categorizes them into groups, such as 'mandatory' or any optional feature
    groups specified by `extra` markers.

    :param package_name:
        The name of the package to analyze.
    :return:
        A dictionary where keys are group names (e.g., 'mandatory',
        'extra_feature') and values are lists of package names corresponding to
        those groups.
    """
    dependencies: defaultdict[str, list[str]] = defaultdict(list)
    package_dist = distribution(package_name)

    if requires := package_dist.requires:
        for req_string in requires:
            req = Requirement(req_string)

            group = "mandatory"
            if match := re.search(r"extra\s*==\s*['\"](?P<extra>\S+)['\"]", str(req.marker)):
                group = match["extra"]

            dependencies[group].append(req_string)
    return dependencies


def find_all_dependencies(package_name: str) -> set[str]:
    """Recursively find all dependencies of a given package, including
    transitive dependencies.

    This function determines all dependencies of the specified package,
    following any transitive dependencies (i.e., dependencies of dependencies)
    and returning a complete set of package names.

    :param package_name:
        The name of the package to analyze.
    :return:
        A set of all dependencies for the package, including transitive
        dependencies.
    """

    def _flatten_groups(grouped_deps: dict[str, list[str]]) -> set[str]:
        _dep_set = set()
        for group, requirements in grouped_deps.items():
            _dep_set |= {Requirement(req_string).name for req_string in requirements}
        return _dep_set

    dep_set: set[str] = {package_name}
    todo = _flatten_groups(grouped_dependencies(package_name))
    while todo:
        req_name = todo.pop()
        if req_name in dep_set:
            continue
        try:
            todo |= _flatten_groups(grouped_dependencies(req_name))
        except PackageNotFoundError:
            continue
        dep_set.add(req_name)
    return dep_set

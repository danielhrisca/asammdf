# -*- coding: utf-8 -*-

from struct import pack
from datetime import datetime, date

from PyQt5 import QtCore, QtGui, QtWidgets


class TreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setUniformRowHeights(True)

        self.mode = "Natural sort"

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            elif len(selected_items) == 1:
                item = selected_items[0]
                checked = item.checkState(0)
                if checked == QtCore.Qt.Checked:
                    item.setCheckState(0, QtCore.Qt.Unchecked)
                else:
                    item.setCheckState(0, QtCore.Qt.Checked)
            else:
                if any(
                    item.checkState(0) == QtCore.Qt.Unchecked for item in selected_items
                ):
                    checked = QtCore.Qt.Checked
                else:
                    checked = QtCore.Qt.Unchecked
                for item in selected_items:
                    item.setCheckState(0, checked)
        else:
            super().keyPressEvent(event)

    def startDrag(self, supportedActions):
        def get_data(item):
            data = set()
            count = item.childCount()

            if count:
                for i in range(count):
                    child = item.child(i)

                    if child.childCount():
                        data = data | get_data(child)
                    else:

                        name = child.name.encode("utf-8")
                        entry = child.entry
                        if entry[1] != 0xFFFFFFFFFFFFFFFF:
                            data.add(
                                (
                                    str(child.mdf_uuid).encode("ascii"),
                                    name,
                                    entry[0],
                                    entry[1],
                                    len(name),
                                )
                            )
            else:
                name = item.name.encode("utf-8")
                entry = item.entry
                if entry[1] != 0xFFFFFFFFFFFFFFFF:
                    data.add(
                        (
                            str(item.mdf_uuid).encode("ascii"),
                            name,
                            entry[0],
                            entry[1],
                            len(name),
                        )
                    )

            return data

        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = set()
        for item in selected_items:
            data = data | get_data(item)

        data = [
            pack(
                f"<12s3q{name_length}s",
                uuid,
                group_index,
                channel_index,
                name_length,
                name,
            )
            for uuid, name, group_index, channel_index, name_length in sorted(data)
        ]

        mimeData.setData(
            "application/octet-stream-asammdf", QtCore.QByteArray(b"".join(data))
        )

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec_(QtCore.Qt.MoveAction)


class FileTreeItem(QtWidgets.QTreeWidgetItem):

    def __init__(self, path, start_time, parent=None):
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()

        super().__init__(parent, [path, start_time])

    def __lt__(self, otherItem):
        column = self.treeWidget().sortColumn()

        if column == 1:
            val1 = datetime.fromisoformat(self.text(column))
            val2 = datetime.fromisoformat(otherItem.text(column))

            return val1 < val2
        else:
            return self.text(column) < otherItem.text(column)

    def __del__(self):
        self.entry = self.name = self.mdf_uuid = None


class FileTreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setUniformRowHeights(True)

        self.mode = "Natural sort"


class ChannelsTreeWidget(QtWidgets.QTreeWidget):
    itemsDeleted = QtCore.pyqtSignal(list)
    set_time_offset = QtCore.pyqtSignal(list)
    items_rearranged = QtCore.pyqtSignal()
    add_channels_request = QtCore.pyqtSignal(list)
    show_properties = QtCore.pyqtSignal(object)
    insert_computation = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.setUniformRowHeights(True)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)
        self.details_enabled = False
        self._has_hidden_items = False

        self.setHeaderHidden(True)
        self.setColumnCount(4)
        self.setDragEnabled(True)

        # self.setColumnWidth(2,40)
        # self.setColumnWidth(3,10)
        self.header().setStretchLastSection(False)

        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            elif len(selected_items) == 1:
                item = selected_items[0]
                checked = item.checkState(0)
                if checked == QtCore.Qt.Checked:
                    item.setCheckState(0, QtCore.Qt.Unchecked)
                else:
                    item.setCheckState(0, QtCore.Qt.Checked)
            else:
                if any(
                    item.checkState(0) == QtCore.Qt.Unchecked for item in selected_items
                ):
                    checked = QtCore.Qt.Checked
                else:
                    checked = QtCore.Qt.Unchecked
                for item in selected_items:
                    item.setCheckState(0, checked)
        else:
            super().keyPressEvent(event)

    def startDrag(self, supportedActions):
        print('start drag')
        def get_data(item):
            data = set()
            count = item.childCount()

            if count:
                for i in range(count):
                    child = item.child(i)

                    if child.childCount():
                        data = data | get_data(child)
                    else:

                        name = child.name.encode("utf-8")
                        entry = child.entry
                        if entry[1] != 0xFFFFFFFFFFFFFFFF:
                            data.add(
                                (
                                    str(child.mdf_uuid).encode("ascii"),
                                    name,
                                    entry[0],
                                    entry[1],
                                    len(name),
                                )
                            )
            else:
                name = item.name.encode("utf-8")
                entry = item.entry
                if entry[1] != 0xFFFFFFFFFFFFFFFF:
                    data.add(
                        (
                            str(item.mdf_uuid).encode("ascii"),
                            name,
                            entry[0],
                            entry[1],
                            len(name),
                        )
                    )

            return data

        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = set()
        for item in selected_items:
            data = data | get_data(item)

        data = [
            pack(
                f"<12s3q{name_length}s",
                uuid,
                group_index,
                channel_index,
                name_length,
                name,
            )
            for uuid, name, group_index, channel_index, name_length in sorted(data)
        ]

        mimeData.setData(
            "application/octet-stream-asammdf", QtCore.QByteArray(b"".join(data))
        )

        print('execut', b"".join(data))
        print(bin(self.dragDropMode()))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec_(QtCore.Qt.MoveAction)

    def dragEnterEvent(self, e):
        print("DragEnter")
        e.accept()

    def dragMoveEvent(self, e):
        print("DragMove")
        e.accept()

    def dropEvent(self, e):
        print('drop', e)
        if e.source() is self:
            print('a')
            super().dropEvent(e)
            self.items_rearranged.emit()
        else:
            print('b')
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
            else:
                super().dropEvent(e)

    def open_menu(self, position):

        item = self.itemAt(position)
        if item is None:
            return

        menu = QtWidgets.QMenu()
        menu.addAction(self.tr(f"{'TO DO'} items in the list"))
        menu.addSeparator()
        menu.addAction(self.tr("Copy name (Ctrl+C)"))
        menu.addAction(self.tr("Copy display properties (Ctrl+Shift+C)"))
        menu.addAction(self.tr("Paste display properties (Ctrl+Shift+P)"))
        menu.addSeparator()
        menu.addAction(self.tr("Enable all"))
        menu.addAction(self.tr("Disable all"))
        menu.addAction(self.tr("Enable all but this"))
        menu.addSeparator()
        if self._has_hidden_items:
            show_hide = "Show disabled items"
        else:
            show_hide = "Hide disabled items"
        menu.addAction(self.tr(show_hide))
        menu.addSeparator()


class ChannelsTreeItem(QtWidgets.QTreeWidgetItem):

    color_changed = QtCore.pyqtSignal(object, str)
    enable_changed = QtCore.pyqtSignal(object, int)
    ylink_changed = QtCore.pyqtSignal(object, int)
    individual_axis_changed = QtCore.pyqtSignal(object, int)

    def __init__(self, entry, name="", computation=None, parent=None, mdf_uuid=None, category="channel", texts=("", "", "", ""), unit="", details="", kind="f",
        precision=3,
        tooltip="", uuid=""):
        super().__init__(parent, list(texts))

        self.entry = entry
        self.name = name
        self.computation = computation
        self.mdf_uuid = mdf_uuid
        self.category = category

        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)

        self.setCheckState(0, QtCore.Qt.Checked)
        self.setCheckState(2, QtCore.Qt.Unchecked)
        self.setCheckState(3, QtCore.Qt.Unchecked)

        self.color = "#ff0000"
        self._value_prefix = ""
        self._value = ""
        self._name = ""

        self.details.setText(details or "\tSource not available")

        self.details.setVisible(False)

        self.uuid = uuid
        self.ranges = {}
        self.unit = unit.strip()
        self.kind = kind
        self.precision = precision

        self._transparent = True
        self._tooltip = tooltip

        self.color_btn.clicked.connect(self.select_color)
        # self.display.stateChanged.connect(self.display_changed)
        self.ylink.stateChanged.connect(self._ylink_changed)
        self.individual_axis.stateChanged.connect(self._individual_axis)

        self.fm = QtGui.QFontMetrics(self.name.font())

        self.setToolTip(self._tooltip or self._name)

        if kind in "SUVui":
            self.fmt = "{}"
        else:
            self.fmt = f"{{:.{self.precision}f}}"

    def set_precision(self, precision):
        if self.kind == "f":
            self.precision = precision
            self.fmt = f"{{:.{self.precision}f}}"

        # def display_changed(self, state):
        #     state = self.display.checkState()
        #     self.enable_changed.emit(self.uuid, state)

    def _individual_axis(self, state):
        state = self.individual_axis.checkState()
        self.individual_axis_changed.emit(self.uuid, state)

    def _ylink_changed(self, state):
        state = self.ylink.checkState()
        self.ylink_changed.emit(self.uuid, state)

    def mouseDoubleClickEvent(self, event):
        dlg = RangeEditor(self.unit, self.ranges)
        dlg.exec_()
        if dlg.pressed_button == "apply":
            self.ranges = dlg.result

    def select_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color))
        if color.isValid():
            self.set_color(color.name())

            self.color_changed.emit(self.uuid, color.name())

    def set_fmt(self, fmt):
        if self.kind in "SUV":
            self.fmt = "{}"
        elif self.kind == "f":
            self.fmt = f"{{:.{self.precision}f}}"
        else:
            if fmt == "hex":
                self.fmt = "0x{:X}"
            elif fmt == "bin":
                self.fmt = "0b{:b}"
            elif fmt == "phys":
                self.fmt = "{}"

    def set_color(self, color):
        self.color = color
        self.set_name(self._name)
        self.set_value(self._value)
        self.color_btn.setStyleSheet(f"background-color: {color};")

        palette = self.name.palette()

        brush = QtGui.QBrush(QtGui.QColor(color))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)

        self.name.setPalette(palette)

    def set_selected(self, on):
        palette = self.name.palette()
        if on:
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        else:
            brush = QtGui.QBrush(QtGui.QColor(self.color))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)

        self.name.setPalette(palette)

    def set_name(self, text=""):
        self.setToolTip(self._tooltip or text)
        self._name = text

    def set_prefix(self, text=""):
        self._value_prefix = text

    def update(self):
        width = self.name.size().width()
        if self.unit:
            self.name.setText(
                self.fm.elidedText(
                    f"{self._name} ({self.unit})", QtCore.Qt.ElideMiddle, width
                )
            )
        else:
            self.name.setText(
                self.fm.elidedText(self._name, QtCore.Qt.ElideMiddle, width)
            )
        self.set_value(self._value, update=True)

    def set_value(self, value, update=False):
        if self._value == value and update is False:
            return

        self._value = value
        if self.ranges and value not in ("", "n.a."):
            for (start, stop), color in self.ranges.items():
                if start <= value <= stop:
                    self.setStyleSheet(f"background-color: {color};")
                    break
            else:
                self.setStyleSheet("background-color: transparent;")
        elif not self._transparent:
            self.setStyleSheet("background-color: transparent;")
        template = "{{}}{}"
        if value not in ("", "n.a."):
            template = template.format(self.fmt)
        else:
            template = template.format("{}")
        try:
            self.value.setText(template.format(self._value_prefix, value))
        except (ValueError, TypeError):
            template = "{}{}"
            self.value.setText(template.format(self._value_prefix, value))

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()
        if modifier == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_C:
            QtWidgets.QApplication.instance().clipboard().setText(self._name)

        elif (
                modifier == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier)
                and key == QtCore.Qt.Key_C
        ):
            QtWidgets.QApplication.instance().clipboard().setText(
                self.get_display_properties()
            )

        elif (
                modifier == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier)
                and key == QtCore.Qt.Key_P
        ):
            info = QtWidgets.QApplication.instance().clipboard().text()
            try:
                info = json.loads(info)
                self.set_color(info["color"])
                self.color_changed.emit(self.uuid, info["color"])
                self.set_fmt(info["format"])
                self.individual_axis.setCheckState(
                    QtCore.Qt.Checked
                    if info["individual_axis"]
                    else QtCore.Qt.Unchecked
                )
                self.ylink.setCheckState(
                    QtCore.Qt.Checked if info["ylink"] else QtCore.Qt.Unchecked
                )
                self.set_precision(info["precision"])

                parent = self.parent().parent().parent().parent().parent()
                sig, index = parent.plot.signal_by_uuid(self.uuid)
                viewbox = parent.plot.view_boxes[index]
                viewbox.setYRange(info["min"], info["max"], padding=0)

                # self.display.setCheckState(
                #     QtCore.Qt.Checked if info["display"] else QtCore.Qt.Unchecked
                # )

                self.ranges = {}

                for key, val in info["ranges"].items():
                    start, stop = [float(e) for e in key.split("|")]
                    self.ranges[(start, stop)] = val

            except:
                pass

        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        width = self.name.size().width()
        if self.unit:
            self.name.setText(
                self.fm.elidedText(
                    f"{self._name} ({self.unit})", QtCore.Qt.ElideMiddle, width
                )
            )
        else:
            self.name.setText(
                self.fm.elidedText(self._name, QtCore.Qt.ElideMiddle, width)
            )

    def text(self):
        return self._name

    def get_display_properties(self):
        info = {
            "color": self.color,
            "precision": self.precision,
            "ylink": self.ylink.checkState() == QtCore.Qt.Checked,
            "individual_axis": self.individual_axis.checkState() == QtCore.Qt.Checked,
            "format": "hex"
            if self.fmt.startswith("0x")
            else "bin"
            if self.fmt.startswith("0b")
            else "phys",
            # "display": self.display.checkState() == QtCore.Qt.Checked,
            "ranges": {
                f"{start}|{stop}": val for (start, stop), val in self.ranges.items()
            },
        }

        parent = self.parent().parent().parent().parent().parent()

        sig, index = parent.plot.signal_by_uuid(self.uuid)

        min_, max_ = parent.plot.view_boxes[index].viewRange()[1]

        info["min"] = float(min_)
        info["max"] = float(max_)

        return json.dumps(info)

    def does_not_exist(self):
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/error.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.color_btn.setIcon(icon)
        self.color_btn.setFlat(True)
        self.color_btn.clicked.disconnect()

    def disconnect_slots(self):
        self.color_changed.disconnect()
        self.enable_changed.disconnect()
        self.ylink_changed.disconnect()
        self.individual_axis_changed.disconnect()



if __name__ == "__main__":
    pass
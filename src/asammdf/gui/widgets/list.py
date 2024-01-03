import json
from struct import pack

from natsort import natsorted
from PySide6 import QtCore, QtGui, QtWidgets

from ...blocks.utils import extract_mime_names


class ListWidget(QtWidgets.QListWidget):
    itemsDeleted = QtCore.Signal(list)
    set_time_offset = QtCore.Signal(list)
    items_rearranged = QtCore.Signal()
    add_channels_request = QtCore.Signal(list)
    show_properties = QtCore.Signal(object)
    insert_computation = QtCore.Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.details_enabled = False

        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)

        self.setAlternatingRowColors(True)

        self.can_delete_items = True
        self.setAcceptDrops(True)

        self.itemSelectionChanged.connect(self.item_selection_changed)

        self.show()
        self._has_hidden_items = False

    def item_selection_changed(self, item=None):
        selection = list(self.selectedItems())
        for row in range(self.count()):
            item = self.item(row)
            if item in selection:
                widget = self.itemWidget(item)
                if widget is not None:
                    widget.set_selected(True)
            else:
                widget = self.itemWidget(item)
                if widget is not None:
                    widget.set_selected(False)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key.Key_Delete and self.can_delete_items:
            selected_items = self.selectedItems()
            deleted = []
            for item in selected_items:
                row = self.row(item)
                item_widget = self.itemWidget(item)
                deleted.append(getattr(item_widget, "uuid", None))
                if hasattr(item_widget, "disconnect_slots"):
                    item_widget.disconnect_slots()
                self.takeItem(row)
            if deleted:
                self.itemsDeleted.emit(deleted)
            event.accept()

        elif key == QtCore.Qt.Key.Key_Space and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            event.accept()
            selected_items = self.selectedItems()
            if not selected_items:
                return

            states = [self.itemWidget(item).display.checkState() for item in selected_items]

            if any(state == QtCore.Qt.CheckState.Unchecked for state in states):
                state = QtCore.Qt.CheckState.Checked
            else:
                state = QtCore.Qt.CheckState.Unchecked
            for item in selected_items:
                wid = self.itemWidget(item)
                wid.display.setCheckState(state)

        elif key == QtCore.Qt.Key.Key_Space and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            event.accept()
            selected_items = self.selectedItems()
            if not selected_items:
                return

            states = [self.itemWidget(item).individual_axis.checkState() for item in selected_items]

            if any(state == QtCore.Qt.CheckState.Unchecked for state in states):
                state = QtCore.Qt.CheckState.Checked
            else:
                state = QtCore.Qt.CheckState.Unchecked
            for item in selected_items:
                wid = self.itemWidget(item)
                wid.individual_axis.setCheckState(state)

        elif modifiers == QtCore.Qt.KeyboardModifier.ControlModifier and key == QtCore.Qt.Key.Key_C:
            selected_items = self.selectedItems()
            if not selected_items:
                event.accept()
                return

            self.itemWidget(selected_items[0]).keyPressEvent(event)

        elif modifiers == (
            QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier
        ) and key in (
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.Key.Key_P,
        ):
            selected_items = self.selectedItems()
            if not selected_items:
                event.accept()
                return
            self.itemWidget(selected_items[0]).keyPressEvent(event)

        else:
            super().keyPressEvent(event)

    def startDrag(self, supportedActions):
        selected_items = self.selectedItems()

        mimeData = QtCore.QMimeData()

        data = []

        for item in selected_items:
            entry = item.entry
            computation = item.computation

            widget = self.itemWidget(item)

            color = widget.color
            unit = widget.unit

            if entry == (-1, -1):
                info = {
                    "name": item.name,
                    "computation": computation,
                    "computed": True,
                    "unit": unit,
                    "color": color,
                }
                info = json.dumps(info).encode("utf-8")
            else:
                info = item.name.encode("utf-8")

            data.append(
                pack(
                    f"<12s3q{len(info)}s",
                    str(item.origin_uuid).encode("ascii"),
                    entry[0],
                    entry[1],
                    len(info),
                    info,
                )
            )

        mimeData.setData("application/octet-stream-asammdf", QtCore.QByteArray(b"".join(data)))

        drag = QtGui.QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec(QtCore.Qt.DropAction.CopyAction)

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat("application/octet-stream-asammdf"):
            e.accept()
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        if e.source() is self:
            super().dropEvent(e)
            self.items_rearranged.emit()
        else:
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

        else:
            menu = QtWidgets.QMenu()
            menu.addAction(self.tr(f"{self.count()} items in the list"))
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

            menu.addAction(self.tr("Add to common Y axis"))
            menu.addAction(self.tr("Remove from common Y axis"))
            menu.addSeparator()
            menu.addAction(self.tr("Set unit"))
            menu.addAction(self.tr("Set precision"))
            menu.addSeparator()
            menu.addAction(self.tr("Relative time base shift"))
            menu.addAction(self.tr("Set time base start offset"))
            menu.addSeparator()
            menu.addAction(self.tr("Insert computation using this channel"))
            menu.addSeparator()
            menu.addAction(self.tr("Delete (Del)"))
            menu.addSeparator()
            menu.addAction(self.tr("Toggle details"))
            menu.addAction(self.tr("File/Computation properties"))

            action = menu.exec_(self.viewport().mapToGlobal(position))

        if action is None:
            return

        if action.text() == "Copy name (Ctrl+C)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_C, QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self.itemWidget(item).keyPressEvent(event)

        elif action.text() == "Copy display properties (Ctrl+Shift+C)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_C,
                QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
            self.itemWidget(item).keyPressEvent(event)

        elif action.text() == "Paste display properties (Ctrl+Shift+P)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_P,
                QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
            )
            self.itemWidget(item).keyPressEvent(event)

        elif action.text() == "Enable all":
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                widget.display.setCheckState(QtCore.Qt.CheckState.Checked)

        elif action.text() == "Disable all":
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                widget.display.setCheckState(QtCore.Qt.CheckState.Unchecked)

        elif action.text() == "Enable all but this":
            selected_items = self.selectedItems()
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                if item in selected_items:
                    widget.display.setCheckState(QtCore.Qt.CheckState.Unchecked)
                else:
                    widget.display.setCheckState(QtCore.Qt.CheckState.Checked)

        elif action.text() == show_hide:
            if self._has_hidden_items:
                for i in range(self.count()):
                    item = self.item(i)
                    item.setHidden(False)
            else:
                for i in range(self.count()):
                    item = self.item(i)
                    widget = self.itemWidget(item)
                    if not widget.display.isChecked():
                        item.setHidden(True)

            self._has_hidden_items = not self._has_hidden_items

        elif action.text() == "Add to common Y axis":
            selected_items = self.selectedItems()
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                if item in selected_items:
                    widget.ylink.setCheckState(QtCore.Qt.CheckState.Checked)

        elif action.text() == "Remove from common Y axis":
            selected_items = self.selectedItems()
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                if item in selected_items:
                    widget.ylink.setCheckState(QtCore.Qt.CheckState.Unchecked)

        elif action.text() == "Set unit":
            selected_items = self.selectedItems()

            unit, ok = QtWidgets.QInputDialog.getText(self, "Set new unit", "Unit:")

            if ok:
                selected_items = self.selectedItems()
                for i in range(self.count()):
                    item = self.item(i)
                    widget = self.itemWidget(item)
                    if item in selected_items:
                        widget.unit = unit
                        widget.update()

        elif action.text() == "Set precision":
            selected_items = self.selectedItems()

            precision, ok = QtWidgets.QInputDialog.getInt(self, "Set new precision (float decimals)", "Precision:")

            if ok and 0 <= precision <= 15:
                for i in range(self.count()):
                    item = self.item(i)
                    widget = self.itemWidget(item)
                    if item in selected_items:
                        widget.set_precision(precision)
                        widget.update()

        elif action.text() in (
            "Relative time base shift",
            "Set time base start offset",
        ):
            selected_items = self.selectedItems()
            if selected_items:
                if action.text() == "Relative time base shift":
                    offset, ok = QtWidgets.QInputDialog.getDouble(
                        self, "Relative offset [s]", "Offset [s]:", decimals=6
                    )
                    absolute = False
                else:
                    offset, ok = QtWidgets.QInputDialog.getDouble(
                        self,
                        "Absolute time start offset [s]",
                        "Offset [s]:",
                        decimals=6,
                    )
                    absolute = True
                if ok:
                    uuids = []

                    for i in range(self.count()):
                        item = self.item(i)
                        widget = self.itemWidget(item)
                        if item in selected_items:
                            uuids.append(widget.uuid)
                    self.set_time_offset.emit([absolute, offset, *uuids])

        elif action.text() == "Delete (Del)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Delete, QtCore.Qt.KeyboardModifier.NoModifier
            )
            self.keyPressEvent(event)

        elif action.text() == "Toggle details":
            self.details_enabled = not self.details_enabled
            for i in range(self.count()):
                item = self.item(i)
                widget = self.itemWidget(item)
                widget.details.setVisible(self.details_enabled)
                item.setSizeHint(widget.sizeHint())

        elif action.text() == "File/Computation properties":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                self.show_properties.emit(self.itemWidget(item).uuid)

        elif action.text() == "Insert computation using this channel":
            selected_items = self.selectedItems()
            if len(selected_items) == 1:
                item = selected_items[0]
                self.insert_computation.emit(self.itemWidget(item)._name)


class MinimalListWidget(QtWidgets.QListWidget):
    itemsDeleted = QtCore.Signal(list)
    itemsPasted = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        self.setAlternatingRowColors(True)

        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)

        self.setAcceptDrops(True)
        self.show()

        self.itemSelectionChanged.connect(self.item_selection_changed)

        self.minimal_menu = False
        self.all_texts = False
        self.placeholder_text = ""

        self.user_editable = True

    def item_selection_changed(self, item=None):
        try:
            selection = list(self.selectedItems())
            for row in range(self.count()):
                item = self.item(row)
                if item in selection:
                    self.itemWidget(item).set_selected(True)
                else:
                    self.itemWidget(item).set_selected(False)
        except:
            pass

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        if key == QtCore.Qt.Key.Key_Delete and self.user_editable:
            selected_items = self.selectedItems()
            deleted = []

            if self.all_texts:
                to_delete = set()
                for item in selected_items:
                    row = self.row(item)
                    deleted.append(row)
                    item_widget = self.itemWidget(item)
                    if hasattr(item_widget, "disconnect_slots"):
                        item_widget.disconnect_slots()
                    to_delete.add(item.text())

                all_texts = set()

                count = self.count()
                for row in range(count):
                    item = self.item(row)
                    all_texts.add(item.text())

                self.clear()
                self.addItems(natsorted(all_texts - to_delete))

            else:
                for item in selected_items:
                    row = self.row(item)
                    deleted.append(row)
                    item_widget = self.itemWidget(item)
                    if hasattr(item_widget, "disconnect_slots"):
                        item_widget.disconnect_slots()
                    self.takeItem(row)

            if deleted:
                self.itemsDeleted.emit(deleted)

            event.accept()

        elif key == QtCore.Qt.Key.Key_C and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            text = []
            for item in self.selectedItems():
                try:
                    text.append(self.itemWidget(item).text())
                except:
                    text.append(item.text())

            if text:
                text = "\n".join(text)
            else:
                text = ""

            QtWidgets.QApplication.instance().clipboard().setText(text)
            event.accept()

        elif (
            key == QtCore.Qt.Key.Key_V
            and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier
            and self.user_editable
        ):
            lines = QtWidgets.QApplication.instance().clipboard().text().splitlines()
            if lines:
                try:
                    self.addItems(lines)
                    self.itemsPasted.emit()
                except:
                    pass
            event.accept()
        else:
            super().keyPressEvent(event)

    def open_menu(self, position):
        menu = QtWidgets.QMenu()

        if self.minimal_menu:
            if self.count() > 0:
                menu.addAction(self.tr(f"{self.count()} items in the list"))
                menu.addSeparator()
                menu.addAction(self.tr("Delete (Del)"))
            else:
                return
        else:
            if self.count() == 0:
                menu.addAction(self.tr(f"{self.count()} items in the list"))
                menu.addSeparator()
                if self.user_editable:
                    menu.addAction(self.tr("Paste names (Ctrl+V)"))
            else:
                menu.addAction(self.tr(f"{self.count()} items in the list"))
                menu.addSeparator()
                menu.addAction(self.tr("Copy names (Ctrl+C)"))
                if self.user_editable:
                    menu.addAction(self.tr("Paste names (Ctrl+V)"))
                    menu.addSeparator()
                    menu.addAction(self.tr("Delete (Del)"))

        action = menu.exec_(self.viewport().mapToGlobal(position))

        if action is None:
            return

        if action.text() == "Delete (Del)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Delete, QtCore.Qt.KeyboardModifier.NoModifier
            )
            self.keyPressEvent(event)
        elif action.text() == "Copy names (Ctrl+C)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_C, QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self.keyPressEvent(event)
        elif action.text() == "Paste names (Ctrl+V)":
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_V, QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self.keyPressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count() == 0 and self.placeholder_text:
            painter = QtGui.QPainter(self.viewport())
            painter.save()
            col = self.palette().placeholderText().color()
            painter.setPen(col)
            fm = self.fontMetrics()
            elided_text = fm.elidedText(
                self.placeholder_text, QtCore.Qt.TextElideMode.ElideRight, self.viewport().width()
            )
            painter.drawText(self.viewport().rect(), QtCore.Qt.AlignmentFlag.AlignCenter, elided_text)
            painter.restore()

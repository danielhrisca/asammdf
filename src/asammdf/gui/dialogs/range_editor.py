from PySide6 import QtCore, QtWidgets

from ..ui.range_editor_dialog import Ui_RangeDialog
from ..widgets.range_widget import RangeWidget


class RangeEditor(Ui_RangeDialog, QtWidgets.QDialog):
    def __init__(self, name, unit="", ranges=None, brush=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowType.WindowSystemMenuHint
            | QtCore.Qt.WindowType.WindowMinMaxButtonsHint
        )

        self.name = name
        self.unit = unit
        self.result = []
        self.pressed_button = None
        self._brush = brush

        if ranges:
            for range_info in ranges:
                widget = RangeWidget(
                    parent=self,
                    name=self.name,
                    brush=brush,
                    **range_info,
                )

                item = QtWidgets.QListWidgetItem()
                item.setSizeHint(widget.sizeHint())
                self.ranges.addItem(item)
                self.ranges.setItemWidget(item, widget)

        self.ranges.setUniformItemSizes(True)

        self.apply_btn.clicked.connect(self.apply)
        self.insert_btn.clicked.connect(self.insert)
        self.cancel_btn.clicked.connect(self.cancel)
        self.reset_btn.clicked.connect(self.reset)

        self.setWindowTitle(f"Edit {self.name} range colors")
        self.show()

    def apply(self, event):
        ranges = []
        count = self.ranges.count()
        for i in range(count):
            item = self.ranges.item(i)
            if item is None:
                continue

            widget = self.ranges.itemWidget(item)
            ranges.append(widget.to_dict(brush=self._brush))

        self.result = ranges
        self.pressed_button = "apply"
        self.close()

    def insert(self, event):
        widget = RangeWidget(self.name)

        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.ranges.addItem(item)
        self.ranges.setItemWidget(item, widget)

    def reset(self, event):
        self.ranges.clear()
        self.result = []

    def cancel(self, event):
        self.result = []
        self.pressed_button = "cancel"
        self.close()

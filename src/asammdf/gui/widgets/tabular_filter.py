import pandas as pd
from PySide6 import QtCore, QtWidgets

from ..dialogs.messagebox import MessageBox
from ..ui.tabular_filter import Ui_TabularFilter


class TabularFilter(Ui_TabularFilter, QtWidgets.QWidget):
    def __init__(self, signals, int_format, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self._target = None

        self.names = [item[0] for item in signals]

        self.dtype_kind = [item[1] for item in signals]

        self.is_bytearray = [item[2] for item in signals]

        self.int_format = int_format

        self.relation.addItems(["AND", "OR"])
        self.column.addItems(self.names)
        self.op.addItems([">", ">=", "<", "<=", "==", "!="])

        self.target.editingFinished.connect(self.validate_target)
        self.column.currentIndexChanged.connect(self.column_changed)

    def column_changed(self, index):
        self.target.setText("")
        self._target = None

    def validate_target(self):
        idx = self.column.currentIndex()
        column_name = self.column.currentText()
        kind = self.dtype_kind[idx]
        target = self.target.text().strip()

        if target:
            if kind in "ui":
                if target.startswith("0x"):
                    try:
                        self._target = int(target, 16)
                    except:
                        MessageBox.warning(
                            None,
                            "Wrong target value",
                            f"{column_name} requires an integer target value",
                        )

                elif target.startswith("0b"):
                    try:
                        self._target = int(target, 2)
                    except:
                        MessageBox.warning(
                            None,
                            "Wrong target value",
                            f"{column_name} requires an integer target value",
                        )
                else:
                    if self.int_format == "hex":
                        try:
                            self._target = int(target, 16)
                            self.target.setText(f"0x{self._target:X}")
                        except:
                            MessageBox.warning(
                                None,
                                "Wrong target value",
                                f"{column_name} requires a hex-format integer target value",
                            )
                    elif self.int_format == "bin":
                        try:
                            self._target = int(target, 2)
                            self.target.setText(f"0b{self._target:b}")
                        except:
                            MessageBox.warning(
                                None,
                                "Wrong target value",
                                f"{column_name} requires a bin-format integer target value",
                            )
                    else:
                        try:
                            self._target = int(target)
                        except:
                            try:
                                self._target = int(target, 16)
                                self.target.setText(f"0x{self._target:X}")
                            except:
                                MessageBox.warning(
                                    None,
                                    "Wrong target value",
                                    f"{column_name} requires an integer target value",
                                )
            elif kind == "f":
                try:
                    self._target = float(target)
                except:
                    MessageBox.warning(
                        None,
                        "Wrong target value",
                        f"{column_name} requires a float target value",
                    )
            elif kind == "O":
                is_bytearray = self.is_bytearray[idx]
                if is_bytearray:
                    try:
                        bytes.fromhex(target.replace(" ", ""))
                    except:
                        MessageBox.warning(
                            None,
                            "Wrong target value",
                            f"{column_name} requires a correct hexstring",
                        )
                    else:
                        target = target.strip().replace(" ", "")
                        target = [target[i : i + 2] for i in range(0, len(target), 2)]

                        target = " ".join(target).upper()
                        if self._target is None:
                            self._target = f'"{target}"'
                            self.target.setText(target)
                        elif self._target.strip('"') != target:
                            self._target = f'"{target}"'
                            self.target.setText(target)
                else:
                    self._target = f'"{target}"'
            elif kind == "S":
                self._target = f'b"{target}"'
            elif kind == "U":
                self._target = f'"{target}"'
            elif kind == "M":
                try:
                    pd.Timestamp(target)
                except:
                    MessageBox.warning(
                        None,
                        "Wrong target value",
                        f"Datetime {column_name} requires a correct pandas Timestamp literal",
                    )
                else:
                    self._target = target

    def to_config(self):
        info = {
            "enabled": self.enabled.checkState() == QtCore.Qt.CheckState.Checked,
            "relation": self.relation.currentText(),
            "column": self.column.currentText(),
            "op": self.op.currentText(),
            "target": str(self._target),
        }

        return info

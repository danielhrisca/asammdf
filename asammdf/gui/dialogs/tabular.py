# -*- coding: utf-8 -*-
from ..ui import resource_rc as resource_rc
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore


class TabularValuesDialog(QtWidgets.QDialog):
    def __init__(self, signals, ranges, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(QtCore.Qt.Window)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle("Tabular values")

        self.table = QtWidgets.QTableWidget(self)

        self.header = []
        for sig in signals:
            self.header.append("t [s]")
            self.header.append("{} ({})".format(sig.name, sig.unit))

        self.table.setColumnCount(2 * len(signals))
        self.table.setRowCount(max(len(sig) for sig in signals))
        self.table.setHorizontalHeaderLabels(self.header)

        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.table.horizontalHeader().setMinimumSectionSize(
            QtWidgets.QHeaderView.Stretch
        )
        self.table.horizontalHeader().setToolTip("")
        self.table.horizontalHeader().setMinimumSectionSize(100)
        self.table.horizontalHeader().sectionClicked.connect(self.show_name)
        self.table.horizontalHeader().entered.connect(self.hover)
        self.table.cellEntered.connect(self.hover)

        for i, sig in enumerate(signals):
            size = len(sig)
            for j in range(size):
                # self.table.setCellWidget(
                #     j,
                #     2*i,
                #     QtWidgets.QLabel(str(sig.timestamps[j]), self.table),
                # )
                #
                # value = sig.samples[j]
                #
                # label = QtWidgets.QLabel(str(sig.samples[j]), self.table)
                #
                # for (start, stop), color in range_.items():
                #     if start <= value < stop:
                #         label.setStyleSheet(
                #             "background-color: {};".format(color))
                #         break
                # else:
                #     label.setStyleSheet("background-color: transparent;")
                #
                # self.table.setCellWidget(
                #     j,
                #     2 * i + 1,
                #     label,
                # )

                self.table.setItem(
                    j, 2 * i, QtWidgets.QTableWidgetItem(str(sig.timestamps[j]))
                )

                self.table.setItem(
                    j, 2 * i + 1, QtWidgets.QTableWidgetItem(str(sig.samples[j]))
                )

        layout.addWidget(self.table)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/info.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        self.setWindowIcon(icon)
        self.setGeometry(240, 60, 1200, 600)

        screen = QtWidgets.QApplication.desktop().screenGeometry()
        self.move((screen.width() - 1200) // 2, (screen.height() - 600) // 2)

    def hover(self, row, column):
        print("hover", row, column)

    def show_name(self, index):
        name = self.header[index // 2]
        widget = self.table.horizontalHeader()
        QtWidgets.QToolTip.showText(widget.mapToGlobal(QtCore.QPoint(0, 0)), name)

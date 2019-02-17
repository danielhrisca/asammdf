# -*- coding: utf-8 -*-
from ..ui import resource_qt5 as resource_rc
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class TabularValuesDialog(QDialog):
    def __init__(self, signals, ranges, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowFlags(Qt.Window)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle("Tabular values")

        self.table = QTableWidget(self)

        self.header = []
        for sig in signals:
            self.header.append("t [s]")
            self.header.append("{} ({})".format(sig.name, sig.unit))

        self.table.setColumnCount(2 * len(signals))
        self.table.setRowCount(max(len(sig) for sig in signals))
        self.table.setHorizontalHeaderLabels(self.header)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setMinimumSectionSize(QHeaderView.Stretch)
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
                #     QLabel(str(sig.timestamps[j]), self.table),
                # )
                #
                # value = sig.samples[j]
                #
                # label = QLabel(str(sig.samples[j]), self.table)
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

                self.table.setItem(j, 2 * i, QTableWidgetItem(str(sig.timestamps[j])))

                self.table.setItem(j, 2 * i + 1, QTableWidgetItem(str(sig.samples[j])))

        layout.addWidget(self.table)

        icon = QIcon()
        icon.addPixmap(QPixmap(":/info.png"), QIcon.Normal, QIcon.Off)

        self.setWindowIcon(icon)
        self.setGeometry(240, 60, 1200, 600)

        screen = QApplication.desktop().screenGeometry()
        self.move((screen.width() - 1200) // 2, (screen.height() - 600) // 2)

    def hover(self, row, column):
        print("hover", row, column)

    def show_name(self, index):
        name = self.header[index // 2]
        widget = self.table.horizontalHeader()
        QToolTip.showText(widget.mapToGlobal(QPoint(0, 0)), name)

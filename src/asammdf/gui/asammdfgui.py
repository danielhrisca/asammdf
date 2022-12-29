# -*- coding: utf-8 -*-
import argparse
import os
import sys

os.environ["QT_API"] = "pyside6"
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

import pyqtgraph
from PySide6 import QtGui, QtWidgets

from asammdf.gui.utils import excepthook
from asammdf.gui.widgets.main import MainWindow
from asammdf.gui.widgets.plot import monkey_patch_pyqtgraph

sys.excepthook = excepthook


def _cmd_line_parser():
    """"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", nargs="*", help="list of measurement files")
    return parser


def main(measurements=None):
    monkey_patch_pyqtgraph()
    parser = _cmd_line_parser()
    args = parser.parse_args(sys.argv[1:])
    app = pyqtgraph.mkQApp()
    app.setOrganizationName("py-asammdf")
    app.setOrganizationDomain("py-asammdf")
    app.setApplicationName("py-asammdf")

    main = MainWindow(args.measurements)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    app.exec()


if __name__ == "__main__":
    main()

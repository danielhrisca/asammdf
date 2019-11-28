# -*- coding: utf-8 -*-
import argparse
import sys

from PyQt5 import QtWidgets

from asammdf.gui.widgets.main import MainWindow
from asammdf.gui.utils import excepthook


sys.excepthook = excepthook


def _cmd_line_parser():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurements", nargs="*", help="list of measurement files",
    )
    return parser


def main(measurements=None):
    parser = _cmd_line_parser()
    args = parser.parse_args(sys.argv[1:])
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("py-asammdf")
    app.setOrganizationDomain("py-asammdf")
    app.setApplicationName("py-asammdf")
    main = MainWindow(args.measurements)

    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    app.exec_()


if __name__ == "__main__":
    main()

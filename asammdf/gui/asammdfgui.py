# -*- coding: utf-8 -*-
import argparse
import sys

try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication

from asammdf.gui.widgets.main import MainWindow
from asammdf.gui.utils import excepthook

sys.excepthook = excepthook

def _cmd_line_parser():
    '''
    return a command line parser. It is used when generating the documentation
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', nargs='*', help='list of measurement files')
    return parser


def main(measurements=None):
    parser = _cmd_line_parser()
    args = parser.parse_args(sys.argv[1:])
    app = QApplication(sys.argv)
    main = MainWindow(args.measurements)
    app.exec_()


if __name__ == "__main__":
    main()

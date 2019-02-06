# -*- coding: utf-8 -*-
import sys

try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt4.QtGui import QApplication

from asammdf.gui.widgets.main import MainWindow
from asammdf.gui.utils import excepthook

sys.excepthook = excepthook


def main():
    app = QApplication(sys.argv)
    main = MainWindow()
    app.exec_()


if __name__ == "__main__":
    main()

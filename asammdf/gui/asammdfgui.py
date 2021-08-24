# -*- coding: utf-8 -*-
import argparse
import sys


from PyQt5 import QtWidgets
import pyqtgraph

def _keys(self, styles):
    def getId(obj):
        try:
            return obj._id
        except AttributeError:
            obj._id = next(pyqtgraph.graphicsItems.ScatterPlotItem.SymbolAtlas._idGenerator)
            return obj._id

    res = [
        (symbol if isinstance(symbol, (str, int)) else getId(symbol), size, getId(pen), getId(brush))
        for symbol, size, pen, brush in styles[:1]
    ]
    
    return res

# speed-up monkey patches
pyqtgraph.graphicsItems.ScatterPlotItem._USE_QRECT = False
pyqtgraph.graphicsItems.ScatterPlotItem.SymbolAtlas._keys = _keys

from asammdf.gui.utils import excepthook
from asammdf.gui.widgets.main import MainWindow

sys.excepthook = excepthook



def _cmd_line_parser():
    """"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", nargs="*", help="list of measurement files")
    return parser


def main(measurements=None):
    parser = _cmd_line_parser()
    args = parser.parse_args(sys.argv[1:])
    app = pyqtgraph.mkQApp()
    app.setOrganizationName("py-asammdf")
    app.setOrganizationDomain("py-asammdf")
    app.setApplicationName("py-asammdf")
    main = MainWindow(args.measurements)

    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    app.exec_()


if __name__ == "__main__":
    main()

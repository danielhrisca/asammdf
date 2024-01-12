import argparse
import os
import sys

os.environ["QT_API"] = "pyside6"
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
os.environ["PYSIDE6_OPTION_PYTHON_ENUM"] = "2"

alternative_sitepacakges = os.environ.get("ASAMMDF_PYTHONPATH", "")

if alternative_sitepacakges:
    os.environ["PYTHONPATH"] = alternative_sitepacakges
    sys.path.insert(0, alternative_sitepacakges)

import pyqtgraph
from PySide6 import QtWidgets

from asammdf.gui.utils import excepthook, set_app_user_model_id
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
    set_app_user_model_id("py-asammdf")

    _main_window = MainWindow(measurements or args.measurements)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    app.exec()


if __name__ == "__main__":
    main()

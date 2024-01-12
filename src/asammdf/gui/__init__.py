import os

os.environ["QT_API"] = "pyside6"
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

if "QT_ENABLE_HIGHDPI_SCALING" not in os.environ:
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

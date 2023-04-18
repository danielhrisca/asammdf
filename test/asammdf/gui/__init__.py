#!/usr/bin/env python
import os
import sys

PYSIDE = "PySide"
PYSIDE2 = "PySide2"
PYSIDE6 = "PySide6"
PYQT4 = "PyQt4"
PYQT5 = "PyQt5"
PYQT6 = "PyQt6"

QT_LIB = os.getenv("PYQTGRAPH_QT_LIB")

if QT_LIB is not None:
    try:
        __import__(QT_LIB)
    except ModuleNotFoundError:
        pass
        # raise ModuleNotFoundError(
        #     f"Environment variable PYQTGRAPH_QT_LIB is set to '{os.getenv('PYQTGRAPH_QT_LIB')}',"
        #     f" but no module with this name was found."
        # )

## Automatically determine which Qt package to use (unless specified by
## environment variable).
## This is done by first checking to see whether one of the libraries
## is already imported. If not, then attempt to import in the order
## specified in libOrder.
libOrder = [PYQT6, PYSIDE6, PYQT5, PYSIDE2]
if QT_LIB is None:
    for lib in libOrder:
        if lib in sys.modules:
            QT_LIB = lib
            break

if QT_LIB is None:
    for lib in libOrder:
        qt = lib + ".QtCore"
        try:
            __import__(qt)
            QT_LIB = lib
            break
        except ImportError:
            pass

if QT_LIB is None:
    raise Exception(
        "PyQtGraph requires one of PyQt5, PyQt6, PySide2 or PySide6; none of these packages could be imported."
    )

mandatory = ["QtCore", "QtGui", "QtQuick", "QtTest", "QtWidgets"]
QT_LIB = __import__(f"{QT_LIB}", fromlist=mandatory)
if not QT_LIB:
    raise ModuleNotFoundError(
        f"At least one of the following modules are not present in {QT_LIB}: {mandatory}"
    )

for m in mandatory:
    globals()[m] = getattr(QT_LIB, m)

# QtCore = getattr(QT_LIB, "QtCore")
# QtGui = getattr(QT_LIB, "QtGui")
# QtTest = getattr(QT_LIB, "QtTest")
# QtWidgets = getattr(QT_LIB, "QtWidgets")

__all__ = mandatory

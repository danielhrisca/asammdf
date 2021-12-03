# -*- coding: utf-8 -*-

from datetime import datetime, timedelta, timezone
import sys
from textwrap import wrap

from PyQt5 import QtCore, QtGui, QtWidgets

LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.functions as fn
from pyqtgraph.graphicsItems.ButtonItem import ButtonItem


class FormatedAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):

        self.plus = self.minus = None

        super().__init__(*args, **kwargs)

        self.format = "phys"
        self.mode = "phys"
        self.text_conversion = None
        self.origin = None

        self.setStyle(autoExpandTextSpace=False, autoReduceTextSpace=False)
        self.adjuster = None

        self.geometryChanged.connect(self.handle_geometry_changed)

        if self.orientation in ("left", "right"):

            self.plus = ButtonItem(":/plus.png", 12, parentItem=self)
            self.minus = ButtonItem(":/minus.png", 12, parentItem=self)

            self.plus.clicked.connect(self.increase_width)
            self.minus.clicked.connect(self.decrease_width)

            if self.scene() is not None:
                self.scene().addItem(self.plus)
                self.scene().addItem(self.minus)

        self.set_pen(self._pen)

    def increase_width(self):
        width = self.width() + 10
        self.setWidth(width)

    def decrease_width(self):
        width = max(self.width() - 10, 48)
        self.setWidth(width)

    def handle_geometry_changed(self):
        if self.adjuster is not None:
            self.adjuster.setRect(self.boundingRect())
            self.adjuster.setPos(self.pos())

    def tickStrings(self, values, scale, spacing):
        strns = []

        if self.text_conversion and self.mode == "phys":
            strns = []
            for val in values:
                nv = self.text_conversion.convert(np.array([val]))[0]

                val = float(val)

                if val.is_integer():
                    val = int(val)

                    if self.format == "hex":
                        val = hex(int(val))
                    elif self.format == "bin":
                        val = bin(int(val))
                    else:
                        val = str(val)
                else:
                    val = f"{val:.6f}"

                if isinstance(nv, bytes):
                    try:
                        strns.append(f'{val}={nv.decode("utf-8")}')
                    except:
                        strns.append(f'{val}={nv.decode("latin-1")}')
                else:

                    strns.append(val)
        else:
            if self.format == "phys":
                strns = super(FormatedAxis, self).tickStrings(values, scale, spacing)

            elif self.format == "hex":
                for val in values:
                    val = float(val)
                    if val.is_integer():
                        val = hex(int(val))
                    else:
                        val = ""
                    strns.append(val)

            elif self.format == "bin":
                for val in values:
                    val = float(val)
                    if val.is_integer():
                        val = bin(int(val))
                    else:
                        val = ""
                    strns.append(val)
            elif self.format == "time":
                strns = [str(timedelta(seconds=val)) for val in values]
            elif self.format == "date":
                strns = (
                    pd.to_datetime(np.array(values) + self.origin.timestamp(), unit="s")
                    .tz_localize("UTC")
                    .tz_convert(LOCAL_TIMEZONE)
                    .astype(str)
                    .to_list()
                )

        return [val[:80] for val in strns]

    def setLabel(self, text=None, units=None, unitPrefix=None, **args):
        """overwrites pyqtgraph setLabel"""
        show_label = False
        if text is not None:
            self.labelText = text
            show_label = True
        if units is not None:
            self.labelUnits = units
            show_label = True
        if show_label:
            self.showLabel()
        if unitPrefix is not None:
            self.labelUnitPrefix = unitPrefix
        if len(args) > 0:
            self.labelStyle = args
        self.label.setHtml(self.labelString())
        self._adjustSize()
        self.picture = None
        self.update()

    def mouseDragEvent(self, event):
        if self.linkedView() is None:
            return
        if self.orientation in ["left", "right"]:
            return self.linkedView().mouseDragEvent(event)
        else:
            return self.linkedView().mouseDragEvent(event)

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            event.accept()
            self.raiseContextMenu(event)
        else:
            lv = self.linkedView()
            if lv is None:
                return
            return lv.mouseClickEvent(event)

    def resizeEvent(self, ev=None):
        if self.orientation in ("left", "right"):
            nudge = 5

            if self.minus is not None:
                br = self.minus.boundingRect()
                p = QtCore.QPointF(0, 0)
                if self.orientation == "left":
                    p.setY(5)
                    p.setX(nudge)
                elif self.orientation == "right":
                    p.setY(5)
                    p.setX(int(self.size().width() - br.height() + nudge))

                self.minus.setPos(p)

            if self.plus is not None:
                br = self.plus.boundingRect()
                p = QtCore.QPointF(0, 0)
                if self.orientation == "left":
                    p.setY(26)
                    p.setX(nudge)
                elif self.orientation == "right":
                    p.setY(26)
                    p.setX(int(self.size().width() - br.height() + nudge))

                self.plus.setPos(p)

        super().resizeEvent(ev)

    def close(self):
        if self.plus is not None:
            self.scene().removeItem(self.plus)
        if self.minus is not None:
            self.scene().removeItem(self.minus)
        self.plus = None
        self.minus = None

        super().close()

    def set_pen(self, pen=None):

        if pen is None:
            pen = fn.mkPen(pen)

        color = pen.color()

        if self.minus is not None:
            p = QtGui.QPainter(self.minus.pixmap)
            p.setBrush(color)
            p.drawRect(QtCore.QRect(0, 24, 64, 15))

        if self.plus is not None:
            p = QtGui.QPainter(self.plus.pixmap)
            p.setBrush(color)
            p.drawRect(QtCore.QRect(0, 24, 64, 15))
            p.drawRect(QtCore.QRect(24, 0, 15, 64))

        if pen is not self._pen:
            self.setPen(pen)

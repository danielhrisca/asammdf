# -*- coding: utf-8 -*-

from math import atan2, degrees

import pyqtgraph as pg
from PySide6 import QtCore, QtGui


class Cursor(pg.InfiniteLine):
    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs,
        )

        self.pen.setWidth(3)
        self.hoverPen.setWidth(3)

    def set_value(self, value):
        self.setPos(value)

    def paint(self, p, *args, skip=True, x_delta=0, height=0):
        if not skip:
            p.setRenderHint(p.RenderHint.Antialiasing)

            pen = self.currentPen
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
            p.setPen(pen)
            vb = self.getViewBox()
            xs = vb.state["viewRange"][0][0]
            x_scale, y_scale = vb.viewPixelSize()

            # x = (x - xs) / x_scale + x_delta
            # is rewriten as

            xs = xs - x_delta * x_scale

            x = (self.value() - xs) / x_scale

            p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            p.drawLine(pg.Point(x, 0), pg.Point(x, height))


class Region(pg.LinearRegionItem):
    def __init__(
        self,
        values=(0, 1),
        orientation="vertical",
        brush=None,
        pen=None,
        hoverBrush=None,
        hoverPen=None,
        movable=True,
        bounds=None,
        span=(0, 1),
        swapMode="sort",
        clipItem=None,
    ):
        pg.GraphicsObject.__init__(self)
        self.orientation = orientation
        self.blockLineSignal = False
        self.moving = False
        self.mouseHovering = False
        self.span = span
        self.swapMode = swapMode
        self.clipItem = clipItem

        self._boundingRectCache = None
        self._clipItemBoundsCache = None

        # note LinearRegionItem.Horizontal and LinearRegionItem.Vertical
        # are kept for backward compatibility.
        lineKwds = dict(
            movable=movable,
            bounds=bounds,
            span=span,
            pen=pen,
            hoverPen=hoverPen,
        )

        self.lines = [
            Cursor(QtCore.QPointF(values[0], 0), angle=90, **lineKwds),
            Cursor(QtCore.QPointF(values[1], 0), angle=90, **lineKwds),
        ]

        for l in self.lines:
            l.setParentItem(self)
            l.sigPositionChangeFinished.connect(self.lineMoveFinished)
        self.lines[0].sigPositionChanged.connect(self._line0Moved)
        self.lines[1].sigPositionChanged.connect(self._line1Moved)

        if brush is None:
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        self.setBrush(brush)

        if hoverBrush is None:
            c = self.brush.color()
            c.setAlpha(min(c.alpha() * 2, 255))
            hoverBrush = pg.functions.mkBrush(c)
        self.setHoverBrush(hoverBrush)

        self.setMovable(movable)

    def paint(self, p, *args, skip=True, x_delta=0, height=0):

        vb = self.getViewBox()
        xs = vb.state["viewRange"][0][0]
        x_scale, y_scale = vb.viewPixelSize()
        xs = xs - x_delta * x_scale

        rect = QtCore.QRectF(
            (self.lines[0].value() - xs) / x_scale,
            0,
            (self.lines[1].value() - self.lines[0].value()) / x_scale,
            height,
        )

        p.setBrush(self.currentBrush)
        p.setPen(pg.functions.mkPen(None))
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.drawRect(rect)
        for line in self.lines:
            line.paint(p, *args, skip=skip, x_delta=x_delta, height=height)

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

        self.setCursor(QtCore.Qt.SplitHCursor)
        self.sigDragged.connect(self.update_mouse_cursor)
        self.sigPositionChangeFinished.connect(self.update_mouse_cursor)

        self._cursor_override = False

    def update_mouse_cursor(self, obj):
        if self.moving:
            if not self._cursor_override:
                QtGui.QGuiApplication.setOverrideCursor(QtCore.Qt.SplitHCursor)
                self._cursor_override = True
        else:
            if self._cursor_override is not None:
                self._cursor_override = False
                QtGui.QGuiApplication.restoreOverrideCursor()

    def set_value(self, value):
        self.setPos(value)

    def paint(self, paint, *args, plot=None, uuid=None):
        if plot:
            paint.setRenderHint(paint.RenderHint.Antialiasing)

            pen = self.currentPen
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
            paint.setPen(pen)

            pen = self.pen
            paint.setPen(pen)

            position = self.value()

            if uuid:

                delta = plot.y_axis.width() + 1
                height = plot.y_axis.height() + plot.x_axis.height()
                width = delta + plot.x_axis.width()

                signal, idx = plot.signal_by_uuid(uuid)
                index = plot.get_timestamp_index(position, signal.timestamps)
                y_value, kind, fmt = signal.value_at_index(index)

                x, y = plot.scale_curve_to_pixmap(position, y_value)

                paint.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, y - 5))
                paint.drawLine(QtCore.QPointF(x, y + 5), QtCore.QPointF(x, height))

                pen.setWidth(1)
                paint.setPen(pen)

                paint.drawLine(QtCore.QPointF(delta, y), QtCore.QPointF(x - 5, y))
                paint.drawLine(QtCore.QPointF(x + 5, y), QtCore.QPointF(width, y))

                pen.setWidth(2)
                paint.setPen(pen)

                paint.drawEllipse(QtCore.QPointF(x, y), 5, 5)

            else:
                x, y = self.scale_curve_to_pixmap(position, 0)
                height = plot.y_axis.height() + plot.x_axis.height()
                paint.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, height))


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

    def paint(self, p, *args, plot=None, uuid=None):
        if plot:

            x1, y1 = plot.scale_curve_to_pixmap(self.lines[0].value(), 0)
            x2, y2 = plot.scale_curve_to_pixmap(self.lines[1].value(), 0)

            height = plot.y_axis.height() + plot.x_axis.height()

            rect = QtCore.QRectF(
                x1,
                0,
                x2 - x1,
                height,
            )

            p.setBrush(self.currentBrush)
            p.setPen(pg.functions.mkPen(None))
            p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
            p.drawRect(rect)
            for line in self.lines:
                line.paint(p, *args, plot=plot, uuid=uuid)

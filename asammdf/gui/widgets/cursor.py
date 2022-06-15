# -*- coding: utf-8 -*-

import pyqtgraph as pg
from PySide6 import QtCore, QtGui


class Cursor(pg.InfiniteLine):
    def __init__(
        self,
        *args,
        show_circle=True,
        show_horizontal_line=True,
        line_width=1,
        color="#ffffff",
        **kwargs
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.line_width = line_width
        self.color = color

        self.setCursor(QtCore.Qt.SplitHCursor)
        self.sigDragged.connect(self.update_mouse_cursor)
        self.sigPositionChangeFinished.connect(self.update_mouse_cursor)

        self._cursor_override = False
        self.show_circle = show_circle
        self.show_horizontal_line = show_horizontal_line

    @property
    def color(self):
        return self.pen.color().name()

    @color.setter
    def color(self, value):
        color = pg.mkColor(value)
        color.setAlpha(200)
        self.pen = QtGui.QPen(color.name())
        self.hoverPen = QtGui.QPen(color.name())
        self.update()

    @property
    def line_width(self):
        return self._line_width

    @line_width.setter
    def line_width(self, value):
        self._line_width = value
        self.update()

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
            paint.setRenderHint(paint.RenderHint.Antialiasing, False)

            pen = self.pen
            pen.setWidth(self.line_width)
            paint.setPen(pen)

            position = self.value()

            rect = plot.viewbox.sceneBoundingRect()
            delta = rect.x()
            height = rect.height()
            width = rect.x() + rect.width()

            if not self.show_circle and not self.show_horizontal_line or not uuid:
                x, y = plot.scale_curve_to_pixmap(
                    position,
                    0,
                    y_range=plot.viewbox.viewRange()[1],
                    x_start=plot.viewbox.viewRange()[0][0],
                    delta=delta,
                )
                paint.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, height))

            else:

                signal, idx = plot.signal_by_uuid(uuid)
                if signal.enable:
                    index = plot.get_timestamp_index(position, signal.timestamps)
                    y_value, kind, fmt = signal.value_at_index(index)
                    if y_value != "n.a.":

                        x, y = plot.scale_curve_to_pixmap(
                            position,
                            y_value,
                            y_range=signal.y_range,
                            x_start=plot.viewbox.viewRange()[0][0],
                            delta=delta,
                        )

                        if self.show_circle:
                            paint.drawLine(
                                QtCore.QPointF(x, 0), QtCore.QPointF(x, y - 5)
                            )
                            paint.drawLine(
                                QtCore.QPointF(x, y + 5), QtCore.QPointF(x, height)
                            )

                            if self.show_horizontal_line:
                                paint.drawLine(
                                    QtCore.QPointF(delta, y), QtCore.QPointF(x - 5, y)
                                )
                                paint.drawLine(
                                    QtCore.QPointF(x + 5, y), QtCore.QPointF(width, y)
                                )

                            paint.setRenderHints(paint.RenderHint.Antialiasing, True)
                            paint.drawEllipse(QtCore.QPointF(x, y), 5, 5)
                            paint.setRenderHints(paint.RenderHint.Antialiasing, False)

                        else:
                            paint.drawLine(
                                QtCore.QPointF(x, 0), QtCore.QPointF(x, height)
                            )
                            if self.show_horizontal_line:
                                paint.drawLine(
                                    QtCore.QPointF(delta, y), QtCore.QPointF(width, y)
                                )

                    else:
                        x, y = plot.scale_curve_to_pixmap(
                            position,
                            0,
                            y_range=plot.viewbox.viewRange()[1],
                            x_start=plot.viewbox.viewRange()[0][0],
                            delta=delta,
                        )
                        paint.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, height))
                else:
                    x, y = plot.scale_curve_to_pixmap(
                        position,
                        0,
                        y_range=plot.viewbox.viewRange()[1],
                        x_start=plot.viewbox.viewRange()[0][0],
                        delta=delta,
                    )
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
        show_circle=True,
        show_horizontal_line=True,
        line_width=1,
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
            show_circle=show_circle,
            show_horizontal_line=show_horizontal_line,
            line_width=line_width,
            color=pen,
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
            rect = plot.viewbox.sceneBoundingRect()
            delta = rect.x()
            height = rect.height()

            x1, y1 = plot.scale_curve_to_pixmap(
                self.lines[0].value(),
                0,
                y_range=plot.viewbox.viewRange()[1],
                x_start=plot.viewbox.viewRange()[0][0],
                delta=delta,
            )
            x2, y2 = plot.scale_curve_to_pixmap(
                self.lines[1].value(),
                0,
                y_range=plot.viewbox.viewRange()[1],
                x_start=plot.viewbox.viewRange()[0][0],
                delta=delta,
            )

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

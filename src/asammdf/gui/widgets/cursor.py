import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from ... import tool as Tool
from ...blocks.utils import escape_xml_string
from ..utils import BLUE


class Bookmark(pg.InfiniteLine):
    def __init__(self, message="", title="", color="#ffffff", tool="", **kwargs):
        self.title = title or "Bookmark"

        if message:
            text = f"{self.title}\nt = {kwargs['pos']}s\n\n{message}\n "
        else:
            text = f"{self.title}\nt = {kwargs['pos']}s\n "

        text = "\n".join([f"  {line}  " for line in text.splitlines()])

        super().__init__(
            movable=False,
            label=text,
            labelOpts={"movable": True},
            **kwargs,
        )

        self.line_width = 2
        self.color = color
        self._visible = True
        self._message = ""
        self.message = message

        if tool and tool == Tool.__tool__:
            self.editable = True
        else:
            self.editable = False

        self.edited = False
        self.deleted = False

        self.fill = pg.mkBrush(BLUE)
        self.border = pg.mkPen(
            {
                "color": color,
                "width": 2,
                "style": QtCore.Qt.PenStyle.DashLine,
            }
        )

    def __hash__(self):
        return hash((self.title, self.message))

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

    def _computeBoundingRect(self):
        # br = UIGraphicsItem.boundingRect(self)
        vr = self.viewRect()  # bounds of containing ViewBox mapped to local coords.
        if vr is None:
            return QtCore.QRectF()

        ## add a 6-pixel radius around the line for mouse interaction.

        px = self.pixelLength(direction=pg.Point(1, 0), ortho=True)  ## get pixel length orthogonal to the line
        if px is None:
            px = 0
        pw = max(self.pen.width() / 2, self.hoverPen.width() / 2)
        w = max(6, self._maxMarkerSize + pw) + 1
        w = w * px
        br = QtCore.QRectF(vr)
        br.setBottom(-w)
        br.setTop(w)

        length = br.width()
        left = br.left() + length * self.span[0]
        right = br.left() + length * self.span[1]
        br.setLeft(left)
        br.setRight(right)
        br = br.normalized()

        vs = self.getViewBox().size()

        if self._bounds != br or self._lastViewSize != vs:
            self._bounds = br
            self._lastViewSize = vs
            self.prepareGeometryChange()

        self._endPoints = (left, right)
        self._lastViewRect = vr

        return self._bounds

    @property
    def line_width(self):
        return self._line_width

    @line_width.setter
    def line_width(self, value):
        self._line_width = value
        self.update()

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, value):
        self._message = value
        if value:
            text = f"{self.title}\nt = {self.value()}s\n\n{value}\n "
        else:
            text = f"{self.title}\nt = {self.value()}s\n "
        text = "\n".join([f"  {line}  " for line in text.splitlines()])

        self.label.setPlainText(text)

    def paint(self, paint, *args, plot=None, uuid=None):
        if plot and self.visible:
            paint.setRenderHint(paint.RenderHint.Antialiasing, False)

            pen = self.pen
            pen.setWidth(self.line_width)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)

            paint.setPen(pen)

            position = self.value()

            rect = plot.viewbox.sceneBoundingRect()
            delta = rect.x()
            height = rect.height()

            px, py = plot.px, plot.py

            plot.px = (plot.x_range[1] - plot.x_range[0]) / rect.width()
            plot.py = rect.height()

            x, y = plot.scale_curve_to_pixmap(
                position,
                0,
                y_range=plot.viewbox.viewRange()[1],
                x_start=plot.viewbox.viewRange()[0][0],
                delta=delta,
            )
            paint.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, height))

            rect = self.label.textItem.sceneBoundingRect()

            black_pen = pg.mkPen("#000000")

            paint.setPen(self.border)
            paint.setBrush(self.fill)
            paint.setRenderHint(paint.RenderHint.Antialiasing, True)
            paint.drawRect(rect)

            paint.setPen(black_pen)

            message = f"{self.title}\nt = {self.value()}s\n\n{self.message}"

            delta = 5  # pixels
            paint.drawText(rect.adjusted(delta, delta, -2 * delta, -2 * delta), message)

            if self.editable:
                paint.setPen(black_pen)
                paint.setBrush(QtGui.QBrush(QtGui.QColor("#000000")))
                paint.setRenderHint(paint.RenderHint.Antialiasing, True)

                rect2 = QtCore.QRectF(
                    rect.x() + rect.width() - 35,
                    rect.y() + 1,
                    18,
                    18,
                )
                paint.drawRect(rect2)
                rect2 = QtCore.QRectF(
                    rect.x() + rect.width() - 18,
                    rect.y() + 1,
                    18,
                    18,
                )
                paint.drawRect(rect2)

                pix = QtGui.QPixmap(":/edit.png").scaled(16, 16)
                paint.drawPixmap(QtCore.QPointF(rect.x() + rect.width() - 34, rect.y() + 1), pix)

                pix = QtGui.QPixmap(":/erase.png").scaled(16, 16)
                paint.drawPixmap(QtCore.QPointF(rect.x() + rect.width() - 17, rect.y() + 1), pix)

            plot.px, plot.py = px, py

    def set_value(self, value):
        self.setPos(value)

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = bool(value)
        self.label.setVisible(self._visible)

    def xml_comment(self):
        return f"""<EVcomment>
    <TX>{escape_xml_string(self.message)}</TX>
    <tool>{Tool.__tool__}</tool>
</EVcomment>"""


class Cursor(pg.InfiniteLine):
    def __init__(
        self,
        *args,
        show_circle=True,
        show_horizontal_line=True,
        line_width=1,
        color="#ffffff",
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.line_width = line_width
        self.color = color

        # disable mouse cursor until https://github.com/pyqtgraph/pyqtgraph/issues/2416 is fixed
        # self.setCursor(QtCore.Qt.CursorShape.SplitHCursor)

        self.sigDragged.connect(self.update_mouse_cursor)
        self.sigPositionChangeFinished.connect(self.update_mouse_cursor)

        self._cursor_override = False
        self.show_circle = show_circle
        self.show_horizontal_line = show_horizontal_line
        self.locked = False

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
                QtGui.QGuiApplication.setOverrideCursor(QtCore.Qt.CursorShape.SplitHCursor)
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

            if self.mouseHovering and self.movable:
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            elif not self.locked:
                pen.setStyle(QtCore.Qt.PenStyle.SolidLine)
            else:
                pen.setStyle(QtCore.Qt.PenStyle.DashDotDotLine)

            paint.setPen(pen)

            position = self.value()

            rect = plot.viewbox.sceneBoundingRect()
            delta = rect.x()
            height = rect.height()
            width = rect.x() + rect.width()

            px, py = plot.px, plot.py

            plot.px = (plot.x_range[1] - plot.x_range[0]) / rect.width()
            plot.py = rect.height()

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
                            paint.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, y - 5))
                            paint.drawLine(QtCore.QPointF(x, y + 5), QtCore.QPointF(x, height))

                            if self.show_horizontal_line:
                                paint.drawLine(QtCore.QPointF(delta, y), QtCore.QPointF(x - 5, y))
                                paint.drawLine(QtCore.QPointF(x + 5, y), QtCore.QPointF(width, y))

                            paint.setRenderHints(paint.RenderHint.Antialiasing, True)
                            paint.drawEllipse(QtCore.QPointF(x, y), 5, 5)
                            paint.setRenderHints(paint.RenderHint.Antialiasing, False)

                        else:
                            paint.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, height))
                            if self.show_horizontal_line:
                                paint.drawLine(QtCore.QPointF(delta, y), QtCore.QPointF(width, y))

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

            plot.px, plot.py = px, py

    def _computeBoundingRect(self):
        # br = UIGraphicsItem.boundingRect(self)
        vr = self.viewRect()  # bounds of containing ViewBox mapped to local coords.
        if vr is None:
            return QtCore.QRectF()

        ## add a 6-pixel radius around the line for mouse interaction.

        px = self.pixelLength(direction=pg.Point(1, 0), ortho=True)  ## get pixel length orthogonal to the line
        if px is None:
            px = 0
        pw = max(self.pen.width() / 2, self.hoverPen.width() / 2)
        w = max(6, self._maxMarkerSize + pw) + 1
        w = w * px
        br = QtCore.QRectF(vr)
        br.setBottom(-w)
        br.setTop(w)

        length = br.width()
        left = br.left() + length * self.span[0]
        right = br.left() + length * self.span[1]
        br.setLeft(left)
        br.setRight(right)
        br = br.normalized()

        vs = self.getViewBox().size()

        if self._bounds != br or self._lastViewSize != vs:
            self._bounds = br
            self._lastViewSize = vs
            self.prepareGeometryChange()

        self._endPoints = (left, right)
        self._lastViewRect = vr

        return self._bounds


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
        lineKwds = {
            "movable": movable,
            "bounds": bounds,
            "span": span,
            "pen": pen,
            "hoverPen": hoverPen,
            "show_circle": show_circle,
            "show_horizontal_line": show_horizontal_line,
            "line_width": line_width,
            "color": pen,
        }

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

            px, py = plot.px, plot.py

            plot.px = (plot.x_range[1] - plot.x_range[0]) / rect.width()
            plot.py = rect.height()

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
            p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceAtop)
            p.drawRect(rect)
            for line in self.lines:
                line.paint(p, *args, plot=plot, uuid=uuid)

            plot.px, plot.py = px, py

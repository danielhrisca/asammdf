from datetime import datetime, timedelta, timezone
from traceback import format_exc

from PySide6 import QtCore, QtGui, QtWidgets

LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.functions as fn
from pyqtgraph.graphicsItems.ButtonItem import ButtonItem
from pyqtgraph.Point import Point

BUTTON_SIZE = 16


class FormatedAxis(pg.AxisItem):
    rangeChanged = QtCore.Signal(object, object)
    scale_editor_requested = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        self.plus = self.minus = None
        self.uuid = kwargs.pop("uuid", None)
        self.background = kwargs.pop("background", fn.mkColor("#000000"))
        self.linked_signal = kwargs.pop("linked_signal", None)

        super().__init__(*args, **kwargs)
        self.setStyle(tickAlpha=0.75)

        self._settings = QtCore.QSettings()

        self.format = "phys"
        self.mode = "phys"
        self.text_conversion = None
        self.origin = None

        self._label_with_unit = ""

        self.locked = kwargs.pop("locked", False)

        if self.orientation in ("left", "right"):
            self.plus = ButtonItem(":/plus.png", BUTTON_SIZE, parentItem=self)
            self.minus = ButtonItem(":/minus.png", BUTTON_SIZE, parentItem=self)

            self.plus.clicked.connect(self.increase_width)
            self.minus.clicked.connect(self.decrease_width)

            if self.scene() is not None:
                self.scene().addItem(self.plus)
                self.scene().addItem(self.minus)

            self.setWidth(48)

        self.set_pen(self._pen)

    def increase_width(self):
        self.setWidth(self.width() + 10)

    def decrease_width(self):
        self.setWidth(self.width() - 10)

    def setWidth(self, w=None):
        if self.orientation in ("left", "right"):
            super().setWidth(max(w, 48))
        else:
            super().setWidth(w)

    def tickStrings(self, values, scale, spacing):
        strns = []
        self.tick_positions = values

        if self.text_conversion and self.mode == "phys":
            strns = []
            for val in values:
                nv = self.text_conversion.convert(np.array([val]), as_bytes=True)[0]

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
                        strns.append(f'{val}={nv.decode("utf-8", errors="replace")}')
                    except:
                        strns.append(f'{val}={nv.decode("latin-1", errors="replace")}')
                else:
                    strns.append(val)
        else:
            if self.format == "phys":
                strns = super().tickStrings(values, scale, spacing)

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
            elif self.format == "ascii":
                for val in values:
                    val = float(val)
                    if val.is_integer():
                        val = int(val)
                        if 0 < val < 0x110000:
                            val = chr(val)
                        else:
                            val = str(val)
                    else:
                        val = ""
                    strns.append(val)
            elif self.format == "time":
                strns = [str(timedelta(seconds=val)) for val in values]
            elif self.format == "date":
                strns = (
                    pd.to_datetime(
                        np.array(values) + self.origin.timestamp(),
                        unit="s",
                        errors="coerce",
                    )
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

    def labelString(self):
        if self.labelUnits == "":
            if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
                units = ""
            else:
                units = "(x%g)" % (1.0 / self.autoSIPrefixScale)
        else:
            units = f"({self.labelUnitPrefix}{self.labelUnits})"

        s = f"{self.labelText} {units}"
        self._label_with_unit = s

        style = ";".join([f"{k}: {self.labelStyle[k]}" for k in self.labelStyle])

        lsbl = f"<span style='{style}'>{s}</span>"
        return lsbl

    def mouseDragEvent(self, event):
        if self.locked:
            return

        if self.orientation in ("left", "right"):
            ev = event
            ev.accept()

            pos = ev.pos()
            lastPos = ev.lastPos()
            dif = pos - lastPos
            dif = dif * -1

            if ev.button() in [
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.MiddleButton,
            ]:
                if self.orientation in ("left", "right"):
                    scale = (self.range[1] - self.range[0]) / self.sceneBoundingRect().height()
                    delta = scale * dif.y()
                else:
                    scale = (self.range[1] - self.range[0]) / self.sceneBoundingRect().width()
                    delta = scale * dif.x()

                self.setRange(self.range[0] - delta, self.range[1] - delta)

            elif ev.button() & QtCore.Qt.MouseButton.RightButton:
                mid = sum(self.range) / 2
                delta = self.range[-1] - self.range[0]

                if self.orientation in ("left", "right"):
                    if dif.y() > 0:
                        delta = 0.94 * delta
                    else:
                        delta = 1.06 * delta

                else:
                    if dif.x() > 0:
                        delta = 0.94 * delta
                    else:
                        delta = 1.06 * delta

                self.setRange(mid - delta / 2, mid + delta / 2)
        else:
            return self.linkedView().mouseDragEvent(event, axis=0, ignore_cursor=True)

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            if self.sceneBoundingRect().contains(event.scenePos()):
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

        return super().resizeEvent(ev)

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
            self.minus.pixmap.fill(color)
            p.setBrush(self.background)
            p.drawRect(QtCore.QRect(4, 24, 56, 15))
            p.end()

        if self.plus is not None:
            p = QtGui.QPainter(self.plus.pixmap)
            self.plus.pixmap.fill(color)
            p.setBrush(self.background)
            p.drawRect(QtCore.QRect(4, 24, 56, 15))
            p.drawRect(QtCore.QRect(24, 4, 15, 56))
            p.end()

        if pen is not self._pen:
            self.setPen(pen)

    def raiseContextMenu(self, ev):
        low, high = self.range

        if self.orientation in ("left", "right"):
            if self.linked_signal is None:
                axis = f"{self.linked_signal.name} Y"
            else:
                axis = "Y"
        else:
            axis = "X"

        menu = QtWidgets.QMenu()
        menu.addAction(f"Edit {axis} axis scaling")
        menu.addSeparator()
        menu.addAction("Apply new axis limits")
        menu.addSeparator()

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        widget.setLayout(layout)
        layout.addWidget(QtWidgets.QLabel("max:"))

        upper = QtWidgets.QDoubleSpinBox()
        upper.setDecimals(9)
        upper.setMinimum(-1e64)
        upper.setMaximum(1e35)
        upper.setValue(high)
        layout.addWidget(upper)

        a = QtWidgets.QWidgetAction(self)
        a.setDefaultWidget(widget)
        menu.addAction(a)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        widget.setLayout(layout)
        layout.addWidget(QtWidgets.QLabel("min:"))

        lower = QtWidgets.QDoubleSpinBox()
        lower.setDecimals(9)
        lower.setMinimum(-1e64)
        lower.setMaximum(1e35)
        lower.setValue(low)
        layout.addWidget(lower)

        a = QtWidgets.QWidgetAction(self)
        a.setDefaultWidget(widget)
        menu.addAction(a)

        action = menu.exec_(ev.screenPos().toPoint())

        if action is None:
            return

        elif action.text() == "Apply new axis limits":
            if self.orientation in ("left", "right"):
                self.setRange(lower.value(), upper.value())
            else:
                self.setRange(lower.value(), upper.value())

        elif action.text() == f"Edit {axis} axis scaling":
            self.scale_editor_requested.emit(self.uuid)

    def set_font_size(self, size):
        font = self.font()
        font.setPointSize(size)

        if self.orientation in ("top", "bottom"):
            metric = QtGui.QFontMetrics(font)
            height = max(metric.height() + 2, 18)
            self.setStyle(tickFont=font, tickTextHeight=height)
        else:
            self.setStyle(tickFont=font)
            self.setFont(font)

        self.label.setFont(font)

    def setRange(self, mn, mx):
        if mn > mx:
            mn, mx = mx, mn
        super().setRange(mn, mx)
        self.rangeChanged.emit(self.uuid, (mn, mx))

    def wheelEvent(self, event):
        if self.locked:
            return

        lv = self.linkedView()

        if lv is None:
            # this is one of the individual axis

            zoom_y_center_on_cursor = self._settings.value("zoom_y_center_on_cursor", False, type=bool)

            pos = event.pos()
            rect = self.boundingRect()

            if zoom_y_center_on_cursor:
                plot, uuid = self.linked_signal

                y_pos_val, sig_y_bottom, sig_y_top = plot.value_at_cursor(uuid)

                delta_proc = (y_pos_val - (sig_y_top + sig_y_bottom) / 2) / (sig_y_top - sig_y_bottom)
                shift = delta_proc * (sig_y_top - sig_y_bottom)
                sig_y_top, sig_y_bottom = sig_y_top + shift, sig_y_bottom + shift

                delta = sig_y_top - sig_y_bottom

                if event.delta() > 0:
                    end = sig_y_top - 0.165 * delta
                    start = sig_y_bottom + 0.165 * delta
                else:
                    end = sig_y_top + 0.165 * delta
                    start = sig_y_bottom - 0.165 * delta

                self.setRange(start, end)

            else:
                y_pos_val = ((rect.height() + rect.y()) - pos.y()) / rect.height() * (
                    self.range[-1] - self.range[0]
                ) + self.range[0]

                ratio = abs((pos.y() - (rect.height() + rect.y())) / rect.height())

                delta = self.range[-1] - self.range[0]

                if event.delta() > 0:
                    delta = 0.66 * delta
                else:
                    delta = 1.33 * delta

                start = y_pos_val - ratio * delta
                end = y_pos_val + (1 - ratio) * delta

                self.setRange(start, end)

            event.accept()
        else:
            # this is the main Y axis or the X axis
            if self.orientation in ("top", "bottom"):
                super().wheelEvent(event)
            else:
                # main Y axis

                if not lv.state["mouseEnabled"][1]:
                    event.ignore()
                else:
                    # the plot is not Y locked
                    pos = event.pos()
                    rect = self.boundingRect()

                    if self._settings.value("zoom_y_center_on_cursor", False, type=bool):
                        plot, uuid = self.linked_signal

                        y_pos_val, sig_y_bottom, sig_y_top = plot.value_at_cursor()

                        if isinstance(y_pos_val, (int, float)):
                            delta_proc = (y_pos_val - (sig_y_top + sig_y_bottom) / 2) / (sig_y_top - sig_y_bottom)
                            shift = delta_proc * (sig_y_top - sig_y_bottom)
                            sig_y_top, sig_y_bottom = sig_y_top + shift, sig_y_bottom + shift

                        delta = sig_y_top - sig_y_bottom

                        if event.delta() > 0:
                            end = sig_y_top - 0.165 * delta
                            start = sig_y_bottom + 0.165 * delta
                        else:
                            end = sig_y_top + 0.165 * delta
                            start = sig_y_bottom - 0.165 * delta

                        self.setRange(start, end)

                    else:
                        y_pos_val = ((rect.height() + rect.y()) - pos.y()) / rect.height() * (
                            self.range[-1] - self.range[0]
                        ) + self.range[0]

                        ratio = abs((pos.y() - (rect.height() + rect.y())) / rect.height())

                        delta = self.range[-1] - self.range[0]

                        if event.delta() > 0:
                            delta = 0.66 * delta
                        else:
                            delta = 1.33 * delta

                        start = y_pos_val - ratio * delta
                        end = y_pos_val + (1 - ratio) * delta

                        self.setRange(start, end)

                    event.accept()

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs, ratio=1.0):
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setRenderHint(p.RenderHint.TextAntialiasing, True)

        ## draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        # p.translate(0.5,0)  ## resolves some damn pixel ambiguity

        ## draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)

        # Draw all text
        if self.style["tickFont"] is not None:
            p.setFont(self.style["tickFont"])
        p.setPen(self.textPen())
        bounding = self.boundingRect().toAlignedRect()
        bounding.setSize(bounding.size() * ratio)
        bounding.moveTo(bounding.topLeft() * ratio)
        p.setClipRect(bounding)
        for rect, flags, text in textSpecs:
            p.drawText(rect, int(flags), text)

    def paint(self, p, opt, widget):
        rect = self.boundingRect()

        width = rect.width()
        ratio = widget.devicePixelRatio() if widget else 1.0
        if self.picture is None:
            try:
                picture = QtGui.QPixmap(int(width * ratio), int(rect.height() * ratio))
                picture.fill(self.background)

                painter = QtGui.QPainter()
                painter.begin(picture)

                if self.isVisible():
                    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
                    if self.style["tickFont"]:
                        painter.setFont(self.style["tickFont"])
                    specs = self.generateDrawSpecs(painter, ratio)

                    if specs is not None:
                        self.drawPicture(painter, *specs, ratio)

                    if self.minus is not None:
                        painter.drawPixmap(
                            QtCore.QPoint(int(rect.x() * ratio) + 5 * ratio, 6 * ratio),
                            self.minus.pixmap.scaled(BUTTON_SIZE * ratio, BUTTON_SIZE * ratio),
                        )
                        painter.drawPixmap(
                            QtCore.QPoint((int(rect.x()) + 5) * ratio, 27 * ratio),
                            self.plus.pixmap.scaled(BUTTON_SIZE * ratio, BUTTON_SIZE * ratio),
                        )

                    if self.orientation in ("left", "right"):
                        painter.setPen(self._pen)

                        label_rect = QtCore.QRectF(
                            1 * ratio,
                            1 * ratio,
                            rect.height() * ratio - (28 + BUTTON_SIZE) * ratio,
                            rect.width() * ratio,
                        )
                        painter.translate(rect.bottomLeft() * ratio)
                        painter.rotate(-90)

                        painter.setRenderHint(painter.RenderHint.TextAntialiasing, True)
                        painter.drawText(
                            label_rect,
                            QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
                            self._label_with_unit,
                        )
                        painter.rotate(90)
                        painter.resetTransform()

            except:
                print(format_exc())

            finally:
                painter.end()

            picture.setDevicePixelRatio(widget.devicePixelRatio() if widget else 1.0)

            self.picture = picture

    def generateDrawSpecs(self, p, ratio):
        """
        Calls tickValues() and tickStrings() to determine where and how ticks should
        be drawn, then generates from this a set of drawing commands to be
        interpreted by drawPicture().
        """
        bounds = self.mapRectFromParent(self.geometry())

        tickBounds = bounds

        if self.orientation == "left":
            span = (bounds.topRight(), bounds.bottomRight())
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == "right":
            span = (bounds.topLeft(), bounds.bottomLeft())
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == "top":
            span = (bounds.bottomLeft(), bounds.bottomRight())
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == "bottom":
            span = (bounds.topLeft(), bounds.topRight())
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1
        else:
            raise ValueError("self.orientation must be in ('left', 'right', 'top', 'bottom')")

        ## determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return

        # Determine major / minor / subminor axis ticks
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            ## parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)

        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        else:
            if axis == 0:
                xScale = -bounds.height() / dif
                offset = self.range[0] * xScale - bounds.height()
            else:
                xScale = bounds.width() / dif
                offset = self.range[0] * xScale

        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)

        tickPositions = []  # remembers positions of previously drawn ticks

        ## compute coordinates to draw ticks
        ## draw three different intervals, long ticks first
        tickSpecs = []
        for i in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i][1]

            ## length of tick
            tickLength = self.style["tickLength"] / ((i * 0.5) + 1.0)

            lineAlpha = self.style["tickAlpha"]
            if lineAlpha is None:
                lineAlpha = 255 / (i + 1)
                if self.grid is not False:
                    lineAlpha *= (
                        self.grid / 255.0 * fn.clip_scalar((0.05 * lengthInPixels / (len(ticks) + 1)), 0.0, 1.0)
                    )
            elif isinstance(lineAlpha, float):
                lineAlpha *= 255
                lineAlpha = max(0, int(round(lineAlpha)))
                lineAlpha = min(255, int(round(lineAlpha)))
            elif isinstance(lineAlpha, int):
                if (lineAlpha > 255) or (lineAlpha < 0):
                    raise ValueError("lineAlpha should be [0..255]")
            else:
                raise TypeError("Line Alpha should be of type None, float or int")

            for v in ticks:
                ## determine actual position to draw this tick
                x = (v * xScale) - offset
                if x < xMin or x > xMax:  ## last check to make sure no out-of-bounds ticks are drawn
                    tickPositions[i].append(None)
                    continue
                tickPositions[i].append(x)

                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                p2[axis] += tickLength * tickDir
                tickPen = self.pen()
                color = tickPen.color()
                color.setAlpha(int(lineAlpha))
                tickPen.setColor(color)
                tickSpecs.append((tickPen, Point(p1), Point(p2)))

        if self.style["stopAxisAtTick"][0] is True:
            minTickPosition = min(map(min, tickPositions))
            if axis == 0:
                stop = max(span[0].y(), minTickPosition)
                span[0].setY(stop)
            else:
                stop = max(span[0].x(), minTickPosition)
                span[0].setX(stop)
        if self.style["stopAxisAtTick"][1] is True:
            maxTickPosition = max(map(max, tickPositions))
            if axis == 0:
                stop = min(span[1].y(), maxTickPosition)
                span[1].setY(stop)
            else:
                stop = min(span[1].x(), maxTickPosition)
                span[1].setX(stop)
        axisSpec = (self.pen(), span[0], span[1])

        textOffset = self.style["tickTextOffset"][axis]  ## spacing between axis and text
        # if self.style['autoExpandTextSpace'] is True:
        # textWidth = self.textWidth
        # textHeight = self.textHeight
        # else:
        # textWidth = self.style['tickTextWidth'] ## space allocated for horizontal text
        # textHeight = self.style['tickTextHeight'] ## space allocated for horizontal text

        textSize2 = 0
        lastTextSize2 = 0
        textRects = []
        textSpecs = []  ## list of draw

        # If values are hidden, return early
        if not self.style["showValues"]:
            return (axisSpec, tickSpecs, textSpecs)

        for i in range(min(len(tickLevels), self.style["maxTextLevel"] + 1)):
            ## Get the list of strings to display for this level
            if tickStrings is None:
                spacing, values = tickLevels[i]
                strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = tickStrings[i]

            if len(strings) == 0:
                continue

            ## ignore strings belonging to ticks that were previously ignored
            for j in range(len(strings)):
                if tickPositions[i][j] is None:
                    strings[j] = None

            ## Measure density of text; decide whether to draw this level
            rects = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    br = p.boundingRect(
                        QtCore.QRectF(0, 0, 100, 100),
                        QtCore.Qt.AlignmentFlag.AlignCenter,
                        s,
                    )
                    ## boundingRect is usually just a bit too large
                    ## (but this probably depends on per-font metrics?)
                    br.setHeight(br.height() * 0.8)

                    rects.append(br)
                    textRects.append(rects[-1])

            if len(textRects) > 0:
                ## measure all text, make sure there's enough room
                if axis == 0:
                    textSize = np.sum([r.height() for r in textRects])
                    textSize2 = np.max([r.width() for r in textRects])
                else:
                    textSize = np.sum([r.width() for r in textRects])
                    textSize2 = np.max([r.height() for r in textRects])
            else:
                textSize = 0
                textSize2 = 0

            if i > 0:  ## always draw top level
                ## If the strings are too crowded, stop drawing text now.
                ## We use three different crowding limits based on the number
                ## of texts drawn so far.
                textFillRatio = float(textSize) / lengthInPixels
                finished = False
                for nTexts, limit in self.style["textFillLimits"]:
                    if len(textSpecs) >= nTexts and textFillRatio >= limit:
                        finished = True
                        break
                if finished:
                    break

            lastTextSize2 = textSize2

            # spacing, values = tickLevels[best]
            # strings = self.tickStrings(values, self.scale, spacing)
            # Determine exactly where tick text should be drawn
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None:  ## this tick was ignored because it is out of bounds
                    continue
                x = tickPositions[i][j]
                # textRect = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignmentFlag.AlignCenter, vstr)
                textRect = rects[j]
                height = textRect.height()
                width = textRect.width()
                # self.textHeight = height
                offset = max(0, self.style["tickLength"]) + textOffset

                if self.orientation == "left":
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                    rect = QtCore.QRectF(tickStop - offset - width, x - (height / 2), width, height)
                elif self.orientation == "right":
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
                    rect = QtCore.QRectF(tickStop + offset, x - (height / 2), width, height)
                elif self.orientation == "top":
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignBottom
                    rect = QtCore.QRectF(x - width / 2.0, tickStop - offset - height, width, height)
                elif self.orientation == "bottom":
                    alignFlags = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop
                    rect = QtCore.QRectF(x - width / 2.0, tickStop + offset, width, height)

                textFlags = alignFlags | QtCore.Qt.TextFlag.TextDontClip
                # p.setPen(self.pen())
                # p.drawText(rect, textFlags, vstr)
                textSpecs.append((rect, textFlags, vstr))

        ## update max text size if needed.
        self._updateMaxTextSize(lastTextSize2)

        self.tickSpecs = tickSpecs

        axisSpec = (
            axisSpec[0],
            axisSpec[1] * ratio,
            axisSpec[2] * ratio,
        )

        bounds.setSize(bounds.size() * ratio)

        for spec in textSpecs:
            spec[0].setSize(spec[0].size() * ratio)
            spec[0].moveTo(spec[0].topLeft() * ratio)

        for i, spec in enumerate(tickSpecs):
            tickSpecs[i] = (
                spec[0],
                spec[1] * ratio,
                spec[2] * ratio,
            )

        return (axisSpec, tickSpecs, textSpecs)

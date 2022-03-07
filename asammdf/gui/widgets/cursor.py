# -*- coding: utf-8 -*-

import pyqtgraph as pg
from math import atan2, degrees

from PySide6 import QtCore, QtGui


class Cursor(pg.InfiniteLine):
    def __init__(self, cursor_unit="s", *args, **kwargs):

        super().__init__(
            *args,
            label=f"{{value:.6f}}{cursor_unit}",
            labelOpts={"position": 0.04},
            **kwargs,
        )

        self.pen.setWidth(3)
        self.hoverPen.setWidth(3)

        # self.addMarker("^", 0)
        # self.addMarker("v", 1)

        self._settings = QtCore.QSettings()
        if self._settings.value("plot_background") == "White":
            self.label.setColor(QtGui.QColor(0, 59, 126))
        else:
            self.label.setColor(QtGui.QColor("#ffffff"))

        self.label.show()

    def set_value(self, value):
        self.setPos(value)

    def paint(self, p, *args, skip=True, delta=0, height=0):
        print('paint cursor ')
        if not skip:
            p.setRenderHint(p.RenderHint.Antialiasing)

            left, right = self._endPoints
            pen = self.currentPen
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
            p.setPen(pen)
            vb = self.getViewBox()
            (xs, _1), (_2, ys) = vb.state["viewRange"]
            x_scale, y_scale = vb.viewPixelSize()

            # x = (x - xs) / x_scale + delta
            # y = (ys - y) / y_scale + 1
            # is rewriten as

            xs = xs - delta * x_scale

            x = (self.value() - xs) / x_scale

            print('draw', (self.value(), x), pen.width(), pen.color().name())
            p.drawLine(pg.Point(x, 0), pg.Point(x, height))
            #
            # if len(self.markers) == 0:
            #     return
            #
            # # paint markers in native coordinate system
            # tr = p.transform()
            # p.resetTransform()
            #
            # start = tr.map(pg.Point(left, 0))
            # end = tr.map(pg.Point(right, 0))
            # up = tr.map(pg.Point(left, 1))
            # dif = end - start
            # length = pg.Point(dif).length()
            # angle = degrees(atan2(dif.y(), dif.x()))
            #
            # p.translate(start)
            # p.rotate(angle)
            #
            # up = up - start
            # det = up.x() * dif.y() - dif.x() * up.y()
            # p.scale(1, 1 if det > 0 else -1)
            #
            # p.setBrush(pg.functions.mkBrush(self.currentPen.color()))
            # # p.setPen(fn.mkPen(None))
            # tr = p.transform()
            # for path, pos, size in self.markers:
            #     p.setTransform(tr)
            #     x = length * pos
            #     p.translate(x, 0)
            #     p.scale(size, size)
            #     p.drawPath(path)
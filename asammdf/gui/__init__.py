try:
    import numpy as np
    from PyQt5 import QtCore
    from pyqtgraph import fn, ScatterPlotItem

    def updateSpots(self, dataSet=None):

        if dataSet is None:
            dataSet = self.data

        invalidate = False
        if self.opts["pxMode"]:
            invalidate = True

            size, symbol, pen, brush = (
                self.opts["size"],
                self.opts["symbol"],
                fn.mkPen(self.opts["pen"]),
                fn.mkBrush(self.opts["brush"]),
            )
            newRectSrc = QtCore.QRectF()
            newRectSrc.pen = pen
            newRectSrc.brush = brush
            newRectSrc.symbol = symbol

            self.fragmentAtlas.symbolMap[
                (symbol, size, id(pen), id(brush))
            ] = newRectSrc
            self.fragmentAtlas.atlasValid = False

            source_rect = np.full(len(dataSet), newRectSrc, dtype="O")
            dataSet["sourceRect"] = source_rect

            self.fragmentAtlas.getAtlas()  # generate atlas so source widths are available.

            dataSet["width"] = np.array(list(map(QtCore.QRectF.width, source_rect))) / 2
            dataSet["targetRect"] = None
            self._maxSpotPxWidth = self.fragmentAtlas.max_width
        else:
            self._maxSpotWidth = 0
            self._maxSpotPxWidth = 0
            self.measureSpotSizes(dataSet)

        if invalidate:
            self.invalidate()

    ScatterPlotItem.updateSpots = updateSpots

except ImportError:
    pass

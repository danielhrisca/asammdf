# -*- coding: utf-8 -*-
import logging

import numpy as np

from ..ui import resource_qt5 as resource_rc


bin_ = bin

try:
    import pyqtgraph as pg

    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *

    from .channel_stats import ChannelStats
    from .plot import Plot

    if not hasattr(pg.InfiniteLine, "addMarker"):
        logger = logging.getLogger("asammdf")
        message = (
            "Old pyqtgraph package: Please install the latest pyqtgraph from the "
            "github develop branch\n"
            "pip install -I --no-deps "
            "https://github.com/pyqtgraph/pyqtgraph/archive/develop.zip"
        )
        logger.warning(message)

    class StandalonePlot(QWidget):
        def __init__(self, signals, with_dots, step_mode, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.splitter = QSplitter(self)
            self.splitter.setOrientation(Qt.Horizontal)
            self.info = None

            self.plot = Plot(signals, with_dots, standalone=True)
            self.splitter.addWidget(self.plot)

            self.plot.range_modified.connect(self.range_modified)
            self.plot.range_removed.connect(self.range_removed)
            self.plot.range_modified_finished.connect(self.range_modified_finished)
            self.plot.cursor_removed.connect(self.cursor_removed)
            self.plot.cursor_moved.connect(self.cursor_moved)
            self.plot.cursor_move_finished.connect(self.cursor_move_finished)
            self.plot.xrange_changed.connect(self.xrange_changed)

            vbox = QVBoxLayout()

            vbox.addWidget(self.splitter, 1)

            hbox = QHBoxLayout()

            for icon, description in (
                (":/cursor.png", "C - Cursor"),
                (":/fit.png", "F - Fit"),
                (":/grid.png", "G - Grid"),
                (":/home.png", "H - Home"),
                (":/zoom-in.png", "I - Zoom-in"),
                (":/zoom-out.png", "O - Zoom-out"),
                (":/info.png", "M - Statistics"),
                (":/range.png", "R - Range"),
                (":/right.png", "← - Move cursor left"),
                (":/left.png", "→ - Move cursor right"),
            ):
                label = QLabel("")
                label.setPixmap(QPixmap(icon).scaled(QSize(16, 16)))

                hbox.addWidget(label)
                label = QLabel(description)
                hbox.addWidget(label)
                hbox.addStretch()

            vbox.addLayout(hbox, 0)
            self.setLayout(vbox)

            icon = QIcon()
            icon.addPixmap(QPixmap(":/info.png"), QIcon.Normal, QIcon.Off)
            self.setWindowIcon(icon)

            self.show()

        def keyPressEvent(self, event):
            key = event.key()

            if key == Qt.Key_M:
                if self.info is None:
                    self.info = ChannelStats()
                    self.splitter.addWidget(self.info)
                    stats = self.plot.get_stats(0)
                    self.info.set_stats(stats)
                else:
                    self.info.setParent(None)
                    self.info.hide()
                    self.info = None
            else:
                super().keyPressEvent(event)

        def cursor_moved(self):
            position = self.plot.cursor1.value()

            x = self.plot.timebase

            if x is not None and len(x):
                dim = len(x)
                position = self.plot.cursor1.value()

                right = np.searchsorted(x, position, side="right")
                if right == 0:
                    next_pos = x[0]
                elif right == dim:
                    next_pos = x[-1]
                else:
                    if position - x[right - 1] < x[right] - position:
                        next_pos = x[right - 1]
                    else:
                        next_pos = x[right]

                y = []

                _, (hint_min, hint_max) = self.plot.viewbox.viewRange()

                for viewbox, sig, curve in zip(
                    self.plot.view_boxes, self.plot.signals, self.plot.curves
                ):
                    if curve.isVisible():
                        index = np.argwhere(sig.timestamps == next_pos).flatten()
                        if len(index):
                            _, (y_min, y_max) = viewbox.viewRange()

                            sample = sig.samples[index[0]]
                            sample = (sample - y_min) / (y_max - y_min) * (
                                hint_max - hint_min
                            ) + hint_min

                            y.append(sample)

                if self.plot.curve.isVisible():
                    timestamps = self.plot.curve.xData
                    samples = self.plot.curve.yData
                    index = np.argwhere(timestamps == next_pos).flatten()
                    if len(index):
                        _, (y_min, y_max) = self.plot.viewbox.viewRange()

                        sample = samples[index[0]]
                        sample = (sample - y_min) / (y_max - y_min) * (
                            hint_max - hint_min
                        ) + hint_min

                        y.append(sample)

                self.plot.viewbox.setYRange(hint_min, hint_max, padding=0)
                self.plot.cursor_hint.setData(x=[next_pos] * len(y), y=y)
                self.plot.cursor_hint.show()

            if self.info:
                stats = self.plot.get_stats(0)
                self.info.set_stats(stats)

        def cursor_move_finished(self):
            x = self.plot.timebase

            if x is not None and len(x):
                dim = len(x)
                position = self.plot.cursor1.value()

                right = np.searchsorted(x, position, side="right")
                if right == 0:
                    next_pos = x[0]
                elif right == dim:
                    next_pos = x[-1]
                else:
                    if position - x[right - 1] < x[right] - position:
                        next_pos = x[right - 1]
                    else:
                        next_pos = x[right]
                self.plot.cursor1.setPos(next_pos)

            self.plot.cursor_hint.setData(x=[], y=[])

        def range_modified(self):
            if self.info:
                stats = self.plot.get_stats(0)
                self.info.set_stats(stats)

        def xrange_changed(self):
            if self.info:
                stats = self.plot.get_stats(0)
                self.info.set_stats(stats)

        def range_modified_finished(self):
            start, stop = self.plot.region.getRegion()

            if self.plot.timebase is not None and len(self.plot.timebase):
                timebase = self.plot.timebase
                dim = len(timebase)

                right = np.searchsorted(timebase, start, side="right")
                if right == 0:
                    next_pos = timebase[0]
                elif right == dim:
                    next_pos = timebase[-1]
                else:
                    if start - timebase[right - 1] < timebase[right] - start:
                        next_pos = timebase[right - 1]
                    else:
                        next_pos = timebase[right]
                start = next_pos

                right = np.searchsorted(timebase, stop, side="right")
                if right == 0:
                    next_pos = timebase[0]
                elif right == dim:
                    next_pos = timebase[-1]
                else:
                    if stop - timebase[right - 1] < timebase[right] - stop:
                        next_pos = timebase[right - 1]
                    else:
                        next_pos = timebase[right]
                stop = next_pos

                self.plot.region.setRegion((start, stop))

        def range_removed(self):
            if self.info:
                stats = self.plot.get_stats(0)
                self.info.set_stats(stats)

        def cursor_removed(self):
            if self.info:
                stats = self.plot.get_stats(0)
                self.info.set_stats(stats)


except ImportError:
    PYQTGRAPH_AVAILABLE = False

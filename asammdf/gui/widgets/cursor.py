# -*- coding: utf-8 -*-

import pyqtgraph as pg


class Cursor(pg.InfiniteLine):
    def __init__(self, *args, **kwargs):

        super().__init__(
            *args, label="{value:.6f}s", labelOpts={"position": 0.04}, **kwargs
        )

        self.addMarker("^", 0)
        self.addMarker("v", 1)

        self.label.show()

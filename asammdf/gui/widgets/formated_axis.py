# -*- coding: utf-8 -*-

import pyqtgraph as pg
import numpy as np


class FormatedAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):

        super(FormatedAxis, self).__init__(*args, **kwargs)

        self.format = "phys"
        self.text_conversion = None

    def tickStrings(self, values, scale, spacing):
        strns = []

        if self.format == "phys":
            strns = super(FormatedAxis, self).tickStrings(values, scale, spacing)
            if self.text_conversion:
                strns = self.text_conversion.convert(np.array(values))
                try:
                    strns = [s.decode("utf-8") for s in strns]
                except:
                    strns = [s.decode("latin-1") for s in strns]

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

        return strns

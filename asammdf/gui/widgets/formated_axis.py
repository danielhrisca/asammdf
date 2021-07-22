# -*- coding: utf-8 -*-

from datetime import datetime, timedelta, timezone
from textwrap import wrap

LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo

import numpy as np
import pandas as pd
import pyqtgraph as pg


class FormatedAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.format = "phys"
        self.mode = "phys"
        self.text_conversion = None
        self.origin = None

    def tickStrings(self, values, scale, spacing):
        strns = []

        if self.text_conversion and self.mode == "phys":
            strns = []
            for val in values:
                nv = self.text_conversion.convert(np.array([val]))[0]

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
                        strns.append(f'{val}={nv.decode("utf-8")}')
                    except:
                        strns.append(f'{val}={nv.decode("latin-1")}')
                else:

                    strns.append(val)
        else:
            if self.format == "phys":
                strns = super(FormatedAxis, self).tickStrings(values, scale, spacing)

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
            elif self.format == "time":
                strns = [str(timedelta(seconds=val)) for val in values]
            elif self.format == "date":
                strns = (
                    pd.to_datetime(np.array(values) + self.origin.timestamp(), unit="s")
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

    def mouseDragEvent(self, event):
        if self.linkedView() is None:
            return
        if self.orientation in ["left", "right"]:
            return self.linkedView().mouseDragEvent(event)
        else:
            return self.linkedView().mouseDragEvent(event)

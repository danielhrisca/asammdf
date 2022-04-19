# -*- coding: utf-8 -*-
import bisect
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from functools import lru_cache, partial, reduce
import logging
import os
from pathlib import Path
from threading import Lock
from time import perf_counter, sleep, time
from traceback import format_exc
import weakref

import numpy as np
import pyqtgraph as pg
import pyqtgraph.canvas.CanvasTemplate_pyside6
import pyqtgraph.canvas.TransformGuiTemplate_pyside6
import pyqtgraph.console.template_pyside6

# imports for pyinstaller
import pyqtgraph.functions as fn
import pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyside6
import pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyside6
import pyqtgraph.GraphicsScene.exportDialogTemplate_pyside6
import pyqtgraph.imageview.ImageViewTemplate_pyside6
from PySide6 import QtCore, QtGui, QtWidgets

PLOT_BUFFER_SIZE = 4000

from ...blocks.utils import target_byte_order
from ..dialogs.range_editor import RangeEditor
from ..utils import FONT_SIZE

try:
    from ...blocks.cutils import positions
except:
    pass


@lru_cache(maxsize=1024)
def polygon_and_ndarray(size):
    polygon = fn.create_qpolygonf(size)
    ndarray = fn.ndarray_from_qpolygonf(polygon)
    return polygon, ndarray


def monkey_patch_pyqtgraph():
    def _keys(self, styles):
        def getId(obj):
            try:
                return obj._id
            except AttributeError:
                obj._id = next(
                    pg.graphicsItems.ScatterPlotItem.SymbolAtlas._idGenerator
                )
                return obj._id

        res = [
            (
                symbol if isinstance(symbol, (str, int)) else getId(symbol),
                size,
                getId(pen),
                getId(brush),
            )
            for symbol, size, pen, brush in styles[:1]
        ]

        return res

    # fixes https://github.com/pyqtgraph/pyqtgraph/issues/2117
    def mouseReleaseEvent(self, ev):
        if self.mouseGrabberItem() is None:
            if ev.button() in self.dragButtons:
                if self.sendDragEvent(ev, final=True):
                    ev.accept()
                self.dragButtons.remove(ev.button())
            else:
                cev = [e for e in self.clickEvents if e.button() == ev.button()]
                if cev:
                    if self.sendClickEvent(cev[0]):
                        ev.accept()
                    try:
                        self.clickEvents.remove(cev[0])
                    except:
                        pass

        if not ev.buttons():
            self.dragItem = None
            self.dragButtons = []
            self.clickEvents = []
            self.lastDrag = None
        QtWidgets.QGraphicsScene.mouseReleaseEvent(self, ev)
        self.sendHoverEvents(ev)

    mkColor_factory = fn.mkColor
    mkBrush_factory = fn.mkBrush
    mkPen_factory = fn.mkPen

    def mkColor(*args):
        try:
            return cached_mkColor_factory(*args)
        except:
            return mkColor_factory(*args)

    @lru_cache(maxsize=512)
    def cached_mkColor_factory(*args):
        return mkColor_factory(*args)

    def mkBrush(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], QtGui.QBrush):
            return args[0]
        try:
            return cached_mkBrush_factory(*args, **kwargs)
        except:
            return mkBrush_factory(*args, **kwargs)

    @lru_cache(maxsize=512)
    def cached_mkBrush_factory(*args, **kargs):
        return mkBrush_factory(*args, **kargs)

    def mkPen(*args, **kwargs):
        try:
            return cached_mkPen_factory(*args, **kwargs)
        except:
            return mkPen_factory(*args, **kwargs)

    @lru_cache(maxsize=512)
    def cached_mkPen_factory(*args, **kargs):
        return mkPen_factory(*args, **kargs)

    # speed-up monkey patches
    pg.graphicsItems.ScatterPlotItem.SymbolAtlas._keys = _keys
    pg.graphicsItems.ScatterPlotItem._USE_QRECT = False
    pg.GraphicsScene.mouseReleaseEvent = mouseReleaseEvent

    fn.mkBrush = mkBrush
    fn.mkColor = mkColor
    fn.mkPen = mkPen


from ...mdf import MDF
from ...signal import Signal
from ..dialogs.define_channel import DefineChannel
from ..ui import resource_rc
from ..utils import COLORS, copy_ranges, extract_mime_names
from .channel_stats import ChannelStats
from .cursor import Cursor, Region
from .dict_to_tree import ComputedChannelInfoWindow
from .formated_axis import FormatedAxis
from .list import ListWidget
from .list_item import ListItem
from .tree import ChannelsTreeItem, ChannelsTreeWidget

bin_ = bin

HERE = Path(__file__).resolve().parent

NOT_FOUND = 0xFFFFFFFF

float64 = np.float64


def simple_min(a, b):
    if b != b:
        return a
    if a <= b:
        return a
    return b


def simple_max(a, b):
    if b != b:
        return a
    if a <= b:
        return b
    return a


def get_descriptions_by_uuid(mime):
    descriptions = {}
    if mime:

        for item in mime:
            descriptions[item["uuid"]] = item
            if item.get("type", "channel") == "group":

                descriptions.update(get_descriptions_by_uuid(item["channels"]))

    return descriptions


class Scatter(pg.ScatterPlotItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exportOpts = {
            "antialias": True,
            "resolutionScale": 0.6,
        }

    def _maskAt(self, obj):
        return np.full(len(self.data["visible"]), True, dtype=bool)


class PlotSignal(Signal):
    def __init__(self, signal, index=0, trim_info=None, duplication=1):
        super().__init__(
            signal.samples,
            signal.timestamps,
            signal.unit,
            signal.name,
            signal.conversion,
            signal.comment,
            signal.raw,
            signal.master_metadata,
            signal.display_names,
            signal.attachment,
            signal.source,
            signal.bit_count,
            signal.stream_sync,
            invalidation_bits=signal.invalidation_bits,
            encoding=signal.encoding,
        )

        self._pos = np.empty(2 * PLOT_BUFFER_SIZE, dtype="i4")
        self._plot_samples = np.empty(2 * PLOT_BUFFER_SIZE, dtype="i1")
        self._plot_timestamps = np.empty(2 * PLOT_BUFFER_SIZE, dtype="f8")

        self._dtype = "i1"

        self.duplication = duplication
        self.uuid = getattr(signal, "uuid", os.urandom(6).hex())
        self.origin_uuid = getattr(signal, "origin_uuid", os.urandom(6).hex())

        self.group_index = getattr(signal, "group_index", NOT_FOUND)
        self.channel_index = getattr(signal, "channel_index", NOT_FOUND)
        self.precision = getattr(signal, "precision", 6)

        self._mode = "raw"
        self._enable = True

        self.format = getattr(signal, "format", "phys")

        self.individual_axis = False
        self.computed = signal.computed
        self.computation = signal.computation

        self.y_link = False
        self.y_range = (0, -1)

        self.trim_info = None

        # take out NaN values
        samples = self.samples
        if samples.dtype.kind not in "SUV":
            nans = np.isnan(samples)
            if np.any(nans):
                self.samples = self.samples[~nans]
                self.timestamps = self.timestamps[~nans]

        if self.samples.dtype.byteorder not in target_byte_order:
            self.samples = self.samples.byteswap().newbyteorder()

        if self.timestamps.dtype.byteorder not in target_byte_order:
            self.timestamps = self.timestamps.byteswap().newbyteorder()

        if self.timestamps.dtype != float64:
            self.timestamps = self.timestamps.astype(float64)

        if self.conversion:
            samples = self.conversion.convert(self.samples)
            if samples.dtype.kind not in "SUV":
                nans = np.isnan(samples)
                if np.any(nans):
                    self.raw_samples = self.samples[~nans]
                    self.phys_samples = samples[~nans]
                    self.timestamps = self.timestamps[~nans]
                    self.samples = self.samples[~nans]
                else:
                    self.raw_samples = self.samples
                    self.phys_samples = samples
            else:
                self.phys_samples = self.raw_samples = self.samples
        else:
            self.phys_samples = self.raw_samples = self.samples

        self.plot_samples = self.phys_samples
        self.plot_timestamps = self.timestamps

        self._stats = {
            "range": (0, -1),
            "range_stats": {},
            "visible": (0, -1),
            "visible_stats": {},
            "fmt": "",
        }

        if getattr(signal, "color", None):
            color = signal.color or COLORS[index % 10]
        else:
            color = COLORS[index % 10]
        self.color = fn.mkColor(color)
        self.color_name = self.color.name()
        self.pen = fn.mkPen(color=color, style=QtCore.Qt.SolidLine)

        if len(self.phys_samples):

            if self.raw_samples.dtype.kind in "SUV":
                self._min_raw = ""
                self._max_raw = ""
                self._avg_raw = ""
                self._rms_raw = ""
                self._std_raw = ""
            else:

                samples = self.raw_samples[np.isfinite(self.raw_samples)]
                if len(samples):
                    self._min_raw = np.nanmin(samples)
                    self._max_raw = np.nanmax(samples)
                    self._avg_raw = np.mean(samples)
                    self._rms_raw = np.sqrt(np.mean(np.square(samples)))
                    self._std_raw = np.std(samples)
                else:
                    self._min_raw = "n.a."
                    self._max_raw = "n.a."
                    self._avg_raw = "n.a."
                    self._rms_raw = "n.a."
                    self._std_raw = "n.a."

            if self.phys_samples is self.raw_samples:
                if self.phys_samples.dtype.kind in "SUV":
                    self.is_string = True
                else:
                    self.is_string = False

                self._min = self._min_raw
                self._max = self._max_raw
                self._avg = self._avg_raw
                self._rms = self._rms_raw
                self._std = self._std_raw

            else:
                if self.phys_samples.dtype.kind in "SUV":
                    self.is_string = True
                    self._min = ""
                    self._max = ""
                    self._avg = ""
                    self._rms = ""
                    self._std = ""
                else:
                    self.is_string = False
                    samples = self.phys_samples[np.isfinite(self.phys_samples)]
                    if len(samples):
                        self._min = np.nanmin(samples)
                        self._max = np.nanmax(samples)
                        self._avg = np.mean(samples)
                        self._rms = np.sqrt(np.mean(np.square(samples)))
                        self._std = np.std(samples)
                    else:
                        self._min = "n.a."
                        self._max = "n.a."
                        self._avg = "n.a."
                        self._rms = "n.a."
                        self._std = "n.a."

            self.empty = False

        else:
            self.empty = True
            if self.phys_samples.dtype.kind in "SUV":
                self.is_string = True
                self._min = ""
                self._max = ""
                self._rms = ""
                self._avg = ""
                self._std = ""
                self._min_raw = ""
                self._max_raw = ""
                self._avg_raw = ""
                self._rms_raw = ""
                self._std_raw = ""
            else:
                self.is_string = False
                self._min = "n.a."
                self._max = "n.a."
                self._rms = "n.a."
                self._avg = "n.a."
                self._std = "n.a."
                self._min_raw = "n.a."
                self._max_raw = "n.a."
                self._avg_raw = "n.a."
                self._rms_raw = "n.a."
                self._std_raw = "n.a."

        self.mode = getattr(signal, "mode", "phys")
        self.trim(*(trim_info or (None, None, 1900)))

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, enable_state):
        if self._enable != enable_state:
            self._enable = enable_state
            if enable_state:
                self._pos = np.empty(2 * PLOT_BUFFER_SIZE, dtype="i4")
                self._plot_samples = np.empty(2 * PLOT_BUFFER_SIZE, dtype=self._dtype)
                self._plot_timestamps = np.empty(2 * PLOT_BUFFER_SIZE, dtype="f8")

            else:
                self._pos = self._plot_samples = self._plot_timestamps = None

    def set_color(self, color):
        self.color = color
        self.pen = fn.mkPen(color=color, style=QtCore.Qt.SolidLine)

    @property
    def min(self):
        if self._mode == "phys":
            _min = self._min
            samples = self.phys_samples
        else:
            _min = self._min_raw
            samples = self.raw_samples

        if _min is not None:
            return _min
        else:
            if samples.dtype.kind in "SUV":
                return ""
            else:
                if len(samples):
                    return np.nanmin(samples)
                else:
                    return "n.a."

    @property
    def max(self):
        if self._mode == "phys":
            _max = self._max
            samples = self.phys_samples
        else:
            _max = self._max_raw
            samples = self.raw_samples

        if _max is not None:
            return _max
        else:
            if samples.dtype.kind in "SUV":
                return ""
            else:
                if len(samples):
                    return np.nanmax(samples)
                else:
                    return "n.a."

    @property
    def avg(self):
        return self._avg if self._mode == "phys" else self._avg_raw

    @avg.setter
    def avg(self, avg):
        self._avg = avg

    @property
    def rms(self):
        return self._rms if self.mode == "phys" else self._rms_raw

    @rms.setter
    def rms(self, rms):
        self._rms = rms

    @property
    def std(self):
        return self._std if self.mode == "phys" else self._std_raw

    def cut(self, start=None, stop=None, include_ends=True, interpolation_mode=0):
        cut_sig = super().cut(start, stop, include_ends, interpolation_mode)

        cut_sig.group_index = self.group_index
        cut_sig.channel_index = self.channel_index
        cut_sig.computed = self.computed
        cut_sig.color = self.color
        cut_sig.computation = self.computation
        cut_sig.precision = self.precision
        cut_sig.mdf_uuif = self.origin_uuid

        return PlotSignal(cut_sig, duplication=self.duplication)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode != self._mode:
            self._mode = mode
            if mode == "raw":
                self.plot_samples = self.raw_samples
                self.plot_timestamps = self.timestamps
            else:
                self.plot_samples = self.phys_samples
                self.plot_timestamps = self.timestamps

            if self.plot_samples.dtype.kind in "SUV":
                self.is_string = True
            else:
                self.is_string = False

    def get_stats(self, cursor=None, region=None, view_region=None):
        stats = {}
        sig = self
        x = sig.timestamps
        size = len(x)

        if size:

            if sig.is_string:
                stats["overall_min"] = ""
                stats["overall_max"] = ""
                stats["overall_average"] = ""
                stats["overall_rms"] = ""
                stats["overall_std"] = ""
                stats["overall_start"] = x[0]
                stats["overall_stop"] = x[-1]
                stats["overall_gradient"] = ""
                stats["overall_integral"] = ""
                stats["overall_delta"] = ""
                stats["overall_start"] = x[0]
                stats["overall_stop"] = x[-1]
                stats["overall_delta_t"] = x[-1] - x[0]
                stats["unit"] = ""
                stats["color"] = sig.color
                stats["name"] = sig.name

                if cursor is not None:
                    position = cursor
                    stats["cursor_t"] = position

                    value, kind, format = self.value_at_timestamp(position)

                    if kind in "SUV":
                        fmt = "{}"
                    elif kind == "f":
                        fmt = f"{{:.{self.precision}f}}"
                    else:
                        if format == "hex":
                            fmt = "0x{:X}"
                        elif format == "bin":
                            fmt = "0b{:b}"
                        elif format == "phys":
                            fmt = "{}"

                    value = fmt.format(value)

                    stats["cursor_value"] = value

                else:
                    stats["cursor_t"] = ""
                    stats["cursor_value"] = ""

                stats["selected_start"] = ""
                stats["selected_stop"] = ""
                stats["selected_delta_t"] = ""
                stats["selected_min"] = ""
                stats["selected_max"] = ""
                stats["selected_average"] = ""
                stats["selected_rms"] = ""
                stats["selected_std"] = ""
                stats["selected_delta"] = ""
                stats["selected_gradient"] = ""
                stats["selected_integral"] = ""
                stats["visible_min"] = ""
                stats["visible_max"] = ""
                stats["visible_average"] = ""
                stats["visible_rms"] = ""
                stats["visible_delta"] = ""
                stats["visible_std"] = ""
                stats["visible_gradient"] = ""
                stats["visible_integral"] = ""
            else:
                if isinstance(sig.min, str):
                    kind = "S"
                    fmt = "{}"
                else:
                    kind = sig.min.dtype.kind
                    format = sig.format
                    if kind in "SUV":
                        fmt = "{}"
                    elif kind == "f":
                        fmt = f"{{:.{self.precision}f}}"
                    else:
                        if format == "hex":
                            fmt = "0x{:X}"
                        elif format == "bin":
                            fmt = "0b{:b}"
                        elif format == "phys":
                            fmt = "{}"

                if size == 1:
                    stats["overall_gradient"] = 0
                    stats["overall_integral"] = 0
                else:
                    stats["overall_gradient"] = (sig.samples[-1] - sig.samples[0]) / (
                        sig.timestamps[-1] - sig.timestamps[0]
                    )
                    stats["overall_integral"] = np.trapz(sig.samples, sig.timestamps)

                stats["overall_min"] = fmt.format(sig.min)
                stats["overall_max"] = fmt.format(sig.max)
                stats["overall_average"] = sig.avg
                stats["overall_rms"] = sig.rms
                stats["overall_std"] = sig.std
                stats["overall_start"] = sig.timestamps[0]
                stats["overall_stop"] = sig.timestamps[-1]
                stats["overall_delta"] = sig.samples[-1] - sig.samples[0]
                stats["overall_delta_t"] = x[-1] - x[0]
                stats["unit"] = sig.unit
                stats["color"] = sig.color
                stats["name"] = sig.name

                if cursor is not None:
                    position = cursor
                    stats["cursor_t"] = position

                    value, kind, format = self.value_at_timestamp(position)

                    if kind in "SUV":
                        fmt = "{}"
                    elif kind == "f":
                        fmt = f"{{:.{self.precision}f}}"
                    else:
                        if format == "hex":
                            fmt = "0x{:X}"
                        elif format == "bin":
                            fmt = "0b{:b}"
                        elif format == "phys":
                            fmt = "{}"

                    value = fmt.format(value)

                    stats["cursor_value"] = value

                else:
                    stats["cursor_t"] = ""
                    stats["cursor_value"] = ""

                if region:
                    start, stop = region

                    new_stats = {}
                    new_stats["selected_start"] = start
                    new_stats["selected_stop"] = stop
                    new_stats["selected_delta_t"] = stop - start

                    cut = sig.cut(start, stop)

                    if self.mode == "raw":
                        samples = cut.raw_samples
                    else:
                        samples = cut.phys_samples

                    idx = np.isfinite(samples).ravel()
                    samples = samples[idx]
                    timestamps = cut.timestamps[idx]

                    size = len(samples)

                    if size:
                        kind = samples.dtype.kind
                        format = self.format

                        if kind in "SUV":
                            fmt = "{}"
                        elif kind == "f":
                            fmt = f"{{:.{self.precision}f}}"
                        else:
                            if format == "hex":
                                fmt = "0x{:X}"
                            elif format == "bin":
                                fmt = "0b{:b}"
                            elif format == "phys":
                                fmt = "{}"

                        new_stats["selected_min"] = fmt.format(np.nanmin(samples))
                        new_stats["selected_max"] = fmt.format(np.nanmax(samples))
                        new_stats["selected_average"] = np.mean(samples)
                        new_stats["selected_std"] = np.std(samples)
                        new_stats["selected_rms"] = np.sqrt(np.mean(np.square(samples)))
                        if kind in "ui":
                            new_stats["selected_delta"] = fmt.format(
                                int(samples[-1]) - int(samples[0])
                            )
                        else:
                            new_stats["selected_delta"] = fmt.format(
                                (samples[-1] - samples[0])
                            )

                        if size == 1:
                            new_stats["selected_gradient"] = 0
                            new_stats["selected_integral"] = 0
                        else:
                            new_stats["selected_gradient"] = (
                                samples[-1] - samples[0]
                            ) / (timestamps[-1] - timestamps[0])
                            new_stats["selected_integral"] = np.trapz(
                                samples, timestamps
                            )

                    else:
                        new_stats["selected_min"] = "n.a."
                        new_stats["selected_max"] = "n.a."
                        new_stats["selected_average"] = "n.a."
                        new_stats["selected_rms"] = "n.a."
                        new_stats["selected_std"] = "n.a."
                        new_stats["selected_gradient"] = "n.a."
                        new_stats["selected_integral"] = "n.a."
                        new_stats["selected_delta"] = "n.a."

                    sig._stats["range"] = (start, stop)
                    sig._stats["range_stats"] = new_stats

                    stats.update(sig._stats["range_stats"])

                else:
                    stats["selected_start"] = ""
                    stats["selected_stop"] = ""
                    stats["selected_delta_t"] = ""
                    stats["selected_min"] = ""
                    stats["selected_max"] = ""
                    stats["selected_average"] = ""
                    stats["selected_rms"] = ""
                    stats["selected_std"] = ""
                    stats["selected_delta"] = ""
                    stats["selected_gradient"] = ""
                    stats["selected_integral"] = ""

                start, stop = view_region

                new_stats = {}
                new_stats["visible_start"] = start
                new_stats["visible_stop"] = stop
                new_stats["visible_delta_t"] = stop - start

                cut = sig.cut(start, stop)

                if self.mode == "raw":
                    samples = cut.raw_samples
                else:
                    samples = cut.phys_samples

                idx = np.isfinite(samples).ravel()
                samples = samples[idx]
                timestamps = cut.timestamps[idx]

                size = len(samples)

                if size:
                    kind = samples.dtype.kind
                    format = self.format

                    if kind in "SUV":
                        fmt = "{}"
                    elif kind == "f":
                        fmt = f"{{:.{self.precision}f}}"
                    else:
                        if format == "hex":
                            fmt = "0x{:X}"
                        elif format == "bin":
                            fmt = "0b{:b}"
                        elif format == "phys":
                            fmt = "{}"

                    new_stats["visible_min"] = fmt.format(np.nanmin(samples))
                    new_stats["visible_max"] = fmt.format(np.nanmax(samples))
                    new_stats["visible_average"] = np.mean(samples)
                    new_stats["visible_std"] = np.std(samples)
                    new_stats["visible_rms"] = np.sqrt(np.mean(np.square(samples)))
                    if kind in "ui":
                        new_stats["visible_delta"] = int(cut.samples[-1]) - int(
                            cut.samples[0]
                        )
                    else:
                        new_stats["visible_delta"] = fmt.format(
                            cut.samples[-1] - cut.samples[0]
                        )

                    if size == 1:
                        new_stats["visible_gradient"] = 0
                        new_stats["visible_integral"] = 0
                    else:
                        new_stats["visible_gradient"] = (samples[-1] - samples[0]) / (
                            timestamps[-1] - timestamps[0]
                        )
                        new_stats["visible_integral"] = np.trapz(samples, timestamps)

                else:
                    new_stats["visible_min"] = "n.a."
                    new_stats["visible_max"] = "n.a."
                    new_stats["visible_average"] = "n.a."
                    new_stats["visible_rms"] = "n.a."
                    new_stats["visible_std"] = "n.a."
                    new_stats["visible_delta"] = "n.a."
                    new_stats["visible_gradient"] = "n.a."
                    new_stats["visible_integral"] = "n.a."

                sig._stats["visible"] = (start, stop)
                sig._stats["visible_stats"] = new_stats

                stats.update(sig._stats["visible_stats"])

        else:
            stats["overall_min"] = "n.a."
            stats["overall_max"] = "n.a."
            stats["overall_average"] = "n.a."
            stats["overall_rms"] = "n.a."
            stats["overall_std"] = "n.a."
            stats["overall_start"] = "n.a."
            stats["overall_stop"] = "n.a."
            stats["overall_gradient"] = "n.a."
            stats["overall_integral"] = "n.a."
            stats["overall_delta"] = "n.a."
            stats["overall_delta_t"] = "n.a."
            stats["unit"] = sig.unit
            stats["color"] = sig.color
            stats["name"] = sig.name

            if cursor is not None:
                position = cursor
                stats["cursor_t"] = position

                stats["cursor_value"] = "n.a."

            else:
                stats["cursor_t"] = ""
                stats["cursor_value"] = ""

            if region is not None:
                start, stop = region

                stats["selected_start"] = start
                stats["selected_stop"] = stop
                stats["selected_delta_t"] = stop - start

                stats["selected_min"] = "n.a."
                stats["selected_max"] = "n.a."
                stats["selected_average"] = "n.a."
                stats["selected_rms"] = "n.a."
                stats["selected_std"] = "n.a."
                stats["selected_delta"] = "n.a."
                stats["selected_gradient"] = "n.a."
                stats["selected_integral"] = "n.a."

            else:
                stats["selected_start"] = ""
                stats["selected_stop"] = ""
                stats["selected_delta_t"] = ""
                stats["selected_min"] = ""
                stats["selected_max"] = ""
                stats["selected_average"] = "n.a."
                stats["selected_rms"] = "n.a."
                stats["selected_std"] = "n.a."
                stats["selected_delta"] = ""
                stats["selected_gradient"] = ""
                stats["selected_integral"] = ""

            start, stop = view_region

            stats["visible_start"] = start
            stats["visible_stop"] = stop
            stats["visible_delta_t"] = stop - start

            stats["visible_min"] = "n.a."
            stats["visible_max"] = "n.a."
            stats["visible_average"] = "n.a."
            stats["visible_rms"] = "n.a."
            stats["visible_std"] = "n.a."
            stats["visible_delta"] = "n.a."
            stats["visible_gradient"] = "n.a."
            stats["visible_integral"] = "n.a."

        #        sig._stats["fmt"] = fmt
        return stats

    def trim_c(self, start=None, stop=None, width=1900, force=False):

        trim_info = (start, stop, width)
        if not force and self.trim_info == trim_info:
            return None

        self.trim_info = trim_info
        sig_timestamps = self.timestamps
        dim = sig_timestamps.size

        if dim:

            if start is None:
                start = sig_timestamps[0]
            if stop is None:
                stop = sig_timestamps[-1]

            if start > stop:
                start, stop = stop, start

            if self._mode == "raw":
                signal_samples = self.raw_samples
            else:
                signal_samples = self.phys_samples

            start_t_sig, stop_t_sig = (
                sig_timestamps[0],
                sig_timestamps[-1],
            )
            if start > stop_t_sig or stop < start_t_sig:
                self.plot_samples = signal_samples[:0]
                self.plot_timestamps = sig_timestamps[:0]
                pos = []
            else:
                start_t = simple_max(start, start_t_sig)
                stop_t = simple_min(stop, stop_t_sig)

                if start_t == start_t_sig:
                    start_ = 0
                else:
                    start_ = np.searchsorted(sig_timestamps, start_t, side="right")

                if stop_t == stop_t_sig:
                    stop_ = dim
                else:
                    stop_ = np.searchsorted(sig_timestamps, stop_t, side="right")

                if stop == start:
                    visible_duplication = 0
                else:

                    visible = int((stop_t - start_t) / (stop - start) * width)

                    if visible:
                        visible_duplication = (stop_ - start_) // visible
                    else:
                        visible_duplication = 0

                if visible_duplication > self.duplication:
                    samples = signal_samples[start_:stop_]
                    timestamps = sig_timestamps[start_:stop_]
                    count, rest = divmod(samples.size, visible_duplication)
                    if rest:
                        count += 1
                    else:
                        rest = visible_duplication
                    steps = visible_duplication

                    if samples.dtype.kind == "f" and samples.itemsize == 2:
                        samples = samples.astype("f8")
                        self._dtype = "f8"

                    if samples.dtype != self._plot_samples.dtype:
                        self._plot_samples = np.empty(
                            2 * PLOT_BUFFER_SIZE, dtype=samples.dtype
                        )

                        self._dtype = samples.dtype

                    if samples.flags.c_contiguous:
                        positions(
                            samples,
                            timestamps,
                            self._plot_samples,
                            self._plot_timestamps,
                            self._pos,
                            steps,
                            count,
                            rest,
                            samples.dtype.kind,
                            samples.itemsize,
                        )
                    else:
                        positions(
                            samples.copy(),
                            timestamps,
                            self._plot_samples,
                            self._plot_timestamps,
                            self._pos,
                            steps,
                            count,
                            rest,
                            samples.dtype.kind,
                            samples.itemsize,
                        )

                    size = 2 * count
                    pos = self._pos[:size]
                    self.plot_samples = self._plot_samples[:size]
                    self.plot_timestamps = self._plot_timestamps[:size]

                else:
                    start_ = simple_min(simple_max(0, start_ - 2), dim - 1)
                    stop_ = simple_min(stop_ + 2, dim)

                    if start_ == 0 and stop_ == dim:
                        self.plot_samples = signal_samples
                        self.plot_timestamps = sig_timestamps

                        pos = None
                    else:

                        self.plot_samples = signal_samples[start_:stop_]
                        self.plot_timestamps = sig_timestamps[start_:stop_]

                        pos = np.arange(start_, stop_)

        else:
            pos = None

        return pos

    def trim_python(self, start=None, stop=None, width=1900, force=False):
        trim_info = (start, stop, width)
        if not force and self.trim_info == trim_info:
            return None

        self.trim_info = trim_info
        sig = self
        dim = len(sig.timestamps)

        if dim:

            if start is None:
                start = sig.timestamps[0]
            if stop is None:
                stop = sig.timestamps[-1]

            if self.mode == "raw":
                signal_samples = self.raw_samples
            else:
                signal_samples = self.phys_samples

            start_t_sig, stop_t_sig = (
                sig.timestamps[0],
                sig.timestamps[-1],
            )
            if start > stop_t_sig or stop < start_t_sig:
                sig.plot_samples = signal_samples[:0]
                sig.plot_timestamps = sig.timestamps[:0]
                pos = []
            else:
                start_t = max(start, start_t_sig)
                stop_t = min(stop, stop_t_sig)

                if start_t == start_t_sig:
                    start_ = 0
                else:
                    start_ = np.searchsorted(sig.timestamps, start_t, side="right")
                if stop_t == stop_t_sig:
                    stop_ = dim
                else:
                    stop_ = np.searchsorted(sig.timestamps, stop_t, side="right")

                try:
                    visible = abs(int((stop_t - start_t) / (stop - start) * width))

                    if visible:
                        visible_duplication = abs((stop_ - start_)) // visible
                    else:
                        visible_duplication = 0
                except:
                    visible_duplication = 0

                while visible_duplication > self.duplication:
                    rows = (stop_ - start_) // visible_duplication
                    stop_2 = start_ + rows * visible_duplication

                    samples = signal_samples[start_:stop_2].reshape(
                        rows, visible_duplication
                    )

                    try:
                        pos_max = samples.argmax(axis=1)
                        pos_min = samples.argmin(axis=1)
                        break
                    except:
                        try:
                            pos_max = np.nanargmax(samples, axis=1)
                            pos_min = np.nanargmin(samples, axis=1)
                            break
                        except ValueError:
                            visible_duplication -= 1

                if visible_duplication > self.duplication:

                    pos = np.dstack([pos_min, pos_max])[0]
                    pos.sort()
                    # pos = np.sort(pos)

                    offsets = np.arange(rows) * visible_duplication

                    pos = (pos.T + offsets).T.ravel()

                    samples = signal_samples[start_:stop_2][pos]

                    timestamps = sig.timestamps[start_:stop_2][pos]

                    if stop_2 != stop_:
                        samples_ = signal_samples[stop_2:stop_]

                        try:
                            pos_max = samples_.argmax()
                            pos_min = samples_.argmin()
                        except:
                            pos_max = np.nanargmax(samples_)
                            pos_min = np.nanargmin(samples_)

                        pos2 = (
                            [pos_min, pos_max]
                            if pos_min < pos_max
                            else [pos_max, pos_min]
                        )

                        _size = len(pos)

                        samples_ = signal_samples[stop_2:stop_][pos2]
                        timestamps_ = sig.timestamps[stop_2:stop_][pos2]

                        samples = np.concatenate((samples, samples_))
                        timestamps = np.concatenate((timestamps, timestamps_))

                        pos2 = p1, p2 = [min(e + stop_2, dim - 1) for e in pos2]

                        # pos = np.concatenate([pos, pos2])

                        new_pos = np.empty(_size + 2, dtype=pos.dtype)
                        new_pos[:_size] = pos
                        new_pos[_size] = p1
                        new_pos[_size + 1] = p2
                        pos = new_pos

                    sig.plot_samples = samples
                    sig.plot_timestamps = timestamps

                else:
                    start_ = min(max(0, start_ - 2), dim - 1)
                    stop_ = min(stop_ + 2, dim)

                    if start_ == 0 and stop_ == dim:
                        sig.plot_samples = signal_samples
                        sig.plot_timestamps = sig.timestamps

                        pos = None
                    else:

                        sig.plot_samples = signal_samples[start_:stop_]
                        sig.plot_timestamps = sig.timestamps[start_:stop_]

                        pos = np.arange(start_, stop_)

        else:
            pos = None

        return pos

    def trim(self, start=None, stop=None, width=1900, force=False):
        if self._enable:
            try:
                return self.trim_c(start, stop, width, force)
            except:
                print(format_exc())
                return self.trim_python(start, stop, width, force)

    def value_at_timestamp(self, timestamp, numeric=False):

        if self.mode == "raw":
            kind = self.raw_samples.dtype.kind
            samples = self.raw_samples
        else:
            kind = self.phys_samples.dtype.kind
            samples = self.phys_samples

        if numeric and kind not in "uif":
            samples = self.raw_samples

        if self.samples.size == 0 or timestamp < self.timestamps[0]:
            value = "n.a."
        else:

            if timestamp > self.timestamps[-1]:
                index = -1
            else:
                index = np.searchsorted(self.timestamps, timestamp, side="left")

            value = samples[index]

            if kind == "S":
                try:
                    value = value.decode("utf-8").strip(" \r\n\t\v\0")
                except:
                    value = value.decode("latin-1").strip(" \r\n\t\v\0")

                value = value or "<empty string>"
            elif kind == "f":
                value = float(value)
            else:
                value = int(value)

        return value, kind, self.format

    def value_at_index(self, index):

        if self.mode == "raw":
            kind = self.raw_samples.dtype.kind
        else:
            kind = self.phys_samples.dtype.kind

        if index is None:
            value = "n.a."
        else:

            if self.mode == "raw":
                value = self.raw_samples[index]
            else:
                value = self.phys_samples[index]

            if kind == "S":
                try:
                    value = value.decode("utf-8").strip(" \r\n\t\v\0")
                except:
                    value = value.decode("latin-1").strip(" \r\n\t\v\0")

                value = value or "<empty string>"
            elif kind == "f":
                value = float(value)
            else:
                value = int(value)

        return value, kind, self.format


from .signal_scale import ScaleDialog


class Plot(QtWidgets.QWidget):

    add_channels_request = QtCore.Signal(list)
    close_request = QtCore.Signal()
    clicked = QtCore.Signal()
    cursor_moved_signal = QtCore.Signal(object, float)
    cursor_removed_signal = QtCore.Signal(object)
    region_moved_signal = QtCore.Signal(object, list)
    region_removed_signal = QtCore.Signal(object)
    show_properties = QtCore.Signal(list)
    splitter_moved = QtCore.Signal(object, int)
    pattern_group_added = QtCore.Signal(object, object)

    def __init__(
        self,
        signals,
        with_dots=False,
        origin=None,
        mdf=None,
        line_interconnect="line",
        hide_missing_channels=False,
        hide_disabled_channels=False,
        x_axis="time",
        allow_cursor=True,
        *args,
        **kwargs,
    ):
        events = kwargs.pop("events", None)
        super().__init__(*args, **kwargs)
        self.closed = False
        self.line_interconnect = line_interconnect
        self.setContentsMargins(0, 0, 0, 0)
        self.pattern = {}
        self.mdf = mdf

        self.ignore_selection_change = False

        self.x_name = "t" if x_axis == "time" else "f"
        self.x_unit = "s" if x_axis == "time" else "Hz"

        self.info_uuid = None

        self._range_start = None
        self._range_stop = None

        self._can_switch_mode = True

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(1, 1, 1, 1)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setSpacing(1)
        vbox.setContentsMargins(1, 1, 1, 1)
        widget = QtWidgets.QWidget()
        self.channel_selection = ChannelsTreeWidget(
            hide_missing_channels=hide_missing_channels,
            hide_disabled_channels=hide_disabled_channels,
            parent=self,
            plot=self,
        )

        widget.setLayout(vbox)

        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(widget)
        self.splitter.setOpaqueResize(False)

        self.plot = _Plot(
            with_dots=with_dots,
            line_interconnect=self.line_interconnect,
            parent=self,
            events=events,
            origin=origin,
            mdf=self.mdf,
            x_axis=x_axis,
            allow_cursor=allow_cursor,
            plot_parent=self,
        )

        self.cursor_info = CursorInfo(
            precision=QtCore.QSettings().value("plot_cursor_precision", 6),
            unit=self.x_unit,
            name=self.x_name,
            plot=self.plot,
        )

        hbox = QtWidgets.QHBoxLayout()
        hbox.setSpacing(3)
        hbox.setContentsMargins(1, 1, 1, 1)

        vbox.addLayout(hbox)

        btn = QtWidgets.QPushButton("")
        btn.clicked.connect(
            lambda x: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress,
                    QtCore.Qt.Key_H,
                    QtCore.Qt.NoModifier,
                )
            )
        )
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/home.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        btn.setIcon(icon)
        btn.setToolTip("Home")
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton("")
        btn.clicked.connect(
            lambda x: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress,
                    QtCore.Qt.Key_F,
                    QtCore.Qt.NoModifier,
                )
            )
        )
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/fit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        btn.setIcon(icon)
        btn.setToolTip("Fit")
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton("")
        btn.clicked.connect(
            lambda x: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress,
                    QtCore.Qt.Key_S,
                    QtCore.Qt.NoModifier,
                )
            )
        )
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/list2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        btn.setIcon(icon)
        btn.setToolTip("Stack")
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton("")
        btn.clicked.connect(
            lambda x: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress,
                    QtCore.Qt.Key_I,
                    QtCore.Qt.NoModifier,
                )
            )
        )
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/zoom-in.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        btn.setIcon(icon)
        btn.setToolTip("Zoom in")
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton("")
        btn.clicked.connect(
            lambda x: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress,
                    QtCore.Qt.Key_O,
                    QtCore.Qt.NoModifier,
                )
            )
        )
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/zoom-out.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        btn.setIcon(icon)
        btn.setToolTip("Zoom out")
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton("")
        btn.clicked.connect(self.increase_font)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/increase-font.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        btn.setIcon(icon)
        btn.setToolTip("Increase font")
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton("")
        btn.clicked.connect(self.decrease_font)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/decrease-font.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        btn.setIcon(icon)
        btn.setToolTip("Decrease font")
        hbox.addWidget(btn)

        self.lock_btn = btn = QtWidgets.QPushButton("")
        btn.clicked.connect(self.set_locked)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/unlocked.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        btn.setIcon(icon)
        btn.setToolTip("The Y axis is unlocked. Press to lock")
        hbox.addWidget(btn)

        self.locked = False

        hbox.addStretch()

        vbox.addWidget(self.channel_selection)
        vbox.addWidget(self.cursor_info)

        self.range_proxy = pg.SignalProxy(
            self.plot.range_modified, rateLimit=16, slot=self.range_modified
        )
        # self.plot.range_modified.connect(self.range_modified)
        self.plot.range_removed.connect(self.range_removed)
        self.plot.range_modified_finished.connect(self.range_modified_finished)
        self.plot.cursor_removed.connect(self.cursor_removed)

        self.cursor_proxy = pg.SignalProxy(
            self.plot.cursor_moved, rateLimit=16, slot=self.cursor_moved
        )
        # self.plot.cursor_moved.connect(self.cursor_moved)
        self.plot.cursor_move_finished.connect(self.cursor_move_finished)
        self.plot.xrange_changed.connect(self.xrange_changed)
        self.plot.computation_channel_inserted.connect(
            self.computation_channel_inserted
        )
        self.plot.curve_clicked.connect(self.curve_clicked)
        self.plot.signals_enable_changed.connect(self._update_visibile_entries)
        self._visible_entries = set()
        self._visible_items = {}
        self._item_cache = {}

        self.splitter.addWidget(self.plot)

        self.info = ChannelStats(self.x_unit)
        self.info.hide()
        self.splitter.addWidget(self.info)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        self.plot.add_channels_request.connect(self.add_channels_request)
        self.setAcceptDrops(True)

        main_layout.addWidget(self.splitter)

        self.show()
        self.hide()

        if signals:
            self.add_new_channels(signals)

        self.channel_selection.color_changed.connect(self.plot.set_color)
        self.channel_selection.unit_changed.connect(self.plot.set_unit)
        self.channel_selection.name_changed.connect(self.plot.set_name)

        self.channel_selection.itemsDeleted.connect(self.channel_selection_reduced)
        # self.channel_selection.itemPressed.connect(self.channel_selection_modified)
        self.channel_selection.currentItemChanged.connect(
            self.channel_selection_row_changed
        )
        self.channel_selection.add_channels_request.connect(self.add_channels_request)
        self.channel_selection.set_time_offset.connect(self.plot.set_time_offset)
        self.channel_selection.show_properties.connect(self._show_properties)
        self.channel_selection.insert_computation.connect(self.plot.insert_computation)

        self.channel_selection.model().dataChanged.connect(
            self.channel_selection_item_changed
        )
        self.channel_selection.items_rearranged.connect(
            self.channel_selection_rearranged
        )
        self.channel_selection.pattern_group_added.connect(self.pattern_group_added_req)
        self.channel_selection.itemDoubleClicked.connect(
            self.channel_selection_item_double_clicked
        )

        self.channel_selection.compute_fft_request.connect(self.compute_fft)
        self.channel_selection.itemExpanded.connect(self.update_current_values)
        self.channel_selection.verticalScrollBar().valueChanged.connect(
            self.update_current_values
        )

        self.keyboard_events = (
            set(
                [
                    (QtCore.Qt.Key_M, int(QtCore.Qt.NoModifier)),
                    (QtCore.Qt.Key_C, int(QtCore.Qt.NoModifier)),
                    (QtCore.Qt.Key_C, int(QtCore.Qt.ControlModifier)),
                    (QtCore.Qt.Key_B, int(QtCore.Qt.ControlModifier)),
                    (QtCore.Qt.Key_H, int(QtCore.Qt.ControlModifier)),
                    (QtCore.Qt.Key_P, int(QtCore.Qt.ControlModifier)),
                    (QtCore.Qt.Key_T, int(QtCore.Qt.ControlModifier)),
                    (QtCore.Qt.Key_G, int(QtCore.Qt.ControlModifier)),
                ]
            )
            | self.plot.keyboard_events
        )

        self.splitter.splitterMoved.connect(self.set_splitter)

        self.show()

    def set_locked(self, event=None, locked=None):
        if locked is None:
            locked = not self.locked
        if locked:
            tooltip = "The Y axis is locked. Press to unlock"
            png = ":/locked.png"
        else:
            tooltip = "The Y axis is unlocked. Press to lock"
            png = ":/unlocked.png"

        self.channel_selection.setColumnHidden(self.channel_selection.CommonAxisColumn, locked)

        self.locked = locked
        self.plot.set_locked(locked)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(png), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lock_btn.setToolTip(tooltip)
        self.lock_btn.setIcon(icon)

    def increase_font(self):
        font = self.font()
        size = font.pointSize()
        pos = bisect.bisect_right(FONT_SIZE, size)
        if pos == len(FONT_SIZE):
            pos -= 1
        new_size = FONT_SIZE[pos]

        self.set_font_size(new_size)

    def decrease_font(self):
        font = self.font()
        size = font.pointSize()
        pos = bisect.bisect_left(FONT_SIZE, size) - 1
        if pos < 0:
            pos = 0
        new_size = FONT_SIZE[pos]

        self.set_font_size(new_size)

    def set_font_size(self, size):
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)
        self.plot.y_axis.set_font_size(size)
        self.plot.x_axis.set_font_size(size)

    def curve_clicked(self, uuid):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while True:
            item = iterator.value()
            if item is None:
                break

            if item.type() == item.Channel and item.uuid == uuid:
                self.channel_selection.setCurrentItem(item)
                break

            iterator += 1

    def channel_selection_rearranged(self, uuids):
        self._update_visibile_entries()

    def channel_selection_item_changed(self, top_left, bottom_right, roles):
        if QtCore.Qt.CheckStateRole not in roles:
            return

        item = self.channel_selection.itemFromIndex(top_left)

        if item.type() != item.Channel:
            return

        column = top_left.column()

        if column == 0:
            enabled = item.checkState(column) == QtCore.Qt.Checked
            if enabled != item.signal.enable:
                item.signal.enable = enabled
                self.plot.set_signal_enable(item.uuid, item.checkState(column))

        elif column == 2:
            if not self.locked:
                enabled = item.checkState(column) == QtCore.Qt.Checked
                if enabled != item.signal.y_link:
                    item.signal.y_link = enabled
                    self.plot.set_common_axis(item.uuid, enabled)

        elif column == 3:
            enabled = item.checkState(column) == QtCore.Qt.Checked
            if enabled != item.signal.individual_axis:
                self.plot.set_individual_axis(item.uuid, enabled)

    def channel_selection_item_double_clicked(self, item, column):
        if item is None:
            return

        type = item.type()
        if type == ChannelsTreeItem.Group:
            dlg = RangeEditor(
                f"channels from <{item._name}>", ranges=item.ranges, parent=self
            )
            dlg.exec_()
            if dlg.pressed_button == "apply":
                item.set_ranges(dlg.result)
                item.update_child_values()
        elif type == ChannelsTreeItem.Channel:
            dlg = RangeEditor(item.signal.name, item.unit, item.ranges, parent=self)
            dlg.exec_()
            if dlg.pressed_button == "apply":
                item.set_ranges(dlg.result)
                item.set_value(item._value, update=True)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def channel_selection_modified(self, item):
        self.channel_selection_row_changed(item, None)

    def channel_selection_row_changed(self, current, previous):
        if not self.closed and not self.ignore_selection_change:
            if current and current.type() == ChannelsTreeItem.Channel:
                item = current
                uuid = item.uuid
                self.info_uuid = uuid

                sig, index = self.plot.signal_by_uuid(uuid)
                if sig.enable:
                    self.plot.set_current_uuid(self.info_uuid)

                    if self.info.isVisible():
                        stats = self.plot.get_stats(self.info_uuid)
                        self.info.set_stats(stats)

    def channel_selection_reduced(self, deleted):

        self.plot.delete_channels(deleted)

        if self.info_uuid in deleted:
            self.info_uuid = None
            self.info.hide()

        count = 0
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while iterator.value():
            count += 1
            iterator += 1

        if not count:
            self.close_request.emit()

        self._update_visibile_entries()

    def cursor_move_finished(self, cursor=None):
        x = self.plot.get_current_timebase()
        if x.size:
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

    def cursor_moved(self, cursor=None):

        if self.plot.cursor1 is None:
            return

        position = self.plot.cursor1.value()

        if not self.plot.region:

            self.cursor_info.update_value()

            for item in self._visible_items.values():
                if item.type() == item.Channel:

                    signal, idx = self.plot.signal_by_uuid(item.uuid)
                    index = self.plot.get_timestamp_index(position, signal.timestamps)

                    value, kind, fmt = signal.value_at_index(index)

                    item.set_prefix("= ")
                    item.kind = kind
                    item.set_fmt(fmt)

                    item.set_value(value, update=True)

        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self.cursor_moved_signal.emit(self, position)

    def cursor_removed(self):

        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while True:
            item = iterator.value()
            if item is None:
                break

            if item.type() == item.Channel and not self.plot.region:
                self.cursor_info.update_value()
                item.set_prefix("")
                item.set_value("")

            iterator += 1

        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self.cursor_removed_signal.emit(self)

    def range_modified(self, region):

        if self.plot.region is None:
            return

        start, stop = self.plot.region.getRegion()

        self.cursor_info.update_value()

        for item in self._visible_items.values():
            if item.type() == item.Channel:

                signal, i = self.plot.signal_by_uuid(item.uuid)

                start_v, kind, fmt = signal.value_at_timestamp(start)
                stop_v, kind, fmt = signal.value_at_timestamp(stop)

                item.set_prefix("Δ = ")
                item.set_fmt(signal.format)

                if "n.a." not in (start_v, stop_v):
                    if kind in "ui":
                        delta = np.int64(stop_v) - np.int64(start_v)
                        item.kind = kind
                        item.set_value(delta)
                        item.set_fmt(fmt)
                    elif kind == "f":
                        delta = stop_v - start_v
                        item.kind = kind
                        item.set_value(delta)
                        item.set_fmt(fmt)
                    else:
                        item.set_value("n.a.")
                else:
                    item.set_value("n.a.")

        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self.region_moved_signal.emit(self, [start, stop])

    def xrange_changed(self, *args):
        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

    def range_modified_finished(self):
        if not self.plot.region:
            return
        start, stop = self.plot.region.getRegion()

        timebase = self.plot.get_current_timebase()

        if timebase.size:
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

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key_M and modifiers == QtCore.Qt.NoModifier:
            ch_size, plt_size, info_size = self.splitter.sizes()

            if self.info.isVisible():
                self.info.hide()
                self.splitter.setSizes((ch_size, plt_size + info_size, 0))

            else:

                self.info.show()
                self.splitter.setSizes(
                    (
                        ch_size,
                        int(0.8 * (plt_size + info_size)),
                        int(0.2 * (plt_size + info_size)),
                    )
                )

        elif (
            key in (QtCore.Qt.Key_B, QtCore.Qt.Key_H, QtCore.Qt.Key_P)
            and modifiers == QtCore.Qt.ControlModifier
        ):

            selected_items = self.channel_selection.selectedItems()
            if not selected_items:
                signals = [(sig, i) for i, sig in enumerate(self.plot.signals)]

            else:
                uuids = [
                    item.uuid
                    for item in selected_items
                    if item.type() == ChannelsTreeItem.Channel
                ]

                signals = [self.plot.signal_by_uuid(uuid) for uuid in uuids]

            if signals:

                if key == QtCore.Qt.Key_B:
                    fmt = "bin"
                elif key == QtCore.Qt.Key_H:
                    fmt = "hex"
                else:
                    fmt = "phys"

                for signal, idx in signals:
                    if signal.plot_samples.dtype.kind in "ui":
                        signal.format = fmt

                        value, kind, fmt = signal.value_at_timestamp(0)

                        widget = self.item_by_uuid(signal.uuid)
                        widget.kind = kind
                        widget.set_fmt(fmt)
                        widget.set_value(update=True)

                        if self.plot.current_uuid == signal.uuid:
                            self.plot.y_axis.format = fmt
                            self.plot.y_axis.picture = None
                            self.plot.y_axis.update()

                        axis = self.plot.get_axis(idx)
                        if isinstance(axis, FormatedAxis):
                            axis.format = fmt
                            axis.picture = None
                            axis.update()

            self.plot.update_plt()

        elif (
            key in (QtCore.Qt.Key_R, QtCore.Qt.Key_S)
            and modifiers == QtCore.Qt.AltModifier
            and self._can_switch_mode
        ):
            self.plot._pixmap = None
            selected_items = self.channel_selection.selectedItems()
            if not selected_items:
                signals = [(sig, i) for i, sig in enumerate(self.plot.signals)]

            else:
                uuids = [
                    item.uuid
                    for item in selected_items
                    if item.type() == ChannelsTreeItem.Channel
                ]

                signals = [self.plot.signal_by_uuid(uuid) for uuid in uuids]

            if signals:

                if key == QtCore.Qt.Key_R:
                    mode = "raw"
                    style = QtCore.Qt.DashLine

                else:
                    mode = "phys"
                    style = QtCore.Qt.SolidLine

                for signal, idx in signals:
                    if signal.mode != mode:
                        signal.pen = fn.mkPen(color=signal.color, style=style)

                        buttom, top = signal.y_range

                        try:
                            min_, max_ = float(signal.min), float(signal.max)
                        except:
                            min_, max_ = 0, 1

                        if max_ != min_ and top != buttom:

                            factor = (top - buttom) / (max_ - min_)
                            offset = (buttom - min_) / (top - buttom)
                        else:
                            factor = 1
                            offset = 0

                        signal.mode = mode

                        try:
                            min_, max_ = float(signal.min), float(signal.max)
                        except:
                            min_, max_ = 0, 1

                        if max_ != min_:

                            delta = (max_ - min_) * factor
                            buttom = min_ + offset * delta
                            top = buttom + delta
                        else:
                            buttom, top = max_ - 1, max_ + 1

                        # TO DO: view.setYRange(buttom, top, padding=0, update=True)

                        if self.plot.current_uuid == signal.uuid:
                            self.plot.y_axis.mode = mode
                            self.plot.y_axis.hide()
                            self.plot.y_axis.show()

            self.plot.update_lines()
            self.plot.update_plt()

            if self.plot.cursor1:
                self.plot.cursor_moved.emit(self.plot.cursor1)

        elif key == QtCore.Qt.Key_I and modifiers == QtCore.Qt.ControlModifier:
            if self.plot.cursor1:
                position = self.plot.cursor1.value()
                comment, submit = QtWidgets.QInputDialog.getMultiLineText(
                    self,
                    "Insert comments",
                    f"Enter the comments for cursor position {position:.9f}s:",
                    "",
                )
                if submit:
                    line = pg.InfiniteLine(
                        pos=position,
                        label=f"t = {position}s\n\n{comment}",
                        pen={
                            "color": "#FF0000",
                            "width": 2,
                            "style": QtCore.Qt.DashLine,
                        },
                        labelOpts={
                            "border": {
                                "color": "#FF0000",
                                "width": 2,
                                "style": QtCore.Qt.DashLine,
                            },
                            "fill": "ff9b37",
                            "color": "#000000",
                            "movable": True,
                        },
                    )
                    self.plot.plotItem.addItem(line, ignoreBounds=True)

        elif key == QtCore.Qt.Key_I and modifiers == QtCore.Qt.AltModifier:
            visible = None
            for item in self.plot.plotItem.items:
                if not isinstance(item, pg.InfiniteLine) or item is self.plot.cursor1:
                    continue

                if visible is None:
                    visible = item.label.isVisible()

                try:
                    if visible:
                        item.label.setVisible(False)
                    else:
                        item.label.setVisible(True)
                except:
                    pass

        elif key == QtCore.Qt.Key_G and modifiers == QtCore.Qt.ControlModifier:
            selected_items = [
                item
                for item in self.channel_selection.selectedItems()
                if item.type() == ChannelsTreeItem.Channel
            ]

            if selected_items:

                uuids = [item.uuid for item in selected_items]

                signals = {}
                indexes = []
                for i, uuid in enumerate(uuids):
                    sig, idx = self.plot.signal_by_uuid(uuid)
                    if i == 0:
                        y_range = sig.y_range
                    indexes.append(idx)
                    signals[sig.name] = sig

                diag = ScaleDialog(signals, y_range, parent=self)

                if diag.exec():
                    y_range = (
                        diag.y_bottom.value(),
                        diag.y_top.value(),
                    )
                    for idx in indexes:
                        self.plot.signals[idx].y_range = y_range

                    self.plot.update_plt()

        elif key == QtCore.Qt.Key_R and modifiers == QtCore.Qt.NoModifier:
            iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
            while True:
                item = iterator.value()
                if item is None:
                    break

                if item.type() == item.Channel:
                    item.set_prefix("")
                    item.set_value("")

                iterator += 1

            self.plot.keyPressEvent(event)

        elif key == QtCore.Qt.Key_C and modifiers in (
            QtCore.Qt.NoModifier,
            QtCore.Qt.ControlModifier,
            QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier,
        ):
            self.channel_selection.keyPressEvent(event)

        elif key == QtCore.Qt.Key_P and modifiers == (
            QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier
        ):
            self.channel_selection.keyPressEvent(event)

        elif (key, int(modifiers)) in self.plot.keyboard_events:
            try:
                self.plot.keyPressEvent(event)
            except:
                print(format_exc())

        else:
            super().keyPressEvent(event)

    def range_removed(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while True:
            item = iterator.value()
            if item is None:
                break

            if item.type() == item.Channel:
                item.set_prefix("")
                item.set_value("")

            iterator += 1

        self._range_start = None
        self._range_stop = None

        self.cursor_info.update_value()

        if self.plot.cursor1:
            self.plot.cursor_moved.emit(self.plot.cursor1)
        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self.region_removed_signal.emit(self)

    def computation_channel_inserted(self):
        sig = self.plot.signals[-1]
        sig.enable = True

        background_color = self.palette().color(QtGui.QPalette.Base)

        item = ChannelsTreeItem(
            ChannelsTreeItem.Channel,
            signal=sig,
            check=QtCore.Qt.Checked if sig.enable else QtCore.Qt.Unchecked,
            background_color=background_color,
        )

        self.channel_selection.addTopLevelItem(item)

        self.info_uuid = sig.uuid

        self.plot.set_current_uuid(self.info_uuid, True)

    def add_new_channels(self, channels, mime_data=None, destination=None):
        def add_new_items(tree, root, items, items_pool):
            children = []
            for info in items:

                pattern = info.get("pattern", None)
                uuid = info["uuid"]
                name = info["name"]

                ranges = copy_ranges(info["ranges"])
                for range_info in ranges:
                    range_info["font_color"] = fn.mkColor(range_info["font_color"])
                    range_info["background_color"] = fn.mkColor(
                        range_info["background_color"]
                    )

                if info.get("type", "channel") == "group":

                    item = ChannelsTreeItem(
                        ChannelsTreeItem.Group, name=name, pattern=pattern, uuid=uuid
                    )  # , root)
                    children.append(item)
                    item.set_ranges(ranges)

                    add_new_items(tree, item, info["channels"], items_pool)

                else:

                    if uuid in items_pool:
                        item = items_pool[uuid]
                        children.append(item)

                        del items_pool[uuid]

            root.addChildren(children)

        descriptions = get_descriptions_by_uuid(mime_data)

        invalid = []

        can_trim = True
        for channel in channels.values():
            diff = np.diff(channel.timestamps)
            invalid_indexes = np.argwhere(diff <= 0).ravel()
            if len(invalid_indexes):
                invalid_indexes = invalid_indexes[:10] + 1
                idx = invalid_indexes[0]
                ts = channel.timestamps[idx - 1 : idx + 2]
                invalid.append(
                    f"{channel.name} @ index {invalid_indexes[:10] - 1} with first time stamp error: {ts}"
                )
                if len(np.argwhere(diff < 0).ravel()):
                    can_trim = False

        if invalid:
            errors = "\n".join(invalid)
            try:
                mdi_title = self.parent().windowTitle()
                title = f"plot <{mdi_title}>"
            except:
                title = "plot window"

            QtWidgets.QMessageBox.warning(
                self,
                f"Channels with corrupted time stamps added to {title}",
                f"The following channels do not have monotonous increasing time stamps:\n{errors}",
            )
            self.plot._can_trim = can_trim

        valid = {}
        invalid = []
        for uuid, channel in channels.items():
            if len(channel):
                samples = channel.samples
                if samples.dtype.kind not in "SUV" and np.all(np.isnan(samples)):
                    invalid.append(channel.name)
                elif channel.conversion:
                    samples = channel.physical().samples
                    if samples.dtype.kind not in "SUV" and np.all(np.isnan(samples)):
                        invalid.append(channel.name)
                    else:
                        valid[uuid] = channel
                else:
                    valid[uuid] = channel
            else:
                valid[uuid] = channel

        if invalid:
            QtWidgets.QMessageBox.warning(
                self,
                "All NaN channels will not be plotted:",
                f"The following channels have all NaN samples and will not be plotted:\n{', '.join(invalid)}",
            )

        channels = valid

        self.adjust_splitter(list(channels.values()))
        # QtCore.QCoreApplication.processEvents()

        channels = self.plot.add_new_channels(channels, descriptions=descriptions)

        enforce_y_axis = False
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while True:
            item = iterator.value()
            if item is None:
                break

            if item.type() == item.Channel:
                if item.checkState(2) == QtCore.Qt.Unchecked:
                    enforce_y_axis = False
                    break
                else:
                    enforce_y_axis = True

            iterator += 1

        children = []

        background_color = self.palette().color(QtGui.QPalette.Base)

        new_items = defaultdict(list)
        for sig_uuid, sig in channels.items():

            description = descriptions.get(sig_uuid, {})

            sig.format = description.get("format", "phys")
            sig.mode = description.get("mode", "phys")

            item = ChannelsTreeItem(
                ChannelsTreeItem.Channel,
                signal=sig,
                check=QtCore.Qt.Checked if sig.enable else QtCore.Qt.Unchecked,
                background_color=background_color,
            )
            item.fmt = description.get("fmt", item.fmt)

            if len(sig):
                value, kind, fmt = sig.value_at_timestamp(sig.timestamps[0])
                item.kind = kind
                item._value = "n.a."
                item.set_value(value, force=True, update=True)

            if mime_data is None:
                children.append(item)
            else:
                new_items[sig_uuid] = item

            try:
                item.set_ranges(
                    [
                        {
                            "font_color": range["color"],
                            "background_color": range["color"],
                            "op1": "<=",
                            "op2": "<=",
                            "value1": float(range["start"]),
                            "value2": float(range["stop"]),
                        }
                        for range in description["ranges"]
                    ]
                )
            except KeyError:
                item.set_ranges(copy_ranges(description.get("ranges", [])))

            for range in item.ranges:
                range["font_color"] = fn.mkColor(range["font_color"])
                range["background_color"] = fn.mkColor(range["background_color"])

            if description:
                individual_axis = description.get("individual_axis", False)
                if individual_axis:
                    item.setCheckState(3, QtCore.Qt.Checked)

                    _, idx = self.plot.signal_by_uuid(sig_uuid)
                    axis = self.plot.get_axis(idx)
                    if isinstance(axis, FormatedAxis):
                        axis.setWidth(description["individual_axis_width"])

                if description.get("common_axis", False):
                    item.setCheckState(2, QtCore.Qt.Checked)

                item.precision = description.get("precision", 3)

            if enforce_y_axis:
                item.setCheckState(2, QtCore.Qt.Checked)

            self.info_uuid = sig_uuid

        if mime_data:
            add_new_items(
                self.channel_selection,
                destination or self.channel_selection.invisibleRootItem(),
                mime_data,
                new_items,
            )

            # still have simple signals to add
            if new_items:
                self.channel_selection.addTopLevelItems(list(new_items.values()))

        elif children:

            if destination is None:
                self.channel_selection.addTopLevelItems(children)
            else:
                destination.addChildren(children)

        self.channel_selection.update_channel_groups_count()
        self.channel_selection.refresh()
        self._update_visibile_entries()

    def channel_item_to_config(self, item):
        widget = item

        channel = {"type": "channel"}

        sig, idx = self.plot.signal_by_uuid(widget.uuid)

        channel["name"] = sig.name
        channel["unit"] = sig.unit
        channel["enabled"] = item.checkState(0) == QtCore.Qt.Checked

        if item.checkState(3) == QtCore.Qt.Checked:
            channel["individual_axis"] = True
            channel["individual_axis_width"] = (
                self.plot.axes[idx].boundingRect().width()
            )
        else:
            channel["individual_axis"] = False

        channel["common_axis"] = item.checkState(2) == QtCore.Qt.Checked
        channel["color"] = sig.color.name()
        channel["computed"] = sig.computed
        channel["ranges"] = copy_ranges(widget.ranges)

        for range_info in channel["ranges"]:
            range_info["background_color"] = range_info["background_color"].name()
            range_info["font_color"] = range_info["font_color"].name()

        channel["precision"] = widget.precision
        channel["fmt"] = widget.fmt
        channel["format"] = sig.format
        channel["mode"] = sig.mode
        if sig.computed:
            channel["computation"] = sig.computation

        channel["y_range"] = [float(e) for e in sig.y_range]
        channel["origin_uuid"] = str(sig.origin_uuid)

        if sig.computed and sig.conversion:
            channel["user_defined_name"] = sig.name
            channel["name"] = sig.computation["expression"].strip("}{")

            channel["conversion"] = {}
            for i in range(sig.conversion.val_param_nr):
                channel["conversion"][f"text_{i}"] = sig.conversion.referenced_blocks[
                    f"text_{i}"
                ].decode("utf-8")
                channel["conversion"][f"val_{i}"] = sig.conversion[f"val_{i}"]

        return channel

    def channel_group_item_to_config(self, item):
        widget = item
        pattern = widget.pattern
        if pattern:
            pattern = dict(pattern)
            ranges = copy_ranges(pattern["ranges"])

            for range_info in ranges:
                range_info["font_color"] = range_info["font_color"].name()
                range_info["background_color"] = range_info["background_color"].name()

            pattern["ranges"] = ranges

        ranges = copy_ranges(widget.ranges)

        for range_info in ranges:
            range_info["font_color"] = range_info["font_color"].name()
            range_info["background_color"] = range_info["background_color"].name()

        channel_group = {
            "type": "group",
            "name": widget._name,
            "enabled": item.checkState(0) == QtCore.Qt.Checked,
            "pattern": pattern,
            "ranges": ranges,
            "origin_uuid": None,
        }

        return channel_group

    def to_config(self):
        def item_to_config(tree, root):
            channels = []

            for i in range(root.childCount()):
                item = root.child(i)
                if item.type() == item.Channel:
                    channel = self.channel_item_to_config(item)

                elif item.type() == item.Group:
                    pattern = item.pattern
                    if pattern:
                        pattern = dict(pattern)
                        ranges = copy_ranges(pattern["ranges"])

                        for range_info in ranges:
                            range_info["font_color"] = range_info["font_color"].name()
                            range_info["background_color"] = range_info[
                                "background_color"
                            ].name()

                        pattern["ranges"] = ranges

                    ranges = copy_ranges(item.ranges)

                    for range_info in ranges:
                        range_info["font_color"] = range_info["font_color"].name()
                        range_info["background_color"] = range_info[
                            "background_color"
                        ].name()

                    channel = self.channel_group_item_to_config(item)
                    channel["channels"] = (
                        item_to_config(tree, item) if item.pattern is None else []
                    )

                channels.append(channel)

            return channels

        pattern = self.pattern
        if pattern:
            ranges = copy_ranges(pattern["ranges"])

            for range_info in ranges:
                range_info["font_color"] = range_info["font_color"].name()
                range_info["background_color"] = range_info["background_color"].name()

            pattern["ranges"] = ranges

        config = {
            "channels": item_to_config(
                self.channel_selection, self.channel_selection.invisibleRootItem()
            )
            if not self.pattern
            else [],
            "pattern": pattern,
            "splitter": [int(e) for e in self.splitter.sizes()[:2]]
            + [
                0,
            ],
            "x_range": [float(e) for e in self.plot.viewbox.viewRange()[0]],
            "y_axis_width": self.plot.y_axis.boundingRect().width(),
            "grid": [
                self.plot.plotItem.ctrl.xGridCheck.isChecked(),
                self.plot.plotItem.ctrl.yGridCheck.isChecked(),
            ],
            "cursor_precision": self.cursor_info.precision,
            "font_size": self.font().pointSize(),
            "locked": self.locked,
            "common_axis_y_range": [float(e) for e in self.plot.common_axis_y_range],
            "channels_header": [
                self.splitter.sizes()[0],
                [self.channel_selection.columnWidth(i) for i in range(5)],
            ],
        }

        return config

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat("application/octet-stream-asammdf"):
            e.accept()
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        if e.source() is self.channel_selection:
            super().dropEvent(e)
        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
            else:
                super().dropEvent(e)

    def item_by_uuid(self, uuid):
        return self._item_cache[uuid]

    def _show_properties(self, uuid):
        for sig in self.plot.signals:
            if sig.uuid == uuid:
                if sig.computed:
                    view = ComputedChannelInfoWindow(sig, self)
                    view.show()

                else:
                    self.show_properties.emit(
                        [sig.group_index, sig.channel_index, sig.origin_uuid]
                    )

    def set_splitter(self, pos, index):
        self.splitter_moved.emit(self, pos)

    def pattern_group_added_req(self, group):
        self.pattern_group_added.emit(self, group)

    def set_timestamp(self, stamp):
        if self.plot.cursor1 is None:
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress,
                QtCore.Qt.Key_C,
                QtCore.Qt.NoModifier,
            )
            self.plot.keyPressEvent(event)

        self.plot.cursor1.setPos(stamp)
        self.cursor_move_finished()

    def compute_fft(self, uuid):
        signal, index = self.plot.signal_by_uuid(uuid)
        window = FFTWindow(PlotSignal(signal), parent=self)
        window.show()

    def clear(self):
        event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier
        )
        self.channel_selection.keyPressEvent(event)
        event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_Delete, QtCore.Qt.NoModifier
        )
        self.channel_selection.keyPressEvent(event)

    def update_current_values(self, *args):
        if self.plot.region:
            self.range_modified(None)
        else:
            self.cursor_moved()

    def adjust_splitter(self, channels=None):
        channels = channels or self.plot.signals

        size = sum(self.splitter.sizes())

        width = 0
        for ch in channels:
            width = max(
                width,
                self.channel_selection.fontMetrics()
                .boundingRect(f"{ch.name} ({ch.unit})")
                .width(),
            )
        width += 170

        if width > self.splitter.sizes()[0]:

            if size - width >= 300:
                self.splitter.setSizes([width, size - width, 0])
            else:
                if size >= 350:
                    self.splitter.setSizes([size - 300, 300, 0])
                elif size >= 100:
                    self.splitter.setSizes([50, size - 50, 0])

    def visible_entries(self):
        return self._visible_entries

    def visible_items(self):
        return self._visible_items

    def _update_visibile_entries(self):

        _item_cache = self._item_cache = {}
        _visible_entries = self._visible_entries = set()
        _visible_items = self._visible_items = {}
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while True:
            item = iterator.value()
            if item is None:
                break
            iterator += 1
            if item.type() == ChannelsTreeItem.Channel:

                _item_cache[item.uuid] = item

                if item.checkState(0) == QtCore.Qt.Checked and item.exists:
                    entry = (item.origin_uuid, item.signal.name, item.uuid)
                    _visible_entries.add(entry)
                    _visible_items[entry] = item
                else:
                    item.set_value("")

        if self.plot.cursor1 is not None:
            self.cursor_moved()

    def close(self):
        self.closed = True

        tree = self.channel_selection
        tree.plot = None
        iterator = QtWidgets.QTreeWidgetItemIterator(tree)
        while True:
            item = iterator.value()
            if item is None:
                break

            item.signal = None

            iterator += 1

        tree.clear()
        self._visible_items.clear()

        for sig in self.plot.signals:
            sig.enable = False
            del sig.plot_samples
            del sig.timestamps
            del sig.plot_timestamps
            del sig.samples
            del sig.phys_samples
            del sig.raw_samples
            sig._raw_samples = None
            sig._phys_samples = None
            sig._timestamps = None
        self.plot.signals.clear()
        self.plot._uuid_map.clear()
        self.plot._timebase_db.clear()
        self.plot.axes = None
        self.plot.plot_parent = None
        self.plot = None
        del self.plot

        super().close()


class _Plot(pg.PlotWidget):
    cursor_moved = QtCore.Signal(object)
    cursor_removed = QtCore.Signal()
    range_removed = QtCore.Signal()
    range_modified = QtCore.Signal(object)
    range_modified_finished = QtCore.Signal(object)
    cursor_move_finished = QtCore.Signal(object)
    xrange_changed = QtCore.Signal(object, object)
    computation_channel_inserted = QtCore.Signal()
    curve_clicked = QtCore.Signal(str)
    signals_enable_changed = QtCore.Signal()

    add_channels_request = QtCore.Signal(list)

    def __init__(
        self,
        signals=None,
        with_dots=False,
        origin=None,
        mdf=None,
        line_interconnect="line",
        x_axis="time",
        plot_parent=None,
        allow_cursor=True,
        *args,
        **kwargs,
    ):
        events = kwargs.pop("events", [])
        super().__init__()

        self.lock = Lock()

        self.plot_parent = plot_parent

        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)

        self._generate_pix = False
        self._grabbing = False
        self.copy_pixmap = True
        self.autoFillBackground()

        self._pixmap = None

        self.locked = False

        self.cursor_unit = "s" if x_axis == "time" else "Hz"

        self.line_interconnect = (
            line_interconnect if line_interconnect != "line" else ""
        )

        self._can_trim = True
        self._can_paint = True
        self.mdf = mdf

        self._update_lines_allowed = True

        self.setAcceptDrops(True)

        self._last_size = self.geometry()
        self._settings = QtCore.QSettings()

        self.setContentsMargins(5, 5, 5, 5)
        self.xrange_changed.connect(self.xrange_changed_handle)
        self.with_dots = with_dots

        self.info = None
        self.current_uuid = 0

        self.standalone = kwargs.get("standalone", False)

        self.region = None
        self.region_lock = None

        self.cursor1 = None
        self.cursor2 = None
        self.signals = []

        self.axes = []
        self._axes_layout_pos = 2

        self.disabled_keys = set()

        self._timebase_db = {}
        self._timestamps_indexes = {}
        self.all_timebase = self.timebase = np.array([])
        for sig in self.signals:
            uuids = self._timebase_db.setdefault(id(sig.timestamps), set())
            uuids.add(sig.uuid)

        #        self._compute_all_timebase()

        self.showGrid(x=True, y=True)

        self.plot_item = self.plotItem
        self.plot_item.hideButtons()
        self.plotItem.showGrid(x=False, y=False)
        self.layout = self.plot_item.layout
        self.scene_ = self.plot_item.scene()
        self.scene_.sigMouseClicked.connect(self._clicked)

        self.viewbox = self.plot_item.vb
        self.viewbox.border = None
        self.viewbox.disableAutoRange()

        self.x_range = (0, 1)
        self._curve = pg.PlotCurveItem(
            np.array([]),
            np.array([]),
            stepMode=self.line_interconnect,
            skipFiniteCheck=False,
            connect="finite",
        )

        self.viewbox.menu.removeAction(self.viewbox.menu.viewAll)
        for ax in self.viewbox.menu.axes:
            self.viewbox.menu.removeAction(ax.menuAction())
        self.plot_item.setMenuEnabled(False, None)

        self.common_axis_items = set()
        self.common_axis_label = ""
        self.common_axis_y_range = (0, 1)

        if allow_cursor:
            start, stop = self.viewbox.viewRange()[0]
            pos = QtCore.QPointF((start + stop) / 2, 0)

            if pg.getConfigOption("background") == "k":
                color = "white"
            else:
                color = "black"

            self.cursor1 = Cursor(
                pos=pos, angle=90, movable=True, pen=color, hoverPen=color
            )
            self.viewbox.addItem(self.cursor1, ignoreBounds=True)

            self.cursor1.sigPositionChanged.connect(self.cursor_moved.emit)
            self.cursor1.sigPositionChangeFinished.connect(
                self.cursor_move_finished.emit
            )
            self.cursor_move_finished.emit(self.cursor1)
            self.cursor1.show()
        else:
            self.cursor1 = None

        self.viewbox.sigYRangeChanged.connect(self.y_changed)
        self.viewbox.sigRangeChangedManually.connect(self.y_changed)

        self.x_axis = FormatedAxis("bottom")

        if x_axis == "time":
            fmt = self._settings.value("plot_xaxis")
            if fmt == "seconds":
                fmt = "phys"
        else:
            fmt = "phys"
        self.x_axis.format = fmt
        self.x_axis.origin = origin

        self.y_axis = FormatedAxis("left")
        self.y_axis.setWidth(48)

        self.plot_item.setAxisItems({"left": self.y_axis, "bottom": self.x_axis})

        self.viewbox_geometry = self.viewbox.sceneBoundingRect()

        self.resizeEvent = self._resizeEvent

        self._uuid_map = {}

        self._enabled_changed_signals = []
        self._enable_timer = QtCore.QTimer()
        self._enable_timer.setSingleShot(True)
        self._enable_timer.timeout.connect(self._signals_enabled_changed_handler)

        # self._paint_timer = QtCore.QTimer()
        # self._paint_timer.setSingleShot(True)
        # self._paint_timer.timeout.connect(self._paint)
        self._inhibit = False

        if signals:
            self.add_new_channels(signals)

        self.viewbox.sigXRangeChanged.connect(self.xrange_changed.emit)

        self.keyboard_events = set(
            [
                (QtCore.Qt.Key_F, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_F, QtCore.Qt.ShiftModifier),
                (QtCore.Qt.Key_G, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_I, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_O, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_X, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_R, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_S, QtCore.Qt.ControlModifier),
                (QtCore.Qt.Key_S, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_S, QtCore.Qt.ShiftModifier),
                (QtCore.Qt.Key_Y, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_Left, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_Right, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_Left, QtCore.Qt.ShiftModifier),
                (QtCore.Qt.Key_Right, QtCore.Qt.ShiftModifier),
                (QtCore.Qt.Key_Left, QtCore.Qt.ControlModifier),
                (QtCore.Qt.Key_Right, QtCore.Qt.ControlModifier),
                (QtCore.Qt.Key_Up, QtCore.Qt.ShiftModifier),
                (QtCore.Qt.Key_Down, QtCore.Qt.ShiftModifier),
                (QtCore.Qt.Key_PageUp, QtCore.Qt.ShiftModifier),
                (QtCore.Qt.Key_PageDown, QtCore.Qt.ShiftModifier),
                (QtCore.Qt.Key_H, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_Insert, QtCore.Qt.NoModifier),
            ]
        )
        self.keyboard_events = {
            (key, int(modif)) for key, modif in self.keyboard_events
        }

        events = events or []

        for i, event_info in enumerate(events):
            color = COLORS[len(COLORS) - (i % len(COLORS)) - 1]
            if isinstance(event_info, (list, tuple)):
                to_display = event_info
                labels = [" - Start", " - End"]
            else:
                to_display = [event_info]
                labels = [""]
            for event, label in zip(to_display, labels):
                description = f't = {event["value"]}s'
                if event["description"]:
                    description += f'\n\n{event["description"]}'
                line = pg.InfiniteLine(
                    pos=event["value"],
                    label=f'{event["type"]}{label}\n{description}',
                    pen={"color": color, "width": 2, "style": QtCore.Qt.DashLine},
                    labelOpts={
                        "border": {
                            "color": color,
                            "width": 2,
                            "style": QtCore.Qt.DashLine,
                        },
                        "fill": "#000000",
                        "color": color,
                        "movable": True,
                    },
                )
                self.plotItem.addItem(line, ignoreBounds=True)

        self.viewbox.sigResized.connect(self.update_views)
        if signals:
            self.update_views()

        self.px = 1
        self.py = 1

    def y_changed(self, *args):
        if len(args) == 1:
            # range manually changed by the user with the wheel
            mask = args[0]
            if mask[1]:
                y_range = self.viewbox.viewRange()[1]
            else:
                return
        else:
            # range changed by the linked axis
            y_range = args[1]

        update = False

        if self.current_uuid in self.common_axis_items:
            for uuid in self.common_axis_items:
                sig, idx = self.signal_by_uuid(uuid)
                if sig.y_range != y_range:
                    update = True
                    if sig.individual_axis:
                        axis = self.get_axis(idx)
                        axis.setRange(*y_range)

                sig.y_range = y_range

        elif self.current_uuid:
            sig, idx = self.signal_by_uuid(self.current_uuid)
            if sig.y_range != y_range:
                update = True
                if sig.individual_axis:
                    axis = self.get_axis(idx)
                    axis.setRange(*y_range)

            sig.y_range = y_range

        if update:
            self.update_plt()

    def set_y_range(self, uuid, y_range):
        sig, _ = self.signal_by_uuid(uuid)
        if (
            uuid == self.current_uuid
            or self.current_uuid in self.common_axis_items
            and uuid in self.common_axis_items
        ):
            if sig.y_range != y_range:
                sig.y_range = y_range
                self.y_axis.setRange(*y_range)
                self.update_plt()
        else:
            if sig.y_range != y_range:
                sig.y_range = y_range
                self.update_plt()

    def update_plt(self, *args, **kwargs):
        self.hide()
        self._pixmap = None
        self._generate_pix = True
        self.viewport().update()
        self.show()

    def set_locked(self, locked):
        self.locked = locked
        self.viewbox.setMouseEnabled(y=not self.locked)

    def set_dots(self, with_dots):

        self.with_dots = with_dots
        self.update_plt()

    def curve_clicked_handle(self, curve, ev, uuid):
        self.curve_clicked.emit(uuid)

    def set_line_interconnect(self, line_interconnect):
        self.line_interconnect = line_interconnect

        self._curve.setData(line_interconnect=line_interconnect)
        self.update_plt()

    def update_lines(self, update=True):

        if update:
            self.update_plt()

    def set_color(self, uuid, color):
        sig, index = self.signal_by_uuid(uuid)

        if sig.mode == "raw":
            style = QtCore.Qt.DashLine
        else:
            style = QtCore.Qt.SolidLine

        sig.pen = fn.mkPen(color=color, style=style)

        if sig.individual_axis:
            self.get_axis(index).set_pen(sig.pen)
            self.get_axis(index).setTextPen(sig.pen)

        if uuid == self.current_uuid:
            self.y_axis.set_pen(sig.pen)
            self.y_axis.setTextPen(sig.pen)

        self.update_plt()

    def set_unit(self, uuid, unit):

        sig, index = self.signal_by_uuid(uuid)
        sig.unit = unit

        sig_axis = [self.get_axis(index)]

        if uuid == self.current_uuid:
            sig_axis.append(self.y_axis)

        for axis in sig_axis:
            if len(sig.name) <= 32:
                if sig.unit:
                    axis.setLabel(f"{sig.name} [{sig.unit}]")
                else:
                    axis.setLabel(f"{sig.name}")
            else:
                if sig.unit:
                    axis.setLabel(f"{sig.name[:29]}...  [{sig.unit}]")
                else:
                    axis.setLabel(f"{sig.name[:29]}...")
            axis.update()

        self.update_plt()

    def set_name(self, uuid, name):

        sig, index = self.signal_by_uuid(uuid)
        sig.name = name

        sig_axis = [self.get_axis(index)]

        if uuid == self.current_uuid:
            sig_axis.append(self.y_axis)

        for axis in sig_axis:
            if len(sig.name) <= 32:
                if sig.unit:
                    axis.setLabel(f"{sig.name} [{sig.unit}]")
                else:
                    axis.setLabel(f"{sig.name}")
            else:
                if sig.unit:
                    axis.setLabel(f"{sig.name[:29]}...  [{sig.unit}]")
                else:
                    axis.setLabel(f"{sig.name[:29]}...")
            axis.update()
        self.update_plt()

    def set_common_axis(self, uuid, state):

        signal, idx = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.Checked, True, 1):
            if not self.common_axis_items:
                self.common_axis_y_range = signal.y_range
            else:
                signal.y_range = self.common_axis_y_range
            self.common_axis_items.add(uuid)
        else:
            self.common_axis_items.remove(uuid)

        self.common_axis_label = ", ".join(
            self.signal_by_uuid(uuid)[0].name for uuid in self.common_axis_items
        )

        self.set_current_uuid(self.current_uuid, True)
        self.update_plt()

    def set_individual_axis(self, uuid, state):

        sig, index = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.Checked, True, 1):
            if sig.enable:
                self.get_axis(index).show()
            sig.individual_axis = True
        else:
            self.get_axis(index).hide()
            sig.individual_axis = False

        self.update_plt()

    def set_signal_enable(self, uuid, state):

        signal, index = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.Checked, True, 1):
            (start, stop), _ = self.viewbox.viewRange()
            width = self.width() - self.y_axis.width()

            signal.enable = True

            signal.trim(start, stop, width)
            if signal.individual_axis:
                self.get_axis(index).show()

            uuids = self._timebase_db.setdefault(id(signal.timestamps), set())
            uuids.add(signal.uuid)

        else:
            signal.enable = False
            if signal.individual_axis:
                self.get_axis(index).hide()

            try:
                self._timebase_db[id(signal.timestamps)].remove(uuid)

                if len(self._timebase_db[id(signal.timestamps)]) == 0:
                    del self._timebase_db[id(signal.timestamps)]
            except:
                pass

        self._enable_timer.start(50)

    def _signals_enabled_changed_handler(self):

        self._compute_all_timebase()
        self.update_lines()
        if self.cursor1:
            self.cursor_move_finished.emit(self.cursor1)
        self.signals_enable_changed.emit()
        self.update_plt()

    def update_views(self):
        geometry = self.viewbox.sceneBoundingRect()
        if geometry != self.viewbox_geometry:
            self._pixmap = None
            self._generate_pix = True
            self.viewbox_geometry = geometry

    def get_stats(self, uuid):
        sig, index = self.signal_by_uuid(uuid)

        return sig.get_stats(
            cursor=self.cursor1.value() if self.cursor1 else None,
            region=self.region.getRegion() if self.region else None,
            view_region=self.viewbox.viewRange()[0],
        )

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if (key, int(modifier)) in self.disabled_keys:
            super().keyPressEvent(event)
        else:
            handled = True
            if key == QtCore.Qt.Key_Y and modifier == QtCore.Qt.NoModifier:
                if self.region is not None:
                    if self.region_lock is not None:
                        self.region_lock = None
                        self.region.lines[0].pen.setStyle(QtCore.Qt.SolidLine)
                        self.region.lines[1].pen.setStyle(QtCore.Qt.SolidLine)
                    else:
                        self.region_lock = self.region.getRegion()[0]
                        self.region.lines[0].pen.setStyle(QtCore.Qt.DashDotDotLine)

                else:
                    self.region_lock = None

            elif key == QtCore.Qt.Key_X and modifier == QtCore.Qt.NoModifier:
                if self.region is not None:
                    self.viewbox.setXRange(*self.region.getRegion(), padding=0)
                    event_ = QtGui.QKeyEvent(
                        QtCore.QEvent.KeyPress, QtCore.Qt.Key_R, QtCore.Qt.NoModifier
                    )
                    self.keyPressEvent(event_)

            elif (
                key == QtCore.Qt.Key_F
                and modifier == QtCore.Qt.NoModifier
                and not self.locked
            ):

                if self.common_axis_items:
                    if any(
                        len(self.signal_by_uuid(uuid)[0].plot_samples)
                        for uuid in self.common_axis_items
                        if self.signal_by_uuid(uuid)[0].enable
                    ):

                        common_min = np.nanmin(
                            [
                                np.nanmin(self.signal_by_uuid(uuid)[0].plot_samples)
                                for uuid in self.common_axis_items
                                if len(self.signal_by_uuid(uuid)[0].plot_samples)
                            ]
                        )
                        common_max = np.nanmax(
                            [
                                np.nanmax(self.signal_by_uuid(uuid)[0].plot_samples)
                                for uuid in self.common_axis_items
                                if len(self.signal_by_uuid(uuid)[0].plot_samples)
                            ]
                        )

                for i, signal in enumerate(self.signals):
                    if len(signal.plot_samples):
                        if signal.uuid in self.common_axis_items:
                            min_ = common_min
                            max_ = common_max
                        else:
                            samples = signal.plot_samples[
                                np.isfinite(signal.plot_samples)
                            ]
                            if len(samples):
                                min_, max_ = (
                                    np.nanmin(samples),
                                    np.nanmax(samples),
                                )
                            else:
                                min_, max_ = 0, 1

                        if min_ != min_:
                            min_ = 0
                        if max_ != max_:
                            max_ = 1

                        signal.y_range = min_, max_

                self.update_plt()

            elif (
                key == QtCore.Qt.Key_F
                and modifier == QtCore.Qt.ShiftModifier
                and not self.locked
            ):

                parent = self.parent().parent()
                uuids = [
                    item.uuid
                    for item in parent.channel_selection.selectedItems()
                    if item.type() == ChannelsTreeItem.Channel
                ]
                uuids = set(uuids)

                if not uuids:
                    return

                for i, signal in enumerate(self.signals):
                    if signal.uuid not in uuids:
                        continue

                    if len(signal.plot_samples):
                        samples = signal.plot_samples[np.isfinite(signal.plot_samples)]
                        if len(samples):
                            min_, max_ = (
                                np.nanmin(samples),
                                np.nanmax(samples),
                            )
                        else:
                            min_, max_ = 0, 1

                        if min_ != min_:
                            min_ = 0
                        if max_ != max_:
                            max_ = 1

                        signal.y_range = min_, max_

                self.update_plt()

            elif key == QtCore.Qt.Key_G and modifier == QtCore.Qt.NoModifier:

                y = self.plotItem.ctrl.yGridCheck.isChecked()
                x = self.plotItem.ctrl.xGridCheck.isChecked()

                if x and y:
                    self.plotItem.showGrid(x=False, y=False)
                elif x:
                    self.plotItem.showGrid(x=True, y=True)
                else:
                    self.plotItem.showGrid(x=True, y=False)

                self.update_plt()

            elif (
                key in (QtCore.Qt.Key_I, QtCore.Qt.Key_O)
                and modifier == QtCore.Qt.NoModifier
            ):

                x_range, _ = self.viewbox.viewRange()
                delta = x_range[1] - x_range[0]
                step = delta * 0.05
                if key == QtCore.Qt.Key_I:
                    step = -step
                if self.cursor1:
                    pos = self.cursor1.value()
                    x_range = pos - delta / 2, pos + delta / 2
                self.viewbox.setXRange(x_range[0] - step, x_range[1] + step, padding=0)

            elif key == QtCore.Qt.Key_R and modifier == QtCore.Qt.NoModifier:
                if self.region is None:
                    if pg.getConfigOption("background") == "k":
                        color = "white"
                    else:
                        color = "black"

                    self.region = Region((0, 0), pen=color, hoverPen=color)
                    self.region.setZValue(-10)
                    self.viewbox.addItem(self.region)
                    self.region.sigRegionChanged.connect(self.range_modified.emit)
                    self.region.sigRegionChangeFinished.connect(
                        self.range_modified_finished_handler
                    )
                    start, stop = self.viewbox.viewRange()[0]
                    start, stop = (
                        start + 0.1 * (stop - start),
                        stop - 0.1 * (stop - start),
                    )
                    self.region.setRegion((start, stop))

                    if self.cursor1 is not None:
                        self.cursor1.hide()
                        self.region.setRegion(
                            tuple(sorted((self.cursor1.value(), stop)))
                        )

                else:
                    self.region_lock = None
                    self.region.setParent(None)
                    self.region.hide()
                    self.region = None
                    self.range_removed.emit()

                    if self.cursor1 is not None:
                        self.cursor1.show()

                self.update_plt()

            elif key == QtCore.Qt.Key_S and modifier == QtCore.Qt.ControlModifier:
                file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Select output measurement file",
                    "",
                    "MDF version 4 files (*.mf4)",
                )

                if file_name:
                    signals = [signal for signal in self.signals if signal.enable]
                    if signals:
                        with MDF() as mdf:
                            groups = {}
                            for sig in signals:
                                id_ = id(sig.timestamps)
                                group_ = groups.setdefault(id_, [])
                                group_.append(sig)

                            for signals in groups.values():
                                sigs = []
                                for signal in signals:
                                    if ":" in signal.name:
                                        sig = signal.copy()
                                        sig.name = sig.name.split(":")[-1].strip()
                                        sigs.append(sig)
                                    else:
                                        sigs.append(signal)
                                mdf.append(sigs, common_timebase=True)
                            mdf.save(file_name, overwrite=True)

            elif (
                key == QtCore.Qt.Key_S
                and modifier == QtCore.Qt.NoModifier
                and not self.locked
            ):

                parent = self.parent().parent()
                uuids = []

                iterator = QtWidgets.QTreeWidgetItemIterator(parent.channel_selection)
                while True:
                    item = iterator.value()
                    if item is None:
                        break

                    if item.type() == ChannelsTreeItem.Channel and item.signal.enable:
                        uuids.append(item.uuid)

                    iterator += 1

                uuids = reversed(uuids)

                count = sum(
                    1
                    for sig in self.signals
                    if sig.min != "n.a."
                    and sig.enable
                    and sig.uuid not in self.common_axis_items
                )

                if any(
                    sig.min != "n.a."
                    and sig.enable
                    and sig.uuid in self.common_axis_items
                    for sig in self.signals
                ):
                    count += 1

                    common_min_ = np.nanmin(
                        [
                            np.nanmin(self.signal_by_uuid(uuid)[0].plot_samples)
                            for uuid in self.common_axis_items
                            if len(self.signal_by_uuid(uuid)[0].plot_samples)
                            and self.signal_by_uuid(uuid)[0].enable
                        ]
                    )
                    common_max_ = np.nanmax(
                        [
                            np.nanmax(self.signal_by_uuid(uuid)[0].plot_samples)
                            for uuid in self.common_axis_items
                            if len(self.signal_by_uuid(uuid)[0].plot_samples)
                            and self.signal_by_uuid(uuid)[0].enable
                        ]
                    )

                if count:

                    position = 0
                    common_axis_handled = False
                    for uuid in uuids:
                        signal, index = self.signal_by_uuid(uuid)

                        if not signal.empty and signal.enable:
                            if signal.uuid in self.common_axis_items:
                                if common_axis_handled:
                                    continue

                                min_ = common_min_
                                max_ = common_max_

                            else:
                                min_ = signal.min
                                max_ = signal.max

                            if min_ == -float("inf") and max_ == float("inf"):
                                min_ = 0
                                max_ = 1
                            elif min_ == -float("inf"):
                                min_ = max_ - 1
                            elif max_ == float("inf"):
                                max_ = min_ + 1

                            if min_ == max_:
                                min_, max_ = min_ - 1, max_ + 1

                            dim = (float(max_) - min_) * 1.1

                            max_ = min_ + dim * count - 0.05 * dim
                            min_ = min_ - 0.05 * dim

                            min_, max_ = (
                                min_ - dim * position,
                                max_ - dim * position,
                            )

                            if signal.uuid in self.common_axis_items:
                                y_range = min_, max_
                                self.common_axis_y_range = y_range
                                for cuuid in self.common_axis_items:
                                    sig, _ = self.signal_by_uuid(cuuid)
                                    sig.y_range = y_range

                                common_axis_handled = True

                            else:
                                signal.y_range = min_, max_

                            position += 1

                else:
                    xrange, _ = self.viewbox.viewRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.setXRange(*xrange, padding=0)
                    self.viewbox.disableAutoRange()

                self.update_plt()

            elif (
                key == QtCore.Qt.Key_S
                and modifier == QtCore.Qt.ShiftModifier
                and not self.locked
            ):

                parent = self.parent().parent()
                uuids = [
                    item.uuid
                    for item in parent.channel_selection.selectedItems()
                    if item.type() == ChannelsTreeItem.Channel
                ]
                uuids = list(reversed(uuids))
                uuids_set = set(uuids)

                if not uuids:
                    return

                count = sum(
                    1
                    for i, sig in enumerate(self.signals)
                    if sig.uuid in uuids_set and sig.min != "n.a." and sig.enable
                )

                if count:

                    common_axis_handled = False
                    position = 0
                    for uuid in uuids:
                        signal, index = self.signal_by_uuid(uuid)

                        if not signal.empty and signal.enable:

                            if uuid in self.common_axis_items:
                                if common_axis_handled:
                                    continue

                                min_ = np.nanmin(
                                    [
                                        np.nanmin(
                                            self.signal_by_uuid(uuid)[0].plot_samples
                                        )
                                        for uuid in self.common_axis_items
                                        if uuid in uuids_set
                                        and len(
                                            self.signal_by_uuid(uuid)[0].plot_samples
                                        )
                                        and self.signal_by_uuid(uuid)[0].enable
                                    ]
                                )
                                max_ = np.nanmax(
                                    [
                                        np.nanmax(
                                            self.signal_by_uuid(uuid)[0].plot_samples
                                        )
                                        for uuid in self.common_axis_items
                                        if uuid in uuids_set
                                        and len(
                                            self.signal_by_uuid(uuid)[0].plot_samples
                                        )
                                        and self.signal_by_uuid(uuid)[0].enable
                                    ]
                                )

                            else:
                                min_ = signal.min
                                max_ = signal.max

                            if min_ == -float("inf") and max_ == float("inf"):
                                min_ = 0
                                max_ = 1
                            elif min_ == -float("inf"):
                                min_ = max_ - 1
                            elif max_ == float("inf"):
                                max_ = min_ + 1

                            if min_ == max_:
                                min_, max_ = min_ - 1, max_ + 1

                            dim = (float(max_) - min_) * 1.1

                            max_ = min_ + dim * count - 0.05 * dim
                            min_ = min_ - 0.05 * dim

                            min_, max_ = (
                                min_ - dim * position,
                                max_ - dim * position,
                            )

                            if signal.uuid in self.common_axis_items:
                                y_range = min_, max_
                                self.common_axis_y_range = y_range
                                for cuuid in self.common_axis_items:
                                    sig, _ = self.signal_by_uuid(cuuid)
                                    sig.y_range = y_range

                                common_axis_handled = True

                            else:
                                signal.y_range = min_, max_

                            position += 1

                else:
                    xrange, _ = self.viewbox.viewRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.setXRange(*xrange, padding=0)
                    self.viewbox.disableAutoRange()

                self.update_plt()

            elif key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right) and modifier in (
                QtCore.Qt.NoModifier,
                QtCore.Qt.ControlModifier,
            ):
                if modifier == QtCore.Qt.ControlModifier:
                    increment = 20
                else:
                    increment = 1
                if self.cursor1:
                    prev_pos = pos = self.cursor1.value()
                    x = self.get_current_timebase()
                    dim = x.size
                    if dim:
                        pos = np.searchsorted(x, pos)
                        if key == QtCore.Qt.Key_Right:
                            pos += increment
                        else:
                            pos -= increment
                        pos = np.clip(pos, 0, dim - increment)
                        pos = x[pos]
                    else:
                        if key == QtCore.Qt.Key_Right:
                            pos += increment
                        else:
                            pos -= increment

                    (left_side, right_side), _ = self.viewbox.viewRange()

                    if pos >= right_side:
                        delta = abs(pos - prev_pos)
                        self.viewbox.setXRange(
                            left_side + delta, right_side + delta, padding=0
                        )
                    elif pos <= left_side:
                        delta = abs(pos - prev_pos)
                        self.viewbox.setXRange(
                            left_side - delta, right_side - delta, padding=0
                        )

                    self.cursor1.set_value(pos)

            elif (
                key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right)
                and modifier == QtCore.Qt.ShiftModifier
            ):
                parent = self.parent().parent()
                uuids = list(
                    set(
                        item.uuid
                        for item in parent.channel_selection.selectedItems()
                        if item.type() == ChannelsTreeItem.Channel
                    )
                )

                if not uuids:
                    return

                start, stop = self.viewbox.viewRange()[0]

                offset = (stop - start) / 100

                if key == QtCore.Qt.Key_Left:
                    offset = -offset

                self.set_time_offset([False, offset, *uuids])

            elif (
                key
                in (
                    QtCore.Qt.Key_Up,
                    QtCore.Qt.Key_Down,
                    QtCore.Qt.Key_PageUp,
                    QtCore.Qt.Key_PageDown,
                )
                and modifier == QtCore.Qt.ShiftModifier
            ):
                parent = self.parent().parent()
                uuids = list(
                    set(
                        item.uuid
                        for item in parent.channel_selection.selectedItems()
                        if item.type() == ChannelsTreeItem.Channel
                    )
                )

                if not uuids:
                    return

                factor = (
                    10 if key in (QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown) else 100
                )

                for uuid in uuids:
                    signal, index = self.signal_by_uuid(uuid)

                    bottom, top = signal.y_range
                    step = (top - bottom) / factor

                    if key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_PageUp):
                        step = -step

                    signal.y_range = bottom + step, top + step

                self.update_plt()

            elif (
                key == QtCore.Qt.Key_H
                and modifier == QtCore.Qt.NoModifier
                and not self.locked
            ):

                if len(self.all_timebase):
                    start_ts = np.amin(self.all_timebase)
                    stop_ts = np.amax(self.all_timebase)

                    self.viewbox.setXRange(start_ts, stop_ts)
                    event_ = QtGui.QKeyEvent(
                        QtCore.QEvent.KeyPress, QtCore.Qt.Key_F, QtCore.Qt.NoModifier
                    )
                    self.keyPressEvent(event_)

                    if self.cursor1:
                        self.cursor_moved.emit(self.cursor1)

            elif key == QtCore.Qt.Key_Insert and modifier == QtCore.Qt.NoModifier:
                self.insert_computation()

            else:
                handled = False

            if handled:
                self.update_plt()
            else:
                self.parent().keyPressEvent(event)

    def range_modified_finished_handler(self, region):
        if self.region_lock is not None:
            for i in range(2):
                if self.region.lines[i].value() == self.region_lock:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.DashDotDotLine)
                else:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.SolidLine)
        self.range_modified_finished.emit(region)

    def trim(self, signals=None, force=False, view_range=None):
        signals = signals or self.signals
        if not self._can_trim:
            return

        if view_range is None:
            (start, stop), _ = self.viewbox.viewRange()
        else:
            start, stop = view_range

        width = self.width() - self.y_axis.width()

        for sig in signals:
            if sig.enable:
                sig.trim(start, stop, width, force)

    def xrange_changed_handle(self, *, force=False):
        self.x_range = tuple(self.viewbox.viewRange()[0])
        self.px = (self.viewport().width() - self.y_axis.width() - 2) / (
            self.x_range[1] - self.x_range[0]
        )

        if self._update_lines_allowed:
            self.trim(force=force)
            self.update_lines()
            self.update_plt()

    def _resizeEvent(self, ev):
        new_size, last_size = self.geometry(), self._last_size
        if new_size != last_size:
            self.px = (self.viewport().width() - self.y_axis.width() - 2) / (
                self.x_range[1] - self.x_range[0]
            )
            self.py = self.viewport().height() - self.x_axis.height() - 2
            self._last_size = new_size
            super().resizeEvent(ev)
            self.xrange_changed_handle()

    def set_current_uuid(self, uuid, force=False):
        axis = self.y_axis
        viewbox = self.viewbox

        sig, index = self.signal_by_uuid(uuid)

        if sig.conversion and hasattr(sig.conversion, "text_0"):
            axis.text_conversion = sig.conversion
        else:
            axis.text_conversion = None
        axis.format = sig.format

        if uuid in self.common_axis_items:
            if self.current_uuid not in self.common_axis_items or force:

                if self._settings.value("plot_background") == "Black":
                    axis.set_pen(fn.mkPen("#FFFFFF"))
                    axis.setTextPen("#FFFFFF")
                else:
                    axis.set_pen(fn.mkPen("#000000"))
                    axis.setTextPen("#000000")
                axis.setLabel(self.common_axis_label)

        else:

            if len(sig.name) <= 32:
                if sig.unit:
                    axis.setLabel(f"{sig.name} [{sig.unit}]")
                else:
                    axis.setLabel(f"{sig.name}")
            else:
                if sig.unit:
                    axis.setLabel(f"{sig.name[:29]}...  [{sig.unit}]")
                else:
                    axis.setLabel(f"{sig.name[:29]}...")

            axis.set_pen(sig.pen)
            axis.setTextPen(sig.pen)
            axis.update()

        self.current_uuid = uuid
        viewbox.setYRange(*sig.y_range, padding=0)
        self.update_plt()

    def _clicked(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        pos = self.plot_item.vb.mapSceneToView(event.scenePos()).x()
        start, stop = self.viewbox.viewRange()[0]
        if not start <= pos <= stop:
            return

        pos = self.plot_item.vb.mapSceneToView(event.scenePos())
        x = pos.x()
        y = event.scenePos().y()

        if (QtCore.Qt.Key_C, int(QtCore.Qt.NoModifier)) not in self.disabled_keys:

            if self.region is not None:
                start, stop = self.region.getRegion()

                if self.region_lock is not None:
                    self.region.setRegion((self.region_lock, pos.x()))
                else:
                    if modifiers == QtCore.Qt.ControlModifier:
                        self.region.setRegion((start, pos.x()))
                    else:
                        self.region.setRegion((pos.x(), stop))

            else:
                if self.cursor1 is not None:
                    self.cursor1.setPos(pos)
                    self.cursor1.sigPositionChangeFinished.emit(self.cursor1)

        delta = self.y_axis.width() + 1
        x_range = self.viewbox.viewRange()[0]
        for sig in self.signals:
            if not sig.enable:
                continue

            val, _1, _2 = sig.value_at_timestamp(x, numeric=True)

            if val == "n.a.":
                continue

            x_val, y_val = self.scale_curve_to_pixmap(
                x, val, y_range=sig.y_range, x_range=x_range, delta=delta
            )

            if abs(y_val - y) <= 15:
                self.curve_clicked.emit(sig.uuid)
                break

    def add_new_channels(self, channels, computed=False, descriptions=None):
        self._can_paint = False
        descriptions = descriptions or {}

        initial_index = len(self.signals)
        self._update_lines_allowed = False

        for sig in channels.values():
            if not hasattr(sig, "computed"):
                sig.computed = computed
                if not computed:
                    sig.computation = {}

        (start, stop), _ = self.viewbox.viewRange()

        width = self.width() - self.y_axis.width()
        trim_info = start, stop, width

        channels = [
            PlotSignal(sig, i, trim_info=trim_info)
            for i, sig in enumerate(channels.values(), len(self.signals))
        ]

        for sig in channels:
            uuids = self._timebase_db.setdefault(id(sig.timestamps), set())
            uuids.add(sig.uuid)
        self.signals.extend(channels)

        self._uuid_map = {sig.uuid: (sig, i) for i, sig in enumerate(self.signals)}

        self._compute_all_timebase()

        if initial_index == 0 and len(self.all_timebase):
            start_t, stop_t = np.amin(self.all_timebase), np.amax(self.all_timebase)
            self.viewbox.setXRange(start_t, stop_t, update=False)

        axis_uuid = None

        for index, sig in enumerate(channels, initial_index):
            description = descriptions.get(sig.uuid, {})
            if description:
                sig.enable = description.get("enabled", True)

            if not sig.empty:
                if description.get("y_range", None):
                    sig.y_range = tuple(description["y_range"])
                else:
                    sig.y_range = sig.min, sig.max
            elif description.get("y_range", None):
                sig.y_range = tuple(description["y_range"])

            self.axes.append(self._axes_layout_pos)
            self._axes_layout_pos += 1

            if initial_index == 0 and index == 0:
                axis_uuid = sig.uuid

        if axis_uuid is not None:
            self.set_current_uuid(sig.uuid)

        self._update_lines_allowed = True
        self._can_paint = True
        self.xrange_changed_handle(force=True)

        return {sig.uuid: sig for sig in channels}

    def _compute_all_timebase(self):
        if self._timebase_db:
            stamps = {id(sig.timestamps): sig.timestamps for sig in self.signals}

            timebases = [
                timestamps
                for id_, timestamps in stamps.items()
                if id_ in self._timebase_db
            ]

            count = len(timebases)

            if count == 0:
                new_timebase = np.array([])
            elif count == 1:
                new_timebase = timebases[0]
            else:
                try:
                    new_timebase = np.unique(np.concatenate(timebases))
                except MemoryError:
                    new_timebase = reduce(np.union1d, timebases)

            self.all_timebase = self.timebase = new_timebase
        else:
            self.all_timebase = self.timebase = np.array([])

    def get_current_timebase(self):
        if self.current_uuid:
            sig, _ = self._uuid_map[self.current_uuid]
            t = sig.timestamps
            if t.size:
                return t
            else:
                return self.all_timebase
        else:
            return self.all_timebase

    def signal_by_uuid(self, uuid):
        return self._uuid_map[uuid]

    def signal_by_name(self, name):
        for i, sig in enumerate(self.signals):
            if sig.name == name:
                return sig, i

        raise Exception(
            f"Signal not found: {name} {[sig.name for sig in self.signals]}"
        )

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat("application/octet-stream-asammdf"):
            e.accept()
        super().dragEnterEvent(e)

    def dropEvent(self, e):
        if e.source() is self.parent().channel_selection:
            super().dropEvent(e)
        else:
            data = e.mimeData()
            if data.hasFormat("application/octet-stream-asammdf"):
                names = extract_mime_names(data)
                self.add_channels_request.emit(names)
            else:
                super().dropEvent(e)

    def delete_channels(self, deleted):
        self._can_paint = False

        needs_timebase_compute = False

        indexes = sorted(
            [(self.signal_by_uuid(uuid)[1], uuid) for uuid in deleted], reverse=True
        )

        for i, uuid in indexes:

            item = self.axes.pop(i)
            if isinstance(item, FormatedAxis):
                self.layout.removeItem(item)
                item.scene().removeItem(item)
                item.unlinkFromView()

            sig = self.signals.pop(i)

            if uuid in self.common_axis_items:
                self.common_axis_items.remove(uuid)

            if sig.enable:
                try:
                    self._timebase_db[id(sig.timestamps)].remove(sig.uuid)

                    if len(self._timebase_db[id(sig.timestamps)]) == 0:
                        del self._timebase_db[id(sig.timestamps)]
                        needs_timebase_compute = True
                except KeyError:
                    pass

        uuids = [sig.uuid for sig in self.signals]

        self._uuid_map = {sig.uuid: (sig, i) for i, sig in enumerate(self.signals)}

        if uuids:
            if self.current_uuid in uuids:
                self.set_current_uuid(self.current_uuid, True)
            else:
                self.set_current_uuid(uuids[0], True)
        else:
            self.current_uuid = None

        if needs_timebase_compute:
            self._compute_all_timebase()

        self._can_paint = True
        self.update_plt()

    def set_time_offset(self, info):
        absolute, offset, *uuids = info

        signals = [sig for sig in self.signals if sig.uuid in uuids]

        if absolute:
            for sig in signals:
                if not len(sig.timestamps):
                    continue
                id_ = id(sig.timestamps)
                delta = sig.timestamps[0] - offset
                sig.timestamps = sig.timestamps - delta

                uuids = self._timebase_db.setdefault(id(sig.timestamps), set())
                uuids.add(sig.uuid)

                self._timebase_db[id_].remove(sig.uuid)
                if len(self._timebase_db[id_]) == 0:
                    del self._timebase_db[id_]
        else:
            for sig in signals:
                if not len(sig.timestamps):
                    continue
                id_ = id(sig.timestamps)

                sig.timestamps = sig.timestamps + offset

                uuids = self._timebase_db.setdefault(id(sig.timestamps), set())
                uuids.add(sig.uuid)

                self._timebase_db[id_].remove(sig.uuid)
                if len(self._timebase_db[id_]) == 0:
                    del self._timebase_db[id_]

        self._compute_all_timebase()

        self.xrange_changed_handle(force=True)

    def insert_computation(self, name=""):
        dlg = DefineChannel(self.signals, self.all_timebase, name, self.mdf, self)
        dlg.setModal(True)
        dlg.exec_()
        sig = dlg.result

        if sig is not None:
            sig.uuid = os.urandom(6).hex()
            sig.group_index = -1
            sig.channel_index = -1
            sig.origin_uuid = os.urandom(6).hex()
            self.add_new_channels({sig.name: sig}, computed=True)
            self.computation_channel_inserted.emit()

    def get_axis(self, index):
        axis = self.axes[index]
        if isinstance(axis, int):
            sig = self.signals[index]
            position = axis

            axis = FormatedAxis(
                "left",
                pen=sig.pen,
                textPen=sig.pen,
                text=sig.name if len(sig.name) <= 32 else "{sig.name[:29]}...",
                units=sig.unit,
                uuid=sig.uuid,
            )
            if sig.conversion and hasattr(sig.conversion, "text_0"):
                axis.text_conversion = sig.conversion

            axis.setRange(*sig.y_range)

            axis.rangeChanged.connect(self.set_y_range)
            self.layout.addItem(axis, 2, position)

            self.axes[index] = axis

        return axis

    def scale_curve_to_pixmap(self, x, y, y_range, x_range, delta):

        if not self.py:
            all_bad = True

        else:
            y_scale = (float(y_range[1]) - float(y_range[0])) / self.py
            x_scale = self.px

            if y_scale * x_scale:
                all_bad = False
            else:
                all_bad = True

        if all_bad:
            try:
                y = np.full(len(y), np.inf)
            except:
                y = np.inf
        else:

            xs = x_range[0]
            ys = y_range[1]

            # x = (x - xs) / x_scale + delta
            # y = (ys - y) / y_scale + 1
            # is rewriten as

            xs = xs - delta * x_scale
            ys = ys + y_scale

            x = (x - xs) / x_scale
            y = (ys - y) / y_scale

        return x, y

    def auto_clip_rect(self, painter):
        painter.setClipRect(self.viewbox.sceneBoundingRect())
        painter.setClipping(True)

    def generatePath(self, x, y):
        if x is None or len(x) == 0 or y is None or len(y) == 0:
            return QtGui.QPainterPath()
        else:
            return self._curve.generatePath(x, y)

    def paintEvent(self, ev):
        if not self._can_paint:
            return

        if self._pixmap is None:
            super().paintEvent(ev)

        if self._generate_pix:
            self._generate_pix = False
            self._grabbing = True
            self._pixmap = self.grab()
            # self._grabbing = False
            paint = QtGui.QPainter()
            paint.begin(self._pixmap)
            paint.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            paint.setRenderHint(paint.RenderHint.Antialiasing, True)

            self.x_range, self.y_range = self.viewbox.viewRange()
            rect = self.viewbox.sceneBoundingRect()

            self.px = (self.x_range[1] - self.x_range[0]) / rect.width()
            self.py = rect.height()

            with_dots = self.with_dots

            self.auto_clip_rect(paint)

            rect = self.viewbox.sceneBoundingRect()

            delta = rect.x()
            x_range = self.x_range

            no_brush = QtGui.QBrush()

            for i, sig in enumerate(self.signals):
                if not sig.enable:
                    continue

                y = sig.plot_samples
                x = sig.plot_timestamps

                x, y = self.scale_curve_to_pixmap(
                    x, y, y_range=sig.y_range, x_range=x_range, delta=delta
                )

                sig.pen.setWidth(1)

                paint.resetTransform()
                paint.translate(0, 0)
                paint.setPen(sig.pen)
                paint.setBrush(no_brush)
                paint.drawPath(self.generatePath(x, y))
                paint.resetTransform()
                paint.translate(0, 0)

                if with_dots:
                    pos = np.isfinite(y)
                    y = y[pos]
                    x = x[pos]

                    _pen = fn.mkPen(sig.color.name())
                    _pen.setWidth(4)
                    _pen.setCapStyle(QtCore.Qt.RoundCap)
                    paint.setPen(_pen)

                    poly, arr = polygon_and_ndarray(x.size)

                    arr[:, 0] = x
                    arr[:, 1] = y
                    paint.drawPoints(poly)

                    paint.setBrush(no_brush)

                item = self.plot_parent.item_by_uuid(sig.uuid)
                if not item:
                    continue

                ranges = item.get_ranges()

                if ranges:
                    for range_info in ranges:
                        val = range_info["value1"]
                        if val is not None and isinstance(val, float):
                            op = range_info["op1"]
                            if op == ">":
                                idx1 = sig.plot_samples < val
                            elif op == ">=":
                                idx1 = sig.plot_samples <= val
                            elif op == "<":
                                idx1 = sig.plot_samples > val
                            elif op == "<=":
                                idx1 = sig.plot_samples >= val
                            elif op == "==":
                                idx1 = sig.plot_samples == val
                            elif op == "!=":
                                idx1 = sig.plot_samples != val
                        else:
                            idx1 = None

                        val = range_info["value2"]
                        if val is not None and isinstance(val, float):
                            op = range_info["op2"]
                            if op == ">":
                                idx2 = sig.plot_samples > val
                            elif op == ">=":
                                idx2 = sig.plot_samples >= val
                            elif op == "<":
                                idx2 = sig.plot_samples < val
                            elif op == "<=":
                                idx2 = sig.plot_samples <= val
                            elif op == "==":
                                idx2 = sig.plot_samples == val
                            elif op == "!=":
                                idx2 = sig.plot_samples != val
                        else:
                            idx2 = None

                        if idx1 is not None or idx2 is not None:
                            if idx1 is None:
                                idx = idx2
                            elif idx2 is None:
                                idx = idx1
                            else:
                                idx = idx1 & idx2

                            if not np.any(idx):
                                continue

                            y = sig.plot_samples.astype("f8")
                            y[~idx] = np.inf
                            x = sig.plot_timestamps

                            x, y = self.scale_curve_to_pixmap(
                                x, y, y_range=sig.y_range, x_range=x_range, delta=delta
                            )

                            color = range_info["font_color"]
                            pen = fn.mkPen(color.name())
                            pen.setWidth(1)

                            paint.resetTransform()
                            paint.translate(0, 0)
                            paint.setPen(pen)
                            paint.setBrush(no_brush)
                            paint.drawPath(self.generatePath(x, y))
                            paint.resetTransform()
                            paint.translate(0, 0)

                            if with_dots:
                                pen.setWidth(4)
                                pen.setCapStyle(QtCore.Qt.RoundCap)
                                paint.setPen(pen)

                                pos = np.isfinite(y)
                                y = y[pos]
                                x = x[pos]

                                poly, arr = polygon_and_ndarray(x.size)

                                arr[:, 0] = x
                                arr[:, 1] = y
                                paint.drawPoints(poly)

                                paint.setBrush(no_brush)

            paint.end()

        if self._pixmap is not None:

            paint = QtGui.QPainter()
            vp = self.viewport()
            paint.begin(vp)
            paint.setCompositionMode(QtGui.QPainter.CompositionMode_Source)

            rect = ev.rect()

            if self.copy_pixmap:
                paint.drawPixmap(rect, self._pixmap.copy(), rect)
            else:
                paint.drawPixmap(rect, self._pixmap, rect)

            paint.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

            self.auto_clip_rect(paint)

            if self.cursor1 is not None and self.cursor1.isVisible():
                self.cursor1.paint(paint, plot=self, uuid=self.current_uuid)

            if self.region is not None:
                self.region.paint(paint, plot=self, uuid=self.current_uuid)

            paint.end()

        self._grabbing = False

    def close(self):
        super().close()

    def get_timestamp_index(self, timestamp, timestamps):
        key = id(timestamps), timestamp
        if key in self._timestamps_indexes:
            return self._timestamps_indexes[key]
        else:
            if timestamps.size:
                if timestamp > timestamps[-1]:
                    index = -1
                else:
                    index = np.searchsorted(timestamps, timestamp, side="left")
            else:
                index = None

            if len(self._timestamps_indexes) > 100000:
                self._timestamps_indexes.clear()
            self._timestamps_indexes[key] = index
            return index


class CursorInfo(QtWidgets.QLabel):
    def __init__(self, precision, name="t", unit="s", plot=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.name = name
        self.unit = unit
        self.plot = plot

        self.setTextFormat(QtCore.Qt.RichText)
        self.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter
        )

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)

        if precision == -1:
            self.setToolTip(f"Cursor information uses maximum precision")
        else:
            self.setToolTip(
                f"Cursor information precision is set to {self.precision} decimals"
            )

    def open_menu(self, point):
        menu = QtWidgets.QMenu(self)
        menu.addAction("Set precision")
        action = menu.exec_(self.mapToGlobal(point))

        if action is None:
            return

        if action.text() == "Set precision":
            precision, ok = QtWidgets.QInputDialog.getInt(
                self, "Set new precision (float decimals)", "Precision:", 6, -1, 15, 1
            )

            if ok:
                self.set_precision(precision)
                QtCore.QSettings().setValue("plot_cursor_precision", precision)

    def update_value(self):

        if not self.plot.region:

            if self.plot.cursor1 is not None:
                position = self.plot.cursor1.value()

                fmt = self.plot.x_axis.format
                if fmt == "phys":
                    if self.precision == -1:
                        cursor_info_text = f"{self.name} = {position}{self.unit}"
                    else:
                        template = f"{self.name} = {{:.{self.precision}f}}{self.unit}"
                        cursor_info_text = template.format(position)
                elif fmt == "time":
                    cursor_info_text = f"{self.name} = {timedelta(seconds=position)}"
                elif fmt == "date":
                    position_date = self.plot.x_axis.origin + timedelta(
                        seconds=position
                    )
                    cursor_info_text = f"{self.name} = {position_date}"
                self.setText(cursor_info_text)
            else:
                self.setText("")

        else:

            start, stop = self.plot.region.getRegion()

            fmt = self.plot.x_axis.format
            if fmt == "phys":
                if self.precision == -1:
                    start_info = f"{start}{self.unit}"
                    stop_info = f"{stop}{self.unit}"
                    delta_info = f"{stop - start}{self.unit}"
                else:
                    template = f"{{:.{self.precision}f}}{self.unit}"
                    start_info = template.format(start)
                    stop_info = template.format(stop)
                    delta_info = template.format(stop - start)

            elif fmt == "time":
                start_info = f"{timedelta(seconds=start)}"
                stop_info = f"{timedelta(seconds=stop)}"
                delta_info = f"{timedelta(seconds=(stop - start))}"
            elif fmt == "date":
                start_info = self.plot.x_axis.origin + timedelta(seconds=start)
                stop_info = self.plot.x_axis.origin + timedelta(seconds=stop)

                delta_info = f"{timedelta(seconds=(stop - start))}"

            self.setText(
                "<html><head/><body>"
                f"<p>{self.name}1 = {start_info}, {self.name}2 = {stop_info}</p>"
                f"<p>Δ{self.name} = {delta_info}</p> "
                "</body></html>"
            )

    def set_precision(self, precision):
        self.precision = precision
        if precision == -1:
            self.setToolTip(f"Cursor information uses maximum precision")
        else:
            self.setToolTip(
                f"Cursor information precision is set to {precision} decimals"
            )
        self.update_value()


from .fft_window import FFTWindow

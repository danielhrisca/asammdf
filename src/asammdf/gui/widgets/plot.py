import bisect
from datetime import timedelta
from functools import lru_cache, partial, reduce
from math import ceil
import os
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from time import perf_counter
from traceback import format_exc
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pyqtgraph as pg
from pyqtgraph import Qt
import pyqtgraph.functions as fn
from PySide6 import QtCore, QtGui, QtWidgets

PLOT_BUFFER_SIZE = 4000

from ... import tool as Tool
from ...blocks.conversion_utils import from_dict, to_dict
from ...blocks.cutils import get_idx_with_edges, positions
from ...blocks.utils import target_byte_order
from ..dialogs.messagebox import MessageBox
from ..utils import FONT_SIZE, value_as_str
from .viewbox import ViewBoxWithCursor


@lru_cache(maxsize=1024)
def polygon_and_ndarray(size):
    polygon = QtGui.QPolygonF()
    polygon.resize(size)

    nbytes = 2 * len(polygon) * 8
    ptr = polygon.data()
    if ptr is None:
        ptr = 0
    buffer = Qt.shiboken.VoidPtr(ptr, nbytes, True)
    ndarray = np.frombuffer(buffer, np.double).reshape((-1, 2))

    return polygon, ndarray


def monkey_patch_pyqtgraph():
    def _keys(self, styles):
        def getId(obj):
            try:
                return obj._id
            except AttributeError:
                obj._id = next(pg.graphicsItems.ScatterPlotItem.SymbolAtlas._idGenerator)
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

    mkColor_factory = fn.mkColor
    mkBrush_factory = fn.mkBrush
    mkPen_factory = fn.mkPen

    def mkColor(*args):
        if isinstance(args[0], QtGui.QColor):
            return QtGui.QColor(args[0])
        else:
            try:
                return cached_mkColor_factory(*args)
            except:
                # print(args, format_exc(), sep='\n')
                return mkColor_factory(*args)

    @lru_cache(maxsize=2048)
    def cached_mkColor_factory(*args):
        return mkColor_factory(*args)

    def mkBrush(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], QtGui.QBrush):
            return args[0]
        try:
            return cached_mkBrush_factory(*args, **kwargs)
        except:
            return mkBrush_factory(*args, **kwargs)

    @lru_cache(maxsize=2048)
    def cached_mkBrush_factory(*args, **kargs):
        return mkBrush_factory(*args, **kargs)

    def mkPen(*args, **kwargs):
        try:
            return cached_mkPen_factory(*args, **kwargs)
        except:
            return mkPen_factory(*args, **kwargs)

    @lru_cache(maxsize=2048)
    def cached_mkPen_factory(*args, **kargs):
        return mkPen_factory(*args, **kargs)

    # speed-up monkey patches
    pg.graphicsItems.ScatterPlotItem.SymbolAtlas._keys = _keys
    pg.graphicsItems.ScatterPlotItem._USE_QRECT = False

    fn.mkBrush = mkBrush
    fn.mkColor = mkColor
    fn.mkPen = mkPen


import asammdf.mdf as mdf_module

from ...blocks.utils import extract_mime_names
from ...signal import Signal
from ..dialogs.define_channel import DefineChannel
from ..utils import COLORS, COLORS_COUNT, copy_ranges
from .channel_stats import ChannelStats
from .cursor import Bookmark, Cursor, Region
from .dict_to_tree import ComputedChannelInfoWindow
from .formated_axis import FormatedAxis
from .tree import ChannelsTreeItem, ChannelsTreeWidget

bin_ = bin

HERE = Path(__file__).resolve().parent

NOT_FOUND = 0xFFFFFFFF
HONEYWELL_SECONDS_PER_CM = 0.1

float64 = np.float64


def simple_min(a, b):
    if b != b:  # noqa: PLR0124
        # b is NaN
        return a
    if a <= b:
        return a
    return b


def simple_max(a, b):
    if b != b:  # noqa: PLR0124
        # b is NaN
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


class PlotSignal(Signal):
    def __init__(self, signal, index=0, trim_info=None, duplication=1, allow_trim=True, allow_nans=False):
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
            invalidation_bits=signal.invalidation_bits,
            encoding=signal.encoding,
            flags=signal.flags,
        )

        self._pos = np.empty(2 * PLOT_BUFFER_SIZE, dtype="i4")
        self._plot_samples = np.empty(2 * PLOT_BUFFER_SIZE, dtype="i1")
        self._plot_timestamps = np.empty(2 * PLOT_BUFFER_SIZE, dtype="f8")

        self.path = None

        self._dtype = "i1"

        self.duplication = duplication
        self.uuid = getattr(signal, "uuid", os.urandom(6).hex())
        self.origin_uuid = getattr(signal, "origin_uuid", os.urandom(6).hex())

        self.group_index = getattr(signal, "group_index", NOT_FOUND)
        self.channel_index = getattr(signal, "channel_index", NOT_FOUND)
        self.precision = getattr(signal, "precision", 3)

        self._mode = "raw"
        self._enable = getattr(signal, "enable", 3)

        self.format = getattr(signal, "format", "phys")

        self.individual_axis = False
        self.computation = signal.computation
        self.original_name = getattr(signal, "original_name", None)

        self.y_link = False
        self.y_range = (0, -1)
        self.home = (0, -1)

        self.trim_info = None

        # take out NaN values
        samples = self.samples
        if samples.dtype.kind not in "SUV":
            if not allow_nans:
                nans = np.isnan(samples)
                if np.any(nans):
                    self.samples = self.samples[~nans]
                    self.timestamps = self.timestamps[~nans]

        if self.samples.dtype.byteorder not in target_byte_order:
            self.samples = self.samples.byteswap().view(self.samples.dtype.newbyteorder())

        if self.timestamps.dtype.byteorder not in target_byte_order:
            self.timestamps = self.timestamps.byteswap().view(self.timestamps.dtype.n())

        if self.timestamps.dtype != float64:
            self.timestamps = self.timestamps.astype(float64)

        self.text_conversion = None

        if self.conversion:
            samples = self.conversion.convert(self.samples, as_bytes=True)
            if samples.dtype.kind not in "SUV":
                if not allow_nans:
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
                    self.raw_samples = self.samples
                    self.phys_samples = samples
            else:
                self.text_conversion = self.conversion
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
            color = signal.color or COLORS[index % COLORS_COUNT]
        else:
            color = COLORS[index % COLORS_COUNT]
        self.color = fn.mkColor(color)
        self.color_name = self.color.name()
        self.pen = fn.mkPen(color=color, style=QtCore.Qt.PenStyle.SolidLine)

        self._min = None
        self._max = None
        self._rms = None
        self._avg = None
        self._std = None
        self._min_raw = None
        self._max_raw = None
        self._avg_raw = None
        self._rms_raw = None
        self._std_raw = None

        self._stats_available = False
        self._compute_basic_stats()

        self.mode = getattr(signal, "mode", "phys")
        if allow_trim:
            self.trim(*(trim_info or (None, None, 1900)))

    @property
    def avg(self):
        if not self._stats_available:
            self._compute_stats()
        return self._avg if self._mode == "phys" else self._avg_raw

    @avg.setter
    def avg(self, avg):
        self._avg = avg

    def _compute_basic_stats(self):
        self._stats_available = False

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
                else:
                    self._min_raw = "n.a."
                    self._max_raw = "n.a."

            if self.phys_samples is self.raw_samples:
                if self.phys_samples.dtype.kind in "SUV":
                    self.is_string = True
                else:
                    self.is_string = False

                self._min = self._min_raw
                self._max = self._max_raw

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

    def _compute_stats(self):
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
        self._stats_available = True

    def cut(self, start=None, stop=None, include_ends=True, interpolation_mode=0):
        cut_sig = super().cut(start, stop, include_ends, interpolation_mode)

        cut_sig.group_index = self.group_index
        cut_sig.channel_index = self.channel_index
        cut_sig.color = self.color
        cut_sig.computation = self.computation
        cut_sig.precision = self.precision
        cut_sig.mdf_uuid = self.origin_uuid

        return PlotSignal(cut_sig, duplication=self.duplication)

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

    def get_stats(self, cursor=None, region=None, view_region=None, precision=6):
        stats = {}
        sig = self
        x = sig.timestamps
        size = len(x)

        if precision == -1:
            precision = 16

        format = sig.format

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

                    value, kind, _ = self.value_at_timestamp(position)

                    stats["cursor_value"] = value_as_str(value, format, self.plot_samples.dtype, precision)

                else:
                    stats["cursor_t"] = ""
                    stats["cursor_value"] = ""

                if region:
                    start, stop = region
                    stats["selected_start"] = value_as_str(start, format, np.dtype("f8"), precision)
                    stats["selected_stop"] = value_as_str(stop, format, np.dtype("f8"), precision)
                    stats["selected_delta_t"] = value_as_str(stop - start, format, np.dtype("f8"), precision)

                    value, kind, _ = self.value_at_timestamp(start)

                    stats["selected_left"] = value_as_str(value, format, self.plot_samples.dtype, precision)

                    value, kind, _ = self.value_at_timestamp(stop)

                    stats["selected_right"] = value_as_str(value, format, self.plot_samples.dtype, precision)

                else:
                    stats["selected_start"] = ""
                    stats["selected_stop"] = ""
                    stats["selected_delta_t"] = ""
                    stats["selected_left"] = ""
                    stats["selected_right"] = ""

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
                if size == 1:
                    stats["overall_gradient"] = 0
                    stats["overall_integral"] = 0
                else:
                    stats["overall_gradient"] = value_as_str(
                        (float(sig.samples[-1]) - float(sig.samples[0])) / (sig.timestamps[-1] - sig.timestamps[0]),
                        format,
                        None,
                        precision,
                    )
                    stats["overall_integral"] = value_as_str(
                        np.trapz(sig.samples, sig.timestamps), format, None, precision
                    )

                stats["overall_min"] = value_as_str(self.min, format, self.plot_samples.dtype, precision)
                stats["overall_max"] = value_as_str(self.max, format, self.plot_samples.dtype, precision)
                stats["overall_average"] = value_as_str(sig.avg, format, None, precision)
                stats["overall_rms"] = value_as_str(sig.rms, format, None, precision)
                stats["overall_std"] = value_as_str(sig.std, format, None, precision)
                stats["overall_start"] = value_as_str(sig.timestamps[0], format, np.dtype("f8"), precision)
                stats["overall_stop"] = value_as_str(sig.timestamps[-1], format, np.dtype("f8"), precision)
                stats["overall_delta"] = value_as_str(
                    sig.samples[-1] - sig.samples[0],
                    format,
                    self.plot_samples.dtype,
                    precision,
                )
                stats["overall_delta_t"] = value_as_str(x[-1] - x[0], format, np.dtype("f8"), precision)
                stats["unit"] = sig.unit
                stats["color"] = sig.color
                stats["name"] = sig.name

                if cursor is not None:
                    position = cursor
                    stats["cursor_t"] = value_as_str(position, format, np.dtype("f8"), precision)

                    value, kind, _ = self.value_at_timestamp(position)

                    stats["cursor_value"] = value_as_str(value, format, self.plot_samples.dtype, precision)

                else:
                    stats["cursor_t"] = ""
                    stats["cursor_value"] = ""

                if region:
                    start, stop = region

                    new_stats = {}
                    new_stats["selected_start"] = value_as_str(start, format, np.dtype("f8"), precision)
                    new_stats["selected_stop"] = value_as_str(stop, format, np.dtype("f8"), precision)
                    new_stats["selected_delta_t"] = value_as_str(stop - start, format, np.dtype("f8"), precision)

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
                        new_stats["selected_left"] = value_as_str(
                            samples[0], format, self.plot_samples.dtype, precision
                        )

                        new_stats["selected_right"] = value_as_str(
                            samples[-1], format, self.plot_samples.dtype, precision
                        )

                        new_stats["selected_min"] = value_as_str(np.nanmin(samples), format, samples.dtype, precision)
                        new_stats["selected_max"] = value_as_str(np.nanmax(samples), format, samples.dtype, precision)
                        new_stats["selected_average"] = value_as_str(np.mean(samples), format, None, precision)
                        new_stats["selected_std"] = value_as_str(np.std(samples), format, None, precision)
                        new_stats["selected_rms"] = value_as_str(
                            np.sqrt(np.mean(np.square(samples))),
                            format,
                            None,
                            precision,
                        )
                        if samples.dtype.kind in "ui":
                            new_stats["selected_delta"] = value_as_str(
                                int(samples[-1]) - int(samples[0]),
                                format,
                                samples.dtype,
                                precision,
                            )
                        else:
                            new_stats["selected_delta"] = value_as_str(
                                samples[-1] - samples[0],
                                format,
                                samples.dtype,
                                precision,
                            )

                        if size == 1:
                            new_stats["selected_gradient"] = 0
                            new_stats["selected_integral"] = 0
                        else:
                            new_stats["selected_gradient"] = value_as_str(
                                (float(samples[-1]) - float(samples[0])) / (timestamps[-1] - timestamps[0]),
                                format,
                                None,
                                precision,
                            )
                            new_stats["selected_integral"] = value_as_str(
                                np.trapz(samples, timestamps), format, None, precision
                            )

                    else:
                        new_stats["selected_min"] = "n.a."
                        new_stats["selected_max"] = "n.a."
                        new_stats["selected_average"] = "n.a."
                        new_stats["selected_left"] = "n.a."
                        new_stats["selected_right"] = "n.a."
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
                    stats["selected_left"] = ""
                    stats["selected_right"] = ""
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
                new_stats["visible_start"] = value_as_str(start, format, np.dtype("f8"), precision)
                new_stats["visible_stop"] = value_as_str(stop, format, np.dtype("f8"), precision)
                new_stats["visible_delta_t"] = value_as_str(stop - start, format, np.dtype("f8"), precision)

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

                    new_stats["visible_min"] = value_as_str(np.nanmin(samples), format, samples.dtype, precision)
                    new_stats["visible_max"] = value_as_str(np.nanmax(samples), format, samples.dtype, precision)
                    new_stats["visible_average"] = value_as_str(np.mean(samples), format, None, precision)
                    new_stats["visible_std"] = value_as_str(np.std(samples), format, None, precision)
                    new_stats["visible_rms"] = value_as_str(
                        np.sqrt(np.mean(np.square(samples))), format, None, precision
                    )
                    if kind in "ui":
                        new_stats["visible_delta"] = value_as_str(
                            int(cut.samples[-1]) - int(cut.samples[0]),
                            format,
                            samples.dtype,
                            precision,
                        )
                    else:
                        new_stats["visible_delta"] = value_as_str(
                            cut.samples[-1] - cut.samples[0],
                            format,
                            samples.dtype,
                            precision,
                        )

                    if size == 1:
                        new_stats["visible_gradient"] = 0
                        new_stats["visible_integral"] = 0
                    else:
                        new_stats["visible_gradient"] = value_as_str(
                            (float(samples[-1]) - float(samples[0])) / (timestamps[-1] - timestamps[0]),
                            format,
                            None,
                            precision,
                        )
                        new_stats["visible_integral"] = value_as_str(
                            np.trapz(samples, timestamps), format, None, precision
                        )

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
                stats["cursor_t"] = value_as_str(position, format, np.dtype("f8"), precision)

                stats["cursor_value"] = "n.a."

            else:
                stats["cursor_t"] = ""
                stats["cursor_value"] = ""

            if region is not None:
                start, stop = region

                stats["selected_start"] = value_as_str(start, format, np.dtype("f8"), precision)
                stats["selected_stop"] = value_as_str(stop, format, np.dtype("f8"), precision)
                stats["selected_delta_t"] = value_as_str(stop - start, format, np.dtype("f8"), precision)

                stats["selected_min"] = "n.a."
                stats["selected_max"] = "n.a."
                stats["selected_left"] = "n.a."
                stats["selected_right"] = "n.a."
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
                stats["selected_left"] = ""
                stats["selected_right"] = ""
                stats["selected_average"] = "n.a."
                stats["selected_rms"] = "n.a."
                stats["selected_std"] = "n.a."
                stats["selected_delta"] = ""
                stats["selected_gradient"] = ""
                stats["selected_integral"] = ""

            start, stop = view_region

            stats["visible_start"] = value_as_str(start, format, np.dtype("f8"), precision)
            stats["visible_stop"] = value_as_str(stop, format, np.dtype("f8"), precision)
            stats["visible_delta_t"] = value_as_str(stop - start, format, np.dtype("f8"), precision)

            stats["visible_min"] = "n.a."
            stats["visible_max"] = "n.a."
            stats["visible_average"] = "n.a."
            stats["visible_rms"] = "n.a."
            stats["visible_std"] = "n.a."
            stats["visible_delta"] = "n.a."
            stats["visible_gradient"] = "n.a."
            stats["visible_integral"] = "n.a."

        stats["region"] = region is not None
        stats["color"] = self.color_name

        return stats

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

    @property
    def rms(self):
        if not self._stats_available:
            self._compute_stats()
        return self._rms if self.mode == "phys" else self._rms_raw

    @rms.setter
    def rms(self, rms):
        self._rms = rms

    def set_color(self, color):
        self.color = color
        self.pen = fn.mkPen(color=color, style=QtCore.Qt.PenStyle.SolidLine)

    def set_home(self, y_range=None):
        self.home = y_range or self.y_range

    @property
    def std(self):
        if not self._stats_available:
            self._compute_stats()
        return self._std if self.mode == "phys" else self._std_raw

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

                if stop_ == start_:
                    visible_duplication = 0
                else:
                    visible = ceil((stop_t - start_t) / (stop - start) * width)

                    if visible:
                        visible_duplication = (stop_ - start_) // abs(visible)
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
                        self._plot_samples = np.empty(2 * PLOT_BUFFER_SIZE, dtype=samples.dtype)

                        self._dtype = samples.dtype

                    if samples.flags.c_contiguous and timestamps.flags.c_contiguous:
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
                        visible_duplication = abs(stop_ - start_) // visible
                    else:
                        visible_duplication = 0
                except:
                    visible_duplication = 0

                while visible_duplication > self.duplication:
                    rows = (stop_ - start_) // visible_duplication
                    stop_2 = start_ + rows * visible_duplication

                    samples = signal_samples[start_:stop_2].reshape(rows, visible_duplication)

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

                        pos2 = [pos_min, pos_max] if pos_min < pos_max else [pos_max, pos_min]

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
            self.path = None
            try:
                return self.trim_c(start, stop, width, force)
            except:
                print(format_exc())
                return self.trim_python(start, stop, width, force)

    def value_at_index(self, index):
        if self.mode == "raw":
            kind = self.raw_samples.dtype.kind
        else:
            kind = self.phys_samples.dtype.kind

        size = len(self)

        if index is None or size == 0:
            value = "n.a."
        else:
            if index >= size:
                index = size - 1

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
                    value = value.decode("utf-8", errors="replace").strip(" \r\n\t\v\0")
                except:
                    value = value.decode("latin-1", errors="replace").strip(" \r\n\t\v\0")

                value = value or "<empty string>"
            elif kind == "f":
                value = float(value)
            else:
                value = int(value)

        return value, kind, self.format

    @property
    def y_range(self):
        return self._y_range

    @y_range.setter
    def y_range(self, value):
        self.path = None
        self._y_range = value


from .signal_scale import ScaleDialog


class Plot(QtWidgets.QWidget):
    add_channels_request = QtCore.Signal(list)
    close_request = QtCore.Signal()
    clicked = QtCore.Signal()
    cursor_moved_signal = QtCore.Signal(object, float)
    cursor_removed_signal = QtCore.Signal(object)
    edit_channel_request = QtCore.Signal(object, object)
    region_moved_signal = QtCore.Signal(object, list)
    region_removed_signal = QtCore.Signal(object)
    show_properties = QtCore.Signal(list)
    splitter_moved = QtCore.Signal(object, int)
    pattern_group_added = QtCore.Signal(object, object)
    verify_bookmarks = QtCore.Signal(list, object)
    x_range_changed_signal = QtCore.Signal(object, object)

    item_double_click_handling = "enable/disable"
    dynamic_columns_width = True
    mouse_mode = "pan"

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
        show_cursor_circle=True,
        show_cursor_horizontal_line=True,
        cursor_line_width=1,
        cursor_color="#ffffff",
        region_values_display_mode="delta",
        owner=None,
        enable_zoom_history=True,
        *args,
        **kwargs,
    ):
        events = kwargs.pop("events", None)

        self.owner = owner
        self.enable_zoom_history = enable_zoom_history

        super().__init__(*args, **kwargs)
        self.closed = False
        self.line_interconnect = line_interconnect
        self.setContentsMargins(0, 0, 0, 0)
        self.pattern = {}
        self.mdf = mdf

        self._settings = QtCore.QSettings()

        self.x_name = "t" if x_axis == "time" else "f"
        self.x_unit = "s" if x_axis == "time" else "Hz"

        self.info_uuid = None

        self._can_switch_mode = True
        self._inhibit_x_range_changed_signal = False
        self._inhibit_timestamp_signals = False
        self._inhibit_timestamp_signals_timer = QtCore.QTimer()
        self._inhibit_timestamp_signals_timer.setSingleShot(True)
        self._inhibit_timestamp_signals_timer.timeout.connect(self._inhibit_timestamp_handler)
        self.can_edit_ranges = True

        self.region_values_display_mode = region_values_display_mode

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

        self.focused_mode = False
        self.show_bookmarks = True

        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(widget)
        self.splitter.setOpaqueResize(False)

        self.plot = PlotGraphics(
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
        self.plot.zoom_changed.connect(self.zoom_changed)

        if self.plot.cursor1 is not None:
            self.plot.cursor1.show_circle = show_cursor_circle
            self.plot.cursor1.show_horizontal_line = show_cursor_horizontal_line
            self.plot.cursor1.line_width = cursor_line_width
            self.plot.cursor1.color = cursor_color

            self.lock = self.plot.lock

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

        btn = QtWidgets.QPushButton("Cmd")

        menu = QtWidgets.QMenu()

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/home.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        menu.addAction(
            icon,
            "Home",
            lambda: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress,
                    QtCore.Qt.Key.Key_W,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )
            ),
        )

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/axis.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        menu.addAction(
            icon,
            "Honeywell",
            lambda: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress,
                    QtCore.Qt.Key.Key_H,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )
            ),
        )

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/fit.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        menu.addAction(
            icon,
            "Fit",
            lambda: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress,
                    QtCore.Qt.Key.Key_F,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )
            ),
        )

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/stack.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        menu.addAction(
            icon,
            "Stack",
            lambda: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress,
                    QtCore.Qt.Key.Key_S,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )
            ),
        )

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/increase-font.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        menu.addAction(icon, "Increase font", self.increase_font)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/decrease-font.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        menu.addAction(icon, "Decrease font", self.decrease_font)

        btn.setMenu(menu)
        hbox.addWidget(btn)
        btn.menu()

        btn = QtWidgets.QPushButton("")
        btn.clicked.connect(
            lambda x: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress,
                    QtCore.Qt.Key.Key_I,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )
            )
        )
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/zoom-in.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        btn.setIcon(icon)
        btn.setToolTip("Zoom in")
        hbox.addWidget(btn)

        btn = QtWidgets.QPushButton("")
        btn.clicked.connect(
            lambda x: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress,
                    QtCore.Qt.Key.Key_O,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )
            )
        )
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/zoom-out.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        btn.setIcon(icon)
        btn.setToolTip("Zoom out")
        hbox.addWidget(btn)

        if self.enable_zoom_history:
            self.undo_btn = btn = QtWidgets.QPushButton("")
            btn.clicked.connect(self.undo_zoom)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/undo.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            btn.setIcon(icon)
            btn.setToolTip("Undo zoom")
            hbox.addWidget(btn)
            btn.setEnabled(False)

            self.redo_btn = btn = QtWidgets.QPushButton("")
            btn.clicked.connect(self.redo_zoom)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/redo.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            btn.setIcon(icon)
            btn.setToolTip("Redo zoom")
            hbox.addWidget(btn)
            btn.setEnabled(False)

        self.lock_btn = btn = QtWidgets.QPushButton("")
        btn.setObjectName("lock_btn")
        btn.clicked.connect(self.set_locked)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/unlocked.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        btn.setIcon(icon)
        btn.setToolTip("The Y axis is unlocked. Press to lock")
        hbox.addWidget(btn)

        self.locked = False

        self.hide_axes_btn = btn = QtWidgets.QPushButton("")
        btn.setObjectName("hide_axes_btn")
        self.hide_axes_btn.clicked.connect(self.hide_axes)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/axis_on.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        btn.setIcon(icon)
        btn.setToolTip("Hide axis")
        hbox.addWidget(self.hide_axes_btn)

        self.selected_channel_value_btn = btn = QtWidgets.QPushButton("")
        btn.setObjectName("selected_channel_value_btn")
        self.selected_channel_value_btn.clicked.connect(self.hide_selected_channel_value)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/number_on.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        btn.setIcon(icon)
        btn.setToolTip("Hide axis")
        hbox.addWidget(self.selected_channel_value_btn)

        self.focused_mode_btn = btn = QtWidgets.QPushButton("")
        btn.setObjectName("focused_mode_btn")
        self.focused_mode_btn.clicked.connect(self.toggle_focused_mode)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/focus_on.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        btn.setIcon(icon)
        btn.setToolTip("Toggle focused mode")
        hbox.addWidget(self.focused_mode_btn)

        self.delta_btn = btn = QtWidgets.QPushButton("")
        btn.setObjectName("delta_btn")
        self.delta_btn.clicked.connect(self.toggle_region_values_display_mode)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/delta_on.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        btn.setIcon(icon)
        btn.setToolTip("Toggle region values display mode")
        hbox.addWidget(self.delta_btn)

        self.bookmark_btn = btn = QtWidgets.QPushButton("")
        btn.setObjectName("bookmark_btn")
        btn.clicked.connect(self.toggle_bookmarks)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/bookmark_on.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        btn.setIcon(icon)
        btn.setToolTip("Toggle bookmarks")
        hbox.addWidget(btn)

        hbox.addStretch()

        self.selected_channel_value = QtWidgets.QLabel("")
        self.selected_channel_value.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.selected_channel_value.setAutoFillBackground(True)
        font = self.selected_channel_value.font()
        font.setBold(True)
        font.setPointSize(24)
        self.selected_channel_value.setFont(font)

        vbox.addWidget(self.selected_channel_value)

        vbox.addWidget(self.channel_selection)
        vbox.addWidget(self.cursor_info)

        self.range_proxy = pg.SignalProxy(self.plot.range_modified, rateLimit=16, slot=self.range_modified)
        # self.plot.range_modified.connect(self.range_modified)
        self.plot.range_removed.connect(self.range_removed)
        self.plot.range_modified_finished.connect(self.range_modified_finished)
        self.plot.cursor_removed.connect(self.cursor_removed)
        self.plot.current_uuid_changed.connect(self.current_uuid_changed)

        self.cursor_proxy = pg.SignalProxy(self.plot.cursor_moved, rateLimit=16, slot=self.cursor_moved)
        # self.plot.cursor_moved.connect(self.cursor_moved)
        self.plot.cursor_move_finished.connect(self.cursor_move_finished)
        self.plot.xrange_changed.connect(self.xrange_changed)
        self.plot.computation_channel_inserted.connect(self.computation_channel_inserted)
        self.plot.curve_clicked.connect(self.curve_clicked)
        self._visible_entries = set()
        self.visible_entries_modified = False
        self.lock = Lock()
        self._visible_items = {}
        self._item_cache = {}

        self._prev_region = None

        self.splitter.addWidget(self.plot)

        self.info = ChannelStats(
            self.x_unit,
            precision=self._settings.value("stats_float_precision", 6, type=int),
        )
        self.info.hide()
        self.info.precision_modified.connect(self.info_precision_modified)
        self.splitter.addWidget(self.info)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        self.plot.add_channels_request.connect(self.add_channels_request)
        self.plot.edit_channel_request.connect(self.edit_channel_request)
        self.setAcceptDrops(True)

        main_layout.addWidget(self.splitter)

        self.show()
        size = sum(self.splitter.sizes())
        self.splitter.setSizes([600, max(size - 600, 1)])

        nameColumnWidth = 5 * self.font().pointSize()
        if signals:
            nameColumnWidth = max([len(signal.name) + 10 for signal in signals])

        unitColumnWidth = 3 * self.font().pointSize()
        if signals:
            unitColumnWidth = max([len(signal.unit) + 10 for signal in signals])

        self.channel_selection.setColumnWidth(self.channel_selection.NameColumn, nameColumnWidth)
        self.channel_selection.setColumnWidth(self.channel_selection.ValueColumn, 83)
        self.channel_selection.setColumnWidth(self.channel_selection.UnitColumn, unitColumnWidth)
        self.channel_selection.setColumnWidth(self.channel_selection.CommonAxisColumn, 35)
        self.channel_selection.setColumnWidth(self.channel_selection.IndividualAxisColumn, 35)
        self.hide()

        if signals:
            self.add_new_channels(signals)

        self.channel_selection.color_changed.connect(self.plot.set_color)
        self.channel_selection.unit_changed.connect(self.plot.set_unit)
        self.channel_selection.name_changed.connect(self.plot.set_name)
        self.channel_selection.conversion_changed.connect(self.set_conversion)

        self.channel_selection.itemsDeleted.connect(self.channel_selection_reduced)
        self.channel_selection.group_activation_changed.connect(self.plot.update)
        self.channel_selection.group_activation_changed.connect(self.cursor_moved)
        self.channel_selection.currentItemChanged.connect(self.channel_selection_row_changed)
        self.channel_selection.itemSelectionChanged.connect(self.channel_selection_changed)
        self.channel_selection.add_channels_request.connect(self.add_channels_request)
        self.channel_selection.set_time_offset.connect(self.plot.set_time_offset)
        self.channel_selection.show_properties.connect(self._show_properties)
        self.channel_selection.insert_computation.connect(self.plot.insert_computation)
        self.channel_selection.edit_computation.connect(self.plot.edit_computation)
        self.channel_selection.itemClicked.connect(self.flash_curve)

        self.channel_selection.model().dataChanged.connect(self.channel_selection_item_changed)

        self.channel_selection.visible_items_changed.connect(self._update_visibile_entries)

        self.channel_selection.pattern_group_added.connect(self.pattern_group_added_req)
        self.channel_selection.double_click.connect(self.channel_selection_item_double_clicked)

        self.channel_selection.compute_fft_request.connect(self.compute_fft)
        self.channel_selection.itemExpanded.connect(self.update_current_values)
        self.channel_selection.verticalScrollBar().valueChanged.connect(self.update_current_values)

        self.keyboard_events = {
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_M,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_C,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.Key.Key_C,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.Key.Key_B,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.Key.Key_H,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.Key.Key_P,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.Key.Key_T,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.Key.Key_G,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_2,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_BracketLeft,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_BracketRight,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_Backspace,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_Backspace,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_W,
            ).toCombined(),
        } | self.plot.keyboard_events

        self.splitter.splitterMoved.connect(self.set_splitter)

        self.hide_selected_channel_value(
            hide=self._settings.value("plot_hide_selected_channel_value", False, type=bool)
        )
        self.toggle_focused_mode(focused=self._settings.value("plot_focused_mode", False, type=bool))
        self.toggle_region_values_display_mode(mode=self._settings.value("plot_region_values_display_mode", "value"))

        self.toggle_bookmarks(hide=not self._settings.value("plot_bookmarks", False, type=bool))
        self.hide_axes(hide=self._settings.value("plot_hide_axes", False, type=bool))
        self.set_locked(locked=self._settings.value("plot_locked", False, type=bool))

        self.zoom_history = []
        self.zoom_history_index = -1
        self.update_zoom = False

        self.show()

    def add_new_channels(self, channels, mime_data=None, destination=None, update=True):
        initial = self.channel_selection.topLevelItemCount() == 0

        def add_new_items(tree, root, items, items_pool):
            children = []
            groups = []

            for info in items:
                pattern = info.get("pattern", None)
                uuid = info["uuid"]
                name = info["name"]
                origin_uuid = info.get("origin_uuid", "000000000000")

                ranges = copy_ranges(info["ranges"])
                for range_info in ranges:
                    range_info["font_color"] = fn.mkColor(range_info["font_color"])
                    range_info["background_color"] = fn.mkColor(range_info["background_color"])

                if info.get("type", "channel") == "group":
                    item = ChannelsTreeItem(
                        ChannelsTreeItem.Group,
                        name=name,
                        pattern=pattern,
                        uuid=uuid,
                        origin_uuid=origin_uuid,
                    )
                    children.append(item)
                    item.set_ranges(ranges)

                    groups.extend(add_new_items(tree, item, info["channels"], items_pool))
                    groups.append((item, info))

                else:
                    if uuid in items_pool:
                        item = items_pool[uuid]
                        children.append(item)

                        del items_pool[uuid]

            if root is None:
                root = self.channel_selection.invisibleRootItem()
                root.addChildren(children)
            else:
                if root.type() == ChannelsTreeItem.Group:
                    root.addChildren(children)
                else:
                    parent = root.parent() or self.channel_selection.invisibleRootItem()
                    index = parent.indexOfChild(root)
                    parent.insertChildren(index, children)

            return groups

        self.plot._can_paint = False

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
                invalid.append(f"{channel.name} @ index {invalid_indexes[:10] - 1} with first time stamp error: {ts}")
                if len(np.argwhere(diff < 0).ravel()):
                    can_trim = False

        if invalid:
            errors = "\n".join(invalid)
            try:
                mdi_title = self.parent().windowTitle()
                title = f"plot <{mdi_title}>"
            except:
                title = "plot window"

            MessageBox.warning(
                self,
                f"Channels with corrupted time stamps added to {title}",
                f"The following channels do not have monotonous increasing time stamps:\n{errors}",
            )
            self.plot._can_trim = can_trim or True  # allow it for now

        valid = {}
        invalid = []
        for uuid, channel in channels.items():
            if len(channel):
                samples = channel.samples
                if samples.dtype.kind not in "SUV" and np.all(np.isnan(samples)):
                    invalid.append(channel.name)
                elif channel.conversion:
                    samples = channel.physical(copy=False).samples
                    if samples.dtype.kind not in "SUV" and np.all(np.isnan(samples)):
                        invalid.append(channel.name)
                    else:
                        valid[uuid] = channel
                else:
                    valid[uuid] = channel
            else:
                valid[uuid] = channel

        if invalid:
            MessageBox.warning(
                self,
                "All NaN channels will not be plotted:",
                f"The following channels have all NaN samples and will not be plotted:\n{', '.join(invalid)}",
            )

        channels = valid

        channels = self.plot.add_new_channels(channels, descriptions=descriptions)

        enforce_y_axis = False
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while item := iterator.value():
            if item.type() == item.Channel:
                if item.checkState(item.CommonAxisColumn) == QtCore.Qt.CheckState.Unchecked:
                    enforce_y_axis = False
                    break
                else:
                    enforce_y_axis = True

            iterator += 1

        children = []

        if self._settings.value("current_theme") == "Dark":
            background_color = QtGui.QColor(0, 0, 0)
        else:
            background_color = QtGui.QColor(255, 255, 255)

        new_items = {}

        items_map = {}
        for sig_uuid, sig in channels.items():
            description = descriptions.get(sig_uuid, {})

            if description:
                sig.format = description.get("format", "phys")
            sig.mode = description.get("mode", "phys")

            if "comment" in description:
                sig.comment = description["comment"] or ""
                sig.flags |= Signal.Flags.user_defined_comment

            item = ChannelsTreeItem(
                ChannelsTreeItem.Channel,
                signal=sig,
                check=QtCore.Qt.CheckState.Checked if sig.enable else QtCore.Qt.CheckState.Unchecked,
                background_color=background_color,
            )

            if len(sig):
                value, kind, fmt = sig.value_at_timestamp(sig.timestamps[0])
                item.kind = kind
                item._value = "n.a."
                item.set_value(value, force=True, update=True)

            if mime_data is None:
                children.append(item)
            else:
                new_items[sig_uuid] = item
            items_map[sig_uuid] = item

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

            self.info_uuid = sig_uuid

        if mime_data:
            destination = destination or self.channel_selection.drop_target

            groups = add_new_items(
                self.channel_selection,
                destination,
                mime_data,
                new_items,
            )

            if groups:
                self.channel_selection.blockSignals(True)

                for item, info in groups:
                    item.setExpanded(info.get("expanded", False))
                    if item.pattern:
                        item.setCheckState(
                            item.NameColumn,
                            QtCore.Qt.CheckState.Checked if info["enabled"] else QtCore.Qt.CheckState.Unchecked,
                        )
                    else:
                        if not item.childCount():
                            item.setCheckState(
                                item.NameColumn,
                                QtCore.Qt.CheckState.Checked if info["enabled"] else QtCore.Qt.CheckState.Unchecked,
                            )
                    if info.get("disabled", False):
                        item.set_disabled(info["disabled"])

                self.channel_selection.blockSignals(False)
                self.channel_selection.refresh()

            # still have simple signals to add
            if new_items:
                self.channel_selection.addTopLevelItems(list(new_items.values()))

        elif children:
            destination = destination or self.channel_selection.drop_target

            if destination is None:
                self.channel_selection.addTopLevelItems(children)
            else:
                if destination.type() == ChannelsTreeItem.Group:
                    destination.addChildren(children)
                else:
                    parent = destination.parent() or self.channel_selection.invisibleRootItem()
                    index = parent.indexOfChild(destination)
                    parent.insertChildren(index, children)

        for sig_uuid, sig in channels.items():
            description = descriptions.get(sig_uuid, {})
            item = items_map[sig_uuid]
            if description:
                individual_axis = description.get("individual_axis", False)
                if individual_axis:
                    item.setCheckState(item.IndividualAxisColumn, QtCore.Qt.CheckState.Checked)

                    _, idx = self.plot.signal_by_uuid(sig_uuid)
                    axis = self.plot.get_axis(idx)
                    if isinstance(axis, FormatedAxis):
                        axis.setWidth(description["individual_axis_width"])

                if description.get("common_axis", False):
                    item.setCheckState(item.CommonAxisColumn, QtCore.Qt.CheckState.Checked)

                item.precision = description.get("precision", 3)

                if description.get("conversion", None):
                    conversion = from_dict(description["conversion"])
                    item.signal.flags |= Signal.Flags.user_defined_conversion
                    item.set_conversion(conversion)

                if description.get("user_defined_name", None):
                    item.name = description["user_defined_name"]

            if enforce_y_axis:
                item.setCheckState(item.CommonAxisColumn, QtCore.Qt.CheckState.Checked)

        if update:
            self.channel_selection.update_channel_groups_count()
            self.channel_selection.refresh()

        self.adjust_splitter(initial=initial)

        self.current_uuid_changed(self.plot.current_uuid)
        self.plot._can_paint = True
        self.plot.update()

    def adjust_splitter(self, initial=False):
        size = sum(self.splitter.sizes())

        if Plot.dynamic_columns_width or initial:
            self.channel_selection.resizeColumnToContents(self.channel_selection.NameColumn)
            self.channel_selection.resizeColumnToContents(self.channel_selection.UnitColumn)

        width = sum(self.channel_selection.columnWidth(col) for col in range(self.channel_selection.columnCount()))

        if width > self.splitter.sizes()[0]:
            if size - width >= 300:
                self.splitter.setSizes([width, size - width, 0])
            else:
                if size >= 350:
                    self.splitter.setSizes([size - 300, 300, 0])
                elif size >= 100:
                    self.splitter.setSizes([50, size - 50, 0])
        else:
            self.splitter.setSizes([width, size - width, 0])

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
            "name": widget.name,
            "enabled": item.checkState(item.NameColumn) == QtCore.Qt.CheckState.Checked,
            "pattern": pattern,
            "ranges": ranges,
            "origin_uuid": item.origin_uuid,
            "expanded": item.isExpanded(),
            "disabled": item.isDisabled(),
        }

        return channel_group

    def channel_item_to_config(self, item):
        widget = item

        channel = {"type": "channel"}

        sig, idx = self.plot.signal_by_uuid(widget.uuid)

        if sig.flags & Signal.Flags.user_defined_name:
            channel["user_defined_name"] = sig.name
            channel["name"] = sig.original_name
        else:
            channel["name"] = sig.name

        channel["unit"] = sig.unit
        channel["flags"] = int(sig.flags)
        channel["enabled"] = item.checkState(item.NameColumn) == QtCore.Qt.CheckState.Checked

        if item.checkState(item.IndividualAxisColumn) == QtCore.Qt.CheckState.Checked:
            channel["individual_axis"] = True
            channel["individual_axis_width"] = self.plot.get_axis(idx).width()
        else:
            channel["individual_axis"] = False

        channel["common_axis"] = item.checkState(item.CommonAxisColumn) == QtCore.Qt.CheckState.Checked
        channel["color"] = sig.color.name()
        channel["computed"] = bool(sig.flags & Signal.Flags.computed)
        channel["ranges"] = copy_ranges(widget.ranges)

        for range_info in channel["ranges"]:
            range_info["background_color"] = range_info["background_color"].name()
            range_info["font_color"] = range_info["font_color"].name()

        channel["precision"] = widget.precision
        channel["fmt"] = widget.fmt
        channel["format"] = widget.format
        channel["mode"] = widget.mode
        if sig.flags & Signal.Flags.computed:
            channel["computation"] = sig.computation

        if sig.flags & Signal.Flags.user_defined_comment:
            channel["comment"] = sig.comment

        if sig.flags & Signal.Flags.user_defined_unit:
            channel["unit"] = sig.unit

        channel["y_range"] = [float(e) for e in sig.y_range]
        channel["origin_uuid"] = str(sig.origin_uuid)

        if sig.flags & Signal.Flags.user_defined_conversion:
            channel["conversion"] = to_dict(sig.conversion)

        return channel

    def channel_selection_changed(self, update=False):
        def set_focused(item):
            if item.type() == item.Channel:
                item.signal.enable = True

            elif item.type() == item.Group:
                for i in range(item.childCount()):
                    set_focused(item.child(i))

        if self.focused_mode:
            for signal in self.plot.signals:
                signal.enable = False

            for item in self.channel_selection.selectedItems():
                set_focused(item)

            self.plot.update()
        else:
            if update:
                for signal in self.plot.signals:
                    signal.enable = False

                iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
                while item := iterator.value():
                    if (
                        item.type() == item.Channel
                        and item.checkState(item.NameColumn) == QtCore.Qt.CheckState.Checked
                        and not item.isDisabled()
                    ):
                        item.signal.enable = True

                    iterator += 1

                self.plot.update()

    def channel_selection_item_changed(self, top_left, bottom_right, roles):
        item = self.channel_selection.itemFromIndex(top_left)

        if item.uuid == self.info_uuid:
            palette = self.selected_channel_value.palette()

            brush = QtGui.QBrush(item.foreground(item.NameColumn))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.WindowText, brush)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.WindowText, brush)

            brush = QtGui.QBrush(item.background(item.NameColumn))
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Window, brush)

            self.selected_channel_value.setPalette(palette)

            value = item.text(item.ValueColumn)
            unit = item.unit

            metrics = QtGui.QFontMetrics(self.selected_channel_value.font())
            elided = metrics.elidedText(
                f"{value} {unit}",
                QtCore.Qt.TextElideMode.ElideRight,
                self.selected_channel_value.width() - 10,
            )

            self.selected_channel_value.setText(elided)  # (f"{value} {unit}")

        if QtCore.Qt.ItemDataRole.CheckStateRole not in roles:
            return

        if item.type() != item.Channel or item.isDisabled():
            return

        column = top_left.column()

        if column == item.NameColumn:
            enabled = item.checkState(column) == QtCore.Qt.CheckState.Checked
            if enabled != item.signal.enable:
                item.signal.enable = enabled
                self.plot.set_signal_enable(item.uuid, item.checkState(column))
            if not enabled and self.channel_selection.hide_disabled_channels:
                item.setHidden(True)

        elif column == item.CommonAxisColumn:
            if not self.locked:
                enabled = item.checkState(column) == QtCore.Qt.CheckState.Checked
                if enabled != item.signal.y_link:
                    item.signal.y_link = enabled
                    self.plot.set_common_axis(item.uuid, enabled)

        elif column == item.IndividualAxisColumn:
            enabled = item.checkState(column) == QtCore.Qt.CheckState.Checked
            if enabled != item.signal.individual_axis:
                self.plot.set_individual_axis(item.uuid, enabled)

    def channel_selection_item_double_clicked(self, item, button):
        if item is None:
            return

        elif item.type() != item.Info:
            if item.type() == item.Channel:
                if not item.isDisabled():
                    if item.checkState(item.NameColumn) == QtCore.Qt.CheckState.Checked:
                        item.setCheckState(item.NameColumn, QtCore.Qt.CheckState.Unchecked)
                    else:
                        item.setCheckState(item.NameColumn, QtCore.Qt.CheckState.Checked)
            elif item.type() == item.Group:
                if (
                    (Plot.item_double_click_handling == "enable/disable" and button == QtCore.Qt.MouseButton.LeftButton)
                    or Plot.item_double_click_handling == "expand/collapse"
                    and button == QtCore.Qt.MouseButton.RightButton
                ):
                    if self.channel_selection.expandsOnDoubleClick():
                        self.channel_selection.setExpandsOnDoubleClick(False)
                    if item.isDisabled():
                        item.set_disabled(False)
                    else:
                        item.set_disabled(True)
                    self.plot.update()
                else:
                    item.setExpanded(not item.isExpanded())

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
            self.info_uuid = None
            self.info.set_stats(None)
            self.selected_channel_value.setText("")
            self.close_request.emit()

    def channel_selection_row_changed(self, current, previous):
        if not self.closed:
            if current and current.type() == ChannelsTreeItem.Channel:
                item = current
                uuid = item.uuid
                self.info_uuid = uuid

                self.plot.set_current_uuid(self.info_uuid)

                if self.info.isVisible():
                    stats = self.plot.get_stats(self.info_uuid)
                    self.info.set_stats(stats)

                if len(self.channel_selection.selectedItems()) == 1:
                    self.flash_curve(current, 0)

    def clear(self):
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_A, QtCore.Qt.KeyboardModifier.ControlModifier
        )
        self.channel_selection.keyPressEvent(event)
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Delete, QtCore.Qt.KeyboardModifier.NoModifier
        )
        self.channel_selection.keyPressEvent(event)

    def close(self):
        self.closed = True

        self.channel_selection.blockSignals(True)
        self.plot.blockSignals(True)
        self.plot._can_paint_global = False
        self.owner = None

        tree = self.channel_selection
        tree.plot = None
        iterator = QtWidgets.QTreeWidgetItemIterator(tree)
        while item := iterator.value():
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

        bookmarks = self.plot.bookmarks
        self.plot.bookmarks = []

        self.verify_bookmarks.emit(bookmarks, self)

        super().close()

    def computation_channel_inserted(self, sig):
        sig.enable = True

        if self.channel_selection.selectedItems():
            item = self.channel_selection.selectedItems()[0]
            item_below = self.channel_selection.itemBelow(item)
            if item_below is None or item_below.parent() != item.parent():
                destination = item.parent()
            else:
                destination = item_below
        else:
            destination = None

        self.add_new_channels({sig.name: sig}, destination=destination)

        self.info_uuid = sig.uuid

        self.plot.set_current_uuid(self.info_uuid, True)

    def compute_fft(self, uuid):
        signal, index = self.plot.signal_by_uuid(uuid)
        try:
            window = FFTWindow(PlotSignal(signal), parent=self)
            window.show()
        except:
            pass

    def current_uuid_changed(self, uuid):
        self.info_uuid = uuid

        if uuid:
            palette = self.selected_channel_value.palette()
            sig, idx = self.plot.signal_by_uuid(uuid)
            brush = QtGui.QBrush(sig.color)
            brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
            palette.setBrush(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.WindowText, brush)
            palette.setBrush(QtGui.QPalette.ColorGroup.Inactive, QtGui.QPalette.ColorRole.WindowText, brush)
            self.selected_channel_value.setPalette(palette)

            item = self.item_by_uuid(uuid)
            if item is not None:
                value = item.text(item.ValueColumn)
                unit = item.unit
                metrics = QtGui.QFontMetrics(self.selected_channel_value.font())
                elided = metrics.elidedText(
                    f"{value} {unit}",
                    QtCore.Qt.TextElideMode.ElideRight,
                    self.selected_channel_value.width() - 10,
                )

                self.selected_channel_value.setText(elided)
        else:
            self.selected_channel_value.setText("")

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

                    item.set_prefix()
                    item.kind = kind
                    item.set_fmt(fmt)

                    item.set_value(value, update=True)

                    if item.uuid == self.info_uuid:
                        value = item.text(item.ValueColumn)
                        unit = item.unit
                        metrics = QtGui.QFontMetrics(self.selected_channel_value.font())
                        elided = metrics.elidedText(
                            f"{value} {unit}",
                            QtCore.Qt.TextElideMode.ElideRight,
                            self.selected_channel_value.width() - 10,
                        )

                        self.selected_channel_value.setText(elided)  # (f"{value} {unit}")

        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        if not self._inhibit_timestamp_signals:
            self.cursor_moved_signal.emit(self, position)

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

    def cursor_removed(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while item := iterator.value():
            if item.type() == item.Channel and not self.plot.region:
                self.cursor_info.update_value()
                item.set_prefix()
                item.set_value("")

            iterator += 1

        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self.cursor_removed_signal.emit(self)

    def curve_clicked(self, uuid):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while item := iterator.value():
            if item.type() == item.Channel and item.uuid == uuid:
                self.channel_selection.clearSelection()
                self.channel_selection.setCurrentItem(item)
                break

            iterator += 1

    def decrease_font(self):
        font = self.font()
        size = font.pointSize()
        pos = bisect.bisect_left(FONT_SIZE, size) - 1
        if pos < 0:
            pos = 0
        new_size = FONT_SIZE[pos]

        self.set_font_size(new_size)

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

    def flash_curve(self, item, column):
        if self.plot.cursor1:
            self.plot.flash_current_signal = 6
            self.plot.update()

    def hide_axes(self, event=None, hide=None):
        if hide is None:
            hide = not self.hide_axes_btn.isFlat()
            self._settings.setValue("plot_hide_axes", hide)

        if hide:
            self.plot.y_axis.hide()
            self.plot.x_axis.hide()
            self.hide_axes_btn.setFlat(True)
            self.hide_axes_btn.setToolTip("Show axes")
        else:
            self.plot.y_axis.show()
            self.plot.x_axis.show()
            self.hide_axes_btn.setFlat(False)
            self.hide_axes_btn.setToolTip("Hide axes")

        if hide:
            png = ":/axis.png"
        else:
            png = ":/axis_on.png"
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(png), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.hide_axes_btn.setIcon(icon)

    def hide_selected_channel_value(self, event=None, hide=None):
        if hide is None:
            hide = not self.selected_channel_value_btn.isFlat()
            self._settings.setValue("plot_hide_selected_channel_value", hide)

        if hide:
            self.selected_channel_value.hide()
            self.selected_channel_value_btn.setFlat(True)
            self.selected_channel_value_btn.setToolTip("Show selected channel value panel")
        else:
            self.selected_channel_value.show()
            self.selected_channel_value_btn.setFlat(False)
            self.selected_channel_value_btn.setToolTip("Hide selected channel value panel")

        if hide:
            png = ":/number.png"
        else:
            png = ":/number_on.png"
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(png), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.selected_channel_value_btn.setIcon(icon)

    def increase_font(self):
        font = self.font()
        size = font.pointSize()
        pos = bisect.bisect_right(FONT_SIZE, size)
        if pos == len(FONT_SIZE):
            pos -= 1
        new_size = FONT_SIZE[pos]

        self.set_font_size(new_size)

    def info_precision_modified(self):
        if not self.closed:
            if self.info_uuid is not None:
                stats = self.plot.get_stats(self.info_uuid)
                self.info.set_stats(stats)

    def _inhibit_timestamp_handler(self):
        self._inhibit_timestamp_signals = False

    def item_by_uuid(self, uuid):
        return self._item_cache.get(uuid, None)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == QtCore.Qt.Key.Key_M and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            event.accept()
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

                if self.info_uuid:
                    stats = self.plot.get_stats(self.info_uuid)
                    self.info.set_stats(stats)

        elif key == QtCore.Qt.Key.Key_2 and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            self.focused_mode = not self.focused_mode
            if self.focused_mode:
                self.focused_mode_btn.setFlat(False)
            else:
                self.focused_mode_btn.setFlat(True)
            self.channel_selection_changed(update=True)

            event.accept()

        elif (
            key in (QtCore.Qt.Key.Key_B, QtCore.Qt.Key.Key_H, QtCore.Qt.Key.Key_P, QtCore.Qt.Key.Key_T)
            and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            selected_items = self.channel_selection.selectedItems() or [
                self.channel_selection.topLevelItem(i) for i in range(self.channel_selection.topLevelItemCount())
            ]

            if key == QtCore.Qt.Key.Key_B:
                fmt = "bin"
            elif key == QtCore.Qt.Key.Key_H:
                fmt = "hex"
            elif key == QtCore.Qt.Key.Key_P:
                fmt = "phys"
            else:
                fmt = "ascii"

            for item in selected_items:
                item_type = item.type()
                if item_type == item.Info:
                    continue

                elif item_type == item.Channel:
                    signal, idx = self.plot.signal_by_uuid(item.uuid)

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

                elif item_type == item.Group:
                    for i in range(item.childCount()):
                        selected_items.append(item.child(i))

                    if item.pattern:
                        item.pattern["integer_format"] = fmt

            if self.info.isVisible():
                stats = self.plot.get_stats(self.info_uuid)
                self.info.set_stats(stats)

            self.current_uuid_changed(self.plot.current_uuid)
            self.plot.update()

            event.accept()

        elif (
            key in (QtCore.Qt.Key.Key_R, QtCore.Qt.Key.Key_S)
            and modifiers == QtCore.Qt.KeyboardModifier.AltModifier
            and self._can_switch_mode
        ):
            selected_items = self.channel_selection.selectedItems()
            if not selected_items:
                signals = [(sig, i) for i, sig in enumerate(self.plot.signals)]
                uuids = [sig.uuid for sig in self.plot.signals]

            else:
                uuids = [item.uuid for item in selected_items if item.type() == ChannelsTreeItem.Channel]

                signals = [self.plot.signal_by_uuid(uuid) for uuid in uuids]

            if signals:
                if key == QtCore.Qt.Key.Key_R:
                    mode = "raw"
                    style = QtCore.Qt.PenStyle.DashLine

                else:
                    mode = "phys"
                    style = QtCore.Qt.PenStyle.SolidLine

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

                        signal.y_range = buttom, top
                        item = self.item_by_uuid(signal.uuid)
                        item._value = None

                        if self.plot.current_uuid == signal.uuid:
                            self.plot.y_axis.mode = mode
                            self.plot.y_axis.picture = None
                            self.plot.y_axis.update()
                            self.plot.viewbox.setYRange(buttom, top, padding=0, update=True)

                        self.plot.get_axis(idx).mode = mode
                        self.plot.get_axis(idx).picture = None
                        self.plot.get_axis(idx).update()

            for uuid in uuids:
                item = self.item_by_uuid(uuid)
                item.setText(item.UnitColumn, item.unit)

            self.plot.update()

            if self.plot.cursor1:
                self.plot.cursor_moved.emit(self.plot.cursor1)

            event.accept()

        elif key == QtCore.Qt.Key.Key_I and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            if self.plot.cursor1:
                position = self.plot.cursor1.value()
                comment, submit = QtWidgets.QInputDialog.getMultiLineText(
                    self,
                    "Insert comments",
                    f"Enter the comments for cursor position {position:.9f}s:",
                    "",
                )
                if submit:
                    visible = True
                    for bookmark in self.plot.bookmarks:
                        visible = bookmark.visible
                        break

                    bookmark = Bookmark(
                        pos=position,
                        message=comment,
                        color="#FF0000",
                        tool=Tool.__tool__,
                    )
                    bookmark.visible = visible
                    bookmark.edited = True

                    self.plot.bookmarks.append(bookmark)
                    self.plot.viewbox.addItem(self.plot.bookmarks[-1])

                    if not visible:
                        self.toggle_bookmarks()

                    self.update()

            event.accept()

        elif key == QtCore.Qt.Key.Key_I and modifiers == QtCore.Qt.KeyboardModifier.AltModifier:
            self.show_bookmarks = not self.show_bookmarks
            if self.show_bookmarks:
                self.bookmark_btn.setFlat(False)
            else:
                self.bookmark_btn.setFlat(True)

            for bookmark in self.plot.bookmarks:
                bookmark.visible = self.show_bookmarks

            self.plot.update()
            event.accept()

        elif key == QtCore.Qt.Key.Key_G and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            selected_items = [
                item for item in self.channel_selection.selectedItems() if item.type() == ChannelsTreeItem.Channel
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
                    offset = diag.offset.value()
                    scale = diag.scaling.value()

                    y_bottom = -offset * scale / 100
                    y_top = y_bottom + scale

                    y_range = y_bottom, y_top

                    # TO DO: should we update the axis here?

                    for idx in indexes:
                        self.plot.signals[idx].y_range = y_range

                    self.zoom_changed()

                    self.plot.update()

            event.accept()

        elif key == QtCore.Qt.Key.Key_R and modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
            iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
            while item := iterator.value():
                if item.type() == item.Channel:
                    item.set_prefix()
                    item.set_value("")

                iterator += 1

            self.plot.keyPressEvent(event)

        elif key == QtCore.Qt.Key.Key_C and modifiers in (
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.KeyboardModifier.ControlModifier,
            QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
        ):
            self.channel_selection.keyPressEvent(event)

        elif (
            key == QtCore.Qt.Key.Key_R
            and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier
            and self.can_edit_ranges
        ):
            self.channel_selection.keyPressEvent(event)

        elif key == QtCore.Qt.Key.Key_V and modifiers in (
            QtCore.Qt.KeyboardModifier.ControlModifier,
            QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
        ):
            self.channel_selection.keyPressEvent(event)

        elif key == QtCore.Qt.Key.Key_BracketLeft and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            self.decrease_font()

        elif key == QtCore.Qt.Key.Key_BracketRight and modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            self.increase_font()

        elif event.keyCombination().toCombined() in self.plot.keyboard_events:
            try:
                self.plot.keyPressEvent(event)
            except:
                print(format_exc())

        elif key == QtCore.Qt.Key.Key_Backspace:
            if modifiers == QtCore.Qt.KeyboardModifier.ShiftModifier:
                self.redo_zoom()
            else:
                self.undo_zoom()

            event.accept()

        elif key == QtCore.Qt.Key.Key_W and modifiers == QtCore.Qt.KeyboardModifier.ShiftModifier:
            if self.enable_zoom_history and self.zoom_history:
                self.zoom_history_index = 0

                snapshot = self.zoom_history[self.zoom_history_index]

                self.plot.block_zoom_signal = True

                for sig in self.plot.signals:
                    y_range = snapshot["y"].get(sig.uuid, None)

                    if y_range is None:
                        continue

                    self.plot.set_y_range(sig.uuid, y_range, emit=False)

                self.plot.viewbox.setXRange(*snapshot["x"], padding=0)

                self.undo_btn.setEnabled(False)
                if len(self.zoom_history) > 1:
                    self.redo_btn.setEnabled(True)

                self.plot.block_zoom_signal = False

            event.accept()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def pattern_group_added_req(self, group):
        self.pattern_group_added.emit(self, group)

    def range_modified(self, region=None):
        if self.plot.region is None:
            return

        start, stop = sorted(self.plot.region.getRegion())

        self.cursor_info.update_value()

        for item in self._visible_items.values():
            if item.type() == item.Channel:
                signal, i = self.plot.signal_by_uuid(item.uuid)

                index = self.plot.get_timestamp_index(start, signal.timestamps)
                start_v, kind, fmt = signal.value_at_index(index)

                index = self.plot.get_timestamp_index(stop, signal.timestamps)
                stop_v, kind, fmt = signal.value_at_index(index)

                if self.region_values_display_mode == "delta":
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

                else:
                    if self.plot.region_lock is not None:
                        if start == self.plot.region_lock:
                            value = stop_v
                        else:
                            value = start_v

                    else:
                        if self._prev_region is None:
                            value = start_v
                        else:
                            if stop == self._prev_region[1]:
                                value = start_v
                            else:
                                value = stop_v

                    item.set_prefix()
                    item.set_fmt(signal.format)

                    if value != "n.a.":
                        if kind in "uif":
                            item.kind = kind
                            item.set_value(value)
                            item.set_fmt(fmt)
                        else:
                            item.set_value("n.a.")
                    else:
                        item.set_value("n.a.")

        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self._prev_region = (start, stop)

        self.region_moved_signal.emit(self, [start, stop])

    def range_modified_finished(self):
        if not self.plot.region:
            return
        start, stop = self.plot.region.getRegion()

        timebase = self.plot.get_current_timebase()

        if timebase.size:
            dim = len(timebase)

            if self.plot.region_lock is None:
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

            else:
                if start == self.plot.region_lock:
                    pos = stop
                else:
                    pos = start

                right = np.searchsorted(timebase, pos, side="right")
                if right == 0:
                    next_pos = timebase[0]
                elif right == dim:
                    next_pos = timebase[-1]
                else:
                    if pos - timebase[right - 1] < timebase[right] - pos:
                        next_pos = timebase[right - 1]
                    else:
                        next_pos = timebase[right]
                pos = next_pos

                self.plot.region.setRegion((self.plot.region_lock, pos))

    def range_removed(self):
        self._prev_region = None
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while item := iterator.value():
            if item.type() == item.Channel:
                item.set_prefix()
                item.set_value("")

            iterator += 1

        self.cursor_info.update_value()

        if self.plot.cursor1:
            self.plot.cursor_moved.emit(self.plot.cursor1)
        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self.region_removed_signal.emit(self)

    def redo_zoom(self):
        if self.enable_zoom_history and self.zoom_history:
            self.zoom_history_index = min(self.zoom_history_index + 1, len(self.zoom_history) - 1)

            snapshot = self.zoom_history[self.zoom_history_index]

            self.plot.block_zoom_signal = True

            for sig in self.plot.signals:
                y_range = snapshot["y"].get(sig.uuid, None)

                if y_range is None:
                    continue

                self.plot.set_y_range(sig.uuid, y_range, emit=False)

            self.plot.viewbox.setXRange(*snapshot["x"], padding=0)

            if self.zoom_history_index == len(self.zoom_history) - 1:
                self.redo_btn.setEnabled(False)
            if len(self.zoom_history) > 1:
                self.undo_btn.setEnabled(True)

            self.plot.block_zoom_signal = False

    def set_conversion(self, uuid, conversion):
        self.plot.set_conversion(uuid, conversion)
        self.cursor_moved()

    def set_font_size(self, size):
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)
        self.channel_selection.set_font_size(size)
        self.plot.y_axis.set_font_size(size)
        self.plot.x_axis.set_font_size(size)

    def set_initial_zoom(self):
        self.zoom_history.clear()
        self.zoom_history_index = -1
        self.zoom_changed(inplace=True)

    def set_locked(self, event=None, locked=None):
        if locked is None:
            locked = not self.locked
            self._settings.setValue("plot_locked", locked)

        if locked:
            tooltip = "The Y axis is locked. Press to unlock"
            png = ":/locked.png"
            self.lock_btn.setFlat(True)
        else:
            tooltip = "The Y axis is unlocked. Press to lock"
            png = ":/unlocked.png"
            self.lock_btn.setFlat(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(png), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.lock_btn.setToolTip(tooltip)
        self.lock_btn.setIcon(icon)

        self.channel_selection.setColumnHidden(self.channel_selection.CommonAxisColumn, locked)

        self.locked = locked
        self.plot.set_locked(locked)

    def set_splitter(self, pos, index):
        self.splitter_moved.emit(self, pos)

    def set_timestamp(self, stamp):
        if self.plot.cursor1 is None:
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_C,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
            self.plot.keyPressEvent(event)

        self._inhibit_timestamp_signals = True
        self._inhibit_timestamp_signals_timer.start(50)
        self.plot.cursor1.setPos(stamp)
        self.cursor_move_finished()

    def _show_properties(self, uuid):
        for sig in self.plot.signals:
            if sig.uuid == uuid:
                if sig.flags & Signal.Flags.computed:
                    try:
                        view = ComputedChannelInfoWindow(sig, self)
                        view.show()
                    except:
                        print(format_exc())
                        raise

                else:
                    self.show_properties.emit([sig.group_index, sig.channel_index, sig])

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
                            range_info["background_color"] = range_info["background_color"].name()

                        pattern["ranges"] = ranges

                    ranges = copy_ranges(item.ranges)

                    for range_info in ranges:
                        range_info["font_color"] = range_info["font_color"].name()
                        range_info["background_color"] = range_info["background_color"].name()

                    channel = self.channel_group_item_to_config(item)
                    channel["channels"] = item_to_config(tree, item) if item.pattern is None else []

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
            "channels": (
                item_to_config(self.channel_selection, self.channel_selection.invisibleRootItem())
                if not self.pattern
                else []
            ),
            "pattern": pattern,
            "splitter": [int(e) for e in self.splitter.sizes()[:2]]
            + [
                0,
            ],
            "x_range": self.plot.viewbox.viewRange()[0],
            "y_axis_width": self.plot.y_axis.width(),
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
            "hide_axes": self.hide_axes_btn.isFlat(),
            "hide_selected_channel_value_panel": self.selected_channel_value_btn.isFlat(),
            "focused_mode": not self.focused_mode_btn.isFlat(),
            "delta_mode": "value" if self.delta_btn.isFlat() else "delta",
            "hide_bookmarks": self.bookmark_btn.isFlat(),
        }

        return config

    def toggle_bookmarks(self, *args, hide=None):
        if hide is not None:
            self.show_bookmarks = hide

        key_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_I,
            QtCore.Qt.KeyboardModifier.AltModifier,
        )
        self.keyPressEvent(key_event)

        if not self.show_bookmarks:
            png = ":/bookmark.png"
        else:
            png = ":/bookmark_on.png"
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(png), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.bookmark_btn.setIcon(icon)

        if hide is None:
            self._settings.setValue("plot_bookmarks", not self.bookmark_btn.isFlat())

    def toggle_focused_mode(self, event=None, focused=None):
        if focused is not None:
            # invert so that the key press event will set the desider focused mode
            self.focused_mode = not focused

        key_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_2, QtCore.Qt.KeyboardModifier.NoModifier
        )
        self.keyPressEvent(key_event)

        if focused is None:
            self._settings.setValue("plot_focused_mode", self.focused_mode)

        if not self.focused_mode:
            self.focused_mode_btn.setFlat(True)
            self.focused_mode_btn.setToolTip("Switch on focused mode")
        else:
            self.focused_mode_btn.setFlat(False)
            self.focused_mode_btn.setToolTip("Switch off focused mode")

        if not self.focused_mode:
            png = ":/focus.png"
        else:
            png = ":/focus_on.png"
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(png), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.focused_mode_btn.setIcon(icon)

    def toggle_region_values_display_mode(self, event=None, mode=None):
        if mode is None:
            self.region_values_display_mode = "delta" if self.region_values_display_mode == "value" else "value"
        else:
            self.region_values_display_mode = mode

        if self.region_values_display_mode == "value":
            self.delta_btn.setFlat(True)
            self.delta_btn.setToolTip("Switch to region cursors delta display mode")
        else:
            self.delta_btn.setFlat(False)
            self.delta_btn.setToolTip("Switch to active region cursor value display mode")

        if self.delta_btn.isFlat():
            png = ":/delta.png"
        else:
            png = ":/delta_on.png"
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(png), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.delta_btn.setIcon(icon)

        if mode is None:
            self._settings.setValue("plot_region_values_display_mode", self.region_values_display_mode)

        self.range_modified()

        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while item := iterator.value():
            if item.type() == ChannelsTreeItem.Channel:
                item.set_value(update=True, force=True)

            iterator += 1

    def undo_zoom(self):
        if self.enable_zoom_history and self.zoom_history:
            self.zoom_history_index = max(self.zoom_history_index - 1, 0)

            snapshot = self.zoom_history[self.zoom_history_index]

            self.plot.block_zoom_signal = True

            for sig in self.plot.signals:
                y_range = snapshot["y"].get(sig.uuid, None)

                if y_range is None:
                    continue

                self.plot.set_y_range(sig.uuid, y_range, emit=False)

            self.plot.viewbox.setXRange(*snapshot["x"], padding=0)

            if self.zoom_history_index == 0:
                self.undo_btn.setEnabled(False)
            if len(self.zoom_history) > 1:
                self.redo_btn.setEnabled(True)

            self.plot.block_zoom_signal = False

    def update_current_values(self, *args):
        if self.plot.region:
            self.range_modified(None)
        else:
            self.cursor_moved()

    def _update_visibile_entries(self):
        with self.lock:
            _item_cache = self._item_cache = {}
            _visible_entries = self._visible_entries = set()
            _visible_items = self._visible_items = {}
            iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)

            while item := iterator.value():
                iterator += 1
                if item.type() == ChannelsTreeItem.Channel:
                    _item_cache[item.uuid] = item

                    if (
                        item.uuid == self.info_uuid
                        or item.exists
                        and (item.checkState(item.NameColumn) == QtCore.Qt.CheckState.Checked or item._is_visible)
                    ):
                        entry = (item.origin_uuid, item.signal.name, item.uuid)
                        _visible_entries.add(entry)
                        _visible_items[entry] = item

            self.visible_entries_modified = True

        if self.plot.cursor1 is not None:
            self.cursor_moved()

    def visible_entries(self):
        return self._visible_entries

    def visible_items(self):
        return self._visible_items

    def xrange_changed(self, *args):
        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        if not self._inhibit_x_range_changed_signal:
            self.x_range_changed_signal.emit(self, self.plot.viewbox.viewRange()[0])

    def zoom_changed(self, inplace=False):
        if self.enable_zoom_history and self.plot.signals and not self.plot.block_zoom_signal:
            snapshot = {
                "x": self.plot.viewbox.viewRange()[0],
                "y": {sig.uuid: sig.y_range for sig in self.plot.signals},
            }

            if inplace:
                if not self.zoom_history:
                    self.zoom_history.append(snapshot)
                    self.zoom_history_index = 0
                else:
                    self.zoom_history[self.zoom_history_index] = snapshot
            else:
                if not self.zoom_history or self.zoom_history[self.zoom_history_index] != snapshot:
                    self.zoom_history = self.zoom_history[: self.zoom_history_index + 1]

                    self.zoom_history.append(snapshot)
                    self.zoom_history_index = len(self.zoom_history) - 1

                self.redo_btn.setEnabled(False)
                if len(self.zoom_history) > 1:
                    self.undo_btn.setEnabled(True)


class PlotGraphics(pg.PlotWidget):
    cursor_moved = QtCore.Signal(object)
    cursor_removed = QtCore.Signal()
    range_removed = QtCore.Signal()
    range_modified = QtCore.Signal(object)
    range_modified_finished = QtCore.Signal(object)
    cursor_move_finished = QtCore.Signal(object)
    xrange_changed = QtCore.Signal(object, object)
    computation_channel_inserted = QtCore.Signal(object)
    curve_clicked = QtCore.Signal(str)
    signals_enable_changed = QtCore.Signal()
    current_uuid_changed = QtCore.Signal(str)
    edit_channel_request = QtCore.Signal(object, object)

    add_channels_request = QtCore.Signal(list)
    zoom_changed = QtCore.Signal(bool)

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
        viewBox = ViewBoxWithCursor(plot=self)
        self.initial_x_range = "adjust"
        super().__init__(viewBox=viewBox)

        # del self.plotItem.vb
        # self.plotItem.vb = ViewBox(parent=self.plotItem)
        #
        # self.plotItem.vb.sigStateChanged.connect(self.plotItem.viewStateChanged)
        # self.plotItem.vb.sigRangeChanged.connect(self.plotItem.sigRangeChanged)
        # self.plotItem.vb.sigXRangeChanged.connect(self.plotItem.sigXRangeChanged)
        # self.plotItem.vb.sigYRangeChanged.connect(self.plotItem.sigYRangeChanged)
        # self.plotItem.layout.addItem(self.plotItem.vb, 2, 1)
        self.plotItem.vb.setLeftButtonAction(Plot.mouse_mode)

        self.lock = Lock()

        self.bookmarks = []

        self.plot_parent = plot_parent

        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        self.autoFillBackground()

        self._pixmap = None

        self.locked = False

        self.cursor_unit = "s" if x_axis == "time" else "Hz"

        self.line_interconnect = line_interconnect if line_interconnect != "line" else ""

        self._can_trim = True
        self._can_paint = True
        self._can_compute_all_timebase = True
        self._can_paint_global = True
        self.mdf = mdf

        self._can_paint = True

        self.setAcceptDrops(True)

        self._last_size = self.geometry()
        self._settings = QtCore.QSettings()

        self.setContentsMargins(5, 5, 5, 5)
        self.xrange_changed.connect(self.xrange_changed_handle)
        self.with_dots = with_dots

        self.current_uuid = None

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

        self.viewbox.sigCursorMoved.connect(self._cursor_moved)
        self.viewbox.sigZoomFinished.connect(self._cursor_zoom_finished)
        self.viewbox.sigZoomChanged.connect(self._cursor_zoom)

        self.x_range = self.y_range = (0, 1)
        self._curve = pg.PlotCurveItem(
            np.array([]),
            np.array([]),
            stepMode=self.line_interconnect,
            skipFiniteCheck=False,
            connect="finite",
        )

        self.scene_.contextMenu = []
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

            self.cursor1 = Cursor(pos=pos, angle=90, movable=True, pen=color, hoverPen=color)

            self.viewbox.cursor = self.cursor1
            self.viewbox.addItem(self.cursor1, ignoreBounds=True)

            self.cursor1.sigPositionChanged.connect(self.cursor_moved.emit)
            self.cursor1.sigPositionChangeFinished.connect(self.cursor_move_finished.emit)
            self.cursor_move_finished.emit(self.cursor1)
            self.cursor1.show()
        else:
            self.cursor1 = None

        self.viewbox.sigYRangeChanged.connect(self.y_changed)
        self.viewbox.sigRangeChangedManually.connect(self.y_changed)

        self.x_axis = FormatedAxis(
            "bottom", maxTickLength=5, background=self.backgroundBrush().color(), linked_signal=(self, None)
        )

        if x_axis == "time":
            fmt = self._settings.value("plot_xaxis")
            if fmt == "seconds" or not fmt:
                fmt = "phys"
        else:
            fmt = "phys"
        self.x_axis.format = fmt
        self.x_axis.origin = origin

        self.y_axis = FormatedAxis(
            "left", maxTickLength=-5, background=self.backgroundBrush().color(), linked_signal=(self, None)
        )
        self.y_axis.setWidth(48)

        self.y_axis.scale_editor_requested.connect(self.open_scale_editor)
        self.y_axis.rangeChanged.connect(self.set_y_range)

        self.plot_item.setAxisItems({"left": self.y_axis, "bottom": self.x_axis})

        def plot_item_wheel_event(event):
            if event is not None:
                pos = event.pos()

                if pos.x() <= self.y_axis.width():
                    self.y_axis.wheelEvent(event)
                else:
                    for axis in self.axes:
                        if isinstance(axis, FormatedAxis) and axis.isVisible():
                            rect = axis.sceneBoundingRect()
                            if rect.contains(pos):
                                axis.wheelEvent(event)
                                break

        def plot_item_mousePressEvent(event):
            if event is not None:
                pos = event.pos()

                if pos.x() <= self.y_axis.width():
                    if not self.locked:
                        self.y_axis.mousePressEvent(event)
                else:
                    for axis in self.axes:
                        if isinstance(axis, FormatedAxis) and axis.isVisible():
                            rect = axis.sceneBoundingRect()
                            if rect.contains(pos):
                                if not self.locked:
                                    axis.mousePressEvent(event)
                                break
                    else:
                        self.plot_item._mousePressEvent(event)

        def plot_item_mouseMoveEvent(event):
            if event is not None:
                pos = event.pos()

                if pos.x() <= self.y_axis.width():
                    self.y_axis.mouseMoveEvent(event)
                else:
                    for axis in self.axes:
                        if isinstance(axis, FormatedAxis) and axis.isVisible():
                            rect = axis.sceneBoundingRect()
                            if rect.contains(pos):
                                if not self.locked:
                                    axis.mouseMoveEvent(event)
                                break
                    else:
                        self.plot_item._mouseMoveEvent(event)

        def plot_item_mouseReleaseEvent(event):
            if event is not None:
                pos = event.pos()

                if pos.x() <= self.y_axis.width():
                    self.y_axis.mouseReleaseEvent(event)
                else:
                    for axis in self.axes:
                        if isinstance(axis, FormatedAxis) and axis.isVisible():
                            rect = axis.sceneBoundingRect()
                            if rect.contains(pos):
                                if not self.locked:
                                    axis.mouseReleaseEvent(event)
                                break
                    else:
                        self.plot_item._mouseReleaseEvent(event)

        self.plot_item.wheelEvent = plot_item_wheel_event
        self.plot_item._mousePressEvent = self.plot_item.mousePressEvent
        self.plot_item._mouseMoveEvent = self.plot_item.mouseMoveEvent
        self.plot_item._mouseReleaseEvent = self.plot_item.mouseReleaseEvent
        self.plot_item.mousePressEvent = plot_item_mousePressEvent
        self.plot_item.mouseMoveEvent = plot_item_mouseMoveEvent
        self.plot_item.mouseReleaseEvent = plot_item_mouseReleaseEvent

        self.viewbox_geometry = self.viewbox.sceneBoundingRect()

        self.viewbox.sigResized.connect(partial(self.xrange_changed_handle, force=True))

        self._uuid_map = {}

        self._enabled_changed_signals = []
        self._enable_timer = QtCore.QTimer()
        self._enable_timer.setSingleShot(True)
        self._enable_timer.timeout.connect(self._signals_enabled_changed_handler)

        self._inhibit = False

        self.viewbox.setXRange(0, 10, update=False)

        if signals:
            self.add_new_channels(signals)

        self.viewbox.sigXRangeChanged.connect(self.xrange_changed.emit)

        self.keyboard_events = {
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_F,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_F,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_G,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_G,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_I,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_O,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_I,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_O,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_X,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_R,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.Key.Key_S,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_S,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_S,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_Y,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_Left,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_Right,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_Left,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_Right,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.Key.Key_Left,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ControlModifier,
                QtCore.Qt.Key.Key_Right,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_Up,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_Down,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_PageUp,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.ShiftModifier,
                QtCore.Qt.Key.Key_PageDown,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_H,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_W,
            ).toCombined(),
            QtCore.QKeyCombination(
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.Key.Key_Insert,
            ).toCombined(),
        }

        events = events or []

        for i, event_info in enumerate(events):
            color = COLORS[COLORS_COUNT - (i % COLORS_COUNT) - 1]
            if isinstance(event_info, (list, tuple)):
                to_display = event_info
                labels = [" - Start", " - End"]
            else:
                to_display = [event_info]
                labels = [""]
            for event, label in zip(to_display, labels):
                bookmark = Bookmark(
                    pos=event["value"],
                    message=event["description"],
                    title=f'{event["type"]}{label}',
                    color=color,
                    tool=event.get("tool", ""),
                )
                self.bookmarks.append(bookmark)
                self.viewbox.addItem(bookmark)

        self.viewbox.sigResized.connect(self.update_views)
        if signals:
            self.update_views()

        self.zoom = None

        self.px = 1
        self.py = 1

        self.last_click = perf_counter()
        self.flash_current_signal = 0
        self.flash_curve_timer = QtCore.QTimer()
        self.flash_curve_timer.setSingleShot(True)
        self.flash_curve_timer.timeout.connect(self.update)

        self.block_zoom_signal = False

    def add_new_channels(self, channels, descriptions=None):
        descriptions = descriptions or {}

        initial_index = len(self.signals)
        self._can_paint = False

        for sig in channels.values():
            if not sig.flags & Signal.Flags.computed:
                sig.computation = {}

        if initial_index == 0:
            start_t, stop_t = np.inf, -np.inf
            for sig in channels.values():
                if len(sig):
                    start_t = min(start_t, sig.timestamps[0])
                    stop_t = max(stop_t, sig.timestamps[-1])

            if (start_t, stop_t) != (np.inf, -np.inf):
                if self.initial_x_range == "adjust":
                    self.viewbox.setXRange(start_t, stop_t, update=False)
                else:
                    delta = self.viewbox.viewRange()[0][1] - self.viewbox.viewRange()[0][0]
                    stop_t = start_t + delta
                    self.viewbox.setXRange(start_t, stop_t, padding=0, update=False)

        (start, stop), _ = self.viewbox.viewRange()

        width = self.viewbox.sceneBoundingRect().width()
        trim_info = start, stop, width

        channels = [
            PlotSignal(sig, i, trim_info=trim_info) for i, sig in enumerate(channels.values(), len(self.signals))
        ]

        self.signals.extend(channels)
        for sig in channels:
            uuids = self._timebase_db.setdefault(id(sig.timestamps), set())
            uuids.add(sig.uuid)
        self._compute_all_timebase()

        self._uuid_map = {sig.uuid: (sig, i) for i, sig in enumerate(self.signals)}

        axis_uuid = None

        for index, sig in enumerate(channels, initial_index):
            description = descriptions.get(sig.uuid, {})
            if description:
                sig.enable = description.get("enabled", True)
                sig.format = description.get("format", "phys")

            y_range = description.get("y_range", None)
            if y_range:
                mn, mx = y_range
                if mn > mx:
                    y_range = mx, mn
                else:
                    y_range = tuple(y_range)

                sig.y_range = y_range

            elif not sig.empty:
                sig.y_range = sig.min, sig.max

            self.axes.append(self._axes_layout_pos)
            self._axes_layout_pos += 1

            if initial_index == 0 and index == 0:
                axis_uuid = sig.uuid

        if axis_uuid is not None:
            self.set_current_uuid(sig.uuid)
            if len(self.all_timebase) and self.cursor1 is not None:
                self.cursor1.set_value(self.all_timebase[0])

        self.viewbox._matrixNeedsUpdate = True
        self.viewbox.updateMatrix()

        self.zoom_changed.emit(True)

        return {sig.uuid: sig for sig in channels}

    def auto_clip_rect(self, painter):
        rect = self.viewbox.sceneBoundingRect()
        painter.setClipRect(rect.x() + 5, rect.y(), rect.width() - 5, rect.height())
        painter.setClipping(True)

    def _clicked(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        pos = self.plot_item.vb.mapSceneToView(event.scenePos()).x()
        start, stop = self.viewbox.viewRange()[0]
        if not start <= pos <= stop:
            return

        scene_pos = event.scenePos()
        pos = self.plot_item.vb.mapSceneToView(scene_pos)
        x = pos.x()
        y = event.scenePos().y()

        for bookmark in self.bookmarks:
            if not bookmark.visible:
                continue

            rect = bookmark.label.textItem.sceneBoundingRect()
            if rect.contains(scene_pos):
                if bookmark.editable:
                    edit_rect = QtCore.QRectF(
                        rect.x() + rect.width() - 35,
                        rect.y() + 1,
                        18,
                        18,
                    )

                    if edit_rect.contains(scene_pos):
                        comment, submit = QtWidgets.QInputDialog.getMultiLineText(
                            self,
                            "Edit bookmark comments",
                            f"Enter new comments for cursor position {bookmark.value():.9f}s:",
                            bookmark.message,
                        )
                        if submit:
                            bookmark.message = comment
                            bookmark.edited = True

                    delete_rect = QtCore.QRectF(
                        rect.x() + rect.width() - 18,
                        rect.y() + 1,
                        18,
                        18,
                    )
                    if delete_rect.contains(scene_pos):
                        bookmark.deleted = True
                        bookmark.visible = False

                break
        else:
            if (
                QtCore.QKeyCombination(QtCore.Qt.Key.Key_C, QtCore.Qt.KeyboardModifier.ControlModifier).toCombined()
                not in self.disabled_keys
            ):
                if self.region is not None:
                    start, stop = self.region.getRegion()

                    if self.region_lock is not None:
                        self.region.setRegion((self.region_lock, pos.x()))
                    else:
                        if modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
                            self.region.setRegion((start, pos.x()))
                        else:
                            self.region.setRegion((pos.x(), stop))

                else:
                    if self.cursor1 is not None:
                        self.cursor1.setPos(pos)
                        self.cursor1.sigPositionChangeFinished.emit(self.cursor1)

            now = perf_counter()
            if modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
                self.select_curve(x, y)
            elif now - self.last_click < 0.3:
                self.select_curve(x, y)

        self.last_click = perf_counter()

    def close(self):
        self._can_paint_global = False
        super().close()

    def _compute_all_timebase(self):
        if self._can_compute_all_timebase:
            if self._timebase_db:
                stamps = {id(sig.timestamps): sig.timestamps for sig in self.signals}

                timebases = [timestamps for id_, timestamps in stamps.items() if id_ in self._timebase_db]

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

    def _cursor_moved(self, event):
        pos = self.plot_item.vb.mapSceneToView(event.scenePos()).x()
        start, stop = self.viewbox.viewRange()[0]
        if not start <= pos <= stop:
            return

        scene_pos = event.scenePos()
        pos = self.plot_item.vb.mapSceneToView(scene_pos)
        x = pos.x()
        y = event.scenePos().y()

        if self.cursor1 is not None:
            self.cursor1.setPos(pos)
            self.cursor1.sigPositionChangeFinished.emit(self.cursor1)

    def _cursor_zoom(self, zoom):
        self.zoom = zoom
        if zoom is not None:
            self.update()

    def _cursor_zoom_finished(self, zoom=None):
        p1, p2, zoom_mode = zoom

        self.block_zoom_signal = True

        if zoom_mode in (self.viewbox.Y_zoom, *self.viewbox.XY_zoom) and not self.locked:
            y1, y2 = sorted([p1.y(), p2.y()])
            y_bottom, y_top = self.viewbox.viewRange()[1]
            r_top = (y2 - y_bottom) / (y_top - y_bottom)
            r_bottom = (y1 - y_bottom) / (y_top - y_bottom)

            for sig in self.signals:
                sig_y_bottom, sig_y_top = sig.y_range
                delta = sig_y_top - sig_y_bottom
                sig_y_top = sig_y_bottom + r_top * delta
                sig_y_bottom = sig_y_bottom + r_bottom * delta

                sig, idx = self.signal_by_uuid(sig.uuid)

                axis = self.axes[idx]
                if isinstance(axis, FormatedAxis):
                    axis.setRange(sig_y_bottom, sig_y_top)
                else:
                    self.set_y_range(sig.uuid, (sig_y_bottom, sig_y_top))

        if zoom_mode in (self.viewbox.X_zoom, *self.viewbox.XY_zoom):
            x1, x2 = sorted([p1.x(), p2.x()])
            self.viewbox.setXRange(x1, x2, padding=0)

        self.block_zoom_signal = False
        self.zoom_changed.emit(False)

    def curve_clicked_handle(self, curve, ev, uuid):
        self.curve_clicked.emit(uuid)

    def delete_channels(self, deleted):
        self._can_paint = False

        needs_timebase_compute = False

        uuid_map = self._uuid_map

        indexes = sorted(
            [(uuid_map[uuid][1], uuid) for uuid in deleted if uuid in uuid_map],
            reverse=True,
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
        self.update()

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat("application/octet-stream-asammdf"):
            e.accept()
        super().dragEnterEvent(e)

    def draw_grids(self, paint, event_rect):
        ratio = self.devicePixelRatio()
        if self.y_axis.grid or self.x_axis.grid:
            rect = self.viewbox.sceneBoundingRect()
            y_delta = rect.y()
            x_delta = rect.x()

            if self.y_axis.grid and self.y_axis.isVisible():
                for pen, p1, p2 in self.y_axis.tickSpecs:
                    pen2 = fn.mkPen(pen)
                    pen2.setStyle(QtCore.Qt.PenStyle.DashLine)
                    y_pos = p1.y() / ratio + y_delta
                    paint.setPen(pen2)
                    paint.drawLine(
                        QtCore.QPointF(0, y_pos),
                        QtCore.QPointF(event_rect.x() + event_rect.width(), y_pos),
                    )

            if self.x_axis.grid and self.x_axis.isVisible():
                for pen, p1, p2 in self.x_axis.tickSpecs:
                    pen2 = fn.mkPen(pen)
                    pen2.setStyle(QtCore.Qt.PenStyle.DashLine)
                    x_pos = p1.x() / ratio + x_delta
                    paint.setPen(pen2)
                    paint.drawLine(
                        QtCore.QPointF(x_pos, 0),
                        QtCore.QPointF(x_pos, event_rect.y() + event_rect.height()),
                    )

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

    def edit_computation(self, item):
        signal = item.signal
        functions = self.plot_parent.owner.functions

        mdf = self.mdf or self.plot_parent.owner.generate_mdf()
        dlg = DefineChannel(
            mdf=mdf,
            computation=signal.computation,
            functions=functions,
            parent=self,
        )
        dlg.setModal(True)
        dlg.exec_()
        computed_channel = dlg.result

        if self.mdf is None:
            mdf.close()

        if computed_channel is not None:
            self.edit_channel_request.emit(computed_channel, item)

    def generatePath(self, x, y, sig=None):
        if sig is None or sig.path is None:
            if x is None or len(x) == 0 or y is None or len(y) == 0:
                path = QtGui.QPainterPath()
            else:
                path = self._curve.generatePath(x, y)
            if sig is not None:
                sig.path = path
        else:
            path = sig.path

        return path

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
                locked=self.locked,
                maxTickLength=5,
                background=self.backgroundBrush().color(),
                linked_signal=(self, sig.uuid),
            )
            axis.scale_editor_requested.connect(self.open_scale_editor)
            if sig.conversion and hasattr(sig.conversion, "text_0"):
                axis.text_conversion = sig.conversion

            axis.setRange(*sig.y_range)

            axis.rangeChanged.connect(self.set_y_range)
            axis.hide()
            self.layout.addItem(axis, 2, position)

            self.axes[index] = axis

        return axis

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

    def get_stats(self, uuid):
        try:
            sig, index = self.signal_by_uuid(uuid)
        except KeyError:
            return {}
        else:
            return sig.get_stats(
                cursor=self.cursor1.value() if self.cursor1 else None,
                region=self.region.getRegion() if self.region else None,
                view_region=self.viewbox.viewRange()[0],
                precision=self._settings.value("stats_float_precision", sig.precision, type=int),
            )

    def get_timestamp_index(self, timestamp, timestamps):
        key = id(timestamps), timestamp
        if key in self._timestamps_indexes:
            return self._timestamps_indexes[key]
        else:
            if timestamps.size:
                if timestamp >= timestamps[-1]:
                    index = -1
                elif timestamp <= timestamps[0]:
                    index = 0
                else:
                    index = np.searchsorted(timestamps, timestamp, side="right") - 1
            else:
                index = None

            if len(self._timestamps_indexes) > 100000:
                self._timestamps_indexes.clear()
            self._timestamps_indexes[key] = index
            return index

    def insert_computation(self, name=""):
        functions = self.plot_parent.owner.functions
        if not functions:
            MessageBox.warning(
                self,
                "Cannot add computed channel",
                "There is no user defined function. Create new function using the Functions Manger (F6)",
            )
            return

        mdf = self.mdf or self.plot_parent.owner.generate_mdf()
        dlg = DefineChannel(
            mdf=mdf,
            computation=None,
            functions=functions,
            parent=self,
        )
        dlg.setModal(True)
        dlg.exec_()
        computed_channel = dlg.result

        if self.mdf is None:
            mdf.close()

        if computed_channel is not None:
            self.add_channels_request.emit([computed_channel])

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if event.keyCombination().toCombined() in self.disabled_keys:
            super().keyPressEvent(event)
        else:
            handled = True
            if key == QtCore.Qt.Key.Key_Y and modifier == QtCore.Qt.KeyboardModifier.NoModifier:
                if self.region is None:
                    event_ = QtGui.QKeyEvent(
                        QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_R, QtCore.Qt.KeyboardModifier.NoModifier
                    )
                    self.keyPressEvent(event_)

                if self.region_lock is not None:
                    self.region_lock = None
                    self.region.lines[0].setMovable(True)
                    self.region.lines[0].locked = False
                    self.region.movable = True
                else:
                    self.region_lock = self.region.getRegion()[0]
                    self.region.lines[0].setMovable(False)
                    self.region.lines[0].locked = True
                    self.region.movable = False

                self.update()

            elif key == QtCore.Qt.Key.Key_X and modifier == QtCore.Qt.KeyboardModifier.NoModifier:
                if self.region is not None:
                    self.viewbox.setXRange(*self.region.getRegion(), padding=0)
                    event_ = QtGui.QKeyEvent(
                        QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_R, QtCore.Qt.KeyboardModifier.NoModifier
                    )
                    self.keyPressEvent(event_)

            elif key == QtCore.Qt.Key.Key_F and modifier == QtCore.Qt.KeyboardModifier.NoModifier and not self.locked:
                self.block_zoom_signal = True
                if self.common_axis_items:
                    if any(
                        len(self.signal_by_uuid(uuid)[0].plot_samples)
                        for uuid in self.common_axis_items
                        if self.signal_by_uuid(uuid)[0].enable
                    ):
                        common_min = np.nanmin(
                            [
                                self.signal_by_uuid(uuid)[0].min
                                for uuid in self.common_axis_items
                                if len(self.signal_by_uuid(uuid)[0].plot_samples)
                            ]
                        )
                        common_max = np.nanmax(
                            [
                                self.signal_by_uuid(uuid)[0].max
                                for uuid in self.common_axis_items
                                if len(self.signal_by_uuid(uuid)[0].plot_samples)
                            ]
                        )
                    else:
                        common_min, common_max = 0, 1

                for i, signal in enumerate(self.signals):
                    if len(signal.plot_samples):
                        if signal.uuid in self.common_axis_items:
                            min_ = common_min
                            max_ = common_max
                        else:
                            samples = signal.plot_samples
                            if len(samples):
                                min_, max_ = signal.min, signal.max
                            else:
                                min_, max_ = 0, 1

                        if min_ != min_:  # noqa: PLR0124
                            # min_ is NaN
                            min_ = 0
                        if max_ != max_:  # noqa: PLR0124
                            # max_ is NaN
                            max_ = 1

                        delta = 0.01 * (max_ - min_)
                        min_, max_ = min_ - delta, max_ + delta

                        signal.y_range = min_, max_

                        if signal.uuid == self.current_uuid:
                            self.viewbox.setYRange(min_, max_, padding=0)

                self.block_zoom_signal = False
                self.zoom_changed.emit(False)
                self.update()

            elif (
                key == QtCore.Qt.Key.Key_F and modifier == QtCore.Qt.KeyboardModifier.ShiftModifier and not self.locked
            ):
                self.block_zoom_signal = True
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
                        samples = signal.plot_samples
                        if len(samples):
                            min_, max_ = signal.min, signal.max
                        else:
                            min_, max_ = 0, 1

                        if min_ != min_:  # noqa: PLR0124
                            # min_ is NaN
                            min_ = 0
                        if max_ != max_:  # noqa: PLR0124
                            # max is NaN
                            max_ = 1

                        delta = 0.01 * (max_ - min_)
                        min_, max_ = min_ - delta, max_ + delta

                        signal.y_range = min_, max_
                        if signal.uuid == self.current_uuid:
                            self.viewbox.setYRange(min_, max_, padding=0)

                self.block_zoom_signal = False
                self.zoom_changed.emit(False)
                self.update()

            elif key == QtCore.Qt.Key.Key_G:
                if modifier == QtCore.Qt.KeyboardModifier.NoModifier:
                    y = self.y_axis.grid
                    x = self.x_axis.grid

                    if x and y:
                        self.y_axis.grid = False
                        self.x_axis.grid = False
                    elif x:
                        self.y_axis.grid = True
                        self.x_axis.grid = True
                    else:
                        self.y_axis.grid = False
                        self.x_axis.grid = True

                    self.x_axis.picture = None
                    self.y_axis.picture = None

                    self.update()

                elif modifier == QtCore.Qt.KeyboardModifier.ShiftModifier:
                    if self.cursor1 is not None:
                        value, ok = QtWidgets.QInputDialog.getDouble(
                            self,
                            "Go to time stamp",
                            "Time stamp",
                            value=self.cursor1.value(),
                            decimals=9,
                        )

                        if ok:
                            self.cursor1.setPos(value)
                            self.cursor_move_finished.emit(self.cursor1)

            elif (
                key in (QtCore.Qt.Key.Key_I, QtCore.Qt.Key.Key_O) and modifier == QtCore.Qt.KeyboardModifier.NoModifier
            ):
                x_range, _ = self.viewbox.viewRange()
                delta = x_range[1] - x_range[0]
                if key == QtCore.Qt.Key.Key_I:
                    step = -delta * 0.25
                else:
                    step = delta * 0.5
                if (
                    self.cursor1
                    and self.cursor1.isVisible()
                    and self._settings.value("zoom_x_center_on_cursor", True, type=bool)
                ):
                    pos = self.cursor1.value()
                    x_range = pos - delta / 2, pos + delta / 2

                self.viewbox.setXRange(x_range[0] - step, x_range[1] + step, padding=0)

            elif (
                key in (QtCore.Qt.Key.Key_I, QtCore.Qt.Key.Key_O)
                and modifier == QtCore.Qt.KeyboardModifier.ShiftModifier
                and not self.locked
            ):
                if key == QtCore.Qt.Key.Key_I:
                    factor = 0.165
                else:
                    factor = -0.165

                self.block_zoom_signal = True

                if self._settings.value("zoom_y_center_on_cursor", True, type=bool):
                    value_info = self.value_at_cursor()
                    if not isinstance(value_info[0], (int, float)):
                        delta_proc = 0
                    else:
                        y, sig_y_bottom, sig_y_top = value_info
                        delta_proc = (y - (sig_y_top + sig_y_bottom) / 2) / (sig_y_top - sig_y_bottom)
                else:
                    delta_proc = 0

                for sig in self.signals:
                    sig_y_bottom, sig_y_top = sig.y_range

                    # center on the signal cursor Y value
                    shift = delta_proc * (sig_y_top - sig_y_bottom)
                    sig_y_top, sig_y_bottom = sig_y_top + shift, sig_y_bottom + shift

                    delta = sig_y_top - sig_y_bottom
                    sig_y_top -= delta * factor
                    sig_y_bottom += delta * factor

                    sig, idx = self.signal_by_uuid(sig.uuid)

                    axis = self.axes[idx]
                    if isinstance(axis, FormatedAxis):
                        axis.setRange(sig_y_bottom, sig_y_top)
                    else:
                        self.set_y_range(sig.uuid, (sig_y_bottom, sig_y_top))

                self.block_zoom_signal = False
                self.zoom_changed.emit(False)

            elif key == QtCore.Qt.Key.Key_R and modifier == QtCore.Qt.KeyboardModifier.NoModifier:
                if self.region is None:
                    color = self.cursor1.pen.color().name()

                    self.region = Region(
                        (0, 0),
                        pen=color,
                        hoverPen=color,
                        show_circle=self.cursor1.show_circle,
                        show_horizontal_line=self.cursor1.show_horizontal_line,
                        line_width=self.cursor1.line_width,
                    )
                    self.region.setZValue(-10)
                    self.viewbox.addItem(self.region)
                    self.region.sigRegionChanged.connect(self.range_modified.emit)
                    self.region.sigRegionChanged.connect(self.range_modified_handler)
                    self.region.sigRegionChangeFinished.connect(self.range_modified_finished_handler)
                    start, stop = self.viewbox.viewRange()[0]
                    start, stop = (
                        start + 0.1 * (stop - start),
                        stop - 0.1 * (stop - start),
                    )
                    self.region.setRegion((start, stop))

                    if self.cursor1 is not None:
                        self.cursor1.hide()
                        self.region.setRegion(tuple(sorted((self.cursor1.value(), stop))))

                else:
                    self.region_lock = None
                    self.region.setParent(None)
                    self.region.hide()
                    self.region.deleteLater()
                    self.region = None
                    self.range_removed.emit()

                    if self.cursor1 is not None:
                        self.cursor1.show()

                self.update()

            elif key == QtCore.Qt.Key.Key_S and modifier == QtCore.Qt.KeyboardModifier.ControlModifier:
                file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save as measurement file",
                    "",
                    "MDF version 4 files (*.mf4 *.mf4z)",
                )

                if file_name:
                    signals = [signal for signal in self.signals if signal.enable]
                    if signals:
                        with mdf_module.MDF() as mdf:
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

                            file_name = Path(file_name)

                            if file_name.suffix.lower() in (".zip", ".mf4z"):
                                tmpf = Path(gettempdir()) / f"{perf_counter()}.mf4"
                                mdf.save(tmpf, overwrite=True, compression=2)

                                zipped_mf4 = ZipFile(file_name, "w", compression=ZIP_DEFLATED)
                                zipped_mf4.write(
                                    str(tmpf),
                                    file_name.with_suffix(".mf4").name,
                                    compresslevel=1,
                                )

                                tmpf.unlink()

                            else:
                                mdf.save(file_name, overwrite=True, compression=2)

            elif key == QtCore.Qt.Key.Key_S and modifier == QtCore.Qt.KeyboardModifier.NoModifier and not self.locked:
                self.block_zoom_signal = True
                parent = self.parent().parent()
                uuids = []

                iterator = QtWidgets.QTreeWidgetItemIterator(parent.channel_selection)
                while item := iterator.value():
                    if item.type() == ChannelsTreeItem.Channel and item.signal.enable:
                        uuids.append(item.uuid)

                    iterator += 1

                uuids = reversed(uuids)

                count = sum(
                    1
                    for sig in self.signals
                    if sig.min != "n.a." and sig.enable and sig.uuid not in self.common_axis_items
                )

                if any(sig.min != "n.a." and sig.enable and sig.uuid in self.common_axis_items for sig in self.signals):
                    count += 1

                    common_min_ = np.nanmin(
                        [
                            self.signal_by_uuid(uuid)[0].min
                            for uuid in self.common_axis_items
                            if len(self.signal_by_uuid(uuid)[0].plot_samples) and self.signal_by_uuid(uuid)[0].enable
                        ]
                    )
                    common_max_ = np.nanmax(
                        [
                            self.signal_by_uuid(uuid)[0].max
                            for uuid in self.common_axis_items
                            if len(self.signal_by_uuid(uuid)[0].plot_samples) and self.signal_by_uuid(uuid)[0].enable
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

                            if signal.uuid == self.current_uuid:
                                self.viewbox.setYRange(min_, max_, padding=0)

                            position += 1

                else:
                    xrange, _ = self.viewbox.viewRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.setXRange(*xrange, padding=0)
                    self.viewbox.disableAutoRange()

                self.block_zoom_signal = False
                self.zoom_changed.emit(False)

                self.update()

            elif (
                key == QtCore.Qt.Key.Key_S and modifier == QtCore.Qt.KeyboardModifier.ShiftModifier and not self.locked
            ):
                self.block_zoom_signal = True
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
                                        self.signal_by_uuid(uuid)[0].min
                                        for uuid in self.common_axis_items
                                        if uuid in uuids_set
                                        and len(self.signal_by_uuid(uuid)[0].plot_samples)
                                        and self.signal_by_uuid(uuid)[0].enable
                                    ]
                                )
                                max_ = np.nanmax(
                                    [
                                        self.signal_by_uuid(uuid)[0].max
                                        for uuid in self.common_axis_items
                                        if uuid in uuids_set
                                        and len(self.signal_by_uuid(uuid)[0].plot_samples)
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

                            if signal.uuid == self.current_uuid:
                                self.viewbox.setYRange(min_, max_, padding=0)

                            position += 1

                else:
                    xrange, _ = self.viewbox.viewRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.setXRange(*xrange, padding=0)
                    self.viewbox.disableAutoRange()

                self.block_zoom_signal = False
                self.zoom_changed.emit(False)

                self.update()

            elif key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right) and modifier in (
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.KeyboardModifier.ControlModifier,
            ):
                if self.region is None:
                    if modifier == QtCore.Qt.KeyboardModifier.ControlModifier:
                        increment = 20
                    else:
                        increment = 1

                    prev_pos = pos = self.cursor1.value()
                    x = self.get_current_timebase()
                    dim = x.size
                    if dim:
                        pos = np.searchsorted(x, pos)
                        if key == QtCore.Qt.Key.Key_Right:
                            pos += increment
                        else:
                            pos -= increment
                        pos = np.clip(pos, 0, dim - increment)
                        pos = x[pos]
                    else:
                        if key == QtCore.Qt.Key.Key_Right:
                            pos += increment
                        else:
                            pos -= increment

                    (left_side, right_side), _ = self.viewbox.viewRange()

                    if pos >= right_side:
                        delta = abs(pos - prev_pos)
                        self.viewbox.setXRange(left_side + delta, right_side + delta, padding=0)
                    elif pos <= left_side:
                        delta = abs(pos - prev_pos)
                        self.viewbox.setXRange(left_side - delta, right_side - delta, padding=0)

                    self.cursor1.set_value(pos)

                else:
                    increment = 1
                    start, stop = self.region.getRegion()

                    if self.region_lock is None:
                        if modifier == QtCore.Qt.KeyboardModifier.ControlModifier:
                            pos = stop
                            second_pos = start
                        else:
                            pos = start
                            second_pos = stop
                    else:
                        if start != stop:
                            pos = start if stop == self.region_lock else stop
                        else:
                            pos = self.region_lock

                    x = self.get_current_timebase()
                    dim = x.size
                    if dim:
                        pos = np.searchsorted(x, pos)
                        if key == QtCore.Qt.Key.Key_Right:
                            pos += increment
                        else:
                            pos -= increment
                        pos = np.clip(pos, 0, dim - increment)
                        pos = x[pos]
                    else:
                        if key == QtCore.Qt.Key.Key_Right:
                            pos += increment
                        else:
                            pos -= increment

                    (left_side, right_side), _ = self.viewbox.viewRange()

                    if pos >= right_side:
                        self.viewbox.setXRange(left_side, pos, padding=0)
                    elif pos <= left_side:
                        self.viewbox.setXRange(pos, right_side, padding=0)

                    if self.region_lock is not None:
                        self.region.setRegion((self.region_lock, pos))
                    else:
                        self.region.setRegion(tuple(sorted((second_pos, pos))))

            elif (
                key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right)
                and modifier == QtCore.Qt.KeyboardModifier.ShiftModifier
            ):
                parent = self.parent().parent()
                uuids = list(
                    {
                        item.uuid
                        for item in parent.channel_selection.selectedItems()
                        if item.type() == ChannelsTreeItem.Channel
                    }
                )

                if not uuids:
                    return

                start, stop = self.viewbox.viewRange()[0]

                offset = (stop - start) / 100

                if key == QtCore.Qt.Key.Key_Left:
                    offset = -offset

                self.set_time_offset([False, offset, *uuids])

            elif (
                key
                in (
                    QtCore.Qt.Key.Key_Up,
                    QtCore.Qt.Key.Key_Down,
                    QtCore.Qt.Key.Key_PageUp,
                    QtCore.Qt.Key.Key_PageDown,
                )
                and modifier == QtCore.Qt.KeyboardModifier.ShiftModifier
            ):
                parent = self.parent().parent()
                uuids = list(
                    {
                        item.uuid
                        for item in parent.channel_selection.selectedItems()
                        if item.type() == ChannelsTreeItem.Channel
                    }
                )

                if not uuids:
                    return

                factor = 10 if key in (QtCore.Qt.Key.Key_PageUp, QtCore.Qt.Key.Key_PageDown) else 100

                for uuid in uuids:
                    signal, index = self.signal_by_uuid(uuid)

                    bottom, top = signal.y_range
                    step = (top - bottom) / factor

                    if key in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_PageUp):
                        step = -step

                    signal.y_range = bottom + step, top + step

                self.zoom_changed.emit(False)

                self.update()

            elif key == QtCore.Qt.Key.Key_H and modifier == QtCore.Qt.KeyboardModifier.NoModifier:
                start_ts, stop_ts = self.viewbox.viewRange()[0]

                if len(self.all_timebase):
                    min_start_ts = np.amin(self.all_timebase)
                    max_stop_ts = np.amax(self.all_timebase)
                else:
                    min_start_ts = start_ts
                    max_stop_ts = stop_ts

                rect = self.viewbox.sceneBoundingRect()
                width = rect.width() - 5

                dpi = QtWidgets.QApplication.primaryScreen().physicalDotsPerInchX()
                dpc = dpi / 2.54

                physical_viewbox_witdh = width / dpc  # cm
                time_width = physical_viewbox_witdh * HONEYWELL_SECONDS_PER_CM

                if self.cursor1.isVisible():
                    mid = self.cursor1.value()
                else:
                    mid = self.region.getRegion()[0]

                if mid - time_width / 2 < min_start_ts:
                    start_ts = min_start_ts
                    stop_ts = min_start_ts + time_width
                elif mid + time_width / 2 > max_stop_ts:
                    start_ts = max_stop_ts - time_width
                    stop_ts = max_stop_ts
                else:
                    start_ts = mid - time_width / 2
                    stop_ts = mid + time_width / 2

                self.viewbox.setXRange(start_ts, stop_ts, padding=0)

                if self.cursor1:
                    self.cursor_moved.emit(self.cursor1)

            elif key == QtCore.Qt.Key.Key_W and modifier == QtCore.Qt.KeyboardModifier.NoModifier:
                if len(self.all_timebase):
                    start_ts = np.amin(self.all_timebase)
                    stop_ts = np.amax(self.all_timebase)

                    self.viewbox.setXRange(start_ts, stop_ts)

                    if self.cursor1:
                        self.cursor_moved.emit(self.cursor1)

            elif key == QtCore.Qt.Key.Key_Insert and modifier == QtCore.Qt.KeyboardModifier.NoModifier:
                self.insert_computation()

            else:
                handled = False

            if not handled:
                event.ignore()
                self.parent().keyPressEvent(event)
            else:
                event.accept()

    def open_scale_editor(self, uuid):
        uuid = uuid or self.current_uuid

        if uuid is None:
            return

        signal, idx = self.signal_by_uuid(uuid)
        signals = {signal.name: signal}

        diag = ScaleDialog(signals, signal.y_range, parent=self)

        if diag.exec():
            offset = diag.offset.value()
            scale = diag.scaling.value()

            y_bottom = -offset * scale / 100
            y_top = y_bottom + scale

            signal.y_range = y_bottom, y_top

            self.zoom_changed.emit(False)
            self.update()

    def paintEvent(self, ev):
        if not self._can_paint or not self._can_paint_global:
            return

        event_rect = ev.rect()

        super().paintEvent(ev)

        if self._pixmap is None:
            ratio = self.devicePixelRatio()

            _pixmap = QtGui.QPixmap(int(event_rect.width() * ratio), int(event_rect.height() * ratio))
            # _pixmap.fill(self.backgroundBrush().color())
            _pixmap.fill(QtCore.Qt.transparent)

            paint = QtGui.QPainter()
            paint.begin(_pixmap)
            paint.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
            paint.setRenderHints(paint.RenderHint.Antialiasing, False)

            self.x_range, self.y_range = self.viewbox.viewRange()

            rect = self.viewbox.sceneBoundingRect()
            rect.setSize(rect.size() * ratio)
            rect.moveTo(rect.topLeft() * ratio)

            self.px = (self.x_range[1] - self.x_range[0]) / rect.width()
            self.py = rect.height()

            with_dots = self.with_dots

            delta = rect.x()
            x_start = self.x_range[0]

            no_brush = QtGui.QBrush()
            pen_width = self._settings.value("line_width", 1, type=int)
            dots_with = self._settings.value("dots_width", 4, type=int)

            paint.resetTransform()
            paint.translate(0, 0)
            paint.setBrush(no_brush)

            flash_current_signal = self.flash_current_signal

            if self._settings.value("curve_dots_cap_style", "square") == "square":
                cap_style = QtCore.Qt.PenCapStyle.SquareCap
            else:
                cap_style = QtCore.Qt.PenCapStyle.RoundCap

            curve = self._curve
            step_mode = self.line_interconnect
            default_connect = curve.opts["connect"]

            for i, sig in enumerate(self.signals):
                if (
                    not sig.enable
                    or flash_current_signal > 0
                    and flash_current_signal % 2 == 0
                    and sig.uuid == self.current_uuid
                ):
                    continue

                y = sig.plot_samples
                x = sig.plot_timestamps

                if len(x):
                    x, y = self.scale_curve_to_pixmap(x, y, y_range=sig.y_range, x_start=x_start, delta=delta)

                    sig.pen.setWidth(pen_width)

                    paint.setPen(sig.pen)

                    pth = self.generatePath(x, y, sig)
                    paint.drawPath(pth)

                    if with_dots:
                        paint.setRenderHints(paint.RenderHint.Antialiasing, True)
                        pos = np.isfinite(y)
                        y = y[pos]
                        x = x[pos]

                        _pen = fn.mkPen(sig.color.name())
                        _pen.setWidth(dots_with)
                        _pen.setCapStyle(cap_style)
                        paint.setPen(_pen)

                        poly, arr = polygon_and_ndarray(x.size)

                        arr[:, 0] = x
                        arr[:, 1] = y
                        paint.drawPoints(poly)
                        paint.setRenderHints(paint.RenderHint.Antialiasing, False)

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

                            if step_mode == "right":
                                idx_with_edges = get_idx_with_edges(idx)
                                _connect = "finite"
                                y = sig.plot_samples.astype("f8")
                                y[~idx_with_edges] = np.inf
                                x = sig.plot_timestamps

                            elif step_mode == "":
                                last = idx[-1]
                                idx_with_edges = np.roll(idx, -1)
                                idx_with_edges[-1] = last
                                _connect = idx_with_edges.view("u1")
                                y = sig.plot_samples.astype("f8")
                                x = sig.plot_timestamps

                            if step_mode == "left":
                                idx_with_edges = np.repeat(get_idx_with_edges(idx), 2)
                                idx_with_edges = np.insert(idx_with_edges, 0, idx_with_edges[0])
                                _connect = idx_with_edges[:-1].view("u1")
                                y = sig.plot_samples.astype("f8")
                                x = sig.plot_timestamps

                            curve.opts["connect"] = _connect

                            x, y = self.scale_curve_to_pixmap(x, y, y_range=sig.y_range, x_start=x_start, delta=delta)

                            color = range_info["font_color"]
                            pen = fn.mkPen(color.name())
                            pen.setWidth(pen_width)

                            paint.setPen(pen)
                            paint.drawPath(self.generatePath(x, y))

                            curve.opts["connect"] = default_connect

                            if with_dots:
                                y = sig.plot_samples.astype("f8")
                                y[~idx] = np.inf

                                x = sig.plot_timestamps

                                x, y = self.scale_curve_to_pixmap(
                                    x, y, y_range=sig.y_range, x_start=x_start, delta=delta
                                )

                                paint.setRenderHints(paint.RenderHint.Antialiasing, True)
                                pen.setWidth(dots_with)
                                pen.setCapStyle(cap_style)
                                paint.setPen(pen)

                                pos = np.isfinite(y)
                                y = y[pos]
                                x = x[pos]

                                poly, arr = polygon_and_ndarray(x.size)

                                arr[:, 0] = x
                                arr[:, 1] = y
                                paint.drawPoints(poly)
                                paint.setRenderHints(paint.RenderHint.Antialiasing, False)

            paint.end()

            _pixmap.setDevicePixelRatio(self.devicePixelRatio())
        else:
            _pixmap = self._pixmap

        paint = QtGui.QPainter()
        vp = self.viewport()
        paint.begin(vp)
        paint.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        paint.setRenderHint(paint.RenderHint.Antialiasing, False)

        if self.y_axis.picture is None:
            self.y_axis.paint(paint, None, None)
        if self.x_axis.picture is None:
            self.x_axis.paint(paint, None, None)

        ratio = self.devicePixelRatio()

        r = self.y_axis.boundingRect()
        r.setSize(self.y_axis.picture.size())
        r.moveTo(r.topLeft() * ratio)
        paint.drawPixmap(
            self.y_axis.sceneBoundingRect(),
            self.y_axis.picture,
            r,
        )

        r = self.x_axis.boundingRect()
        r.setSize(self.x_axis.picture.size())
        r.moveTo(r.topLeft() * ratio)
        paint.drawPixmap(
            self.x_axis.sceneBoundingRect(),
            self.x_axis.picture,
            r,
        )

        for ax in self.axes:
            if isinstance(ax, FormatedAxis) and ax.isVisible():
                if ax.picture is None:
                    ax.paint(paint, None, None)

                r = ax.boundingRect()
                r.setSize(ax.picture.size())
                r.moveTo(r.topLeft() * ratio)

                paint.drawPixmap(
                    ax.sceneBoundingRect(),
                    ax.picture,
                    r,
                )

        r = self.viewbox.sceneBoundingRect()
        r.setLeft(r.left() + 5)
        r.setSize(r.size() * ratio)
        r.moveTo(r.topLeft() * ratio)

        t = self.viewbox.sceneBoundingRect()
        t.setLeft(t.left() + 5)

        self.auto_clip_rect(paint)
        self.draw_grids(paint, event_rect)

        paint.setClipping(False)
        paint.drawPixmap(t.toRect(), _pixmap, r.toRect())
        paint.setClipping(True)

        if self.zoom is None:
            if self.cursor1 is not None and self.cursor1.isVisible():
                self.cursor1.paint(paint, plot=self, uuid=self.current_uuid)

            if self.region is not None:
                self.region.paint(paint, plot=self, uuid=self.current_uuid)

            for bookmark in self.bookmarks:
                if bookmark.visible:
                    bookmark.paint(paint, plot=self, uuid=self.current_uuid)

        else:
            p1, p2, zoom_mode = self.zoom

            old_px, old_py = self.px, self.py

            rect = self.viewbox.sceneBoundingRect()

            self.px = (self.x_range[1] - self.x_range[0]) / rect.width()
            self.py = rect.height()

            delta = rect.x()
            height = rect.height()
            width = rect.width()

            x1, y1 = self.scale_curve_to_pixmap(
                p1.x(),
                p1.y(),
                y_range=self.viewbox.viewRange()[1],
                x_start=self.viewbox.viewRange()[0][0],
                delta=delta,
            )
            x2, y2 = self.scale_curve_to_pixmap(
                p2.x(),
                p2.y(),
                y_range=self.viewbox.viewRange()[1],
                x_start=self.viewbox.viewRange()[0][0],
                delta=delta,
            )

            rect = None

            if zoom_mode == self.viewbox.X_zoom or zoom_mode in self.viewbox.XY_zoom and self.locked:
                x1, x2 = sorted([x1, x2])
                rect = QtCore.QRectF(
                    x1,
                    0,
                    x2 - x1,
                    height,
                )

            elif zoom_mode == self.viewbox.Y_zoom and not self.locked:
                y1, y2 = sorted([y1, y2])
                rect = QtCore.QRectF(
                    0,
                    y1,
                    width + delta,
                    y2 - y1,
                )
            elif zoom_mode in self.viewbox.XY_zoom:
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])

                rect = QtCore.QRectF(
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                )

            if rect is not None:
                color = fn.mkColor(0x62, 0xB2, 0xE2)
                paint.setPen(fn.mkPen(color))
                color = fn.mkColor(0x62, 0xB2, 0xE2, 50)
                paint.setBrush(fn.mkBrush(color))

                paint.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceAtop)
                paint.drawRect(rect)

            self.px, self.py = old_px, old_py

        paint.end()

        if self._pixmap is None:
            self._pixmap = _pixmap

        if self.cursor1 and self.flash_current_signal > 0:
            self.flash_current_signal -= 1
            self.flash_curve_timer.start(50)

    def range_modified_finished_handler(self, region):
        if self.region_lock is not None:
            for i in range(2):
                if self.region.lines[i].value() == self.region_lock:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.PenStyle.DashDotDotLine)
                else:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.PenStyle.SolidLine)
        self.range_modified_finished.emit(region)

    def range_modified_handler(self, region):
        if self.region_lock is not None:
            for i in range(2):
                if self.region.lines[i].value() == self.region_lock:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.PenStyle.DashDotDotLine)
                else:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.PenStyle.SolidLine)

    def scale_curve_to_pixmap(self, x, y, y_range, x_start, delta):
        if self.py:
            y_low, y_high = y_range

            y_scale = (np.float64(y_high) - np.float64(y_low)) / np.float64(self.py)
            x_scale = self.px

            if y_scale * x_scale:
                all_bad = False
            else:
                all_bad = True
        else:
            all_bad = True

        if all_bad:
            try:
                y = np.full(len(y), np.inf)
            except:
                y = np.inf
        else:
            # xs = x_start
            # ys = y_high
            # x = (x - xs) / x_scale + delta
            # y = (ys - y) / y_scale + 1
            # is rewriten as

            xs = x_start - delta * x_scale
            ys = y_high + y_scale

            x = (x - xs) / x_scale
            y = (ys - y) / y_scale

        return x, y

    def select_curve(self, x, y):
        ratio = self.devicePixelRatio()
        rect = self.viewbox.sceneBoundingRect()
        rect.setSize(rect.size() * ratio)
        rect.moveTo(rect.topLeft() * ratio)

        delta = rect.x()
        x_start = self.x_range[0]

        y = y * ratio

        candidates = []
        for sig in self.signals:
            if not sig.enable:
                continue

            val, _1, _2 = sig.value_at_timestamp(x, numeric=True)

            if val == "n.a.":
                continue

            x_val, y_val = self.scale_curve_to_pixmap(x, val, y_range=sig.y_range, x_start=x_start, delta=delta)

            candidates.append((abs(y_val - y), sig.uuid))

        if candidates:
            candidates.sort()
            self.curve_clicked.emit(candidates[0][1])

    def set_color(self, uuid, color):
        sig, index = self.signal_by_uuid(uuid)

        if sig.mode == "raw":
            style = QtCore.Qt.PenStyle.DashLine
        else:
            style = QtCore.Qt.PenStyle.SolidLine

        sig.pen = fn.mkPen(color=color, style=style)

        if sig.individual_axis:
            self.get_axis(index).set_pen(sig.pen)
            self.get_axis(index).setTextPen(sig.pen)

        if uuid == self.current_uuid:
            self.y_axis.set_pen(sig.pen)
            self.y_axis.setTextPen(sig.pen)

        self.update()

    def set_common_axis(self, uuid, state):
        signal, idx = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.CheckState.Checked, True, 1):
            if not self.common_axis_items:
                self.common_axis_y_range = signal.y_range
            else:
                signal.y_range = self.common_axis_y_range
            self.common_axis_items.add(uuid)
        else:
            self.common_axis_items.remove(uuid)

        self.common_axis_label = ", ".join(self.signal_by_uuid(uuid)[0].name for uuid in self.common_axis_items)

        self.set_current_uuid(self.current_uuid, True)
        self.update()

    def set_conversion(self, uuid, conversion):
        sig, index = self.signal_by_uuid(uuid)

        axis = self.axes[index]
        if isinstance(axis, FormatedAxis):
            if sig.conversion and hasattr(sig.conversion, "text_0"):
                axis.text_conversion = sig.conversion
            else:
                axis.text_conversion = None

            axis.picture = None

        if uuid == self.current_uuid:
            axis = self.y_axis
            if sig.conversion and hasattr(sig.conversion, "text_0"):
                axis.text_conversion = sig.conversion
            else:
                axis.text_conversion = None

            axis.picture = None

        sig.trim(*sig.trim_info, force=True)

        self.update()

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

        self.current_uuid_changed.emit(uuid)
        self.update()

    def set_dots(self, with_dots):
        self.with_dots = with_dots
        self.update()

    def set_individual_axis(self, uuid, state):
        sig, index = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.CheckState.Checked, True, 1):
            if sig.enable:
                self.get_axis(index).show()
            sig.individual_axis = True
        else:
            self.get_axis(index).hide()
            sig.individual_axis = False

        self.update()

    def set_line_interconnect(self, line_interconnect):
        self.line_interconnect = line_interconnect

        self._curve.setData(stepMode=line_interconnect)
        for sig in self.signals:
            sig.path = None
        self.update()

    def set_locked(self, locked):
        self.locked = locked
        for axis in self.axes:
            if isinstance(axis, FormatedAxis):
                axis.locked = locked

        self.viewbox.setMouseEnabled(y=not self.locked)

    def set_name(self, uuid, name):
        sig, index = self.signal_by_uuid(uuid)
        sig.name = name or ""

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
        self.update()

    def set_signal_enable(self, uuid, state):
        signal, index = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.CheckState.Checked, True, 1):
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

        for sig in self.signals:
            sig.path = None

        self._enable_timer.start(50)

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

                sig.trim(*sig.trim_info, force=True)
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

                sig.trim(*sig.trim_info, force=True)

        self._compute_all_timebase()

        self.update()

    def set_unit(self, uuid, unit):
        sig, index = self.signal_by_uuid(uuid)
        sig.unit = unit or ""

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

        self.update()

    def set_y_range(self, uuid, y_range, emit=True):
        update = False

        if uuid is None:
            # y axis was changed
            if self.current_uuid is None:
                return
            uuid = self.current_uuid
            sig, idx = self.signal_by_uuid(uuid)
            if sig.y_range != y_range:
                sig.y_range = y_range
                axis = self.axes[idx]
                if isinstance(axis, FormatedAxis):
                    axis.setRange(*y_range)
                update = True
        else:
            sig, idx = self.signal_by_uuid(uuid)
            if sig.y_range != y_range:
                sig.y_range = y_range
                if uuid == self.current_uuid:
                    self.y_axis.setRange(*y_range)
                update = True

        if uuid in self.common_axis_items:
            for uuid in self.common_axis_items:
                sig, idx = self.signal_by_uuid(uuid)
                if sig.y_range != y_range:
                    sig.y_range = y_range
                    axis = self.axes[idx]
                    if isinstance(axis, FormatedAxis):
                        axis.setRange(*y_range)
                    update = True
        if emit:
            self.zoom_changed.emit(False)
        if update:
            self.update()

    def signal_by_name(self, name):
        for i, sig in enumerate(self.signals):
            if sig.name == name:
                return sig, i

        raise Exception(f"Signal not found: {name} {[sig.name for sig in self.signals]}")

    def signal_by_uuid(self, uuid):
        return self._uuid_map[uuid]

    def _signals_enabled_changed_handler(self):
        self._compute_all_timebase()
        if self.cursor1:
            self.cursor_move_finished.emit(self.cursor1)
        self.signals_enable_changed.emit()
        self.update()

    def trim(self, signals=None, force=False, view_range=None):
        signals = signals or self.signals
        if not self._can_trim:
            return

        if view_range is None:
            (start, stop), _ = self.viewbox.viewRange()
        else:
            start, stop = view_range

        width = self.viewbox.sceneBoundingRect().width()

        for sig in signals:
            if sig.enable:
                sig.trim(start, stop, width, force)

    def update(self, *args, pixmap=None, **kwargs):
        self._pixmap = pixmap
        if self.viewbox:
            self.viewbox.update()

    def update_views(self):
        geometry = self.viewbox.sceneBoundingRect()
        if geometry != self.viewbox_geometry:
            self.viewbox_geometry = geometry

    def value_at_cursor(self, uuid=None):
        uuid = uuid or self.current_uuid

        if not uuid:
            y, sig_y_bottom, sig_y_top = "n.a.", 0, 1
        elif self.cursor1:
            if not self.cursor1.isVisible():
                cursor = self.region.lines[0]
            else:
                cursor = self.cursor1

            timestamp = cursor.value()
            sig, idx = self.signal_by_uuid(uuid)
            sig_y_bottom, sig_y_top = sig.y_range
            y, *_ = sig.value_at_timestamp(timestamp, numeric=True)

        else:
            sig, idx = self.signal_by_uuid(uuid)
            sig_y_bottom, sig_y_top = sig.y_range

            if not len(sig):
                y = "n.a."
            else:
                y = sig.phys_samples[-1]

                if y.dtype.kind not in "uif":
                    y = sig.raw_samples[-1]

        return y, sig_y_bottom, sig_y_top

    def xrange_changed_handle(self, *args, force=False):
        if self._can_paint:
            self.trim(force=force)
            self.update()

        self.zoom_changed.emit(False)

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
            self.update()


class CursorInfo(QtWidgets.QLabel):
    def __init__(self, precision, name="t", unit="s", plot=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.name = name
        self.unit = unit
        self.plot = plot

        self.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignTrailing
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)

        if precision == -1:
            self.setToolTip("Cursor information uses maximum precision")
        else:
            self.setToolTip(f"Cursor information precision is set to {self.precision} decimals")

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
        cursor_info_text = ""
        if not self.plot.region:
            if self.plot.cursor1 is not None:
                position = self.plot.cursor1.value()

                fmt = self.plot.x_axis.format
                if fmt == "phys" or not fmt:
                    if self.precision == -1:
                        cursor_info_text = f"{self.name} = {position}{self.unit}"
                    else:
                        template = f"{self.name} = {{:.{self.precision}f}}{self.unit}"
                        cursor_info_text = template.format(position)
                elif fmt == "time":
                    cursor_info_text = f"{self.name} = {timedelta(seconds=position)}"
                elif fmt == "date":
                    position_date = self.plot.x_axis.origin + timedelta(seconds=position)
                    cursor_info_text = f"{self.name} = {position_date}"

                if cursor_info_text:
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
            self.setToolTip("Cursor information uses maximum precision")
        else:
            self.setToolTip(f"Cursor information precision is set to {precision} decimals")
        self.update_value()


try:
    import scipy  # noqa: F401

    from .fft_window import FFTWindow
except ImportError:
    pass

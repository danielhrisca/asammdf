# -*- coding: utf-8 -*-
import bisect
from collections import defaultdict
from datetime import timedelta
from functools import lru_cache, partial, reduce
import os
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from time import perf_counter
from traceback import format_exc
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pyqtgraph as pg
import pyqtgraph.canvas.CanvasTemplate_pyside6
import pyqtgraph.canvas.TransformGuiTemplate_pyside6

try:
    import pyqtgraph.console.template_pyside6
except ImportError:
    import pyqtgraph.console.template_generic

# imports for pyinstaller
import pyqtgraph.functions as fn

try:
    import pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyside6
    import pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyside6
    import pyqtgraph.GraphicsScene.exportDialogTemplate_pyside6
    import pyqtgraph.imageview.ImageViewTemplate_pyside6
except ImportError:
    import pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_generic
    import pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_generic
    import pyqtgraph.GraphicsScene.exportDialogTemplate_generic
    import pyqtgraph.imageview.ImageViewTemplate_generic

from PySide6 import QtCore, QtGui, QtWidgets

PLOT_BUFFER_SIZE = 4000

from ...blocks.conversion_utils import from_dict, to_dict
from ...blocks.utils import target_byte_order
from ..utils import FONT_SIZE, value_as_str

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
from ..utils import COLORS, copy_ranges, extract_mime_names
from .channel_stats import ChannelStats
from .cursor import Cursor, Region
from .dict_to_tree import ComputedChannelInfoWindow
from .formated_axis import FormatedAxis
from .tree import ChannelsTreeItem, ChannelsTreeWidget

bin_ = bin

HERE = Path(__file__).resolve().parent

NOT_FOUND = 0xFFFFFFFF
HONEYWELL_SECONDS_PER_CM = 0.1

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
            user_defined_comment=signal.user_defined_comment,
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
        self._enable = True

        self.format = getattr(signal, "format", "phys")

        self.individual_axis = False
        self.computed = signal.computed
        self.computation = signal.computation

        self.y_link = False
        self.y_range = (0, -1)
        self.home = (0, -1)

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

        self.text_conversion = None

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
            color = signal.color or COLORS[index % 10]
        else:
            color = COLORS[index % 10]
        self.color = fn.mkColor(color)
        self.color_name = self.color.name()
        self.pen = fn.mkPen(color=color, style=QtCore.Qt.SolidLine)

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

        self.mode = getattr(signal, "mode", "phys")
        self.trim(*(trim_info or (None, None, 1900)))

    @property
    def avg(self):
        if not self._stats_available:
            self._compute_stats()
        return self._avg if self._mode == "phys" else self._avg_raw

    @avg.setter
    def avg(self, avg):
        self._avg = avg

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
        cut_sig.computed = self.computed
        cut_sig.color = self.color
        cut_sig.computation = self.computation
        cut_sig.precision = self.precision
        cut_sig.mdf_uuif = self.origin_uuid

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

    def get_stats(self, cursor=None, region=None, view_region=None):
        stats = {}
        sig = self
        x = sig.timestamps
        size = len(x)

        float_fmt = f"{{:.{self.precision}f}}"
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

                    stats["cursor_value"] = value_as_str(
                        value, format, self.plot_samples.dtype, self.precision
                    )

                else:
                    stats["cursor_t"] = ""
                    stats["cursor_value"] = ""

                if region:
                    start, stop = region
                    stats["selected_start"] = start
                    stats["selected_stop"] = stop
                    stats["selected_delta_t"] = stop - start

                    value, kind, _ = self.value_at_timestamp(start)

                    stats["selected_left"] = value_as_str(
                        value, format, self.plot_samples.dtype, self.precision
                    )

                    value, kind, _ = self.value_at_timestamp(stop)

                    stats["selected_right"] = value_as_str(
                        value, format, self.plot_samples.dtype, self.precision
                    )

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
                    stats["overall_gradient"] = float_fmt.format(
                        (float(sig.samples[-1]) - float(sig.samples[0]))
                        / (sig.timestamps[-1] - sig.timestamps[0])
                    )
                    stats["overall_integral"] = float_fmt.format(
                        np.trapz(sig.samples, sig.timestamps)
                    )

                stats["overall_min"] = value_as_str(
                    self.min, format, self.plot_samples.dtype, self.precision
                )
                stats["overall_max"] = value_as_str(
                    self.max, format, self.plot_samples.dtype, self.precision
                )
                stats["overall_average"] = float_fmt.format(sig.avg)
                stats["overall_rms"] = float_fmt.format(sig.rms)
                stats["overall_std"] = float_fmt.format(sig.std)
                stats["overall_start"] = sig.timestamps[0]
                stats["overall_stop"] = sig.timestamps[-1]
                stats["overall_delta"] = value_as_str(
                    sig.samples[-1] - sig.samples[0],
                    format,
                    self.plot_samples.dtype,
                    self.precision,
                )
                stats["overall_delta_t"] = x[-1] - x[0]
                stats["unit"] = sig.unit
                stats["color"] = sig.color
                stats["name"] = sig.name

                if cursor is not None:
                    position = cursor
                    stats["cursor_t"] = position

                    value, kind, _ = self.value_at_timestamp(position)

                    stats["cursor_value"] = value_as_str(
                        value, format, self.plot_samples.dtype, self.precision
                    )

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

                        new_stats["selected_left"] = value_as_str(
                            samples[0], format, self.plot_samples.dtype, self.precision
                        )

                        new_stats["selected_right"] = value_as_str(
                            samples[-1], format, self.plot_samples.dtype, self.precision
                        )

                        new_stats["selected_min"] = value_as_str(
                            np.nanmin(samples), format, samples.dtype, self.precision
                        )
                        new_stats["selected_max"] = value_as_str(
                            np.nanmax(samples), format, samples.dtype, self.precision
                        )
                        new_stats["selected_average"] = float_fmt.format(
                            np.mean(samples)
                        )
                        new_stats["selected_std"] = float_fmt.format(np.std(samples))
                        new_stats["selected_rms"] = float_fmt.format(
                            np.sqrt(np.mean(np.square(samples)))
                        )
                        if samples.dtype.kind in "ui":
                            new_stats["selected_delta"] = value_as_str(
                                int(samples[-1]) - int(samples[0]),
                                format,
                                samples.dtype,
                                self.precision,
                            )
                        else:
                            new_stats["selected_delta"] = value_as_str(
                                samples[-1] - samples[0],
                                format,
                                samples.dtype,
                                self.precision,
                            )

                        if size == 1:
                            new_stats["selected_gradient"] = 0
                            new_stats["selected_integral"] = 0
                        else:
                            new_stats["selected_gradient"] = float_fmt.format(
                                (float(samples[-1]) - float(samples[0]))
                                / (timestamps[-1] - timestamps[0])
                            )
                            new_stats["selected_integral"] = float_fmt.format(
                                np.trapz(samples, timestamps)
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

                    new_stats["visible_min"] = value_as_str(
                        np.nanmin(samples), format, samples.dtype, self.precision
                    )
                    new_stats["visible_max"] = value_as_str(
                        np.nanmax(samples), format, samples.dtype, self.precision
                    )
                    new_stats["visible_average"] = float_fmt.format(np.mean(samples))
                    new_stats["visible_std"] = float_fmt.format(np.std(samples))
                    new_stats["visible_rms"] = float_fmt.format(
                        np.sqrt(np.mean(np.square(samples)))
                    )
                    if kind in "ui":
                        new_stats["visible_delta"] = value_as_str(
                            int(cut.samples[-1]) - int(cut.samples[0]),
                            format,
                            samples.dtype,
                            self.precision,
                        )
                    else:
                        new_stats["visible_delta"] = value_as_str(
                            cut.samples[-1] - cut.samples[0],
                            format,
                            samples.dtype,
                            self.precision,
                        )

                    if size == 1:
                        new_stats["visible_gradient"] = 0
                        new_stats["visible_integral"] = 0
                    else:
                        new_stats["visible_gradient"] = float_fmt.format(
                            (float(samples[-1]) - float(samples[0]))
                            / (timestamps[-1] - timestamps[0])
                        )
                        new_stats["visible_integral"] = float_fmt.format(
                            np.trapz(samples, timestamps)
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
        self.pen = fn.mkPen(color=color, style=QtCore.Qt.SolidLine)

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
                    value = value.decode("latin-1", errors="replace").strip(
                        " \r\n\t\v\0"
                    )

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

    item_double_click_handling = "enable/disable"

    def __init__(
        self,
        signals,
        with_dots=False,
        origin=None,
        mdf=None,
        line_interconnect="line",
        line_width=1,
        hide_missing_channels=False,
        hide_disabled_channels=False,
        x_axis="time",
        allow_cursor=True,
        show_cursor_circle=True,
        show_cursor_horizontal_line=True,
        cursor_line_width=1,
        cursor_color="#ffffff",
        region_values_display_mode="delta",
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

        self.x_name = "t" if x_axis == "time" else "f"
        self.x_unit = "s" if x_axis == "time" else "Hz"

        self.info_uuid = None

        self._can_switch_mode = True
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
        icon.addPixmap(QtGui.QPixmap(":/home.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        menu.addAction(
            icon,
            "Home",
            lambda: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress,
                    QtCore.Qt.Key_W,
                    QtCore.Qt.NoModifier,
                )
            ),
        )

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/axis.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        menu.addAction(
            icon,
            "Honeywell",
            lambda: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress,
                    QtCore.Qt.Key_H,
                    QtCore.Qt.NoModifier,
                )
            ),
        )

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/fit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        menu.addAction(
            icon,
            "Fit",
            lambda: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress,
                    QtCore.Qt.Key_F,
                    QtCore.Qt.NoModifier,
                )
            ),
        )

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/stack.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        menu.addAction(
            icon,
            "Stack",
            lambda: self.plot.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress,
                    QtCore.Qt.Key_S,
                    QtCore.Qt.NoModifier,
                )
            ),
        )

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/increase-font.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        menu.addAction(icon, "Increase font", self.increase_font)

        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/decrease-font.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        menu.addAction(icon, "Decrease font", self.decrease_font)

        btn.setMenu(menu)
        hbox.addWidget(btn)
        btn.menu()

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

        self.hide_axes_btn = btn = QtWidgets.QPushButton("")
        self.hide_axes_btn.clicked.connect(self.hide_axes)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/axis.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        btn.setIcon(icon)
        btn.setToolTip("Hide axis")
        hbox.addWidget(self.hide_axes_btn)

        self.selected_channel_value_btn = btn = QtWidgets.QPushButton("")
        self.selected_channel_value_btn.clicked.connect(
            self.hide_selected_channel_value
        )
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/number.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        btn.setIcon(icon)
        btn.setToolTip("Hide axis")
        hbox.addWidget(self.selected_channel_value_btn)

        self.focused_mode_btn = btn = QtWidgets.QPushButton("")
        self.focused_mode_btn.clicked.connect(self.toggle_focused_mode)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(":/focus.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        btn.setIcon(icon)
        btn.setToolTip("Toggle focused mode")
        hbox.addWidget(self.focused_mode_btn)

        self.delta_btn = btn = QtWidgets.QPushButton("")
        self.delta_btn.clicked.connect(self.toggle_region_values_display_mode)
        icon = QtGui.QIcon()
        pix = QtGui.QPixmap(64, 64)
        color = QtGui.QColor("#000000")
        color.setAlpha(0)
        pix.fill(color)
        painter = QtGui.QPainter(pix)
        font = painter.font()
        font.setPointSize(48)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#61b2e2"))
        painter.drawText(QtCore.QPoint(12, 52), "Δ")
        painter.end()
        icon.addPixmap(pix, QtGui.QIcon.Normal, QtGui.QIcon.Off)
        btn.setIcon(icon)
        btn.setToolTip("Toggle region values display mode")
        hbox.addWidget(self.delta_btn)

        hbox.addStretch()

        self.selected_channel_value = QtWidgets.QLabel("")
        self.selected_channel_value.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        self.selected_channel_value.setAutoFillBackground(True)
        font = self.selected_channel_value.font()
        font.setBold(True)
        font.setPointSize(24)
        self.selected_channel_value.setFont(font)

        vbox.addWidget(self.selected_channel_value)

        vbox.addWidget(self.channel_selection)
        vbox.addWidget(self.cursor_info)

        self.range_proxy = pg.SignalProxy(
            self.plot.range_modified, rateLimit=16, slot=self.range_modified
        )
        # self.plot.range_modified.connect(self.range_modified)
        self.plot.range_removed.connect(self.range_removed)
        self.plot.range_modified_finished.connect(self.range_modified_finished)
        self.plot.cursor_removed.connect(self.cursor_removed)
        self.plot.current_uuid_changed.connect(self.current_uuid_changed)

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
        self._visible_entries = set()
        self._visible_items = {}
        self._item_cache = {}

        self._prev_region = None

        self.splitter.addWidget(self.plot)

        self.info = ChannelStats(self.x_unit)
        self.info.hide()
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

        self.channel_selection.setColumnWidth(
            self.channel_selection.NameColumn, nameColumnWidth
        )
        self.channel_selection.setColumnWidth(self.channel_selection.ValueColumn, 83)
        self.channel_selection.setColumnWidth(
            self.channel_selection.UnitColumn, unitColumnWidth
        )
        self.channel_selection.setColumnWidth(
            self.channel_selection.CommonAxisColumn, 35
        )
        self.channel_selection.setColumnWidth(
            self.channel_selection.IndividualAxisColumn, 35
        )
        self.hide()

        if signals:
            self.add_new_channels(signals)

        self.channel_selection.color_changed.connect(self.plot.set_color)
        self.channel_selection.unit_changed.connect(self.plot.set_unit)
        self.channel_selection.name_changed.connect(self.plot.set_name)
        self.channel_selection.conversion_changed.connect(self.plot.set_conversion)

        self.channel_selection.itemsDeleted.connect(self.channel_selection_reduced)
        self.channel_selection.group_activation_changed.connect(self.plot.update)
        self.channel_selection.currentItemChanged.connect(
            self.channel_selection_row_changed
        )
        self.channel_selection.itemSelectionChanged.connect(
            self.channel_selection_changed
        )
        self.channel_selection.add_channels_request.connect(self.add_channels_request)
        self.channel_selection.set_time_offset.connect(self.plot.set_time_offset)
        self.channel_selection.show_properties.connect(self._show_properties)
        self.channel_selection.insert_computation.connect(self.plot.insert_computation)
        self.channel_selection.edit_computation.connect(self.plot.edit_computation)

        self.channel_selection.model().dataChanged.connect(
            self.channel_selection_item_changed
        )

        self.channel_selection.visible_items_changed.connect(
            self._update_visibile_entries
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
                    (QtCore.Qt.Key_2, int(QtCore.Qt.NoModifier)),
                ]
            )
            | self.plot.keyboard_events
        )

        self.splitter.splitterMoved.connect(self.set_splitter)
        self.line_width = line_width

        self.hide_selected_channel_value(hide=False)
        self.toggle_focused_mode(focused=False)
        self.toggle_region_values_display_mode(mode="value")

        self.show()

    def add_new_channels(self, channels, mime_data=None, destination=None, update=True):
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
                    range_info["background_color"] = fn.mkColor(
                        range_info["background_color"]
                    )

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

                    groups.extend(
                        add_new_items(tree, item, info["channels"], items_pool)
                    )
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

        channels = self.plot.add_new_channels(channels, descriptions=descriptions)

        enforce_y_axis = False
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while True:
            item = iterator.value()
            if item is None:
                break

            if item.type() == item.Channel:
                if item.checkState(item.CommonAxisColumn) == QtCore.Qt.Unchecked:
                    enforce_y_axis = False
                    break
                else:
                    enforce_y_axis = True

            iterator += 1

        children = []

        background_color = self.palette().color(QtGui.QPalette.Base)

        new_items = {}

        items_map = {}
        for sig_uuid, sig in channels.items():

            description = descriptions.get(sig_uuid, {})

            if description:
                sig.format = description.get("format", "phys")
            sig.mode = description.get("mode", "phys")

            if "comment" in description:
                sig.comment = description["comment"]
                sig.user_defined_comment = True

            item = ChannelsTreeItem(
                ChannelsTreeItem.Channel,
                signal=sig,
                check=QtCore.Qt.Checked if sig.enable else QtCore.Qt.Unchecked,
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
                            QtCore.Qt.Checked
                            if info["enabled"]
                            else QtCore.Qt.Unchecked,
                        )
                    else:
                        if not item.childCount():
                            item.setCheckState(
                                item.NameColumn,
                                QtCore.Qt.Checked
                                if info["enabled"]
                                else QtCore.Qt.Unchecked,
                            )

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
                    parent = (
                        destination.parent()
                        or self.channel_selection.invisibleRootItem()
                    )
                    index = parent.indexOfChild(destination)
                    parent.insertChildren(index, children)

        for sig_uuid, sig in channels.items():
            description = descriptions.get(sig_uuid, {})
            item = items_map[sig_uuid]
            if description:
                individual_axis = description.get("individual_axis", False)
                if individual_axis:
                    item.setCheckState(item.IndividualAxisColumn, QtCore.Qt.Checked)

                    _, idx = self.plot.signal_by_uuid(sig_uuid)
                    axis = self.plot.get_axis(idx)
                    if isinstance(axis, FormatedAxis):
                        axis.setWidth(description["individual_axis_width"])

                if description.get("common_axis", False):
                    item.setCheckState(item.CommonAxisColumn, QtCore.Qt.Checked)

                item.precision = description.get("precision", 3)

                if description.get("conversion", None):
                    conversion = from_dict(description["conversion"])
                    conversion.is_user_defined = True
                    item.set_conversion(conversion)

            if enforce_y_axis:
                item.setCheckState(item.CommonAxisColumn, QtCore.Qt.Checked)

        if update:
            self.channel_selection.update_channel_groups_count()
            self.channel_selection.refresh()

        self.adjust_splitter()

        self.current_uuid_changed(self.plot.current_uuid)
        self.plot._can_paint = True
        self.plot.update()

    def adjust_splitter(self):
        size = sum(self.splitter.sizes())

        self.channel_selection.resizeColumnToContents(self.channel_selection.NameColumn)
        self.channel_selection.resizeColumnToContents(self.channel_selection.UnitColumn)

        width = sum(
            self.channel_selection.columnWidth(col)
            for col in range(self.channel_selection.columnCount())
        )

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
            "enabled": item.checkState(item.NameColumn) == QtCore.Qt.Checked,
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

        channel["name"] = sig.name
        channel["unit"] = sig.unit
        channel["enabled"] = item.checkState(item.NameColumn) == QtCore.Qt.Checked

        if item.checkState(item.IndividualAxisColumn) == QtCore.Qt.Checked:
            channel["individual_axis"] = True
            channel["individual_axis_width"] = (
                self.plot.get_axis(idx).boundingRect().width()
            )
        else:
            channel["individual_axis"] = False

        channel["common_axis"] = (
            item.checkState(item.CommonAxisColumn) == QtCore.Qt.Checked
        )
        channel["color"] = sig.color.name()
        channel["computed"] = sig.computed
        channel["ranges"] = copy_ranges(widget.ranges)

        for range_info in channel["ranges"]:
            range_info["background_color"] = range_info["background_color"].name()
            range_info["font_color"] = range_info["font_color"].name()

        channel["precision"] = widget.precision
        channel["fmt"] = widget.fmt
        channel["format"] = widget.format
        channel["mode"] = widget.mode
        if sig.computed:
            channel["computation"] = sig.computation

        if sig.user_defined_comment:
            channel["comment"] = sig.comment

        channel["y_range"] = [float(e) for e in sig.y_range]
        channel["origin_uuid"] = str(sig.origin_uuid)

        if sig.conversion and sig.conversion.is_user_defined:
            channel["conversion"] = to_dict(sig.conversion)

        return channel

    def channel_selection_changed(self, update=False):
        if self.focused_mode:
            for signal in self.plot.signals:
                signal.enable = False
            for item in self.channel_selection.selectedItems():
                if item.type() == item.Channel:
                    item.signal.enable = True
            self.plot.update()
        else:
            if update:
                for signal in self.plot.signals:
                    signal.enable = False

                iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
                while True:
                    item = iterator.value()
                    if item is None:
                        break

                    if (
                        item.type() == item.Channel
                        and item.checkState(item.NameColumn) == QtCore.Qt.Checked
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
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)

            brush = QtGui.QBrush(item.background(item.NameColumn))
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)

            self.selected_channel_value.setPalette(palette)

            value = item.text(item.ValueColumn)
            unit = item.unit
            self.selected_channel_value.setText(f"{value} {unit}")

        if QtCore.Qt.CheckStateRole not in roles:
            return

        if item.type() != item.Channel or item.isDisabled():
            return

        column = top_left.column()

        if column == item.NameColumn:
            enabled = item.checkState(column) == QtCore.Qt.Checked
            if enabled != item.signal.enable:
                item.signal.enable = enabled
                self.plot.set_signal_enable(item.uuid, item.checkState(column))

        elif column == item.CommonAxisColumn:
            if not self.locked:
                enabled = item.checkState(column) == QtCore.Qt.Checked
                if enabled != item.signal.y_link:
                    item.signal.y_link = enabled
                    self.plot.set_common_axis(item.uuid, enabled)

        elif column == item.IndividualAxisColumn:
            enabled = item.checkState(column) == QtCore.Qt.Checked
            if enabled != item.signal.individual_axis:
                self.plot.set_individual_axis(item.uuid, enabled)

    def channel_selection_item_double_clicked(self, item, column):
        if item is None:
            return

        elif item.type() != item.Info and column not in (
            item.CommonAxisColumn,
            item.IndividualAxisColumn,
        ):
            if item.type() == item.Channel:
                if not item.isDisabled():
                    if item.checkState(item.NameColumn) == QtCore.Qt.Checked:
                        item.setCheckState(item.NameColumn, QtCore.Qt.Unchecked)
                    else:
                        item.setCheckState(item.NameColumn, QtCore.Qt.Checked)
            elif item.type() == item.Group:

                if Plot.item_double_click_handling == "enable/disable":
                    if self.channel_selection.expandsOnDoubleClick():
                        self.channel_selection.setExpandsOnDoubleClick(False)
                    if item.isDisabled():
                        item.set_disabled(False)
                        item.setIcon(item.NameColumn, QtGui.QIcon(":/open.png"))
                    else:
                        item.set_disabled(True)
                        item.setIcon(item.NameColumn, QtGui.QIcon(":/erase.png"))
                    self.plot.update()
                elif Plot.item_double_click_handling == "expand/collapse":
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

    def clear(self):
        event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier
        )
        self.channel_selection.keyPressEvent(event)
        event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_Delete, QtCore.Qt.NoModifier
        )
        self.channel_selection.keyPressEvent(event)

    def close(self):
        self.closed = True

        self.channel_selection.blockSignals(True)
        self.plot.blockSignals(True)

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
            brush.setStyle(QtCore.Qt.SolidPattern)
            palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
            self.selected_channel_value.setPalette(palette)

            item = self.item_by_uuid(uuid)
            if item is not None:
                value = item.text(item.ValueColumn)
                unit = item.unit
                self.selected_channel_value.setText(f"{value} {unit}")
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
                        self.selected_channel_value.setText(f"{value} {unit}")

        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

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
        while True:
            item = iterator.value()
            if item is None:
                break

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
        while True:
            item = iterator.value()
            if item is None:
                break

            if item.type() == item.Channel and item.uuid == uuid:
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

    def hide_axes(self, event=None, hide=None):
        if hide is None:
            hide = not self.hide_axes_btn.isFlat()

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

    def hide_selected_channel_value(self, event=None, hide=None):
        if hide is None:
            hide = not self.selected_channel_value_btn.isFlat()

        if hide:
            self.selected_channel_value.hide()
            self.selected_channel_value_btn.setFlat(True)
            self.selected_channel_value_btn.setToolTip(
                "Show selected channel value panel"
            )
        else:
            self.selected_channel_value.show()
            self.selected_channel_value_btn.setFlat(False)
            self.selected_channel_value_btn.setToolTip(
                "Hide selected channel value panel"
            )

    def increase_font(self):
        font = self.font()
        size = font.pointSize()
        pos = bisect.bisect_right(FONT_SIZE, size)
        if pos == len(FONT_SIZE):
            pos -= 1
        new_size = FONT_SIZE[pos]

        self.set_font_size(new_size)

    def item_by_uuid(self, uuid):
        return self._item_cache.get(uuid, None)

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

                stats = self.plot.get_stats(self.info_uuid)
                self.info.set_stats(stats)

        elif key == QtCore.Qt.Key_2 and modifiers == QtCore.Qt.NoModifier:
            self.focused_mode = not self.focused_mode
            if self.focused_mode:
                self.focused_mode_btn.setFlat(False)
            else:
                self.focused_mode_btn.setFlat(True)
            self.channel_selection_changed(update=True)

        elif (
            key in (QtCore.Qt.Key_B, QtCore.Qt.Key_H, QtCore.Qt.Key_P, QtCore.Qt.Key_T)
            and modifiers == QtCore.Qt.ControlModifier
        ):

            selected_items = (
                self.channel_selection.selectedItems()
                or self.channel_selection.invisibleRootItem()
            )

            if key == QtCore.Qt.Key_B:
                fmt = "bin"
            elif key == QtCore.Qt.Key_H:
                fmt = "hex"
            elif key == QtCore.Qt.Key_P:
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

        elif (
            key in (QtCore.Qt.Key_R, QtCore.Qt.Key_S)
            and modifiers == QtCore.Qt.AltModifier
            and self._can_switch_mode
        ):

            selected_items = self.channel_selection.selectedItems()
            if not selected_items:
                signals = [(sig, i) for i, sig in enumerate(self.plot.signals)]
                uuids = [sig.uuid for sig in self.plot.signals]

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

                        signal.y_range = buttom, top
                        item = self.item_by_uuid(signal.uuid)
                        item._value = None

                        if self.plot.current_uuid == signal.uuid:
                            self.plot.y_axis.mode = mode
                            self.plot.y_axis.picture = None
                            self.plot.y_axis.update()
                            self.plot.viewbox.setYRange(
                                buttom, top, padding=0, update=True
                            )

                        self.plot.get_axis(idx).mode = mode
                        self.plot.get_axis(idx).picture = None
                        self.plot.get_axis(idx).update()

            for uuid in uuids:
                item = self.item_by_uuid(uuid)
                item.setText(item.UnitColumn, item.unit)

            self.plot.update()

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
                    offset = diag.offset.value()
                    scale = diag.scaling.value()

                    y_bottom = -offset * scale / 100
                    y_top = y_bottom + scale

                    y_range = y_bottom, y_top

                    for idx in indexes:
                        self.plot.signals[idx].y_range = y_range

                    self.plot.update()

        elif key == QtCore.Qt.Key_R and modifiers == QtCore.Qt.NoModifier:
            iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
            while True:
                item = iterator.value()
                if item is None:
                    break

                if item.type() == item.Channel:
                    item.set_prefix()
                    item.set_value("")

                iterator += 1

            self.plot.keyPressEvent(event)

        elif key == QtCore.Qt.Key_C and modifiers in (
            QtCore.Qt.NoModifier,
            QtCore.Qt.ControlModifier,
            QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier,
        ):
            self.channel_selection.keyPressEvent(event)

        elif (
            key == QtCore.Qt.Key_R
            and modifiers == QtCore.Qt.ControlModifier
            and self.can_edit_ranges
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

    @property
    def line_width(self):
        return self.plot.line_width

    @line_width.setter
    def line_width(self, value):
        self.plot.line_width = value
        self.plot.dot_width = value + 4
        self.plot.update()

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

                start_v, kind, fmt = signal.value_at_timestamp(start)
                stop_v, kind, fmt = signal.value_at_timestamp(stop)

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
        while True:
            item = iterator.value()
            if item is None:
                break

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

    def set_font_size(self, size):
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)
        self.channel_selection.set_font_size(size)
        self.plot.y_axis.set_font_size(size)
        self.plot.x_axis.set_font_size(size)

    def set_locked(self, event=None, locked=None):
        if locked is None:
            locked = not self.locked
        if locked:
            tooltip = "The Y axis is locked. Press to unlock"
            png = ":/locked.png"
            self.lock_btn.setFlat(True)
        else:
            tooltip = "The Y axis is unlocked. Press to lock"
            png = ":/unlocked.png"
            self.lock_btn.setFlat(False)

        self.channel_selection.setColumnHidden(
            self.channel_selection.CommonAxisColumn, locked
        )

        self.locked = locked
        self.plot.set_locked(locked)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(png), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.lock_btn.setToolTip(tooltip)
        self.lock_btn.setIcon(icon)

    def set_splitter(self, pos, index):
        self.splitter_moved.emit(self, pos)

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

    def _show_properties(self, uuid):
        for sig in self.plot.signals:
            if sig.uuid == uuid:
                if sig.computed:
                    try:
                        view = ComputedChannelInfoWindow(sig, self)
                        view.show()
                    except:
                        print(format_exc())
                        raise

                else:
                    self.show_properties.emit(
                        [sig.group_index, sig.channel_index, sig.origin_uuid]
                    )

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
            "hide_axes": not self.plot.y_axis.isVisible(),
            "hide_selected_channel_value_panel": not self.selected_channel_value.isVisible(),
        }

        return config

    def toggle_focused_mode(self, event=None, focused=None):
        if focused is not None:
            # invert so that the key press event will set the desider focused mode
            self.focused_mode = not focused

        event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress, QtCore.Qt.Key_2, QtCore.Qt.NoModifier
        )
        self.keyPressEvent(event)

        if not self.focused_mode:
            self.focused_mode_btn.setFlat(True)
            self.focused_mode_btn.setToolTip("Switch on focused mode")
        else:
            self.focused_mode_btn.setFlat(False)
            self.focused_mode_btn.setToolTip("Switch off focused mode")

    def toggle_region_values_display_mode(self, event=None, mode=None):
        if mode is None:
            self.region_values_display_mode = (
                "delta" if self.region_values_display_mode == "value" else "value"
            )
        else:
            self.region_values_display_mode = mode

        if self.region_values_display_mode == "value":
            self.delta_btn.setFlat(True)
            self.delta_btn.setToolTip("Switch to region cursors delta display mode")
        else:
            self.delta_btn.setFlat(False)
            self.delta_btn.setToolTip(
                "Switch to active region cursor value display mode"
            )

        self.range_modified()

        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while True:
            item = iterator.value()
            if item is None:
                break
            if item.type() == ChannelsTreeItem.Channel:
                item.set_value(update=True, force=True)

            iterator += 1

    def update_current_values(self, *args):
        if self.plot.region:
            self.range_modified(None)
        else:
            self.cursor_moved()

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

                if (
                    item.uuid == self.info_uuid
                    or item.exists
                    and (
                        item.checkState(item.NameColumn) == QtCore.Qt.Checked
                        or item._is_visible
                    )
                ):
                    entry = (item.origin_uuid, item.signal.name, item.uuid)
                    _visible_entries.add(entry)
                    _visible_items[entry] = item

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


class _Plot(pg.PlotWidget):
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

        self.autoFillBackground()

        self.line_width = 1
        self.dot_width = 4

        self._pixmap = None

        self.locked = False

        self.cursor_unit = "s" if x_axis == "time" else "Hz"

        self.line_interconnect = (
            line_interconnect if line_interconnect != "line" else ""
        )

        self._can_trim = True
        self._can_paint = True
        self._can_paint_global = True
        self.mdf = mdf

        self._can_paint = True

        self.setAcceptDrops(True)

        self._last_size = self.geometry()
        self._settings = QtCore.QSettings()

        self.setContentsMargins(5, 5, 5, 5)
        self.xrange_changed.connect(self.xrange_changed_handle)
        self.with_dots = with_dots

        self.info = None
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

        self.x_range = self.y_range = (0, 1)
        self._curve = pg.PlotCurveItem(
            np.array([]),
            np.array([]),
            stepMode=self.line_interconnect,
            skipFiniteCheck=False,
            connect="finite",
        )

        self.viewbox.menu.removeAction(self.viewbox.menu.viewAll)
        self.scene_.contextMenu = []
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

        self.x_axis = FormatedAxis(
            "bottom", maxTickLength=5, background=self.backgroundBrush().color()
        )

        if x_axis == "time":
            fmt = self._settings.value("plot_xaxis")
            if fmt == "seconds":
                fmt = "phys"
        else:
            fmt = "phys"
        self.x_axis.format = fmt
        self.x_axis.origin = origin

        self.y_axis = FormatedAxis(
            "left", maxTickLength=-5, background=self.backgroundBrush().color()
        )
        self.y_axis.setWidth(48)

        self.y_axis.scale_editor_requested.connect(self.open_scale_editor)
        self.y_axis.rangeChanged.connect(self.set_y_range)

        self.plot_item.setAxisItems({"left": self.y_axis, "bottom": self.x_axis})

        def plot_item_wheel_event(event):
            if event is not None:
                x = event.pos().x()

                if x <= self.y_axis.width():
                    self.y_axis.wheelEvent(event)
                else:
                    for axis in self.axes:
                        if isinstance(axis, FormatedAxis) and axis.isVisible():
                            rect = axis.sceneBoundingRect()
                            if rect.x() <= x <= rect.x() + rect.width():
                                axis.wheelEvent(event)
                                break

        def plot_item_mousePressEvent(event):
            if event is not None:
                x = event.pos().x()

                if x <= self.y_axis.width():
                    self.y_axis.mousePressEvent(event)
                else:
                    for axis in self.axes:
                        if isinstance(axis, FormatedAxis) and axis.isVisible():
                            rect = axis.sceneBoundingRect()
                            if rect.x() <= x <= rect.x() + rect.width():
                                axis.mousePressEvent(event)
                                break
                    else:
                        self.plot_item._mousePressEvent(event)

        def plot_item_mouseMoveEvent(event):
            if event is not None:
                x = event.pos().x()

                if x <= self.y_axis.width():
                    self.y_axis.mouseMoveEvent(event)
                else:
                    for axis in self.axes:
                        if isinstance(axis, FormatedAxis) and axis.isVisible():
                            rect = axis.sceneBoundingRect()
                            if rect.x() <= x <= rect.x() + rect.width():
                                axis.mouseMoveEvent(event)
                                break
                    else:
                        self.plot_item._mouseMoveEvent(event)

        def plot_item_mouseReleaseEvent(event):
            if event is not None:
                x = event.pos().x()

                if x <= self.y_axis.width():
                    self.y_axis.mouseReleaseEvent(event)
                else:
                    for axis in self.axes:
                        if isinstance(axis, FormatedAxis) and axis.isVisible():
                            rect = axis.sceneBoundingRect()
                            if rect.x() <= x <= rect.x() + rect.width():
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
                (QtCore.Qt.Key_W, QtCore.Qt.NoModifier),
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

        self.last_click = perf_counter()

    def add_new_channels(self, channels, computed=False, descriptions=None):
        descriptions = descriptions or {}

        initial_index = len(self.signals)
        self._can_paint = False

        for sig in channels.values():
            if not hasattr(sig, "computed"):
                sig.computed = computed
                if not computed:
                    sig.computation = {}

        if initial_index == 0:
            start_t, stop_t = 1, -1
            for sig in channels.values():
                if len(sig):
                    start_t = min(start_t, sig.timestamps[0])
                    stop_t = max(stop_t, sig.timestamps[-1])

            if (start_t, stop_t) != (1, -1):
                self.viewbox.setXRange(start_t, stop_t, update=False)

        (start, stop), _ = self.viewbox.viewRange()

        width = self.viewbox.sceneBoundingRect().width()
        trim_info = start, stop, width

        channels = [
            PlotSignal(sig, i, trim_info=trim_info)
            for i, sig in enumerate(channels.values(), len(self.signals))
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

        now = perf_counter()
        if modifiers == QtCore.Qt.ShiftModifier:
            self.select_curve(x, y)
        elif now - self.last_click < 0.3:
            self.select_curve(x, y)

        self.last_click = perf_counter()

    def close(self):
        super().close()

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
        if self.y_axis.grid or self.x_axis.grid:

            rect = self.viewbox.sceneBoundingRect()
            y_delta = rect.y()
            x_delta = rect.x()

            if self.y_axis.grid:
                for pen, p1, p2 in self.y_axis.tickSpecs:
                    y_pos = p1.y() + y_delta
                    paint.setPen(pen)
                    paint.drawLine(
                        QtCore.QPointF(0, y_pos),
                        QtCore.QPointF(event_rect.x() + event_rect.width(), y_pos),
                    )

            if self.x_axis.grid:
                for pen, p1, p2 in self.x_axis.tickSpecs:
                    x_pos = p1.x() + x_delta
                    paint.setPen(pen)
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

        computed_signals = {
            sig.name: sig
            for sig in self.signals
            if sig.computed and sig.uuid != signal.uuid
        }
        dlg = DefineChannel(
            mdf=self.mdf,
            name=signal.name,
            computation=signal.computation,
            computed_signals=computed_signals,
            parent=self,
        )
        dlg.setModal(True)
        dlg.exec_()
        computed_channel = dlg.result

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
        sig, index = self.signal_by_uuid(uuid)

        return sig.get_stats(
            cursor=self.cursor1.value() if self.cursor1 else None,
            region=self.region.getRegion() if self.region else None,
            view_region=self.viewbox.viewRange()[0],
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

        computed_signals = {sig.name: sig for sig in self.signals if sig.computed}
        dlg = DefineChannel(
            mdf=self.mdf,
            name=name,
            computation=None,
            computed_signals=computed_signals,
            parent=self,
        )
        dlg.setModal(True)
        dlg.exec_()
        computed_channel = dlg.result

        if computed_channel is not None:
            self.add_channels_request.emit([computed_channel])

    def keyPressEvent(self, event):
        key = event.key()
        modifier = event.modifiers()

        if (key, int(modifier)) in self.disabled_keys:
            super().keyPressEvent(event)
        else:
            handled = True
            if key == QtCore.Qt.Key_Y and modifier == QtCore.Qt.NoModifier:
                if self.region is None:
                    event_ = QtGui.QKeyEvent(
                        QtCore.QEvent.KeyPress, QtCore.Qt.Key_R, QtCore.Qt.NoModifier
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

                        if signal.uuid == self.current_uuid:
                            self.viewbox.setYRange(min_, max_, padding=0)

                self.update()

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
                        if signal.uuid == self.current_uuid:
                            self.viewbox.setYRange(min_, max_, padding=0)

                self.update()

            elif key == QtCore.Qt.Key_G and modifier == QtCore.Qt.NoModifier:

                y = self.plotItem.ctrl.yGridCheck.isChecked()
                x = self.plotItem.ctrl.xGridCheck.isChecked()

                if x and y:
                    self.plotItem.showGrid(x=False, y=False)
                elif x:
                    self.plotItem.showGrid(x=True, y=True)
                else:
                    self.plotItem.showGrid(x=True, y=False)

                self.update()

            elif (
                key in (QtCore.Qt.Key_I, QtCore.Qt.Key_O)
                and modifier == QtCore.Qt.NoModifier
            ):

                x_range, _ = self.viewbox.viewRange()
                delta = x_range[1] - x_range[0]
                if key == QtCore.Qt.Key_I:
                    step = -delta * 0.25
                else:
                    step = delta * 0.5
                if self.cursor1:
                    pos = self.cursor1.value()
                    x_range = pos - delta / 2, pos + delta / 2
                self.viewbox.setXRange(x_range[0] - step, x_range[1] + step, padding=0)

            elif key == QtCore.Qt.Key_R and modifier == QtCore.Qt.NoModifier:
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

                self.update()

            elif key == QtCore.Qt.Key_S and modifier == QtCore.Qt.ControlModifier:
                file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Select output measurement file",
                    "",
                    "MDF version 4 files (*.mf4 *.mf4z)",
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

                            file_name = Path(file_name)

                            if file_name.suffix.lower() in (".zip", ".mf4z"):

                                tmpf = Path(gettempdir()) / f"{perf_counter()}.mf4"
                                mdf.save(tmpf, overwrite=True, compression=2)

                                zipped_mf4 = ZipFile(
                                    file_name, "w", compression=ZIP_DEFLATED
                                )
                                zipped_mf4.write(
                                    str(tmpf),
                                    file_name.with_suffix(".mf4").name,
                                    compresslevel=1,
                                )

                                tmpf.unlink()

                            else:
                                mdf.save(file_name, overwrite=True, compression=2)

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

                            if signal.uuid == self.current_uuid:
                                self.viewbox.setYRange(min_, max_, padding=0)

                            position += 1

                else:
                    xrange, _ = self.viewbox.viewRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.setXRange(*xrange, padding=0)
                    self.viewbox.disableAutoRange()

                self.update()

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

                            if signal.uuid == self.current_uuid:
                                self.viewbox.setYRange(min_, max_, padding=0)

                            position += 1

                else:
                    xrange, _ = self.viewbox.viewRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.setXRange(*xrange, padding=0)
                    self.viewbox.disableAutoRange()

                self.update()

            elif key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right) and modifier in (
                QtCore.Qt.NoModifier,
                QtCore.Qt.ControlModifier,
            ):

                if self.region is None:

                    if modifier == QtCore.Qt.ControlModifier:
                        increment = 20
                    else:
                        increment = 1

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

                else:
                    increment = 1
                    start, stop = self.region.getRegion()

                    if self.region_lock is None:

                        if modifier == QtCore.Qt.ControlModifier:
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
                        self.viewbox.setXRange(left_side, pos, padding=0)
                    elif pos <= left_side:
                        self.viewbox.setXRange(pos, right_side, padding=0)

                    if self.region_lock is not None:
                        self.region.setRegion((self.region_lock, pos))
                    else:
                        self.region.setRegion(tuple(sorted((second_pos, pos))))

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

                self.update()

            elif key == QtCore.Qt.Key_H and modifier == QtCore.Qt.NoModifier:
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

            elif key == QtCore.Qt.Key_W and modifier == QtCore.Qt.NoModifier:

                if len(self.all_timebase):
                    start_ts = np.amin(self.all_timebase)
                    stop_ts = np.amax(self.all_timebase)

                    self.viewbox.setXRange(start_ts, stop_ts)

                    if self.cursor1:
                        self.cursor_moved.emit(self.cursor1)

            elif key == QtCore.Qt.Key_Insert and modifier == QtCore.Qt.NoModifier:
                self.insert_computation()

            else:
                handled = False

            if not handled:
                self.parent().keyPressEvent(event)

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

            self.update()

    def paintEvent(self, ev):
        if not self._can_paint or not self._can_paint_global:
            return

        event_rect = ev.rect()

        super().paintEvent(ev)

        if self._pixmap is None:
            self._pixmap = QtGui.QPixmap(event_rect.width(), event_rect.height())
            self._pixmap.fill(self.backgroundBrush().color())

            paint = QtGui.QPainter()
            paint.begin(self._pixmap)
            paint.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            paint.setRenderHints(paint.RenderHint.Antialiasing, False)

            self.x_range, self.y_range = self.viewbox.viewRange()

            rect = self.viewbox.sceneBoundingRect()

            self.px = (self.x_range[1] - self.x_range[0]) / rect.width()
            self.py = rect.height()

            with_dots = self.with_dots

            self.auto_clip_rect(paint)

            rect = self.viewbox.sceneBoundingRect()

            delta = rect.x()
            x_start = self.x_range[0]

            no_brush = QtGui.QBrush()
            pen_width = self.line_width
            dots_with = self.dot_width

            paint.resetTransform()
            paint.translate(0, 0)
            paint.setBrush(no_brush)

            for i, sig in enumerate(self.signals):
                if not sig.enable:
                    continue

                y = sig.plot_samples
                x = sig.plot_timestamps

                x, y = self.scale_curve_to_pixmap(
                    x, y, y_range=sig.y_range, x_start=x_start, delta=delta
                )

                sig.pen.setWidth(pen_width)

                paint.setPen(sig.pen)

                paint.drawPath(self.generatePath(x, y, sig))

                if with_dots:
                    paint.setRenderHints(paint.RenderHint.Antialiasing, True)
                    pos = np.isfinite(y)
                    y = y[pos]
                    x = x[pos]

                    _pen = fn.mkPen(sig.color.name())
                    _pen.setWidth(dots_with)
                    _pen.setCapStyle(QtCore.Qt.RoundCap)
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

                            y = sig.plot_samples.astype("f8")
                            y[~idx] = np.inf
                            x = sig.plot_timestamps

                            x, y = self.scale_curve_to_pixmap(
                                x, y, y_range=sig.y_range, x_start=x_start, delta=delta
                            )

                            color = range_info["font_color"]
                            pen = fn.mkPen(color.name())
                            pen.setWidth(pen_width)

                            paint.setPen(pen)
                            paint.drawPath(self.generatePath(x, y))

                            if with_dots:
                                paint.setRenderHints(
                                    paint.RenderHint.Antialiasing, True
                                )
                                pen.setWidth(dots_with)
                                pen.setCapStyle(QtCore.Qt.RoundCap)
                                paint.setPen(pen)

                                pos = np.isfinite(y)
                                y = y[pos]
                                x = x[pos]

                                poly, arr = polygon_and_ndarray(x.size)

                                arr[:, 0] = x
                                arr[:, 1] = y
                                paint.drawPoints(poly)
                                paint.setRenderHints(
                                    paint.RenderHint.Antialiasing, False
                                )
            paint.end()

        paint = QtGui.QPainter()
        vp = self.viewport()
        paint.begin(vp)
        paint.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        paint.setRenderHint(paint.RenderHint.Antialiasing, False)

        if self.y_axis.picture is None:
            self.y_axis.paint(paint, None, None)
        if self.x_axis.picture is None:
            self.x_axis.paint(paint, None, None)

        paint.drawPixmap(
            self.y_axis.sceneBoundingRect(),
            self.y_axis.picture,
            self.y_axis.boundingRect(),
        )
        paint.drawPixmap(
            self.x_axis.sceneBoundingRect(),
            self.x_axis.picture,
            self.x_axis.boundingRect(),
        )

        for ax in self.axes:
            if isinstance(ax, FormatedAxis) and ax.isVisible():
                if ax.picture is None:
                    ax.paint(paint, None, None)
                paint.drawPixmap(
                    ax.sceneBoundingRect(),
                    ax.picture,
                    ax.boundingRect(),
                )

        self.auto_clip_rect(paint)

        paint.drawPixmap(event_rect, self._pixmap, event_rect)

        self.draw_grids(paint, event_rect)

        if self.cursor1 is not None and self.cursor1.isVisible():
            self.cursor1.paint(paint, plot=self, uuid=self.current_uuid)

        if self.region is not None:
            self.region.paint(paint, plot=self, uuid=self.current_uuid)

        paint.end()

    def range_modified_finished_handler(self, region):
        if self.region_lock is not None:
            for i in range(2):
                if self.region.lines[i].value() == self.region_lock:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.DashDotDotLine)
                else:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.SolidLine)
        self.range_modified_finished.emit(region)

    def range_modified_handler(self, region):
        if self.region_lock is not None:
            for i in range(2):
                if self.region.lines[i].value() == self.region_lock:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.DashDotDotLine)
                else:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.SolidLine)

    def scale_curve_to_pixmap(self, x, y, y_range, x_start, delta):

        if self.py:
            y_low, y_high = y_range
            y_scale = (y_high - y_low) / self.py
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
        delta = self.viewbox.sceneBoundingRect().x()
        x_start = self.viewbox.viewRange()[0][0]

        candidates = []
        for sig in self.signals:
            if not sig.enable:
                continue

            val, _1, _2 = sig.value_at_timestamp(x, numeric=True)

            if val == "n.a.":
                continue

            x_val, y_val = self.scale_curve_to_pixmap(
                x, val, y_range=sig.y_range, x_start=x_start, delta=delta
            )

            candidates.append((abs(y_val - y), sig.uuid))

        if candidates:
            candidates.sort()
            self.curve_clicked.emit(candidates[0][1])

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

        self.update()

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

    def set_dots(self, with_dots):
        self.with_dots = with_dots
        self.update()

    def set_individual_axis(self, uuid, state):

        sig, index = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.Checked, True, 1):
            if sig.enable:
                self.get_axis(index).show()
            sig.individual_axis = True
        else:
            self.get_axis(index).hide()
            sig.individual_axis = False

        self.update()

    def set_line_interconnect(self, line_interconnect):
        self.line_interconnect = line_interconnect

        self._curve.setData(line_interconnect=line_interconnect)
        self.update()

    def set_locked(self, locked):
        self.locked = locked
        for axis in self.axes:
            if isinstance(axis, FormatedAxis):
                axis.locked = locked

        self.viewbox.setMouseEnabled(y=not self.locked)

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
        self.update()

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

        self.update()

    def set_y_range(self, uuid, y_range):
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

        if update:
            self.update()

    def signal_by_name(self, name):
        for i, sig in enumerate(self.signals):
            if sig.name == name:
                return sig, i

        raise Exception(
            f"Signal not found: {name} {[sig.name for sig in self.signals]}"
        )

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

    def update(self, *args, **kwargs):
        self._pixmap = None
        self.viewbox.update()
        # super().update()

    def update_views(self):
        geometry = self.viewbox.sceneBoundingRect()
        if geometry != self.viewbox_geometry:
            self.viewbox_geometry = geometry

    def xrange_changed_handle(self, *args, force=False):

        if self._can_paint:
            self.trim(force=force)
            self.update()

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


try:
    import scipy

    from .fft_window import FFTWindow
except ImportError:
    pass

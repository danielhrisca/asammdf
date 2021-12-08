# -*- coding: utf-8 -*-
from copy import deepcopy
from datetime import timedelta
from functools import partial, reduce
import logging
import os
from pathlib import Path
from time import perf_counter, sleep
from traceback import format_exc

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# imports for pyinstaller
import pyqtgraph.canvas.CanvasTemplate_pyqt5
import pyqtgraph.canvas.TransformGuiTemplate_pyqt5
import pyqtgraph.console.template_pyqt5
import pyqtgraph.functions as fn
import pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt5
import pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5
import pyqtgraph.GraphicsScene.exportDialogTemplate_pyqt5
import pyqtgraph.imageview.ImageViewTemplate_pyqt5

try:
    from ...blocks.cutils import positions
except:
    pass


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


# fixes https://github.com/pyqtgraph/pyqtgraph/issues/2117
def mouseReleaseEvent(self, ev):
    if self.mouseGrabberItem() is None:
        if ev.button() in self.dragButtons:
            if self.sendDragEvent(ev, final=True):
                # print "sent drag event"
                ev.accept()
            self.dragButtons.remove(ev.button())
        else:
            cev = [e for e in self.clickEvents if e.button() == ev.button()]
            if cev:
                if self.sendClickEvent(cev[0]):
                    # print "sent click event"
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


# speed-up monkey patches
pg.graphicsItems.ScatterPlotItem.SymbolAtlas._keys = _keys
pg.graphicsItems.ScatterPlotItem._USE_QRECT = False
pg.GraphicsScene.mouseReleaseEvent = mouseReleaseEvent


from ...blocks import v4_constants as v4c
from ...mdf import MDF
from ...signal import Signal
from ..dialogs.define_channel import DefineChannel
from ..ui import resource_rc as resource_rc
from ..utils import COLORS, copy_ranges, extract_mime_names
from .channel_display import ChannelDisplay
from .channel_group_display import ChannelGroupDisplay
from .channel_stats import ChannelStats
from .cursor import Cursor
from .dict_to_tree import ComputedChannelInfoWindow
from .formated_axis import FormatedAxis
from .list import ListWidget
from .list_item import ListItem
from .tree import ChannelsGroupTreeItem, ChannelsTreeItem, ChannelsTreeWidget

bin_ = bin

HERE = Path(__file__).resolve().parent

FAKE = -2


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


class PlotSignal(Signal):
    def __init__(self, signal, index=0, fast=False, trim_info=None, duplication=1):
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

        self.duplication = duplication
        self.uuid = getattr(signal, "uuid", os.urandom(6).hex())
        self.mdf_uuid = getattr(signal, "mdf_uuid", os.urandom(6).hex())

        self.group_index = getattr(signal, "group_index", FAKE)
        self.channel_index = getattr(signal, "channel_index", FAKE)
        self.precision = getattr(signal, "precision", 6)

        self._mode = "raw"

        self.enable = True
        self.format = "phys"

        self.individual_axis = False
        self.computed = signal.computed
        self.computation = signal.computation

        self.trim_info = None

        # take out NaN values
        samples = self.samples
        if samples.dtype.kind not in "SUV":
            nans = np.isnan(samples)
            if np.any(nans):
                self.samples = self.samples[~nans]
                self.timestamps = self.timestamps[~nans]

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

        if not fast:
            if hasattr(signal, "color"):
                color = signal.color or COLORS[index % 10]
            else:
                color = COLORS[index % 10]
            self.color = color
            self.pen = pg.mkPen(color=color, style=QtCore.Qt.SolidLine)

        if len(self.phys_samples) and not fast:

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

        self.mode = "phys"
        if not fast:
            self.trim(*(trim_info or (None, None, 1900)))

        self.size = len(self.samples)

    @property
    def min(self):
        return self._min if self._mode == "phys" else self._min_raw

    @min.setter
    def min(self, min):
        self._min = min

    @property
    def max(self):
        return self._max if self._mode == "phys" else self._max_raw

    @max.setter
    def max(self, max):
        self._max = max

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

    @std.setter
    def std(self, std):
        self._std = std

    def cut(
        self, start=None, stop=None, include_ends=True, interpolation_mode=0, fast=False
    ):
        cut_sig = super().cut(start, stop, include_ends, interpolation_mode)

        cut_sig.group_index = self.group_index
        cut_sig.channel_index = self.channel_index
        cut_sig.computed = self.computed
        cut_sig.color = self.color
        cut_sig.computation = self.computation
        cut_sig.precision = self.precision
        cut_sig.mdf_uuif = self.mdf_uuid

        return PlotSignal(cut_sig, duplication=self.duplication, fast=fast)

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

    def trim_c(self, start=None, stop=None, width=1900):

        trim_info = (start, stop, width)
        if self.trim_info == trim_info:
            return None

        self.trim_info = trim_info
        sig = self
        sig_timestamps = sig.timestamps
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
                sig.plot_samples = signal_samples[:0]
                sig.plot_timestamps = sig_timestamps[:0]
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

                    pos = np.empty(2 * count, dtype="i4")

                    positions(samples.astype("f8"), pos, steps, count, rest)

                    sig.plot_samples = samples[pos]
                    sig.plot_timestamps = timestamps[pos]

                else:
                    start_ = simple_min(simple_max(0, start_ - 2), dim - 1)
                    stop_ = simple_min(stop_ + 2, dim)

                    if start_ == 0 and stop_ == dim:
                        sig.plot_samples = signal_samples
                        sig.plot_timestamps = sig_timestamps

                        pos = None
                    else:

                        sig.plot_samples = signal_samples[start_:stop_]
                        sig.plot_timestamps = sig_timestamps[start_:stop_]

                        pos = np.arange(start_, stop_)

        else:
            pos = None

        return pos

    def trim_python(self, start=None, stop=None, width=1900):
        trim_info = (start, stop, width)
        if self.trim_info == trim_info:
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

    def trim(self, start=None, stop=None, width=1900):
        try:
            return self.trim_c(start, stop, width)
        except:
            return self.trim_python(start, stop, width)

    def value_at_timestamp(self, stamp):
        if len(self.timestamps) and self.timestamps[-1] < stamp:
            values = self.samples[-1:]
        else:
            values = super().cut(stamp, stamp).samples

        if len(values) == 0:
            value = "n.a."
            kind = values.dtype.kind
        else:

            if self.mode != "raw":
                if self.conversion:
                    values = self.conversion.convert(values)
            value = values[0]

            kind = values.dtype.kind
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


class Plot(QtWidgets.QWidget):

    add_channels_request = QtCore.pyqtSignal(list)
    close_request = QtCore.pyqtSignal()
    clicked = QtCore.pyqtSignal()
    cursor_moved_signal = QtCore.pyqtSignal(object, float)
    cursor_removed_signal = QtCore.pyqtSignal(object)
    region_moved_signal = QtCore.pyqtSignal(object, list)
    region_removed_signal = QtCore.pyqtSignal(object)
    show_properties = QtCore.pyqtSignal(list)
    splitter_moved = QtCore.pyqtSignal(object, int)
    pattern_group_added = QtCore.pyqtSignal(object, object)

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
        *args,
        **kwargs,
    ):
        events = kwargs.pop("events", None)
        super().__init__(*args, **kwargs)
        self.line_interconnect = line_interconnect
        self.setContentsMargins(0, 0, 0, 0)
        self.pattern = {}
        self.mdf = mdf

        self.x_name = "t" if x_axis == "time" else "f"
        self.x_unit = "s" if x_axis == "time" else "Hz"

        self.info_uuid = None

        self._range_start = None
        self._range_stop = None

        self._can_switch_mode = True

        self._accept_cursor_update = False
        self._last_cursor_update = perf_counter()

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
        )

        self.cursor_info = CursorInfo(
            precision=QtCore.QSettings().value("plot_cursor_precision", 6),
            unit=self.x_unit,
            name=self.x_name,
            plot=self.plot,
        )

        vbox.addWidget(self.channel_selection)
        vbox.addWidget(self.cursor_info)

        self.plot.range_modified.connect(self.range_modified)
        self.plot.range_removed.connect(self.range_removed)
        self.plot.range_modified_finished.connect(self.range_modified_finished)
        self.plot.cursor_removed.connect(self.cursor_removed)
        self.plot.cursor_moved.connect(self.cursor_moved)
        self.plot.cursor_move_finished.connect(self.cursor_move_finished)
        self.plot.xrange_changed.connect(self.xrange_changed)
        self.plot.computation_channel_inserted.connect(
            self.computation_channel_inserted
        )
        self.plot.curve_clicked.connect(self.curve_clicked)

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

        if signals:
            self.add_new_channels(signals)

        self.channel_selection.itemsDeleted.connect(self.channel_selection_reduced)
        self.channel_selection.itemPressed.connect(self.channel_selection_modified)
        self.channel_selection.currentItemChanged.connect(
            self.channel_selection_row_changed
        )
        self.channel_selection.add_channels_request.connect(self.add_channels_request)
        self.channel_selection.set_time_offset.connect(self.plot.set_time_offset)
        self.channel_selection.show_properties.connect(self._show_properties)
        self.channel_selection.insert_computation.connect(self.plot.insert_computation)

        self.channel_selection.itemChanged.connect(self.channel_selection_item_changed)
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
                    (QtCore.Qt.Key_M, QtCore.Qt.NoModifier),
                    (QtCore.Qt.Key_B, QtCore.Qt.ControlModifier),
                    (QtCore.Qt.Key_H, QtCore.Qt.ControlModifier),
                    (QtCore.Qt.Key_P, QtCore.Qt.ControlModifier),
                ]
            )
            | self.plot.keyboard_events
        )

        self.splitter.splitterMoved.connect(self.set_splitter)

    def curve_clicked(self, uuid):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while iterator.value():
            item = iterator.value()
            widget = self.channel_selection.itemWidget(item, 1)
            if isinstance(widget, ChannelDisplay) and widget.uuid == uuid:
                self.channel_selection.setCurrentItem(item)
                break

            iterator += 1

    def channel_selection_rearranged(self, uuids):
        uuids = set(uuids)

        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while iterator.value():
            item = iterator.value()
            widget = self.channel_selection.itemWidget(item, 1)
            if isinstance(widget, ChannelDisplay) and widget.uuid in uuids:
                widget.color_changed.connect(self.plot.set_color)
                widget.enable_changed.connect(self.plot.set_signal_enable)
                widget.ylink_changed.connect(self.plot.set_common_axis)
                widget.individual_axis_changed.connect(self.plot.set_individual_axis)

            iterator += 1

    def channel_selection_item_changed(self, item, column):
        if item is not None and column == 0:
            state = item.checkState(0)
            widget = self.channel_selection.itemWidget(item, 1)
            if isinstance(widget, ChannelDisplay):
                widget.enable_changed.emit(widget.uuid, state)

    def channel_selection_item_double_clicked(self, item, column):
        if isinstance(item, ChannelsGroupTreeItem):
            item.show_info()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def channel_selection_modified(self, item):
        if isinstance(item, ChannelsTreeItem):
            uuid = self.channel_selection.itemWidget(item, 1).uuid
            self.info_uuid = uuid

            sig, index = self.plot.signal_by_uuid(uuid)
            if sig.enable:

                self.plot.curves[index].hide()
                self.plot.update_signal_curve(sig, index)
                for i in range(3):
                    QtWidgets.QApplication.processEvents()
                    sleep(0.01)
                self.plot.curves[index].show()
                self.plot.update_signal_curve(sig, index)

                self.plot.set_current_uuid(self.info_uuid)
                if self.info.isVisible():
                    stats = self.plot.get_stats(self.info_uuid)
                    self.info.set_stats(stats)

    def channel_selection_row_changed(self, current, previous):
        if isinstance(current, ChannelsTreeItem):
            item = current
            uuid = self.channel_selection.itemWidget(item, 1).uuid
            self.info_uuid = uuid

            sig, index = self.plot.signal_by_uuid(uuid)
            if sig.enable:

                self.plot.curves[index].hide()
                self.plot.update_signal_curve(sig, index)
                for i in range(3):
                    QtWidgets.QApplication.processEvents()
                    sleep(0.01)
                self.plot.curves[index].show()
                self.plot.update_signal_curve(sig, index)

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

    def cursor_move_finished(self):
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
            self._accept_cursor_update = True
            self.plot.cursor1.setPos(next_pos)

        # self.plot.cursor_hint.setData(x=[], y=[])
        self.plot.cursor_hint.hide()

    def cursor_moved(self):
        if self.plot.cursor1 is None or (
            perf_counter() - self._last_cursor_update < 0.030
            and not self._accept_cursor_update
        ):
            return

        position = self.plot.cursor1.value()

        if not self.plot.region:

            self.cursor_info.update_value()

            iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
            while iterator.value():
                item = iterator.value()
                iterator += 1
                if not self.channel_selection.is_item_visible(item):
                    continue

                widget = self.channel_selection.itemWidget(item, 1)
                if isinstance(widget, ChannelDisplay):

                    signal, idx = self.plot.signal_by_uuid(widget.uuid)
                    value, kind, fmt = signal.value_at_timestamp(position)

                    widget.set_prefix("= ")
                    widget.kind = kind
                    widget.set_fmt(fmt)

                    widget.set_value(value, update=True)

        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self._last_cursor_update = perf_counter()
        self._accept_cursor_update = False
        self.cursor_moved_signal.emit(self, position)

    def cursor_removed(self):

        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while iterator.value():
            item = iterator.value()
            widget = self.channel_selection.itemWidget(item, 1)

            if isinstance(widget, ChannelDisplay) and not self.plot.region:
                self.cursor_info.update_value()
                widget.set_prefix("")
                widget.set_value("")

            iterator += 1

        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self.cursor_removed_signal.emit(self)

    def range_modified(self):
        if self.plot.region is None or (
            perf_counter() - self._last_cursor_update < 0.030
            and not self._accept_cursor_update
        ):
            return

        start, stop = self.plot.region.getRegion()

        self.cursor_info.update_value()

        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while iterator.value():
            item = iterator.value()
            iterator += 1

            if not self.channel_selection.is_item_visible(item):
                continue

            widget = self.channel_selection.itemWidget(item, 1)
            if isinstance(widget, ChannelDisplay):

                signal, i = self.plot.signal_by_uuid(widget.uuid)

                start_v, kind, fmt = signal.value_at_timestamp(start)
                stop_v, kind, fmt = signal.value_at_timestamp(stop)

                widget.set_prefix("Δ = ")
                widget.set_fmt(signal.format)

                if "n.a." not in (start_v, stop_v):
                    if kind in "ui":
                        delta = np.int64(stop_v) - np.int64(start_v)
                        widget.kind = kind
                        widget.set_value(delta)
                        widget.set_fmt(fmt)
                    elif kind == "f":
                        delta = stop_v - start_v
                        widget.kind = kind
                        widget.set_value(delta)
                        widget.set_fmt(fmt)
                    else:
                        widget.set_value("n.a.")
                else:
                    widget.set_value("n.a.")

        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self._last_cursor_update = perf_counter()
        self._accept_cursor_update = False

        self.region_moved_signal.emit(self, [start, stop])

    def xrange_changed(self):
        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

    def range_modified_finished(self):
        if not self.plot.region:
            return
        start, stop = self.plot.region.getRegion()

        timebase = self.plot.get_current_timebase()

        if timebase.size:
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

            self._accept_cursor_update = True
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
                    self.channel_selection.itemWidget(item, 1).uuid
                    for item in selected_items
                ]

                signals = [self.plot.signal_by_uuid(uuid) for uuid in uuids]

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

                    widget = self.widget_by_uuid(signal.uuid)
                    widget.kind = kind
                    widget.set_fmt(fmt)

                    if self.plot.current_uuid == signal.uuid:
                        self.plot.y_axis.format = fmt
                        self.plot.y_axis.hide()
                        self.plot.y_axis.show()
            if self.plot.cursor1:
                self.plot.cursor_moved.emit()

        elif (
            key in (QtCore.Qt.Key_R, QtCore.Qt.Key_S)
            and modifiers == QtCore.Qt.AltModifier
            and self._can_switch_mode
        ):
            selected_items = self.channel_selection.selectedItems()
            if not selected_items:
                signals = [(sig, i) for i, sig in enumerate(self.plot.signals)]

            else:
                uuids = [
                    self.channel_selection.itemWidget(item, 1).uuid
                    for item in selected_items
                ]

                signals = [self.plot.signal_by_uuid(uuid) for uuid in uuids]

            if key == QtCore.Qt.Key_R:
                mode = "raw"
                style = QtCore.Qt.DashLine

            else:
                mode = "phys"
                style = QtCore.Qt.SolidLine

            for signal, idx in signals:
                if signal.mode != mode:
                    signal.pen = pg.mkPen(color=signal.color, style=style)

                    view = self.plot.view_boxes[idx]
                    _, (buttom, top) = view.viewRange()

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

                    view.setYRange(buttom, top, padding=0, update=True)

                    if self.plot.current_uuid == signal.uuid:
                        self.plot.y_axis.mode = mode
                        self.plot.y_axis.hide()
                        self.plot.y_axis.show()

            self.plot.update_lines()

            if self.plot.cursor1:
                self.plot.cursor_moved.emit()

        elif key == QtCore.Qt.Key_I and modifiers == QtCore.Qt.ControlModifier:
            if self.plot.cursor1:
                position = self.plot.cursor1.value()
                comment, _ = QtWidgets.QInputDialog.getMultiLineText(
                    self,
                    "Insert comments",
                    f"Enter the comments for cursor position {position:.9f}s:",
                    "",
                )
                line = pg.InfiniteLine(
                    pos=position,
                    label=f"t = {position}s\n\n{comment}",
                    pen={"color": "#FF0000", "width": 2, "style": QtCore.Qt.DashLine},
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

        elif (key, modifiers) in self.plot.keyboard_events:
            self.plot.keyPressEvent(event)
        else:
            event.ignore()

    def range_removed(self):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while iterator.value():
            item = iterator.value()
            widget = self.channel_selection.itemWidget(item, 1)
            if isinstance(widget, ChannelDisplay):
                widget.set_prefix("")
                widget.set_value("")

            iterator += 1

        self._range_start = None
        self._range_stop = None

        self.cursor_info.update_value()

        if self.plot.cursor1:
            self.plot.cursor_moved.emit()
        if self.info.isVisible():
            stats = self.plot.get_stats(self.info_uuid)
            self.info.set_stats(stats)

        self.region_removed_signal.emit(self)

    def computation_channel_inserted(self):
        sig = self.plot.signals[-1]

        name, unit = sig.name, sig.unit
        item = ChannelsTreeItem((-1, -1), name, sig.computation)
        tooltip = getattr(sig, "tooltip", "")
        it = ChannelDisplay(
            sig.uuid,
            unit,
            sig.samples.dtype.kind,
            3,
            tooltip,
            "",
            item=item,
            parent=self,
        )
        it.setAttribute(QtCore.Qt.WA_StyledBackground)

        font = QtGui.QFont()
        font.setItalic(True)
        it.name.setFont(font)

        it.set_name(name)
        it.set_value("")
        it.set_color(sig.color)
        item.setSizeHint(1, it.sizeHint())
        self.channel_selection.addTopLevelItem(item)
        self.channel_selection.setItemWidget(item, 1, it)

        it.color_changed.connect(self.plot.set_color)
        it.unit_changed.connect(self.plot.set_unit)
        it.name_changed.connect(self.plot.set_name)
        it.enable_changed.connect(self.plot.set_signal_enable)
        it.ylink_changed.connect(self.plot.set_common_axis)
        it.individual_axis_changed.connect(self.plot.set_individual_axis)
        it.unit_changed.connect(self.plot.set_unit)

        it.enable_changed.emit(sig.uuid, 1)
        it.enable_changed.emit(sig.uuid, 0)
        it.enable_changed.emit(sig.uuid, 1)

        self.info_uuid = sig.uuid

        self.plot.set_current_uuid(self.info_uuid, True)

    def add_new_channels(self, channels, mime_data=None, destination=None):

        size = sum(self.splitter.sizes())
        if size >= 600:
            self.splitter.setSizes([500, size - 500, 0])
        else:
            self.splitter.setSizes([size - 100, 100, 0])

        def add_new_items(tree, root, items, items_pool):
            for (name, group_index, channel_index, mdf_uuid, type_, ranges) in items:

                ranges = deepcopy(ranges)
                for range_info in ranges:
                    range_info["font_color"] = QtGui.QColor(range_info["font_color"])
                    range_info["background_color"] = QtGui.QColor(
                        range_info["background_color"]
                    )

                if type_ == "group":
                    pattern = group_index
                    item = ChannelsGroupTreeItem(name, pattern)
                    widget = ChannelGroupDisplay(
                        name, pattern, item=item, ranges=ranges
                    )
                    widget.item = item
                    root.addChild(item)
                    tree.setItemWidget(item, 1, widget)

                    add_new_items(tree, item, channel_index, items_pool)

                else:

                    key = (name, group_index, channel_index, mdf_uuid)

                    if isinstance(name, dict):
                        key = (name["name"],) + key[1:]
                        if key in items_pool:
                            item, widget = items_pool[key]
                            widget.item = item
                            widget.set_ranges(ranges)

                            root.addChild(item)
                            tree.setItemWidget(item, 1, widget)

                            del items_pool[key]
                    else:

                        if key in items_pool:
                            item, widget = items_pool[key]
                            widget.item = item
                            widget.set_ranges(ranges)

                            root.addChild(item)
                            tree.setItemWidget(item, 1, widget)

                            del items_pool[key]

        for sig in channels:
            sig.uuid = os.urandom(6).hex()

        invalid = []

        can_trim = True
        for channel in channels:
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

        valid = []
        invalid = []
        for channel in channels:
            if len(channel):
                samples = channel.samples
                if samples.dtype.kind not in "SUV" and np.all(np.isnan(samples)):
                    invalid.append(channel.name)
                elif channel.conversion:
                    samples = channel.physical().samples
                    if samples.dtype.kind not in "SUV" and np.all(np.isnan(samples)):
                        invalid.append(channel.name)
                    else:
                        valid.append(channel)
                else:
                    valid.append(channel)
            else:
                valid.append(channel)

        if invalid:
            QtWidgets.QMessageBox.warning(
                self,
                "All NaN channels will not be plotted:",
                f"The following channels have all NaN samples and will not be plotted:\n{', '.join(invalid)}",
            )

        channels = valid

        channels = self.plot.add_new_channels(channels)

        enforce_y_axis = False
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while iterator.value():
            item = iterator.value()
            widget = self.channel_selection.itemWidget(item, 1)
            if isinstance(widget, ChannelDisplay):
                if widget.ylink.checkState() == QtCore.Qt.Unchecked:
                    enforce_y_axis = False
                    break
                else:
                    enforce_y_axis = True

            iterator += 1

        new_items = {}
        for sig in channels:

            item = ChannelsTreeItem(
                (sig.group_index, sig.channel_index),
                sig.name,
                sig.computation,
                None,
                sig.mdf_uuid,
                check=QtCore.Qt.Checked if sig.enable else QtCore.Qt.Unchecked,
            )

            # item.setData(QtCore.Qt.UserRole, sig.name)
            tooltip = getattr(sig, "tooltip", "") or f"{sig.name}\n{sig.comment}"
            if sig.source:
                src = sig.source
                source_type = v4c.SOURCE_TYPE_TO_STRING[src.source_type]
                bus_type = v4c.BUS_TYPE_TO_STRING[src.bus_type]
                details = f"     {source_type} source on bus {bus_type}: name=[{src.name}] path=[{src.path}]"
            else:
                details = ""

            if len(sig.samples) and sig.conversion:
                kind = sig.conversion.convert(sig.samples[:1]).dtype.kind
            else:
                kind = sig.samples.dtype.kind
            it = ChannelDisplay(
                sig.uuid, sig.unit, kind, 3, tooltip, details, parent=self, item=item
            )
            if self.channel_selection.details_enabled:
                it.details.setVisible(True)
            it.setAttribute(QtCore.Qt.WA_StyledBackground)

            if len(sig):
                it.set_value(sig.samples[0])

            if sig.computed:
                font = QtGui.QFont()
                font.setItalic(True)
                it.name.setFont(font)

            it.set_name(sig.name)
            it.set_color(sig.color)
            item.setSizeHint(1, it.sizeHint())

            if mime_data is None:

                if destination is None:
                    self.channel_selection.addTopLevelItem(item)
                else:
                    destination.addChild(item)

                self.channel_selection.setItemWidget(item, 1, it)
            else:
                new_items[(item.name, *item.entry, item.mdf_uuid)] = (item, it)

            it.color_changed.connect(self.plot.set_color)
            it.enable_changed.connect(self.plot.set_signal_enable)
            it.ylink_changed.connect(self.plot.set_common_axis)
            it.unit_changed.connect(self.plot.set_unit)
            it.name_changed.connect(self.plot.set_name)
            if enforce_y_axis:
                it.ylink.setCheckState(QtCore.Qt.Checked)
            it.individual_axis_changed.connect(self.plot.set_individual_axis)

            item.setFlags(
                item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled
            )
            self.info_uuid = sig.uuid

        if mime_data:
            add_new_items(
                self.channel_selection,
                destination or self.channel_selection.invisibleRootItem(),
                mime_data,
                new_items,
            )

            # still have simple signals to add
            if new_items:
                for item, widget in new_items.values():
                    self.channel_selection.addTopLevelItem(item)
                    widget.item = item
                    self.channel_selection.setItemWidget(item, 1, widget)

        self.channel_selection.update_channel_groups_count()
        self.channel_selection.refresh()

    def to_config(self):
        def item_to_config(tree, root):
            channels = []

            for i in range(root.childCount()):
                item = root.child(i)
                widget = tree.itemWidget(item, 1)
                if isinstance(widget, ChannelDisplay):

                    channel = {"type": "channel"}

                    sig, idx = self.plot.signal_by_uuid(widget.uuid)

                    channel["name"] = sig.name
                    channel["unit"] = sig.unit
                    channel["enabled"] = item.checkState(0) == QtCore.Qt.Checked

                    if widget.individual_axis.checkState() == QtCore.Qt.Checked:
                        channel["individual_axis"] = True
                        channel["individual_axis_width"] = (
                            self.plot.axes[idx].boundingRect().width()
                        )
                    else:
                        channel["individual_axis"] = False

                    channel["common_axis"] = (
                        widget.ylink.checkState() == QtCore.Qt.Checked
                    )
                    channel["color"] = sig.color
                    channel["computed"] = sig.computed
                    channel["ranges"] = copy_ranges(widget.ranges)

                    for range_info in channel["ranges"]:
                        range_info["background_color"] = range_info[
                            "background_color"
                        ].name()
                        range_info["font_color"] = range_info["font_color"].name()

                    channel["precision"] = widget.precision
                    channel["fmt"] = widget.fmt
                    channel["mode"] = sig.mode
                    if sig.computed:
                        channel["computation"] = sig.computation

                    view = self.plot.view_boxes[idx]
                    channel["y_range"] = [float(e) for e in view.viewRange()[1]]
                    channel["mdf_uuid"] = str(sig.mdf_uuid)

                    if sig.computed and sig.conversion:
                        channel["user_defined_name"] = sig.name
                        channel["name"] = sig.computation["expression"].strip("}{")

                        channel["conversion"] = {}
                        for i in range(sig.conversion.val_param_nr):
                            channel["conversion"][
                                f"text_{i}"
                            ] = sig.conversion.referenced_blocks[f"text_{i}"].decode(
                                "utf-8"
                            )
                            channel["conversion"][f"val_{i}"] = sig.conversion[
                                f"val_{i}"
                            ]

                elif isinstance(widget, ChannelGroupDisplay):
                    pattern = item.pattern
                    if pattern:
                        ranges = copy_ranges(pattern["ranges"])

                        for range_info in ranges:
                            range_info["font_color"] = range_info["font_color"].name()
                            range_info["background_color"] = range_info[
                                "background_color"
                            ].name()

                        pattern["ranges"] = ranges

                    ranges = copy_ranges(widget.ranges)

                    for range_info in ranges:
                        range_info["font_color"] = range_info["font_color"].name()
                        range_info["background_color"] = range_info[
                            "background_color"
                        ].name()

                    channel = {
                        "type": "group",
                        "name": widget.name.text().rsplit("\t[")[0],
                        "channels": item_to_config(tree, item)
                        if item.pattern is None
                        else [],
                        "enabled": item.checkState(0) == QtCore.Qt.Checked,
                        "pattern": pattern,
                        "ranges": ranges,
                    }

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

    def widget_by_uuid(self, uuid):
        iterator = QtWidgets.QTreeWidgetItemIterator(self.channel_selection)
        while iterator.value():
            item = iterator.value()
            widget = self.channel_selection.itemWidget(item, 1)

            if isinstance(widget, ChannelDisplay) and widget.uuid == uuid:
                break

            iterator += 1
        else:
            widget = None
        return widget

    def _show_properties(self, uuid):
        for sig in self.plot.signals:
            if sig.uuid == uuid:
                if sig.computed:
                    view = ComputedChannelInfoWindow(sig, self)
                    view.show()

                else:
                    self.show_properties.emit(
                        [sig.group_index, sig.channel_index, sig.mdf_uuid]
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
            self.range_modified()
        else:
            self.cursor_moved()


class _Plot(pg.PlotWidget):
    cursor_moved = QtCore.pyqtSignal()
    cursor_removed = QtCore.pyqtSignal()
    range_removed = QtCore.pyqtSignal()
    range_modified = QtCore.pyqtSignal()
    range_modified_finished = QtCore.pyqtSignal()
    cursor_move_finished = QtCore.pyqtSignal()
    xrange_changed = QtCore.pyqtSignal()
    computation_channel_inserted = QtCore.pyqtSignal()
    curve_clicked = QtCore.pyqtSignal(str)

    add_channels_request = QtCore.pyqtSignal(list)

    def __init__(
        self,
        signals=None,
        with_dots=False,
        origin=None,
        mdf=None,
        line_interconnect="line",
        x_axis="time",
        *args,
        **kwargs,
    ):
        events = kwargs.pop("events", [])
        super().__init__()

        self.cursor_unit = "s" if x_axis == "time" else "Hz"

        self.line_interconnect = (
            line_interconnect if line_interconnect != "line" else ""
        )

        self._last_update = perf_counter()
        self._can_trim = True
        self.mdf = mdf

        self.setAcceptDrops(True)

        self._last_size = self.geometry()
        self._settings = QtCore.QSettings()

        self.setContentsMargins(5, 5, 5, 5)
        self.xrange_changed.connect(self.xrange_changed_handle)
        self.with_dots = with_dots
        if self.with_dots:
            self.curvetype = pg.PlotDataItem
        else:
            self.curvetype = pg.PlotCurveItem
        self.info = None
        self.current_uuid = 0

        self.standalone = kwargs.get("standalone", False)

        self.region = None
        self.region_lock = None
        self.cursor1 = None
        self.cursor2 = None
        self.signals = signals or []

        self.axes = []
        self._axes_layout_pos = 2

        self.disabled_keys = set()

        self._timebase_db = {}
        self.all_timebase = self.timebase = np.array([])
        for sig in self.signals:
            uuids = self._timebase_db.setdefault(id(sig.timestamps), set())
            uuids.add(sig.uuid)

        #        self._compute_all_timebase()

        self.showGrid(x=True, y=True)

        self.plot_item = self.plotItem
        self.plot_item.hideAxis("left")
        self.plot_item.hideAxis("bottom")
        self.plotItem.showGrid(x=False, y=False)
        self.layout = self.plot_item.layout
        self.scene_ = self.plot_item.scene()
        self.scene_.sigMouseClicked.connect(self._clicked)
        self.viewbox = self.plot_item.vb

        self.viewbox.menu.removeAction(self.viewbox.menu.viewAll)
        for ax in self.viewbox.menu.axes:
            self.viewbox.menu.removeAction(ax.menuAction())
        self.plot_item.setMenuEnabled(False, None)

        self.common_axis_items = set()
        self.common_axis_label = ""
        self.common_viewbox = pg.ViewBox(enableMenu=False)
        self.scene_.addItem(self.common_viewbox)
        self.common_viewbox.setXLink(self.viewbox)

        axis = self.layout.itemAt(3, 1)
        axis.setParent(None)
        self.x_axis = FormatedAxis("bottom")
        self.layout.removeItem(self.x_axis)
        self.layout.addItem(self.x_axis, 3, 1)
        self.x_axis.linkToView(axis.linkedView())
        self.plot_item.axes["bottom"]["item"] = self.x_axis
        if x_axis == "time":
            fmt = self._settings.value("plot_xaxis")
            if fmt == "seconds":
                fmt = "phys"
        else:
            fmt = "phys"
        self.x_axis.format = fmt
        self.x_axis.origin = origin

        axis = self.layout.itemAt(2, 0)
        axis.setParent(None)
        self.y_axis = FormatedAxis("left")
        self.y_axis.setWidth(48)
        self.layout.removeItem(axis)
        self.layout.addItem(self.y_axis, 2, 0)
        self.y_axis.linkToView(axis.linkedView())
        self.plot_item.axes["left"]["item"] = self.y_axis

        self.cursor_hint = pg.PlotDataItem(
            [],
            [],
            pen="#000000",
            symbolBrush="#000000",
            symbolPen="w",
            symbol="s",
            symbolSize=8,
        )
        self.viewbox.addItem(self.cursor_hint)

        self.view_boxes = []
        self.curves = []

        self._prev_geometry = self.viewbox.sceneBoundingRect()

        self.resizeEvent = self._resizeEvent

        self._uuid_map = {}

        self._enabled_changed_signals = []
        self._enable_timer = QtCore.QTimer()
        self._enable_timer.setSingleShot(True)
        self._enable_timer.timeout.connect(self._signals_enabled_changed_handler)

        if signals:
            self.add_new_channels(signals)

        self.viewbox.sigXRangeChanged.connect(self.xrange_changed.emit)

        self.keyboard_events = set(
            [
                (QtCore.Qt.Key_C, QtCore.Qt.NoModifier),
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
                (QtCore.Qt.Key_H, QtCore.Qt.NoModifier),
                (QtCore.Qt.Key_Insert, QtCore.Qt.NoModifier),
            ]
        )

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

    def update_signal_curve(self, signal, signal_index, bounds=False):
        sig = signal
        color = sig.color
        t = sig.plot_timestamps
        plot_samples = sig.plot_samples
        pen = sig.pen

        curve = self.curves[signal_index]

        if not isinstance(curve, self.curvetype):
            curve = self.curvetype(
                t,
                plot_samples,
                pen=pen,
                symbolBrush=color,
                symbolPen=color,
                symbol="o",
                symbolSize=4,
                clickable=True,
                mouseWidth=30,
                dynamicRangeLimit=None,
                stepMode=self.line_interconnect,
            )
            if self.with_dots:
                curve.curve.setClickable(True, 30)

            curve.sigClicked.connect(partial(self.curve_clicked.emit, signal.uuid))

            self.view_boxes[signal_index].removeItem(self.curves[signal_index])

            self.curves[signal_index] = curve

            self.view_boxes[signal_index].addItem(curve)
        else:
            if (
                self.with_dots
                or self.line_interconnect != curve.opts["stepMode"]
                or not t.size
            ):
                curve.setData(
                    x=t,
                    y=plot_samples,
                    stepMode=self.line_interconnect,
                    symbolBrush=color,
                    symbolPen=color,
                    pen=pen,
                    skipFiniteCheck=True,
                )
                curve.update()
            else:
                sig_min = sig.min
                sig_max = sig.max

                if curve._boundsCache[0] is not None:
                    curve._boundsCache[0][1] = (t[0], t[-1])
                    curve._boundsCache[1][1] = (sig_min, sig_max)

                else:
                    curve._boundsCache = [
                        [(1.0, None), (t[0], t[-1])],
                        [(1.0, None), (sig_min, sig_max)],
                    ]

                curve.xData = t
                curve.yData = plot_samples
                curve.path = None
                if bounds:
                    curve._boundingRect = QtCore.QRectF(
                        t[0], sig_min, t[-1] - t[0], sig_max - sig_min
                    )
                else:
                    curve._boundingRect = None
                curve.opts["pen"] = pen
                curve.prepareGeometryChange()
                curve.update()

    def update_lines(self, bounds=False):
        self.curvetype = pg.PlotDataItem if self.with_dots else pg.PlotCurveItem

        if self.curves:
            for sig in self.signals:
                _, signal_index = self.signal_by_uuid(sig.uuid)

                if not sig.enable:
                    self.curves[signal_index].hide()
                else:
                    self.update_signal_curve(sig, signal_index, bounds=bounds)
                    self.curves[signal_index].show()

    def set_color(self, uuid, color):
        sig, index = self.signal_by_uuid(uuid)
        curve = self.curves[index]
        sig.color = color

        if sig.mode == "raw":
            style = QtCore.Qt.DashLine
        else:
            style = QtCore.Qt.SolidLine

        sig.pen = pg.mkPen(color=color, style=style)
        curve.setPen(sig.pen)
        curve.setBrush(color)

        if self.curvetype == pg.PlotDataItem:
            curve.curve.setPen(sig.pen)
            curve.curve.setBrush(color)
            curve.scatter.setPen(sig.pen)
            curve.scatter.setBrush(color)

            if sig.individual_axis:
                self.axes[index].set_pen(sig.pen)
                self.axes[index].setTextPen(sig.pen)

        if uuid == self.current_uuid:
            self.y_axis.set_pen(sig.pen)
            self.y_axis.setTextPen(sig.pen)

    def set_unit(self, uuid, unit):
        sig, index = self.signal_by_uuid(uuid)
        sig.unit = unit

        sig_axis = [self.axes[index]]

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

    def set_name(self, uuid, name):
        sig, index = self.signal_by_uuid(uuid)
        sig.name = name

        sig_axis = [self.axes[index]]

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

    def set_common_axis(self, uuid, state):
        _, index = self.signal_by_uuid(uuid)
        viewbox = self.view_boxes[index]
        if state in (QtCore.Qt.Checked, True, 1):
            viewbox.setYRange(*self.common_viewbox.viewRange()[1], padding=0)
            viewbox.setYLink(self.common_viewbox)
            self.common_axis_items.add(uuid)
        else:
            self.view_boxes[index].setYLink(None)
            self.common_axis_items.remove(uuid)

        self.common_axis_label = ", ".join(
            self.signal_by_uuid(uuid)[0].name for uuid in self.common_axis_items
        )

        self.set_current_uuid(self.current_uuid, True)

    def set_individual_axis(self, uuid, state):

        _, index = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.Checked, True, 1):
            if self.signals[index].enable:
                self.axes[index].show()
            self.signals[index].individual_axis = True
        else:
            self.axes[index].hide()
            self.signals[index].individual_axis = False

    def set_signal_enable(self, uuid, state):

        sig, index = self.signal_by_uuid(uuid)

        if state in (QtCore.Qt.Checked, True, 1):
            (start, stop), _ = self.viewbox.viewRange()
            width = self.width() - self.y_axis.width()
            signal = self.signals[index]

            signal.enable = True
            signal.trim(start, stop, width)
            self.view_boxes[index].setXLink(self.viewbox)
            if signal.individual_axis:
                self.axes[index].show()

            uuids = self._timebase_db.setdefault(id(sig.timestamps), set())
            uuids.add(sig.uuid)

        else:
            self.signals[index].enable = False
            self.view_boxes[index].setXLink(None)
            self.axes[index].hide()

            try:
                self._timebase_db[id(sig.timestamps)].remove(uuid)

                if len(self._timebase_db[id(sig.timestamps)]) == 0:
                    del self._timebase_db[id(sig.timestamps)]
            except:
                pass

        self._enable_timer.start(50)

    def _signals_enabled_changed_handler(self):
        self._compute_all_timebase()
        self.update_lines()
        if self.cursor1:
            self.cursor_move_finished.emit()

    def update_views(self):
        geometry = self.viewbox.sceneBoundingRect()
        if geometry != self._prev_geometry:
            for view_box in self.view_boxes:
                view_box.setGeometry(geometry)
            self._prev_geometry = geometry

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

        if (key, modifier) in self.disabled_keys:
            super().keyPressEvent(event)
        else:
            if key == QtCore.Qt.Key_C and modifier == QtCore.Qt.NoModifier:
                if self.cursor1 is None:
                    start, stop = self.viewbox.viewRange()[0]
                    self.cursor1 = Cursor(
                        self.cursor_unit, pos=0, angle=90, movable=True
                    )
                    self.plotItem.addItem(self.cursor1, ignoreBounds=True)
                    self.cursor1.sigPositionChanged.connect(self.cursor_moved.emit)
                    self.cursor1.sigPositionChangeFinished.connect(
                        self.cursor_move_finished.emit
                    )
                    self.cursor1.setPos((start + stop) / 2)
                    self.cursor_move_finished.emit()

                    if self.region is not None:
                        self.cursor1.hide()

                else:
                    self.plotItem.removeItem(self.cursor1)
                    self.cursor1.setParent(None)
                    self.cursor1 = None
                    self.cursor_removed.emit()

            elif key == QtCore.Qt.Key_Y and modifier == QtCore.Qt.NoModifier:
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

            elif key == QtCore.Qt.Key_F and modifier == QtCore.Qt.NoModifier:
                if self.common_axis_items:
                    if any(
                        len(self.signal_by_uuid(uuid)[0].plot_samples)
                        for uuid in self.common_axis_items
                        if self.signal_by_uuid(uuid)[0].enable
                    ):
                        with_common_axis = True

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
                    else:
                        with_common_axis = False

                for i, (viewbox, signal) in enumerate(
                    zip(self.view_boxes, self.signals)
                ):
                    if len(signal.plot_samples):
                        if signal.uuid in self.common_axis_items:
                            if with_common_axis:
                                min_ = common_min
                                max_ = common_max
                                with_common_axis = False
                            else:
                                continue
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

                        viewbox.setYRange(min_, max_, padding=0)

                if self.cursor1:
                    self.cursor_moved.emit()

            elif key == QtCore.Qt.Key_F and modifier == QtCore.Qt.ShiftModifier:
                parent = self.parent().parent()
                uuids = [
                    parent.channel_selection.itemWidget(item, 1).uuid
                    for item in parent.channel_selection.selectedItems()
                    if isinstance(item, ChannelsTreeItem)
                ]
                uuids = set(uuids)

                if not uuids:
                    return

                for i, (viewbox, signal) in enumerate(
                    zip(self.view_boxes, self.signals)
                ):
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

                        viewbox.setYRange(min_, max_, padding=0)

                if self.cursor1:
                    self.cursor_moved.emit()

            elif key == QtCore.Qt.Key_G and modifier == QtCore.Qt.NoModifier:
                y = self.plotItem.ctrl.yGridCheck.isChecked()
                x = self.plotItem.ctrl.xGridCheck.isChecked()

                if x and y:
                    self.plotItem.showGrid(x=False, y=False)
                elif x:
                    self.plotItem.showGrid(x=True, y=True)
                else:
                    self.plotItem.showGrid(x=True, y=False)

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

                    self.region = pg.LinearRegionItem((0, 0))
                    self.region.setZValue(-10)
                    self.plotItem.addItem(self.region)
                    self.region.sigRegionChanged.connect(self.range_modified.emit)
                    self.region.sigRegionChangeFinished.connect(
                        self.range_modified_finished_handler
                    )
                    for line in self.region.lines:
                        line.addMarker("^", 0)
                        line.addMarker("v", 1)
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

            elif key == QtCore.Qt.Key_S and modifier == QtCore.Qt.NoModifier:

                parent = self.parent().parent()
                uuids = []

                iterator = QtWidgets.QTreeWidgetItemIterator(parent.channel_selection)
                while iterator.value():
                    item = iterator.value()
                    if isinstance(item, ChannelsTreeItem):
                        uuids.append(parent.channel_selection.itemWidget(item, 1).uuid)

                    iterator += 1

                uuids = reversed(uuids)

                count = sum(
                    1
                    for i, (sig, curve) in enumerate(zip(self.signals, self.curves))
                    if sig.min != "n.a."
                    and curve.isVisible()
                    and sig.uuid not in self.common_axis_items
                )

                if any(
                    sig.min != "n.a."
                    and curve.isVisible()
                    and sig.uuid in self.common_axis_items
                    for (sig, curve) in zip(self.signals, self.curves)
                ):
                    count += 1
                    with_common_axis = True
                else:
                    with_common_axis = False

                if count:

                    position = 0
                    for uuid in uuids:
                        signal, index = self.signal_by_uuid(uuid)
                        viewbox = self.view_boxes[index]

                        if not signal.empty and signal.enable:
                            if (
                                with_common_axis
                                and signal.uuid in self.common_axis_items
                            ):
                                with_common_axis = False

                                min_ = np.nanmin(
                                    [
                                        np.nanmin(
                                            self.signal_by_uuid(uuid)[0].plot_samples
                                        )
                                        for uuid in self.common_axis_items
                                        if len(
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
                                        if len(
                                            self.signal_by_uuid(uuid)[0].plot_samples
                                        )
                                        and self.signal_by_uuid(uuid)[0].enable
                                    ]
                                )

                            else:
                                if signal.uuid in self.common_axis_items:
                                    continue
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

                            viewbox.setYRange(min_, max_, padding=0)

                            position += 1

                else:
                    xrange, _ = self.viewbox.viewRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.setXRange(*xrange, padding=0)
                    self.viewbox.disableAutoRange()
                if self.cursor1:
                    self.cursor_moved.emit()

            elif key == QtCore.Qt.Key_S and modifier == QtCore.Qt.ShiftModifier:

                parent = self.parent().parent()
                uuids = [
                    parent.channel_selection.itemWidget(item, 1).uuid
                    for item in parent.channel_selection.selectedItems()
                    if isinstance(item, ChannelsTreeItem)
                ]
                uuids = list(reversed(uuids))
                uuids_set = set(uuids)

                if not uuids:
                    return

                count = sum(
                    1
                    for i, (sig, curve) in enumerate(zip(self.signals, self.curves))
                    if sig.uuid in uuids_set and sig.min != "n.a." and curve.isVisible()
                )

                if count:

                    position = 0
                    for uuid in uuids:
                        signal, index = self.signal_by_uuid(uuid)
                        viewbox = self.view_boxes[index]

                        if not signal.empty and signal.enable:

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

                            viewbox.setYRange(min_, max_, padding=0)

                            position += 1

                else:
                    xrange, _ = self.viewbox.viewRange()
                    self.viewbox.autoRange(padding=0)
                    self.viewbox.setXRange(*xrange, padding=0)
                    self.viewbox.disableAutoRange()
                if self.cursor1:
                    self.cursor_moved.emit()

            elif (
                key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right)
                and modifier == QtCore.Qt.NoModifier
            ):
                if self.cursor1:
                    prev_pos = pos = self.cursor1.value()
                    x = self.get_current_timebase()
                    dim = x.size
                    if dim:
                        pos = np.searchsorted(x, pos)
                        if key == QtCore.Qt.Key_Right:
                            pos += 1
                        else:
                            pos -= 1
                        pos = np.clip(pos, 0, dim - 1)
                        pos = x[pos]
                    else:
                        if key == QtCore.Qt.Key_Right:
                            pos += 1
                        else:
                            pos -= 1

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
                    else:
                        delta = 0

                    self.cursor1.set_value(pos)

            elif key == QtCore.Qt.Key_H and modifier == QtCore.Qt.NoModifier:
                if len(self.all_timebase):
                    start_ts = np.amin(self.all_timebase)
                    stop_ts = np.amax(self.all_timebase)

                    self.viewbox.setXRange(start_ts, stop_ts)
                    event_ = QtGui.QKeyEvent(
                        QtCore.QEvent.KeyPress, QtCore.Qt.Key_F, QtCore.Qt.NoModifier
                    )
                    self.keyPressEvent(event_)

                    if self.cursor1:
                        self.cursor_moved.emit()

            elif key == QtCore.Qt.Key_Insert and modifier == QtCore.Qt.NoModifier:
                self.insert_computation()

            else:
                self.parent().keyPressEvent(event)

    def range_modified_finished_handler(self):
        if self.region_lock is not None:
            for i in range(2):
                if self.region.lines[i].value() == self.region_lock:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.DashDotDotLine)
                else:
                    self.region.lines[i].pen.setStyle(QtCore.Qt.SolidLine)
        self.range_modified_finished.emit()

    def trim(self, signals=None):
        signals = signals or self.signals
        if not self._can_trim:
            return
        (start, stop), _ = self.viewbox.viewRange()

        width = self.width() - self.y_axis.width()

        for sig in signals:
            if sig.enable:
                # sig.trim_orig(start, stop, width)
                try:
                    sig.trim(start, stop, width)
                except:
                    sig.trim_python(start, stop, width)

    def xrange_changed_handle(self):
        self.trim()
        self.update_lines()

    def _resizeEvent(self, ev):

        new_size, last_size = self.geometry(), self._last_size
        if new_size != last_size:
            self._last_size = new_size
            self.xrange_changed_handle()
            super().resizeEvent(ev)

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
                for sig_, vbox in zip(self.signals, self.view_boxes):
                    if sig_.uuid not in self.common_axis_items:
                        vbox.setYLink(None)

                vbox = self.view_boxes[index]
                viewbox.setYRange(*vbox.viewRange()[1], padding=0)
                self.common_viewbox.setYRange(*vbox.viewRange()[1], padding=0)
                self.common_viewbox.setYLink(viewbox)

                if self._settings.value("plot_background") == "Black":
                    axis.set_pen(fn.mkPen("#FFFFFF"))
                    axis.setTextPen("#FFFFFF")
                else:
                    axis.set_pen(fn.mkPen("#000000"))
                    axis.setTextPen("#000000")
                axis.setLabel(self.common_axis_label)

        else:
            self.common_viewbox.setYLink(None)
            for sig_, vbox in zip(self.signals, self.view_boxes):
                if sig_.uuid not in self.common_axis_items:
                    vbox.setYLink(None)

            viewbox.setYRange(*self.view_boxes[index].viewRange()[1], padding=0)
            self.view_boxes[index].setYLink(viewbox)
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
        # axis.setWidth()

    def _clicked(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        pos = self.plot_item.vb.mapSceneToView(event.scenePos()).x()
        start, stop = self.viewbox.viewRange()[0]
        if not start <= pos <= stop:
            return

        if (QtCore.Qt.Key_C, QtCore.Qt.NoModifier) not in self.disabled_keys:

            if self.region is None:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    pos = self.plot_item.vb.mapSceneToView(event.scenePos())

                    if self.cursor1 is not None:
                        self.plotItem.removeItem(self.cursor1)
                        self.cursor1.setParent(None)
                        self.cursor1 = None

                    self.cursor1 = Cursor(
                        self.cursor_unit, pos=pos, angle=90, movable=True
                    )
                    self.plotItem.addItem(self.cursor1, ignoreBounds=True)
                    self.cursor1.sigPositionChanged.connect(self.cursor_moved.emit)
                    self.cursor1.sigPositionChangeFinished.connect(
                        self.cursor_move_finished.emit
                    )
                    self.cursor_move_finished.emit()

            else:
                pos = self.plot_item.vb.mapSceneToView(event.scenePos())
                start, stop = self.region.getRegion()

                if self.region_lock is not None:
                    self.region.setRegion((self.region_lock, pos.x()))
                else:
                    if modifiers == QtCore.Qt.ControlModifier:
                        self.region.setRegion((start, pos.x()))
                    else:
                        self.region.setRegion((pos.x(), stop))

    def add_new_channels(self, channels, computed=False):

        geometry = self.viewbox.sceneBoundingRect()
        initial_index = len(self.signals)

        for sig in channels:
            if not hasattr(sig, "computed"):
                sig.computed = computed
                if not computed:
                    sig.computation = {}

        (start, stop), _ = self.viewbox.viewRange()

        width = self.width() - self.y_axis.width()
        trim_info = start, stop, width

        channels = [
            PlotSignal(sig, i, trim_info=trim_info)
            for i, sig in enumerate(channels, len(self.signals))
        ]

        for sig in channels:
            uuids = self._timebase_db.setdefault(id(sig.timestamps), set())
            uuids.add(sig.uuid)
        self.signals.extend(channels)

        self._uuid_map = {sig.uuid: (sig, i) for i, sig in enumerate(self.signals)}

        self._compute_all_timebase()

        if initial_index == 0 and len(self.all_timebase):
            start_t, stop_t = np.amin(self.all_timebase), np.amax(self.all_timebase)
            self.viewbox.setXRange(start_t, stop_t)

        axis_uuid = None

        for index, sig in enumerate(channels, initial_index):

            axis = FormatedAxis(
                "left",  # "right",
                pen=sig.pen,
                textPen=sig.pen,
                text=sig.name if len(sig.name) <= 32 else "{sig.name[:29]}...",
                units=sig.unit,
            )
            if sig.conversion and hasattr(sig.conversion, "text_0"):
                axis.text_conversion = sig.conversion

            view_box = pg.ViewBox(enableMenu=False)
            view_box.setGeometry(geometry)
            view_box.disableAutoRange()

            axis.linkToView(view_box)
            #            if len(sig.name) <= 32:
            #                axis.labelText = sig.name
            #            else:
            #                axis.labelText = f"{sig.name[:29]}..."
            #            axis.labelUnits = sig.unit
            #            axis.labelStyle = {"color": color}
            #
            #            axis.setLabel(axis.labelText, sig.unit, color=color)

            self.layout.addItem(axis, 2, self._axes_layout_pos)
            self._axes_layout_pos += 1

            # self.layout.addItem(view_box, 2, 1)
            self.scene_.addItem(view_box)

            t = sig.plot_timestamps

            curve = self.curvetype(
                t,
                sig.plot_samples,
                pen=sig.pen,
                symbolBrush=sig.color,
                symbolPen=sig.pen,
                symbol="o",
                symbolSize=4,
                clickable=True,
                mouseWidth=30,
                dynamicRangeLimit=None,
                stepMode=self.line_interconnect,
                #                connect='finite',
            )
            curve.hide()
            if self.with_dots:
                curve.curve.setClickable(True, 30)

            curve.sigClicked.connect(partial(self.curve_clicked.emit, sig.uuid))

            self.view_boxes.append(view_box)
            self.curves.append(curve)
            if not sig.empty:
                view_box.setYRange(sig.min, sig.max, padding=0, update=True)

            #            (start, stop), _ = self.viewbox.viewRange()
            #            view_box.setXRange(start, stop, padding=0, update=True)

            self.axes.append(axis)
            axis.hide()
            view_box.addItem(curve)

            if initial_index == 0 and index == 0:
                axis_uuid = sig.uuid

        for index, sig in enumerate(channels, initial_index):
            self.view_boxes[index].setXLink(self.viewbox)

        for curve in self.curves[initial_index:]:
            curve.show()

        if axis_uuid is not None:
            self.set_current_uuid(sig.uuid)

        return channels

    def _compute_all_timebase(self):
        if self._timebase_db:
            timebases = [
                sig.timestamps
                for sig in self.signals
                if id(sig.timestamps) in self._timebase_db
            ]
            if timebases:
                try:
                    new_timebase = np.unique(np.concatenate(timebases))
                except MemoryError:
                    new_timebase = reduce(np.union1d, timebases)
            else:
                new_timebase = np.array([])
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

        needs_timebase_compute = False

        indexes = sorted(
            [(self.signal_by_uuid(uuid)[1], uuid) for uuid in deleted], reverse=True
        )

        for i, uuid in indexes:
            item = self.curves.pop(i)
            item.sigClicked.disconnect()
            item.hide()
            item.setParent(None)
            self.view_boxes[i].removeItem(item)

            item = self.axes.pop(i)
            self.layout.removeItem(item)
            item.scene().removeItem(item)
            item.unlinkFromView()

            item = self.view_boxes.pop(i)
            item.setXLink(None)
            item.setYLink(None)
            self.plotItem.scene().removeItem(item)
            self.layout.removeItem(item)

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

        self.xrange_changed_handle()

        self._compute_all_timebase()

        self.xrange_changed_handle()

    def insert_computation(self, name=""):
        dlg = DefineChannel(self.signals, self.all_timebase, name, self.mdf, self)
        dlg.setModal(True)
        dlg.exec_()
        sig = dlg.result

        if sig is not None:
            sig.uuid = os.urandom(6).hex()
            sig.group_index = -1
            sig.channel_index = -1
            sig.mdf_uuid = os.urandom(6).hex()
            self.add_new_channels([sig], computed=True)
            self.computation_channel_inserted.emit()


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

# -*- coding: utf-8 -*-
""" asammdf *Signal* class module for time correct signal processing """

import logging
from textwrap import fill

import numpy as np

from .blocks.utils import MdfException, extract_cncomment_xml, SignalSource
from .blocks.conversion_utils import from_dict
from .blocks import v2_v3_blocks as v3b
from .blocks import v4_blocks as v4b

from .version import __version__

logger = logging.getLogger("asammdf")


class Signal(object):
    """
    The *Signal* represents a channel described by it's samples and timestamps.
    It can perform arithmetic operations against other *Signal* or numeric types.
    The operations are computed in respect to the timestamps (time correct).
    The non-float signals are not interpolated, instead the last value relative
    to the current timestamp is used.
    *samples*, *timestamps* and *name* are mandatory arguments.

    Parameters
    ----------
    samples : numpy.array | list | tuple
        signal samples
    timestamps : numpy.array | list | tuple
        signal timestamps
    unit : str
        signal unit
    name : str
        signal name
    conversion : dict | channel conversion block
        dict that contains extra conversion information about the signal ,
        default *None*
    comment : str
        signal comment, default ''
    raw : bool
        signal samples are raw values, with no physical conversion applied
    master_metadata : list
        master name and sync type
    display_name : str
        display name used by mdf version 3
    attachment : bytes, name
        channel attachment and name from MDF version 4
    source : SignalSource
        source information named tuple
    bit_count : int
        bit count; useful for integer channels
    stream_sync : bool
        the channel is a synchronisation for the attachment stream (mdf v4 only)
    invalidation_bits : numpy.array | None
        channel invalidation bits, default *None*
    encoding : str | None
        encoding for string signals; default *None*

    """

    def __init__(
        self,
        samples=None,
        timestamps=None,
        unit="",
        name="",
        conversion=None,
        comment="",
        raw=True,
        master_metadata=None,
        display_name="",
        attachment=(),
        source=None,
        bit_count=None,
        stream_sync=False,
        invalidation_bits=None,
        encoding=None,
    ):

        if samples is None or timestamps is None or not name:
            message = (
                '"samples", "timestamps" and "name" are mandatory '
                "for Signal class __init__: samples={samples}\n"
                "timestamps={timestamps}\nname={name}"
            )
            raise MdfException(message)
        else:
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples)
            if not isinstance(timestamps, np.ndarray):
                timestamps = np.array(timestamps, dtype=np.float64)
            if samples.shape[0] != timestamps.shape[0]:
                message = "{} samples and timestamps length mismatch ({} vs {})"
                message = message.format(name, samples.shape[0], timestamps.shape[0])
                logger.exception(message)
                raise MdfException(message)

            self.samples = samples
            self.timestamps = timestamps
            self.unit = unit
            self.name = name
            self.comment = comment
            self._plot_axis = None
            self.raw = raw
            self.master_metadata = master_metadata
            self.display_name = display_name
            self.attachment = attachment
            self.encoding = encoding
            self.group_index = -1
            self.channel_index = -1

            if source:
                if not isinstance(source, SignalSource):
                    source = source.to_common_source()
            self.source = source

            if bit_count is None:
                self.bit_count = samples.dtype.itemsize * 8
            else:
                self.bit_count = bit_count

            self.stream_sync = stream_sync

            if invalidation_bits is not None:
                if not isinstance(
                    invalidation_bits, np.ndarray
                ):
                    invalidation_bits = np.array(invalidation_bits)
                if invalidation_bits.shape[0] != samples.shape[0]:
                    message = "{} samples and invalidation bits length mismatch ({} vs {})"
                    message = message.format(name, samples.shape[0], invalidation_bits.shape[0])
                    logger.exception(message)
                    raise MdfException(message)
            self.invalidation_bits = invalidation_bits

            if conversion:
                if not isinstance(
                    conversion, (v4b.ChannelConversion, v3b.ChannelConversion)
                ):
                    conversion = from_dict(conversion)

            self.conversion = conversion

    def __repr__(self):
        return f"""<Signal {self.name}:
\tsamples={self.samples}
\ttimestamps={self.timestamps}
\tinvalidation_bits={self.invalidation_bits}
\tunit="{self.unit}"
\tconversion={self.conversion}
\tsource={self.source}
\tcomment="{self.comment}"
\tmastermeta="{self.master_metadata}"
\traw={self.raw}
\tdisplay_name={self.display_name}
\tattachment={self.attachment}>
"""

    def plot(self, validate=True):
        """ plot Signal samples. Pyqtgraph is used if it is available; in this
        case see the GUI plot documentation to see the available commands"""
        try:

            from .gui.plot import plot

            plot(self, validate=True)
            return

        except:
            raise
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import axes3d
                from matplotlib.widgets import Slider
            except ImportError:
                logging.warning("Signal plotting requires pyqtgraph or matplotlib")
                return

        if len(self.samples.shape) <= 1 and self.samples.dtype.names is None:
            fig = plt.figure()
            fig.canvas.set_window_title(self.name)
            fig.text(
                0.95,
                0.05,
                f"asammdf {__version__}",
                fontsize=8,
                color="red",
                ha="right",
                va="top",
                alpha=0.5,
            )

            name = self.name

            if self.comment:
                comment = self.comment.replace("$", "")
                comment = extract_cncomment_xml(comment)
                comment = fill(comment, 120).replace("\\n", " ")

                title = f"{name}\n({comment})"
                plt.title(title)
            else:
                plt.title(name)
            try:
                if not self.master_metadata:
                    plt.xlabel("Time [s]")
                    plt.ylabel(f"[{self.unit}]")
                    plt.plot(self.timestamps, self.samples, "b")
                    plt.plot(self.timestamps, self.samples, "b.")
                    plt.grid(True)
                    plt.show()
                else:
                    master_name, sync_type = self.master_metadata
                    if sync_type in (0, 1):
                        plt.xlabel(f"{master_name} [s]")
                    elif sync_type == 2:
                        plt.xlabel(f"{master_name} [deg]")
                    elif sync_type == 3:
                        plt.xlabel(f"{master_name} [m]")
                    elif sync_type == 4:
                        plt.xlabel(f"{master_name} [index]")
                    plt.ylabel(f"[{self.unit}]")
                    plt.plot(self.timestamps, self.samples, "b")
                    plt.plot(self.timestamps, self.samples, "b.")
                    plt.grid(True)
                    plt.show()
            except ValueError:
                plt.close(fig)
        else:
            try:
                names = self.samples.dtype.names
                if self.samples.dtype.names is None or len(names) == 1:

                    if names:
                        samples = self.samples[names[0]]
                    else:
                        samples = self.samples

                    shape = samples.shape[1:]

                    fig = plt.figure()
                    fig.canvas.set_window_title(self.name)
                    fig.text(
                        0.95,
                        0.05,
                        f"asammdf {__version__}",
                        fontsize=8,
                        color="red",
                        ha="right",
                        va="top",
                        alpha=0.5,
                    )

                    if self.comment:
                        comment = self.comment.replace("$", "")
                        plt.title(f"{self.name}\n({comment})")
                    else:
                        plt.title(self.name)

                    ax = fig.add_subplot(111, projection="3d")

                    # Grab some test data.
                    X = np.array(range(shape[1]))
                    Y = np.array(range(shape[0]))
                    X, Y = np.meshgrid(X, Y)

                    Z = samples[0]

                    # Plot a basic wireframe.
                    self._plot_axis = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

                    # Place Sliders on Graph
                    ax_a = plt.axes([0.25, 0.1, 0.65, 0.03])

                    # Create Sliders & Determine Range
                    sa = Slider(
                        ax_a,
                        "Time [s]",
                        self.timestamps[0],
                        self.timestamps[-1],
                        valinit=self.timestamps[0],
                    )

                    def update(val):
                        self._plot_axis.remove()
                        idx = np.searchsorted(self.timestamps, sa.val, side="right")
                        Z = samples[idx - 1]
                        self._plot_axis = ax.plot_wireframe(
                            X, Y, Z, rstride=1, cstride=1
                        )
                        fig.canvas.draw_idle()

                    sa.on_changed(update)

                    plt.show()

                else:
                    fig = plt.figure()
                    fig.canvas.set_window_title(self.name)
                    fig.text(
                        0.95,
                        0.05,
                        f"asammdf {__version__}",
                        fontsize=8,
                        color="red",
                        ha="right",
                        va="top",
                        alpha=0.5,
                    )

                    if self.comment:
                        comment = self.comment.replace("$", "")
                        plt.title(f"{self.name}\n({comment})")
                    else:
                        plt.title(self.name)

                    ax = fig.add_subplot(111, projection="3d")

                    samples = self.samples[names[0]]
                    axis1 = self.samples[names[1]]
                    axis2 = self.samples[names[2]]

                    # Grab some test data.
                    X, Y = np.meshgrid(axis2[0], axis1[0])

                    Z = samples[0]

                    # Plot a basic wireframe.
                    self._plot_axis = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

                    # Place Sliders on Graph
                    ax_a = plt.axes([0.25, 0.1, 0.65, 0.03])

                    # Create Sliders & Determine Range
                    sa = Slider(
                        ax_a,
                        "Time [s]",
                        self.timestamps[0],
                        self.timestamps[-1],
                        valinit=self.timestamps[0],
                    )

                    def update(val):
                        self._plot_axis.remove()
                        idx = np.searchsorted(self.timestamps, sa.val, side="right")
                        Z = samples[idx - 1]
                        X, Y = np.meshgrid(axis2[idx - 1], axis1[idx - 1])
                        self._plot_axis = ax.plot_wireframe(
                            X, Y, Z, rstride=1, cstride=1
                        )
                        fig.canvas.draw_idle()

                    sa.on_changed(update)

                    plt.show()

            except Exception as err:
                print(err)

    def cut(self, start=None, stop=None, include_ends=True, interpolation_mode=0):
        """
        Cuts the signal according to the *start* and *stop* values, by using
        the insertion indexes in the signal's *time* axis.

        Parameters
        ----------
        start : float
            start timestamp for cutting
        stop : float
            stop timestamp for cutting
        include_ends : bool
            include the *start* and *stop* timestamps after cutting the signal.
            If *start* and *stop* are found in the original timestamps, then
            the new samples will be computed using interpolation. Default *True*
        interpolation_mode : int
            interpolation mode for integer signals; default 0

                * 0 - repeat previous samples
                * 1 - linear interpolation

        Returns
        -------
        result : Signal
            new *Signal* cut from the original

        Examples
        --------
        >>> new_sig = old_sig.cut(1.0, 10.5)
        >>> new_sig.timestamps[0], new_sig.timestamps[-1]
        0.98, 10.48

        """
        ends = (start, stop)
        if len(self) == 0:
            result = Signal(
                np.array([], dtype=self.samples.dtype),
                np.array([], dtype=self.timestamps.dtype),
                self.unit,
                self.name,
                self.conversion,
                self.comment,
                self.raw,
                self.master_metadata,
                self.display_name,
                self.attachment,
                self.source,
                self.bit_count,
                self.stream_sync,
                encoding=self.encoding,
            )

        elif start is None and stop is None:
            # return the channel uncut
            result = Signal(
                self.samples.copy(),
                self.timestamps.copy(),
                self.unit,
                self.name,
                self.conversion,
                self.comment,
                self.raw,
                self.master_metadata,
                self.display_name,
                self.attachment,
                self.source,
                self.bit_count,
                self.stream_sync,
                invalidation_bits=self.invalidation_bits.copy()
                if self.invalidation_bits is not None
                else None,
                encoding=self.encoding,
            )

        else:
            if start is None:
                # cut from begining to stop
                if stop < self.timestamps[0]:
                    result = Signal(
                        np.array([], dtype=self.samples.dtype),
                        np.array([], dtype=self.timestamps.dtype),
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_name,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        self.stream_sync,
                        encoding=self.encoding,
                    )

                else:
                    stop = np.searchsorted(self.timestamps, stop, side="right")
                    if (
                        include_ends
                        and ends[-1] not in self.timestamps
                        and ends[-1] < self.timestamps[-1]
                    ):
                        interpolated = self.interp(
                            [ends[1]], interpolation_mode=interpolation_mode
                        )
                        samples = np.append(
                            self.samples[:stop], interpolated.samples, axis=0
                        )
                        timestamps = np.append(self.timestamps[:stop], ends[1])
                        if self.invalidation_bits is not None:
                            invalidation_bits = np.append(
                                self.invalidation_bits[:stop],
                                interpolated.invalidation_bits,
                            )
                        else:
                            invalidation_bits = None
                    else:
                        samples = self.samples[:stop].copy()
                        timestamps = self.timestamps[:stop].copy()
                        if self.invalidation_bits is not None:
                            invalidation_bits = self.invalidation_bits[:stop].copy()
                        else:
                            invalidation_bits = None
                    result = Signal(
                        samples,
                        timestamps,
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_name,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        self.stream_sync,
                        invalidation_bits=invalidation_bits,
                        encoding=self.encoding,
                    )

            elif stop is None:
                # cut from start to end
                if start > self.timestamps[-1]:
                    result = Signal(
                        np.array([], dtype=self.samples.dtype),
                        np.array([], dtype=self.timestamps.dtype),
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_name,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        self.stream_sync,
                        encoding=self.encoding,
                    )

                else:
                    start = np.searchsorted(self.timestamps, start, side="left")
                    if (
                        include_ends
                        and ends[0] not in self.timestamps
                        and ends[0] > self.timestamps[0]
                    ):
                        interpolated = self.interp(
                            [ends[0]], interpolation_mode=interpolation_mode
                        )
                        samples = np.append(
                            interpolated.samples, self.samples[start:], axis=0
                        )
                        timestamps = np.append(ends[0], self.timestamps[start:])
                        if self.invalidation_bits is not None:
                            invalidation_bits = np.append(
                                interpolated.invalidation_bits,
                                self.invalidation_bits[start:],
                            )
                        else:
                            invalidation_bits = None
                    else:
                        samples = self.samples[start:].copy()
                        timestamps = self.timestamps[start:].copy()
                        if self.invalidation_bits is not None:
                            invalidation_bits = self.invalidation_bits[start:].copy()
                        else:
                            invalidation_bits = None
                    result = Signal(
                        samples,
                        timestamps,
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_name,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        self.stream_sync,
                        invalidation_bits=invalidation_bits,
                        encoding=self.encoding,
                    )

            else:
                # cut between start and stop
                if start > self.timestamps[-1] or stop < self.timestamps[0]:
                    result = Signal(
                        np.array([], dtype=self.samples.dtype),
                        np.array([], dtype=self.timestamps.dtype),
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_name,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        self.stream_sync,
                        encoding=self.encoding,
                    )
                else:
                    start = np.searchsorted(self.timestamps, start, side="left")
                    stop = np.searchsorted(self.timestamps, stop, side="right")

                    if start == stop:
                        if include_ends:
                            interpolated = self.interp(
                                np.unique(ends), interpolation_mode=interpolation_mode
                            )
                            samples = interpolated.samples
                            timestamps = np.array(
                                np.unique(ends), dtype=self.timestamps.dtype
                            )
                            invalidation_bits = interpolated.invalidation_bits
                        else:
                            samples = np.array([], dtype=self.samples.dtype)
                            timestamps = np.array([], dtype=self.timestamps.dtype)
                            if self.invalidation_bits is not None:
                                invalidation_bits = np.array([], dtype=bool)
                            else:
                                invalidation_bits = None
                    else:
                        samples = self.samples[start:stop].copy()
                        timestamps = self.timestamps[start:stop].copy()
                        if self.invalidation_bits is not None:
                            invalidation_bits = self.invalidation_bits[
                                start:stop
                            ].copy()
                        else:
                            invalidation_bits = None

                        if (
                            include_ends
                            and ends[-1] not in self.timestamps
                            and ends[-1] < self.timestamps[-1]
                        ):
                            interpolated = self.interp(
                                [ends[1]], interpolation_mode=interpolation_mode
                            )
                            samples = np.append(samples, interpolated.samples, axis=0)
                            timestamps = np.append(timestamps, ends[1])
                            if invalidation_bits is not None:
                                invalidation_bits = np.append(
                                    invalidation_bits, interpolated.invalidation_bits
                                )

                        if (
                            include_ends
                            and ends[0] not in self.timestamps
                            and ends[0] > self.timestamps[0]
                        ):
                            interpolated = self.interp(
                                [ends[0]], interpolation_mode=interpolation_mode
                            )
                            samples = np.append(interpolated.samples, samples, axis=0)
                            timestamps = np.append(ends[0], timestamps)

                            if invalidation_bits is not None:
                                invalidation_bits = np.append(
                                    interpolated.invalidation_bits, invalidation_bits
                                )

                    result = Signal(
                        samples,
                        timestamps,
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_name,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        self.stream_sync,
                        invalidation_bits=invalidation_bits,
                        encoding=self.encoding,
                    )

        return result

    def extend(self, other):
        """ extend signal with samples from another signal

        Parameters
        ----------
        other : Signal

        Returns
        -------
        signal : Signal
            new extended *Signal*

        """
        if len(self.timestamps):
            last_stamp = self.timestamps[-1]
        else:
            last_stamp = 0
        if len(other):
            other_first_sample = other.timestamps[0]
            if last_stamp >= other_first_sample:
                timestamps = other.timestamps + last_stamp
            else:
                timestamps = other.timestamps

            if self.invalidation_bits is None and other.invalidation_bits is None:
                invalidation_bits = None
            elif self.invalidation_bits is None and other.invalidation_bits is not None:
                invalidation_bits = np.concatenate(
                    (np.zeros(len(self), dtype=bool), other.invalidation_bits)
                )
            elif self.invalidation_bits is not None and other.invalidation_bits is None:
                invalidation_bits = np.concatenate(
                    (self.invalidation_bits, np.zeros(len(other), dtype=bool))
                )
            else:
                invalidation_bits = np.append(
                    self.invalidation_bits, other.invalidation_bits
                )

            result = Signal(
                np.append(self.samples, other.samples, axis=0),
                np.append(self.timestamps, timestamps),
                self.unit,
                self.name,
                self.conversion,
                self.comment,
                self.raw,
                self.master_metadata,
                self.display_name,
                self.attachment,
                self.source,
                self.bit_count,
                self.stream_sync,
                invalidation_bits=invalidation_bits,
                encoding=self.encoding,
            )
        else:
            result = self

        return result

    def interp(self, new_timestamps, interpolation_mode=0):
        """ returns a new *Signal* interpolated using the *new_timestamps*

        Parameters
        ----------
        new_timestamps : np.array
            timestamps used for interpolation
        interpolation_mode : int
            interpolation mode for integer signals; default 0

                * 0 - repeat previous samples
                * 1 - linear interpolation

        Returns
        -------
        signal : Signal
            new interpolated *Signal*

        """
        if not len(self.samples) or not len(new_timestamps):
            return Signal(
                self.samples[:0].copy(),
                self.timestamps[:0].copy(),
                self.unit,
                self.name,
                comment=self.comment,
                conversion=self.conversion,
                raw=self.raw,
                master_metadata=self.master_metadata,
                display_name=self.display_name,
                attachment=self.attachment,
                stream_sync=self.stream_sync,
                invalidation_bits=None,
                encoding=self.encoding,
            )
        else:

            if len(self.samples.shape) > 1:
                idx = np.searchsorted(self.timestamps, new_timestamps, side="right")
                idx -= 1
                idx = np.clip(idx, 0, idx[-1])
                s = self.samples[idx]

                if self.invalidation_bits is not None:
                    invalidation_bits = self.invalidation_bits[idx]
                else:
                    invalidation_bits = None
            else:

                kind = self.samples.dtype.kind
                if kind == "f":
                    s = np.interp(new_timestamps, self.timestamps, self.samples)

                    if self.invalidation_bits is not None:
                        idx = np.searchsorted(
                            self.timestamps, new_timestamps, side="right"
                        )
                        idx -= 1
                        idx = np.clip(idx, 0, idx[-1])
                        invalidation_bits = self.invalidation_bits[idx]
                    else:
                        invalidation_bits = None
                elif kind in "ui":
                    if interpolation_mode == 1:
                        s = np.interp(
                            new_timestamps, self.timestamps, self.samples
                        ).astype(self.samples.dtype)
                        if self.invalidation_bits is not None:
                            idx = np.searchsorted(
                                self.timestamps, new_timestamps, side="right"
                            )
                            idx -= 1
                            idx = np.clip(idx, 0, idx[-1])
                            invalidation_bits = self.invalidation_bits[idx]
                        else:
                            invalidation_bits = None
                    else:
                        idx = np.searchsorted(
                            self.timestamps, new_timestamps, side="right"
                        )
                        idx -= 1
                        idx = np.clip(idx, 0, idx[-1])
                        s = self.samples[idx]

                        if self.invalidation_bits is not None:
                            invalidation_bits = self.invalidation_bits[idx]
                        else:
                            invalidation_bits = None
                else:
                    idx = np.searchsorted(self.timestamps, new_timestamps, side="right")
                    idx -= 1
                    idx = np.clip(idx, 0, idx[-1])
                    s = self.samples[idx]

                    if self.invalidation_bits is not None:
                        invalidation_bits = self.invalidation_bits[idx]
                    else:
                        invalidation_bits = None

            return Signal(
                s,
                new_timestamps,
                self.unit,
                self.name,
                comment=self.comment,
                conversion=self.conversion,
                source=self.source,
                raw=self.raw,
                master_metadata=self.master_metadata,
                display_name=self.display_name,
                attachment=self.attachment,
                stream_sync=self.stream_sync,
                invalidation_bits=invalidation_bits,
                encoding=self.encoding,
            )

    def __apply_func(self, other, func_name):
        """ delegate operations to the *samples* attribute, but in a time
        correct manner by considering the *timestamps*

        """

        if isinstance(other, Signal):
            if len(self) and len(other):
                start = max(self.timestamps[0], other.timestamps[0])
                stop = min(self.timestamps[-1], other.timestamps[-1])
                s1 = self.cut(start, stop)
                s2 = other.cut(start, stop)
            else:
                s1 = self
                s2 = other
            time = np.union1d(s1.timestamps, s2.timestamps)
            s = self.interp(time).samples
            o = other.interp(time).samples
            func = getattr(s, func_name)
            s = func(o)
        elif other is None:
            s = self.samples
            time = self.timestamps
        else:
            func = getattr(self.samples, func_name)
            s = func(other)
            time = self.timestamps
        return Signal(
            samples=s,
            timestamps=time,
            unit=self.unit,
            name=self.name,
            conversion=self.conversion,
            raw=self.raw,
            master_metadata=self.master_metadata,
            display_name=self.display_name,
            attachment=self.attachment,
            stream_sync=self.stream_sync,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=self.encoding,
        )

    def __pos__(self):
        return self

    def __neg__(self):
        return Signal(
            np.negative(self.samples),
            self.timestamps,
            unit=self.unit,
            name=self.name,
            conversion=self.conversion,
            raw=self.raw,
            master_metadata=self.master_metadata,
            display_name=self.display_name,
            attachment=self.attachment,
            stream_sync=self.stream_sync,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=self.encoding,
        )

    def __round__(self, n):
        return Signal(
            np.around(self.samples, n),
            self.timestamps,
            unit=self.unit,
            name=self.name,
            conversion=self.conversion,
            raw=self.raw,
            master_metadata=self.master_metadata,
            display_name=self.display_name,
            attachment=self.attachment,
            stream_sync=self.stream_sync,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=self.encoding,
        )

    def __sub__(self, other):
        return self.__apply_func(other, "__sub__")

    def __isub__(self, other):
        return self.__sub__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __add__(self, other):
        return self.__apply_func(other, "__add__")

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        return self.__apply_func(other, "__truediv__")

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __rtruediv__(self, other):
        return self.__apply_func(other, "__rtruediv__")

    def __mul__(self, other):
        return self.__apply_func(other, "__mul__")

    def __imul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        return self.__apply_func(other, "__floordiv__")

    def __ifloordiv__(self, other):
        return self.__truediv__(other)

    def __rfloordiv__(self, other):
        return 1 / self.__apply_func(other, "__rfloordiv__")

    def __mod__(self, other):
        return self.__apply_func(other, "__mod__")

    def __pow__(self, other):
        return self.__apply_func(other, "__pow__")

    def __and__(self, other):
        return self.__apply_func(other, "__and__")

    def __or__(self, other):
        return self.__apply_func(other, "__or__")

    def __xor__(self, other):
        return self.__apply_func(other, "__xor__")

    def __invert__(self):
        s = ~self.samples
        time = self.timestamps
        return Signal(
            s,
            time,
            unit=self.unit,
            name=self.name,
            conversion=self.conversion,
            raw=self.raw,
            master_metadata=self.master_metadata,
            display_name=self.display_name,
            attachment=self.attachment,
            stream_sync=self.stream_sync,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=self.encoding,
        )

    def __lshift__(self, other):
        return self.__apply_func(other, "__lshift__")

    def __rshift__(self, other):
        return self.__apply_func(other, "__rshift__")

    def __lt__(self, other):
        return self.__apply_func(other, "__lt__")

    def __le__(self, other):
        return self.__apply_func(other, "__le__")

    def __gt__(self, other):
        return self.__apply_func(other, "__gt__")

    def __ge__(self, other):
        return self.__apply_func(other, "__ge__")

    def __eq__(self, other):
        return self.__apply_func(other, "__eq__")

    def __ne__(self, other):
        return self.__apply_func(other, "__ne__")

    def __iter__(self):
        for item in (self.samples, self.timestamps, self.unit, self.name):
            yield item

    def __reversed__(self):
        return enumerate(zip(reversed(self.samples), reversed(self.timestamps)))

    def __len__(self):
        return len(self.samples)

    def __abs__(self):
        return Signal(
            np.fabs(self.samples),
            self.timestamps,
            unit=self.unit,
            name=self.name,
            conversion=self.conversion,
            raw=self.raw,
            master_metadata=self.master_metadata,
            display_name=self.display_name,
            attachment=self.attachment,
            stream_sync=self.stream_sync,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
        )

    def __getitem__(self, val):
        return self.samples[val]

    def __setitem__(self, idx, val):
        self.samples[idx] = val

    def astype(self, np_type):
        """ returns new *Signal* with samples of dtype *np_type*

        Parameters
        ----------
        np_type : np.dtype
            new numpy dtye

        Returns
        -------
        signal : Signal
            new *Signal* with the samples of *np_type* dtype

        """
        return Signal(
            self.samples.astype(np_type),
            self.timestamps,
            unit=self.unit,
            name=self.name,
            conversion=self.conversion,
            raw=self.raw,
            master_metadata=self.master_metadata,
            display_name=self.display_name,
            attachment=self.attachment,
            stream_sync=self.stream_sync,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=self.encoding,
        )

    def physical(self):
        """
        get the physical samples values

        Returns
        -------
        phys : Signal
            new *Signal* with physical values

        """

        if not self.raw or self.conversion is None:
            samples = self.samples.copy()
            encoding = None
        else:
            samples = self.conversion.convert(self.samples)
            if samples.dtype.kind == 'S':
                encoding = 'utf-8' if self.conversion.id == b'##CC' else 'latin-1'
            else:
                encoding = None

        return Signal(
            samples,
            self.timestamps.copy(),
            unit=self.unit,
            name=self.name,
            conversion=None,
            raw=False,
            master_metadata=self.master_metadata,
            display_name=self.display_name,
            attachment=self.attachment,
            stream_sync=self.stream_sync,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=encoding,
        )

    def validate(self, copy=True):
        """ appply invalidation bits if they are available for this signal

        Parameters
        ----------
        copy (True) : bool
            return a copy of the result

            .. versionadded:: 5.12.0

        """
        if self.invalidation_bits is None:
            signal = self

        else:
            idx = np.nonzero(~self.invalidation_bits)[0]
            signal = Signal(
                self.samples[idx],
                self.timestamps[idx],
                self.unit,
                self.name,
                self.conversion,
                self.comment,
                self.raw,
                self.master_metadata,
                self.display_name,
                self.attachment,
                self.source,
                self.bit_count,
                self.stream_sync,
                invalidation_bits=None,
                encoding=self.encoding,
            )

        if copy:
            signal = signal.copy()

        return signal

    def copy(self):
        """ copy all attributes to a new Signal """
        return Signal(
            self.samples.copy(),
            self.timestamps.copy(),
            self.unit,
            self.name,
            self.conversion,
            self.comment,
            self.raw,
            self.master_metadata,
            self.display_name,
            self.attachment,
            self.source,
            self.bit_count,
            self.stream_sync,
            invalidation_bits=self.invalidation_bits.copy()
            if self.invalidation_bits is not None
            else None,
            encoding=self.encoding,
        )


if __name__ == "__main__":
    pass

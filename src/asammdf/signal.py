""" asammdf *Signal* class module for time correct signal processing """

from __future__ import annotations

from collections.abc import Iterator
import logging
from textwrap import fill
from typing import Any

import numpy as np
from numpy.core.defchararray import encode
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .blocks import v2_v3_blocks as v3b
from .blocks import v4_blocks as v4b
from .blocks.conversion_utils import from_dict
from .blocks.options import FloatInterpolation, IntegerInterpolation
from .blocks.source_utils import Source
from .blocks.utils import extract_xml_comment, MdfException, SignalFlags
from .types import (
    ChannelConversionType,
    FloatInterpolationModeType,
    IntInterpolationModeType,
    SourceType,
    SyncType,
)
from .version import __version__

logger = logging.getLogger("asammdf")


class Signal:
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
    display_names : dict
        display names used by mdf version 3
    attachment : bytes, name
        channel attachment and name from MDF version 4
    source : Source
        source information named tuple
    bit_count : int
        bit count; useful for integer channels
    invalidation_bits : numpy.array | None
        channel invalidation bits, default *None*
    encoding : str | None
        encoding for string signals; default *None*
    flags : Signal.Flags
        flags for user defined attributes and stream sync

    """

    Flags = SignalFlags

    def __init__(
        self,
        samples: ArrayLike | None = None,
        timestamps: ArrayLike | None = None,
        unit: str = "",
        name: str = "",
        conversion: dict[str, Any] | ChannelConversionType | None = None,
        comment: str = "",
        raw: bool = True,
        master_metadata: tuple[str, SyncType] | None = None,
        display_names: dict[str, str] | None = None,
        attachment: tuple[bytes, str | None, str | None] | None = None,
        source: SourceType | None = None,
        bit_count: int | None = None,
        invalidation_bits: ArrayLike | None = None,
        encoding: str | None = None,
        group_index: int = -1,
        channel_index: int = -1,
        flags: Flags = Flags.no_flags,
        virtual_conversion: dict[str, Any] | ChannelConversionType | None = None,
        virtual_master_conversion: dict[str, Any] | ChannelConversionType | None = None,
    ) -> None:
        if samples is None or timestamps is None or not name:
            message = (
                '"samples", "timestamps" and "name" are mandatory '
                f"for Signal class __init__: samples={samples}\n"
                f"timestamps={timestamps}\nname={name}"
            )
            raise MdfException(message)
        else:
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples)
                kind = samples.dtype.kind
                if kind == "U":
                    if encoding is None:
                        encodings = ["utf-8", "latin-1"]
                    else:
                        encodings = [encoding, "utf-8", "latin-1"]
                    for encoding in encodings:
                        try:
                            samples = encode(samples, encoding)
                            break
                        except:
                            continue
                    else:
                        samples = encode(samples, encodings[0], errors="ignore")
                elif kind == "O":
                    samples = samples.astype(bytes)

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
            self.flags = flags
            self._plot_axis = None
            self.raw = raw
            self.master_metadata = master_metadata
            self.display_names = display_names or {}
            self.attachment = attachment
            self.encoding = encoding
            self.group_index = group_index
            self.channel_index = channel_index

            if source:
                if not isinstance(source, Source):
                    source = Source.from_source(source)
            self.source = source

            if bit_count is None:
                self.bit_count = samples.dtype.itemsize * 8
            else:
                self.bit_count = bit_count

            if invalidation_bits is not None:
                if not isinstance(invalidation_bits, np.ndarray):
                    invalidation_bits = np.array(invalidation_bits)
                if invalidation_bits.shape[0] != samples.shape[0]:
                    message = "{} samples and invalidation bits length mismatch ({} vs {})"
                    message = message.format(name, samples.shape[0], invalidation_bits.shape[0])
                    logger.exception(message)
                    raise MdfException(message)
            self.invalidation_bits = invalidation_bits

            if conversion:
                if not isinstance(conversion, (v4b.ChannelConversion, v3b.ChannelConversion)):
                    conversion = from_dict(conversion)

            self.conversion = conversion

            if self.flags & self.Flags.virtual:
                if not isinstance(virtual_conversion, (v4b.ChannelConversion, v3b.ChannelConversion)):
                    conversion = from_dict(virtual_conversion)
                self.virtual_conversion = conversion
            else:
                self.virtual_conversion = None

            if self.flags & self.Flags.virtual_master:
                if not isinstance(virtual_master_conversion, (v4b.ChannelConversion, v3b.ChannelConversion)):
                    conversion = from_dict(virtual_master_conversion)
                self.virtual_master_conversion = conversion
            else:
                self.virtual_master_conversion = None

    def __repr__(self):
        return f"""<Signal {self.name}:
\tsamples={self.samples}
\ttimestamps={self.timestamps}
\tinvalidation_bits={self.invalidation_bits}
\tunit="{self.unit}"
\tconversion={self.conversion}
\tsource={self.source}
\tcomment="{self.comment}"
\tflags="{self.flags}"
\tmastermeta="{self.master_metadata}"
\traw={self.raw}
\tdisplay_names={self.display_names}
\tattachment={self.attachment}>
"""

    def plot(self, validate: bool = True, index_only: bool = False) -> None:
        """plot Signal samples. Pyqtgraph is used if it is available; in this
        case see the GUI plot documentation to see the available commands

        Parameters
        ----------
        validate (True): bool
            apply the invalidation bits

        index_only (False) : bool
            use index based X axis. This can be useful if the master (usually
            time based) is corrupted with NaN, inf or if it is not strictly
            increasing

        """
        try:
            from .gui.plot import plot

            plot(self, validate=True, index_only=False)
            return

        except:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.widgets import Slider
            except ImportError:
                logging.warning("Signal plotting requires pyqtgraph or matplotlib")
                return

        if len(self.samples.shape) <= 1 and self.samples.dtype.names is None:
            fig = plt.figure()
            fig.canvas.manager.set_window_title(self.name)
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
                comment = extract_xml_comment(comment)
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
                        self._plot_axis = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
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
                        self._plot_axis = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
                        fig.canvas.draw_idle()

                    sa.on_changed(update)

                    plt.show()

            except Exception as err:
                print(err)

    def cut(
        self,
        start: float | None = None,
        stop: float | None = None,
        include_ends: bool = True,
        integer_interpolation_mode: (
            IntInterpolationModeType | IntegerInterpolation
        ) = IntegerInterpolation.REPEAT_PREVIOUS_SAMPLE,
        float_interpolation_mode: (
            FloatInterpolationModeType | FloatInterpolation
        ) = FloatInterpolation.LINEAR_INTERPOLATION,
    ) -> Signal:
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

        integer_interpolation_mode : int
            interpolation mode for integer signals; default 0

                * 0 - repeat previous samples
                * 1 - linear interpolation
                * 2 - hybrid interpolation: channels with integer data type (raw values) that have a
                  conversion that outputs float values will use linear interpolation, otherwise
                  the previous sample is used

                .. versionadded:: 6.2.0

        float_interpolation_mode : int
            interpolation mode for float channels; default 1

                * 0 - repeat previous sample
                * 1 - use linear interpolation

                .. versionadded:: 6.2.0

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

        integer_interpolation_mode = IntegerInterpolation(integer_interpolation_mode)
        float_interpolation_mode = FloatInterpolation(float_interpolation_mode)

        original_start, original_stop = (start, stop)

        if self.samples.size == 0:
            result = Signal(
                np.array([], dtype=self.samples.dtype),
                np.array([], dtype=self.timestamps.dtype),
                self.unit,
                self.name,
                self.conversion,
                self.comment,
                self.raw,
                self.master_metadata,
                self.display_names,
                self.attachment,
                self.source,
                self.bit_count,
                encoding=self.encoding,
                group_index=self.group_index,
                channel_index=self.channel_index,
                flags=self.flags,
                virtual_conversion=self.virtual_conversion,
                virtual_master_conversion=self.virtual_master_conversion,
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
                self.display_names,
                self.attachment,
                self.source,
                self.bit_count,
                invalidation_bits=self.invalidation_bits.copy() if self.invalidation_bits is not None else None,
                encoding=self.encoding,
                group_index=self.group_index,
                channel_index=self.channel_index,
                flags=self.flags,
                virtual_conversion=self.virtual_conversion,
                virtual_master_conversion=self.virtual_master_conversion,
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
                        self.display_names,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        encoding=self.encoding,
                        group_index=self.group_index,
                        channel_index=self.channel_index,
                        flags=self.flags,
                        virtual_conversion=self.virtual_conversion,
                        virtual_master_conversion=self.virtual_master_conversion,
                    )

                else:
                    stop = np.searchsorted(self.timestamps, stop, side="right")
                    if include_ends and original_stop not in self.timestamps and original_stop < self.timestamps[-1]:
                        interpolated = self.interp(
                            [original_stop],
                            integer_interpolation_mode=integer_interpolation_mode,
                            float_interpolation_mode=float_interpolation_mode,
                        )

                        if len(interpolated):
                            samples = np.append(self.samples[:stop], interpolated.samples, axis=0)
                            timestamps = np.append(self.timestamps[:stop], interpolated.timestamps)
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

                    if samples.dtype != self.samples.dtype:
                        samples = samples.astype(self.samples.dtype)

                    result = Signal(
                        samples,
                        timestamps,
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_names,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        invalidation_bits=invalidation_bits,
                        encoding=self.encoding,
                        group_index=self.group_index,
                        channel_index=self.channel_index,
                        flags=self.flags,
                        virtual_conversion=self.virtual_conversion,
                        virtual_master_conversion=self.virtual_master_conversion,
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
                        self.display_names,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        encoding=self.encoding,
                        group_index=self.group_index,
                        channel_index=self.channel_index,
                        flags=self.flags,
                        virtual_conversion=self.virtual_conversion,
                        virtual_master_conversion=self.virtual_master_conversion,
                    )

                else:
                    start = np.searchsorted(self.timestamps, start, side="left")
                    if include_ends and original_start not in self.timestamps and original_start > self.timestamps[0]:
                        interpolated = self.interp(
                            [original_start],
                            integer_interpolation_mode=integer_interpolation_mode,
                            float_interpolation_mode=float_interpolation_mode,
                        )
                        if len(interpolated):
                            samples = np.append(interpolated.samples, self.samples[start:], axis=0)
                            timestamps = np.append(interpolated.timestamps, self.timestamps[start:])
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

                    if samples.dtype != self.samples.dtype:
                        samples = samples.astype(self.samples.dtype)

                    result = Signal(
                        samples,
                        timestamps,
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_names,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        invalidation_bits=invalidation_bits,
                        encoding=self.encoding,
                        group_index=self.group_index,
                        channel_index=self.channel_index,
                        flags=self.flags,
                        virtual_conversion=self.virtual_conversion,
                        virtual_master_conversion=self.virtual_master_conversion,
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
                        self.display_names,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        encoding=self.encoding,
                        group_index=self.group_index,
                        channel_index=self.channel_index,
                        flags=self.flags,
                        virtual_conversion=self.virtual_conversion,
                        virtual_master_conversion=self.virtual_master_conversion,
                    )
                else:
                    start = np.searchsorted(self.timestamps, start, side="left")
                    stop = np.searchsorted(self.timestamps, stop, side="right")

                    if start == stop:
                        if include_ends:
                            if original_start == original_stop:
                                ends = np.array([original_start], dtype=self.timestamps.dtype)
                            else:
                                ends = np.array(
                                    [original_start, original_stop],
                                    dtype=self.timestamps.dtype,
                                )

                            interpolated = self.interp(
                                ends,
                                integer_interpolation_mode=integer_interpolation_mode,
                                float_interpolation_mode=float_interpolation_mode,
                            )
                            samples = interpolated.samples
                            timestamps = ends
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
                            invalidation_bits = self.invalidation_bits[start:stop].copy()
                        else:
                            invalidation_bits = None

                        if (
                            include_ends
                            and original_stop not in self.timestamps
                            and original_stop < self.timestamps[-1]
                        ):
                            interpolated = self.interp(
                                [original_stop],
                                integer_interpolation_mode=integer_interpolation_mode,
                                float_interpolation_mode=float_interpolation_mode,
                            )

                            if len(interpolated):
                                samples = np.append(samples, interpolated.samples, axis=0)
                                timestamps = np.append(timestamps, interpolated.timestamps)
                                if invalidation_bits is not None:
                                    invalidation_bits = np.append(
                                        invalidation_bits,
                                        interpolated.invalidation_bits,
                                    )

                        if (
                            include_ends
                            and original_start not in self.timestamps
                            and original_start > self.timestamps[0]
                        ):
                            interpolated = self.interp(
                                [original_start],
                                integer_interpolation_mode=integer_interpolation_mode,
                                float_interpolation_mode=float_interpolation_mode,
                            )

                            if len(interpolated):
                                samples = np.append(interpolated.samples, samples, axis=0)
                                timestamps = np.append(interpolated.timestamps, timestamps)

                                if invalidation_bits is not None:
                                    invalidation_bits = np.append(
                                        interpolated.invalidation_bits,
                                        invalidation_bits,
                                    )

                    if samples.dtype != self.samples.dtype:
                        samples = samples.astype(self.samples.dtype)

                    result = Signal(
                        samples,
                        timestamps,
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_names,
                        self.attachment,
                        self.source,
                        self.bit_count,
                        invalidation_bits=invalidation_bits,
                        encoding=self.encoding,
                        group_index=self.group_index,
                        channel_index=self.channel_index,
                        flags=self.flags,
                        virtual_conversion=self.virtual_conversion,
                        virtual_master_conversion=self.virtual_master_conversion,
                    )

        return result

    def extend(self, other: Signal) -> Signal:
        """extend signal with samples from another signal

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
                invalidation_bits = np.concatenate((np.zeros(len(self), dtype=bool), other.invalidation_bits))
            elif self.invalidation_bits is not None and other.invalidation_bits is None:
                invalidation_bits = np.concatenate((self.invalidation_bits, np.zeros(len(other), dtype=bool)))
            else:
                invalidation_bits = np.append(self.invalidation_bits, other.invalidation_bits)

            result = Signal(
                np.append(self.samples, other.samples, axis=0),
                np.append(self.timestamps, timestamps),
                self.unit,
                self.name,
                self.conversion,
                self.comment,
                self.raw,
                self.master_metadata,
                self.display_names,
                self.attachment,
                self.source,
                self.bit_count,
                invalidation_bits=invalidation_bits,
                encoding=self.encoding,
                group_index=self.group_index,
                channel_index=self.channel_index,
                flags=self.flags,
                virtual_conversion=self.virtual_conversion,
                virtual_master_conversion=self.virtual_master_conversion,
            )
        else:
            result = self

        return result

    def interp(
        self,
        new_timestamps: NDArray[Any],
        integer_interpolation_mode: (
            IntInterpolationModeType | IntegerInterpolation
        ) = IntegerInterpolation.REPEAT_PREVIOUS_SAMPLE,
        float_interpolation_mode: (
            FloatInterpolationModeType | FloatInterpolation
        ) = FloatInterpolation.LINEAR_INTERPOLATION,
    ) -> Signal:
        """returns a new *Signal* interpolated using the *new_timestamps*

        Parameters
        ----------
        new_timestamps : np.array
            timestamps used for interpolation

        integer_interpolation_mode : int
            interpolation mode for integer signals; default 0

                * 0 - repeat previous samples
                * 1 - linear interpolation
                * 2 - hybrid interpolation: channels with integer data type (raw values) that have a
                  conversion that outputs float values will use linear interpolation, otherwise
                  the previous sample is used

                .. versionadded:: 6.2.0

        float_interpolation_mode : int
            interpolation mode for float channels; default 1

                * 0 - repeat previous sample
                * 1 - use linear interpolation

                .. versionadded:: 6.2.0

        Returns
        -------
        signal : Signal
            new interpolated *Signal*

        """

        integer_interpolation_mode = IntegerInterpolation(integer_interpolation_mode)
        float_interpolation_mode = FloatInterpolation(float_interpolation_mode)

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
                display_names=self.display_names,
                attachment=self.attachment,
                invalidation_bits=None,
                encoding=self.encoding,
                group_index=self.group_index,
                channel_index=self.channel_index,
                flags=self.flags,
                virtual_conversion=self.virtual_conversion,
                virtual_master_conversion=self.virtual_master_conversion,
            )
        else:
            # # we need to validate first otherwise we can get false invalid data
            # # if the new timebase and the invalidation bits are aligned in an
            # # infavorable way
            #
            # if self.invalidation_bits is not None:
            #     signal = self.validate()
            #     has_invalidation = True
            # else:
            #     signal = self
            #     has_invalidation = False

            signal = self
            invalidation_bits = signal.invalidation_bits

            if not len(signal.samples):
                return Signal(
                    self.samples[:0].copy(),
                    self.timestamps[:0].copy(),
                    self.unit,
                    self.name,
                    comment=self.comment,
                    conversion=self.conversion,
                    raw=self.raw,
                    master_metadata=self.master_metadata,
                    display_names=self.display_names,
                    attachment=self.attachment,
                    invalidation_bits=None if invalidation_bits is None else np.array([], dtype=bool),
                    encoding=self.encoding,
                    group_index=self.group_index,
                    channel_index=self.channel_index,
                    flags=self.flags,
                    virtual_conversion=self.virtual_conversion,
                    virtual_master_conversion=self.virtual_master_conversion,
                )

            if len(signal.samples.shape) > 1:
                idx = np.searchsorted(signal.timestamps, new_timestamps, side="right")
                idx -= 1
                idx[idx < 0] = 0
                s = signal.samples[idx]
                if invalidation_bits is not None:
                    invalidation_bits = invalidation_bits[idx]
            else:
                kind = signal.samples.dtype.kind

                if kind == "f":
                    if float_interpolation_mode == FloatInterpolation.REPEAT_PREVIOUS_SAMPLE:
                        idx = np.searchsorted(signal.timestamps, new_timestamps, side="right")
                        idx -= 1
                        idx[idx < 0] = 0
                        s = signal.samples[idx]

                        if invalidation_bits is not None:
                            invalidation_bits = invalidation_bits[idx]

                    else:
                        s = np.interp(new_timestamps, signal.timestamps, signal.samples)

                        if invalidation_bits is not None:
                            idx = np.searchsorted(signal.timestamps, new_timestamps, side="right")
                            idx -= 1
                            idx[idx < 0] = 0
                            invalidation_bits = invalidation_bits[idx]

                elif kind in "ui":
                    if integer_interpolation_mode == IntegerInterpolation.HYBRID_INTERPOLATION:
                        if signal.raw and signal.conversion:
                            kind = signal.conversion.convert(signal.samples[:1]).dtype.kind
                            if kind == "f":
                                integer_interpolation_mode = IntegerInterpolation.LINEAR_INTERPOLATION

                    if integer_interpolation_mode == IntegerInterpolation.HYBRID_INTERPOLATION:
                        integer_interpolation_mode = IntegerInterpolation.REPEAT_PREVIOUS_SAMPLE

                    if integer_interpolation_mode == IntegerInterpolation.LINEAR_INTERPOLATION:
                        s = np.interp(new_timestamps, signal.timestamps, signal.samples).astype(signal.samples.dtype)

                        if invalidation_bits is not None:
                            idx = np.searchsorted(signal.timestamps, new_timestamps, side="right")
                            idx -= 1
                            idx[idx < 0] = 0
                            invalidation_bits = invalidation_bits[idx]

                    elif integer_interpolation_mode == IntegerInterpolation.REPEAT_PREVIOUS_SAMPLE:
                        idx = np.searchsorted(signal.timestamps, new_timestamps, side="right")
                        idx -= 1
                        idx[idx < 0] = 0

                        s = signal.samples[idx]

                        if invalidation_bits is not None:
                            invalidation_bits = invalidation_bits[idx]

                else:
                    idx = np.searchsorted(signal.timestamps, new_timestamps, side="right")
                    idx -= 1
                    idx[idx < 0] = 0
                    s = signal.samples[idx]

                    if invalidation_bits is not None:
                        invalidation_bits = invalidation_bits[idx]

            if s.dtype != self.samples.dtype:
                s = s.astype(self.samples.dtype)

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
                display_names=self.display_names,
                attachment=self.attachment,
                invalidation_bits=invalidation_bits,
                encoding=self.encoding,
                group_index=self.group_index,
                channel_index=self.channel_index,
                flags=self.flags,
                virtual_conversion=self.virtual_conversion,
                virtual_master_conversion=self.virtual_master_conversion,
            )

    def __apply_func(self, other: Signal | NDArray[Any] | None, func_name: str) -> Signal:
        """delegate operations to the *samples* attribute, but in a time
        correct manner by considering the *timestamps*

        """

        if isinstance(other, Signal):
            if len(self) and len(other):
                start = max(self.timestamps[0], other.timestamps[0])
                stop = min(self.timestamps[-1], other.timestamps[-1])
                s1 = self.physical().cut(start, stop)
                s2 = other.physical().cut(start, stop)
            else:
                s1 = self
                s2 = other
            time = np.union1d(s1.timestamps, s2.timestamps)
            s = s1.interp(time).samples
            o = s2.interp(time).samples
            func = getattr(s, func_name)
            conversion = None
            s = func(o)
        elif other is None:
            s = self.samples
            conversion = self.conversion
            time = self.timestamps
        else:
            func = getattr(self.samples, func_name)
            s = func(other)
            conversion = self.conversion
            time = self.timestamps
        return Signal(
            samples=s,
            timestamps=time,
            unit=self.unit,
            name=self.name,
            conversion=conversion,
            raw=self.raw,
            master_metadata=self.master_metadata,
            display_names=self.display_names,
            attachment=self.attachment,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=self.encoding,
            group_index=self.group_index,
            channel_index=self.channel_index,
            flags=self.flags,
            virtual_conversion=self.virtual_conversion,
            virtual_master_conversion=self.virtual_master_conversion,
        )

    def __pos__(self) -> Signal:
        return self

    def __neg__(self) -> Signal:
        return Signal(
            np.negative(self.samples),
            self.timestamps,
            unit=self.unit,
            name=self.name,
            conversion=self.conversion,
            raw=self.raw,
            master_metadata=self.master_metadata,
            display_names=self.display_names,
            attachment=self.attachment,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=self.encoding,
            flags=self.flags,
            virtual_conversion=self.virtual_conversion,
            virtual_master_conversion=self.virtual_master_conversion,
        )

    def __round__(self, n: int) -> Signal:
        return Signal(
            np.around(self.samples, n),
            self.timestamps,
            unit=self.unit,
            name=self.name,
            conversion=self.conversion,
            raw=self.raw,
            master_metadata=self.master_metadata,
            display_names=self.display_names,
            attachment=self.attachment,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=self.encoding,
            flags=self.flags,
            virtual_conversion=self.virtual_conversion,
            virtual_master_conversion=self.virtual_master_conversion,
        )

    def __sub__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__sub__")

    def __isub__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__sub__(other)

    def __rsub__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return -self.__sub__(other)

    def __add__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__add__")

    def __iadd__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__add__(other)

    def __radd__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__add__(other)

    def __truediv__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__truediv__")

    def __itruediv__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__truediv__(other)

    def __rtruediv__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__rtruediv__")

    def __mul__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__mul__")

    def __imul__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__mul__(other)

    def __rmul__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__mul__(other)

    def __floordiv__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__floordiv__")

    def __ifloordiv__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__truediv__(other)

    def __rfloordiv__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return 1 / self.__apply_func(other, "__rfloordiv__")

    def __mod__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__mod__")

    def __pow__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__pow__")

    def __and__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__and__")

    def __or__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__or__")

    def __xor__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__xor__")

    def __invert__(self) -> Signal:
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
            display_names=self.display_names,
            attachment=self.attachment,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=self.encoding,
            flags=self.flags,
            virtual_conversion=self.virtual_conversion,
            virtual_master_conversion=self.virtual_master_conversion,
        )

    def __lshift__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__lshift__")

    def __rshift__(self, other: Signal | NDArray[Any] | None) -> Signal:
        return self.__apply_func(other, "__rshift__")

    def __lt__(self, other: Signal | NDArray[Any] | None) -> bool:
        return self.__apply_func(other, "__lt__")

    def __le__(self, other: Signal | NDArray[Any] | None) -> bool:
        return self.__apply_func(other, "__le__")

    def __gt__(self, other: Signal | NDArray[Any] | None) -> bool:
        return self.__apply_func(other, "__gt__")

    def __ge__(self, other: Signal | NDArray[Any] | None) -> bool:
        return self.__apply_func(other, "__ge__")

    def __eq__(self, other: Signal | NDArray[Any] | None) -> bool:
        return self.__apply_func(other, "__eq__")

    def __ne__(self, other: Signal | NDArray[Any] | None) -> bool:
        return self.__apply_func(other, "__ne__")

    def __iter__(self) -> Iterator[Any]:
        yield from (self.samples, self.timestamps, self.unit, self.name)

    def __reversed__(self) -> Iterator[tuple[int, tuple[Any, Any]]]:
        return enumerate(zip(reversed(self.samples), reversed(self.timestamps)))

    def __len__(self) -> int:
        return len(self.samples)

    def __abs__(self) -> Signal:
        return Signal(
            np.fabs(self.samples),
            self.timestamps,
            unit=self.unit,
            name=self.name,
            conversion=self.conversion,
            raw=self.raw,
            master_metadata=self.master_metadata,
            display_names=self.display_names,
            attachment=self.attachment,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            flags=self.flags,
            virtual_conversion=self.virtual_conversion,
            virtual_master_conversion=self.virtual_master_conversion,
        )

    def __getitem__(self, val: int) -> Any:
        return self.samples[val]

    def __setitem__(self, idx: int, val: Any) -> None:
        self.samples[idx] = val

    def astype(self, np_type: DTypeLike) -> Signal:
        """returns new *Signal* with samples of dtype *np_type*

        Parameters
        ----------
        np_type : np.dtype
            new numpy dtype

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
            display_names=self.display_names,
            attachment=self.attachment,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=self.encoding,
            flags=self.flags,
            virtual_conversion=self.virtual_conversion,
            virtual_master_conversion=self.virtual_master_conversion,
        )

    def physical(self, copy: bool = True) -> Signal:
        """
        get the physical samples values

        Parameters
        ----------
        copy : bool
            copy the samples and timestamps in the returned Signal

            .. versionadded:: 7.4.0

        Returns
        -------
        phys : Signal
            new *Signal* with physical values

        """

        if not self.raw or self.conversion is None:
            if copy:
                samples = self.samples.copy()
            else:
                samples = self.samples
            encoding = None
        else:
            samples = self.conversion.convert(self.samples)
            if samples.dtype.kind == "S":
                encoding = "utf-8" if self.conversion.id == b"##CC" else "latin-1"
            else:
                encoding = None

        return Signal(
            samples,
            self.timestamps.copy() if copy else self.timestamps,
            unit=self.unit,
            name=self.name,
            conversion=None,
            raw=False,
            master_metadata=self.master_metadata,
            display_names=self.display_names,
            attachment=self.attachment,
            invalidation_bits=self.invalidation_bits,
            source=self.source,
            encoding=encoding,
            group_index=self.group_index,
            channel_index=self.channel_index,
            flags=self.flags,
            virtual_conversion=self.virtual_conversion,
            virtual_master_conversion=self.virtual_master_conversion,
        )

    def validate(self, copy: bool = True) -> Signal:
        """apply invalidation bits if they are available for this signal

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
            if len(idx) == len(self.samples):
                signal = self
            else:
                signal = Signal(
                    self.samples[idx],
                    self.timestamps[idx],
                    self.unit,
                    self.name,
                    self.conversion,
                    self.comment,
                    self.raw,
                    self.master_metadata,
                    self.display_names,
                    self.attachment,
                    self.source,
                    self.bit_count,
                    invalidation_bits=None,
                    encoding=self.encoding,
                    group_index=self.group_index,
                    channel_index=self.channel_index,
                    flags=self.flags,
                    virtual_conversion=self.virtual_conversion,
                    virtual_master_conversion=self.virtual_master_conversion,
                )

        if copy:
            signal = signal.copy()

        return signal

    def copy(self) -> Signal:
        """copy all attributes to a new Signal"""
        return Signal(
            self.samples.copy(),
            self.timestamps.copy(),
            self.unit,
            self.name,
            self.conversion,
            self.comment,
            self.raw,
            self.master_metadata,
            self.display_names,
            self.attachment,
            self.source,
            self.bit_count,
            invalidation_bits=self.invalidation_bits.copy() if self.invalidation_bits is not None else None,
            encoding=self.encoding,
            group_index=self.group_index,
            channel_index=self.channel_index,
            flags=self.flags,
            virtual_conversion=self.virtual_conversion,
            virtual_master_conversion=self.virtual_master_conversion,
        )


if __name__ == "__main__":
    pass

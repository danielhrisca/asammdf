"""asammdf `Signal` class module for time-correct signal processing"""

from collections.abc import Iterator
import logging
from pathlib import Path
from textwrap import fill
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import Any, overload

from .blocks import v2_v3_blocks as v3b
from .blocks import v4_blocks as v4b
from .blocks.conversion_utils import from_dict
from .blocks.options import FloatInterpolation, IntegerInterpolation
from .blocks.source_utils import Source
from .blocks.types import (
    ChannelConversionType,
    FloatInterpolationModeType,
    IntInterpolationModeType,
    SourceType,
)
from .blocks.utils import extract_xml_comment, MdfException, SignalFlags
from .version import __version__

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

try:
    encode = np.strings.encode
except:
    encode = np.char.encode

logger = logging.getLogger("asammdf")

ORIGIN_UNKNOWN = (-1, -1)


class InvalidationArray(np.ndarray[tuple[int], np.dtype[np.bool]]):
    ORIGIN_UNKNOWN = ORIGIN_UNKNOWN

    def __new__(cls, input_array: ArrayLike, origin: tuple[int, int] = ORIGIN_UNKNOWN) -> "InvalidationArray":
        obj = np.asarray(input_array).view(cls)
        obj.origin = origin
        return obj

    def __array_finalize__(self, obj: NDArray[np.bool] | None) -> None:
        if obj is None:
            return
        self.origin: tuple[int, int] = getattr(obj, "origin", ORIGIN_UNKNOWN)


class Signal:
    """The `Signal` represents a channel described by its samples and
    timestamps. It can perform arithmetic operations against other `Signal`
    objects or numeric types. The operations are computed in respect to the
    timestamps (time-correct). The non-float signals are not interpolated,
    instead the last value relative to the current timestamp is used.
    `samples`, `timestamps` and `name` are mandatory arguments.

    Parameters
    ----------
    samples : array-like
        Signal samples.
    timestamps : array-like
        Signal timestamps.
    unit : str, optional
        Signal unit.
    name : str
        Signal name.
    conversion : dict | channel conversion block, optional
        Dict that contains extra conversion information about the signal.
    comment : str, optional
        Signal comment.
    raw : bool, default True
        Signal samples are raw values, with no physical conversion applied.
    master_metadata : tuple, optional
        Master name and sync type.
    display_names : dict, optional
        Display names used by MDF version 3.
    attachment : tuple, optional
        Channel attachment and name from MDF version 4.
    source : Source, optional
        Source information named tuple.
    bit_count : int, optional
        Bit count; useful for integer channels.
    invalidation_bits : array-like, optional
        Channel invalidation bits.
    encoding : str, optional
        Encoding for string signals.
    flags : int, optional
        Flags for user-defined attributes and stream sync.
    """

    Flags = SignalFlags

    def __init__(
        self,
        samples: ArrayLike,
        timestamps: ArrayLike,
        unit: str = "",
        name: str = "",
        conversion: dict[str, object] | ChannelConversionType | None = None,
        comment: str = "",
        raw: bool = True,
        master_metadata: tuple[str, int] | None = None,
        display_names: dict[str, str] | None = None,
        attachment: tuple[bytes | str, Path, bytes | str] | None = None,
        source: SourceType | None = None,
        bit_count: int | None = None,
        invalidation_bits: ArrayLike | None = None,
        encoding: str | None = None,
        group_index: int = -1,
        channel_index: int = -1,
        flags: int = Flags.no_flags,
        virtual_conversion: dict[str, object] | ChannelConversionType | None = None,
        virtual_master_conversion: dict[str, object] | ChannelConversionType | None = None,
    ) -> None:
        if not name:
            message = (
                '"samples", "timestamps" and "name" are mandatory '
                f"for Signal class __init__: samples={samples!r}\n"
                f"timestamps={timestamps!r}\nname={name}"
            )
            raise MdfException(message)
        else:
            self.samples: NDArray[Any]
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples)
                kind = samples.dtype.kind
                if kind == "U":
                    if encoding is None:
                        encodings = ["utf-8", "latin-1"]
                    else:
                        encodings = [encoding, "utf-8", "latin-1"]
                    for _encoding in encodings:
                        try:
                            self.samples = encode(samples, _encoding)
                            break
                        except:
                            continue
                    else:
                        self.samples = encode(samples, encodings[0], errors="ignore")
                elif kind == "O":
                    self.samples = samples.astype(np.bytes_)
                else:
                    self.samples = samples
            else:
                self.samples = samples

            self.timestamps: NDArray[Any]
            if not isinstance(timestamps, np.ndarray):
                self.timestamps = np.array(timestamps, dtype=np.float64)
            else:
                self.timestamps = timestamps

            if self.samples.shape[0] != self.timestamps.shape[0]:
                message = "{} samples and timestamps length mismatch ({} vs {})"
                message = message.format(name, self.samples.shape[0], self.timestamps.shape[0])
                logger.exception(message)
                raise MdfException(message)

            self.unit = unit
            self.name = name
            self.comment = comment
            self.flags = flags
            self._plot_axis: Line3DCollection | None = None
            self.raw = raw
            self.master_metadata = master_metadata
            self.display_names = display_names or {}
            self.attachment = attachment
            self.encoding = encoding
            self.group_index = group_index
            self.channel_index = channel_index
            self._invalidation_bits = InvalidationArray(invalidation_bits) if invalidation_bits is not None else None

            self.source: Source | None
            if source:
                if not isinstance(source, Source):
                    self.source = Source.from_source(source)
                else:
                    self.source = source
            else:
                self.source = None

            if bit_count is None:
                self.bit_count = self.samples.dtype.itemsize * 8
            else:
                self.bit_count = bit_count

            self.conversion: ChannelConversionType | None
            if conversion:
                if not isinstance(conversion, (v4b.ChannelConversion, v3b.ChannelConversion)):
                    self.conversion = from_dict(conversion)
                else:
                    self.conversion = conversion
            else:
                self.conversion = None

            self.virtual_conversion: ChannelConversionType | None
            if self.flags & self.Flags.virtual:
                if not isinstance(virtual_conversion, (v4b.ChannelConversion, v3b.ChannelConversion)):
                    self.virtual_conversion = from_dict(virtual_conversion)
                else:
                    self.virtual_conversion = virtual_conversion
            else:
                self.virtual_conversion = None

            self.virtual_master_conversion: ChannelConversionType | None
            if self.flags & self.Flags.virtual_master:
                if not isinstance(virtual_master_conversion, (v4b.ChannelConversion, v3b.ChannelConversion)):
                    self.virtual_master_conversion = from_dict(virtual_master_conversion)
                else:
                    self.virtual_master_conversion = virtual_master_conversion
            else:
                self.virtual_master_conversion = None

    @property
    def invalidation_bits(self) -> InvalidationArray | None:
        return self._invalidation_bits

    @invalidation_bits.setter
    def invalidation_bits(self, value: ArrayLike | None) -> None:
        if value is None:
            self._invalidation_bits = None

        else:
            if not isinstance(value, InvalidationArray):
                value = InvalidationArray(value)

            if value.shape[0] != self.samples.shape[0]:
                message = "{} samples and invalidation bits length mismatch ({} vs {})"
                message = message.format(self.name, self.samples.shape[0], value.shape[0])
                logger.exception(message)
                raise MdfException(message)

            self._invalidation_bits = value

    def __repr__(self) -> str:
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

    def plot(self, validate: bool = True) -> None:
        """Plot Signal samples. Pyqtgraph is used if it is available; in this
        case see the GUI plot documentation to see the available commands.

        Parameters
        ----------
        validate : bool, default True
            Apply the invalidation bits.
        """
        try:
            from .gui.plot import plot

            plot(self, validate=True)
            return

        except:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.widgets import Slider
                from mpl_toolkits.mplot3d import Axes3D
            except ImportError:
                logging.warning("Signal plotting requires pyqtgraph or matplotlib")
                return

        if len(self.samples.shape) <= 1 and self.samples.dtype.names is None:
            fig = plt.figure()
            if fig.canvas.manager:
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
                    match sync_type:
                        case 0 | 1:
                            plt.xlabel(f"{master_name} [s]")
                        case 2:
                            plt.xlabel(f"{master_name} [deg]")
                        case 3:
                            plt.xlabel(f"{master_name} [m]")
                        case 4:
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
                if names is None or len(names) == 1:
                    if names:
                        samples = self.samples[names[0]]
                    else:
                        samples = self.samples

                    shape = samples.shape[1:]

                    fig = plt.figure()
                    if fig.canvas.manager:
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

                    if self.comment:
                        comment = self.comment.replace("$", "")
                        plt.title(f"{self.name}\n({comment})")
                    else:
                        plt.title(self.name)

                    ax: Axes3D = fig.add_subplot(111, projection="3d")

                    # Grab some test data.
                    X = np.array(range(shape[1]))
                    Y = np.array(range(shape[0]))
                    X, Y = np.meshgrid(X, Y)

                    Z = samples[0]

                    # Plot a basic wireframe.
                    self._plot_axis = plot_axis = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

                    # Place Sliders on Graph
                    ax_a = plt.axes((0.25, 0.1, 0.65, 0.03))

                    # Create Sliders & Determine Range
                    sa = Slider(
                        ax_a,
                        "Time [s]",
                        self.timestamps[0],
                        self.timestamps[-1],
                        valinit=self.timestamps[0],
                    )

                    def update(val: float) -> None:
                        plot_axis.remove()
                        idx = np.searchsorted(self.timestamps, sa.val, side="right")
                        Z = samples[idx - 1]
                        self._plot_axis = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
                        fig.canvas.draw_idle()

                    sa.on_changed(update)

                    plt.show()

                else:
                    fig = plt.figure()
                    if fig.canvas.manager:
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
                    self._plot_axis = plot_axis = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

                    # Place Sliders on Graph
                    ax_a = plt.axes((0.25, 0.1, 0.65, 0.03))

                    # Create Sliders & Determine Range
                    sa = Slider(
                        ax_a,
                        "Time [s]",
                        self.timestamps[0],
                        self.timestamps[-1],
                        valinit=self.timestamps[0],
                    )

                    def update(val: float) -> None:
                        plot_axis.remove()
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
    ) -> "Signal":
        """Cut the signal according to the `start` and `stop` values, by using
        the insertion indexes in the signal's time axis.

        Parameters
        ----------
        start : float, optional
            Start timestamp for cutting.
        stop : float, optional
            Stop timestamp for cutting.
        include_ends : bool, default True
            Include the `start` and `stop` timestamps after cutting the signal.
            If `start` and `stop` are not found in the original timestamps,
            then the new samples will be computed using interpolation.
        integer_interpolation_mode : int, default 0
            Interpolation mode for integer signals.

            * 0 - repeat previous sample
            * 1 - linear interpolation
            * 2 - hybrid interpolation: channels with integer data type (raw
              values) that have a conversion that outputs float values will use
              linear interpolation, otherwise the previous sample is used

            .. versionadded:: 6.2.0

        float_interpolation_mode : int, default 1
            Interpolation mode for float channels.

            * 0 - repeat previous sample
            * 1 - linear interpolation

            .. versionadded:: 6.2.0

        Returns
        -------
        result : Signal
            New `Signal` cut from the original.

        Examples
        --------
        >>> from asammdf import Signal
        >>> import numpy as np
        >>> old_sig = Signal(np.arange(0.03, 100, 0.05), np.arange(0.03, 100, 0.05), name='SIG')
        >>> new_sig = old_sig.cut(1.0, 10.5)
        >>> new_sig.timestamps[0], new_sig.timestamps[-1]
        (1.0, 10.5)

        >>> new_sig = old_sig.cut(1.0, 10.5, include_ends=False)
        >>> new_sig.timestamps[0], new_sig.timestamps[-1]
        (1.03, 10.48)

        >>> new_sig = old_sig.cut(1.0, 10.5, float_interpolation_mode=0)
        >>> new_sig.samples[0], new_sig.samples[-1]
        (0.98, 10.48)
        """

        integer_interpolation_mode = IntegerInterpolation(integer_interpolation_mode)
        float_interpolation_mode = FloatInterpolation(float_interpolation_mode)

        if self.samples.size == 0:
            return Signal(
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

        invalidation_bits: NDArray[np.bool] | None
        if start is None:
            if stop is None:
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
                    stop_idx = np.searchsorted(self.timestamps, stop, side="right")
                    if include_ends and stop not in self.timestamps and stop < self.timestamps[-1]:
                        interpolated = self.interp(
                            [stop],
                            integer_interpolation_mode=integer_interpolation_mode,
                            float_interpolation_mode=float_interpolation_mode,
                        )

                        if len(interpolated):
                            samples = np.append(self.samples[:stop_idx], interpolated.samples, axis=0)
                            timestamps = np.append(self.timestamps[:stop_idx], interpolated.timestamps)
                            if self.invalidation_bits is not None and interpolated.invalidation_bits is not None:
                                invalidation_bits = InvalidationArray(
                                    np.append(
                                        self.invalidation_bits[:stop_idx],
                                        interpolated.invalidation_bits,
                                    ),
                                    self.invalidation_bits.origin,
                                )
                            else:
                                invalidation_bits = None
                    else:
                        samples = self.samples[:stop_idx].copy()
                        timestamps = self.timestamps[:stop_idx].copy()
                        if self.invalidation_bits is not None:
                            invalidation_bits = self.invalidation_bits[:stop_idx].copy()
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
            if stop is None:
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
                    start_idx = np.searchsorted(self.timestamps, start, side="left")
                    if include_ends and start not in self.timestamps and start > self.timestamps[0]:
                        interpolated = self.interp(
                            [start],
                            integer_interpolation_mode=integer_interpolation_mode,
                            float_interpolation_mode=float_interpolation_mode,
                        )
                        if len(interpolated):
                            samples = np.append(interpolated.samples, self.samples[start_idx:], axis=0)
                            timestamps = np.append(interpolated.timestamps, self.timestamps[start_idx:])
                            if self.invalidation_bits is not None and interpolated.invalidation_bits is not None:
                                invalidation_bits = InvalidationArray(
                                    np.append(
                                        interpolated.invalidation_bits,
                                        self.invalidation_bits[start_idx:],
                                    ),
                                    self.invalidation_bits.origin,
                                )
                            else:
                                invalidation_bits = None
                    else:
                        samples = self.samples[start_idx:].copy()
                        timestamps = self.timestamps[start_idx:].copy()
                        if self.invalidation_bits is not None:
                            invalidation_bits = self.invalidation_bits[start_idx:].copy()
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
                    start_idx = np.searchsorted(self.timestamps, start, side="left")
                    stop_idx = np.searchsorted(self.timestamps, stop, side="right")

                    if start_idx == stop_idx:
                        if include_ends:
                            if start == stop:
                                ends = np.array([start], dtype=self.timestamps.dtype)
                            else:
                                ends = np.array(
                                    [start, stop],
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
                                invalidation_bits = self.invalidation_bits[0:0]
                            else:
                                invalidation_bits = None
                    else:
                        samples = self.samples[start_idx:stop_idx].copy()
                        timestamps = self.timestamps[start_idx:stop_idx].copy()
                        if self.invalidation_bits is not None:
                            invalidation_bits = self.invalidation_bits[start_idx:stop_idx].copy()
                        else:
                            invalidation_bits = None

                        if include_ends and stop not in self.timestamps and stop < self.timestamps[-1]:
                            interpolated = self.interp(
                                [stop],
                                integer_interpolation_mode=integer_interpolation_mode,
                                float_interpolation_mode=float_interpolation_mode,
                            )

                            if len(interpolated):
                                samples = np.append(samples, interpolated.samples, axis=0)
                                timestamps = np.append(timestamps, interpolated.timestamps)
                                if invalidation_bits is not None and interpolated.invalidation_bits is not None:
                                    invalidation_bits = InvalidationArray(
                                        np.append(
                                            invalidation_bits,
                                            interpolated.invalidation_bits,
                                        ),
                                        interpolated.invalidation_bits.origin,
                                    )

                        if include_ends and start not in self.timestamps and start > self.timestamps[0]:
                            interpolated = self.interp(
                                [start],
                                integer_interpolation_mode=integer_interpolation_mode,
                                float_interpolation_mode=float_interpolation_mode,
                            )

                            if len(interpolated):
                                samples = np.append(interpolated.samples, samples, axis=0)
                                timestamps = np.append(interpolated.timestamps, timestamps)

                                if invalidation_bits is not None and interpolated.invalidation_bits is not None:
                                    invalidation_bits = InvalidationArray(
                                        np.append(
                                            interpolated.invalidation_bits,
                                            invalidation_bits,
                                        ),
                                        interpolated.invalidation_bits.origin,
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

    def extend(self, other: "Signal") -> "Signal":
        """Extend Signal with samples from another Signal.

        Parameters
        ----------
        other : Signal

        Returns
        -------
        signal : Signal
            New extended `Signal`.
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

            if self.invalidation_bits is None:
                if other.invalidation_bits is None:
                    invalidation_bits = None
                else:
                    invalidation_bits = InvalidationArray(
                        np.concatenate((np.zeros(len(self), dtype=bool), other.invalidation_bits)),
                        other.invalidation_bits.origin,
                    )
            else:
                if other.invalidation_bits is None:
                    invalidation_bits = InvalidationArray(
                        np.concatenate((self.invalidation_bits, np.zeros(len(other), dtype=bool))),
                        self.invalidation_bits.origin,
                    )
                else:
                    invalidation_bits = InvalidationArray(
                        np.append(self.invalidation_bits, other.invalidation_bits), self.invalidation_bits.origin
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
        new_timestamps: NDArray[Any] | list[float],
        integer_interpolation_mode: (
            IntInterpolationModeType | IntegerInterpolation
        ) = IntegerInterpolation.REPEAT_PREVIOUS_SAMPLE,
        float_interpolation_mode: (
            FloatInterpolationModeType | FloatInterpolation
        ) = FloatInterpolation.LINEAR_INTERPOLATION,
    ) -> "Signal":
        """Return a new `Signal` interpolated using the `new_timestamps`.

        Parameters
        ----------
        new_timestamps : np.ndarray | list
            Timestamps used for interpolation.

        integer_interpolation_mode : int, default 0
            Interpolation mode for integer signals.

            * 0 - repeat previous sample
            * 1 - linear interpolation
            * 2 - hybrid interpolation: channels with integer data type (raw
              values) that have a conversion that outputs float values will use
              linear interpolation, otherwise the previous sample is used

            .. versionadded:: 6.2.0

        float_interpolation_mode : int, default 1
            Interpolation mode for float channels.

            * 0 - repeat previous sample
            * 1 - linear interpolation

            .. versionadded:: 6.2.0

        Returns
        -------
        signal : Signal
            New interpolated `Signal`.
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
            invalidation_bits: NDArray[np.bool] | None = signal.invalidation_bits

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

    def __apply_func(self, other: Union["Signal", NDArray[Any], float] | None, func_name: str) -> "Signal":
        """Delegate operations to the `samples` attribute, but in a
        time-correct manner by considering the `timestamps`.
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
            s1 = s1.interp(time)
            s2 = s2.interp(time)

            invalidation_bits: NDArray[np.bool] | None
            if s1.invalidation_bits is not None or s2.invalidation_bits is not None:
                if s1.invalidation_bits is None:
                    invalidation_bits = s2.invalidation_bits
                elif s2.invalidation_bits is None:
                    invalidation_bits = s1.invalidation_bits
                else:
                    invalidation_bits = s1.invalidation_bits | s2.invalidation_bits
            else:
                invalidation_bits = None

            func = getattr(s1.samples, func_name)
            conversion = None
            s = func(s2.samples)

        elif other is None:
            s = self.samples
            conversion = self.conversion
            time = self.timestamps
            invalidation_bits = self.invalidation_bits

        else:
            func = getattr(self.samples, func_name)
            s = func(other)
            conversion = self.conversion
            time = self.timestamps
            invalidation_bits = self.invalidation_bits

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
            invalidation_bits=invalidation_bits,
            source=self.source,
            encoding=self.encoding,
            group_index=self.group_index,
            channel_index=self.channel_index,
            flags=self.flags,
            virtual_conversion=self.virtual_conversion,
            virtual_master_conversion=self.virtual_master_conversion,
        )

    def __pos__(self) -> "Signal":
        return self

    def __neg__(self) -> "Signal":
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

    def __round__(self, n: int) -> "Signal":
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

    def __sub__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__sub__")

    def __isub__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__sub__(other)

    def __rsub__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return -self.__sub__(other)

    def __add__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__add__")

    def __iadd__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__add__(other)

    def __radd__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__add__(other)

    def __truediv__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__truediv__")

    def __itruediv__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__truediv__(other)

    def __rtruediv__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__rtruediv__")

    def __mul__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__mul__")

    def __imul__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__mul__(other)

    def __rmul__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__mul__(other)

    def __floordiv__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__floordiv__")

    def __ifloordiv__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__truediv__(other)

    def __rfloordiv__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return 1 / self.__apply_func(other, "__rfloordiv__")

    def __mod__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__mod__")

    def __pow__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__pow__")

    def __and__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__and__")

    def __or__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__or__")

    def __xor__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__xor__")

    def __invert__(self) -> "Signal":
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

    def __lshift__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__lshift__")

    def __rshift__(self, other: Union["Signal", NDArray[Any], float] | None) -> "Signal":
        return self.__apply_func(other, "__rshift__")

    def __lt__(self, other: Union["Signal", NDArray[Any]] | None) -> "Signal":
        return self.__apply_func(other, "__lt__")

    def __le__(self, other: Union["Signal", NDArray[Any]] | None) -> "Signal":
        return self.__apply_func(other, "__le__")

    def __gt__(self, other: Union["Signal", NDArray[Any]] | None) -> "Signal":
        return self.__apply_func(other, "__gt__")

    def __ge__(self, other: Union["Signal", NDArray[Any]] | None) -> "Signal":
        return self.__apply_func(other, "__ge__")

    def __eq__(self, other: Union["Signal", NDArray[Any]] | None) -> "Signal":  # type: ignore[override]
        return self.__apply_func(other, "__eq__")

    def __ne__(self, other: Union["Signal", NDArray[Any]] | None) -> "Signal":  # type: ignore[override]
        return self.__apply_func(other, "__ne__")

    def __iter__(self) -> Iterator[NDArray[Any] | str]:
        yield from (self.samples, self.timestamps, self.unit, self.name)

    def __reversed__(self) -> Iterator[tuple[int, tuple[Any, Any]]]:
        return enumerate(zip(reversed(self.samples), reversed(self.timestamps), strict=False))

    def __len__(self) -> int:
        return len(self.samples)

    def __abs__(self) -> "Signal":
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

    @overload
    def __getitem__(self, val: str) -> NDArray[Any]: ...

    @overload
    def __getitem__(self, val: int | slice) -> "Signal": ...

    def __getitem__(self, val: int | slice | str) -> Union[NDArray[Any], "Signal"]:
        if isinstance(val, str):
            return self.samples[val]
        else:
            return Signal(
                self.samples[val],
                self.timestamps[val],
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
                invalidation_bits=self.invalidation_bits[val] if self.invalidation_bits is not None else None,
                encoding=self.encoding,
                group_index=self.group_index,
                channel_index=self.channel_index,
                flags=self.flags,
                virtual_conversion=self.virtual_conversion,
                virtual_master_conversion=self.virtual_master_conversion,
            )

    def __setitem__(self, idx: Any, val: Any) -> None:
        self.samples[idx] = val

    def astype(self, np_type: DTypeLike) -> "Signal":
        """Return a new `Signal` with samples of dtype `np_type`.

        Parameters
        ----------
        np_type : np.dtype
            New numpy dtype.

        Returns
        -------
        signal : Signal
            New `Signal` with the samples of dtype `np_type`.
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

    def physical(self, copy: bool = True, ignore_value2text_conversions: bool = False) -> "Signal":
        """Get the physical sample values.

        Parameters
        ----------
        copy : bool, default True
            Copy the samples and timestamps in the returned Signal.

            .. versionadded:: 7.4.0

        ignore_value2text_conversions : bool, default False
            Make sure that the output signal has numeric samples by ignoring
            the value to text conversions.

            .. versionadded:: 8.3.0

        Returns
        -------
        phys : Signal
            New `Signal` with physical values.
        """

        if not self.raw or self.conversion is None:
            if copy:
                samples = self.samples.copy()
            else:
                samples = self.samples
            encoding = None
        else:
            samples = self.conversion.convert(self.samples, ignore_value2text_conversions=ignore_value2text_conversions)
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

    def validate(self, copy: bool = True) -> "Signal":
        """Apply invalidation bits if they are available for this signal.

        Parameters
        ----------
        copy : bool, default True
            Return a copy of the result.

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

    def copy(self) -> "Signal":
        """Copy all attributes to a new Signal."""
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

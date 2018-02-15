# -*- coding: utf-8 -*-
""" asammdf *Signal* class module for time correct signal processing """

import numpy as np
import warnings

from numexpr import evaluate

from .utils import MdfException
from .version import __version__


class SignalConversions(object):
    """
    types of generic conversions found in the `Signal` conversion attribute.
    This holds all the conversion types found in the mdf versions 3 and 4
    """

    CONVERSION_NONE = 0
    CONVERSION_LINEAR = 1
    CONVERSION_RATIONAL = 2
    CONVERSION_ALGEBRAIC = 3
    CONVERSION_POLYNOMIAL = 4
    CONVERSION_TAB = 5
    CONVERSION_TABI = 6
    CONVERSION_TABX = 7
    CONVERSION_RTAB = 8
    CONVERSION_RTABX = 9
    CONVERSION_TTAB = 10
    CONVERSION_TRANS = 11
    CONVERSION_EXPO = 12
    CONVERSION_LOGH = 13


class Signal(object):
    """
    The *Signal* represents a hannel described by it's samples and timestamps.
    It can perform aritmethic operations agains other *Signal* or numeric types.
    The operations are computed in respect to the timestamps (time correct).
    The non-float signals are not interpolated, instead the last value relative
    to the current timestamp is used.
    *samples*, *timstamps* and *name* are mandatory arguments.

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
    conversion : dict
        dict that contains extra conversionrmation about the signal ,
        default *None*
    comment : str
        signal comment, default ''
    raw : bool
        signal samples are raw values, with no physical conversion applied

    """

    __slots__ = [
        'samples',
        'timestamps',
        'unit',
        'name',
        'conversion',
        'comment',
        '_plot_axis',
        'raw',
    ]

    def __init__(self,
                 samples=None,
                 timestamps=None,
                 unit='',
                 name='',
                 conversion=None,
                 comment='',
                 raw=False):

        if samples is None or timestamps is None or name == '':
            message = ('"samples", "timestamps" and "name" are mandatory '
                       'for Signal class __init__: samples={}\n'
                       'timestamps={}\nname={}')
            raise MdfException(message.format(samples, timestamps, name))
        else:
            if isinstance(samples, (list, tuple)):
                samples = np.array(samples)
            if isinstance(timestamps, (list, tuple)):
                timestamps = np.array(timestamps, dtype=np.float64)
            if not samples.shape[0] == timestamps.shape[0]:
                message = 'samples and timestamps length missmatch ({} vs {})'
                message = message.format(samples.shape[0], timestamps.shape[0])
                raise MdfException(message)
            self.samples = samples
            self.timestamps = timestamps
            self.unit = unit
            self.name = name
            self.conversion = conversion
            self.comment = comment
            self._plot_axis = None
            self.raw = raw

#    def physical(self):
#        """ get Signal with physical conversion appplied
#        to its samples
#
#        """
#        if self.raw:
#            pass
#        else:
#            return self

    def __str__(self):
        string = """<Signal {}:
\tsamples={}
\ttimestamps={}
\tunit="{}"
\tconversion={}
\tcomment="{}"
\traw={}>
"""
        return string.format(
            self.name,
            self.samples,
            self.timestamps,
            self.unit,
            self.conversion,
            self.comment,
            self.raw,
        )

    def __repr__(self):
        string = (
            'Signal(name={}, samples={}, timestamps={}, '
            'unit={}, conversion={}, comment={}, raw={})'
        )
        return string.format(
            self.name,
            repr(self.samples),
            repr(self.timestamps),
            self.unit,
            self.conversion,
            self.comment,
            self.raw,
        )

    def plot(self):
        """ plot Signal samples """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import axes3d
            from matplotlib.widgets import Slider
        except ImportError:
            warnings.warn("Signal plotting requires matplotlib")
            return

        if len(self.samples.shape) <= 1 and self.samples.dtype.names is None:
            fig = plt.figure()
            fig.canvas.set_window_title(self.name)
            fig.text(0.95, 0.05, 'asammdf {}'.format(__version__),
                     fontsize=8, color='red',
                     ha='right', va='top', alpha=0.5)

            if self.comment:
                comment = self.comment.replace('$', '')
                plt.title('{}\n({})'.format(self.name, comment))
            else:
                plt.title(self.name)
            plt.xlabel('Time [s]')
            plt.ylabel('[{}]'.format(self.unit))
            plt.plot(self.timestamps, self.samples, 'b')
            plt.plot(self.timestamps, self.samples, 'b.')
            plt.grid(True)
            plt.show()
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
                        'asammdf {}'.format(__version__),
                        fontsize=8,
                        color='red',
                        ha='right',
                        va='top',
                        alpha=0.5,
                    )

                    if self.comment:
                        comment = self.comment.replace('$', '')
                        plt.title('{}\n({})'.format(self.name, comment))
                    else:
                        plt.title(self.name)

                    ax = fig.add_subplot(111, projection='3d')

                    # Grab some test data.
                    X = np.array(range(shape[1]))
                    Y = np.array(range(shape[0]))
                    X, Y = np.meshgrid(X, Y)

                    Z = samples[0]

                    # Plot a basic wireframe.
                    self._plot_axis = ax.plot_wireframe(X, Y, Z,
                                                        rstride=1, cstride=1)

                    # Place Sliders on Graph
                    ax_a = plt.axes([0.25, 0.1, 0.65, 0.03])

                    # Create Sliders & Determine Range
                    sa = Slider(ax_a,
                                'Time [s]',
                                self.timestamps[0],
                                self.timestamps[-1],
                                valinit=self.timestamps[0])

                    def update(val):
                        self._plot_axis.remove()
                        idx = np.searchsorted(self.timestamps,
                                              sa.val,
                                              side='right')
                        Z = samples[idx-1]
                        self._plot_axis = ax.plot_wireframe(
                            X,
                            Y,
                            Z,
                            rstride=1,
                            cstride=1,
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
                        'asammdf {}'.format(__version__),
                        fontsize=8,
                        color='red',
                        ha='right',
                        va='top',
                        alpha=0.5,
                    )

                    if self.comment:
                        comment = self.comment.replace('$', '')
                        plt.title('{}\n({})'.format(self.name, comment))
                    else:
                        plt.title(self.name)

                    ax = fig.add_subplot(111, projection='3d')

                    samples = self.samples[names[0]]
                    axis1 = self.samples[names[1]]
                    axis2 = self.samples[names[2]]

                    # Grab some test data.
                    X, Y = np.meshgrid(axis2[0], axis1[0])

                    Z = samples[0]

                    # Plot a basic wireframe.
                    self._plot_axis = ax.plot_wireframe(
                        X,
                        Y,
                        Z,
                        rstride=1,
                        cstride=1,
                    )

                    # Place Sliders on Graph
                    ax_a = plt.axes([0.25, 0.1, 0.65, 0.03])

                    # Create Sliders & Determine Range
                    sa = Slider(ax_a,
                                'Time [s]',
                                self.timestamps[0],
                                self.timestamps[-1],
                                valinit=self.timestamps[0])

                    def update(val):
                        self._plot_axis.remove()
                        idx = np.searchsorted(self.timestamps,
                                              sa.val,
                                              side='right')
                        Z = samples[idx-1]
                        X, Y = np.meshgrid(axis2[idx-1], axis1[idx-1])
                        self._plot_axis = ax.plot_wireframe(
                            X,
                            Y,
                            Z,
                            rstride=1,
                            cstride=1,
                        )
                        fig.canvas.draw_idle()

                    sa.on_changed(update)

                    plt.show()

            except Exception as err:
                print(err)

    def cut(self, start=None, stop=None):
        """
        Cuts the signal according to the *start* and *stop* values, by using
        the insertion indexes in the signal's *time* axis.

        Parameters
        ----------
        start : float
            start timestamp for cutting
        stop : float
            stop timestamp for cutting

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

        if start is None and stop is None:
            # return the channel uncut
            result = self

        else:
            if start is None:
                # cut from beggining to stop
                stop = np.searchsorted(self.timestamps, stop, side='right')
                if stop:
                    result = Signal(
                        self.samples[: stop],
                        self.timestamps[:stop],
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                    )
                else:
                    result = Signal(
                        np.array([]),
                        np.array([]),
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                    )

            elif stop is None:
                # cut from start to end
                start = np.searchsorted(self.timestamps, start, side='left')
                result = Signal(
                    self.samples[start:],
                    self.timestamps[start:],
                    self.unit,
                    self.name,
                    self.conversion,
                    self.comment,
                    self.raw,
                )

            else:
                # cut between start and stop
                start_ = np.searchsorted(self.timestamps, start, side='left')
                stop_ = np.searchsorted(self.timestamps, stop, side='right')
                if stop_ == start_:

                    if (len(self.timestamps)
                            and stop >= self.timestamps[0]
                            and start <= self.timestamps[-1]):
                        # start and stop are found between 2 signal samples
                        # so return the previous sample
                        result = Signal(
                            self.samples[start_: start_ + 1],
                            self.timestamps[start_: start_ + 1],
                            self.unit,
                            self.name,
                            self.conversion,
                            self.comment,
                            self.raw,
                        )
                    else:
                        # signal is empty or start and stop are outside the
                        # signal time base
                        result = Signal(
                            np.array([]),
                            np.array([]),
                            self.unit,
                            self.name,
                            self.conversion,
                            self.comment,
                            self.raw,
                        )
                else:
                    result = Signal(
                        self.samples[start_: stop_],
                        self.timestamps[start_: stop_],
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
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
            delta = last_stamp / len(self) + last_stamp
        else:
            last_stamp = 0
            delta = 0
        if len(other):
            other_first_sample = other.timestamps[0]
            if last_stamp >= other_first_sample:
                timestamps = other.timestamps + delta - other_first_sample
            else:
                timestamps = other.timestamps

            result = Signal(
                np.append(self.samples, other.samples),
                np.append(self.timestamps, timestamps),
                self.unit,
                self.name,
                self.conversion,
                self.comment,
                self.raw,
            )
        else:
            result = self

        return result

    def interp(self, new_timestamps):
        """ returns a new *Signal* interpolated using the *new_timestamps*

        Parameters
        ----------
        new_timestamps : np.array
            timestamps used for interpolation

        Returns
        -------
        signal : Signal
            new interpolated *Signal*

        """
        if self.samples.dtype.kind == 'f':
            s = np.interp(new_timestamps, self.timestamps, self.samples)
        else:
            idx = np.searchsorted(
                self.timestamps,
                new_timestamps,
                side='right',
            )
            idx -= 1
            idx = np.clip(idx, 0, idx[-1])
            s = self.samples[idx]
        return Signal(
            s,
            new_timestamps,
            self.unit,
            self.name,
            self.conversion,
            self.raw,
        )

    def __apply_func(self, other, func_name):
        """ delegate operations to the *samples* attribute, but in a time
        correct manner by considering the *timestamps*

        """

        if isinstance(other, Signal):
            time = np.union1d(self.timestamps, other.timestamps)
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
            s,
            time,
            self.unit,
            self.name,
            self.conversion,
            self.raw,
        )

    def __pos__(self):
        return self

    def __neg__(self):
        return Signal(
            np.negative(self.samples),
            self.timestamps,
            self.unit,
            self.name,
            self.conversion,
            self.raw,
        )

    def __round__(self, n):
        return Signal(
            np.around(self.samples, n),
            self.timestamps,
            self.unit,
            self.name,
            self.conversion,
            self.raw,
        )

    def __sub__(self, other):
        return self.__apply_func(other, '__sub__')

    def __isub__(self, other):
        return self.__sub__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __add__(self, other):
        return self.__apply_func(other, '__add__')

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return self.__apply_func(other, '__mul__')

    def __imul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__apply_func(other, '__truediv__')

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __rtruediv__(self, other):
        return self.__apply_func(other, '__rtruediv__')

    def __mod__(self, other):
        return self.__apply_func(other, '__mod__')

    def __pow__(self, other):
        return self.__apply_func(other, '__pow__')

    def __and__(self, other):
        return self.__apply_func(other, '__and__')

    def __or__(self, other):
        return self.__apply_func(other, '__or__')

    def __xor__(self, other):
        return self.__apply_func(other, '__xor__')

    def __invert__(self):
        s = ~self.samples
        time = self.timestamps
        return Signal(
            s,
            time,
            self.unit,
            self.name,
            self.conversion,
            self.raw,
        )

    def __lshift__(self, other):
        return self.__apply_func(other, '__lshift__')

    def __rshift__(self, other):
        return self.__apply_func(other, '__rshift__')

    def __lt__(self, other):
        return self.__apply_func(other, '__lt__')

    def __le__(self, other):
        return self.__apply_func(other, '__le__')

    def __gt__(self, other):
        return self.__apply_func(other, '__gt__')

    def __ge__(self, other):
        return self.__apply_func(other, '__ge__')

    def __eq__(self, other):
        return self.__apply_func(other, '__eq__')

    def __ne__(self, other):
        return self.__apply_func(other, '__ne__')

    def __iter__(self):
        for item in (
                self.samples,
                self.timestamps,
                self.unit,
                self.name):
            yield item

    def __reversed__(self):
        return enumerate(zip(reversed(self.samples),
                             reversed(self.timestamps)))

    def __len__(self):
        return len(self.samples)

    def __abs__(self):
        return Signal(
            np.fabs(self.samples),
            self.timestamps,
            self.unit,
            self.name,
            self.conversion,
            self.raw,
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
            self.unit,
            self.name,
            self.conversion,
            self.raw,
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
        else:
            conversion = self.conversion
            conv_type = conversion['type']
            if conv_type == SignalConversions.CONVERSION_LINEAR:
                samples = self.samples * conversion['a'] + conversion['b']

            elif conv_type == SignalConversions.CONVERSION_TABI:
                samples = np.interp(
                    self.samples,
                    conversion['raw'],
                    conversion['phys'],
                )

            elif conv_type == SignalConversions.CONVERSION_TAB:
                samples = np.interp(
                    self.samples,
                    conversion['raw'],
                    conversion['phys'],
                )

                idx = np.searchsorted(
                    conversion['raw'],
                    self.samples,
                )
                idx = np.clip(
                    idx,
                    0,
                    len(conversion['raw']) - 1,
                )
                samples = conversion['phys'][idx]

            elif conv_type == SignalConversions.CONVERSION_RATIONAL:
                P1 = conversion['P1']
                P2 = conversion['P2']
                P3 = conversion['P3']
                P4 = conversion['P4']
                P5 = conversion['P5']
                P6 = conversion['P6']
                coefs = (P2, P3, P5, P6)
                if coefs == (0, 0, 0, 0):
                    if P1 != P4:
                        samples = evaluate('P4 * X / P1')
                else:
                    samples = evaluate('(P2 - (P4 * (X - P5 -P6))) / (P3* (X - P5 - P6) - P1)')

            elif conv_type == SignalConversions.CONVERSION_POLYNOMIAL:
                P1 = conversion['P1']
                P2 = conversion['P2']
                P3 = conversion['P3']
                P4 = conversion['P4']
                P5 = conversion['P5']
                P6 = conversion['P6']
                if (P1, P2, P3, P4, P5, P6) != (0, 1, 0, 0, 0, 1):
                    X = self.samples
                    samples = evaluate('(P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)')
                else:
                    samples = self.samples.copy()

            elif conv_type in (
                    SignalConversions.CONVERSION_EXPO,
                    SignalConversions.CONVERSION_LOGH):
                P1 = conversion['P1']
                P2 = conversion['P2']
                P3 = conversion['P3']
                P4 = conversion['P4']
                P5 = conversion['P5']
                P6 = conversion['P6']
                P7 = conversion['P7']

                if conv_type == SignalConversions.CONVERSION_EXPO:
                    func = np.log
                else:
                    func = np.exp

                if P4 == 0:
                    samples = func(((self.samples - P7) * P6 - P3) / P1) / P2
                elif P1 == 0:
                    samples = func((P3 / (self.samples - P7) - P6) / P4) / P5
                else:
                    message = 'wrong conversion {}'
                    message = message.format(conversion)
                    raise ValueError(message)

            elif conv_type == SignalConversions.CONVERSION_TABX:
                phys = np.insert(conversion['phys'], 0, conversion['default'])
                raw = np.insert(conversion['raw'], 0, conversion['raw'][0] - 1)
                indexes = np.searchsorted(raw, self.samples)
                np.place(indexes, indexes >= len(raw), 0)

                samples = phys[indexes]

            elif conv_type == SignalConversions.CONVERSION_ALGEBRAIC:
                formula = conversion['formula']

                if 'X1' in formula:
                    X1 = self.samples
                    samples = evaluate(formula)
                else:
                    X = self.samples
                    samples = evaluate(formula)

            else:
                samples = self.samples.copy()

        return Signal(
            samples,
            self.timestamps.copy(),
            unit=self.unit,
            name=self.name,
            raw=False,
            comment=self.comment,
        )


if __name__ == '__main__':
    pass

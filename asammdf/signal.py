# -*- coding: utf-8 -*-
""" asammdf *Signal* class module for time correct signal processing """

import logging
from textwrap import fill

import numpy as np

from .utils import MdfException, extract_cncomment_xml
from . import v2_v3_blocks as v3b
from . import v4_constants as v4c
from . import v4_blocks as v4b

from .version import __version__

logger = logging.getLogger('asammdf')


class Signal(object):
    """
    The *Signal* represents a channel described by it's samples and timestamps.
    It can perform arithmetic operations against other *Signal* or numeric types.
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

    """

    def __init__(self,
                 samples=None,
                 timestamps=None,
                 unit='',
                 name='',
                 conversion=None,
                 comment='',
                 raw=True,
                 master_metadata=None,
                 display_name='',
                 attachment=(),
                 source=None,
                 bit_count=None):

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
                message = '{} samples and timestamps length mismatch ({} vs {})'
                message = message.format(
                    name,
                    samples.shape[0],
                    timestamps.shape[0],
                )
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
            self.source = source
            if bit_count is None:
                self.bit_count = samples.dtype.itemsize * 8
            else:
                self.bit_count = bit_count

            if not isinstance(conversion, (v4b.ChannelConversion, v3b.ChannelConversion)):
                if conversion is None:
                    pass

                elif 'a' in conversion:
                    conversion['conversion_type'] = v4c.CONVERSION_TYPE_LIN
                    conversion = v4b.ChannelConversion(
                        **conversion
                    )

                elif 'formula' in conversion:
                    conversion['conversion_type'] = v4c.CONVERSION_TYPE_ALG
                    conversion = v4b.ChannelConversion(
                        **conversion
                    )

                elif all(
                        key in conversion
                        for key in ['P{}'.format(i) for i in range(1, 7)]):
                    conversion['conversion_type'] = v4c.CONVERSION_TYPE_RAT
                    conversion = v4b.ChannelConversion(
                        **conversion
                    )

                elif 'raw_0' in conversion and 'phys_0' in conversion:
                    conversion['conversion_type'] = v4c.CONVERSION_TYPE_TAB
                    nr = 0
                    while 'phys_{}'.format(nr) in conversion:
                        nr += 1
                    conversion['val_param_nr'] = nr * 2
                    conversion = v4b.ChannelConversion(
                        **conversion
                    )

                elif 'upper_0' in conversion and 'phys_0' in conversion:
                    conversion['conversion_type'] = v4c.CONVERSION_TYPE_RTAB
                    nr = 0
                    while 'phys_{}'.format(nr) in conversion:
                        nr += 1
                    conversion['val_param_nr'] = nr * 3 + 1
                    conversion = v4b.ChannelConversion(
                        **conversion
                    )

                elif 'val_0' in conversion and 'text_0' in conversion:
                    conversion['conversion_type'] = v4c.CONVERSION_TYPE_TABX
                    nr = 0
                    while 'text_{}'.format(nr) in conversion:
                        nr += 1
                    conversion['ref_param_nr'] = nr + 1
                    conversion = v4b.ChannelConversion(
                        **conversion
                    )

                elif 'upper_0' in conversion and 'text_0' in conversion:
                    conversion['conversion_type'] = v4c.CONVERSION_TYPE_RTABX
                    nr = 0
                    while 'text_{}'.format(nr) in conversion:
                        nr += 1
                    conversion['ref_param_nr'] = nr + 1
                    conversion = v4b.ChannelConversion(
                        **conversion
                    )

                else:
                    conversion = v4b.ChannelConversion(
                        conversion_type=v4c.CONVERSION_TYPE_NON
                    )

            self.conversion = conversion

    def __repr__(self):
        string = """<Signal {}:
\tsamples={}
\ttimestamps={}
\tunit="{}"
\tconversion={}
\tsource={}
\tcomment="{}"
\tmastermeta="{}"
\traw={}
\tdisplay_name={}>
"""
        return string.format(
            self.name,
            self.samples,
            self.timestamps,
            self.unit,
            self.conversion,
            self.source,
            self.comment,
            self.master_metadata,
            self.raw,
            self.display_name,
        )

    def plot(self):
        """ plot Signal samples """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import axes3d
            from matplotlib.widgets import Slider
        except ImportError:
            logging.warning("Signal plotting requires matplotlib")
            return

        if len(self.samples.shape) <= 1 and self.samples.dtype.names is None:
            fig = plt.figure()
            fig.canvas.set_window_title(self.name)
            fig.text(0.95, 0.05, 'asammdf {}'.format(__version__),
                     fontsize=8, color='red',
                     ha='right', va='top', alpha=0.5)

            name = self.name

            if self.comment:
                comment = self.comment.replace('$', '')
                comment = extract_cncomment_xml(comment)
                comment = fill(comment, 120).replace('\\n', ' ')

                title = '{}\n({})'.format(name, comment)
                plt.title(title)
            else:
                plt.title(name)
            try:
                if not self.master_metadata:
                    plt.xlabel('Time [s]')
                    plt.ylabel('[{}]'.format(self.unit))
                    plt.plot(self.timestamps, self.samples, 'b')
                    plt.plot(self.timestamps, self.samples, 'b.')
                    plt.grid(True)
                    plt.show()
                else:
                    master_name, sync_type = self.master_metadata
                    if sync_type in (0, 1):
                        plt.xlabel('{} [s]'.format(master_name))
                    elif sync_type == 2:
                        plt.xlabel('{} [deg]'.format(master_name))
                    elif sync_type == 3:
                        plt.xlabel('{} [m]'.format(master_name))
                    elif sync_type == 4:
                        plt.xlabel('{} [index]'.format(master_name))
                    plt.ylabel('[{}]'.format(self.unit))
                    plt.plot(self.timestamps, self.samples, 'b')
                    plt.plot(self.timestamps, self.samples, 'b.')
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
        if len(self) == 0:
            result = Signal(
                np.array([]),
                np.array([]),
                self.unit,
                self.name,
                self.conversion,
                self.comment,
                self.raw,
                self.master_metadata,
                self.display_name,
                self.attachment,
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
            )

        else:
            if start is None:
                # cut from begining to stop
                stop = np.searchsorted(self.timestamps, stop, side='right')
                if stop:
                    result = Signal(
                        self.samples[:stop],
                        self.timestamps[:stop],
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_name,
                        self.attachment,
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
                        self.master_metadata,
                        self.display_name,
                        self.attachment,
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
                    self.master_metadata,
                    self.display_name,
                    self.attachment,
                )

            else:
                # cut between start and stop
                if start > self.timestamps[-1]:
                    result = Signal(
                        np.array([]),
                        np.array([]),
                        self.unit,
                        self.name,
                        self.conversion,
                        self.comment,
                        self.raw,
                        self.master_metadata,
                        self.display_name,
                        self.attachment,
                    )
                else:
                    start_ = np.searchsorted(self.timestamps, start, side='left')
                    start_ = max(0, start_)
                    stop_ = np.searchsorted(self.timestamps, stop, side='right')

                    if start not in self.timestamps and start_ == stop_:
                        start_ -= 1
                    if stop_ == start_:
                        if (len(self.timestamps)
                                and stop >= self.timestamps[0]
                                and start <= self.timestamps[-1]):
                            # start and stop are found between 2 signal samples
                            # so return the previous sample
                            result = Signal(
                                self.samples[start_ - 1: start_],
                                self.timestamps[start_ - 1: start_],
                                self.unit,
                                self.name,
                                self.conversion,
                                self.comment,
                                self.raw,
                                self.master_metadata,
                                self.display_name,
                                self.attachment,
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
                                self.master_metadata,
                                self.display_name,
                                self.attachment,
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
                            self.master_metadata,
                            self.display_name,
                            self.attachment,
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
                self.master_metadata,
                self.display_name,
                self.attachment,
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
        if not len(self.samples) or not len(new_timestamps):
            return Signal(
                self.samples.copy(),
                self.timestamps.copy(),
                self.unit,
                self.name,
                comment=self.comment,
                conversion=self.conversion,
                raw=self.raw,
                master_metadata=self.master_metadata,
                display_name=self.display_name,
                attachment=self.attachment,
            )
        else:
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
                comment=self.comment,
                conversion=self.conversion,
                raw=self.raw,
                master_metadata=self.master_metadata,
                display_name=self.display_name,
                attachment=self.attachment,
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
            self.master_metadata,
            self.display_name,
            attachment=self.attachment,
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
            self.master_metadata,
            self.display_name,
            self.attachment,
        )

    def __round__(self, n):
        return Signal(
            np.around(self.samples, n),
            self.timestamps,
            self.unit,
            self.name,
            self.conversion,
            self.raw,
            self.master_metadata,
            self.display_name,
            self.attachment,
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
            self.master_metadata,
            self.display_name,
            self.attachment,
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
            self.master_metadata,
            self.display_name,
            self.attachment,
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
            self.master_metadata,
            self.display_name,
            self.attachment,
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
            samples = self.conversion.convert(self.samples)

        return Signal(
            samples,
            self.timestamps.copy(),
            unit=self.unit,
            name=self.name,
            raw=False,
            comment=self.comment,
            master_metadata=self.master_metadata,
            display_name=self.display_name,
            attachment=self.attachment,
        )


if __name__ == '__main__':
    pass

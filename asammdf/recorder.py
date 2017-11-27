# -*- coding: utf-8 -*-
""" MDF version 4 recorder """

from collections import namedtuple
from time import perf_counter
from warnings import warn

from numpy import array, float64

from .mdf import MDF
from .signal import Signal


__all__ = [
    'ChannelDescriptor',
    'Deposit',
    'Recorder',
]


ChannelDescriptor = namedtuple('ChannelDescriptor', ['name', 'unit', 'description', 'dtype'])


class Deposit(list):
    """ list wrapper that keeps track of the relative timestamp when
    appending new samples

    Parameters
    ----------
    dtype : numpy.dtype
        numpy dtype of record

    """

    def __init__(self):
        super(Deposit, self).__init__()

        self.origin = None

    def append(self, samples):
        """ add new samples

        Parameters
        ----------
        samples : list | tuple
            iterable of samples for to be added to this Deposit

        """
        super(Deposit, self).append([perf_counter() - self.origin,] + list(samples))


class Recorder(object):
    """ record MDF version 4 file.

    Parameters
    ----------
    filename : str
        output file name

    """

    def __init__(self, filename):
        self.name = filename
        self.groups = []
        self.start = None
        self.active = False

    def register(self, signals, description):
        """ registers signals for streaming in a data group

        Parameters
        ----------
        signals : list
            list of `ChannelDescriptor`
        description : str
            acquisition description for this channel group

        Returns
        -------
        deposit : list
            list that will hold the data group samples. The caller will use
            this list to append new samples.

        """
        group = {}
        if not signals:
            warn('register called with empty list')
            return
        elif not all(isinstance(signal, ChannelDescriptor) for signal in signals):
            warn('The signal list must contain only ChannelDescriptor namedtuples')
            return
        else:
            group['descriptors'] = signals
            group['deposit'] = Deposit()
            group['description'] = description

            self.groups.append(group)

        return group['deposit']

    def start(self):
        if not self.groups:
            warn('no signals were registered for streaming')
        else:
            self.active = True
            self.start = perf_counter()
            for group in self.groups:
                group['deposit'].origin = self.start

    def stop(self):
        if self.active is False:
            warn('stop was called but the streaming was not previously started')
        else:
            self.active = False

            mdf = MDF(version='4.10')

            for group in self.groups:
                deposit = group['deposit']
                description = group['description']

                t = [samples[0] for samples in deposit]
                t = array(t, dtype=float64)

                signals = []

                for i, descriptor in enumerate(group['descriptors'], 1):
                    values = [samples[i] for samples in deposit]
                    values = array(values, dtype=descriptor.dtype)

                    signal = Signal(
                        samples=values,
                        timestamps=t,
                        unit=descriptor.unit,
                        name=descriptor.name,
                    )

                    signals.append(signal)

                mdf.append(signals, description)

                deposit[:] = []
                signals = []
                t = None

            mdf.save(self.name)

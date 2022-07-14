# -*- coding: utf-8 -*-
""" asammdf virtual channels functions"""
from functools import reduce
import sys

import numpy as np

from .signal import Signal


def generate_from_numpy_function(fname):
    func = getattr(np, fname)

    def retfunc(signal, *args):
        if isinstance(signal, Signal):
            result = signal.copy()
            result.samples = func(result.samples, *args)
        else:
            result = func(signal, *args)
        return result

    retfunc.__name__ = fname
    retfunc.__doc__ = func.__doc__

    return retfunc


def generate_cast_function(fname):
    dtype = getattr(np, fname)

    def retfunc(signal):
        if isinstance(signal, Signal):
            result = signal.copy()
            result.samples = result.samples.astype(dtype)
        else:
            result = signal.astype(dtype)
        return result

    retfunc.__name__ = fname

    return retfunc


def interpolate_signals(signals):
    sig_objects = [
        (i, sig) for (i, sig) in enumerate(signals) if isinstance(sig, Signal)
    ]

    t_min, t_max = -np.inf, np.inf
    for i, sig in sig_objects:
        if not len(sig):
            continue
        if sig.timestamps[0] > t_min:
            t_min = sig.timestamps[0]
        if sig.timestamps[-1] < t_max:
            t_max = sig.timestamps[-1]

    for j, (i, sig) in enumerate(sig_objects):
        sig_objects[j] = i, sig.cut(t_min, t_max)

    timebase = reduce(np.union1d, [sig.t for i, sig in sig_objects])

    for j, (i, sig) in enumerate(sig_objects):
        sig_objects[j] = i, sig.interp(timebase)

    for i, sig in sig_objects:
        signals[i] = sig

    return signals, timebase


for fname in (
    "absolute",
    "arccos",
    "arcsin",
    "arctan",
    "arctan2",
    "around",
    "cbrt",
    "ceil",
    "clip",
    "cos",
    "cumprod",
    "cumsum",
    "deg2rad",
    "degrees",
    "diff",
    "exp",
    "fix",
    "floor",
    "gradient",
    "log",
    "log10",
    "log2",
    "rad2deg",
    "radians",
    "rint",
    "sin",
    "sqrt",
    "square",
    "tan",
    "trunc",
    "sign",
    "max",
    "min",
):
    func = generate_from_numpy_function(fname)
    globals()[fname] = func

for fname in (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
):
    func = generate_cast_function(fname)
    globals()[fname] = func


def average(*args):
    if len(args):
        return sum(args) / len(args)
    else:
        return 0


def maximum(*signals):
    if len(signals) == 1 and isinstance(signals[0], (list, tuple)):
        signals = signals[0]

    signals, timebase = interpolate_signals(signals)

    signals = [sig.samples if isinstance(sig, Signal) else sig for sig in signals]

    result = Signal(reduce(np.maximum, signals), timebase, name="result")

    return result


def minimum(*signals):
    if len(signals) == 1 and isinstance(signals[0], (list, tuple)):
        signals = signals[0]

    signals, timebase = interpolate_signals(signals)

    signals = [sig.samples if isinstance(sig, Signal) else sig for sig in signals]

    result = Signal(reduce(np.minimum, signals), timebase, name="result")

    return result

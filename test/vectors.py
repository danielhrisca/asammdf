# -*- coding: utf-8 -*-
import numpy as np

from asammdf.signal import Signal


# create 3 Signal objects
timestamps = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

# unit8
s_uint8 = Signal(samples=np.array([0, 1, 2, 3, 4], dtype=np.uint8),
                 timestamps=timestamps,
                 name='Uint8_Signal',
                 unit='u1',
                 info='s_uint8 info',
                 comment='s_uint8 comment')
# int32
s_int32 = Signal(samples=np.array([-20, -10, 0, 10, 20], dtype=np.int32),
                 timestamps=timestamps,
                 name='Int32_Signal',
                 unit='i4',
                 info='s_int32 info',
                 comment='s_int32 comment')

# float64
s_float64 = Signal(samples=np.array([-20, -10, 0, 10, 20], dtype=np.float64),
                   timestamps=timestamps,
                   name='Float64_Signal',
                   unit='f8',
                   info='s_float64 info',
                   comment='s_float64 comment')

# create signal list
signals = [s_uint8, s_int32, s_float64]

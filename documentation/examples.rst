.. raw:: html

    <style> .red {color:red} </style>
    <style> .blue {color:blue} </style>
    <style> .green {color:green} </style>
    <style> .cyan {color:cyan} </style>
    <style> .magenta {color:magenta} </style>
    <style> .orange {color:orange} </style>
    <style> .brown {color:brown} </style>
    
.. role:: red
.. role:: blue
.. role:: green
.. role:: cyan
.. role:: magenta
.. role:: orange
.. role:: brown

.. _examples:
Examples
========

Working with MDF
----------------

.. code-block:: python

    from asammdf import MDF, Signal
    import numpy as np


    # create 3 Signal objects

    timestamps = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

    # unit8
    s_uint8 = Signal(samples=np.array([0, 1, 2, 3, 4], dtype=np.uint8),
                     timestamps=timestamps,
                     name='Uint8_Signal',
                     unit='u1')
    # int32
    s_int32 = Signal(samples=np.array([-20, -10, 0, 10, 20], dtype=np.int32),
                     timestamps=timestamps,
                     name='Int32_Signal',
                     unit='i4')

    # float64
    s_float64 = Signal(samples=np.array([-20, -10, 0, 10, 20], dtype=np.int32),
                       timestamps=timestamps,
                       name='Float64_Signal',
                       unit='f8')

    # create empty MDf version 4.00 file
    mdf4 = MDF(version='4.00')

    # append the 3 signals to the new file
    signals = [s_uint8, s_int32, s_float64]
    mdf4.append(signals, 'Created by Python')

    # save new file
    mdf4.save('my_new_file.mf4')

    # convert new file to mdf version 3.10 with compression of raw channel data
    mdf3 = mdf4.convert(to='3.10', compression=True)
    print(mdf3.version)
    # prints >>> 3.10

    # get the float signal
    sig = mdf3.get('Float64_Signal')
    print(sig)
    # prints >>> Signal { name="Float64_Signal":	s=[-20 -10   0  10  20]	t=[ 0.1         0.2         0.30000001  0.40000001  0.5       ]	unit="f8"	conversion=None }

    
Working with Signal
-------------------
    
.. code-block:: python

    from asammdf import Signal
    import numpy as np


    # create 3 Signal objects with different time stamps

    # unit8 with 100ms time raster
    timestamps = np.array([0.1 * t for t in range(5)], dtype=np.float32)
    s_uint8 = Signal(samples=np.array([t for t in range(5)], dtype=np.uint8),
                     timestamps=timestamps,
                     name='Uint8_Signal',
                     unit='u1')

    # int32 with 50ms time raster
    timestamps = np.array([0.05 * t for t in range(10)], dtype=np.float32)
    s_int32 = Signal(samples=np.array(list(range(-500, 500, 100)), dtype=np.int32),
                     timestamps=timestamps,
                     name='Int32_Signal',
                     unit='i4')

    # float64 with 300ms time raster
    timestamps = np.array([0.3 * t for t in range(3)], dtype=np.float32)
    s_float64 = Signal(samples=np.array(list(range(2000, -1000, -1000)), dtype=np.int32),
                       timestamps=timestamps,
                       name='Float64_Signal',
                       unit='f8')

    prod = s_float64 * s_uint8
    prod.name = 'Uint8_Signal * Float64_Signal'
    prod.unit = '*'
    prod.plot()

    pow2 = s_uint8 ** 2
    pow2.name = 'Uint8_Signal ^ 2'
    pow2.unit = 'u1^2'
    pow2.plot()

    allsum = s_uint8 + s_int32 + s_float64
    allsum.name = 'Uint8_Signal + Int32_Signal + Float64_Signal'
    allsum.unit = '+'
    allsum.plot()

    # inplace operations
    pow2 *= -1
    pow2.name = '- Uint8_Signal ^ 2'
    pow2.plot()


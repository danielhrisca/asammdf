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

    from __future__ import print_function, division
    from asammdf import MDF, Signal, configure
    import numpy as np

    # configure asammdf to optimize disk space usage
    configure(integer_compacting=True)
    # configure asammdf to split data blocks on 10KB blocks
    configure(split_data_blocks=True, split_threshold=10*1024)


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
    s_float64 = Signal(samples=np.array([-20, -10, 0, 10, 20], dtype=np.float64),
                       timestamps=timestamps,
                       name='Float64_Signal',
                       unit='f8')

    # create empty MDf version 4.00 file
    mdf4 = MDF(version='4.10')

    # append the 3 signals to the new file
    signals = [s_uint8, s_int32, s_float64]
    mdf4.append(signals, 'Created by Python')

    # save new file
    mdf4.save('my_new_file.mf4', overwrite=True)

    # convert new file to mdf version 3.10 with lower possible RAM usage
    mdf3 = mdf4.convert(to='3.10', memory='minimum')
    print(mdf3.version)

    # get the float signal
    sig = mdf3.get('Float64_Signal')
    print(sig)

    # cut measurement from 0.3s to end of measurement
    mdf4_cut = mdf4.cut(start=0.3)
    mdf4_cut.get('Float64_Signal').plot()

    # cut measurement from start of measurement to 0.4s
    mdf4_cut = mdf4.cut(stop=0.45)
    mdf4_cut.get('Float64_Signal').plot()

    # filter some signals from the file
    mdf4 = mdf4.filter(['Int32_Signal', 'Uint8_Signal'])

    # save using zipped transpose deflate blocks
    mdf4.save('out.mf4', compression=2, overwrite=True)


Working with Signal
-------------------

.. code-block:: python

    from __future__ import print_function, division
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

    # map signals
    xs = np.linspace(-1, 1, 50)
    ys = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(xs, ys)
    vals = np.linspace(0, 180. / np.pi, 100)
    phi = np.ones((len(vals), 50, 50), dtype=np.float64)
    for i, val in enumerate(vals):
        phi[i] *= val
    R = 1 - np.sqrt(X**2 + Y**2)
    samples = np.cos(2 * np.pi * X + phi) * R
    print(phi.shape, samples.shape)
    timestamps = np.arange(0, 2, 0.02)

    s_map = Signal(samples=samples,
                   timestamps=timestamps,
                   name='Variable Map Signal',
                   unit='dB')
    s_map.plot()


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

    # cut signal
    s_int32.plot()
    cut_signal = s_int32.cut(start=0.2, stop=0.35)
    cut_signal.plot()


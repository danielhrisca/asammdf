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

--------
Examples
--------

Working with MDF
================

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
    s_float64 = Signal(samples=np.array([-20, -10, 0, 10, 20], dtype=np.float64),
                       timestamps=timestamps,
                       name='Float64_Signal',
                       unit='f8')

    # create empty MDf version 4.00 file
    with MDF(version='4.10') as mdf4:

        # append the 3 signals to the new file
        signals = [s_uint8, s_int32, s_float64]
        mdf4.append(signals, comment='Created by Python')

        # save new file
        mdf4.save('my_new_file.mf4', overwrite=True)

        # convert new file to mdf version 3.10
        mdf3 = mdf4.convert(version='3.10')
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
===================

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
    prod *= -1
    prod.name = '- Uint8_Signal * Float64_Signal'
    prod.plot()

    # cut signal
    s_int32.plot()
    cut_signal = s_int32.cut(start=0.2, stop=0.35)
    cut_signal.plot()
    
    
MF4 demo file generator
=======================

.. code-block:: python

    from asammdf import MDF, SUPPORTED_VERSIONS, Signal
    import numpy as np

    cycles = 100
    sigs = []

    mdf = MDF()

    t = np.arange(cycles, dtype=np.float64)

    # no conversion
    sig = Signal(
        np.ones(cycles, dtype=np.uint64),
        t,
        name='Channel_no_conversion',
        unit='s',
        conversion=None,
        comment='Unsigned 64 bit channel {}',
    )
    sigs.append(sig)

    # linear
    conversion = {
        'a': 2,
        'b': -0.5,
    }
    sig = Signal(
        np.ones(cycles, dtype=np.int64),
        t,
        name='Channel_linear_conversion',
        unit='Nm',
        conversion=conversion,
        comment='Signed 64bit channel with linear conversion',
    )
    sigs.append(sig)


    # algebraic
    conversion = {
        'formula': '2 * sin(X)',
    }
    sig = Signal(
        np.arange(cycles, dtype=np.int32) / 100.0,
        t,
        name='Channel_algebraic',
        unit='eV',
        conversion=conversion,
        comment='Sinus channel with algebraic conversion',
    )
    sigs.append(sig)

    # rational
    conversion = {
        'P1': 0,
        'P2': 4,
        'P3': -0.5,
        'P4': 0,
        'P5': 0,
        'P6': 1,
    }
    sig = Signal(
        np.ones(cycles, dtype=np.int64),
        t,
        name='Channel_rational_conversion',
        unit='Nm',
        conversion=conversion,
        comment='Channel with rational conversion',
    )
    sigs.append(sig)

    # string channel
    sig = [
        'String channel sample {}'.format(j).encode('ascii')
        for j in range(cycles)
    ]
    sig = Signal(
        np.array(sig),
        t,
        name='Channel_string',
        comment='String channel',
        encoding='latin-1',
    )
    sigs.append(sig)

    # byte array
    ones = np.ones(cycles, dtype=np.dtype('(8,)u1'))
    sig = Signal(
        ones*111,
        t,
        name='Channel_bytearay',
        comment='Byte array channel',
    )
    sigs.append(sig)

    # tabular
    vals = 20
    conversion = {
        'raw_{}'.format(i): i
        for i in range(vals)
    }
    conversion.update(
        {
            'phys_{}'.format(i): -i
            for i in range(vals)
        }
    )
    sig = Signal(
        np.arange(cycles, dtype=np.uint32) % 20,
        t,
        name='Channel_tabular',
        unit='-',
        conversion=conversion,
        comment='Tabular channel',
    )
    sigs.append(sig)

    # value to text
    vals = 20
    conversion = {
        'val_{}'.format(i): i
        for i in range(vals)
    }
    conversion.update(
        {
            'text_{}'.format(i): 'key_{}'.format(i).encode('ascii')
            for i in range(vals)
        }
    )
    conversion['default'] = b'default key'
    sig = Signal(
        np.arange(cycles, dtype=np.uint32) % 30,
        t,
        name='Channel_value_to_text',
        conversion=conversion,
        comment='Value to text channel',
    )
    sigs.append(sig)

    # tabular with range
    vals = 20
    conversion = {
        'lower_{}'.format(i): i * 10
        for i in range(vals)
    }
    conversion.update(
        {
            'upper_{}'.format(i): (i + 1) * 10
            for i in range(vals)
        }
    )
    conversion.update(
        {
            'phys_{}'.format(i): i
            for i in range(vals)
        }
    )
    conversion['default'] = -1
    sig = Signal(
        2 * np.arange(cycles, dtype=np.float64),
        t,
        name='Channel_value_range_to_value',
        unit='order',
        conversion=conversion,
        comment='Value range to value channel',
    )
    sigs.append(sig)

    # value range to text
    vals = 20
    conversion = {
        'lower_{}'.format(i): i * 10
        for i in range(vals)
    }
    conversion.update(
        {
            'upper_{}'.format(i): (i + 1) * 10
            for i in range(vals)
        }
    )
    conversion.update(
        {
            'text_{}'.format(i): 'Level {}'.format(i)
            for i in range(vals)
        }
    )
    conversion['default'] = b'Unknown level'
    sig = Signal(
        6 * np.arange(cycles, dtype=np.uint64) % 240,
        t,
        name='Channel_value_range_to_text',
        conversion=conversion,
        comment='Value range to text channel',
    )
    sigs.append(sig)


    mdf.append(sigs, comment='single dimensional channels', common_timebase=True)



    sigs = []

    # lookup tabel with axis
    samples = [
        np.ones((cycles, 2, 3), dtype=np.uint64) * 1,
        np.ones((cycles, 2), dtype=np.uint64) * 2,
        np.ones((cycles, 3), dtype=np.uint64) * 3,
    ]

    types = [
        ('Channel_lookup_with_axis', '(2, 3)<u8'),
        ('channel_axis_1', '(2, )<u8'),
        ('channel_axis_2', '(3, )<u8'),
    ]

    sig = Signal(
        np.rec.fromarrays(samples, dtype=np.dtype(types)),
        t,
        name='Channel_lookup_with_axis',
        unit='A',
        comment='Array channel with axis',
    )
    sigs.append(sig)

    # lookup tabel with default axis
    samples = [
        np.ones((cycles, 2, 3), dtype=np.uint64) * 4,
    ]

    types = [
        ('Channel_lookup_with_default_axis', '(2, 3)<u8'),
    ]

    sig = Signal(
        np.rec.fromarrays(samples, dtype=np.dtype(types)),
        t,
        name='Channel_lookup_with_default_axis',
        unit='mA',
        comment='Array channel with default axis',
    )
    sigs.append(sig)

    # structure channel composition
    samples = [
        np.ones(cycles, dtype=np.uint8) * 10,
        np.ones(cycles, dtype=np.uint16) * 20,
        np.ones(cycles, dtype=np.uint32) * 30,
        np.ones(cycles, dtype=np.uint64) * 40,
        np.ones(cycles, dtype=np.int8) * -10,
        np.ones(cycles, dtype=np.int16) * -20,
        np.ones(cycles, dtype=np.int32) * -30,
        np.ones(cycles, dtype=np.int64) * -40,
    ]

    types = [
        ('struct_channel_0', np.uint8),
        ('struct_channel_1', np.uint16),
        ('struct_channel_2', np.uint32),
        ('struct_channel_3', np.uint64),
        ('struct_channel_4', np.int8),
        ('struct_channel_5', np.int16),
        ('struct_channel_6', np.int32),
        ('struct_channel_7', np.int64),
    ]

    sig = Signal(
        np.rec.fromarrays(samples, dtype=np.dtype(types)),
        t,
        name='Channel_structure_composition',
        comment='Structure channel composition',
    )
    sigs.append(sig)


    # nested structures
    l4_arr = [
        np.ones(cycles, dtype=np.float64) * 41,
        np.ones(cycles, dtype=np.float64) * 42,
        np.ones(cycles, dtype=np.float64) * 43,
        np.ones(cycles, dtype=np.float64) * 44,
    ]

    types = [
        ('level41', np.float64),
        ('level42', np.float64),
        ('level43', np.float64),
        ('level44', np.float64),
    ]

    l4_arr = np.rec.fromarrays(l4_arr, dtype=types)

    l3_arr = [
        l4_arr,
        l4_arr,
        l4_arr,
    ]

    types = [
        ('level31', l4_arr.dtype),
        ('level32', l4_arr.dtype),
        ('level33', l4_arr.dtype),
    ]

    l3_arr = np.rec.fromarrays(l3_arr, dtype=types)


    l2_arr = [
        l3_arr,
        l3_arr,
    ]

    types = [
        ('level21', l3_arr.dtype),
        ('level22', l3_arr.dtype),
    ]

    l2_arr = np.rec.fromarrays(l2_arr, dtype=types)


    l1_arr = [
        l2_arr,
    ]

    types = [
        ('level11', l2_arr.dtype),
    ]

    l1_arr = np.rec.fromarrays(l1_arr, dtype=types)


    sigs.append(
        Signal(
            l1_arr,
            t,
            name='Nested_structures',
        )
    )

    mdf.append(sigs, comment='arrays', common_timebase=True)

    mdf.save('demo.mf4', overwrite=True)




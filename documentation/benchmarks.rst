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

.. _benchmarks:

Intro
-----

The benchmarks were done using two test files (available here https://github.com/danielhrisca/asammdf/issues/14) (for mdf version 3 and 4) of around 170MB.
The files contain 183 data groups and a total of 36424 channels.

*asamdf 2.8.0* was compared against *mdfreader 2.7.2* (latest versions from PyPI).
*mdfreader* seems to be the most used Python package to handle MDF files, and it also supports both version 3 and 4 of the standard.

The three benchmark cathegories are file open, file save and extracting the data for all channels inside the file(36424 calls).
For each cathegory two aspect were noted: elapsed time and peak RAM usage.

Dependencies
------------
You will need the following packages to be able to run the benchmark script

* psutil
* mdfreader

Usage
-----
Extract the test files from the archive, or provide a folder that contains the files "test.mdf" and "test.mf4".
Run the module *bench.py* ( see --help option for available options )


x64 Python results
------------------
Benchmark environment

* 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 18:41:36) [MSC v.1900 64 bit (AMD64)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results

* full =  asammdf MDF object created with memory=full (everything loaded into RAM)
* low =  asammdf MDF object created with memory=low (raw channel data not loaded into RAM, but metadata loaded to RAM)
* minimum =  asammdf MDF object created with memory=full (lowest possible RAM usage)
* compress = mdfreader mdf object created with compression=blosc
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels

Raw data
^^^^^^^^


================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 772      319
asammdf 2.8.0 low mdfv3                                  656      165
asammdf 2.8.0 minimum mdfv3                              441       77
mdfreader 2.7.2 mdfv3                                   1783      428
mdfreader 2.7.2 compress mdfv3                          3330      127
mdfreader 2.7.2 noDataLoading mdfv3                      699      167
asammdf 2.8.0 full mdfv4                                1903      381
asammdf 2.8.0 low mdfv4                                 1783      216
asammdf 2.8.0 minimum mdfv4                             1348       88
mdfreader 2.7.2 mdfv4                                   4849      442
mdfreader 2.7.2 compress mdfv4                          6347      138
mdfreader 2.7.2 noDataLoading mdfv4                     3425      176
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 359      321
asammdf 2.8.0 low mdfv3                                  415      172
asammdf 2.8.0 minimum mdfv3                              993       86
mdfreader 2.7.2 mdfv3                                   8402      456
mdfreader 2.7.2 compress mdfv3                          8364      424
asammdf 2.8.0 full mdfv4                                 497      387
asammdf 2.8.0 low mdfv4                                  507      228
asammdf 2.8.0 minimum mdfv4                             2179       97
mdfreader 2.7.2 mdfv4                                   7958      460
mdfreader 2.7.2 compress mdfv4                          8170      417
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 772      325
asammdf 2.8.0 low mdfv3                                 3784      179
asammdf 2.8.0 minimum mdfv3                             5076       92
mdfreader 2.7.2 mdfv3                                     65      428
mdfreader 2.7.2 nodata mdfv3                            1231      379
mdfreader 2.7.2 compress mdfv3                           487      127
asammdf 2.8.0 full mdfv4                                 800      389
asammdf 2.8.0 low mdfv4                                 7025      226
asammdf 2.8.0 minimum mdfv4                             9518      100
mdfreader 2.7.2 mdfv4                                     71      442
mdfreader 2.7.2 nodata mdfv4                            1575      404
mdfreader 2.7.2 compress mdfv4                           508      145
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full v3 to v4                             3461      751
asammdf 2.8.0 low v3 to v4                              4092      331
asammdf 2.8.0 minimum v3 to v4                          4852      163
asammdf 2.8.0 full v4 to v3                             3732      753
asammdf 2.8.0 low v4 to v3                              4348      313
asammdf 2.8.0 minimum v4 to v3                          7136      134
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full v3                                   8152     1312
asammdf 2.8.0 low v3                                    9839      456
asammdf 2.8.0 minimum v3                               11694      228
mdfreader 2.7.2 v3                                     10352     2927
mdfreader 2.7.2 compress v3                            15314     2940
asammdf 2.8.0 full v4                                  11938     1434
asammdf 2.8.0 low v4                                   13154      549
asammdf 2.8.0 minimum v4                               17188      229
mdfreader 2.7.2 v4                                     16536     2941
mdfreader 2.7.2 compress v4                            21261     2951
================================================== ========= ========



Graphical results
^^^^^^^^^^^^^^^^^

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Open'
    aspect = 'time'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])


    arr = ram if aspect == 'ram' else time


    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Open'
    aspect = 'ram'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Save'
    aspect = 'time'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Save'
    aspect = 'ram'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Get'
    aspect = 'time'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Get'
    aspect = 'ram'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Convert'
    aspect = 'time'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Convert'
    aspect = 'ram'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Merge'
    aspect = 'time'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Merge'
    aspect = 'ram'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


x86 Python results
------------------
Benchmark environment

* 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 17:54:52) [MSC v.1900 32 bit (Intel)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results

* full =  asammdf MDF object created with memory=full (everything loaded into RAM)
* low =  asammdf MDF object created with memory=low (raw channel data not loaded into RAM, but metadata loaded to RAM)
* minimum =  asammdf MDF object created with memory=full (lowest possible RAM usage)
* compress = mdfreader mdf object created with compression=blosc
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



Raw data
^^^^^^^^


================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 918      264
asammdf 2.8.0 low mdfv3                                  898      110
asammdf 2.8.0 minimum mdfv3                              577       56
mdfreader 2.7.2 mdfv3                                   2462      395
mdfreader 2.7.2 compress mdfv3                          4174       97
mdfreader 2.7.2 noDataLoading mdfv3                      911      105
asammdf 2.8.0 full mdfv4                                2644      302
asammdf 2.8.0 low mdfv4                                 2269      137
asammdf 2.8.0 minimum mdfv4                             1883       62
mdfreader 2.7.2 mdfv4                                   5869      403
mdfreader 2.7.2 compress mdfv4                          7367      101
mdfreader 2.7.2 noDataLoading mdfv4                     3897      110
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 452      267
asammdf 2.8.0 low mdfv3                                  495      118
asammdf 2.8.0 minimum mdfv3                             1206       62
mdfreader 2.7.2 mdfv3                                   9258      415
asammdf 2.8.0 full mdfv4                                 642      307
asammdf 2.8.0 low mdfv4                                  693      146
asammdf 2.8.0 minimum mdfv4                             2642       71
mdfreader 2.7.2 mdfv4                                   8548      422
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 889      268
asammdf 2.8.0 low mdfv3                                12707      120
asammdf 2.8.0 minimum mdfv3                            13644       66
mdfreader 2.7.2 mdfv3                                     80      395
mdfreader 2.7.2 nodata mdfv3                            1413      310
mdfreader 2.7.2 compress mdfv3                           529       97
asammdf 2.8.0 full mdfv4                                 968      307
asammdf 2.8.0 low mdfv4                                14475      144
asammdf 2.8.0 minimum mdfv4                            17057       69
mdfreader 2.7.2 mdfv4                                     72      403
mdfreader 2.7.2 nodata mdfv4                            1806      325
mdfreader 2.7.2 compress mdfv4                           562      107
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full v3 to v4                             4048      642
asammdf 2.8.0 low v3 to v4                              4551      219
asammdf 2.8.0 minimum v3 to v4                          5847      121
asammdf 2.8.0 full v4 to v3                             4394      639
asammdf 2.8.0 low v4 to v3                              5239      198
asammdf 2.8.0 minimum v4 to v3                          8392       98
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full v3                                  10061     1168
asammdf 2.8.0 low v3                                   11245      323
asammdf 2.8.0 minimum v3                               13618      186
asammdf 2.8.0 full v4                                  14144     1226
asammdf 2.8.0 low v4                                   15410      355
asammdf 2.8.0 minimum v4                               21417      170
================================================== ========= ========

* mdfreader got a MemoryError

Graphical results
^^^^^^^^^^^^^^^^^

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Open'
    aspect = 'time'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Open'
    aspect = 'ram'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Save'
    aspect = 'time'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Save'
    aspect = 'ram'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Get'
    aspect = 'time'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Get'
    aspect = 'ram'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Convert'
    aspect = 'time'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Convert'
    aspect = 'ram'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Merge'
    aspect = 'time'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.8.0_mdfreader_2.7.2.txt'
    topic = 'Merge'
    aspect = 'ram'
    for_doc = True

    with open(res, 'r') as f:
        lines = f.readlines()

    platform = 'x86' if '32 bit' in lines[2] else 'x64'

    idx = [i for i, line in enumerate(lines) if line.startswith('==')]

    table_spans = {'open': [idx[1] + 1, idx[2]],
                   'save': [idx[4] + 1, idx[5]],
                   'get': [idx[7] + 1, idx[8]],
                   'convert' : [idx[10] + 1, idx[11]],
                   'merge' : [idx[13] + 1, idx[14]]}


    start, stop = table_spans[topic.lower()]

    cat = [l[:50].strip(' \t\n\r\0*') for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip(' \t\n\r\0*')) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip(' \t\n\r\0*')) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 3.8 / 12 * len(cat) + 1.2)

    asam_pos = [i for i, c in enumerate(cat) if c.startswith('asam')]
    mdfreader_pos = [i for i, c in enumerate(cat) if c.startswith('mdfreader')]

    ax.barh(asam_pos, arr[asam_pos], color='green', ecolor='green')
    ax.barh(mdfreader_pos, arr[mdfreader_pos], color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time [ms]' if aspect == 'time' else 'RAM [MB]')
    if topic == 'Get':
        ax.set_title('Get all channels (36424 calls) - {}'.format('time' if aspect == 'time' else 'ram usage'))
    else:
        ax.set_title('{} test file - {}'.format(topic, 'time' if aspect == 'time' else 'ram usage'))
    ax.xaxis.grid()

    fig.subplots_adjust(bottom=0.72/fig.get_figheight(), top=1-0.48/fig.get_figheight(), left=0.4, right=0.9)

    if aspect == 'time':
        if topic == 'Get':
            name = '{}_get_all_channels.png'.format(platform)
        else:
            name = '{}_{}.png'.format(platform, topic.lower())
    else:
        if topic == 'Get':
            name = '{}_get_all_channels_ram_usage.png'.format(platform)
        else:
            name = '{}_{}_ram_usage.png'.format(platform, topic.lower())

    plt.show()

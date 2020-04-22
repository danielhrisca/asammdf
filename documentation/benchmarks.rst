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

----------
Benchmarks
----------


Test setup
==========

The benchmarks were done using two test files (available `here <https://github.com/danielhrisca/asammdf/issues/14>`_) (for mdf version 3 and 4) of around 170MB.
The files contain 183 data groups and a total of 36424 channels.

*asamdf 5.20.0* was compared against *mdfreader 4.0*.

For each category two aspect were noted: elapsed time and peak RAM usage.

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
==================






Benchmark environment

* 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
* Windows-10-10.0.17763-SP0
* Intel64 Family 6 Model 158 Stepping 10, GenuineIntel
* numpy 1.18.3
* 16GB installed RAM

Notations used in the results

* compress = mdfreader mdf object created with compression=blosc
* nodata = mdfreader mdf object read with no_data_loading=True

Files used for benchmark:

* mdf version 3.10
    * 167 MB file size
    * 183 groups
    * 36424 channels
* mdf version 4.00
    * 183 MB file size
    * 183 groups
    * 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.20.0 mdfv3                                     269      136
mdfreader 4.0 mdfv3                                     1531      453
mdfreader 4.0 no_data_loading mdfv3                      687      206
mdfreader 4.0 compress mdfv3                            5398      154
asammdf 5.20.0 mdfv4                                     409      149
mdfreader 4.0 mdfv4                                     3910      484
mdfreader 4.0 no_data_loading mdfv4                     2836      272
mdfreader 4.0 compress mdfv4                            7898      182
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.20.0 mdfv3                                     393      136
mdfreader 4.0 mdfv3                                     3996      482
mdfreader 4.0 no_data_loading mdfv3                     4658      544
mdfreader 4.0 compress mdfv3                            4133      451
asammdf 5.20.0 mdfv4                                     341      149
mdfreader 4.0 mdfv4                                     2241      503
mdfreader 4.0 no_data_loading mdfv4                     3461      581
mdfreader 4.0 compress mdfv4                            2497      456
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.20.0 mdfv3                                    3892      137
mdfreader 4.0 mdfv3                                       73      453
mdfreader 4.0 nodata mdfv3                             11587      243
mdfreader 4.0 compress mdfv3                             517      154
asammdf 5.20.0 mdfv4                                    6940      149
mdfreader 4.0 mdfv4                                       61      485
mdfreader 4.0 nodata mdfv4                             17585      305
mdfreader 4.0 compress mdfv4                             527      184
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.20.0.dev-31 v3 to v4                          3252      194
asammdf 5.20.0.dev-31 v4 to v410                        2901      180
asammdf 5.20.0.dev-31 v4 to v420                        3273      218
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.20.0.dev-31 v3                                8339      205
mdfreader 4.0 v3                                          0*       0*
mdfreader 4.0 nodata v3                                   0*       0*
mdfreader 4.0 compress v3                                 0*       0*
asammdf 5.20.0.dev-31 v4                                7641      210
mdfreader 4.0 v4                                          0*       0*
mdfreader 4.0 nodata v4                                   0*       0*
mdfreader 4.0 compress v4                                 0*       0*
================================================== ========= ========



Graphical results
-----------------

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x64_asammdf_5.20.0_mdfreader_4.0.rst'
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
                   'mdf' : [idx[10] + 1, idx[11]],
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

    res = '../benchmarks/results/x64_asammdf_5.20.0_mdfreader_4.0.rst'
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

    res = '../benchmarks/results/x64_asammdf_5.20.0_mdfreader_4.0.rst'
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

    res = '../benchmarks/results/x64_asammdf_5.20.0_mdfreader_4.0.rst'
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

    res = '../benchmarks/results/x64_asammdf_5.20.0_mdfreader_4.0.rst'
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

    res = '../benchmarks/results/x64_asammdf_5.20.0_mdfreader_4.0.rst'
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

    res = '../benchmarks/results/x64_asammdf_5.20.0_mdfreader_4.0.rst'
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

    res = '../benchmarks/results/x64_asammdf_5.20.0_mdfreader_4.0.rst'
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

    res = '../benchmarks/results/x64_asammdf_5.20.0_mdfreader_4.0.rst'
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

    res = '../benchmarks/results/x64_asammdf_5.20.0_mdfreader_4.0.rst'
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


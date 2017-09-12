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

The benchmarks were done using two test files (for mdf version 3 and 4) of around 170MB. 
The files contain 183 data groups and a total of 36424 channels.

*asamdf 2.5.2* was compared against *mdfreader 0.2.5* (latest versions from PyPI). 
*mdfreader* seems to be the most used Python package to handle MDF files, and it also supports both version 3 and 4 of the standard.

The three benchmark cathegories are file open, file save and extracting the data for all channels inside the file(36424 calls).
For each cathegory two aspect were noted: elapsed time and peak RAM usage.

Dependencies
------------
You will need the following packages to be able to run the benchmark script

* psutil
* mdfreader

x64 Python results
------------------
The test environment used for 64 bit tests had:

* 3.6.2 (v3.6.2:5fd33b5, Jul  8 2017, 04:57:36) [MSC v.1900 64 bit (AMD64)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results:

* nodata = asammdf MDF object created with load_measured_data=False (raw channel data not loaded into RAM)

Files used for benchmark:

* 183 groups
* 36424 channels

Raw data
^^^^^^^^

================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 mdfv3                                      821      371
asammdf 2.5.2 nodata mdfv3                               653      191
mdfreader 0.2.5 mdfv3                                   2909      537
asammdf 2.5.2 mdfv4                                     1694      455
asammdf 2.5.2 nodata mdfv4                              1297      260
mdfreader 0.2.5 mdfv4                                  31074      748
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 mdfv3                                      393      373
asammdf 2.5.2 nodata mdfv3                               383      198
mdfreader 0.2.5 mdfv3                                  21464     1997
asammdf 2.5.2 mdfv4                                      586      465
asammdf 2.5.2 nodata mdfv4                               550      275
mdfreader 0.2.5 mdfv4                                  19036     2795
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 mdfv3                                      613      381
asammdf 2.5.2 nodata mdfv3                              9161      207
mdfreader 0.2.5 mdfv3                                     28      536
asammdf 2.5.2 mdfv4                                      606      464
asammdf 2.5.2 nodata mdfv4                             12403      275
mdfreader 0.2.5 mdfv4                                     40      748
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 v3 to v4                                  4773      885
asammdf 2.5.2 v3 to v4 nodata                          21903      605
asammdf 2.5.2 v4 to v3                                  4823      882
asammdf 2.5.2 v4 to v3 nodata                          26090      740
================================================== ========= ========

Graphical results
^^^^^^^^^^^^^^^^^

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    
    res = '../benchmarks/results/x64_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])


    arr = ram if aspect == 'ram' else time


    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x64_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x64_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x64_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x64_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x64_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x64_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x64_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x64_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x64_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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
The test environment used for 32 bit tests had:

* 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 17:54:52) [MSC v.1900 32 bit (Intel)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results:

* nodata = asammdf MDF object created with load_measured_data=False (raw channel data not loaded into RAM)

Files used for benchmark:

* 183 groups
* 36424 channels


Raw data
^^^^^^^^


================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 mdfv3                                     1009      289
asammdf 2.5.2 nodata mdfv3                               663      118
mdfreader 0.2.5 mdfv3                                   3705      454
asammdf 2.5.2 mdfv4                                     2031      343
asammdf 2.5.2 nodata mdfv4                              1690      161
mdfreader 0.2.5 mdfv4                                  42315      576
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 mdfv3                                      439      293
asammdf 2.5.2 nodata mdfv3                               462      126
mdfreader 0.2.5 mdfv3                                  19759     1224
asammdf 2.5.2 mdfv4                                      691      354
asammdf 2.5.2 nodata mdfv4                               712      174
mdfreader 0.2.5 mdfv4                                  17415     1686
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 mdfv3                                      807      298
asammdf 2.5.2 nodata mdfv3                             18500      132
mdfreader 0.2.5 mdfv3                                     36      454
asammdf 2.5.2 mdfv4                                      804      349
asammdf 2.5.2 nodata mdfv4                             21315      171
mdfreader 0.2.5 mdfv4                                     49      577
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 v3 to v4                                  5834      709
asammdf 2.5.2 v3 to v4 nodata                          28427      494
asammdf 2.5.2 v4 to v3                                  5474      710
asammdf 2.5.2 v4 to v3 nodata                          30423      638
================================================== ========= ========


Graphical results
^^^^^^^^^^^^^^^^^

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    res = '../benchmarks/results/x86_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x86_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x86_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x86_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x86_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x86_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x86_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x86_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x86_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

    res = '../benchmarks/results/x86_asammdf_2.5.2_mdfreader_0.2.5.txt'
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

    cat = [l[:50].strip() for l in lines[start: stop]]
    time = np.array([int(l[50:61].strip()) for l in lines[start: stop]])
    ram = np.array([int(l[61:].strip()) for l in lines[start: stop]])

    if aspect == 'ram':
        arr = ram
    else:
        arr = time

    y_pos = list(range(len(cat)))

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 4.5)

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

    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.4, right=0.9)

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

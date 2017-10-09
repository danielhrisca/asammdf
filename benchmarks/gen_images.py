# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:32:00 2017

@author: uidn3651
"""
import matplotlib.pyplot as plt
import numpy as np

def f(res, topic, aspect, for_doc=False):
    import matplotlib.pyplot as plt
    import numpy as np

    res = res
    topic = topic
    aspect = aspect
    for_doc = for_doc

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
    fig.set_size_inches(9, 3.4 / 12 * len(cat) + 1.2)

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

    if for_doc:
        plt.show()
    else:
        plt.savefig(name, dpi=300)


for file in (r'e:\02__PythonWorkspace\asammdf\benchmarks\results\x64_asammdf_2.6.0_mdfreader_0.2.6.txt',
              r'e:\02__PythonWorkspace\asammdf\benchmarks\results\x86_asammdf_2.6.0_mdfreader_0.2.6.txt'):
    for topic in ('Open', 'Save', 'Get', 'Convert', 'Merge'):
        for aspect in ('time', 'ram'):
            f(file, topic, aspect)
"""
benchmark asammdf vs mdfreader
"""
import argparse
import multiprocessing
import os
import platform
import sys
import traceback

from io import StringIO

try:
    import resource
except ImportError:
    pass

import psutil
import numpy as np

from asammdf import MDF, Signal
from asammdf import __version__ as asammdf_version
import asammdf.blocks.v4_constants as v4c
import asammdf.blocks.v4_blocks as v4b
import asammdf.blocks.v2_v3_constants as v3c
import asammdf.blocks.v2_v3_blocks as v3b
from mdfreader import Mdf as MDFreader
from mdfreader import __version__ as mdfreader_version


PYVERSION = sys.version_info[0]

if PYVERSION > 2:
    from time import perf_counter
else:
    from time import clock as perf_counter


class MyList(list):
    """ list that prints the items that are appended or extended """

    def append(self, item):
        """ append item and print it to stdout """
        print(item)
        super(MyList, self).append(item)

    def extend(self, items):
        """ extend items and print them to stdout
        using the new line separator
        """
        print('\n'.join(items))
        super(MyList, self).extend(items)


class Timer():
    """ measures the RAM usage and elased time. The information is saved in
    the output attribute and any Exception text is saved in the error attribute

    Parameters
    ----------
    topic : str
        timer title; only used if Exceptions are raised during execution
    message : str
        execution item description
    fmt : str
        output fmt; can be "rst" (rstructured text) or "md" (markdown)
    """

    def __init__(self, topic, message, fmt='rst'):
        self.topic = topic
        self.message = message
        self.output = ''
        self.error = ''
        self.fmt = fmt
        self.start = None

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type_, value, tracebackobj):
        elapsed_time = int((perf_counter() - self.start) * 1000)
        process = psutil.Process(os.getpid())

        if platform.system() == 'Windows':
            ram_usage = int(process.memory_info().peak_wset / 1024 / 1024)
        else:
            ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ram_usage = int(ram_usage / 1024)

        if tracebackobj:
            info = StringIO()
            traceback.print_tb(tracebackobj, None, info)
            info.seek(0)
            info = info.read()
            self.error = '{} : {}\n{}\t \n{}{}'.format(
                self.topic,
                self.message,
                type_,
                value,
                info,
            )
            if self.fmt == 'rst':
                self.output = '{:<50} {:>9} {:>8}'.format(self.message,
                                                          '0*',
                                                          '0*')
            elif self.fmt == 'md':
                self.output = '|{:<50}|{:>9}|{:>8}|'.format(self.message,
                                                            '0*',
                                                            '0*')
        else:
            if self.fmt == 'rst':
                self.output = '{:<50} {:>9} {:>8}'.format(self.message,
                                                          elapsed_time,
                                                          ram_usage)
            elif self.fmt == 'md':
                self.output = '|{:<50}|{:>9}|{:>8}|'.format(self.message,
                                                            elapsed_time,
                                                            ram_usage)

        return True


def generate_test_files(version='4.10'):
    cycles = 3000
    channels_count = 2000
    mdf = MDF(version=version)

    if version <= '3.30':
        filename = r'test.mdf'
    else:
        filename = r'test.mf4'

    if os.path.exists(filename):
        return filename

    t = np.arange(cycles, dtype=np.float64)

    cls = v4b.ChannelConversion if version >= '4.00' else v3b.ChannelConversion

    # no conversion
    sigs = []
    for i in range(channels_count):
        sig = Signal(
            np.ones(cycles, dtype=np.uint64) * i,
            t,
            name='Channel_{}'.format(i),
            unit='unit_{}'.format(i),
            conversion=None,
            comment='Unsigned int 16bit channel {}'.format(i),
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # linear
    sigs = []
    for i in range(channels_count):
        conversion = {
            'conversion_type': v4c.CONVERSION_TYPE_LIN if version >= '4.00' else v3c.CONVERSION_TYPE_LINEAR,
            'a': float(i),
            'b': -0.5,
        }
        sig = Signal(
            np.ones(cycles, dtype=np.int64),
            t,
            name='Channel_{}'.format(i),
            unit='unit_{}'.format(i),
            conversion=cls(**conversion),
            comment='Signed 16bit channel {} with linear conversion'.format(i),
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # algebraic
    sigs = []
    for i in range(channels_count):
        conversion = {
            'conversion_type': v4c.CONVERSION_TYPE_ALG if version >= '4.00' else v3c.CONVERSION_TYPE_FORMULA,
            'formula': '{} * sin(X)'.format(i),
        }
        sig = Signal(
            np.arange(cycles, dtype=np.int32) / 100.0,
            t,
            name='Channel_{}'.format(i),
            unit='unit_{}'.format(i),
            conversion=cls(**conversion),
            comment='Sinus channel {} with algebraic conversion'.format(i),
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # rational
    sigs = []
    for i in range(channels_count):
        conversion = {
            'conversion_type': v4c.CONVERSION_TYPE_RAT if version >= '4.00' else v3c.CONVERSION_TYPE_RAT,
            'P1': 0,
            'P2': i,
            'P3': -0.5,
            'P4': 0,
            'P5': 0,
            'P6': 1,
        }
        sig = Signal(
            np.ones(cycles, dtype=np.int64),
            t,
            name='Channel_{}'.format(i),
            unit='unit_{}'.format(i),
            conversion=cls(**conversion),
            comment='Channel {} with rational conversion'.format(i),
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # string
    sigs = []
    for i in range(channels_count):
        sig = [
            'Channel {} sample {}'.format(i, j).encode('ascii')
            for j in range(cycles)
        ]
        sig = Signal(
            np.array(sig),
            t,
            name='Channel_{}'.format(i),
            unit='unit_{}'.format(i),
            comment='String channel {}'.format(i),
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # byte array
    sigs = []
    ones = np.ones(cycles, dtype=np.dtype('(8,)u1'))
    for i in range(channels_count):
        sig = Signal(
            ones*(i%255),
            t,
            name='Channel_{}'.format(i),
            unit='unit_{}'.format(i),
            comment='Byte array channel {}'.format(i),
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    # value to text
    sigs = []
    ones = np.ones(cycles, dtype=np.uint64)
    conversion = {
        'raw': np.arange(255, dtype=np.float64),
        'phys': np.array([
            'Value {}'.format(i).encode('ascii')
            for i in range(255)
        ]),
        'conversion_type': v4c.CONVERSION_TYPE_TABX if version >= '4.00' else v3c.CONVERSION_TYPE_TABX,
        'links_nr': 260,
        'ref_param_nr': 255,
    }

    for i in range(255):
        conversion['val_{}'.format(i)] = conversion['param_val_{}'.format(i)] = conversion['raw'][i]
        conversion['text_{}'.format(i)] = conversion['phys'][i]
    conversion['text_{}'.format(255)] = 'Default'

    for i in range(channels_count):
        sig = Signal(
            ones * i,
            t,
            name='Channel_{}'.format(i),
            unit='unit_{}'.format(i),
            comment='Value to text channel {}'.format(i),
            conversion=cls(**conversion),
            raw=True,
        )
        sigs.append(sig)
    mdf.append(sigs, common_timebase=True)

    mdf.save(filename, overwrite=True)


def open_mdf3(output, fmt):

    with Timer('Open file',
               f'asammdf {asammdf_version} mdfv3',
               fmt) as timer:
        MDF(r'test.mdf')
    output.send([timer.output, timer.error])


def open_mdf4(output, fmt):

    with Timer('Open file',
               f'asammdf {asammdf_version} mdfv4',
               fmt) as timer:
        MDF(r'test.mf4')
    output.send([timer.output, timer.error])


def save_mdf3(output, fmt):

    x = MDF(r'test.mdf')
    with Timer('Save file',
               f'asammdf {asammdf_version} mdfv3',
               fmt) as timer:
        x.save(r'x.mdf', overwrite=True)
    output.send([timer.output, timer.error])


def save_mdf4(output, fmt):

    x = MDF(r'test.mf4')
    with Timer('Save file',
               f'asammdf {asammdf_version} mdfv4',
               fmt) as timer:
        x.save(r'x.mf4', overwrite=True)
    output.send([timer.output, timer.error])


def get_all_mdf3(output, fmt):

    x = MDF(r'test.mdf')
    with Timer('Get all channels',
               f'asammdf {asammdf_version} mdfv3',
               fmt) as timer:
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                x.get(group=i, index=j, samples_only=True)
    output.send([timer.output, timer.error])


def get_all_mdf4(output, fmt):

    x = MDF(r'test.mf4')
    with Timer('Get all channels',
               f'asammdf {asammdf_version} mdfv4',
               fmt) as timer:
        t = perf_counter()
        counter = 0
        to_break = False
        for i, gp in enumerate(x.groups):
            if to_break:
                break
            for j in range(len(gp['channels'])):
                t2 = perf_counter()
                if t2 - t > 60:
                    timer.message += ' {}/s'.format(counter/(t2-t))
                    to_break = True
                    break
                x.get(group=i, index=j, samples_only=True)
                counter += 1
    output.send([timer.output, timer.error])
    

def convert_v3_v4(output, fmt):

    with MDF(r'test.mdf') as x:
        with Timer('Convert file',
                   f'asammdf {asammdf_version} v3 to v4',
                   fmt) as timer:
            x.convert('4.10')
    output.send([timer.output, timer.error])

def convert_v4_v3(output, fmt):

    with MDF(r'test.mf4') as x:
        with Timer('Convert file',
                   f'asammdf {asammdf_version} v4 to v3',
                   fmt) as timer:
            y = x.convert('4.10')
            y.close()
    output.send([timer.output, timer.error])


def merge_v3(output, fmt):

    files = [r'test.mdf', ] * 3
    with Timer('Merge 3 files',
               f'asammdf {asammdf_version} v3',
               fmt) as timer:
        MDF.concatenate(files, outversion='3.30')
    output.send([timer.output, timer.error])


def merge_v4(output, fmt):
    files = [r'test.mf4', ] * 3

    with Timer('Merge 3 files',
               f'asammdf {asammdf_version} v4',
               fmt) as timer:
        MDF.concatenate(files, outversion='4.10')
    output.send([timer.output, timer.error])


#
# mdfreader
#


def open_reader3(output, fmt):

    with Timer('Open file',
               'mdfreader {} mdfv3'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mdf')
    output.send([timer.output, timer.error])


def open_reader3_nodata(output, fmt):

    with Timer('Open file',
               'mdfreader {} no_data_loading mdfv3'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mdf', no_data_loading=True)
    output.send([timer.output, timer.error])


def open_reader3_compression(output, fmt):

    with Timer('Open file',
               'mdfreader {} compress mdfv3'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mdf', compression='blosc')
    output.send([timer.output, timer.error])


def open_reader4(output, fmt):

    with Timer('Open file',
               'mdfreader {} mdfv4'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mf4')
    output.send([timer.output, timer.error])


def open_reader4_nodata(output, fmt):

    with Timer('Open file',
               'mdfreader {} no_data_loading mdfv4'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mf4', no_data_loading=True)
    output.send([timer.output, timer.error])


def open_reader4_compression(output, fmt):

    with Timer('Open file',
               'mdfreader {} compress mdfv4'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mf4', compression='blosc')
    output.send([timer.output, timer.error])


def save_reader3(output, fmt):

    x = MDFreader(r'test.mdf')
    with Timer('Save file',
               'mdfreader {} mdfv3'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mdf')
    output.send([timer.output, timer.error])


def save_reader3_nodata(output, fmt):

    x = MDFreader(r'test.mdf', no_data_loading=True)
    with Timer('Save file',
               'mdfreader {} no_data_loading mdfv3'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mdf')
    output.send([timer.output, timer.error])


def save_reader3_compression(output, fmt):
    with Timer('Save file',
               'mdfreader {} compress mdfv3'.format(mdfreader_version),
               fmt) as outer_timer:
        x = MDFreader(r'test.mdf', compression='blosc')
        with Timer('Save file',
                'mdfreader {} compress mdfv3'.format(mdfreader_version),
                fmt) as timer:
            x.write(r'x.mdf')
        output.send([timer.output, timer.error])
    if outer_timer.error:
        output.send([timer.output, timer.error])


def save_reader4(output, fmt):

    x = MDFreader(r'test.mf4')
    with Timer('Save file',
               'mdfreader {} mdfv4'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mf4')
    output.send([timer.output, timer.error])


def save_reader4_nodata(output, fmt):

    x = MDFreader(r'test.mf4', no_data_loading=True)
    with Timer('Save file',
               'mdfreader {} no_data_loading mdfv4'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mf4')
    output.send([timer.output, timer.error])


def save_reader4_compression(output, fmt):

    x = MDFreader(r'test.mf4', compression='blosc')
    with Timer('Save file',
               'mdfreader {} compress mdfv4'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mf4')
    output.send([timer.output, timer.error])


def get_all_reader3(output, fmt):

    x = MDFreader(r'test.mdf')
    with Timer('Get all channels',
               'mdfreader {} mdfv3'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.get_channel_data(s)
    output.send([timer.output, timer.error])


def get_all_reader3_nodata(output, fmt):

    x = MDFreader(r'test.mdf', no_data_loading=True)
    with Timer('Get all channels',
               'mdfreader {} nodata mdfv3'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.get_channel_data(s)
    output.send([timer.output, timer.error])


def get_all_reader3_compression(output, fmt):

    x = MDFreader(r'test.mdf', compression='blosc')
    with Timer('Get all channels',
               'mdfreader {} compress mdfv3'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.get_channel_data(s)

        with open('D:\\TMP\\f.txt', 'w') as f:
            f.write('OK')
    output.send([timer.output, timer.error])


def get_all_reader4(output, fmt):

    x = MDFreader(r'test.mf4')
    with Timer('Get all channels',
               'mdfreader {} mdfv4'.format(mdfreader_version),
               fmt) as timer:
        t = perf_counter()
        counter = 0
        to_break = False
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += ' {}/s'.format(counter/(t2-t))
                to_break = True
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])


def get_all_reader4_nodata(output, fmt):

    x = MDFreader(r'test.mf4', no_data_loading=True)
    with Timer('Get all channels',
               'mdfreader {} nodata mdfv4'.format(mdfreader_version),
               fmt) as timer:
        t = perf_counter()
        counter = 0
        to_break = False
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += ' {}/s'.format(counter/(t2-t))
                to_break = True
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])


def get_all_reader4_compression(output, fmt):

    x = MDFreader(r'test.mf4', compression='blosc')
    with Timer('Get all channels',
               'mdfreader {} compress mdfv4'.format(mdfreader_version),
               fmt) as timer:
        t = perf_counter()
        counter = 0
        to_break = False
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += ' {}/s'.format(counter/(t2-t))
                to_break = True
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])


def merge_reader_v3(output, fmt):

    files = [r'test.mdf', ] * 3
    with Timer('Merge 3 files',
               'mdfreader {} v3'.format(mdfreader_version),
               fmt) as timer:
        x1 = MDFreader(files[0])
        x1.resample(0.01)
        x2 = MDFreader(files[1])
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2])
        x2.resample(0.01)
        x1.merge_mdf(x2)
    output.send([timer.output, timer.error])


def merge_reader_v3_compress(output, fmt):

    files = [r'test.mdf', ] * 3
    with Timer('Merge 3 files',
               'mdfreader {} compress v3'.format(mdfreader_version),
               fmt) as timer:
        x1 = MDFreader(files[0], compression='blosc')
        x1.resample(0.01)
        x2 = MDFreader(files[1], compression='blosc')
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2], compression='blosc')
        x2.resample(0.01)
        x1.merge_mdf(x2)
    output.send([timer.output, timer.error])


def merge_reader_v3_nodata(output, fmt):

    files = [r'test.mdf', ] * 3
    with Timer('Merge 3 files',
               'mdfreader {} nodata v3'.format(mdfreader_version),
               fmt) as timer:
        x1 = MDFreader(files[0], no_data_loading=True)
        x1.resample(0.01)
        x2 = MDFreader(files[1], no_data_loading=True)
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2], no_data_loading=True)
        x2.resample(0.01)
        x1.merge_mdf(x2)
    output.send([timer.output, timer.error])


def merge_reader_v4(output, fmt):
    files = [r'test.mf4', ] * 3

    with Timer('Merge 3 files',
               'mdfreader {} v4'.format(mdfreader_version),
               fmt) as timer:
        x1 = MDFreader(files[0])
        x1.resample(0.01)
        x2 = MDFreader(files[1])
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2])
        x2.resample(0.01)
        x1.merge_mdf(x2)

    output.send([timer.output, timer.error])


def merge_reader_v4_compress(output, fmt):

    files = [r'test.mf4', ] * 3
    with Timer('Merge 3 files',
               'mdfreader {} compress v4'.format(mdfreader_version),
               fmt) as timer:
        x1 = MDFreader(files[0], compression='blosc')
        x1.resample(0.01)
        x2 = MDFreader(files[1], compression='blosc')
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2], compression='blosc')
        x2.resample(0.01)
        x1.merge_mdf(x2)

    output.send([timer.output, timer.error])

def merge_reader_v4_nodata(output, fmt):

    files = [r'test.mf4', ] * 3
    with Timer('Merge 3 files',
               'mdfreader {} nodata v4'.format(mdfreader_version),
               fmt) as timer:
        x1 = MDFreader(files[0], no_data_loading=True)
        x1.resample(0.01)
        x2 = MDFreader(files[1], no_data_loading=True)
        x2.resample(0.01)
        x1.merge_mdf(x2)
        x2 = MDFreader(files[2], no_data_loading=True)
        x2.resample(0.01)
        x1.merge_mdf(x2)

    output.send([timer.output, timer.error])

#
# utility functions
#


def filter_asam(output, fmt):
    with Timer('Filter file',
               f'asammdf {asammdf_version} mdfv4',
               fmt) as timer:
        x = MDF(r'test.mf4').filter([f'Channel_{i}_{j}5' for i in range(10) for j in range(1, 20)])
        t = perf_counter()
        counter = 0
        to_break = False
        for i, gp in enumerate(x.groups):
            if to_break:
                break
            for j in range(len(gp['channels'])):
                t2 = perf_counter()
                if t2 - t > 60:
                    timer.message += ' {}/s'.format(counter/(t2-t))
                    to_break = True
                    break
                x.get(group=i, index=j, samples_only=True)
                counter += 1
    output.send([timer.output, timer.error])


def filter_reader4(output, fmt):

    with Timer('Filter file',
               'mdfreader {} mdfv4'.format(mdfreader_version),
               fmt) as timer:
        x = MDFreader(r'test.mf4', channel_list=[f'Channel_{i}_{j}5' for i in range(10) for j in range(1, 20)])
        t = perf_counter()
        counter = 0
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += ' {}/s'.format(counter/(t2-t))
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])

def filter_reader4_compression(output, fmt):

    with Timer('Filter file',
               'mdfreader {} compression mdfv4'.format(mdfreader_version),
               fmt) as timer:
        x = MDFreader(r'test.mf4', compression='blosc', channel_list=[f'Channel_{i}_{j}5' for i in range(10) for j in range(1, 20)])
        t = perf_counter()
        counter = 0
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += ' {}/s'.format(counter/(t2-t))
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])

def filter_reader4_nodata(output, fmt):

    with Timer('Filter file',
               'mdfreader {} nodata mdfv4'.format(mdfreader_version),
               fmt) as timer:
        x = MDFreader(r'test.mf4', no_data_loading=True, channel_list=[f'Channel_{i}_{j}5' for i in range(10) for j in range(1, 20)])
        t = perf_counter()
        counter = 0
        for s in x:
            t2 = perf_counter()
            if t2 - t > 60:
                timer.message += ' {}/s'.format(counter/(t2-t))
                break
            x.get_channel_data(s)
            counter += 1
    output.send([timer.output, timer.error])


def cut_asam(output, fmt):
    x = MDF(r'test.mf4')
    t = x.get_master(0)
    start, stop = 0.2 * (t[-1] - t[0]) + t[0], 0.8 * (t[-1] - t[0]) + t[0]
    with Timer('Cut file',
               f'asammdf {asammdf_version} mdfv4',
               fmt) as timer:
        x = x.cut(start=start, stop=stop)

    output.send([timer.output, timer.error])


def cut_reader4(output, fmt):

    x = MDFreader(r'test.mf4')
    t = x.get_channel_data(list(x.masterChannelList)[0])
    begin, end = 0.2 * (t[-1] - t[0]) + t[0], 0.8 * (t[-1] - t[0]) + t[0]
    with Timer('Cut file',
               'mdfreader {} mdfv4'.format(mdfreader_version),
               fmt) as timer:

        x.cut(begin=begin, end=end)
    output.send([timer.output, timer.error])

def cut_reader4_compression(output, fmt):

    x = MDFreader(r'test.mf4', compression='blosc')
    t = x.get_channel_data(list(x.masterChannelList)[0])
    begin, end = 0.2 * (t[-1] - t[0]) + t[0], 0.8 * (t[-1] - t[0]) + t[0]
    with Timer('Cut file',
               'mdfreader {} compression mdfv4'.format(mdfreader_version),
               fmt) as timer:

        x.cut(begin=begin, end=end)
    output.send([timer.output, timer.error])

def cut_reader4_nodata(output, fmt):

    x = MDFreader(r'test.mf4', no_data_loading=True)
    t = x.get_channel_data(list(x.masterChannelList)[0])
    begin, end = 0.2 * (t[-1] - t[0]) + t[0], 0.8 * (t[-1] - t[0]) + t[0]
    with Timer('Cut file',
               'mdfreader {} nodata mdfv4'.format(mdfreader_version),
               fmt) as timer:

        x.cut(begin=begin, end=end)
    output.send([timer.output, timer.error])


def table_header(topic, fmt='rst'):
    output = []
    if fmt == 'rst':
        result = '{:<50} {:>9} {:>8}'.format(topic, 'Time [ms]', 'RAM [MB]')
        output.append('')
        output.append('{} {} {}'.format('='*50, '='*9, '='*8))
        output.append(result)
        output.append('{} {} {}'.format('='*50, '='*9, '='*8))
    elif fmt == 'md':
        result = '|{:<50}|{:>9}|{:>8}|'.format(topic, 'Time [ms]', 'RAM [MB]')
        output.append('')
        output.append(result)
        output.append('|{}|{}|{}|'.format('-'*50, '-'*9, '-'*8))
    return output


def table_end(fmt='rst'):
    if fmt == 'rst':
        return ['{} {} {}'.format('='*50, '='*9, '='*8), '']
    elif fmt == 'md':
        return ['', ]


def main(text_output, fmt):
    if os.path.dirname(__file__):
        os.chdir(os.path.dirname(__file__))
    for version in ('3.30', '4.10'):
        generate_test_files(version)

    mdf = MDF('test.mdf', 'minimum')
    v3_size = os.path.getsize('test.mdf') // 1024 // 1024
    v3_groups = len(mdf.groups)
    v3_channels = sum (
        len(gp['channels'])
        for gp in mdf.groups
    )
    v3_version = mdf.version

    mdf = MDF('test.mf4', 'minimum')
    v4_size = os.path.getsize('test.mf4') // 1024 // 1024
    v4_groups = len(mdf.groups)
    v4_channels = sum(
        len(gp['channels'])
        for gp in mdf.groups
    )
    v4_version = mdf.version

    listen, send = multiprocessing.Pipe()
    output = MyList()
    errors = []

    installed_ram = round(psutil.virtual_memory().total / 1024 / 1024 / 1024)

    output.append('\n\nBenchmark environment\n')
    output.append('* {}'.format(sys.version))
    output.append('* {}'.format(platform.platform()))
    output.append('* {}'.format(platform.processor()))
    output.append('* numpy {}'.format(np.__version__))
    output.append('* {}GB installed RAM\n'.format(installed_ram))
    output.append('Notations used in the results\n')
    output.append(('* compress = mdfreader mdf object created with '
                   'compression=blosc'))
    output.append(('* nodata = mdfreader mdf object read with '
                   'no_data_loading=True'))
    output.append('\nFiles used for benchmark:\n')
    output.append('* mdf version {}'.format(v3_version))
    output.append('    * {} MB file size'.format(v3_size))
    output.append('    * {} groups'.format(v3_groups))
    output.append('    * {} channels'.format(v3_channels))
    output.append('* mdf version {}'.format(v4_version))
    output.append('    * {} MB file size'.format(v4_size))
    output.append('    * {} groups'.format(v4_groups))
    output.append('    * {} channels\n\n'.format(v4_channels))

    OPEN, SAVE, GET, CONVERT, MERGE, FILTER, CUT = 1, 1, 1, 1, 1, 0, 0

    tests = (
          open_mdf3,
          open_reader3,
          open_reader3_compression,
          open_reader3_nodata,
         open_mdf4,
          open_reader4,
          open_reader4_compression,
          open_reader4_nodata,
    )

    if tests and OPEN:
        output.extend(table_header('Open file', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
         save_mdf3,
        save_reader3,
        save_reader3_nodata,
        save_reader3_compression,
         save_mdf4,
        save_reader4,
        save_reader4_nodata,
        save_reader4_compression,
    )

    if tests and SAVE:
        output.extend(table_header('Save file', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        get_all_mdf3,
        get_all_reader3,
        get_all_reader3_nodata,
        get_all_reader3_compression,

        get_all_mdf4,

        get_all_reader4,
        get_all_reader4_compression,
        get_all_reader4_nodata,


    )

    if tests and GET:
        output.extend(table_header('Get all channels (36424 calls)', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
         convert_v3_v4,
         convert_v4_v3,
    )

    if tests and CONVERT:
        output.extend(table_header('Convert file', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
          merge_v3,
          merge_reader_v3,
          merge_reader_v3_compress,
          merge_reader_v3_nodata,
         merge_v4,
          merge_reader_v4,
          merge_reader_v4_nodata,
          merge_reader_v4_compress,
    )

    if tests and MERGE:
        output.extend(table_header('Merge 3 files', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
          filter_asam,
          filter_reader4,
          filter_reader4_compression,
          filter_reader4_nodata,
    )

    if tests and FILTER:
        output.extend(table_header('Filter 200 channels', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))


    tests = (
          cut_asam,
          cut_reader4,
          cut_reader4_compression,
          cut_reader4_nodata,
    )

    if tests and CUT:
        output.extend(table_header('Cut file from 20% to 80%', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))




    errors = [err for err in errors if err]
    if errors:
        print('\n\nERRORS\n', '\n'.join(errors))

    if text_output:
        arch = 'x86' if platform.architecture()[0] == '32bit' else 'x64'
        file = '{}_asammdf_{}_mdfreader_{}.{}'.format(arch,
                                                      asammdf_version,
                                                      mdfreader_version,
                                                      fmt)
        with open(file, 'w') as out:
            out.write('\n'.join(output))

    for file in ('x.mdf', 'x.mf4'):
        if PYVERSION >= 3:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
        else:
            try:
                os.remove(file)
            except IOError:
                pass


def _cmd_line_parser():
    '''
    return a command line parser. It is used when generating the documentation
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        help=('path to test files, '
                              'if not provided the script folder is used'))
    parser.add_argument('--text_output',
                        action='store_true',
                        help='option to save the results to text file')
    parser.add_argument('--format',
                        default='rst',
                        nargs='?',
                        choices=['rst', 'md'],
                        help='text formatting')

    return parser

if __name__ == '__main__':
    cmd_parser = _cmd_line_parser()
    args = cmd_parser.parse_args(sys.argv[1:])

    main(args.text_output, args.format)

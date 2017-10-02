"""
bnachmark asammdf vs mdfreader
"""
from __future__ import print_function, division
import argparse
import multiprocessing
import os
import platform
import sys
import traceback

from io import StringIO

PYVERSION = sys.version_info[0]

if PYVERSION > 2:
    from time import perf_counter
else:
    from time import clock as perf_counter

try:
    import resource
except ImportError:
    pass

import psutil

from asammdf import MDF
from asammdf import __version__ as asammdf_version
from mdfreader import mdf as MDFreader
from mdfreader import __version__ as mdfreader_version


class Timer():
    def __init__(self, topic, message, format='rst'):
        self.topic = topic
        self.message = message
        self.output = ''

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, tracebackobj):
        elapsed_time = (perf_counter() - self.start) * 1000
        process = psutil.Process(os.getpid())

        if format == 'rst':
            if platform.system() == 'Windows':
                self.output = '{:<50} {:>9} {:>8}'.format(self.message, int(elapsed_time), int(process.memory_info().peak_wset / 1024 / 1024))
            else:
                self.output = '{:<50} {:>9} {:>8}'.format(self.message, int(elapsed_time), int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
        elif format == 'md':
            if platform.system() == 'Windows':
                self.output = '|{}|{}|{}|'.format(self.message, int(elapsed_time), int(process.memory_info().peak_wset / 1024 / 1024))
            else:
                self.output = '|{}|{}|{}|'.format(self.message, int(elapsed_time), int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

        if tracebackobj:
            info = StringIO()
            traceback.print_tb(tracebackobj, None, info)
            info.seek(0)
            info = info.read()
            print('{} : {} --> Excption during run:\n{}'.format(self.topic, self.message, info))

        return True


def open_mdf3(path, output):
    os.chdir(path)
    with Timer('Open file', 'asammdf {} mdfv3'.format(asammdf_version)) as timer:
        x = MDF(r'test.mdf')
    output.append(timer.output)

def save_mdf3(path, output):
    os.chdir(path)
    x = MDF(r'test.mdf')
    with Timer('Save file', 'asammdf {} mdfv3'.format(asammdf_version)) as timer:
        x.save(r'x.mdf', overwrite=True)
    output.append(timer.output)

def get_all_mdf3(path, output):
    os.chdir(path)
    x = MDF(r'test.mdf')
    with Timer('Get all channels', 'asammdf {} mdfv3'.format(asammdf_version)) as timer:
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)
    output.append(timer.output)

def all_mdf3(path, output):
    os.chdir(path)
    with Timer('asammdf {} mdfv3'.format(asammdf_version)) as timer:
        x = MDF(r'test.mdf')
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)
        x.save(r'x.mdf', overwrite=True)
    output.append(timer.output)

def all_mdf3_nodata(path, output):
    os.chdir(path)
    with Timer('asammdf {} mdfv3 nodata'.format(asammdf_version)) as timer:
        x = MDF(r'test.mdf', load_measured_data=False)
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)
        x.save(r'x.mdf', overwrite=True)
    output.append(timer.output)

def all_mdf4(path, output):
    os.chdir(path)
    with Timer('asammdf {} mdfv4'.format(asammdf_version)) as timer:
        x = MDF(r'test.mf4')
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)
        x.save(r'x.mf4', overwrite=True)
    output.append(timer.output)

def all_mdf4_nodata(path, output):
    os.chdir(path)
    with Timer('asammdf {} mdfv4 nodata'.format(asammdf_version)) as timer:
        x = MDF(r'test.mf4', load_measured_data=False)
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)
        x.save(r'x.mf4', overwrite=True)
    output.append(timer.output)

def open_mdf3_nodata(path, output):
    os.chdir(path)
    with Timer('Open file', 'asammdf {} nodata mdfv3'.format(asammdf_version)) as timer:
        x = MDF(r'test.mdf', load_measured_data=False)
    output.append(timer.output)

def save_mdf3_nodata(path, output):
    os.chdir(path)
    x = MDF(r'test.mdf', load_measured_data=False)
    with Timer('Save file', 'asammdf {} nodata mdfv3'.format(asammdf_version)) as timer:
        x.save(r'x.mdf', overwrite=True)
    output.append(timer.output)

def get_all_mdf3_nodata(path, output):
    os.chdir(path)
    x = MDF(r'test.mdf', load_measured_data=False)
    with Timer('Get all channels', 'asammdf {} nodata mdfv3'.format(asammdf_version)) as timer:

        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)
    output.append(timer.output)

def open_mdf4(path, output):
    os.chdir(path)
    with Timer('Open file','asammdf {} mdfv4'.format(asammdf_version)) as timer:
        x = MDF(r'test.mf4')
    output.append(timer.output)

def save_mdf4(path, output):
    os.chdir(path)
    x = MDF(r'test.mf4')
    with Timer('Save file', 'asammdf {} mdfv4'.format(asammdf_version)) as timer:
        x.save(r'x.mf4', overwrite=True)
    output.append(timer.output)

def get_all_mdf4(path, output):
    os.chdir(path)
    x = MDF(r'test.mf4')
    with Timer('asammdf {} mdfv4'.format(asammdf_version)) as timer:
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)
    output.append(timer.output)

def open_mdf4_nodata(path, output):
    os.chdir(path)
    with Timer('Open file','asammdf {} nodata mdfv4'.format(asammdf_version)) as timer:
        x = MDF(r'test.mf4', load_measured_data=False)
    output.append(timer.output)

def save_mdf4_nodata(path, output):
    os.chdir(path)
    x = MDF(r'test.mf4', load_measured_data=False)
    with Timer('Save file', 'asammdf {} nodata mdfv4'.format(asammdf_version)) as timer:
        x.save(r'x.mf4', overwrite=True)
    output.append(timer.output)

def get_all_mdf4_nodata(path, output):
    os.chdir(path)
    x = MDF(r'test.mf4', load_measured_data=False)
    with Timer('Get all channels', 'asammdf {} nodata mdfv4'.format(asammdf_version)) as timer:
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)
    output.append(timer.output)

def convert_v3_v4(path, output):
    os.chdir(path)
    with MDF(r'test.mdf') as x:
        with Timer('Convert file', 'asammdf {} v3 to v4'.format(asammdf_version)) as timer:
            y = x.convert('4.10')
    output.append(timer.output)

def convert_v3_v4_nodata(path, output):
    os.chdir(path)
    with MDF(r'test.mdf', load_measured_data=False) as x:
        with Timer('Convert file', 'asammdf {} v3 to v4 nodata'.format(asammdf_version)) as timer:
            y = x.convert(to='4.10', load_measured_data=False)
            y.close()
    output.append(timer.output)

def convert_v4_v3(path, output):
    os.chdir(path)
    with MDF(r'test.mf4') as x:
        with Timer('Convert file', 'asammdf {} v4 to v3'.format(asammdf_version)) as timer:
            y = x.convert('3.30')
            y.close()
    output.append(timer.output)

def convert_v4_v3_nodata(path, output):
    os.chdir(path)
    with MDF(r'test.mf4', load_measured_data=False) as x:
        with Timer('Convert file', 'asammdf {} v4 to v3 nodata'.format(asammdf_version)) as timer:
            y = x.convert('3.30', load_measured_data=False)
            y.close()
    output.append(timer.output)

def merge_v3(path, output):
    os.chdir(path)
    files = [r'test.mdf', ] * 2
    with Timer('Merge files', 'asammdf {} v3'.format(asammdf_version)) as timer:
        y = MDF.merge(files)
    output.append(timer.output)

def merge_v3_nodata(path, output):
    os.chdir(path)
    files = [r'test.mdf', ] * 2
    with Timer('Merge files', 'asammdf {} v3 nodata'.format(asammdf_version)) as timer:
        y = MDF.merge(files, load_measured_data=False)
        y.close()
    output.append(timer.output)

def merge_v4(path, output):
    files = [r'test.mf4', ] * 2
    os.chdir(path)
    with Timer('Merge files', 'asammdf {} v4'.format(asammdf_version)) as timer:
        y = MDF.merge(files)
    output.append(timer.output)

def merge_v4_nodata(path, output):
    files = [r'test.mf4', ] * 2
    os.chdir(path)
    with Timer('Merge files', 'asammdf {} v4 nodata'.format(asammdf_version)) as timer:
        y = MDF.merge(files, load_measured_data=False)
        y.close()
    output.append(timer.output)

################
# mdf reader
#################


def open_reader4(path, output):
    os.chdir(path)
    with Timer('Open file','mdfreader {} mdfv4'.format(mdfreader_version)) as timer:
        x = MDFreader(r'test.mf4')
    output.append(timer.output)

def save_reader4(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mf4')
    with Timer('Save file', 'mdfreader {} mdfv4'.format(mdfreader_version)) as timer:
        x.write(r'x.mf4')
    output.append(timer.output)

def save_reader4_nodata(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mf4', noDataLoading=True)
    with Timer('Save file', 'mdfreader {} noDataLoading mdfv4'.format(mdfreader_version)) as timer:
        x.write(r'x.mf4')
    output.append(timer.output)

def save_reader4_compression(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mf4', compression='blosc')
    with Timer('Save file', 'mdfreader {} compression mdfv4'.format(mdfreader_version)) as timer:
        x.write(r'x.mf4')
    output.append(timer.output)

def save_reader4_compression_bcolz(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mf4', compression=6)
    with Timer('Save file', 'mdfreader {} compression bcolz 6 mdfv4'.format(mdfreader_version)) as timer:
        x.write(r'x.mf4')
    output.append(timer.output)


def get_all_reader4(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mf4')
    with Timer('Get all channels', 'mdfreader {} mdfv4'.format(mdfreader_version)) as timer:
        for s in x:
            y = x.getChannelData(s)
    output.append(timer.output)

def all_reader4(path, output):
    os.chdir(path)
    with Timer('mdfreader {} mdfv4'.format(mdfreader_version)) as timer:
        x = MDFreader(r'test.mf4')
        for s in x:
            y = x.getChannelData(s)
        x.write('x.mf4')
    output.append(timer.output)

def all_reader4_nodata(path, output):
    os.chdir(path)
    with Timer('mdfreader {} noDataLoading mdfv4'.format(mdfreader_version)) as timer:
        x = MDFreader(r'test.mf4', noDataLoading=True)
        for s in x:
            y = x.getChannelData(s)
        x.write('x.mf4')
    output.append(timer.output)

def get_all_reader4_nodata(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mf4', noDataLoading=True)
    with Timer('Get all channels', 'mdfreader {} nodata mdfv4'.format(mdfreader_version)) as timer:
        for s in x:
            y = x.getChannelData(s)
    output.append(timer.output)

def get_all_reader4_compression(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mf4', compression='blosc')
    with Timer('Get all channels', 'mdfreader {} compression mdfv4'.format(mdfreader_version)) as timer:
        for s in x:
            y = x.getChannelData(s)
    output.append(timer.output)

def get_all_reader4_compression_bcolz(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mf4', compression=6)
    with Timer('Get all channels', 'mdfreader {} compression bcolz 6 mdfv4'.format(mdfreader_version)) as timer:
        for s in x:
            y = x.getChannelData(s)
    output.append(timer.output)

def open_reader4_nodata(path, output):
    os.chdir(path)
    with Timer('Open file','mdfreader {} noDataLoading mdfv4'.format(mdfreader_version)) as timer:
        x = MDFreader(r'test.mf4', noDataLoading=True)
    output.append(timer.output)

def open_reader4_compression(path, output):
    os.chdir(path)
    with Timer('Open file','mdfreader {} compression mdfv4'.format(mdfreader_version)) as timer:
        x = MDFreader(r'test.mf4', compression='blosc')
    output.append(timer.output)

def open_reader4_compression_bcolz(path, output):
    os.chdir(path)
    with Timer('Open file','mdfreader {} compression bcolz 6 mdfv4'.format(mdfreader_version)) as timer:
        x = MDFreader(r'test.mf4', compression=6)
    output.append(timer.output)


def open_reader3(path, output):
    os.chdir(path)
    with Timer('Open file','mdfreader {} mdfv3'.format(mdfreader_version)) as timer:
        x = MDFreader(r'test.mdf')
    output.append(timer.output)

def save_reader3(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mdf')
    with Timer('Save file','mdfreader {} mdfv3'.format(mdfreader_version)) as timer:
        x.write(r'x.mdf')
    output.append(timer.output)

def save_reader3_nodata(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mdf', noDataLoading=True)
    with Timer('Save file', 'mdfreader {} noDataLoading mdfv3'.format(mdfreader_version)) as timer:
        x.write(r'x.mdf')
    output.append(timer.output)

def save_reader3_compression(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mdf', compression='blosc')
    with Timer('Save file', 'mdfreader {} compression mdfv3'.format(mdfreader_version)) as timer:
        x.write(r'x.mdf')
    output.append(timer.output)

def save_reader3_compression_bcolz(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mdf', compression=6)
    with Timer('Save file', 'mdfreader {} compression bcolz 6 mdfv3'.format(mdfreader_version)) as timer:
        x.write(r'x.mdf')
    output.append(timer.output)

def get_all_reader3(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mdf')
    with Timer('Get all channels', 'mdfreader {} mdfv3'.format(mdfreader_version)) as timer:
        for s in x:
            y = x.getChannelData(s)
    output.append(timer.output)

def get_all_reader3_nodata(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mdf', noDataLoading=True)
    with Timer('Get all channels', 'mdfreader {} nodata mdfv3'.format(mdfreader_version)) as timer:
        for s in x:
            y = x.getChannelData(s)
    output.append(timer.output)

def get_all_reader3_compression(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mdf', compression='blosc')
    with Timer('Get all channels', 'mdfreader {} compression mdfv3'.format(mdfreader_version)) as timer:
        for s in x:
            y = x.getChannelData(s)
    output.append(timer.output)

def get_all_reader3_compression_bcolz(path, output):
    os.chdir(path)
    x = MDFreader(r'test.mdf', compression=6)
    with Timer('Get all channels', 'mdfreader {} compression bcolz 6 mdfv3'.format(mdfreader_version)) as timer:
        for s in x:
            y = x.getChannelData(s)
    output.append(timer.output)

def open_reader3_nodata(path, output):
    os.chdir(path)
    with Timer('Open file','mdfreader {} noDataLoading mdfv3'.format(mdfreader_version)) as timer:
        x = MDFreader(r'test.mdf', noDataLoading=True)
    output.append(timer.output)

def open_reader3_compression(path, output):
    os.chdir(path)
    with Timer('Open file','mdfreader {} compression mdfv3'.format(mdfreader_version)) as timer:
        x = MDFreader(r'test.mdf', compression='blosc')
    output.append(timer.output)

def open_reader3_compression_bcolz(path, output):
    os.chdir(path)
    with Timer('Open file','mdfreader {} compression bcolz 6 mdfv3'.format(mdfreader_version)) as timer:
        x = MDFreader(r'test.mdf', compression=6)
    output.append(timer.output)


def table_header(topic, format='rst'):
    output = []
    if format == 'rst':
        output.append('')
        output.append('{} {} {}'.format('='*50, '='*9, '='*8))
        output.append('{:<50} {:>9} {:>8}'.format(topic, 'Time [ms]', 'RAM [MB]'))
        output.append('{} {} {}'.format('='*50, '='*9, '='*8))
    elif format == 'md':
        output.append('')
        output.append('|{}|{}|{}|'.format(topic, 'Time [ms]', 'RAM [MB]'))
        output.append('|-|-|-|')
    return output

def table_end(format='rst'):
    if format == 'rst':
        return ['{} {} {}'.format('='*50, '='*9, '='*8), '']
    elif format == 'md':
        return ['',]


def main(path, output_stream, format):
    manager = multiprocessing.Manager()
    output = manager.list()

    output.append('Benchmark environment\n')
    output.append('* {}'.format(sys.version))
    output.append('* {}'.format(platform.platform()))
    output.append('* {}'.format(platform.processor()))
    output.append('* {}GB installed RAM\n'.format(round(psutil.virtual_memory().total / 1024 / 1024 / 1024)))
    output.append('Notations used in the results\n')
    output.append('* nodata = asammdf MDF object created with load_measured_data=False (raw channel data not loaded into RAM)')
    output.append('* compression = mdfreader mdf object created with compression=blosc')
    output.append('* compression bcolz 6 = mdfreader mdf object created with compression=6')
    output.append('* noDataLoading = mdfreader mdf object read with noDataLoading=True')
    output.append('\nFiles used for benchmark:\n')
    output.append('* 183 groups')
    output.append('* 36424 channels\n\n')

    tests = (
                 open_mdf3,
                 open_mdf3_nodata,
                 open_reader3,
                 open_reader3_compression,
                 open_reader3_compression_bcolz,
                 open_reader3_nodata,
                 open_mdf4,
                 open_mdf4_nodata,
                 open_reader4,
                 open_reader4_compression,
                 open_reader4_compression_bcolz,
                 open_reader4_nodata
                 )
    if tests:
        print('Running open file tests')
        output.extend(table_header('Open file', format))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(path, output))
            thr.start()
            thr.join()
        output.extend(table_end(format))

    tests = (
                 save_mdf3,
                 save_mdf3_nodata,
                 save_reader3,
                 save_reader3_nodata,
                 save_reader3_compression,
                 save_reader3_compression_bcolz,
                 save_mdf4,
                 save_mdf4_nodata,
                 save_reader4,
                 save_reader4_nodata,
                 save_reader4_compression,
                 save_reader4_compression_bcolz,
                 )
    if tests:
        print('Running save file tests')
        output.extend(table_header('Save file', format))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(path, output))
            thr.start()
            thr.join()
        output.extend(table_end(format))

    tests = (
                 get_all_mdf3,
                 get_all_mdf3_nodata,
                 get_all_reader3,
                 get_all_reader3_nodata,
                 get_all_reader3_compression,
                 get_all_reader3_compression_bcolz,
                 get_all_mdf4,
                 get_all_mdf4_nodata,
                 get_all_reader4,
                 get_all_reader4_nodata,
                 get_all_reader4_compression,
                 get_all_reader4_compression_bcolz,
                 )
    if tests:
        print('Running get all channels file tests')
        output.extend(table_header('Get all channels (36424 calls)', format))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(path, output))
            thr.start()
            thr.join()
        output.extend(table_end(format))

    tests = (
                 convert_v3_v4,
                 convert_v3_v4_nodata,
                 convert_v4_v3,
                 convert_v4_v3_nodata,
                 )
    if tests:
        print('Running convert file tests')
        output.extend(table_header('Convert file', format))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(path, output))
            thr.start()
            thr.join()
        output.extend(table_end(format))

    tests = (
                 merge_v3,
                 merge_v3_nodata,
                 merge_v4,
                 merge_v4_nodata,
                 )
    if tests:
        print('Running merge files tests')
        output.extend(table_header('Merge files', format))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(path, output))
            thr.start()
            thr.join()
        output.extend(table_end(format))

    print('All tests executed')

    if output_stream == 'stdout':
        print('\n'.join(output))
    else:
        file = os.path.join('results', '{}_asammdf_{}_mdfreader_{}.txt'.format(
                                    'x86' if platform.architecture()[0] == '32bit' else 'x64',
                                    asammdf_version,
                                    mdfreader_version))
        with open(file, 'w') as out:
            out.write('\n'.join(output))

def _cmd_line_parser():
    '''
    return a command line parser. It is used when generating the documentation
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default = os.path.dirname(__file__),
                        help='path to test files')
    parser.add_argument('--output',
                        default='stdout',
                        nargs='?',
                        choices=['stdout', 'text'],
                        help='output stream')
    parser.add_argument('--format',
                        default='rst',
                        nargs='?',
                        choices=['rst', 'md'],
                        help='text formatting')

    return parser

if __name__ == '__main__':

    parser = _cmd_line_parser()
    args = parser.parse_args(sys.argv[1:])

    main(args.path, args.output, args.format)

    os.chdir(args.path)
    for file in ('x.mdf', 'x.mf4'):
        try:
            os.remove(file)
        except:
            pass

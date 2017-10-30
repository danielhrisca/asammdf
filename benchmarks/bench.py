"""
benchmark asammdf vs mdfreader
"""
from __future__ import print_function, division
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

from asammdf import MDF
from asammdf import __version__ as asammdf_version
from mdfreader import mdf as MDFreader
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

        if self.fmt == 'rst':
            self.output = '{:<50} {:>9} {:>8}'.format(self.message,
                                                      elapsed_time,
                                                      ram_usage)
        elif self.fmt == 'md':
            self.output = '|{:<50}|{:>9}|{:>8}|'.format(self.message,
                                                        elapsed_time,
                                                        ram_usage)

        if tracebackobj:
            info = StringIO()
            traceback.print_tb(tracebackobj, None, info)
            info.seek(0)
            info = info.read()
            self.error = '{} : {}\n{}'.format(self.topic, self.message, info)

        return True


def open_mdf3(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'asammdf {} mdfv3'.format(asammdf_version),
               fmt) as timer:
        MDF(r'test.mdf')
    output.send([timer.output, timer.error])


def open_mdf3_nodata(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'asammdf {} nodata mdfv3'.format(asammdf_version),
               fmt) as timer:
        MDF(r'test.mdf', load_measured_data=False)
    output.send([timer.output, timer.error])


def open_mdf4(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'asammdf {} mdfv4'.format(asammdf_version),
               fmt) as timer:
        MDF(r'test.mf4')
    output.send([timer.output, timer.error])


def open_mdf4_nodata(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'asammdf {} nodata mdfv4'.format(asammdf_version),
               fmt) as timer:
        MDF(r'test.mf4', load_measured_data=False)
    output.send([timer.output, timer.error])


def save_mdf3(path, output, fmt):
    os.chdir(path)
    x = MDF(r'test.mdf')
    with Timer('Save file',
               'asammdf {} mdfv3'.format(asammdf_version),
               fmt) as timer:
        x.save(r'x.mdf', overwrite=True)
    output.send([timer.output, timer.error])


def save_mdf3_nodata(path, output, fmt):
    os.chdir(path)
    x = MDF(r'test.mdf', load_measured_data=False)
    with Timer('Save file',
               'asammdf {} nodata mdfv3'.format(asammdf_version),
               fmt) as timer:
        x.save(r'x.mdf', overwrite=True)
    output.send([timer.output, timer.error])


def save_mdf4(path, output, fmt):
    os.chdir(path)
    x = MDF(r'test.mf4')
    with Timer('Save file',
               'asammdf {} mdfv4'.format(asammdf_version),
               fmt) as timer:
        x.save(r'x.mf4', overwrite=True)
    output.send([timer.output, timer.error])


def save_mdf4_nodata(path, output, fmt):
    os.chdir(path)
    x = MDF(r'test.mf4', load_measured_data=False)
    with Timer('Save file',
               'asammdf {} nodata mdfv4'.format(asammdf_version),
               fmt) as timer:
        x.save(r'x.mf4', overwrite=True)
    output.send([timer.output, timer.error])



def get_all_mdf3(path, output, fmt):
    os.chdir(path)
    x = MDF(r'test.mdf')
    with Timer('Get all channels',
               'asammdf {} mdfv3'.format(asammdf_version),
               fmt) as timer:
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                x.get(group=i, index=j, samples_only=True)
    output.send([timer.output, timer.error])


def get_all_mdf3_nodata(path, output, fmt):
    os.chdir(path)
    x = MDF(r'test.mdf', load_measured_data=False)
    with Timer('Get all channels',
               'asammdf {} nodata mdfv3'.format(asammdf_version),
               fmt) as timer:

        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                x.get(group=i, index=j, samples_only=True)
    output.send([timer.output, timer.error])


def get_all_mdf4(path, output, fmt):
    os.chdir(path)
    x = MDF(r'test.mf4')
    with Timer('Get all channels',
               'asammdf {} mdfv4'.format(asammdf_version),
               fmt) as timer:
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                x.get(group=i, index=j, samples_only=True)
    output.send([timer.output, timer.error])


def get_all_mdf4_nodata(path, output, fmt):
    os.chdir(path)
    x = MDF(r'test.mf4', load_measured_data=False)
    with Timer('Get all channels',
               'asammdf {} nodata mdfv4'.format(asammdf_version),
               fmt) as timer:
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                x.get(group=i, index=j, samples_only=True)
    output.send([timer.output, timer.error])


def convert_v3_v4(path, output, fmt):
    os.chdir(path)
    with MDF(r'test.mdf') as x:
        with Timer('Convert file',
                   'asammdf {} v3 to v4'.format(asammdf_version),
                   fmt) as timer:
            x.convert('4.10')
    output.send([timer.output, timer.error])


def convert_v3_v4_nodata(path, output, fmt):
    os.chdir(path)
    with MDF(r'test.mdf', load_measured_data=False) as x:
        with Timer('Convert file',
                   'asammdf {} v3 to v4 nodata'.format(asammdf_version),
                   fmt) as timer:
            y = x.convert(to='4.10', load_measured_data=False)
            y.close()
    output.send([timer.output, timer.error])


def convert_v4_v3(path, output, fmt):
    os.chdir(path)
    with MDF(r'test.mf4') as x:
        with Timer('Convert file',
                   'asammdf {} v4 to v3'.format(asammdf_version),
                   fmt) as timer:
            y = x.convert('3.30')
            y.close()
    output.send([timer.output, timer.error])


def convert_v4_v3_nodata(path, output, fmt):
    os.chdir(path)
    with MDF(r'test.mf4', load_measured_data=False) as x:
        with Timer('Convert file',
                   'asammdf {} v4 to v3 nodata'.format(asammdf_version),
                   fmt) as timer:
            y = x.convert('3.30', load_measured_data=False)
            y.close()
    output.send([timer.output, timer.error])


def merge_v3(path, output, fmt):
    os.chdir(path)
    files = [r'test.mdf', ] * 2
    with Timer('Merge files',
               'asammdf {} v3'.format(asammdf_version),
               fmt) as timer:
        MDF.merge(files)
    output.send([timer.output, timer.error])


def merge_v3_nodata(path, output, fmt):
    os.chdir(path)
    files = [r'test.mdf', ] * 2
    with Timer('Merge files',
               'asammdf {} v3 nodata'.format(asammdf_version),
               fmt) as timer:
        y = MDF.merge(files, load_measured_data=False)
        y.close()
    output.send([timer.output, timer.error])


def merge_v4(path, output, fmt):
    files = [r'test.mf4', ] * 2
    os.chdir(path)
    with Timer('Merge files',
               'asammdf {} v4'.format(asammdf_version),
               fmt) as timer:
        MDF.merge(files)
    output.send([timer.output, timer.error])


def merge_v4_nodata(path, output, fmt):
    files = [r'test.mf4', ] * 2
    os.chdir(path)
    with Timer('Merge files',
               'asammdf {} v4 nodata'.format(asammdf_version),
               fmt) as timer:
        y = MDF.merge(files, load_measured_data=False)
        y.close()
    output.send([timer.output, timer.error])

#
# mdfreader
#


def open_reader3(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'mdfreader {} mdfv3'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mdf')
    output.send([timer.output, timer.error])


def open_reader3_nodata(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'mdfreader {} noDataLoading mdfv3'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mdf', noDataLoading=True)
    output.send([timer.output, timer.error])


def open_reader3_compression(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'mdfreader {} compress mdfv3'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mdf', compression='blosc')
    output.send([timer.output, timer.error])


def open_reader3_compression_bcolz(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'mdfreader {} compress bcolz 6 mdfv3'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mdf', compression=6)
    output.send([timer.output, timer.error])


def open_reader4(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'mdfreader {} mdfv4'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mf4')
    output.send([timer.output, timer.error])


def open_reader4_nodata(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'mdfreader {} noDataLoading mdfv4'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mf4', noDataLoading=True)
    output.send([timer.output, timer.error])


def open_reader4_compression(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'mdfreader {} compress mdfv4'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mf4', compression='blosc')
    output.send([timer.output, timer.error])


def open_reader4_compression_bcolz(path, output, fmt):
    os.chdir(path)
    with Timer('Open file',
               'mdfreader {} compress bcolz 6 mdfv4'.format(mdfreader_version),
               fmt) as timer:
        MDFreader(r'test.mf4', compression=6)
    output.send([timer.output, timer.error])


def save_reader3(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mdf')
    with Timer('Save file',
               'mdfreader {} mdfv3'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mdf')
    output.send([timer.output, timer.error])


def save_reader3_nodata(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mdf', noDataLoading=True)
    with Timer('Save file',
               'mdfreader {} noDataLoading mdfv3'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mdf')
    output.send([timer.output, timer.error])


def save_reader3_compression(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mdf', compression='blosc')
    with Timer('Save file',
               'mdfreader {} compress mdfv3'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mdf')
    output.send([timer.output, timer.error])


def save_reader3_compression_bcolz(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mdf', compression=6)
    with Timer('Save file',
               'mdfreader {} compress bcolz 6 mdfv3'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mdf')
    output.send([timer.output, timer.error])


def save_reader4(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mf4')
    with Timer('Save file',
               'mdfreader {} mdfv4'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mf4')
    output.send([timer.output, timer.error])


def save_reader4_nodata(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mf4', noDataLoading=True)
    with Timer('Save file',
               'mdfreader {} noDataLoading mdfv4'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mf4')
    output.send([timer.output, timer.error])


def save_reader4_compression(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mf4', compression='blosc')
    with Timer('Save file',
               'mdfreader {} compress mdfv4'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mf4')
    output.send([timer.output, timer.error])


def save_reader4_compression_bcolz(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mf4', compression=6)
    with Timer('Save file',
               'mdfreader {} compress bcolz6 mdfv4'.format(mdfreader_version),
               fmt) as timer:
        x.write(r'x.mf4')
    output.send([timer.output, timer.error])


def get_all_reader3(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mdf')
    with Timer('Get all channels',
               'mdfreader {} mdfv3'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.getChannelData(s)
    output.send([timer.output, timer.error])


def get_all_reader3_nodata(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mdf', noDataLoading=True)
    with Timer('Get all channels',
               'mdfreader {} nodata mdfv3'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.getChannelData(s)
    output.send([timer.output, timer.error])


def get_all_reader3_compression(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mdf', compression='blosc')
    with Timer('Get all channels',
               'mdfreader {} compress mdfv3'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.getChannelData(s)
    output.send([timer.output, timer.error])


def get_all_reader3_compression_bcolz(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mdf', compression=6)
    with Timer('Get all channels',
               'mdfreader {} compress bcolz 6 mdfv3'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.getChannelData(s)
    output.send([timer.output, timer.error])


def get_all_reader4(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mf4')
    with Timer('Get all channels',
               'mdfreader {} mdfv4'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.getChannelData(s)
    output.send([timer.output, timer.error])


def get_all_reader4_nodata(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mf4', noDataLoading=True)
    with Timer('Get all channels',
               'mdfreader {} nodata mdfv4'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.getChannelData(s)
    output.send([timer.output, timer.error])


def get_all_reader4_compression(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mf4', compression='blosc')
    with Timer('Get all channels',
               'mdfreader {} compress mdfv4'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.getChannelData(s)
    output.send([timer.output, timer.error])


def get_all_reader4_compression_bcolz(path, output, fmt):
    os.chdir(path)
    x = MDFreader(r'test.mf4', compression=6)
    with Timer('Get all channels',
               'mdfreader {} compress bcolz 6 mdfv4'.format(mdfreader_version),
               fmt) as timer:
        for s in x:
            x.getChannelData(s)
    output.send([timer.output, timer.error])

#
# utility functions
#


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


def main(path, text_output, fmt):
    listen, send = multiprocessing.Pipe()
    output = MyList()
    errors = []

    if not path:
        path = os.path.dirname(__file__)

    installed_ram = round(psutil.virtual_memory().total / 1024 / 1024 / 1024)

    output.append('Benchmark environment\n')
    output.append('* {}'.format(sys.version))
    output.append('* {}'.format(platform.platform()))
    output.append('* {}'.format(platform.processor()))
    output.append('* {}GB installed RAM\n'.format(installed_ram))
    output.append('Notations used in the results\n')
    output.append(('* nodata = asammdf MDF object created with '
                   'load_measured_data=False '
                   '(raw channel data not loaded into RAM)'))
    output.append(('* compress = mdfreader mdf object created with '
                   'compression=blosc'))
    output.append(('* compression bcolz 6 = mdfreader mdf object created with '
                   'compression=6'))
    output.append(('* noDataLoading = mdfreader mdf object read with '
                   'noDataLoading=True'))
    output.append('\nFiles used for benchmark:\n')
    output.append('* 183 groups')
    output.append('* 36424 channels\n\n')

    tests = (
        open_mdf3,
        open_mdf3_nodata,
        open_reader3,
#        open_reader3_compression,
#        open_reader3_compression_bcolz,
#        open_reader3_nodata,
        open_mdf4,
        open_mdf4_nodata,
        open_reader4,
#        open_reader4_compression,
#        open_reader4_compression_bcolz,
#        open_reader4_nodata,
    )

    if tests:
        output.extend(table_header('Open file', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(path, send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        save_mdf3,
        save_mdf3_nodata,
        save_reader3,
        # save_reader3_nodata,
#        save_reader3_compression,
#        save_reader3_compression_bcolz,
        save_mdf4,
        save_mdf4_nodata,
        save_reader4,
        # save_reader4_nodata,
#        save_reader4_compression,
#        save_reader4_compression_bcolz,
    )

    if tests:
        output.extend(table_header('Save file', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(path, send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        get_all_mdf3,
        get_all_mdf3_nodata,
#        get_all_reader3,
#        # get_all_reader3_nodata,
#        get_all_reader3_compression,
#        get_all_reader3_compression_bcolz,
        get_all_mdf4,
        get_all_mdf4_nodata,
#        get_all_reader4,
#        # get_all_reader4_nodata,
#        get_all_reader4_compression,
#        get_all_reader4_compression_bcolz,
    )

    if tests:
        output.extend(table_header('Get all channels (36424 calls)', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(path, send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        convert_v3_v4,
        convert_v3_v4_nodata,
        convert_v4_v3,
        convert_v4_v3_nodata,
    )

    if tests:
        output.extend(table_header('Convert file', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(path, send, fmt))
            thr.start()
            thr.join()
            result, err = listen.recv()
            output.append(result)
            errors.append(err)
        output.extend(table_end(fmt))

    tests = (
        merge_v3,
        merge_v3_nodata,
        merge_v4,
        merge_v4_nodata,
    )

    if tests:
        output.extend(table_header('Merge files', fmt))
        for func in tests:
            thr = multiprocessing.Process(target=func, args=(path, send, fmt))
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

    os.chdir(path)
    for file in ('x.mdf', 'x.mf4'):
        try:
            os.remove(file)
        except FileNotFoundError:
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
                        help='text fmtting')

    return parser

if __name__ == '__main__':

    cmd_parser = _cmd_line_parser()
    args = cmd_parser.parse_args(sys.argv[1:])

    main(args.path, args.text_output, args.format)

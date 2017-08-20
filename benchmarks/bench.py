"""
bnachmark asammdf vs mdfreader
"""
from __future__ import print_function, division
import sys
PYVERSION = sys.version_info[0]

if PYVERSION > 2:
    from time import perf_counter
else:
    from time import clock as perf_counter

from asammdf import MDF
from asammdf import __version__ as asammdf_version
from mdfreader import mdf as MDFreader
from mdfreader import __version__ as mdfreader_version
import os
import multiprocessing
import psutil
import sys
import platform

class Timer():
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = perf_counter()
        return None

    def __exit__(self, type, value, traceback):
        elapsed_time = (perf_counter() - self.start) * 1000
        process = psutil.Process(os.getpid())
        print('{:<50} {:>9} {:>8}'.format(self.message, int(elapsed_time), int(process.memory_info().peak_wset / 1024 / 1024)))

path = r'D:\TMP'

def open_mdf3():
    os.chdir(path)
    with Timer('asammdf {} mdfv3'.format(asammdf_version)):
        x = MDF(r'test.mdf')

def save_mdf3():
    os.chdir(path)
    x = MDF(r'test.mdf')
    with Timer('asammdf {} mdfv3'.format(asammdf_version)):
        x.save(r'x.mdf')

def get_all_mdf3():
    os.chdir(path)
    x = MDF(r'test.mdf')
    with Timer('asammdf {} mdfv3'.format(asammdf_version)):
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)

def open_mdf3_nodata():
    os.chdir(path)
    with Timer('asammdf {} nodata mdfv3'.format(asammdf_version)):
        x = MDF(r'test.mdf', load_measured_data=False)

def get_all_mdf3_nodata():
    os.chdir(path)
    x = MDF(r'test.mdf', load_measured_data=False)
    with Timer('asammdf {} nodata mdfv3'.format(asammdf_version)):

        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)

def open_mdf3_compressed():
    os.chdir(path)
    with Timer('asammdf {} compression mdfv3'.format(asammdf_version)):
        x = MDF(r'test.mdf', compression=True)

def save_mdf3_compressed():
    os.chdir(path)
    x = MDF(r'test.mdf', compression=True)
    with Timer('asammdf {} compression mdfv3'.format(asammdf_version)):
        x.save(r'x.mdf')

def get_all_mdf3_compressed():
    os.chdir(path)
    x = MDF(r'test.mdf', compression=True)
    with Timer('asammdf {} compression mdfv3'.format(asammdf_version)):
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)

def open_mdf4():
    os.chdir(path)
    with Timer('asammdf {} mdfv4'.format(asammdf_version)):
        x = MDF(r'test.mf4')

def save_mdf4():
    os.chdir(path)
    x = MDF(r'test.mf4')
    with Timer('asammdf {} mdfv4'.format(asammdf_version)):
        x.save(r'x.mf4')

def get_all_mdf4():
    os.chdir(path)
    x = MDF(r'test.mf4')
    with Timer('asammdf {} mdfv4'.format(asammdf_version)):
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)

def open_mdf4_nodata():
    os.chdir(path)
    with Timer('asammdf {} nodata mdfv4'.format(asammdf_version)):
        x = MDF(r'test.mf4', load_measured_data=False)

def get_all_mdf4_nodata():
    os.chdir(path)
    x = MDF(r'test.mf4', load_measured_data=False)
    with Timer('asammdf {} nodata mdfv4'.format(asammdf_version)):
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j, samples_only=True)


def open_mdf4_compressed():
    os.chdir(path)
    with Timer('asammdf {} compression mdfv4'.format(asammdf_version)):
        x = MDF(r'test.mf4', compression=True)

def save_mdf4_compressed():
    os.chdir(path)
    x = MDF(r'test.mf4', compression=True)
    with Timer('asammdf {} compression mdfv4'.format(asammdf_version)):
        x.save(r'x.mf4')

def get_all_mdf4_compressed():
    os.chdir(path)
    x = MDF(r'test.mf4', compression=True)s
    with Timer('asammdf {} compression mdfv4'.format(asammdf_version)):
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get_channel_data(group=i, index=j, samples_only=True)

################
# mdf reader
#################


def open_reader4():
    os.chdir(path)
    with Timer('mdfreader {} mdfv4'.format(mdfreader_version)):
        x = MDFreader(r'test.mf4')

def save_reader4():
    os.chdir(path)
    x = MDFreader(r'test.mf4')
    with Timer('mdfreader {} mdfv4'.format(mdfreader_version)):
        x.write(r'x.mf4')

def get_all_reader4():
    os.chdir(path)
    x = MDFreader(r'test.mf4')
    with Timer('mdfreader {} mdfv4'.format(mdfreader_version)):
        for s in x:
            y = x.getChannelData(s)

def get_all_reader4_nodata():
    os.chdir(path)
    x = MDFreader(r'test.mf4', noDataLoading=True)
    with Timer('mdfreader {} nodata mdfv4'.format(mdfreader_version)):
        for s in x:
            y = x.getChannelData(s)

def open_reader4_nodata():
    os.chdir(path)
    with Timer('mdfreader {} noDataLoading mdfv4'.format(mdfreader_version)):
        x = MDFreader(r'test.mf4', noDataLoading=True)

def open_reader4_compression():
    os.chdir(path)
    with Timer('mdfreader {} compression mdfv4'.format(mdfreader_version)):
        x = MDFreader(r'test.mf4', compression='blosc')

def open_reader4_compression_bcolz():
    os.chdir(path)
    with Timer('mdfreader {} compression bcolz 6 mdfv4'.format(mdfreader_version)):
        x = MDFreader(r'test.mf4', compression=6)


def open_reader3():
    os.chdir(path)
    with Timer('mdfreader {} mdfv3'.format(mdfreader_version)):
        x = MDFreader(r'test.mdf')

def save_reader3():
    os.chdir(path)
    x = MDFreader(r'test.mdf')
    with Timer('mdfreader {} mdfv3'.format(mdfreader_version)):
        x.write(r'x.mdf')

def get_all_reader3():
    os.chdir(path)
    x = MDFreader(r'test.mdf')
    with Timer('mdfreader {} mdfv3'.format(mdfreader_version)):
        for s in x:
            y = x.getChannelData(s)

def get_all_reader3_nodata():
    os.chdir(path)
    x = MDFreader(r'test.mdf', noDataLoading=True)
    with Timer('mdfreader {} nodata mdfv3'.format(mdfreader_version)):
        for s in x:
            y = x.getChannelData(s)

def open_reader3_nodata():
    os.chdir(path)
    with Timer('mdfreader {} noDataLoading mdfv3'.format(mdfreader_version)):
        x = MDFreader(r'test.mdf', noDataLoading=True)

def open_reader3_compression():
    os.chdir(path)
    with Timer('mdfreader {} compression mdfv3'.format(mdfreader_version)):
        x = MDFreader(r'test.mdf', compression='blosc')

def open_reader3_compression_bcolz():
    os.chdir(path)
    with Timer('mdfreader {} compression bcolz 6 mdfv3'.format(mdfreader_version)):
        x = MDFreader(r'test.mdf', compression=6)


def main():
    print('Benchmark environment\n')
    print('* {}'.format(sys.version))
    print('* {}'.format(platform.platform()))
    print('* {}'.format(platform.processor()))
    print('* {}GB installed RAM\n'.format(round(psutil.virtual_memory().total / 1024 / 1024 / 1024)))
    print('Notations used in the results\n')
    print('* nodata = MDF object created with load_measured_data=False (raw channel data not loaded into RAM)')
    print('* compression = MDF object created with compression=True/blosc')
    print('* compression bcolz 6 = MDF object created with compression=6')
    print('* noDataLoading = MDF object read with noDataLoading=True')
    print('\nFiles used for benchmark:')
    print('* 183 groups')
    print('* 36424 channels\n\n')

    print('{} {} {}'.format('='*50, '='*9, '='*8))
    print('{:<50} {:>9} {:>8}'.format('Open file', 'Time [ms]', 'RAM [MB]'))
    print('{} {} {}'.format('='*50, '='*9, '='*8))
    for func in (open_mdf3,
                 open_mdf3_compressed,
                 open_mdf3_nodata,
                 open_reader3,
                 open_reader3_compression,
                 open_reader3_compression_bcolz,
                 open_reader3_nodata,
                 open_mdf4,
                 open_mdf4_compressed,
                 open_mdf4_nodata,
                 open_reader4,
                 open_reader4_compression,
                 open_reader4_compression_bcolz,
                 open_reader4_nodata
                 ):
        thr = multiprocessing.Process(target=func, args=())
        thr.start()
        thr.join()
    print('{} {} {}'.format('='*50, '='*9, '='*8))

    print('\n')

    print('{} {} {}'.format('='*50, '='*9, '='*8))
    print('{:<50} {:>9} {:>8}'.format('Save file', 'Time [ms]', 'RAM [MB]'))
    print('{} {} {}'.format('='*50, '='*9, '='*8))
    for func in (save_mdf3,
                 save_mdf3_compressed,
                 save_reader3,
                 save_mdf4,
                 save_mdf4_compressed,
                 save_reader4):
        thr = multiprocessing.Process(target=func, args=())
        thr.start()
        thr.join()
    print('{} {} {}'.format('='*50, '='*9, '='*8))

    print('\n')

    print('{} {} {}'.format('='*50, '='*9, '='*8))
    print('{:<50} {:>9} {:>8}'.format('Get all channels (36424 calls)', 'Time [ms]', 'RAM [MB]'))
    print('{} {} {}'.format('='*50, '='*9, '='*8))
    for func in (
                 get_all_mdf3,
                 get_all_mdf3_compressed,
                 get_all_mdf3_nodata,
                 get_all_reader3,
                 get_all_reader3_nodata,
                 get_all_mdf4,
                 get_all_mdf4_compressed,
                 get_all_mdf4_nodata,
                 get_all_reader4,
                 get_all_reader4_nodata
                 ):
        thr = multiprocessing.Process(target=func, args=())
        thr.start()
        thr.join()
    print('{} {} {}'.format('='*50, '='*9, '='*8))


if __name__ == '__main__':
    main()

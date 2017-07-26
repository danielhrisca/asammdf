"""
bnachmark asammdf vs mdfreader
"""

from time import perf_counter
from asammdf import MDF
from mdfreader import mdf as MDFreader
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

path = r'E:\TMP'

def open_mdf3():
    os.chdir(path)
    with Timer('asammdf 2.0.0 mdfv3'):
        x = MDF(r'test.mdf')

def save_mdf3():
    os.chdir(path)
    x = MDF(r'test.mdf')
    with Timer('asammdf 2.0.0 mdfv3'):
        x.save(r'x.mdf')

def get_all_mdf3():
    os.chdir(path)
    x = MDF(r'test.mdf')
    with Timer('asammdf 2.0.0 mdfv3'):
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j)

def open_mdf3_nodata():
    os.chdir(path)
    with Timer('asammdf 2.0.0 nodata mdfv3'):
        x = MDF(r'test.mdf', load_measured_data=False)

def get_all_mdf3_nodata():
    os.chdir(path)
    x = MDF(r'test.mdf', load_measured_data=False)
    with Timer('asammdf 2.0.0 nodata mdfv3'):

        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j)

def open_mdf3_compressed():
    os.chdir(path)
    with Timer('asammdf 2.0.0 compression mdfv3'):
        x = MDF(r'test.mdf', compression=True)

def save_mdf3_compressed():
    os.chdir(path)
    x = MDF(r'test.mdf', compression=True)
    with Timer('asammdf 2.0.0 compression mdfv3'):
        x.save(r'x.mdf')

def get_all_mdf3_compressed():
    os.chdir(path)
    x = MDF(r'test.mdf', compression=True)
    with Timer('asammdf 2.0.0 compression mdfv3'):
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j)

def open_mdf4():
    os.chdir(path)
    with Timer('asammdf 2.0.0 mdfv4'):
        x = MDF(r'test.mf4')

def save_mdf4():
    os.chdir(path)
    x = MDF(r'test.mf4')
    with Timer('asammdf 2.0.0 mdfv4'):
        x.save(r'x.mf4')

def get_all_mdf4():
    os.chdir(path)
    x = MDF(r'test.mf4')
    with Timer('asammdf 2.0.0 mdfv4'):
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j)

def open_mdf4_nodata():
    os.chdir(path)
    with Timer('asammdf 2.0.0 nodata mdfv4'):
        x = MDF(r'test.mf4', load_measured_data=False)

def get_all_mdf4_nodata():
    os.chdir(path)
    x = MDF(r'test.mf4', load_measured_data=False)
    with Timer('asammdf 2.0.0 nodata mdfv4'):
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j)

def open_mdf4_compressed():
    os.chdir(path)
    with Timer('asammdf 2.0.0 compression mdfv4'):
        x = MDF(r'test.mf4', compression=True)

def save_mdf4_compressed():
    os.chdir(path)
    x = MDF(r'test.mf4', compression=True)
    with Timer('asammdf 2.0.0 compression mdfv4'):
        x.save(r'x.mf4')

def get_all_mdf4_compressed():
    os.chdir(path)
    x = MDF(r'test.mf4', compression=True)
    with Timer('asammdf 2.0.0 compression mdfv4'):
        for i, gp in enumerate(x.groups):
            for j in range(len(gp['channels'])):
                y = x.get(group=i, index=j)

#################
# mdf reader
#################


def open_reader4():
    os.chdir(path)
    with Timer('mdfreader 0.2.5 mdfv4'):
        x = MDFreader(r'test.mf4')

def save_reader4():
    os.chdir(path)
    x = MDFreader(r'test.mf4')
    with Timer('mdfreader 0.2.5 mdfv4'):
        x.write(r'x.mf4')

def get_all_reader4():
    os.chdir(path)
    x = MDFreader(r'test.mf4')
    with Timer('mdfreader 0.2.5 mdfv4'):
        for s in x:
            y = x.getChannelData(s)

def open_reader4_nodata():
    os.chdir(path)
    with Timer('mdfreader 0.2.5 noconvert mdfv4'):
        x = MDFreader(r'test.mf4', convertAfterRead=False)


def open_reader3():
    os.chdir(path)
    with Timer('mdfreader 0.2.5 mdfv3'):
        x = MDFreader(r'test.mdf')

def save_reader3():
    os.chdir(path)
    x = MDFreader(r'test.mdf')
    with Timer('mdfreader 0.2.5 mdfv3'):
        x.write(r'x.mdf')

def get_all_reader3():
    os.chdir(path)
    x = MDFreader(r'test.mdf')
    with Timer('mdfreader 0.2.5 mdfv3'):
        for s in x:
            y = x.getChannelData(s)

def open_reader3_nodata():
    os.chdir(path)
    with Timer('mdfreader 0.2.5 no convert mdfv3'):
        x = MDFreader(r'test.mdf', convertAfterRead=False)


def main():
    print('Benchmark environment\n')
    print('* {}'.format(sys.version))
    print('* {}'.format(platform.platform()))
    print('* {}'.format(platform.processor()))
    print('* {}GB installed RAM\n'.format(round(psutil.virtual_memory().total / 1024 / 1024 / 1024)))
    print('Notations used in the results\n')
    print('* nodata = MDF object created with load_measured_data=False (raw channel data no loaded into RAM)')
    print('* compression = MDF object created with compression=True (raw channel data loaded into RAM and compressed)')
    print('* noconvert = MDF object created with convertAfterRead=False')
    print('\nFiles used for benchmark:')
    print('* 183 groups')
    print('* 36424 channels\n')

    print('{} {} {}'.format('='*50, '='*9, '='*8))
    print('{:<50} {:>9} {:>8}'.format('Open file', 'Time [ms]', 'RAM [MB]'))
    print('{} {} {}'.format('='*50, '='*9, '='*8))
    for func in (open_mdf3,
                 open_mdf3_compressed,
                 open_mdf3_nodata,
                 open_reader3,
                 open_reader3_nodata,
                 open_mdf4,
                 open_mdf4_compressed,
                 open_mdf4_nodata,
                 open_reader4,
                 open_reader4_nodata):
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
    for func in (get_all_mdf3,
                 get_all_mdf3_compressed,
                 get_all_mdf3_nodata,
                 get_all_reader3,
                 get_all_mdf4,
                 get_all_mdf4_compressed,
                 get_all_mdf4_nodata,
                 get_all_reader4):
        thr = multiprocessing.Process(target=func, args=())
        thr.start()
        thr.join()
    print('{} {} {}'.format('='*50, '='*9, '='*8))


if __name__ == '__main__':
    main()

*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports both MDF version 3 and 4 formats. 

*asammdf* works on Python 2.7, and Python >= 3.4

Project goals
=============
The main goals for this library are:

* to be faster than the other Python based mdf libraries
* to have clean and easy to understand code base

Features
========

* create new mdf files from scratch
* append new channels
* read unsorted MDF v3 and v4 files
* filter a subset of channels from original mdf file
* cut measurement to specified time interval
* convert to different mdf version
* export to Excel, HDF5, Matlab and CSV
* merge multiple files sharing the same internal structure
* read and save mdf version 4.10 files containing zipped data blocks
* full support (read, append, save) for the following map types (multidimensional array channels):

    * mdf version 3 channels with CDBLOCK
    * mdf version 4 structure channel composition
    * mdf version 4 channel arrays with CNTemplate storage and one of the array types:
    
        * 0 - array
        * 1 - scaling axis
        * 2 - look-up
        
* add and extract attachments for mdf version 4
* files are loaded in RAM for fast operations
* handle large files (exceeding the available RAM) using *load_measured_data* = *False* argument
* extract channel data, master channel and extra channel information as *Signal* objects for unified operations with v3 and v4 files
* time domain operation using the *Signal* class

    * Pandas data frames are good if all the channels have the same time based
    * usually a measurement will have channels from different sources at different rates
    * the *Signal* class facilitates operations with such channels

Major features not implemented (yet)
====================================

* for version 3

    * functionality related to sample reduction block (but the class is defined)
    
* for version 4

    * handling of bus logging measurements
    * handling of unfinnished measurements (mdf 4)
    * full support mdf 4 channel arrays
    * xml schema for TXBLOCK and MDBLOCK
    * partial conversions
    * event blocks

Usage
=====

.. code-block: python

   from asammdf import MDF
   
   mdf = MDF('sample.mdf')
   speed = mdf.get('WheelSpeed')
   speed.plot()
   
   important_signals = ['WheelSpeed', 'VehicleSpeed', 'VehicleAcceleration']
   # get short measurement with a subset of channels from 10s to 12s 
   short = mdf.filter(important_signals).cut(start=10, stop=12)
   
   # convert to version 4.10 and save to disk
   short.convert('4.10').save('important signals.mf4')

 
Check the *examples* folder for extended usage demo.

Documentation
=============
http://asammdf.readthedocs.io/en/latest

Installation
============
*asammdf* is available on 

* github: https://github.com/danielhrisca/asammdf/
* PyPI: https://pypi.org/project/asammdf/
    
.. code-block: python

   pip install asammdf

    
Dependencies
============
asammdf uses the following libraries

* numpy : the heart that makes all tick
* numexpr : for algebraic and rational channel conversions
* matplotlib : for Signal plotting
* wheel : for installation in virtual environments

optional dependencies needed for exports

* pandas : for DataFrame export
* h5py : for HDF5 export
* xlsxwriter : for Excel export
* scipy : for Matlab .mat export


Benchmarks
==========

Python 3 x86
------------

Benchmark environment

* 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 17:54:52) [MSC v.1900 32 bit (Intel)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results

* nodata = asammdf MDF object created with load_measured_data=False (raw channel data not loaded into RAM)
* compression = mdfreader mdf object created with compression=blosc
* compression bcolz 6 = mdfreader mdf object created with compression=6
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.2 mdfv3                                     1191      286
asammdf 2.6.2 nodata mdfv3                               706      118
mdfreader 0.2.6 mdfv3                                   3910      458
mdfreader 0.2.6 compression mdfv3                       5040      185
mdfreader 0.2.6 compression bcolz 6 mdfv3               5274      941
mdfreader 0.2.6 noDataLoading mdfv3                     2033      120
asammdf 2.6.2 mdfv4                                     2237      330
asammdf 2.6.2 nodata mdfv4                              1969      150
mdfreader 0.2.6 mdfv4                                   7759      870
mdfreader 0.2.6 compression mdfv4                       9439      587
mdfreader 0.2.6 compression bcolz 6 mdfv4               7679     1294
mdfreader 0.2.6 noDataLoading mdfv4                     4878      522
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.2 mdfv3                                      434      290
asammdf 2.6.2 nodata mdfv3                               475      125
mdfreader 0.2.6 mdfv3                                   9329      481
mdfreader 0.2.6 compression mdfv3                       9743      452
mdfreader 0.2.6 compression bcolz 6 mdfv3               9806      941
asammdf 2.6.2 mdfv4                                      639      334
asammdf 2.6.2 nodata mdfv4                               636      159
mdfreader 0.2.6 mdfv4                                   7679      891
mdfreader 0.2.6 compression mdfv4                       7436      852
mdfreader 0.2.6 compression bcolz 6 mdfv4               7027     1312
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.2 mdfv3                                      804      294
asammdf 2.6.2 nodata mdfv3                             19036      130
mdfreader 0.2.6 mdfv3                                     78      458
mdfreader 0.2.6 noDataLoading mdfv3                 18000000      118
mdfreader 0.2.6 compression mdfv3                        724      188
mdfreader 0.2.6 compression bcolz 6 mdfv3                305      943
asammdf 2.6.2 mdfv4                                      883      335
asammdf 2.6.2 nodata mdfv4                             26520      160
mdfreader 0.2.6 mdfv4                                     77      870
mdfreader 0.2.6 noDataLoading mdfv4                 18000000      523
mdfreader 0.2.6 compression mdfv4                        684      594
mdfreader 0.2.6 compression bcolz 6 mdfv4                355     1302
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.2 v3 to v4                                  6359      685
asammdf 2.6.2 v3 to v4 nodata                          31124      479
asammdf 2.6.2 v4 to v3                                  5778      680
asammdf 2.6.2 v4 to v3 nodata                          36685      627
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.2 v3                                       13305     1228
asammdf 2.6.2 v3 nodata                                54322      343
asammdf 2.6.2 v4                                       16648     1267
asammdf 2.6.2 v4 nodata                                72303      364
================================================== ========= ========





Python 3 x64
------------
Benchmark environment

* 3.6.2 (v3.6.2:5fd33b5, Jul  8 2017, 04:57:36) [MSC v.1900 64 bit (AMD64)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results

* nodata = asammdf MDF object created with load_measured_data=False (raw channel data not loaded into RAM)
* compression = mdfreader mdf object created with compression=blosc
* compression bcolz 6 = mdfreader mdf object created with compression=6
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.2 mdfv3                                      757      364
asammdf 2.6.2 nodata mdfv3                               537      188
mdfreader 0.2.6 mdfv3                                   2619      545
mdfreader 0.2.6 compression mdfv3                       3928      269
mdfreader 0.2.6 compression bcolz 6 mdfv3               3826     1041
mdfreader 0.2.6 noDataLoading mdfv3                     1408      198
asammdf 2.6.2 mdfv4                                     1785      435
asammdf 2.6.2 nodata mdfv4                              1460      244
mdfreader 0.2.6 mdfv4                                   5246     1308
mdfreader 0.2.6 compression mdfv4                       6468     1023
mdfreader 0.2.6 compression bcolz 6 mdfv4               6689     1746
mdfreader 0.2.6 noDataLoading mdfv4                     3798      944
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.2 mdfv3                                      367      367
asammdf 2.6.2 nodata mdfv3                               375      194
mdfreader 0.2.6 mdfv3                                   8522      577
mdfreader 0.2.6 compression mdfv3                       8144      542
mdfreader 0.2.6 compression bcolz 6 mdfv3               7676     1040
asammdf 2.6.2 mdfv4                                      457      440
asammdf 2.6.2 nodata mdfv4                               473      255
mdfreader 0.2.6 mdfv4                                   6006     1091
mdfreader 0.2.6 compression mdfv4                       6271     1288
mdfreader 0.2.6 compression bcolz 6 mdfv4               5932     1763
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.2 mdfv3                                      593      373
asammdf 2.6.2 nodata mdfv3                              9008      203
mdfreader 0.2.6 mdfv3                                     63      545
mdfreader 0.2.6 noDataLoading mdfv3                 18000000      198
mdfreader 0.2.6 compression mdfv3                        631      271
mdfreader 0.2.6 compression bcolz 6 mdfv3                261     1043
asammdf 2.6.2 mdfv4                                      623      443
asammdf 2.6.2 nodata mdfv4                             16745      258
mdfreader 0.2.6 mdfv4                                     60     1308
mdfreader 0.2.6 noDataLoading mdfv4                 18000000      943
mdfreader 0.2.6 compression mdfv4                        631     1032
mdfreader 0.2.6 compression bcolz 6 mdfv4                281     1754
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.2 v3 to v4                                  4540      833
asammdf 2.6.2 v3 to v4 nodata                          22162      578
asammdf 2.6.2 v4 to v3                                  4909      837
asammdf 2.6.2 v4 to v3 nodata                          30383      723
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.2 v3                                       10287     1442
asammdf 2.6.2 v3 nodata                                30281      526
asammdf 2.6.2 v4                                       13297     1523
asammdf 2.6.2 v4 nodata                                51197      587
================================================== ========= ========


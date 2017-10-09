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
asammdf 2.6.0 mdfv3                                      888      287
asammdf 2.6.0 nodata mdfv3                               609      118
mdfreader 0.2.6 mdfv3                                   3457      458
mdfreader 0.2.6 compression mdfv3                       4665      184
mdfreader 0.2.6 compression bcolz 6 mdfv3               4619      940
mdfreader 0.2.6 noDataLoading mdfv3                     1890      120
asammdf 2.6.0 mdfv4                                     1971      330
asammdf 2.6.0 nodata mdfv4                              1630      150
mdfreader 0.2.6 mdfv4                                   6414      870
mdfreader 0.2.6 compression mdfv4                       7495      587
mdfreader 0.2.6 compression bcolz 6 mdfv4               7473     1294
mdfreader 0.2.6 noDataLoading mdfv4                     4418      523
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.0 mdfv3                                      450      290
asammdf 2.6.0 nodata mdfv3                               457      125
mdfreader 0.2.6 mdfv3                                   9455      481
mdfreader 0.2.6 noDataLoading mdfv3                     1314      289
mdfreader 0.2.6 compression mdfv3                       9263      451
mdfreader 0.2.6 compression bcolz 6 mdfv3               9305      941
asammdf 2.6.0 mdfv4                                      617      334
asammdf 2.6.0 nodata mdfv4                               601      159
mdfreader 0.2.6 mdfv4                                   7063      890
mdfreader 0.2.6 noDataLoading mdfv4                     1452      694
mdfreader 0.2.6 compression mdfv4                       7227      851
mdfreader 0.2.6 compression bcolz 6 mdfv4               6954     1312
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.0 mdfv3                                      754      294
asammdf 2.6.0 nodata mdfv3                             18843      130
mdfreader 0.2.6 mdfv3                                     80      458
mdfreader 0.2.6 noDataLoading mdfv3                 18000000      118
mdfreader 0.2.6 compression mdfv3                        690      188
mdfreader 0.2.6 compression bcolz 6 mdfv3                317      943
asammdf 2.6.0 mdfv4                                      784      335
asammdf 2.6.0 nodata mdfv4                             20635      160
mdfreader 0.2.6 mdfv4                                     79      870
mdfreader 0.2.6 noDataLoading mdfv4                 18000000      523
mdfreader 0.2.6 compression mdfv4                        704      594
mdfreader 0.2.6 compression bcolz 6 mdfv4                333     1302
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.0 v3 to v4                                  5720      679
asammdf 2.6.0 v3 to v4 nodata                          28738      479
asammdf 2.6.0 v4 to v3                                  5731      682
asammdf 2.6.0 v4 to v3 nodata                          30795      627
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.0 v3                                       12988     1206
asammdf 2.6.0 v3 nodata                                53020      322
asammdf 2.6.0 v4                                       15434     1244
asammdf 2.6.0 v4 nodata                                60260      344
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
asammdf 2.6.0 mdfv3                                      780      364
asammdf 2.6.0 nodata mdfv3                               562      187
mdfreader 0.2.6 mdfv3                                   2825      545
mdfreader 0.2.6 compression mdfv3                       4198      268
mdfreader 0.2.6 compression bcolz 6 mdfv3               4041     1040
mdfreader 0.2.6 noDataLoading mdfv3                     1466      198
asammdf 2.6.0 mdfv4                                     1717      435
asammdf 2.6.0 nodata mdfv4                              1351      244
mdfreader 0.2.6 mdfv4                                   5589     1308
mdfreader 0.2.6 compression mdfv4                       6794     1023
mdfreader 0.2.6 compression bcolz 6 mdfv4               6853     1747
mdfreader 0.2.6 noDataLoading mdfv4                     4035      943
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.0 mdfv3                                      392      365
asammdf 2.6.0 nodata mdfv3                               421      195
mdfreader 0.2.6 mdfv3                                   8355      577
mdfreader 0.2.6 noDataLoading mdfv3                     1159      370
mdfreader 0.2.6 compression mdfv3                       8510      543
mdfreader 0.2.6 compression bcolz 6 mdfv3               8150     1041
asammdf 2.6.0 mdfv4                                      438      441
asammdf 2.6.0 nodata mdfv4                               479      255
mdfreader 0.2.6 mdfv4                                   6491     1329
mdfreader 0.2.6 noDataLoading mdfv4                     1142     1117
mdfreader 0.2.6 compression mdfv4                       6637     1288
mdfreader 0.2.6 compression bcolz 6 mdfv4               6349     1764
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.0 mdfv3                                      601      373
asammdf 2.6.0 nodata mdfv3                              9213      203
mdfreader 0.2.6 mdfv3                                     71      545
mdfreader 0.2.6 noDataLoading mdfv3                 18000000      198
mdfreader 0.2.6 compression mdfv3                        663      272
mdfreader 0.2.6 compression bcolz 6 mdfv3                275     1041
asammdf 2.6.0 mdfv4                                      650      443
asammdf 2.6.0 nodata mdfv4                             13256      257
mdfreader 0.2.6 mdfv4                                     63     1307
mdfreader 0.2.6 noDataLoading mdfv4                 18000000      943
mdfreader 0.2.6 compression mdfv4                        657     1031
mdfreader 0.2.6 compression bcolz 6 mdfv4                292     1754
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.0 v3 to v4                                  4658      832
asammdf 2.6.0 v3 to v4 nodata                          22138      578
asammdf 2.6.0 v4 to v3                                  5026      838
asammdf 2.6.0 v4 to v3 nodata                          26169      723
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.0 v3                                       10739     1390
asammdf 2.6.0 v3 nodata                                31730      478
asammdf 2.6.0 v4                                       13171     1482
asammdf 2.6.0 v4 nodata                                43173      545
================================================== ========= ========


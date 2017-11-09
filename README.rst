*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports both MDF version 3 and 4 formats. 

*asammdf* works on Python 2.7, and Python >= 3.4

Project goals
=============
The main goals for this library are:

* to be faster than the other Python based mdf libraries
* to have clean and easy to understand code base
* to have minimal 3-rd party dependencies

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
* split large data blocks (configurable size) for mdf version 4
* disk space savings by compacting 1-dimensional integer channels (configurable)
* full support (read, append, save) for the following map types (multidimensional array channels):

    * mdf version 3 channels with CDBLOCK
    * mdf version 4 structure channel composition
    * mdf version 4 channel arrays with CNTemplate storage and one of the array types:
    
        * 0 - array
        * 1 - scaling axis
        * 2 - look-up
        
* add and extract attachments for mdf version 4
* files are loaded in RAM for fast operations
* handle large files (exceeding the available RAM) using *memory* = *minimum* argument
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
    * full support for remaining mdf 4 channel arrays types
    * xml schema for TXBLOCK and MDBLOCK
    * partial conversions
    * event blocks
    * channels with default X axis
    * chanenls with reference to attachment

Usage
=====

.. code-block:: python

   from asammdf import MDF
   
   mdf = MDF('sample.mdf')
   speed = mdf.get('WheelSpeed')
   speed.plot()
   
   important_signals = ['WheelSpeed', 'VehicleSpeed', 'VehicleAcceleration']
   # get short measurement with a subset of channels from 10s to 12s 
   short = mdf.filter(important_signals).cut(start=10, stop=12)
   
   # convert to version 4.10 and save to disk
   short.convert('4.10').save('important signals.mf4')
   
   # plot some channels from a huge file
   efficient = MDF('huge.mf4', load_measured_data=False)
   for signal in efficient.select(['Sensor1', 'Voltage3']):
       signal.plot()
   

 
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

Graphical results can be seen here at http://asammdf.readthedocs.io/en/latest/benchmarks.html


Python 3 x86
------------
Benchmark environment

* 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 17:54:52) [MSC v.1900 32 bit (Intel)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results

* full =  asammdf MDF object created with memory=full (everything loaded into RAM)
* low =  asammdf MDF object created with memory=low (raw channel data not loaded into RAM, but metadata loaded to RAM)
* minimum =  asammdf MDF object created with memory=full (lowest possible RAM usage)
* compress = mdfreader mdf object created with compression=blosc
* compression bcolz 6 = mdfreader mdf object created with compression=6
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.7.0 full mdfv3                                 892      279
asammdf 2.7.0 low mdfv3                                  794      126
asammdf 2.7.0 minimum mdfv3                              523       71
mdfreader 0.2.7 mdfv3                                   2978      421
mdfreader 0.2.7 compress mdfv3                          4625      152
mdfreader 0.2.7 compress bcolz 6 mdfv3                  4308     1307
mdfreader 0.2.7 noDataLoading mdfv3                      812      121
asammdf 2.7.0 full mdfv4                                2296      318
asammdf 2.7.0 low mdfv4                                 2139      152
asammdf 2.7.0 minimum mdfv4                             1599       77
mdfreader 0.2.7 mdfv4                                   5662      421
mdfreader 0.2.7 compress mdfv4                          6847      137
mdfreader 0.2.7 compress bcolz 6 mdfv4                  7033     1200
mdfreader 0.2.7 noDataLoading mdfv4                     3759      134
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.7.0 full mdfv3                                 395      282
asammdf 2.7.0 low mdfv3                                  492      133
asammdf 2.7.0 minimum mdfv3                             1197       78
mdfreader 0.2.7 mdfv3                                   9073      435
mdfreader 0.2.7 noDataLoading mdfv3                    10121      464
mdfreader 0.2.7 compress mdfv3                          9323      407
mdfreader 0.2.7 compress bcolz 6 mdfv3                  9053     1307
asammdf 2.7.0 full mdfv4                                 550      322
asammdf 2.7.0 low mdfv4                                  639      162
asammdf 2.7.0 minimum mdfv4                             2672       86
mdfreader 0.2.7 mdfv4                                   8705      440
mdfreader 0.2.7 noDataLoading mdfv4                     7930      500
mdfreader 0.2.7 compress mdfv4                          8836      401
mdfreader 0.2.7 compress bcolz6 mdfv4                   8609     1214
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.7.0 full mdfv3                                 854      284
asammdf 2.7.0 low mdfv3                                12495      136
asammdf 2.7.0 minimum mdfv3                            13589       82
mdfreader 0.2.7 mdfv3                                     76      421
mdfreader 0.2.7 nodata mdfv3                            1419      327
mdfreader 0.2.7 compress mdfv3                           699      153
mdfreader 0.2.7 compress bcolz 6 mdfv3                   294     1307
asammdf 2.7.0 full mdfv4                                 885      323
asammdf 2.7.0 low mdfv4                                15095      160
asammdf 2.7.0 minimum mdfv4                            18019       85
mdfreader 0.2.7 mdfv4                                     72      421
mdfreader 0.2.7 nodata mdfv4                            1914      351
mdfreader 0.2.7 compress mdfv4                           706      142
mdfreader 0.2.7 compress bcolz 6 mdfv4                   314     1205
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.7.0 full v3 to v4                             3997      383
asammdf 2.7.0 low v3 to v4                              4474      234
asammdf 2.7.0 minimum v3 to v4                          5185      182
asammdf 2.7.0 full v4 to v3                             4634      378
asammdf 2.7.0 low v4 to v3                              5111      213
asammdf 2.7.0 minimum v4 to v3                          7996      140
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.7.0 full v3                                  10048     1184
asammdf 2.7.0 low v3                                   11128      339
asammdf 2.7.0 minimum v3                               13078      201
mdfreader 0.2.7 v3                                        0*       0*
asammdf 2.7.0 full v4                                  14038     1241
asammdf 2.7.0 low v4                                   15429      371
asammdf 2.7.0 minimum v4                               20086      185
mdfreader 0.2.7 v4                                        0*       0*
================================================== ========= ========

* mdfreader got a MemoryError



Python 3 x64
------------
Benchmark environment

* 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 18:41:36) [MSC v.1900 64 bit (AMD64)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results

* full =  asammdf MDF object created with memory=full (everything loaded into RAM)
* low =  asammdf MDF object created with memory=low (raw channel data not loaded into RAM, but metadata loaded to RAM)
* minimum =  asammdf MDF object created with memory=full (lowest possible RAM usage)
* compress = mdfreader mdf object created with compression=blosc
* compression bcolz 6 = mdfreader mdf object created with compression=6
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.7.0 full mdfv3                                 737      339
asammdf 2.7.0 low mdfv3                                  648      187
asammdf 2.7.0 minimum mdfv3                              395       98
mdfreader 0.2.7 mdfv3                                   2310      465
mdfreader 0.2.7 compress mdfv3                          3565      200
mdfreader 0.2.7 compress bcolz 6 mdfv3                  3706     1535
mdfreader 0.2.7 noDataLoading mdfv3                      658      188
asammdf 2.7.0 full mdfv4                                1840      403
asammdf 2.7.0 low mdfv4                                 1765      238
asammdf 2.7.0 minimum mdfv4                             1261      110
mdfreader 0.2.7 mdfv4                                   4660      467
mdfreader 0.2.7 compress mdfv4                          5813      181
mdfreader 0.2.7 compress bcolz 6 mdfv4                  6113     1433
mdfreader 0.2.7 noDataLoading mdfv4                     3226      211
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.7.0 full mdfv3                                 329      342
asammdf 2.7.0 low mdfv3                                  383      194
asammdf 2.7.0 minimum mdfv3                              926      107
mdfreader 0.2.7 mdfv3                                   8053      482
mdfreader 0.2.7 noDataLoading mdfv3                     8762      566
mdfreader 0.2.7 compress mdfv3                          7975      451
mdfreader 0.2.7 compress bcolz 6 mdfv3                  7875     1534
asammdf 2.7.0 full mdfv4                                 412      408
asammdf 2.7.0 low mdfv4                                  464      248
asammdf 2.7.0 minimum mdfv4                             2003      118
mdfreader 0.2.7 mdfv4                                   7498      485
mdfreader 0.2.7 noDataLoading mdfv4                     6767      595
mdfreader 0.2.7 compress mdfv4                          7701      441
mdfreader 0.2.7 compress bcolz6 mdfv4                   7517     1444
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.7.0 full mdfv3                                 635      346
asammdf 2.7.0 low mdfv3                                 3222      199
asammdf 2.7.0 minimum mdfv3                             4347      113
mdfreader 0.2.7 mdfv3                                     58      464
mdfreader 0.2.7 nodata mdfv3                            1117      403
mdfreader 0.2.7 compress mdfv3                           599      199
mdfreader 0.2.7 compress bcolz 6 mdfv3                   248     1534
asammdf 2.7.0 full mdfv4                                 687      410
asammdf 2.7.0 low mdfv4                                 6612      248
asammdf 2.7.0 minimum mdfv4                             8661      122
mdfreader 0.2.7 mdfv4                                     56      467
mdfreader 0.2.7 nodata mdfv4                            1506      444
mdfreader 0.2.7 compress mdfv4                           598      187
mdfreader 0.2.7 compress bcolz 6 mdfv4                   278     1439
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.7.0 full v3 to v4                             3505      498
asammdf 2.7.0 low v3 to v4                              3697      352
asammdf 2.7.0 minimum v3 to v4                          4426      267
asammdf 2.7.0 full v4 to v3                             3788      497
asammdf 2.7.0 low v4 to v3                              4225      334
asammdf 2.7.0 minimum v4 to v3                          6625      210
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.7.0 full v3                                   7828     1333
asammdf 2.7.0 low v3                                    9350      476
asammdf 2.7.0 minimum v3                               11020      249
mdfreader 0.2.7 v3                                     11437     2963
asammdf 2.7.0 full v4                                  11869     1455
asammdf 2.7.0 low v4                                   12764      571
asammdf 2.7.0 minimum v4                               16559      249
mdfreader 0.2.7 v4                                     16126     2966
================================================== ========= ========





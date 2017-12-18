*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports MDF versions 2 (.dat), 3 (.mdf) and 4 (.mf4). 

*asammdf* works on Python 2.7, and Python >= 3.4 (Travis CI tests done with Python 2.7 and Python >= 3.5)

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
    * handling of unfinished measurements (mdf 4)
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
   efficient = MDF('huge.mf4', memory='minimum')
   for signal in efficient.select(['Sensor1', 'Voltage3']):
       signal.plot()
   

 
Check the *examples* folder for extended usage demo, or the documentation
http://asammdf.readthedocs.io/en/master/examples.html

Documentation
=============
http://asammdf.readthedocs.io/en/master

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
* pandas : for DataFrame export

optional dependencies needed for exports

* h5py : for HDF5 export
* xlsxwriter : for Excel export
* scipy : for Matlab .mat export


Benchmarks
==========

Graphical results can be seen here at http://asammdf.readthedocs.io/en/master/benchmarks.html


Python 3 x86
------------
Benchmark environment

* 3.6.3 (v3.6.3:2c5fed8, Oct  3 2017, 17:26:49) [MSC v.1900 32 bit (Intel)]
* Windows-10-10.0.16299-SP0
* Intel64 Family 6 Model 69 Stepping 1, GenuineIntel
* 16GB installed RAM

Notations used in the results

* full =  asammdf MDF object created with memory=full (everything loaded into RAM)
* low =  asammdf MDF object created with memory=low (raw channel data not loaded into RAM, but metadata loaded to RAM)
* minimum =  asammdf MDF object created with memory=full (lowest possible RAM usage)
* compress = mdfreader mdf object created with compression=blosc
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full mdfv3                                1259      260
asammdf 2.8.1 low mdfv3                                 1076      106
asammdf 2.8.1 minimum mdfv3                              767       52
mdfreader 2.7.3 mdfv3                                   3146      392
mdfreader 2.7.3 noDataLoading mdfv3                     1159      102
asammdf 2.8.1 full mdfv4                                2792      299
asammdf 2.8.1 low mdfv4                                 2645      133
asammdf 2.8.1 minimum mdfv4                             2070       58
mdfreader 2.7.3 mdfv4                                   7372      397
mdfreader 2.7.3 noDataLoading mdfv4                     4526      104
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full mdfv3                                 581      263
asammdf 2.8.1 low mdfv3                                  688      114
asammdf 2.8.1 minimum mdfv3                             1931       58
mdfreader 2.7.3 mdfv3                                   8902      412
mdfreader 2.7.3 noDataLoading mdfv3                    10490      420
asammdf 2.8.1 full mdfv4                                 843      303
asammdf 2.8.1 low mdfv4                                  959      143
asammdf 2.8.1 minimum mdfv4                             3698       67
mdfreader 2.7.3 mdfv4                                   8084      417
mdfreader 2.7.3 noDataLoading mdfv4                     9524      426
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full mdfv3                                1278      265
asammdf 2.8.1 low mdfv3                                18354      116
asammdf 2.8.1 minimum mdfv3                            19288       63
mdfreader 2.7.3 mdfv3                                    117      392
asammdf 2.8.1 full mdfv4                                1266      303
asammdf 2.8.1 low mdfv4                                20515      141
asammdf 2.8.1 minimum mdfv4                            23939       65
mdfreader 2.7.3 mdfv4                                    116      398
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full v3 to v4                             5667      638
asammdf 2.8.1 low v3 to v4                              6483      215
asammdf 2.8.1 minimum v3 to v4                          8301      117
asammdf 2.8.1 full v4 to v3                             6910      635
asammdf 2.8.1 low v4 to v3                              7938      195
asammdf 2.8.1 minimum v4 to v3                         12352       94
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full v3                                  14564     1165
asammdf 2.8.1 low v3                                   16148      319
asammdf 2.8.1 minimum v3                               19046      180
mdfreader 2.7.3 v3                                     16765      928
asammdf 2.8.1 full v4                                  21262     1223
asammdf 2.8.1 low v4                                   23150      352
asammdf 2.8.1 minimum v4                               30687      166
mdfreader 2.7.3 v4                                     25437      919
================================================== ========= ========




Python 3 x64
------------
Benchmark environment

* 3.6.2 (v3.6.2:5fd33b5, Jul  8 2017, 04:57:36) [MSC v.1900 64 bit (AMD64)]
* Windows-10-10.0.16299-SP0
* Intel64 Family 6 Model 69 Stepping 1, GenuineIntel
* 16GB installed RAM

Notations used in the results

* full =  asammdf MDF object created with memory=full (everything loaded into RAM)
* low =  asammdf MDF object created with memory=low (raw channel data not loaded into RAM, but metadata loaded to RAM)
* minimum =  asammdf MDF object created with memory=full (lowest possible RAM usage)
* compress = mdfreader mdf object created with compression=blosc
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full mdfv3                                1100      327
asammdf 2.8.1 low mdfv3                                  980      174
asammdf 2.8.1 minimum mdfv3                              599       86
mdfreader 2.7.3 mdfv3                                   2567      436
mdfreader 2.7.3 compress mdfv3                          4324      135
mdfreader 2.7.3 noDataLoading mdfv3                      973      176
asammdf 2.8.1 full mdfv4                                2613      390
asammdf 2.8.1 low mdfv4                                 2491      225
asammdf 2.8.1 minimum mdfv4                             1749       97
mdfreader 2.7.3 mdfv4                                   6457      448
mdfreader 2.7.3 compress mdfv4                          8219      147
mdfreader 2.7.3 noDataLoading mdfv4                     4221      180
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full mdfv3                                 676      327
asammdf 2.8.1 low mdfv3                                  541      181
asammdf 2.8.1 minimum mdfv3                             1363       95
mdfreader 2.7.3 mdfv3                                   8013      465
mdfreader 2.7.3 noDataLoading mdfv3                     8948      476
mdfreader 2.7.3 compress mdfv3                          7629      432
asammdf 2.8.1 full mdfv4                                 672      395
asammdf 2.8.1 low mdfv4                                  736      237
asammdf 2.8.1 minimum mdfv4                             3127      107
mdfreader 2.7.3 mdfv4                                   7237      467
mdfreader 2.7.3 noDataLoading mdfv4                     8332      473
mdfreader 2.7.3 compress mdfv4                          6791      426
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full mdfv3                                 967      333
asammdf 2.8.1 low mdfv3                                 5690      186
asammdf 2.8.1 minimum mdfv3                             7296       99
mdfreader 2.7.3 mdfv3                                     95      436
mdfreader 2.7.3 compress mdfv3                           531      135
asammdf 2.8.1 full mdfv4                                 988      397
asammdf 2.8.1 low mdfv4                                10572      234
asammdf 2.8.1 minimum mdfv4                            13803      108
mdfreader 2.7.3 mdfv4                                     95      448
mdfreader 2.7.3 compress mdfv4                           534      148
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full v3 to v4                             4986      759
asammdf 2.8.1 low v3 to v4                              5573      340
asammdf 2.8.1 minimum v3 to v4                          7049      171
asammdf 2.8.1 full v4 to v3                             5705      761
asammdf 2.8.1 low v4 to v3                              6510      321
asammdf 2.8.1 minimum v4 to v3                         10434      142
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full v3                                  12251     1320
asammdf 2.8.1 low v3                                   14453      464
asammdf 2.8.1 minimum v3                               16830      236
mdfreader 2.7.3 v3                                     15635      983
mdfreader 2.7.3 compress v3                            20812      993
asammdf 2.8.1 full v4                                  18172     1441
asammdf 2.8.1 low v4                                   20083      558
asammdf 2.8.1 minimum v4                               26374      237
mdfreader 2.7.3 v4                                     23450      981
mdfreader 2.7.3 compress v4                            28421      985
================================================== ========= ========





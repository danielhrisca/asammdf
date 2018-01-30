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
asammdf 2.8.1 full mdfv3                                1207      260
asammdf 2.8.1 low mdfv3                                 1065      107
asammdf 2.8.1 minimum mdfv3                              746       52
mdfreader 2.7.4 mdfv3                                   3061      392
mdfreader 2.7.4 noDataLoading mdfv3                     1154      106
asammdf 2.8.1 full mdfv4                                2811      298
asammdf 2.8.1 low mdfv4                                 2708      134
asammdf 2.8.1 minimum mdfv4                             2081       58
mdfreader 2.7.4 mdfv4                                   7293      397
mdfreader 2.7.4 noDataLoading mdfv4                     4557      109
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full mdfv3                                 564      264
asammdf 2.8.1 low mdfv3                                  628      115
asammdf 2.8.1 minimum mdfv3                             1780       58
mdfreader 2.7.4 mdfv3                                   9021      412
mdfreader 2.7.4 noDataLoading mdfv3                       0*       0*
asammdf 2.8.1 full mdfv4                                 798      303
asammdf 2.8.1 low mdfv4                                  916      143
asammdf 2.8.1 minimum mdfv4                             3992       67
mdfreader 2.7.4 mdfv4                                   8069      417
mdfreader 2.7.4 noDataLoading mdfv4                     9646      434
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full mdfv3                                1226      265
asammdf 2.8.1 low mdfv3                                17517      117
asammdf 2.8.1 minimum mdfv3                            19145       63
mdfreader 2.7.4 mdfv3                                    120      392
mdfreader 2.7.4 nodata mdfv3                           30561      130
asammdf 2.8.1 full mdfv4                                1234      304
asammdf 2.8.1 low mdfv4                                20214      141
asammdf 2.8.1 minimum mdfv4                            23583       65
mdfreader 2.7.4 mdfv4                                    115      397
mdfreader 2.7.4 nodata mdfv4                           38428      123
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full v3 to v4                             5507      638
asammdf 2.8.1 low v3 to v4                              6345      215
asammdf 2.8.1 minimum v3 to v4                          8098      118
asammdf 2.8.1 full v4 to v3                             6761      635
asammdf 2.8.1 low v4 to v3                              7732      194
asammdf 2.8.1 minimum v4 to v3                         12232       94
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full v3                                  14283     1166
asammdf 2.8.1 low v3                                   15639      320
asammdf 2.8.1 minimum v3                               18547      181
mdfreader 2.7.4 v3                                     16451      929
mdfreader 2.7.4 nodata v3                                 0*       0*
asammdf 2.8.1 full v4                                  20925     1223
asammdf 2.8.1 low v4                                   22659      352
asammdf 2.8.1 minimum v4                               29923      166
mdfreader 2.7.4 v4                                     25032      919
mdfreader 2.7.4 nodata v4                              24316      948
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
asammdf 2.8.1 full mdfv3                                1054      317
asammdf 2.8.1 low mdfv3                                  919      164
asammdf 2.8.1 minimum mdfv3                              592       76
mdfreader 2.7.4 mdfv3                                   2545      426
mdfreader 2.7.4 compress mdfv3                          4188      126
mdfreader 2.7.4 noDataLoading mdfv3                     1015      173
asammdf 2.8.1 full mdfv4                                2438      380
asammdf 2.8.1 low mdfv4                                 2311      215
asammdf 2.8.1 minimum mdfv4                             1649       87
mdfreader 2.7.4 mdfv4                                   6176      438
mdfreader 2.7.4 compress mdfv4                          7940      137
mdfreader 2.7.4 noDataLoading mdfv4                     4013      180
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full mdfv3                                 507      319
asammdf 2.8.1 low mdfv3                                  515      171
asammdf 2.8.1 minimum mdfv3                             1263       84
mdfreader 2.7.4 mdfv3                                   7590      454
mdfreader 2.7.4 noDataLoading mdfv3                       0*       0*
mdfreader 2.7.4 compress mdfv3                          7236      423
asammdf 2.8.1 full mdfv4                                 599      385
asammdf 2.8.1 low mdfv4                                  703      227
asammdf 2.8.1 minimum mdfv4                             3157       97
mdfreader 2.7.4 mdfv4                                   6764      457
mdfreader 2.7.4 noDataLoading mdfv4                     8053      476
mdfreader 2.7.4 compress mdfv4                          6677      416
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full mdfv3                                1016      323
asammdf 2.8.1 low mdfv3                                 5599      177
asammdf 2.8.1 minimum mdfv3                             7105       91
mdfreader 2.7.4 mdfv3                                    102      426
mdfreader 2.7.4 nodata mdfv3                           16651      208
mdfreader 2.7.4 compress mdfv3                           515      126
asammdf 2.8.1 full mdfv4                                1080      388
asammdf 2.8.1 low mdfv4                                10658      225
asammdf 2.8.1 minimum mdfv4                            13554       98
mdfreader 2.7.4 mdfv4                                     91      438
mdfreader 2.7.4 nodata mdfv4                           26847      204
mdfreader 2.7.4 compress mdfv4                           517      138
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full v3 to v4                             4995      750
asammdf 2.8.1 low v3 to v4                              5646      330
asammdf 2.8.1 minimum v3 to v4                          6902      161
asammdf 2.8.1 full v4 to v3                             5750      751
asammdf 2.8.1 low v4 to v3                              6572      313
asammdf 2.8.1 minimum v4 to v3                         10229      133
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.1 full v3                                  12050     1311
asammdf 2.8.1 low v3                                   14122      454
asammdf 2.8.1 minimum v3                               16537      227
mdfreader 2.7.4 v3                                     14710      974
mdfreader 2.7.4 compress v3                            19571      982
asammdf 2.8.1 full v4                                  17569     1431
asammdf 2.8.1 low v4                                   19297      548
asammdf 2.8.1 minimum v4                               25442      227
mdfreader 2.7.4 v4                                     22324      971
mdfreader 2.7.4 nodata v4                              21581     1013
mdfreader 2.7.4 compress v4                            26916      974
================================================== ========= ========

*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports MDF versions 2 (.dat), 3 (.mdf) and 4 (.mf4). 

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

* 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 17:54:52) [MSC v.1900 32 bit (Intel)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
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
asammdf 2.8.0 full mdfv3                                 918      264
asammdf 2.8.0 low mdfv3                                  898      110
asammdf 2.8.0 minimum mdfv3                              577       56
mdfreader 2.7.2 mdfv3                                   2462      395
mdfreader 2.7.2 compress mdfv3                          4174       97
mdfreader 2.7.2 noDataLoading mdfv3                      911      105
asammdf 2.8.0 full mdfv4                                2644      302
asammdf 2.8.0 low mdfv4                                 2269      137
asammdf 2.8.0 minimum mdfv4                             1883       62
mdfreader 2.7.2 mdfv4                                   5869      403
mdfreader 2.7.2 compress mdfv4                          7367      101
mdfreader 2.7.2 noDataLoading mdfv4                     3897      110
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 452      267
asammdf 2.8.0 low mdfv3                                  495      118
asammdf 2.8.0 minimum mdfv3                             1206       62
mdfreader 2.7.2 mdfv3                                   9258      415
asammdf 2.8.0 full mdfv4                                 642      307
asammdf 2.8.0 low mdfv4                                  693      146
asammdf 2.8.0 minimum mdfv4                             2642       71
mdfreader 2.7.2 mdfv4                                   8548      422
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 889      268
asammdf 2.8.0 low mdfv3                                12707      120
asammdf 2.8.0 minimum mdfv3                            13644       66
mdfreader 2.7.2 mdfv3                                     80      395
mdfreader 2.7.2 nodata mdfv3                            1413      310
mdfreader 2.7.2 compress mdfv3                           529       97
asammdf 2.8.0 full mdfv4                                 968      307
asammdf 2.8.0 low mdfv4                                14475      144
asammdf 2.8.0 minimum mdfv4                            17057       69
mdfreader 2.7.2 mdfv4                                     72      403
mdfreader 2.7.2 nodata mdfv4                            1806      325
mdfreader 2.7.2 compress mdfv4                           562      107
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full v3 to v4                             4048      642
asammdf 2.8.0 low v3 to v4                              4551      219
asammdf 2.8.0 minimum v3 to v4                          5847      121
asammdf 2.8.0 full v4 to v3                             4394      639
asammdf 2.8.0 low v4 to v3                              5239      198
asammdf 2.8.0 minimum v4 to v3                          8392       98
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full v3                                  10061     1168
asammdf 2.8.0 low v3                                   11245      323
asammdf 2.8.0 minimum v3                               13618      186
asammdf 2.8.0 full v4                                  14144     1226
asammdf 2.8.0 low v4                                   15410      355
asammdf 2.8.0 minimum v4                               21417      170
================================================== ========= ========

Observations

* mdfreader got a MemoryError in the merge tests



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
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 772      319
asammdf 2.8.0 low mdfv3                                  656      165
asammdf 2.8.0 minimum mdfv3                              441       77
mdfreader 2.7.2 mdfv3                                   1783      428
mdfreader 2.7.2 compress mdfv3                          3330      127
mdfreader 2.7.2 noDataLoading mdfv3                      699      167
asammdf 2.8.0 full mdfv4                                1903      381
asammdf 2.8.0 low mdfv4                                 1783      216
asammdf 2.8.0 minimum mdfv4                             1348       88
mdfreader 2.7.2 mdfv4                                   4849      442
mdfreader 2.7.2 compress mdfv4                          6347      138
mdfreader 2.7.2 noDataLoading mdfv4                     3425      176
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 359      321
asammdf 2.8.0 low mdfv3                                  415      172
asammdf 2.8.0 minimum mdfv3                              993       86
mdfreader 2.7.2 mdfv3                                   8402      456
mdfreader 2.7.2 compress mdfv3                          8364      424
asammdf 2.8.0 full mdfv4                                 497      387
asammdf 2.8.0 low mdfv4                                  507      228
asammdf 2.8.0 minimum mdfv4                             2179       97
mdfreader 2.7.2 mdfv4                                   7958      460
mdfreader 2.7.2 compress mdfv4                          8170      417
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full mdfv3                                 772      325
asammdf 2.8.0 low mdfv3                                 3784      179
asammdf 2.8.0 minimum mdfv3                             5076       92
mdfreader 2.7.2 mdfv3                                     65      428
mdfreader 2.7.2 nodata mdfv3                            1231      379
mdfreader 2.7.2 compress mdfv3                           487      127
asammdf 2.8.0 full mdfv4                                 800      389
asammdf 2.8.0 low mdfv4                                 7025      226
asammdf 2.8.0 minimum mdfv4                             9518      100
mdfreader 2.7.2 mdfv4                                     71      442
mdfreader 2.7.2 nodata mdfv4                            1575      404
mdfreader 2.7.2 compress mdfv4                           508      145
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full v3 to v4                             3461      751
asammdf 2.8.0 low v3 to v4                              4092      331
asammdf 2.8.0 minimum v3 to v4                          4852      163
asammdf 2.8.0 full v4 to v3                             3732      753
asammdf 2.8.0 low v4 to v3                              4348      313
asammdf 2.8.0 minimum v4 to v3                          7136      134
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.8.0 full v3                                   8152     1312
asammdf 2.8.0 low v3                                    9839      456
asammdf 2.8.0 minimum v3                               11694      228
mdfreader 2.7.2 v3                                     10352     2927
mdfreader 2.7.2 compress v3                            15314     2940
asammdf 2.8.0 full v4                                  11938     1434
asammdf 2.8.0 low v4                                   13154      549
asammdf 2.8.0 minimum v4                               17188      229
mdfreader 2.7.2 v4                                     16536     2941
mdfreader 2.7.2 compress v4                            21261     2951
================================================== ========= ========





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
* disk space savings by compacting 1-dimensional integer channels
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

Graphical results can be seen here at http://asammdf.readthedocs.io/en/stable/benchmarks.html


Python 3 x86
------------
Benchmark environment

* 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 17:54:52) [MSC v.1900 32 bit (Intel)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results

* nodata = asammdf MDF object created with load_measured_data=False (raw channel data not loaded into RAM)
* compress = mdfreader mdf object created with compression=blosc
* compression bcolz 6 = mdfreader mdf object created with compression=6
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.5 mdfv3                                      916      286
asammdf 2.6.5 nodata mdfv3                               623      118
mdfreader 0.2.6 mdfv3                                   3373      458
mdfreader 0.2.6 compress mdfv3                          4526      184
mdfreader 0.2.6 compress bcolz 6 mdfv3                  4518      940
mdfreader 0.2.6 noDataLoading mdfv3                     1833      120
asammdf 2.6.5 mdfv4                                     2214      330
asammdf 2.6.5 nodata mdfv4                              1695      150
mdfreader 0.2.6 mdfv4                                   6348      870
mdfreader 0.2.6 compress mdfv4                          7262      586
mdfreader 0.2.6 compress bcolz 6 mdfv4                  7552     1294
mdfreader 0.2.6 noDataLoading mdfv4                     4797      522
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.5 mdfv3                                      462      290
asammdf 2.6.5 nodata mdfv3                               521      125
mdfreader 0.2.6 mdfv3                                   9175      481
mdfreader 0.2.6 compress mdfv3                          9727      452
mdfreader 0.2.6 compress bcolz 6 mdfv3                  9284      940
asammdf 2.6.5 mdfv4                                      657      334
asammdf 2.6.5 nodata mdfv4                               710      159
mdfreader 0.2.6 mdfv4                                   6706      891
mdfreader 0.2.6 compress mdfv4                          7030      851
mdfreader 0.2.6 compress bcolz6 mdfv4                   6693     1311
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.5 mdfv3                                      791      291
asammdf 2.6.5 nodata mdfv3                             18430      128
mdfreader 0.2.6 mdfv3                                     78      457
mdfreader 0.2.6 compress mdfv3                           738      187
mdfreader 0.2.6 compress bcolz 6 mdfv3                   299      941
asammdf 2.6.5 mdfv4                                      863      334
asammdf 2.6.5 nodata mdfv4                             20637      157
mdfreader 0.2.6 mdfv4                                     77      869
mdfreader 0.2.6 compress mdfv4                           653      593
mdfreader 0.2.6 compress bcolz 6 mdfv4                   313     1301
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.5 v3 to v4                                  3843      680
asammdf 2.6.5 v3 to v4 nodata                           4656      242
asammdf 2.6.5 v4 to v3                                  4261      681
asammdf 2.6.5 v4 to v3 nodata                           5231      225
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.5 v3                                       10058     1248
asammdf 2.6.5 v3 nodata                                11174      363
asammdf 2.6.5 v4                                       14232     1282
asammdf 2.6.5 v4 nodata                                14629      380
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
* compress = mdfreader mdf object created with compression=blosc
* compression bcolz 6 = mdfreader mdf object created with compression=6
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.5 mdfv3                                      779      364
asammdf 2.6.5 nodata mdfv3                               551      187
mdfreader 0.2.6 mdfv3                                   2672      545
mdfreader 0.2.6 compress mdfv3                          3844      267
mdfreader 0.2.6 compress bcolz 6 mdfv3                  3886     1040
mdfreader 0.2.6 noDataLoading mdfv3                     1400      198
asammdf 2.6.5 mdfv4                                     1883      435
asammdf 2.6.5 nodata mdfv4                              1457      244
mdfreader 0.2.6 mdfv4                                   5371     1307
mdfreader 0.2.6 compress mdfv4                          6470     1023
mdfreader 0.2.6 compress bcolz 6 mdfv4                  6894     1746
mdfreader 0.2.6 noDataLoading mdfv4                     4078      943
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.5 mdfv3                                      356      366
asammdf 2.6.5 nodata mdfv3                               398      195
mdfreader 0.2.6 mdfv3                                  10164      577
mdfreader 0.2.6 compress mdfv3                         12341      542
mdfreader 0.2.6 compress bcolz 6 mdfv3                 11427      958
asammdf 2.6.5 mdfv4                                      805      440
asammdf 2.6.5 nodata mdfv4                               522      255
mdfreader 0.2.6 mdfv4                                   7256     1328
mdfreader 0.2.6 compress mdfv4                          7010     1288
mdfreader 0.2.6 compress bcolz6 mdfv4                   6688     1763
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.5 mdfv3                                      657      370
asammdf 2.6.5 nodata mdfv3                              9647      200
mdfreader 0.2.6 mdfv3                                     67      544
mdfreader 0.2.6 compress mdfv3                           698      270
mdfreader 0.2.6 compress bcolz 6 mdfv3                   267     1042
asammdf 2.6.5 mdfv4                                      736      443
asammdf 2.6.5 nodata mdfv4                             13552      254
mdfreader 0.2.6 mdfv4                                     64     1307
mdfreader 0.2.6 compress mdfv4                           631     1031
mdfreader 0.2.6 compress bcolz 6 mdfv4                   304     1753
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.5 v3 to v4                                  3675      823
asammdf 2.6.5 v3 to v4 nodata                           4607      379
asammdf 2.6.5 v4 to v3                                  4442      831
asammdf 2.6.5 v4 to v3 nodata                           5105      366
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.5 v3                                        8605     1449
asammdf 2.6.5 v3 nodata                                11089      544
asammdf 2.6.5 v4                                       13469     1536
asammdf 2.6.5 v4 nodata                                15565      600
================================================== ========= ========




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
asammdf 2.6.4 mdfv3                                      926      286
asammdf 2.6.4 nodata mdfv3                               615      118
mdfreader 0.2.6 mdfv3                                   3345      458
mdfreader 0.2.6 compress mdfv3                          4520      185
mdfreader 0.2.6 compress bcolz 6 mdfv3                  4635      941
mdfreader 0.2.6 noDataLoading mdfv3                     1867      120
asammdf 2.6.4 mdfv4                                     2250      330
asammdf 2.6.4 nodata mdfv4                              1706      150
mdfreader 0.2.6 mdfv4                                   6413      869
mdfreader 0.2.6 compress mdfv4                          7368      586
mdfreader 0.2.6 compress bcolz 6 mdfv4                  7733     1294
mdfreader 0.2.6 noDataLoading mdfv4                     4474      523
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.4 mdfv3                                      407      290
asammdf 2.6.4 nodata mdfv3                               447      126
mdfreader 0.2.6 mdfv3                                   8865      481
mdfreader 0.2.6 compress mdfv3                          8919      451
mdfreader 0.2.6 compress bcolz 6 mdfv3                  8548      941
asammdf 2.6.4 mdfv4                                      578      334
asammdf 2.6.4 nodata mdfv4                               617      159
mdfreader 0.2.6 mdfv4                                   6758      891
mdfreader 0.2.6 compress mdfv4                          6999      852
mdfreader 0.2.6 compress bcolz6 mdfv4                   6639     1312
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.4 mdfv3                                      818      291
asammdf 2.6.4 nodata mdfv3                             18416      128
mdfreader 0.2.6 mdfv3                                     77      458
mdfreader 0.2.6 compress mdfv3                           665      188
mdfreader 0.2.6 compress bcolz 6 mdfv3                   291      943
asammdf 2.6.4 mdfv4                                      860      335
asammdf 2.6.4 nodata mdfv4                             25362      157
mdfreader 0.2.6 mdfv4                                    162      794
mdfreader 0.2.6 compress mdfv4                           710      593
mdfreader 0.2.6 compress bcolz 6 mdfv4                   336     1301
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.4 v3 to v4                                  4389      680
asammdf 2.6.4 v3 to v4 nodata                          26231      472
asammdf 2.6.4 v4 to v3                                  4586      681
asammdf 2.6.4 v4 to v3 nodata                          34042      622
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.4 v3                                       10262     1243
asammdf 2.6.4 v3 nodata                                48898      352
asammdf 2.6.4 v4                                       14443     1281
asammdf 2.6.4 v4 nodata                                67092      377
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
asammdf 2.6.4 mdfv3                                      781      364
asammdf 2.6.4 nodata mdfv3                               570      187
mdfreader 0.2.6 mdfv3                                   2675      545
mdfreader 0.2.6 compress mdfv3                          3791      268
mdfreader 0.2.6 compress bcolz 6 mdfv3                  3910     1040
mdfreader 0.2.6 noDataLoading mdfv3                     1436      199
asammdf 2.6.4 mdfv4                                     1921      435
asammdf 2.6.4 nodata mdfv4                              1476      244
mdfreader 0.2.6 mdfv4                                   5520     1307
mdfreader 0.2.6 compress mdfv4                          6529     1024
mdfreader 0.2.6 compress bcolz 6 mdfv4                  6757     1746
mdfreader 0.2.6 noDataLoading mdfv4                     3948      943
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.4 mdfv3                                      375      365
asammdf 2.6.4 nodata mdfv3                               360      194
mdfreader 0.2.6 mdfv3                                   7983      578
mdfreader 0.2.6 compress mdfv3                          7966      543
mdfreader 0.2.6 compress bcolz 6 mdfv3                  7566     1041
asammdf 2.6.4 mdfv4                                      493      440
asammdf 2.6.4 nodata mdfv4                               444      256
mdfreader 0.2.6 mdfv4                                   6015     1329
mdfreader 0.2.6 compress mdfv4                          6105     1288
mdfreader 0.2.6 compress bcolz6 mdfv4                   5875     1763
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.4 mdfv3                                      636      370
asammdf 2.6.4 nodata mdfv3                              8535      200
mdfreader 0.2.6 mdfv3                                     59      545
mdfreader 0.2.6 compress mdfv3                           605      270
mdfreader 0.2.6 compress bcolz 6 mdfv3                   255     1042
asammdf 2.6.4 mdfv4                                      675      443
asammdf 2.6.4 nodata mdfv4                             16774      254
mdfreader 0.2.6 mdfv4                                     61     1308
mdfreader 0.2.6 compress mdfv4                           598     1030
mdfreader 0.2.6 compress bcolz 6 mdfv4                   276     1753
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.4 v3 to v4                                  3420      823
asammdf 2.6.4 v3 to v4 nodata                          18877      572
asammdf 2.6.4 v4 to v3                                  4009      832
asammdf 2.6.4 v4 to v3 nodata                          28683      718
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.4 v3                                        8251     1448
asammdf 2.6.4 v3 nodata                                27406      535
asammdf 2.6.4 v4                                       12183     1537
asammdf 2.6.4 v4 nodata                                48747      602
================================================== ========= ========



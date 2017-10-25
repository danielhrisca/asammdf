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
asammdf 2.6.3 mdfv3                                      951      286
asammdf 2.6.3 nodata mdfv3                               639      118
mdfreader 0.2.6 mdfv3                                   3490      458
mdfreader 0.2.6 compress mdfv3                          4624      185
mdfreader 0.2.6 compress bcolz 6 mdfv3                  4654      941
mdfreader 0.2.6 noDataLoading mdfv3                     1884      120
asammdf 2.6.3 mdfv4                                     2251      330
asammdf 2.6.3 nodata mdfv4                              1791      150
mdfreader 0.2.6 mdfv4                                   6447      869
mdfreader 0.2.6 compress mdfv4                          7549      586
mdfreader 0.2.6 compress bcolz 6 mdfv4                  7730     1294
mdfreader 0.2.6 noDataLoading mdfv4                     4553      522
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.3 mdfv3                                      448      290
asammdf 2.6.3 nodata mdfv3                               467      125
mdfreader 0.2.6 mdfv3                                   8992      481
mdfreader 0.2.6 compress mdfv3                          9228      452
mdfreader 0.2.6 compress bcolz 6 mdfv3                  8751      941
asammdf 2.6.3 mdfv4                                      630      334
asammdf 2.6.3 nodata mdfv4                               628      159
mdfreader 0.2.6 mdfv4                                   6880      891
mdfreader 0.2.6 compress mdfv4                          7101      852
mdfreader 0.2.6 compress bcolz6 mdfv4                   6839     1311
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.3 mdfv3                                      779      291
asammdf 2.6.3 nodata mdfv3                             18127      128
mdfreader 0.2.6 mdfv3                                     80      458
mdfreader 0.2.6 noDataLoading mdfv3                 18000000      118
mdfreader 0.2.6 compress mdfv3                           684      187
mdfreader 0.2.6 compress bcolz 6 mdfv3                   298      942
asammdf 2.6.3 mdfv4                                      801      335
asammdf 2.6.3 nodata mdfv4                             25176      157
mdfreader 0.2.6 mdfv4                                     78      870
mdfreader 0.2.6 noDataLoading mdfv4                 18000000      523
mdfreader 0.2.6 compress mdfv4                           686      593
mdfreader 0.2.6 compress bcolz 6 mdfv4                   319     1301
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.3 v3 to v4                                  5884      682
asammdf 2.6.3 v3 to v4 nodata                          27892      479
asammdf 2.6.3 v4 to v3                                  5836      680
asammdf 2.6.3 v4 to v3 nodata                          35283      627
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.3 v3                                       13305     1228
asammdf 2.6.3 v3 nodata                                52775      346
asammdf 2.6.3 v4                                       16069     1267
asammdf 2.6.3 v4 nodata                                70402      364
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
asammdf 2.6.3 mdfv3                                      792      364
asammdf 2.6.3 nodata mdfv3                               568      188
mdfreader 0.2.6 mdfv3                                   2693      545
mdfreader 0.2.6 compress mdfv3                          3855      267
mdfreader 0.2.6 compress bcolz 6 mdfv3                  3865     1040
mdfreader 0.2.6 noDataLoading mdfv3                     1438      199
asammdf 2.6.3 mdfv4                                     1866      435
asammdf 2.6.3 nodata mdfv4                              1480      244
mdfreader 0.2.6 mdfv4                                   5394     1307
mdfreader 0.2.6 compress mdfv4                          6541     1023
mdfreader 0.2.6 compress bcolz 6 mdfv4                  6670     1746
mdfreader 0.2.6 noDataLoading mdfv4                     3940      944
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.3 mdfv3                                      346      365
asammdf 2.6.3 nodata mdfv3                               374      194
mdfreader 0.2.6 mdfv3                                   7861      576
mdfreader 0.2.6 compress mdfv3                          7935      543
mdfreader 0.2.6 compress bcolz 6 mdfv3                  7563     1041
asammdf 2.6.3 mdfv4                                      475      441
asammdf 2.6.3 nodata mdfv4                               443      256
mdfreader 0.2.6 mdfv4                                   5979     1329
mdfreader 0.2.6 compress mdfv4                          6194     1287
mdfreader 0.2.6 compress bcolz6 mdfv4                   5884     1763
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.3 mdfv3                                      590      370
asammdf 2.6.3 nodata mdfv3                              8521      199
mdfreader 0.2.6 mdfv3                                     59      545
mdfreader 0.2.6 noDataLoading mdfv3                 18000000      198
mdfreader 0.2.6 compress mdfv3                           609      270
mdfreader 0.2.6 compress bcolz 6 mdfv3                   252     1042
asammdf 2.6.3 mdfv4                                      627      443
asammdf 2.6.3 nodata mdfv4                             16623      254
mdfreader 0.2.6 mdfv4                                     60     1307
mdfreader 0.2.6 noDataLoading mdfv4                 18000000      943
mdfreader 0.2.6 compress mdfv4                           591     1030
mdfreader 0.2.6 compress bcolz 6 mdfv4                   277     1753
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.3 v3 to v4                                  4674      833
asammdf 2.6.3 v3 to v4 nodata                          20945      578
asammdf 2.6.3 v4 to v3                                  5057      835
asammdf 2.6.3 v4 to v3 nodata                          30132      723
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.6.3 v3                                       10545     1439
asammdf 2.6.3 v3 nodata                                30476      526
asammdf 2.6.3 v4                                       13780     1524
asammdf 2.6.3 v4 nodata                                51810      587
================================================== ========= ========


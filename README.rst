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

* read sorted and unsorted MDF v3 and v4 files
* files are loaded in RAM for fast operations
* handle large files (exceeding the available RAM) using *load_measured_data* = *False* argument
* extract channel data, master channel and extra channel information as *Signal* objects for unified operations with v3 and v4 files
* time domain operation using the *Signal* class

    * Pandas data frames are good if all the channels have the same time based
    * usually a measuremetn will have channels from different sources at different rates
    * the *Signal* class facilitates operations with such channels
    
* remove data group by index or by specifing a channel name inside the target data group
* create new mdf files from scratch
* append new channels
* filter a subset of channels from original mdf file
* cut measurement to specified time interval
* convert to different mdf version
* export to Excel, HDF5, Matlab and CSV
* merge multiple files sharing the same internal structure
* add and extract attachments
* mdf 4.10 zipped blocks
* mdf 4 structure channels

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
asammdf 2.5.4 mdfv3                                      898      289
asammdf 2.5.4 nodata mdfv3                               631      121
mdfreader 0.2.6 mdfv3                                   3431      460
mdfreader 0.2.6 compression mdfv3                       4722      184
mdfreader 0.2.6 compression bcolz 6 mdfv3               4624      940
mdfreader 0.2.6 noDataLoading mdfv3                     1824      120
asammdf 2.5.4 mdfv4                                     1943      333
asammdf 2.5.4 nodata mdfv4                              1547      153
mdfreader 0.2.6 mdfv4                                   6326      881
mdfreader 0.2.6 compression mdfv4                       7354      594
mdfreader 0.2.6 compression bcolz 6 mdfv4               7379     1303
mdfreader 0.2.6 noDataLoading mdfv4                     4343      530
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.4 mdfv3                                      434      293
asammdf 2.5.4 nodata mdfv3                               461      128
mdfreader 0.2.6 mdfv3                                   8901      483
mdfreader 0.2.6 noDataLoading mdfv3                    10331      483
mdfreader 0.2.6 compression mdfv3                       9247      450
mdfreader 0.2.6 compression bcolz 6 mdfv3               8775      941
asammdf 2.5.4 mdfv4                                      687      339
asammdf 2.5.4 nodata mdfv4                               775      162
mdfreader 0.2.6 mdfv4                                   6943      901
mdfreader 0.2.6 noDataLoading mdfv4                     8039      901
mdfreader 0.2.6 compression mdfv4                       7061      860
mdfreader 0.2.6 compression bcolz 6 mdfv4               6811     1320
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.4 mdfv3                                      754      298
asammdf 2.5.4 nodata mdfv3                             18535      134
mdfreader 0.2.6 mdfv3                                     79      460
mdfreader 0.2.6 nodata mdfv3                          108596      333
mdfreader 0.2.6 compression mdfv3                        673      188
mdfreader 0.2.6 compression bcolz 6 mdfv3                298      942
asammdf 2.5.4 mdfv4                                      759      339
asammdf 2.5.4 nodata mdfv4                             20622      163
mdfreader 0.2.6 mdfv4                                     78      880
mdfreader 0.2.6 nodata mdfv4                          155000      752
mdfreader 0.2.6 compression mdfv4                        677      602
mdfreader 0.2.6 compression bcolz 6 mdfv4                322     1310
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.4 v3 to v4                                  5772      693
asammdf 2.5.4 v3 to v4 nodata                          28056      486
asammdf 2.5.4 v4 to v3                                  5828      692
asammdf 2.5.4 v4 to v3 nodata                          32825      630
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.4 v3                                       13135     1220
asammdf 2.5.4 v3 nodata                                52395      336
asammdf 2.5.4 v4                                       15282     1259
asammdf 2.5.4 v4 nodata                                59918      359
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
asammdf 2.5.4 mdfv3                                      744      368
asammdf 2.5.4 nodata mdfv3                               536      192
mdfreader 0.2.6 mdfv3                                   2763      546
mdfreader 0.2.6 compression mdfv3                       4007      267
mdfreader 0.2.6 compression bcolz 6 mdfv3               3897     1039
mdfreader 0.2.6 noDataLoading mdfv3                     1493      197
asammdf 2.5.4 mdfv4                                     1793      439
asammdf 2.5.4 nodata mdfv4                              1317      249
mdfreader 0.2.6 mdfv4                                   5520     1319
mdfreader 0.2.6 compression mdfv4                       7009     1031
mdfreader 0.2.6 compression bcolz 6 mdfv4               7082     1755
mdfreader 0.2.6 noDataLoading mdfv4                     4724      952
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.4 mdfv3                                      459      369
asammdf 2.5.4 nodata mdfv3                               524      200
mdfreader 0.2.6 mdfv3                                   8607      579
mdfreader 0.2.6 noDataLoading mdfv3                     9265      578
mdfreader 0.2.6 compression mdfv3                       8242      542
mdfreader 0.2.6 compression bcolz 6 mdfv3               7787     1039
asammdf 2.5.4 mdfv4                                      572      446
asammdf 2.5.4 nodata mdfv4                               512      260
mdfreader 0.2.6 mdfv4                                   6248     1341
mdfreader 0.2.6 noDataLoading mdfv4                     7095     1340
mdfreader 0.2.6 compression mdfv4                       6455     1296
mdfreader 0.2.6 compression bcolz 6 mdfv4               6067     1771
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.4 mdfv3                                      605      379
asammdf 2.5.4 nodata mdfv3                              9065      209
mdfreader 0.2.6 mdfv3                                     66      546
mdfreader 0.2.6 nodata mdfv3                           80570      418
mdfreader 0.2.6 compression mdfv3                        628      270
mdfreader 0.2.6 compression bcolz 6 mdfv3                273     1040
asammdf 2.5.4 mdfv4                                      611      448
asammdf 2.5.4 nodata mdfv4                             12484      262
mdfreader 0.2.6 mdfv4                                     64     1319
mdfreader 0.2.6 nodata mdfv4                          117087     1189
mdfreader 0.2.6 compression mdfv4                        637     1041
mdfreader 0.2.6 compression bcolz 6 mdfv4                301     1762
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.4 v3 to v4                                  4640      849
asammdf 2.5.4 v3 to v4 nodata                          21774      589
asammdf 2.5.4 v4 to v3                                  4842      854
asammdf 2.5.4 v4 to v3 nodata                          26222      728
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.4 v3                                       10062     1408
asammdf 2.5.4 v3 nodata                                30880      497
asammdf 2.5.4 v4                                       13109     1503
asammdf 2.5.4 v4 nodata                                41532      565
================================================== ========= ========

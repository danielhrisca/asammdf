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
* export to Excel, HDF5 and CSV
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
    * mdf 4 channel arrays
    * xml schema for TXBLOCK and MDBLOCK

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
asammdf 2.5.2 mdfv3                                      951      289
asammdf 2.5.2 nodata mdfv3                               667      119
mdfreader 0.2.5 mdfv3                                   3628      454
asammdf 2.5.2 mdfv4                                     2142      343
asammdf 2.5.2 nodata mdfv4                              1585      161
mdfreader 0.2.5 mdfv4                                  42997      577
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 mdfv3                                      530      293
asammdf 2.5.2 nodata mdfv3                               457      126
mdfreader 0.2.5 mdfv3                                  19931     1225
asammdf 2.5.2 mdfv4                                      705      354
asammdf 2.5.2 nodata mdfv4                               707      175
mdfreader 0.2.5 mdfv4                                  17803     1686
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 mdfv3                                      865      298
asammdf 2.5.2 nodata mdfv3                             18442      132
mdfreader 0.2.5 mdfv3                                     36      454
asammdf 2.5.2 mdfv4                                      803      349
asammdf 2.5.2 nodata mdfv4                             20397      171
mdfreader 0.2.5 mdfv4                                     48      577
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 v3 to v4                                  5824      690
asammdf 2.5.2 v3 to v4 nodata                          28060      486
asammdf 2.5.2 v4 to v3                                  5591      710
asammdf 2.5.2 v4 to v3 nodata                          30169      638
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 v3                                       13152     1226
asammdf 2.5.2 v3 nodata                                53144      348
asammdf 2.5.2 v4                                       15031     1296
asammdf 2.5.2 v4 nodata                                60374      389
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
asammdf 2.5.2 mdfv3                                      832      372
asammdf 2.5.2 nodata mdfv3                               615      191
mdfreader 0.2.5 mdfv3                                   2876      537
asammdf 2.5.2 mdfv4                                     1690      455
asammdf 2.5.2 nodata mdfv4                              1289      260
mdfreader 0.2.5 mdfv4                                  30755      748
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 mdfv3                                      393      373
asammdf 2.5.2 nodata mdfv3                               452      198
mdfreader 0.2.5 mdfv3                                  21409     1997
asammdf 2.5.2 mdfv4                                      552      465
asammdf 2.5.2 nodata mdfv4                               563      275
mdfreader 0.2.5 mdfv4                                  18899     2795
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 mdfv3                                      633      381
asammdf 2.5.2 nodata mdfv3                              8703      207
mdfreader 0.2.5 mdfv3                                     29      537
asammdf 2.5.2 mdfv4                                      633      464
asammdf 2.5.2 nodata mdfv4                             12241      274
mdfreader 0.2.5 mdfv4                                     40      748
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 v3 to v4                                  4504      856
asammdf 2.5.2 v3 to v4 nodata                          21983      592
asammdf 2.5.2 v4 to v3                                  4731      883
asammdf 2.5.2 v4 to v3 nodata                          26840      740
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.2 v3                                       10572     1428
asammdf 2.5.2 v3 nodata                                31014      519
asammdf 2.5.2 v4                                       12852     1558
asammdf 2.5.2 v4 nodata                                41054      610
================================================== ========= ========
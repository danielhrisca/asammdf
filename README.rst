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
asammdf 2.5.3 mdfv3                                      897      281
asammdf 2.5.3 nodata mdfv3                               648      112
mdfreader 0.2.5 mdfv3                                   3836      454
asammdf 2.5.3 mdfv4                                     2098      331
asammdf 2.5.3 nodata mdfv4                              1588      151
mdfreader 0.2.5 mdfv4                                  45415      577
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.3 mdfv3                                      469      285
asammdf 2.5.3 nodata mdfv3                               526      119
mdfreader 0.2.5 mdfv3                                  20328     1224
asammdf 2.5.3 mdfv4                                      752      337
asammdf 2.5.3 nodata mdfv4                               751      160
mdfreader 0.2.5 mdfv4                                  18135     1686
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.3 mdfv3                                      846      289
asammdf 2.5.3 nodata mdfv3                             19460      126
mdfreader 0.2.5 mdfv3                                     37      454
asammdf 2.5.3 mdfv4                                      809      337
asammdf 2.5.3 nodata mdfv4                             20778      161
mdfreader 0.2.5 mdfv4                                     49      577
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.3 v3 to v4                                  6121      673
asammdf 2.5.3 v3 to v4 nodata                          29340      476
asammdf 2.5.3 v4 to v3                                  5645      690
asammdf 2.5.3 v4 to v3 nodata                          32115      628
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.3 v3                                       13392     1201
asammdf 2.5.3 v3 nodata                                54040      327
asammdf 2.5.3 v4                                       15031     1265
asammdf 2.5.3 v4 nodata                                60397      364
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
asammdf 2.5.3 mdfv3                                      876      357
asammdf 2.5.3 nodata mdfv3                               636      181
mdfreader 0.2.5 mdfv3                                   3295      537
asammdf 2.5.3 mdfv4                                     1889      436
asammdf 2.5.3 nodata mdfv4                              1498      245
mdfreader 0.2.5 mdfv4                                  34732      748
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.3 mdfv3                                      486      359
asammdf 2.5.3 nodata mdfv3                               538      188
mdfreader 0.2.5 mdfv3                                  25780     1996
asammdf 2.5.3 mdfv4                                      628      442
asammdf 2.5.3 nodata mdfv4                               579      257
mdfreader 0.2.5 mdfv4                                  21399     2795
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.3 mdfv3                                      732      367
asammdf 2.5.3 nodata mdfv3                             10205      198
mdfreader 0.2.5 mdfv3                                     35      537
asammdf 2.5.3 mdfv4                                      688      445
asammdf 2.5.3 nodata mdfv4                             14187      258
mdfreader 0.2.5 mdfv4                                     45      748
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.3 v3 to v4                                  5056      828
asammdf 2.5.3 v3 to v4 nodata                          24569      576
asammdf 2.5.3 v4 to v3                                  5300      851
asammdf 2.5.3 v4 to v3 nodata                          29128      725
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.5.3 v3                                       11408     1387
asammdf 2.5.3 v3 nodata                                35575      487
asammdf 2.5.3 v4                                       14531     1507
asammdf 2.5.3 v4 nodata                                44399      568
================================================== ========= ========
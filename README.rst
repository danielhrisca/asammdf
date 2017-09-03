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

    * for low memory computers or for large data files there is the option to load only the metadata and leave the raw channel data (the samples) unread; this of course will mean slower channel data access speed

* extract channel data, master channel and extra channel information as *Signal* objects for unified operations with v3 and v4 files
* time domain operation using the *Signal* class

    * Pandas data frames are good if all the channels have the same time based
    * usually a measuremetn will have channels from different sources at different rates
    * the *Signal* class facilitates operations with such channels
    
* remove data group by index or by specifing a channel name inside the target data group
* create new mdf files from scratch
* append new channels
* filter a subset of channels from original mdf file
* convert to different mdf version
* export to Excel, HDF5 and CSV
* add and extract attachments
* mdf 4.10 zipped blocks
* mdf 4 structure channels

Major features still not implemented
====================================

* for version 3

    * functionality related to sample reduction block (but the class is defined)
    
* for version 4

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

* nodata = MDF object created with load_measured_data=False (raw channel data not loaded into RAM)
* compression = MDF object created with compression=blosc
* compression bcolz 6 = MDF object created with compression=6
* noDataLoading = MDF object read with noDataLoading=True

Files used for benchmark:
* 183 groups
* 36424 channels


================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.4.3 mdfv3                                      927      289
asammdf 2.4.3 nodata mdfv3                               647      118
mdfreader 0.2.5 mdfv3                                   3583      455
asammdf 2.4.3 mdfv4                                     1956      343
asammdf 2.4.3 nodata mdfv4                              1509      161
mdfreader 0.2.5 mdfv4                                  41613      578
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.4.3 mdfv3                                      415      292
asammdf 2.4.3 nodata mdfv3                               437      126
mdfreader 0.2.5 mdfv3                                  19103     1225
asammdf 2.4.3 mdfv4                                      667      351
asammdf 2.4.3 nodata mdfv4                               714      169
mdfreader 0.2.5 mdfv4                                  16612     1687
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.4.3 mdfv3                                      935      297
asammdf 2.4.3 nodata mdfv3                             18635      131
mdfreader 0.2.5 mdfv3                                     34      455
asammdf 2.4.3 mdfv4                                      827      349
asammdf 2.4.3 nodata mdfv4                             20404      170
mdfreader 0.2.5 mdfv4                                     46      578
================================================== ========= ========


Python 3 x64
------------

Benchmark environment

* 3.6.2 (v3.6.2:5fd33b5, Jul  8 2017, 04:57:36) [MSC v.1900 64 bit (AMD64)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results

* nodata = MDF object created with load_measured_data=False (raw channel data not loaded into RAM)
* compression = MDF object created with compression=blosc
* compression bcolz 6 = MDF object created with compression=6
* noDataLoading = MDF object read with noDataLoading=True

Files used for benchmark:
* 183 groups
* 36424 channels


================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.4.3 mdfv3                                      768      372
asammdf 2.4.3 nodata mdfv3                               597      190
mdfreader 0.2.5 mdfv3                                   2742      536
asammdf 2.4.3 mdfv4                                     1655      455
asammdf 2.4.3 nodata mdfv4                              1292      260
mdfreader 0.2.5 mdfv4                                  29513      748
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.4.3 mdfv3                                      384      373
asammdf 2.4.3 nodata mdfv3                               379      196
mdfreader 0.2.5 mdfv3                                  20888     1996
asammdf 2.4.3 mdfv4                                      518      462
asammdf 2.4.3 nodata mdfv4                               502      272
mdfreader 0.2.5 mdfv4                                  18299     2795
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.4.3 mdfv3                                      662      381
asammdf 2.4.3 nodata mdfv3                              8735      208
mdfreader 0.2.5 mdfv3                                     27      536
asammdf 2.4.3 mdfv4                                      609      464
asammdf 2.4.3 nodata mdfv4                             12104      273
mdfreader 0.2.5 mdfv4                                     38      748
================================================== ========= ========
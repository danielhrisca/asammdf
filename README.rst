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
* add and extract attachments
* mdf 4.10 zipped blocks
* mdf 4 structure channels

Major features still not implemented
====================================

* functionality related to sample reduction block (but the class is defined)
* mdf 3 channel dependency save and append (only reading is implemented)
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
http://asammdf.readthedocs.io/en/stable

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
* pandas : for DataFrame export

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
* compression = MDF object created with compression=True/blosc
* compression bcolz 6 = MDF object created with compression=6
* noDataLoading = MDF object read with noDataLoading=True

Files used for benchmark:
* 183 groups
* 36424 channels


================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.3.0 mdfv3                                     1126      294
asammdf 2.3.0 nodata mdfv3                               917      123
mdfreader 0.2.5 mdfv3                                   3743      455
asammdf 2.3.0 mdfv4                                     2359      348
asammdf 2.3.0 nodata mdfv4                              1906      166
mdfreader 0.2.5 mdfv4                                  43166      577
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.3.0 mdfv3                                      420      297
asammdf 2.3.0 nodata mdfv3                               445      130
mdfreader 0.2.5 mdfv3                                  20078     1224
asammdf 2.3.0 mdfv4                                      711      357
asammdf 2.3.0 nodata mdfv4                               738      175
mdfreader 0.2.5 mdfv4                                  17822     1687
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.3.0 mdfv3                                      777      299
asammdf 2.3.0 nodata mdfv3                             18662      132
mdfreader 0.2.5 mdfv3                                     36      455
asammdf 2.3.0 mdfv4                                      681      354
asammdf 2.3.0 nodata mdfv4                             20439      176
mdfreader 0.2.5 mdfv4                                     51      578
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
asammdf 2.3.0 mdfv3                                     1011      379
asammdf 2.3.0 nodata mdfv3                               725      198
mdfreader 0.2.5 mdfv3                                   2973      537
asammdf 2.3.0 mdfv4                                     1890      464
asammdf 2.3.0 nodata mdfv4                              1542      268
mdfreader 0.2.5 mdfv4                                  32192      748
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.3.0 mdfv3                                      359      379
asammdf 2.3.0 nodata mdfv3                               352      205
mdfreader 0.2.5 mdfv3                                  21777     1997
asammdf 2.3.0 mdfv4                                      525      471
asammdf 2.3.0 nodata mdfv4                               542      280
mdfreader 0.2.5 mdfv4                                  19591     2795
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.3.0 mdfv3                                      589      383
asammdf 2.3.0 nodata mdfv3                              8841      209
mdfreader 0.2.5 mdfv3                                     28      537
asammdf 2.3.0 mdfv4                                      494      468
asammdf 2.3.0 nodata mdfv4                             12330      280
mdfreader 0.2.5 mdfv4                                     39      748
================================================== ========= ========

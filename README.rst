*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports both MDF version 3 and 4 formats. 

*asammdf* works on Python 2.7, and Python >= 3.4

Project goals
=============
The main goals for this library are:

* to be faster than the other Python based mdf libraries
* to have clean and eays to understand code base

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
* append new channels
* convert to different mdf version
* add and extract attachments
* mdf 4.10 zipped blocks

Major features still not implemented
====================================

* functionality related to sample reduction block (but the class is defined)
* mdf 3 channel dependency functionality
* functionality related to trigger blocks (but the class is defined)
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
http://asammdf.readthedocs.io/en/2.1.0/

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
* numexpr : for formula based channel conversions
* blosc : optionally used for in memmory raw channel data compression
* matplotlib : for Signal plotting

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

* nodata = MDF object created with load_measured_data=False (raw channel data no loaded into RAM)
* compression = MDF object created with compression=True (raw channel data loaded into RAM and compressed)
* noconvert = MDF object created with convertAfterRead=False

Files used for benchmark:
* 183 groups
* 36424 channels


================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.1.0 mdfv3                                     1031      284
asammdf 2.1.0 compression mdfv3                         1259      192
asammdf 2.1.0 nodata mdfv3                               584      114
mdfreader 0.2.5 mdfv3                                   3809      455
mdfreader 0.2.5 no convert mdfv3                        3498      321
asammdf 2.1.0 mdfv4                                     2109      341
asammdf 2.1.0 compression mdfv4                         2405      239
asammdf 2.1.0 nodata mdfv4                              1686      159
mdfreader 0.2.5 mdfv4                                  44400      578
mdfreader 0.2.5 noconvert mdfv4                        43867      449
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.1.0 mdfv3                                      713      286
asammdf 2.1.0 compression mdfv3                          926      194
mdfreader 0.2.5 mdfv3                                  19862     1226
asammdf 2.1.0 mdfv4                                     1109      347
asammdf 2.1.0 compression mdfv4                         1267      246
mdfreader 0.2.5 mdfv4                                  17518     1656
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.1.0 mdfv3                                     3943      295
asammdf 2.1.0 compression mdfv3                        29682      203
asammdf 2.1.0 nodata mdfv3                             23215      129
mdfreader 0.2.5 mdfv3                                     38      455
asammdf 2.1.0 mdfv4                                     3227      351
asammdf 2.1.0 compression mdfv4                        26070      250
asammdf 2.1.0 nodata mdfv4                             21619      171
mdfreader 0.2.5 mdfv4                                     51      578
================================================== ========= ========


Python 3 x64
------------

Benchmark environment

* 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 18:41:36) [MSC v.1900 64 bit (AMD64)]
* Windows-10-10.0.14393-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

Notations used in the results

* nodata = MDF object created with load_measured_data=False (raw channel data no loaded into RAM)
* compression = MDF object created with compression=True (raw channel data loaded into RAM and compressed)
* noconvert = MDF object created with convertAfterRead=False

Files used for benchmark:
* 183 groups
* 36424 channels


================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.1.0 mdfv3                                      801      352
asammdf 2.1.0 compression mdfv3                          946      278
asammdf 2.1.0 nodata mdfv3                               490      172
mdfreader 0.2.5 mdfv3                                   2962      525
mdfreader 0.2.5 no convert mdfv3                        2740      392
asammdf 2.1.0 mdfv4                                     1674      440
asammdf 2.1.0 compression mdfv4                         1916      343
asammdf 2.1.0 nodata mdfv4                              1360      245
mdfreader 0.2.5 mdfv4                                  31915      737
mdfreader 0.2.5 noconvert mdfv4                        31425      607
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.1.0 mdfv3                                      575      353
asammdf 2.1.0 compression mdfv3                          705      276
mdfreader 0.2.5 mdfv3                                  21591     1985
asammdf 2.1.0 mdfv4                                      913      447
asammdf 2.1.0 compression mdfv4                         1160      352
mdfreader 0.2.5 mdfv4                                  18666     2782
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 2.1.0 mdfv3                                     2835      363
asammdf 2.1.0 compression mdfv3                        18188      287
asammdf 2.1.0 nodata mdfv3                             11926      188
mdfreader 0.2.5 mdfv3                                     29      525
asammdf 2.1.0 mdfv4                                     2338      450
asammdf 2.1.0 compression mdfv4                        15566      355
asammdf 2.1.0 nodata mdfv4                             12598      260
mdfreader 0.2.5 mdfv4                                     39      737
================================================== ========= ========

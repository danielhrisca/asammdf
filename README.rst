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

Major features still not implemented
====================================

* functionality related to sample reduction block (but the class is defined)
* mdf 3 channel dependency functionality
* functionality related to trigger blocks (but the class is defined)
* handling of unfinnished measurements (mdf 4)
* compressed data blocks for mdf >= 4.10
* mdf 4 attachment blocks
* mdf 4 channel arrays
* mdf 4 VLSD channels and SDBLOCKs
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
http://asammdf.readthedocs.io/en/2.0.0/

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

3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 18:41:36) [MSC v.1900 64 bit (AMD64)]

Windows-7-6.1.7601-SP1

Intel64 Family 6 Model 94 Stepping 3, GenuineIntel

16 installed RAM


* nodata = MDF object created with load_measured_data=False (raw channel data no loaded into RAM)
* compression = MDF object created with compression=True (raw channel data loaded into RAM and compressed)
* noconvert = MDF object created with convertAfterRead=False

Files used for benchmark:

* 183 groups
* 36424 channels

========================================          =========       ========
Open file                                         Time [ms]       RAM [MB]
========================================          =========       ========
asammdf 2.0.0 mdfv3                                     721            352
asammdf 2.0.0 compression mdfv3                        1008            275
asammdf 2.0.0 nodata mdfv3                              641            199
mdfreader 0.2.5 mdfv3                                  2996            526
mdfreader 0.2.5 no convert mdfv3                       2846            393
asammdf 2.0.0 mdfv4                                    1634            439
asammdf 2.0.0 compression mdfv4                        1917            343
asammdf 2.0.0 nodata mdfv4                             1594            274
mdfreader 0.2.5 mdfv4                                 31023            739
mdfreader 0.2.5 noconvert mdfv4                       30693            609
========================================          =========       ========


========================================          =========       ========
Save file                                         Time [ms]       RAM [MB]
========================================          =========       ========
asammdf 2.0.0 mdfv3                                     472            353
asammdf 2.0.0 compression mdfv3                         667            275
mdfreader 0.2.5 mdfv3                                 18910           2003
asammdf 2.0.0 mdfv4                                     686            447
asammdf 2.0.0 compression mdfv4                         836            352
mdfreader 0.2.5 mdfv4                                 16631           2802
========================================          =========       ========


========================================          =========       ========
Get all channels                                  Time [ms]       RAM [MB]
========================================          =========       ========
asammdf 2.0.0 mdfv3                                    2492            362
asammdf 2.0.0 compression mdfv3                       14474            285
asammdf 2.0.0 nodata mdfv3                             9621            215
mdfreader 0.2.5 mdfv3                                    31            526
asammdf 2.0.0 mdfv4                                    2066            450
asammdf 2.0.0 compression mdfv4                       16944            359
asammdf 2.0.0 nodata mdfv4                            12364            292
mdfreader 0.2.5 mdfv4                                    39            739
========================================          =========       ========

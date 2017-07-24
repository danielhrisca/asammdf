*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports both MDF version 3 and 4 formats. 

Project goals
=============
The main goals for this library are:

* to be faster than the other Python based mdf libraries
* clean and simple data types

Dependencies
============
asammdf uses the following libraries

* numpy : the heart that makes all tick
* numexpr : for formula based channel conversions
* blosc : optionally used for in memmory raw channel data compression
* matplotlib : for Signal plotting

Usage
=====

```python

   from asammdf import MDF3
   mdf = MDF3('sample.mdf')
   speed = mdf.get_signal_by_name('WheelSpeed')
   
```

Check the *examples* folder for extended usage demo.

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

Documentation
=============
http://asammdf.readthedocs.io/en/stable

Installation
============
*asammdf* is available on 

    * github: https://github.com/danielhrisca/asammdf/
    * PyPI: https://pypi.org/project/asammdf/
    
.. code-block:: python

    pip install asammdf

Benchmarks
==========
using a more complex file of 170MB with 180 data groups and 36000 channels with Python 3.6.1 32bit 

    * file load:

        * asammdf 1.1.0 : 950ms
        * asammdf 1.1.0 with compression : 1600s
        * asammdf 1.1.0 without loading raw channel data: 750ms
        * mdfreader 0.2.4 : 3600ms
        * mdfreader 0.2.4 without channel conversion : 3330ms

    * file save:

        * asammdf 1.1.0 : 722ms
        * mdfreader 0.2.4 : 18800ms

    * get channel data (10000 calls):

        * asammdf 1.1.0 : 918ms
        * mdfreader 0.2.4 : 11ms

    * RAM usage:

        * asammdf 1.1.0 : 345MB
        * asammdf 1.1.0 with compression : 280MB
        * asammdf 1.1.0 without loading raw channel data: 150MB
        * mdfreader 0.2.4 : 480MB
        * mdfreader 0.2.4 without channel conversion: 365MB

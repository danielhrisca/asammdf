*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. Currently only version 3 is supported.

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

Usage
=====

.. code-block:: python

    from asammdf import MDF3
    mdf = MDF3('sample.mdf')
    speed = mdf.get_signal_by_name('WheelSpeed')

Features
========

* read sorted and unsorted MDF v3 files
* files are loaded in RAM for fast operations
    * for low memory computers or for large data files there is the option to load only the metadata and leave the raw channel data (the samples) unread; this of course will mean slower channel data access speed
* extract channel data, master channel and extra channel information (unit, conversion rule)
* remove channel by name
* remove data group by specifing a channel name inside the target data group
* append new channels

Benchmarks
==========
using a more complex file of 170MB with 180 data groups and 36000 channels with Python 3.6.1 32bit 

* file load:
    * asammdf 1.0.0 : 1040ms
    * mdfreader 0.2.4 : 3986ms
* file save:
    * asammdf 1.0.0 : 722ms
    * mdfreader 0.2.4 : 18800ms
* get channel data (10000 calls):
    * asammdf 1.0.0 : 918ms
    * mdfreader 0.2.4 : 11ms
* RAM usage with loaded raw channel data:
    * asammdf 1.0.0 : 280MB
    * mdfreader 0.2.4 : 441MB
* RAM usage without loaded raw channel data:
    * asammdf 1.0.0 : 118MB
    * mdfreader 0.2.4 : 300MB

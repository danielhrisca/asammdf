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
* numexpr : for fromula based channel conversions
* blosc : optionally used for in memmory raw channel data compression

Usage
=====

.. code-block:: python

    from asammdf import MDF3
    mdf = MDF3('sample.mdf')
    speed = mdf.get_signal_by_name('WheelSpeed')


Benchmarks
==========
using a more complex file of 170MB with 180 data groups and 36000 channels the file 

* file load:
    * asammdf 1.0.0 : 1040ms
    * mdfreader 0.2.2 : 3986ms
        
* file save:
    * asammdf 1.0.0 : 722ms
    * mdfreader 0.2.2 : 18800ms
        
* get channel data (10000 calls):
    * asammdf 1.0.0 : 918ms
    * mdfreader 0.2.2 : 11ms

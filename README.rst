*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports both MDF version 3 and 4 formats. 

Project goals
=============
The main goals for this library are:

* to be faster than the other Python based mdf libraries
* clean and simple data types

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

.. code-block:: python

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
    
.. code-block:: python

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
![Using Python 3.6.1 x64](benchmarks/asam 2.0.0 vs reader 0.2.5 Pyhton3.6.1x64 SSD i7-6820.txt)

![Open MDF file](https://raw.githubusercontent.com/danielhrisca/asammdf/master/benchmarks/open.png)

using a more complex file of 170MB with 180 data groups and 36000 channels with Python 3.6.1 64bit 

    * mdf version 3
    
        * file load:

            * asammdf 2.0.0 : 800ms
            * asammdf 2.0.0 with compression : 1050s
            * asammdf 2.0.0 without loading raw channel data: 600ms
            * mdfreader 0.2.5 : 3200ms
            * mdfreader 0.2.5 without channel conversion : 2850ms

        * file save:

            * asammdf 2.0.0 : 520ms
            * asammdf 2.0.0 with compression : 610s
            * mdfreader 0.2.5 : 19600ms

        * get channel data (10000 calls):

            * asammdf 2.0.0 : 918ms
            * mdfreader 0.2.5 : 11ms

        * RAM usage:

            * asammdf 2.0.0 : 334MB
            * asammdf 2.0.0 with compression : 262MB
            * asammdf 2.0.0 without loading raw channel data: 76MB
            * mdfreader 0.2.5 : 510MB
            * mdfreader 0.2.5 without channel conversion: 887MB
            
    * mdf version 4
    
        * file load:

            * asammdf 2.0.0 : 2280ms
            * asammdf 2.0.0 with compression : 3130s
            * asammdf 2.0.0 without loading raw channel data: 2540ms
            * mdfreader 0.2.5 : 30426ms
            * mdfreader 0.2.5 without channel conversion : 30000ms

        * file save:

            * asammdf 2.0.0 : 980ms
            * asammdf 2.0.0 with compression : 1150s
            * mdfreader 0.2.5 : 17100ms

        * get channel data (10000 calls):

            * asammdf 2.0.0 : 918ms
            * mdfreader 0.2.5 : 11ms

        * RAM usage:

            * asammdf 2.0.0 : 1123MB
            * asammdf 2.0.0 with compression : 480MB
            * asammdf 2.0.0 without loading raw channel data: 455MB
            * mdfreader 0.2.5 : 577MB
            * mdfreader 0.2.5 without channel conversion: 2891MB

[![Build Status](https://travis-ci.org/danielhrisca/asammdf.svg?branch=development)](https://travis-ci.org/danielhrisca/asammdf)
[![PyPI version](https://badge.fury.io/py/asammdf.svg)](https://badge.fury.io/py/asammdf)
[![Documentation Status](http://readthedocs.org/projects/asammdf/badge/?version=development)](http://asammdf.readthedocs.io/en/development/?badge=stable)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/a3da21da90ca43a5b72fc24b56880c99?branch=development)](https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=badger) 
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/a3da21da90ca43a5b72fc24b56880c99?branch=development)](https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=Badge_Coverage)


*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports MDF versions 2 (.dat), 3 (.mdf) and 4 (.mf4). 

*asammdf* works on Python 2.7, and Python >= 3.4 (Travis CI tests done with Python 2.7 and Python >= 3.5)


Project goals
=============
The main goals for this library are:

* to be faster than the other Python based mdf libraries
* to have clean and easy to understand code base
* to have minimal 3-rd party dependencies

Features
========

* create new mdf files from scratch
* append new channels
* read unsorted MDF v3 and v4 files
* filter a subset of channels from original mdf file
* cut measurement to specified time interval
* convert to different mdf version
* export to Excel, HDF5, Matlab and CSV
* merge multiple files sharing the same internal structure
* read and save mdf version 4.10 files containing zipped data blocks
* split large data blocks (configurable size) for mdf version 4
* disk space savings by compacting 1-dimensional integer channels (configurable)
* full support (read, append, save) for the following map types (multidimensional array channels):

    * mdf version 3 channels with CDBLOCK
    * mdf version 4 structure channel composition
    * mdf version 4 channel arrays with CNTemplate storage and one of the array types:
    
        * 0 - array
        * 1 - scaling axis
        * 2 - look-up
        
* add and extract attachments for mdf version 4
* files are loaded in RAM for fast operations
* handle large files (exceeding the available RAM) using *memory* = *minimum* argument
* extract channel data, master channel and extra channel information as *Signal* objects for unified operations with v3 and v4 files
* time domain operation using the *Signal* class

    * Pandas data frames are good if all the channels have the same time based
    * usually a measurement will have channels from different sources at different rates
    * the *Signal* class facilitates operations with such channels

Major features not implemented (yet)
====================================
* for version 3

    * functionality related to sample reduction block (but the class is defined)
    
* for version 4

    * handling of bus logging measurements
    * handling of unfinished measurements (mdf 4)
    * xml schema for TXBLOCK and MDBLOCK
    * full support for remaining mdf 4 channel arrays types
    * partial conversions
    * event blocks
    * channels with default X axis
    * channels with reference to attachment

Usage
=====

```python
   from asammdf import MDF
   
   mdf = MDF('sample.mdf')
   speed = mdf.get('WheelSpeed')
   speed.plot()
   
   important_signals = ['WheelSpeed', 'VehicleSpeed', 'VehicleAcceleration']
   # get short measurement with a subset of channels from 10s to 12s 
   short = mdf.filter(important_signals).cut(start=10, stop=12)
   
   # convert to version 4.10 and save to disk
   short.convert('4.10').save('important signals.mf4')
   
   # plot some channels from a huge file
   efficient = MDF('huge.mf4', , memory='minimum')
   for signal in efficient.select(['Sensor1', 'Voltage3']):
       signal.plot()
   
```  
 
Check the *examples* folder for extended usage demo, or the documentation
http://asammdf.readthedocs.io/en/development/examples.html

Documentation
=============
http://asammdf.readthedocs.io/en/development

Installation
============
*asammdf* is available on 

* github: https://github.com/danielhrisca/asammdf/
* PyPI: https://pypi.org/project/asammdf/
    
```
   pip install asammdf
```
    
Dependencies
============
asammdf uses the following libraries

* numpy : the heart that makes all tick
* numexpr : for algebraic and rational channel conversions
* matplotlib : for Signal plotting
* wheel : for installation in virtual environments
* pandas : for DataFrame export

optional dependencies needed for exports

* h5py : for HDF5 export
* xlsxwriter : for Excel export
* scipy : for Matlab .mat export

Benchmarks
==========

http://asammdf.readthedocs.io/en/development/benchmarks.html


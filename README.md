
<img align=left src="https://raw.githubusercontent.com/danielhrisca/asammdf/master/asammdf.png" width="192" height="192" /> 

<p align=center>
   
*asammdf* is a fast parser and editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports MDF versions 2 (.dat), 3 (.mdf) and 4 (.mf4). 

*asammdf* works on Python 2.7, and Python >= 3.4

*asammdf* was tested succesfully on both Linux and Windows

_

</p>

# Status

! | Travis CI  | Coverage  |  Codacy  | ReadTheDocs 
--|--|--|--|--
master | [![Build Status](https://travis-ci.org/danielhrisca/asammdf.svg?branch=master)](https://travis-ci.org/danielhrisca/asammdf) | [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/a3da21da90ca43a5b72fc24b56880c99?branch=master)](https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=Badge_Coverage) | [![Codacy Badge](https://api.codacy.com/project/badge/Grade/a3da21da90ca43a5b72fc24b56880c99?branch=master)](https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=badger) |  [![Documentation Status](http://readthedocs.org/projects/asammdf/badge/?version=master)](http://asammdf.readthedocs.io/en/master/?badge=stable) |  
development| [![Build Status](https://travis-ci.org/danielhrisca/asammdf.svg?branch=development)](https://travis-ci.org/danielhrisca/asammdf) | [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/a3da21da90ca43a5b72fc24b56880c99?branch=development)](https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=Badge_Coverage) | [![Codacy Badge](https://api.codacy.com/project/badge/Grade/a3da21da90ca43a5b72fc24b56880c99?branch=development)](https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=badger) | [![Documentation Status](http://readthedocs.org/projects/asammdf/badge/?version=development)](http://asammdf.readthedocs.io/en/master/?badge=stable) |   

PyPI| conda-forge  |  anaconda-cloud 
--|--|--
[![PyPI version](https://badge.fury.io/py/asammdf.svg)](https://badge.fury.io/py/asammdf)  | [![conda-forge version](https://anaconda.org/conda-forge/asammdf/badges/version.svg)](https://anaconda.org/conda-forge/asammdf) | [![anaconda-cloud version](https://anaconda.org/daniel.hrisca/asammdf/badges/version.svg)](https://anaconda.org/daniel.hrisca/asammdf)


# Project goals
The main goals for this library are:

* to be faster than the other Python based mdf libraries
* to have clean and easy to understand code base
* to have minimal 3-rd party dependencies

# Features

* create new mdf files from scratch
* append new channels
* read unsorted MDF v3 and v4 files
* read CAN bus logging files
* filter a subset of channels from original mdf file
* cut measurement to specified time interval
* convert to different mdf version
* export to Excel, HDF5, Matlab and CSV
* merge multiple files sharing the same internal structure
* read and save mdf version 4.10 files containing zipped data blocks
* space optimizations for saved files (no duplicated blocks)
* split large data blocks (configurable size) for mdf version 4
* full support (read, append, save) for the following map types (multidimensional array channels):

    * mdf version 3 channels with CDBLOCK
    * mdf version 4 structure channel composition
    * mdf version 4 channel arrays with CNTemplate storage and one of the array types:
    
        * 0 - array
        * 1 - scaling axis
        * 2 - look-up
        
* add and extract attachments for mdf version 4
* handle large files (for example merging two fileas, each with 14000 channels and 5GB size, on a RaspberryPi) using *memory* = *minimum* argument
* extract channel data, master channel and extra channel information as *Signal* objects for unified operations with v3 and v4 files
* time domain operation using the *Signal* class

    * Pandas data frames are good if all the channels have the same time based
    * a measurement will usually have channels from different sources at different rates
    * the *Signal* class facilitates operations with such channels

# Major features not implemented (yet)

* for version 3

    * functionality related to sample reduction block
    
* for version 4

    * functionality related to sample reduction block
    * handling of channel hierarchy
    * full handling of bus logging measurements
    * handling of unfinished measurements (mdf 4)
    * full support for remaining mdf 4 channel arrays types
    * xml schema for MDBLOCK
    * full handling of event blocks
    * channels with default X axis
    * chanenls with reference to attachment

# Usage

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
http://asammdf.readthedocs.io/en/master/examples.html

# Documentation
http://asammdf.readthedocs.io/en/master

# Contributing
Please have a look over the [contributing guidelines](https://github.com/danielhrisca/asammdf/blob/master/CONTRIBUTING.md)

## Contributors
Thanks to all who contributed with commits to *asammdf*:
* Julien Grave [JulienGrv](https://github.com/JulienGrv)
* Jed Frey [jed-frey](https://github.com/jed-frey)
* Mihai [yahym](https://github.com/yahym)
* Jack Weinstein [jacklev](https://github.com/jacklev)
* Isuru Fernando [isuruf](https://github.com/isuruf)
* Felix Kohlgrüber [fkohlgrueber](https://github.com/fkohlgrueber)

# Installation
*asammdf* is available on 

* github: https://github.com/danielhrisca/asammdf/
* PyPI: https://pypi.org/project/asammdf/
* conda-forge: https://anaconda.org/conda-forge/asammdf
    
```
   pip install asammdf
   # or for anaconda
   conda install -c conda-forge asammdf 
```
    
# Dependencies
asammdf uses the following libraries

* numpy : the heart that makes all tick
* numexpr : for algebraic and rational channel conversions
* matplotlib : for Signal plotting
* wheel : for installation in virtual environments
* pandas : for DataFrame export
* canmatrix : to handle CAN bus logging measurements

optional dependencies needed for exports

* h5py : for HDF5 export
* xlsxwriter : for Excel export
* scipy : for Matlab .mat export

other optional dependencies

* chardet : to detect non-standard unicode encodings

# Benchmarks

http://asammdf.readthedocs.io/en/master/benchmarks.html


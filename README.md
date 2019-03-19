
<img align=left src="https://raw.githubusercontent.com/danielhrisca/asammdf/master/asammdf.png" width="192" height="192" />

<p align=center>

*asammdf* is a fast parser and editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files.

*asammdf* supports MDF versions 2 (.dat), 3 (.mdf) and 4 (.mf4).

*asammdf* works on Python >= 3.6 (for Python 2.7, 3.4 and 3.5 see the 4.x.y releases)


</p>

<img align=left src="https://raw.githubusercontent.com/danielhrisca/asammdf/master/gui.png"/>


# Status

! | Travis CI  | Appveyor | CoverAlls  |  Codacy  | ReadTheDocs 
--|--|--|--|--|--
master | [![Build Status](https://travis-ci.org/danielhrisca/asammdf.svg?branch=)](https://travis-ci.org/danielhrisca/asammdf) | [![Build status](https://ci.appveyor.com/api/projects/status/racx048r4cnwa2lg/branch/master?svg=true)](https://ci.appveyor.com/project/danielhrisca/asammdf/branch/master) | [![Coverage Status](https://coveralls.io/repos/github/danielhrisca/asammdf/badge.svg?branch=master)](https://coveralls.io/github/danielhrisca/asammdf?branch=master) | [![Codacy Badge](https://api.codacy.com/project/badge/Grade/a3da21da90ca43a5b72fc24b56880c99?branch=master)](https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=badger) |  [![Documentation Status](http://readthedocs.org/projects/asammdf/badge/?version=master)](http://asammdf.readthedocs.io/en/master/?badge=stable) | 

PyPI| conda-forge
--|--
[![PyPI version](https://badge.fury.io/py/asammdf.svg)](https://badge.fury.io/py/asammdf)  | [![conda-forge version](https://anaconda.org/conda-forge/asammdf/badges/version.svg)](https://anaconda.org/conda-forge/asammdf)


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
* export to pandas, Excel, HDF5, Matlab (v4, v5 and v7.3),CSV and parquet
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

 * graphical interface to visualize channels and perform operations with the files

# Major features not implemented (yet)

* for version 3

    * functionality related to sample reduction block: the samples reduction blocks are simply ignored

* for version 4

    * functionality related to sample reduction block: the samples reduction blocks are simply ignored
    * handling of channel hierarchy: channel hierarchy is ignored
    * full handling of bus logging measurements: currently only CAN bus logging is implemented with the
      ability to *get* signals defined in the attached CAN database (.arxml or .dbc)
    * handling of unfinished measurements (mdf 4): warnings are logged based on the unfinished status flags
      but no further steps are taken to sanitize the measurement
    * full support for remaining mdf 4 channel arrays types
    * xml schema for MDBLOCK: most metadata stored in the comment blocks will not be available
    * full handling of event blocks: events are transfered to the new files (in case of calling methods
      that return new *MDF* objects) but no new events can be created
    * channels with default X axis: the defaukt X axis is ignored and the channel group's master channel
      is used

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
efficient = MDF('huge.mf4')
for signal in efficient.select(['Sensor1', 'Voltage3']):
   signal.plot()
```

Check the *examples* folder for extended usage demo, or the documentation
http://asammdf.readthedocs.io/en/master/examples.html

# Documentation
http://asammdf.readthedocs.io/en/master

# Contributing & Support
Please have a look over the [contributing guidelines](CONTRIBUTING.md)

If you enjoy this library please consider making a donation to the
[numpy project](https://www.flipcause.com/secure/cause_pdetails/MzUwMQ==).

## Contributors
Thanks to all who contributed with commits to *asammdf*:
* Julien Grave [JulienGrv](https://github.com/JulienGrv)
* Jed Frey [jed-frey](https://github.com/jed-frey)
* Mihai [yahym](https://github.com/yahym)
* Jack Weinstein [jackjweinstein](https://github.com/jackjweinstein)
* Isuru Fernando [isuruf](https://github.com/isuruf)
* Felix Kohlgrüber [fkohlgrueber](https://github.com/fkohlgrueber)
* Stanislav Frolov [stanifrolov](https://github.com/stanifrolov)
* Thomas Kastl [kasuteru](https://github.com/kasuteru)
* venden [venden](https://github.com/venden)
* Marat K. [kopytjuk](https://github.com/kopytjuk>)
* freakatzz [freakatzz](https://github.com/freakatzz)

# Installation
*asammdf* is available on

* github: https://github.com/danielhrisca/asammdf/
* PyPI: https://pypi.org/project/asammdf/
* conda-forge: https://anaconda.org/conda-forge/asammdf

```shell
pip install asammdf
# or for anaconda
conda install -c conda-forge asammdf
```

# Dependencies
asammdf uses the following libraries

* numpy : the heart that makes all tick
* numexpr : for algebraic and rational channel conversions
* wheel : for installation in virtual environments
* pandas : for DataFrame export
* canmatrix : to handle CAN bus logging measurements
* natsort

optional dependencies needed for exports

* h5py : for HDF5 export
* xlsxwriter : for Excel export
* scipy : for Matlab v4 and v5 .mat export
* hdf5storage : for Matlab v7.3 .mat export
* fastparquet : for parquet export

other optional dependencies

* cChardet : to detect non-standard unicode encodings
* PyQt5 : for GUI tool
* pyqtgraph : for GUI tool and Signal plotting
* matplotlib : as fallback for Signal plotting

# Benchmarks

http://asammdf.readthedocs.io/en/master/benchmarks.html

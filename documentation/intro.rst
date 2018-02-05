------------
Introduction
------------

Project goals
=============
The main goals for this library are:

* to be faster than the other Python based mdf libraries
* to have clean and easy to understand code base

Features
========

* create new mdf files from scratch
* append new channels
* read unsorted MDF v2, v3 and v4 files
* filter a subset of channels from original mdf file
* cut measurement to specified time interval
* convert to different mdf version
* export to Excel, HDF5, Matlab and CSV
* merge multiple files sharing the same internal structure
* read and save mdf version 4.10 files containing zipped data blocks
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
    * handling of unfinnished measurements (mdf 4)
    * full support for remaining mdf 4 channel arrays types
    * xml schema for TXBLOCK and MDBLOCK
    * partial conversions
    * event blocks
    * channels with default X axis
    * chanenls with reference to attachment
    
    
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

other optional dependencies

* chardet : to detect non-standard unicode encodings


Installation
============
*asammdf* is available on 

    * github: https://github.com/danielhrisca/asammdf/
    * PyPI: https://pypi.org/project/asammdf/

.. code-block:: python

    pip install asammdf

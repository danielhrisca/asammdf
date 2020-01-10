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
* read unsorted MDF v3 and v4 files
* read CAN bus logging files
* extract CAN signals from anonymous CAN bu logging measurements
* filter a subset of channels from original mdf file
* cut measurement to specified time interval
* convert to different mdf version
* export to pandas, HDF5, Matlab (v4, v5 and v7.3), CSV and parquet
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
* handle large files (for example merging two fileas, each with 14000 channels and 5GB size, on a RaspberryPi)
* extract channel data, master channel and extra channel information as *Signal* objects for unified operations with v3 and v4 files
* time domain operation using the *Signal* class

    * Pandas data frames are good if all the channels have the same time base
    * a measurement will usually have channels from different sources at different rates
    * the *Signal* class facilitates operations with such channels

Major features not implemented (yet)
====================================

* for version 3

    * functionality related to sample reduction block: the sample reduction blocks are simply ignored

* for version 4

    * functionality related to sample reduction block: the sample reduction blocks are simply ignored
    * handling of channel hierarchy: channel hierarchy is ignored
    * full handling of bus logging measurements: currently only CAN bus logging is implemented with the
      ability to *get* signals defined in the attached CAN database (.arxml or .dbc). Signals can also
      be extracted from an anonymous CAN logging measurement by providing a CAN database (.dbc or .arxml)
    * handling of unfinished measurements (mdf 4): warnings are logged based on the unfinished status flags
      but no further steps are taken to sanitize the measurement
    * full support for remaining mdf 4 channel arrays types
    * xml schema for MDBLOCK: most metadata stored in the comment blocks will not be available
    * full handling of event blocks: events are transferred to the new files (in case of calling methods
      that return new *MDF* objects) but no new events can be created
    * channels with default X axis: the default X axis is ignored and the channel group's master channel
      is used


Dependencies
============
asammdf uses the following libraries

* numpy : the heart that makes all tick 
* numexpr : for algebraic and rational channel conversions
* wheel : for installation in virtual environments
* pandas : for DataFrame export
* canmatrix : to handle CAN bus logging measurements
* natsort
* cChardet : to detect non-standard unicode encodings
* lxml : for canmatrix arxml support
* lz4 : to speed up the disk IO peformance

optional dependencies needed for exports

* h5py : for HDF5 export
* scipy : for Matlab v4 and v5 .mat export
* hdf5storage : for Matlab v7.3 .mat export
* fastparquet : for parquet export

other optional dependencies

* PyQt5 : for GUI tool
* pyqtgraph : for GUI tool and Signal plotting (preferably the latest develop branch code)
* matplotlib : as fallback for Signal plotting


Installation
============
*asammdf* is available on

    * github: https://github.com/danielhrisca/asammdf/
    * PyPI: https://pypi.org/project/asammdf/
    * conda-forge: https://anaconda.org/conda-forge/asammdf

    .. code:: python

       pip install asammdf
       # or for anaconda
       conda install -c conda-forge asammdf

Contributing & Support
======================
Please have a look over the `contributing guidelines <https://github.com/danielhrisca/asammdf/blob/master/CONTRIBUTING.md>`_

If you enjoy this library please consider making a donation to the 
`numpy project <https://www.flipcause.com/secure/cause_pdetails/MzUwMQ==>`_

Contributors
------------
Thanks to all who contributed with commits to *asammdf*

* Julien Grave `JulienGrv <https://github.com/JulienGrv>`_.
* Jed Frey `jed-frey <https://github.com/jed-frey>`_.
* Mihai `yahym <https://github.com/yahym>`_.
* Jack Weinstein `jackjweinstein <https://github.com/jackjweinstein>`_.
* Isuru Fernando `isuruf <https://github.com/isuruf>`_.
* Felix Kohlgrüber `fkohlgrueber <https://github.com/fkohlgrueber>`_.
* Stanislav Frolov `stanifrolov <https://github.com/stanifrolov>`_.
* Thomas Kastl `kasuteru <https://github.com/kasuteru>`_.
* venden `venden <https://github.com/venden>`_.
* Marat K. `kopytjuk` <https://github.com/kopytjuk>`_.
* freakatzz `freakatzz` <https://github.com/freakatzz>`_.



------------
Introduction
------------

Project goals
=============
The main goals for this library are:

* to be faster than the other Python-based mdf libraries
* to have clean and easy-to-understand code base
* to have minimal 3rd party dependencies

Features
========

* create new mdf files from scratch
* append new channels
* read unsorted MDF v3 and v4 files
* read CAN and LIN bus logging files
* extract CAN and LIN signals from anonymous bus logging measurements
* filter a subset of channels from original mdf file
* cut measurement to specified time interval
* convert to different mdf version
* export to pandas, HDF5, Matlab (v7.3), CSV and parquet
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
* handle large files (for example merging two files, each with 14000 channels and 5GB size, on a RaspberryPi)
* extract channel data, master channel and extra channel information as `Signal` objects for unified operations with v3 and v4 files
* time domain operation using the `Signal` class

  * pandas DataFrames are good if all the channels have the same time base
  * a measurement will usually have channels from different sources at different rates
  * the `Signal` class facilitates operations with such channels

* graphical interface to visualize channels and perform operations with the files

Major features not implemented (yet)
====================================

* for version 3

  * functionality related to sample reduction block: the sample reduction blocks are simply ignored

* for version 4

  * experimental support for MDF v4.20 column oriented storage
  * functionality related to sample reduction block: the sample reduction blocks are simply ignored
  * handling of channel hierarchy: channel hierarchy is ignored
  * full handling of bus logging measurements: currently only CAN and LIN bus logging are implemented with the
    ability to *get* signals defined in the attached CAN/LIN database (.arxml or .dbc). Signals can also
    be extracted from an anonymous bus logging measurement by providing a CAN or LIN database (.dbc or .arxml)
  * handling of unfinished measurements (mdf 4): finalization is attempted when the file is loaded, however
    not all the finalization steps are supported
  * full support for remaining mdf 4 channel arrays types
  * xml schema for MDBLOCK: most metadata stored in the comment blocks will not be available
  * full handling of event blocks: events are transferred to the new files (in case of calling methods
    that return new `MDF` objects) but no new events can be created
  * channels with default X axis: the default X axis is ignored and the channel group's master channel
    is used
  * attachment encryption/decryption using user provided encryption/decryption functions; this is not
    part of the MDF v4 spec and is only supported by this library


Dependencies
============
`asammdf` uses the following libraries

* numpy : the heart that makes all tick 
* numexpr : for algebraic and rational channel conversions
* wheel : for installation in virtual environments
* pandas : for DataFrame export
* canmatrix : to handle CAN/LIN bus logging measurements
* natsort
* lxml : for canmatrix arxml support
* lz4 : to speed up the disk IO performance
* python-dateutil : measurement start time handling

Optional dependencies needed for exports

* h5py : for HDF5 export
* hdf5storage : for Matlab v7.3 .mat export
* pyarrow : for parquet export
* scipy: for Matlab v4 and v5 .mat export

Other optional dependencies

* PySide6 : for GUI tool
* pyqtgraph : for GUI tool and Signal plotting
* matplotlib : as fallback for Signal plotting
* faust-cchardet : to detect non-standard Unicode encodings
* chardet : to detect non-standard Unicode encodings 
* pyqtlet2 : for the GPS window
* isal : for faster zlib compression/decompression
* fsspec : access files stored in the cloud


Installation
============
`asammdf` is available on

* GitHub: https://github.com/danielhrisca/asammdf/
* PyPI: https://pypi.org/project/asammdf/
* conda-forge: https://anaconda.org/conda-forge/asammdf

.. code:: python

   pip install asammdf
   # for the GUI
   pip install asammdf[gui]
   # or for anaconda
   conda install -c conda-forge asammdf

In case a wheel is not present for your OS/Python versions and you
lack the proper compiler setup to compile the C-extension code, then
you can simply copy-paste the package code to your site-packages. In this
way the Python fallback code will be used instead of the compiled C-extension code.


Contributing & Support
======================
Please have a look at the `contributing guidelines <https://github.com/danielhrisca/asammdf/blob/master/CONTRIBUTING.md>`_.

If you enjoy this library please consider making a donation to the 
`numpy project <https://numfocus.org/donate-to-numpy>`_ or to `danielhrisca using liberapay <https://liberapay.com/danielhrisca/donate>`_.

Contributors
------------
Thanks to all who contributed with commits to `asammdf`


.. raw:: html

    <a href="https://github.com/danielhrisca/asammdf/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=danielhrisca/asammdf" />
    </a>




.. asammdf documentation master file, created by
   sphinx-quickstart on Wed Jul 12 06:05:15 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to asammdf's documentation!
===================================

*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports both MDF version 3 and 4 formats. 

*asammdf* works on Python 2.7, and Python >= 3.4


Project goals
-------------
The main goals for this library are:

* to be faster than the other Python based mdf libraries
* to have clean and easy to understand code base

Features
--------

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
* filter a subset of channels from original mdf file
* convert to different mdf version
* export to Excel, HDF5 and CSV
* add and extract attachments
* mdf 4.10 zipped blocks

Major features still not implemented
------------------------------------

* functionality related to sample reduction block (but the class is defined)
* mdf 3 channel dependency append (reading and saving file with CDBLOCKs is implemented)
* handling of unfinnished measurements (mdf 4)
* mdf 4 channel arrays
* xml schema for TXBLOCK and MDBLOCK
    
Dependencies
------------
asammdf uses the following libraries
    
* numpy : the heart that makes all tick
* numexpr : for algebraic and rational channel conversions
* matplotlib : for Signal plotting

optional dependencies needed for exports

* pandas : for DataFrame export
* h5py : for HDF5 export
* xlsxwriter : for Excel export


Installation
------------
*asammdf* is available on 

    * github: https://github.com/danielhrisca/asammdf/
    * PyPI: https://pypi.org/project/asammdf/

.. code-block:: python

    pip install asammdf


API
--------

.. toctree::
   :maxdepth: 1
   
   mdf
   signal
   examples

Benchmarks
----------
*asammdf* relies heavily on *dict* objects. Starting with Python 3.6 the *dict* objects are more compact and ordered (implementation detail); *asammdf* uses takes advantage of those changes
so for best performance it is advised to use Python >= 3.6.

.. toctree::
   :maxdepth: 2
   
   benchmarks


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

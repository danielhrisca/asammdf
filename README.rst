
.. image:: https://raw.githubusercontent.com/danielhrisca/asammdf/development/asammdf.png
    :height: 200px
    :width: 200px
    :align: center

----

*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports MDF versions 2 (.dat), 3 (.mdf) and 4 (.mf4). 

*asammdf* works on Python 2.7, and Python >= 3.4 (Travis CI tests done with Python 2.7 and Python >= 3.5)


----

.. image:: https://raw.githubusercontent.com/danielhrisca/asammdf/development/gui.png

Status
======

+-------------+----------------+-------------------+-----------------+---------------+
|             | Travis CI      | Coverage          | Codacy          | ReadTheDocs   |
+=============+================+===================+=================+===============+
| master      | |Build Master| | |Coverage Master| | |Codacy Master| | |Docs Master| |
+-------------+----------------+-------------------+-----------------+---------------+
| development | |Build Status| | |Coverage Badge|  | |Codacy Badge|  | |Docs Status| |
+-------------+----------------+-------------------+-----------------+---------------+

+----------------+-----------------------+
| PyPI           | conda-forge           |
+================+=======================+
| |PyPI version| | |conda-forge version| |
+----------------+-----------------------+


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
* read CAN bus logging files
* filter a subset of channels from original mdf file
* cut measurement to specified time interval
* convert to different mdf version
* export to pandas, Excel, HDF5, Matlab (v4, v5 and v7.3) and CSV
* merge (concatenate) multiple files sharing the same internal structure
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
* handle large files (for example merging two files, each with 14000 channels and 5GB size, on a RaspberryPi) using *memory* = *minimum* argument
* extract channel data, master channel and extra channel information as *Signal* objects for unified operations with v3 and v4 files
* time domain operation using the *Signal* class

    * Pandas data frames are good if all the channels have the same time based
    * a measurement will usually have channels from different sources at different rates
    * the *Signal* class facilitates operations with such channels
    
 * graphical interface to visualize channels and perform operations with the files


Major features not implemented (yet)
====================================

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

Usage
=====

.. code-block:: python

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
   efficient = MDF('huge.mf4', memory='minimum')
   for signal in efficient.select(['Sensor1', 'Voltage3']):
       signal.plot()
   

 
Check the *examples* folder for extended usage demo, or the documentation
http://asammdf.readthedocs.io/en/master/examples.html

Documentation
=============
http://asammdf.readthedocs.io/en/master

Contributing
============
Please have a look over the [contributing guidelines](https://github.com/danielhrisca/asammdf/blob/master/CONTRIBUTING.md)

Contributors
------------
Thanks to all who contributed with commits to *asammdf*

* Julien Grave `JulienGrv <https://github.com/JulienGrv>`_.
* Jed Frey `jed-frey <https://github.com/jed-frey>`_.
* Mihai `yahym <https://github.com/yahym>`_.
* Jack Weinstein `jacklev <https://github.com/jacklev>`_.
* Isuru Fernando `isuruf <https://github.com/isuruf>`_.
* Felix Kohlgrüber `fkohlgrueber <https://github.com/fkohlgrueber>`_.

Installation
============
*asammdf* is available on 

* github: https://github.com/danielhrisca/asammdf/
* PyPI: https://pypi.org/project/asammdf/
* conda-forge: https://anaconda.org/conda-forge/asammdf
    
.. code-block: python

   pip install asammdf
   # or for anaconda
   conda install -c conda-forge asammdf

    
Dependencies
============
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
* scipy : for Matlab v4 and v5 .mat export
* hdf5storage : for Matlab v7.3 .mat export

other optional dependencies

* chardet : to detect non-standard unicode encodings
* PyQt4 or PyQt5 : for GUI tool
* pyqtgraph : for GUI tool


Benchmarks
==========

Graphical results can be seen here at http://asammdf.readthedocs.io/en/master/benchmarks.html


Python 3 x64
------------
Benchmark environment

* 3.6.4 |Anaconda, Inc.| (default, Jan 16 2018, 18:10:19) [GCC 7.2.0]
* Linux-4.13.0-37-generic-x86_64-with-debian-stretch-sid
* x86_64
* 8GB installed RAM

Notations used in the results

* full =  asammdf MDF object created with memory=full (everything loaded into RAM)
* low =  asammdf MDF object created with memory=low (raw channel data not loaded into RAM, but metadata loaded to RAM)
* minimum =  asammdf MDF object created with memory=full (lowest possible RAM usage)
* compress = mdfreader mdf object created with compression=blosc
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* mdf version 3.10
    * 167 MB file size
    * 183 groups
    * 36424 channels
* mdf version 4.00
    * 183 MB file size
    * 183 groups
    * 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.0.0.dev full mdfv3                            2013      331
asammdf 4.0.0.dev low mdfv3                             1913      178
asammdf 4.0.0.dev minimum mdfv3                          617       64
mdfreader 2.7.7 mdfv3                                   2892      241
mdfreader 2.7.7 compress mdfv3                          2947      234
mdfreader 2.7.7 noDataLoading mdfv3                     1652      175
asammdf 4.0.0.dev full mdfv4                            3504      304
asammdf 4.0.0.dev low mdfv4                             3292      140
asammdf 4.0.0.dev minimum mdfv4                         2663       64
mdfreader 2.7.7 mdfv4                                   8215      440
mdfreader 2.7.7 compress mdfv4                          8535      309
mdfreader 2.7.7 noDataLoading mdfv4                     5413      182
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.0.0.dev full mdfv3                            1083      338
asammdf 4.0.0.dev low mdfv3                             1308      185
asammdf 4.0.0.dev minimum mdfv3                         3936       68
mdfreader 2.7.7 mdfv3                                     0*       0*
mdfreader 2.7.7 noDataLoading mdfv3                       0*       0*
mdfreader 2.7.7 compress mdfv3                            0*       0*
asammdf 4.0.0.dev full mdfv4                            1279      309
asammdf 4.0.0.dev low mdfv4                             1679      149
asammdf 4.0.0.dev minimum mdfv4                         3776       74
mdfreader 2.7.7 mdfv4                                   6710      465
mdfreader 2.7.7 noDataLoading mdfv4                     9615      483
mdfreader 2.7.7 compress mdfv4                          7191      463
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.0.0.dev full mdfv3                            2182      342
asammdf 4.0.0.dev low mdfv3                             9133      195
asammdf 4.0.0.dev minimum mdfv3                        13574       81
mdfreader 2.7.7 mdfv3                                      4      241
mdfreader 2.7.7 nodata mdfv3                            2319      204
mdfreader 2.7.7 compress mdfv3                            42      234
asammdf 4.0.0.dev full mdfv4                            2072      311
asammdf 4.0.0.dev low mdfv4                            11022      151
asammdf 4.0.0.dev minimum mdfv4                        18972       82
mdfreader 2.7.7 mdfv4                                    114      440
mdfreader 2.7.7 nodata mdfv4                           23070      208
mdfreader 2.7.7 compress mdfv4                           255      313
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.0.0.dev full v3 to v4                         6898      674
asammdf 4.0.0.dev low v3 to v4                          7447      343
asammdf 4.0.0.dev minimum v3 to v4                     11432      116
asammdf 4.0.0.dev full v4 to v3                         7294      601
asammdf 4.0.0.dev low v4 to v3                          6613      251
asammdf 4.0.0.dev minimum v4 to v3                     13755      110
================================================== ========= ========


================================================== ========= ========
Merge 2 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.0.0.dev full v3                              14380     1107
asammdf 4.0.0.dev low v3                               13896      425
asammdf 4.0.0.dev minimum v3                           20179      138
mdfreader 2.7.7 v3                                      6081      251
mdfreader 2.7.7 compress v3                             6285      250
mdfreader 2.7.7 nodata v3                                 0*       0*
asammdf 4.0.0.dev full v4                              18774     1054
asammdf 4.0.0.dev low v4                               26612      349
asammdf 4.0.0.dev minimum v4                           34256      135
mdfreader 2.7.7 v4                                     28264      960
mdfreader 2.7.7 nodata v4                              24660      998
mdfreader 2.7.7 compress v4                            22881      959
================================================== ========= ========


.. |Build Master| image:: https://travis-ci.org/danielhrisca/asammdf.svg?branch=master
   :target: https://travis-ci.org/danielhrisca/asammdf
.. |Coverage Master| image:: https://api.codacy.com/project/badge/Coverage/a3da21da90ca43a5b72fc24b56880c99?branch=master
   :target: https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=Badge_Coverage
.. |Codacy Master| image:: https://api.codacy.com/project/badge/Grade/a3da21da90ca43a5b72fc24b56880c99?branch=master
   :target: https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=badger
.. |Docs Master| image:: http://readthedocs.org/projects/asammdf/badge/?version=master
   :target: http://asammdf.readthedocs.io/en/master/?badge=stable
.. |Build Status| image:: https://travis-ci.org/danielhrisca/asammdf.svg?branch=development
   :target: https://travis-ci.org/danielhrisca/asammdf
.. |Coverage Badge| image:: https://api.codacy.com/project/badge/Coverage/a3da21da90ca43a5b72fc24b56880c99?branch=development
   :target: https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=Badge_Coverage
.. |Codacy Badge| image:: https://api.codacy.com/project/badge/Grade/a3da21da90ca43a5b72fc24b56880c99?branch=development
   :target: https://www.codacy.com/app/danielhrisca/asammdf?utm_source=github.com&utm_medium=referral&utm_content=danielhrisca/asammdf&utm_campaign=badger
.. |Docs Status| image:: http://readthedocs.org/projects/asammdf/badge/?version=development
   :target: http://asammdf.readthedocs.io/en/master/?badge=stable
.. |PyPI version| image:: https://badge.fury.io/py/asammdf.svg
   :target: https://badge.fury.io/py/asammdf
.. |conda-forge version| image:: https://anaconda.org/conda-forge/asammdf/badges/version.svg
   :target: https://anaconda.org/conda-forge/asammdf
.. |anaconda-cloud version| image:: https://anaconda.org/daniel.hrisca/asammdf/badges/version.svg
   :target: https://anaconda.org/daniel.hrisca/asammdf


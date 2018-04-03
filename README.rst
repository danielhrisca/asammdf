*asammdf* is a fast parser/editor for ASAM (Associtation for Standardisation of Automation and Measuring Systems) MDF (Measurement Data Format) files. 

*asammdf* supports MDF versions 2 (.dat), 3 (.mdf) and 4 (.mf4). 

*asammdf* works on Python 2.7, and Python >= 3.4 (Travis CI tests done with Python 2.7 and Python >= 3.5)


Status
======

+-------------+----------------+-------------------+-----------------+---------------+
|             | Travis CI      | Coverage          | Codacy          | ReadTheDocs   |
+=============+================+===================+=================+===============+
| master      | |Build Master| | |Coverage Master| | |Codacy Master| | |Docs Master| |
+-------------+----------------+-------------------+-----------------+---------------+
| development | |Build Status| | |Coverage Badge|  | |Codacy Badge|  | |Docs Status| |
+-------------+----------------+-------------------+-----------------+---------------+

+----------------+-----------------------+--------------------------+
| PyPI           | conda-forge           | anaconda-cloud           |
+================+=======================+==========================+
| |PyPI version| | |conda-forge version| | |anaconda-cloud version| |
+----------------+-----------------------+--------------------------+


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
* export to Excel, HDF5, Matlab, CSV and pandas
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


Major features not implemented (yet)
====================================

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
Thanks to all who contributed with commits to *asammdf*:
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
* scipy : for Matlab .mat export

other optional dependencies

* chardet : to detect non-standard unicode encodings


Benchmarks
==========

Graphical results can be seen here at http://asammdf.readthedocs.io/en/master/benchmarks.html


Python 3 x64
------------
Benchmark environment

* 3.6.4 (default, Jan  5 2018, 02:35:40) [GCC 7.2.1 20171224]
* Linux-4.15.0-1-MANJARO-x86_64-with-arch-Manjaro-Linux
* 
* 4GB installed RAM

Notations used in the results

* full =  asammdf MDF object created with memory=full (everything loaded into RAM)
* low =  asammdf MDF object created with memory=low (raw channel data not loaded into RAM, but metadata loaded to RAM)
* minimum =  asammdf MDF object created with memory=full (lowest possible RAM usage)
* compress = mdfreader mdf object created with compression=blosc
* noDataLoading = mdfreader mdf object read with noDataLoading=True

Files used for benchmark:

* 183 groups
* 36424 channels



================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 3.0.0    full mdfv3                              706      256
asammdf 3.0.0    low mdfv3                               637      103
asammdf 3.0.0    minimum mdfv3                           612       64
mdfreader 2.7.5 mdfv3                                   2201      414
mdfreader 2.7.5 compress mdfv3                          1871      281
mdfreader 2.7.5 noDataLoading mdfv3                      948      160
asammdf 3.0.0    full mdfv4                             2599      296
asammdf 3.0.0    low mdfv4                              2485      131
asammdf 3.0.0    minimum mdfv4                          1376       64
mdfreader 2.7.5 mdfv4                                   5706      435
mdfreader 2.7.5 compress mdfv4                          5453      303
mdfreader 2.7.5 noDataLoading mdfv4                     3904      181
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 3.0.0    full mdfv3                              468      258
asammdf 3.0.0    low mdfv3                               363      110
asammdf 3.0.0    minimum mdfv3                           919       80
mdfreader 2.7.5 mdfv3                                   6424      451
mdfreader 2.7.5 noDataLoading mdfv3                     7364      510
mdfreader 2.7.5 compress mdfv3                          6624      449
asammdf 3.0.0    full mdfv4                              984      319
asammdf 3.0.0    low mdfv4                              1028      156
asammdf 3.0.0    minimum mdfv4                          2786       80
mdfreader 2.7.5 mdfv4                                   3355      460
mdfreader 2.7.5 noDataLoading mdfv4                     5153      483
mdfreader 2.7.5 compress mdfv4                          3773      457
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 3.0.0    full mdfv3                             1196      269
asammdf 3.0.0    low mdfv3                              5230      121
asammdf 3.0.0    minimum mdfv3                          6871       85
mdfreader 2.7.5 mdfv3                                     77      414
mdfreader 2.7.5 noDataLoading mdfv3                    13036      195
mdfreader 2.7.5 compress mdfv3                           184      281
asammdf 3.0.0    full mdfv4                             1207      305
asammdf 3.0.0    low mdfv4                              5613      144
asammdf 3.0.0    minimum mdfv4                          7725       80
mdfreader 2.7.5 mdfv4                                     74      435
mdfreader 2.7.5 noDataLoading mdfv4                    14140      207
mdfreader 2.7.5 compress mdfv4                           171      307
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 3.0.0    full v3 to v4                          3712      565
asammdf 3.0.0    low v3 to v4                           4091      228
asammdf 3.0.0    minimum v3 to v4                       6740      126
asammdf 3.0.0    full v4 to v3                          3787      571
asammdf 3.0.0    low v4 to v3                           4546      222
asammdf 3.0.0    minimum v4 to v3                       8369      115
================================================== ========= ========


================================================== ========= ========
Merge files                                        Time [ms] RAM [MB]
================================================== ========= ========
asammdf 3.0.0    full v3                                7297      975
asammdf 3.0.0    low v3                                 7766      282
asammdf 3.0.0    minimum v3                            11363      163
mdfreader 2.7.5 mdfv3                                  13039     1301
mdfreader 2.7.5 compress mdfv3                         12877     1298
mdfreader 2.7.5 noDataLoading mdfv3                    12981     1421
asammdf 3.0.0    full v4                               11313     1025
asammdf 3.0.0    low v4                                12155      322
asammdf 3.0.0    minimum v4                            18787      152
mdfreader 2.7.5 mdfv4                                  21423     1309
mdfreader 2.7.5 noDataLoading mdfv4                    20142     1352
mdfreader 2.7.5 compress mdfv4                         20600     1309
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


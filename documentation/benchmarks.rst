.. raw:: html

    <style> .red {color:red} </style>
    <style> .blue {color:blue} </style>
    <style> .green {color:green} </style>
    <style> .cyan {color:cyan} </style>
    <style> .magenta {color:magenta} </style>
    <style> .orange {color:orange} </style>
    <style> .brown {color:brown} </style>
    
.. role:: red
.. role:: blue
.. role:: green
.. role:: cyan
.. role:: magenta
.. role:: orange
.. role:: brown

.. _bench:

Benchmarks
==========
The benchmarks were done using two test files (for mdf version 3 and 4) of around 170MB. 
The files contain 183 data groups and a total of 36424 channels.

*asamdf 2.0.0* was compared against *mdfreader 0.2.5*. 
*mdfreader* seems to be the most used Python package to handle MDF files, and it also supports both version 3 and 4 of the standard.

The three benchmark cathegories are file open, file save and extracting the data for all channels inside the file(36424 calls).
For each cathegory two aspect were noted: elapsed time and peak RAM usage.

x64 Python results
------------------
The test environment used for 64 bit tests had:

* Python 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 18:41:36) [MSC v.1900 64 bit (AMD64)]
* Windows-7-6.1.7601-SP1
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel (i7-6820Q)
* 16GB installed RAM

The notations used in the results have the following meaning:

* nodata = MDF object created with load_measured_data=False (raw channel data no loaded into RAM)
* compression = MDF object created with compression=True (raw channel data loaded into RAM and compressed)
* noconvert = MDF object created with convertAfterRead=False

========================================          =========       ========
Open file                                         Time [ms]       RAM [MB]
========================================          =========       ========
asammdf 2.0.0 mdfv3                                     721            352
asammdf 2.0.0 compression mdfv3                        1008            275
asammdf 2.0.0 nodata mdfv3                              641            199
mdfreader 0.2.5 mdfv3                                  2996            526
mdfreader 0.2.5 no convert mdfv3                       2846            393
asammdf 2.0.0 mdfv4                                    1634            439
asammdf 2.0.0 compression mdfv4                        1917            343
asammdf 2.0.0 nodata mdfv4                             1594            274
mdfreader 0.2.5 mdfv4                                 31023            739
mdfreader 0.2.5 noconvert mdfv4                       30693            609
========================================          =========       ========


========================================          =========       ========
Save file                                         Time [ms]       RAM [MB]
========================================          =========       ========
asammdf 2.0.0 mdfv3                                     472            353
asammdf 2.0.0 compression mdfv3                         667            275
mdfreader 0.2.5 mdfv3                                 18910           2003
asammdf 2.0.0 mdfv4                                     686            447
asammdf 2.0.0 compression mdfv4                         836            352
mdfreader 0.2.5 mdfv4                                 16631           2802
========================================          =========       ========


========================================          =========       ========
Get all channels                                  Time [ms]       RAM [MB]
========================================          =========       ========
asammdf 2.0.0 mdfv3                                    2492            362
asammdf 2.0.0 compression mdfv3                       14474            285
asammdf 2.0.0 nodata mdfv3                             9621            215
mdfreader 0.2.5 mdfv3                                    31            526
asammdf 2.0.0 mdfv4                                    2066            450
asammdf 2.0.0 compression mdfv4                       16944            359
asammdf 2.0.0 nodata mdfv4                            12364            292
mdfreader 0.2.5 mdfv4                                    39            739
========================================          =========       ========

x86 Python results
------------------
The test environment used for 32 bit tests had:

* Python 3.6.1 (v3.6.1:69c0db5, Mar 21 2017, 17:54:52) [MSC v.1900 32 bit (Intel)]
* Windows-7-6.1.7601-SP1
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel (i7-6820Q)
* 16GB installed RAM

The notations used in the results have the following meaning:

* nodata = MDF object created with load_measured_data=False (raw channel data no loaded into RAM)
* compression = MDF object created with compression=True (raw channel data loaded into RAM and compressed)
* noconvert = MDF object created with convertAfterRead=False


========================================          =========       ========
Open file                                         Time [ms]       RAM [MB]
========================================          =========       ========
asammdf 2.0.0 mdfv3                                     851            283
asammdf 2.0.0 compression mdfv3                        1149            190
asammdf 2.0.0 nodata mdfv3                              765            129
mdfreader 0.2.5 mdfv3                                  3633            453
mdfreader 0.2.5 no convert mdfv3                       3309            319
asammdf 2.0.0 mdfv4                                    1854            339
asammdf 2.0.0 compression mdfv4                        2191            236
asammdf 2.0.0 nodata mdfv4                             1772            173
mdfreader 0.2.5 mdfv4                                 42177            576
mdfreader 0.2.5 noconvert mdfv4                       41799            447
========================================          =========       ========


========================================          =========       ========
Save file                                         Time [ms]       RAM [MB]
========================================          =========       ========
asammdf 2.0.0 mdfv3                                     564            286
asammdf 2.0.0 compression mdfv3                         756            194
mdfreader 0.2.5 mdfv3                                 17499           1236
asammdf 2.0.0 mdfv4                                     906            347
asammdf 2.0.0 compression mdfv4                        1112            244
mdfreader 0.2.5 mdfv4                                 15027           1698
========================================          =========       ========


========================================          =========       ========
Get all channels                                  Time [ms]       RAM [MB]
========================================          =========       ========
asammdf 2.0.0 mdfv3                                    3224            293
asammdf 2.0.0 compression mdfv3                       25019            201
asammdf 2.0.0 nodata mdfv3                            18824            144
mdfreader 0.2.5 mdfv3                                    35            454
asammdf 2.0.0 mdfv4                                    2513            349
asammdf 2.0.0 compression mdfv4                       25140            250
asammdf 2.0.0 nodata mdfv4                            19862            188
mdfreader 0.2.5 mdfv4                                    50            576
========================================          =========       ========


Benchmark environment

* 3.6.6 (default, Sep 12 2018, 18:26:19) 
[GCC 8.0.1 20180414 (experimental) [trunk revision 259383]]
* Linux-4.15.0-38-generic-x86_64-with-Ubuntu-18.04-bionic
* x86_64
* 15GB installed RAM

Notations used in the results

* full =  asammdf MDF object created with memory=full (everything loaded into RAM)
* low =  asammdf MDF object created with memory=low (raw channel data not loaded into RAM, but metadata loaded to RAM)
* minimum =  asammdf MDF object created with memory=full (lowest possible RAM usage)
* compress = mdfreader mdf object created with compression=blosc
* no_data_loading = mdfreader mdf object read with no_data_loading=True

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
asammdf 4.2.1dev full mdfv3                             1305      340
asammdf 4.2.1dev low mdfv3                              1236      187
asammdf 4.2.1dev minimum mdfv3                           598       80
mdfreader 3.0 mdfv3                                     2231      435
mdfreader 3.0 compress mdfv3                            1987      301
mdfreader 3.0 no_data_loading mdfv3                     1035      183
asammdf 4.2.1dev full mdfv4                             1284      310
asammdf 4.2.1dev low mdfv4                              1145      145
asammdf 4.2.1dev minimum mdfv4                          1089       81
mdfreader 3.0 mdfv4                                     5509      460
mdfreader 3.0 compress mdfv4                            5202      328
mdfreader 3.0 no_data_loading mdfv4                     3552      193
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.2.1dev full mdfv3                              707      340
asammdf 4.2.1dev low mdfv3                               775      193
asammdf 4.2.1dev minimum mdfv3                          1959       86
mdfreader 3.0 mdfv3                                     5631      472
mdfreader 3.0 no_data_loading mdfv3                     6499      529
mdfreader 3.0 compress mdfv3                            5802      471
asammdf 4.2.1dev full mdfv4                              770      319
asammdf 4.2.1dev low mdfv4                               969      161
asammdf 4.2.1dev minimum mdfv4                          1959       96
mdfreader 3.0 mdfv4                                     4038      486
mdfreader 3.0 no_data_loading mdfv4                     5733      501
mdfreader 3.0 compress mdfv4                            4117      483
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.2.1dev full mdfv3                             1448      352
asammdf 4.2.1dev low mdfv3                              6566      204
asammdf 4.2.1dev minimum mdfv3                          7904       94
mdfreader 3.0 mdfv3                                       86      435
mdfreader 3.0 nodata mdfv3                             10937      216
mdfreader 3.0 compress mdfv3                             192      301
asammdf 4.2.1dev full mdfv4                             1532      317
asammdf 4.2.1dev low mdfv4                              7031      157
asammdf 4.2.1dev minimum mdfv4                          8222       93
mdfreader 3.0 mdfv4                                       98      460
mdfreader 3.0 compress mdfv4                             203      334
mdfreader 3.0 nodata mdfv4                             14895      226
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.2.1dev full v3 to v4                          4946      679
asammdf 4.2.1dev low v3 to v4                           5074      349
asammdf 4.2.1dev minimum v3 to v4                       7463      122
asammdf 4.2.1dev full v4 to v3                          4448      607
asammdf 4.2.1dev low v4 to v3                           4685      257
asammdf 4.2.1dev minimum v4 to v3                       7211      115
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.2.1dev full v3                               15050     1648
asammdf 4.2.1dev low v3                                14921      622
asammdf 4.2.1dev minimum v3                            20606      167
mdfreader 3.0 v3                                       19063     1314
mdfreader 3.0 compress v3                              19101     1313
mdfreader 3.0 nodata v3                                18661     1434
asammdf 4.2.1dev full v4                               14612     1509
asammdf 4.2.1dev low v4                                14546      450
asammdf 4.2.1dev minimum v4                            26211      164
mdfreader 3.0 v4                                       29422     1349
mdfreader 3.0 nodata v4                                28634     1330
mdfreader 3.0 compress v4                              29076     1299
================================================== ========= ========

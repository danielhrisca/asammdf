

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
asammdf 4.3.0 full mdfv3                                1304      340
asammdf 4.3.0 low mdfv3                                 1226      187
asammdf 4.3.0 minimum mdfv3                              587       80
mdfreader 3.0 mdfv3                                     2251      434
mdfreader 3.0 compress mdfv3                            2038      301
mdfreader 3.0 no_data_loading mdfv3                     1027      183
asammdf 4.3.0 full mdfv4                                1292      310
asammdf 4.3.0 low mdfv4                                 1161      146
asammdf 4.3.0 minimum mdfv4                             1082       81
mdfreader 3.0 mdfv4                                     5447      460
mdfreader 3.0 compress mdfv4                            5190      328
mdfreader 3.0 no_data_loading mdfv4                     3545      193
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.3.0 full mdfv3                                 983      340
asammdf 4.3.0 low mdfv3                                  827      194
asammdf 4.3.0 minimum mdfv3                             2009       86
mdfreader 3.0 mdfv3                                     5629      472
mdfreader 3.0 no_data_loading mdfv3                     6618      529
mdfreader 3.0 compress mdfv3                            6086      471
asammdf 4.3.0 full mdfv4                                 806      319
asammdf 4.3.0 low mdfv4                                  983      161
asammdf 4.3.0 minimum mdfv4                             1979       96
mdfreader 3.0 mdfv4                                     4162      485
mdfreader 3.0 no_data_loading mdfv4                     5774      501
mdfreader 3.0 compress mdfv4                            4209      483
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.3.0 full mdfv3                                1463      352
asammdf 4.3.0 low mdfv3                                 6353      204
asammdf 4.3.0 minimum mdfv3                             7671       94
mdfreader 3.0 mdfv3                                       85      434
mdfreader 3.0 nodata mdfv3                             10770      216
mdfreader 3.0 compress mdfv3                             189      301
asammdf 4.3.0 full mdfv4                                1553      317
asammdf 4.3.0 low mdfv4                                 7455      157
asammdf 4.3.0 minimum mdfv4                             8694       92
mdfreader 3.0 mdfv4                                       87      460
mdfreader 3.0 compress mdfv4                             204      333
mdfreader 3.0 nodata mdfv4                             15565      226
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.3.0 full v3 to v4                             5001      679
asammdf 4.3.0 low v3 to v4                              5118      349
asammdf 4.3.0 minimum v3 to v4                          7497      122
asammdf 4.3.0 full v4 to v3                             4531      607
asammdf 4.3.0 low v4 to v3                              4756      257
asammdf 4.3.0 minimum v4 to v3                          7226      115
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.3.0 full v3                                  14903     1172
asammdf 4.3.0 low v3                                   14862      372
asammdf 4.3.0 minimum v3                               18476      131
mdfreader 3.0 v3                                       19055     1314
mdfreader 3.0 compress v3                              19076     1313
mdfreader 3.0 nodata v3                                18650     1433
asammdf 4.3.0 full v4                                  14754     1107
asammdf 4.3.0 low v4                                   14883      287
asammdf 4.3.0 minimum v4                               19386      130
mdfreader 3.0 v4                                       29473     1348
mdfreader 3.0 nodata v4                                28672     1330
mdfreader 3.0 compress v4                              29113     1298
================================================== ========= ========

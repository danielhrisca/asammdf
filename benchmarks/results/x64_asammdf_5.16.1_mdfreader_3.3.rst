

Benchmark environment

* 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
* Windows-10-10.0.17763-SP0
* Intel64 Family 6 Model 158 Stepping 10, GenuineIntel
* numpy 1.17.2
* 16GB installed RAM

Notations used in the results

* compress = mdfreader mdf object created with compression=blosc
* nodata = mdfreader mdf object read with no_data_loading=True

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
asammdf 5.16.1 mdfv3                                     289      145
mdfreader 3.3 mdfv3                                     1581      461
mdfreader 3.3 no_data_loading mdfv3                      717      214
mdfreader 3.3 compress mdfv3                            1317      329
asammdf 5.16.1 mdfv4                                     420      157
mdfreader 3.3 mdfv4                                     3914      493
mdfreader 3.3 no_data_loading mdfv4                     2818      280
mdfreader 3.3 compress mdfv4                            3712      365
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.16.1 mdfv3                                     421      144
mdfreader 3.3 mdfv3                                     4076      490
mdfreader 3.3 no_data_loading mdfv3                     4586      552
mdfreader 3.3 compress mdfv3                            4202      488
asammdf 5.16.1 mdfv4                                     347      157
mdfreader 3.3 mdfv4                                     2320      512
mdfreader 3.3 no_data_loading mdfv4                     3470      588
mdfreader 3.3 compress mdfv4                            2440      507
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.16.1 mdfv3                                    3682      146
mdfreader 3.3 mdfv3                                       51      461
mdfreader 3.3 nodata mdfv3                             28372      251
mdfreader 3.3 compress mdfv3                             167      329
asammdf 5.16.1 mdfv4                                    6792      157
mdfreader 3.3 mdfv4                                       93      493
mdfreader 3.3 nodata mdfv4                             32486      314
mdfreader 3.3 compress mdfv4                             244      368
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.16.1 v3 to v4                                 3143      205
asammdf 5.16.1 v4 to v410                               2665      180
asammdf 5.16.1 v4 to v420                               3145      222
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.16.1 v3                                       7853      213
mdfreader 3.3 v3                                       16464     1367
mdfreader 3.3 nodata v3                                15816     1482
mdfreader 3.3 compress v3                              16459     1315
asammdf 5.16.1 v4                                       7216      208
mdfreader 3.3 v4                                       24205     1407
mdfreader 3.3 nodata v4                                24063     1455
mdfreader 3.3 compress v4                              24381     1343
================================================== ========= ========

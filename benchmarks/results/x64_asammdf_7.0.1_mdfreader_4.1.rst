Benchmark environment

* 3.9.4 (tags/v3.9.4:1f2e308, Apr  6 2021, 13:40:21) [MSC v.1928 64 bit (AMD64)]
* Windows-10-10.0.19041-SP0
* Intel64 Family 6 Model 158 Stepping 10, GenuineIntel
* numpy 1.21.2
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
asammdf 7.0.1 mdfv3                                      369      186
mdfreader 4.1 mdfv3                                     1741      498
mdfreader 4.1 no_data_loading mdfv3                      646      248
mdfreader 4.1 compress mdfv3                            1463      365
asammdf 7.0.1 mdfv4                                      468      199
mdfreader 4.1 mdfv4                                     4350      520
mdfreader 4.1 no_data_loading mdfv4                     2892      310
mdfreader 4.1 compress mdfv4                            4105      391
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 7.0.1 mdfv3                                      378      186
mdfreader 4.1 mdfv3                                     4310      527
mdfreader 4.1 no_data_loading mdfv3                     5070      586
mdfreader 4.1 compress mdfv3                            4456      525
asammdf 7.0.1 mdfv4                                      331      366
mdfreader 4.1 mdfv4                                     2254      539
mdfreader 4.1 no_data_loading mdfv4                     3591      618
mdfreader 4.1 compress mdfv4                            2400      535
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 7.0.1 mdfv3                                     3354      187
mdfreader 4.1 mdfv3                                       40      498
mdfreader 4.1 nodata mdfv3                             12686      283
mdfreader 4.1 compress mdfv3                             154      366
asammdf 7.0.1 mdfv4                                     5243      364
mdfreader 4.1 mdfv4                                       51      520
mdfreader 4.1 nodata mdfv4                             20210      336
mdfreader 4.1 compress mdfv4                             170      396
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 7.0.1 v3 to v4                                  2186      232
asammdf 7.0.1 v4 to v410                                2008      394
asammdf 7.0.1 v4 to v420                                2359      438
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 7.0.1 v3                                        6449      224
mdfreader 4.1 v3                                          0*       0*
mdfreader 4.1 nodata v3                                   0*       0*
mdfreader 4.1 compress v3                                 0*       0*
asammdf 7.0.1 v4                                        6713      409
mdfreader 4.1 v4                                       34746     1156
mdfreader 4.1 nodata v4                                37608     1266
mdfreader 4.1 compress v4                              34184     1151
================================================== ========= ========


Benchmark environment

* 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)]
* Windows-10-10.0.17763-SP0
* Intel64 Family 6 Model 158 Stepping 10, GenuineIntel
* numpy 1.16.2
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
asammdf 5.12.0 mdfv3                                     257      155
mdfreader 3.3 mdfv3                                     1621      474
mdfreader 3.3 compress mdfv3                            3251      174
mdfreader 3.3 no_data_loading mdfv3                      714      222
asammdf 5.12.0 mdfv4                                     401      167
mdfreader 3.3 mdfv4                                     4408      497
mdfreader 3.3 compress mdfv4                            6013      196
mdfreader 3.3 no_data_loading mdfv4                     2901      234
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.12.0 mdfv3                                     414      154
mdfreader 3.3 mdfv3                                     4129      501
mdfreader 3.3 no_data_loading mdfv3                     4885      562
mdfreader 3.3 compress mdfv3                            4339      470
asammdf 5.12.0 mdfv4                                     351      168
mdfreader 3.3 mdfv4                                     2399      515
mdfreader 3.3 no_data_loading mdfv4                     3647      533
mdfreader 3.3 compress mdfv4                            2535      474
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.12.0 mdfv3                                    3953      155
mdfreader 3.3 mdfv3                                       47      474
mdfreader 3.3 nodata mdfv3                             13157      257
mdfreader 3.3 compress mdfv3                             512      174
asammdf 5.12.0 mdfv4                                    7528      167
mdfreader 3.3 mdfv4                                       84      496
mdfreader 3.3 compress mdfv4                             560      201
mdfreader 3.3 nodata mdfv4                             41172      260
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.12.0 v3 to v4                                 2173      203
asammdf 5.12.0 v4 to v3                                 2043      191
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.12.0 v3                                       6090      245
mdfreader 3.3 v3                                       17474     1378
mdfreader 3.3 compress v3                              30164     1327
mdfreader 3.3 nodata v3                                16876     1482
asammdf 5.12.0 v4                                       5995      291
mdfreader 3.3 v4                                       25128     1406
mdfreader 3.3 nodata v4                                24789     1390
mdfreader 3.3 compress v4                              37270     1348
================================================== ========= ========

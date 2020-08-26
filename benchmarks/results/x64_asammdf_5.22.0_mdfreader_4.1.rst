Benchmark environment

* 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
* Windows-10-10.0.18362-SP0
* Intel64 Family 6 Model 158 Stepping 10, GenuineIntel
* numpy 1.19.1
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
asammdf 5.22.0 mdfv3                                     277      135
mdfreader 4.1 mdfv3                                     1564      451
mdfreader 4.1 no_data_loading mdfv3                      706      204
mdfreader 4.1 compress mdfv3                            1403      319
asammdf 5.22.0 mdfv4                                     432      147
mdfreader 4.1 mdfv4                                     4084      483
mdfreader 4.1 no_data_loading mdfv4                     2966      270
mdfreader 4.1 compress mdfv4                            3835      355
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.22.0 mdfv3                                     395      134
mdfreader 4.1 mdfv3                                     4056      479
mdfreader 4.1 no_data_loading mdfv3                     4818      542
mdfreader 4.1 compress mdfv3                            4313      479
asammdf 5.22.0 mdfv4                                     374      147
mdfreader 4.1 mdfv4                                     2270      502
mdfreader 4.1 no_data_loading mdfv4                     3424      578
mdfreader 4.1 compress mdfv4                            2475      497
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.22.0 mdfv3                                    3978      136
mdfreader 4.1 mdfv3                                       46      451
mdfreader 4.1 nodata mdfv3                             11929      241
mdfreader 4.1 compress mdfv3                             166      319
asammdf 5.22.0 mdfv4                                    7060      147
mdfreader 4.1 mdfv4                                       60      483
mdfreader 4.1 nodata mdfv4                             17991      303
mdfreader 4.1 compress mdfv4                             173      356
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.22.0 v3 to v4                                 2749      192
asammdf 5.22.0 v4 to v410                               2239      177
asammdf 5.22.0 v4 to v420                               2611      216
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.22.0 v3                                       7604      202
mdfreader 4.1 v3                                          0*       0*
mdfreader 4.1 nodata v3                                   0*       0*
mdfreader 4.1 compress v3                                 0*       0*
asammdf 5.22.0 v4                                       6990      206
mdfreader 4.1 v4                                       32816     1123
mdfreader 4.1 nodata v4                                32998     1229
mdfreader 4.1 compress v4                              32908     1118
================================================== ========= ========


Benchmark environment

* 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
* Windows-10-10.0.17763-SP0
* Intel64 Family 6 Model 158 Stepping 10, GenuineIntel
* numpy 1.18.3
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
asammdf 5.20.0 mdfv3                                     269      136
mdfreader 4.0 mdfv3                                     1531      453
mdfreader 4.0 no_data_loading mdfv3                      687      206
mdfreader 4.0 compress mdfv3                            5398      154
asammdf 5.20.0 mdfv4                                     409      149
mdfreader 4.0 mdfv4                                     3910      484
mdfreader 4.0 no_data_loading mdfv4                     2836      272
mdfreader 4.0 compress mdfv4                            7898      182
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.20.0 mdfv3                                     393      136
mdfreader 4.0 mdfv3                                     3996      482
mdfreader 4.0 no_data_loading mdfv3                     4658      544
mdfreader 4.0 compress mdfv3                            4133      451
asammdf 5.20.0 mdfv4                                     341      149
mdfreader 4.0 mdfv4                                     2241      503
mdfreader 4.0 no_data_loading mdfv4                     3461      581
mdfreader 4.0 compress mdfv4                            2497      456
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.20.0 mdfv3                                    3892      137
mdfreader 4.0 mdfv3                                       73      453
mdfreader 4.0 nodata mdfv3                             11587      243
mdfreader 4.0 compress mdfv3                             517      154
asammdf 5.20.0 mdfv4                                    6940      149
mdfreader 4.0 mdfv4                                       61      485
mdfreader 4.0 nodata mdfv4                             17585      305
mdfreader 4.0 compress mdfv4                             527      184
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.20.0.dev-31 v3 to v4                          3252      194
asammdf 5.20.0.dev-31 v4 to v410                        2901      180
asammdf 5.20.0.dev-31 v4 to v420                        3273      218
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.20.0.dev-31 v3                                8339      205
mdfreader 4.0 v3                                          0*       0*
mdfreader 4.0 nodata v3                                   0*       0*
mdfreader 4.0 compress v3                                 0*       0*
asammdf 5.20.0.dev-31 v4                                7641      210
mdfreader 4.0 v4                                          0*       0*
mdfreader 4.0 nodata v4                                   0*       0*
mdfreader 4.0 compress v4                                 0*       0*
================================================== ========= ========

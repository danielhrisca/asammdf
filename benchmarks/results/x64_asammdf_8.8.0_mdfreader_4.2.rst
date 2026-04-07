Benchmark environment

* 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)]
* Windows-11-10.0.26200-SP0
* AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD
* numpy 2.4.3
* 15GB installed RAM

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
asammdf 8.8.0 mdfv3                                      313      215
mdfreader 4.2 mdfv3                                     1785      523
mdfreader 4.2 no_data_loading mdfv3                      644      252
mdfreader 4.2 compress mdfv3                            1369      389
asammdf 8.8.0 mdfv4                                      449      228
mdfreader 4.2 mdfv4                                     3513      544
mdfreader 4.2 no_data_loading mdfv4                     2348      288
mdfreader 4.2 compress mdfv4                            3326      413
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 8.8.0 mdfv3                                      319      373
mdfreader 4.2 mdfv3                                     6241      548
mdfreader 4.2 no_data_loading mdfv3                     6859      600
mdfreader 4.2 compress mdfv3                            6432      546
asammdf 8.8.0 mdfv4                                      456      395
mdfreader 4.2 mdfv4                                    10259      562
mdfreader 4.2 no_data_loading mdfv4                     4174      611
mdfreader 4.2 compress mdfv4                            1791      558
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 8.8.0 mdfv3                                     1536      376
mdfreader 4.2 mdfv3                                       35      523
mdfreader 4.2 nodata mdfv3                             14466      286
mdfreader 4.2 compress mdfv3                             109      389
asammdf 8.8.0 mdfv4                                     7820      394
mdfreader 4.2 mdfv4                                       45      543
mdfreader 4.2 nodata mdfv4                             18291      312
mdfreader 4.2 compress mdfv4                             113      415
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 8.8.0 v3 to v4                                  1957      442
asammdf 8.8.0 v4 to v410                                1981      438
asammdf 8.8.0 v4 to v420                                2029      438
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 8.8.0 v3                                        5020      648
mdfreader 4.2 v3                                       20639     1140
mdfreader 4.2 nodata v3                                   0*       0*
mdfreader 4.2 compress v3                              19284     1135
asammdf 8.8.0 v4                                        6535      675
mdfreader 4.2 v4                                       27437     1170
mdfreader 4.2 nodata v4                                28783     1246
mdfreader 4.2 compress v4                              25897     1165
================================================== ========= ========

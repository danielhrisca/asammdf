

Benchmark environment

* 3.6.7 (default, Oct 22 2018, 11:32:17) [GCC 8.2.0]
* Linux-4.15.0-43-generic-x86_64-with-Ubuntu-18.04-bionic
* x86_64
* 15GB installed RAM

Notations used in the results

* full =  asammdf MDF object created with memory=full (everything loaded into RAM)
* low =  asammdf MDF object created with memory=low (raw channel data not loaded into RAM, but metadata loaded to RAM)
* minimum =  asammdf MDF object created with memory=full (lowest possible RAM usage)
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
asammdf 4.7.0 full mdfv3                             1290      335
asammdf 4.7.0 low mdfv3                              1203      182
asammdf 4.7.0 minimum mdfv3                           578       75
mdfreader 3.0 mdfv3                                     2304      430
mdfreader 3.0 compress mdfv3                            2067      296
mdfreader 3.0 no_data_loading mdfv3                     1104      178
asammdf 4.7.0 full mdfv4                             1311      305
asammdf 4.7.0 low mdfv4                              1220      140
asammdf 4.7.0 minimum mdfv4                           972       77
mdfreader 3.0 mdfv4                                     5483      456
mdfreader 3.0 compress mdfv4                            5163      324
mdfreader 3.0 no_data_loading mdfv4                     3515      188
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.7.0 full mdfv3                              774      336
asammdf 4.7.0 low mdfv3                               863      189
asammdf 4.7.0 minimum mdfv3                          2102       80
mdfreader 3.0 mdfv3                                     5800      468
mdfreader 3.0 no_data_loading mdfv3                     6733      524
mdfreader 3.0 compress mdfv3                            5947      466
asammdf 4.7.0 full mdfv4                              933      313
asammdf 4.7.0 low mdfv4                              1027      155
asammdf 4.7.0 minimum mdfv4                          2015       92
mdfreader 3.0 mdfv4                                     4257      481
mdfreader 3.0 no_data_loading mdfv4                     6040      496
mdfreader 3.0 compress mdfv4                            4352      478
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.7.0 full mdfv3                             1662      345
asammdf 4.7.0 low mdfv3                              7073      197
asammdf 4.7.0 minimum mdfv3                          8624       86
mdfreader 3.0 mdfv3                                       88      430
mdfreader 3.0 nodata mdfv3                             11710      212
mdfreader 3.0 compress mdfv3                             199      296
asammdf 4.7.0 full mdfv4                             1462      313
asammdf 4.7.0 low mdfv4                              7866      153
asammdf 4.7.0 minimum mdfv4                          9462       88
mdfreader 3.0 mdfv4                                      101      456
mdfreader 3.0 compress mdfv4                             222      329
mdfreader 3.0 nodata mdfv4                             16240      221
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.7.0 full v3 to v4                          4312      645
asammdf 4.7.0 low v3 to v4                           4423      315
asammdf 4.7.0 minimum v3 to v4                       6588      117
asammdf 4.7.0 full v4 to v3                          3608      573
asammdf 4.7.0 low v4 to v3                           3608      224
asammdf 4.7.0 minimum v4 to v3                       6176      112
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 4.7.0 full v3                               13630     1125
asammdf 4.7.0 low v3                                13466      329
asammdf 4.7.0 minimum v3                            15622      125
mdfreader 3.0 v3                                       19818     1310
mdfreader 3.0 compress v3                              20245     1309
mdfreader 3.0 nodata v3                                19546     1429
asammdf 4.7.0 full v4                               12837     1074
asammdf 4.7.0 low v4                                12760      254
asammdf 4.7.0 minimum v4                            15506      130
mdfreader 3.0 v4                                       29927     1344
mdfreader 3.0 nodata v4                                29324     1377
mdfreader 3.0 compress v4                              29627     1344
================================================== ========= ========




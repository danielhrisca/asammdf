Benchmark environment

* 3.7.1 (v3.7.1:260ec2c36a, Oct 20 2018, 14:57:15) [MSC v.1915 64 bit (AMD64)]
* Windows-10-10.0.16299-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* numpy 1.16.1
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
asammdf 5.0.0    mdfv3                                   329      111
mdfreader 3.0 mdfv3                                     1942      425
mdfreader 3.0 compress mdfv3                            1667      293
mdfreader 3.0 no_data_loading mdfv3                      864      171
asammdf 5.0.0    mdfv4                                   455      123
mdfreader 3.0 mdfv4                                     5217      448
mdfreader 3.0 compress mdfv4                            4999      320
mdfreader 3.0 no_data_loading mdfv4                     3644      178
================================================== ========= ========


================================================== ========= ========
Save file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.0.0    mdfv3                                   544      110
mdfreader 3.0 mdfv3                                     6017      453
mdfreader 3.0 no_data_loading mdfv3                     6808      513
mdfreader 3.0 compress mdfv3                            6124      452
asammdf 5.0.0    mdfv4                                   449      124
mdfreader 3.0 mdfv4                                     3409      467
mdfreader 3.0 no_data_loading mdfv4                     4841      485
mdfreader 3.0 compress mdfv4                            3597      465
================================================== ========= ========


================================================== ========= ========
Get all channels (36424 calls)                     Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.0.0    mdfv3                                  4783      111
mdfreader 3.0 mdfv3                                       59      425
mdfreader 3.0 nodata mdfv3                             15259      207
mdfreader 3.0 compress mdfv3                             200      292
asammdf 5.0.0    mdfv4                                  8210      123
mdfreader 3.0 mdfv4                                       80      448
mdfreader 3.0 compress mdfv4                             226      322
mdfreader 3.0 nodata mdfv4                             22078      208
================================================== ========= ========


================================================== ========= ========
Convert file                                       Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.0.0    v3 to v4                               3090      157
asammdf 5.0.0    v4 to v3                               2665      151
================================================== ========= ========


================================================== ========= ========
Merge 3 files                                      Time [ms] RAM [MB]
================================================== ========= ========
asammdf 5.0.0    v3                                     8409      173
mdfreader 3.0 v3                                       18133     1329
mdfreader 3.0 compress v3                              18285     1277
mdfreader 3.0 nodata v3                                17846     1432
asammdf 5.0.0    v4                                     8902      174
mdfreader 3.0 v4                                       29391     1356
mdfreader 3.0 nodata v4                                28493     1334
mdfreader 3.0 compress v4                              29109     1298
================================================== ========= ========
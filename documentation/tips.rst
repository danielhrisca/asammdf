.. raw:: html

    <style> .red {color:red} </style>
    <style> .blue {color:blue} </style>
    <style> .green {color:green} </style>
    <style> .cyan {color:cyan} </style>
    <style> .magenta {color:magenta} </style>
    <style> .orange {color:orange} </style>
    <style> .brown {color:brown} </style>

.. role:: red
.. role:: blue
.. role:: green
.. role:: cyan
.. role:: magenta
.. role:: orange
.. role:: brown

----
Tips
----

    
Impact of *memory* argument
===========================

By default when the *MDF* object is created all data is loaded into RAM (memory='full').
This will give you the best performance from *asammdf*. 

However if you reach the physical memory limit *asammdf* gives you two options:

    * memory='low' : only the metadata is loaded into RAM, the raw channel data is loaded when needed
    * memory='minimum' : only minimal data is loaded into RAM.


*MDF* created with *memory='full'*
----------------------------------

Advantages

* best performance if all channels are used (for example *cut*, *convert*, *export* or *merge* methods)

Disadvantages

* higher RAM usage, there is the chance of MemoryError for large files
* data is not accessed in chunks 
* time can be wasted if only a small number of channels is retrieved from the file (for example *filter*, *get* or *select* methods)

Use case

* when data fits inside the system RAM


*MDF* created with *memory='low'*
---------------------------------

Advantages

* lower RAM usage than memory='full'
* can handle files that do not fit in the available physical memory
* middle ground between 'full' speed and 'minimum' memory usage

Disadvantages

* slower performance for retrieving channel data
* must call *close* method to release the temporary file used in case of appending.

.. note::

    it is advised to use the *MDF* context manager in this case

Use case

* when 'full' data exceeds available RAM
* it is advised to avoid getting individual channels when using this option
* best performance / memory usage ratio when using *cut*, *convert*, *filter*, *merge* or *select* methods

.. note::

    See benchmarks for the effects of using the flag

*MDF* created with *memory='minimum'*
-------------------------------------

Advantages

* lowest RAM usage
* the only choice when dealing with huge files (10's of thousands of channels and GB of sample data)
* handle big files on 32 bit Python ()

Disadvantages

* slightly slower performance compared to memory='low'
* must call *close* method to release the temporary file used in case of appending.

.. note::

    See benchmarks for the effects of using the flag
    
    
Chunked data access
===================
When the *MDF* is created with the option "full" all the samples are loaded into RAM 
and are processed as a single block. For large files this can lead to MemoryError exceptions
(for example trying to merge several GB sized files).

*asammdf* optimizes memory usage for options "low" and "minimum" by processing samples
in fragments. The read fragment size was tuned based on experimental measurements and should
give a good compromise between execution time and memory usage. 

You can further tune the read fragment size using the *configure* method, to favor execution speed 
(using larger fragment sizes) or memory usage (using lower fragment sizes).


Optimized methods
=================
The *MDF* methods (*cut*, *filter*, *select*) are optimized and should be used instead of calling *get* for several channels.
For "low" and "minimum" options the time savings can be dramatic.


Faster file loading
===================

BytesIO and *memory='full'*
---------------------------
In case of files with high block count (large number of channels, or large number of data blocks) you can speed up the
loading in case of *full* memory option, at the expense of higher RAM usage by reading the file into a *BytesIO* object
and feeding it to the *MDF* class

.. code::python

    with open(file_name, 'rb') as fin:
        mdf = MDF(BytesIO(fin.read()))
            
Using a test file with the size of 3.2GB that contained ~580000 channels the loading time and RAM usage were

* Python 3.6.3 (v3.6.3:2c5fed8, Oct  3 2017, 18:11:49) [MSC v.1900 64 bit (AMD64)]
* Windows-10-10.0.15063-SP0
* Intel64 Family 6 Model 94 Stepping 3, GenuineIntel
* 16GB installed RAM

================================================== ========= ========
Open file                                          Time [ms] RAM [MB]
================================================== ========= ========
asammdf 3.5.1.dev mdfv4                                62219     4335
asammdf w BytesIO 3.5.1.dev mdfv4                      31232     7409
================================================== ========= ========

Skip XML parsing for MDF4 files
-------------------------------
MDF4 uses the XML channel comment to define the channel's display name (this acts
as an alias for the channel name). XML pasring is an expensive operation that can
have a big impact on the loading performance of measurements with high channel
count. 

You can use the keyword only argument *use_display_names* when creating MDF
objects to skip the XML parsing. This means that the display names will not be
available when calling the *get* method.

Using a test file that contained ~36000 channels the loading times were

======================================================= ========= ========
Open file                                               Time [ms] RAM [MB]
======================================================= ========= ========
asammdf 3.5.1.dev full mdfv4    use_display_names=True       6086      335
asammdf 3.5.1.dev low mdfv4     use_display_names=True       5590      170
asammdf 3.5.1.dev minimum mdfv4 use_display_names=True       4694       61
asammdf 3.5.1.dev full mdfv4    use_display_names=False      2020      328
asammdf 3.5.1.dev low mdfv4     use_display_names=False      1912      163
asammdf 3.5.1.dev minimum mdfv4 use_display_names=False       966       59
======================================================= ========= ========

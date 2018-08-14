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
* time can be wasted if only a small number of channels is retreived from the file (for example *filter*, *get* or *select* methods)

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
* best performance / memory usage ratio when using *cut*, *convert*, *flter*, *merge* or *select* methods

.. note::

    See benchmarks for the effects of using the flag

*MDF* created with *memory='minimum'*
-------------------------------------

Advantages

* lowest RAM usage
* the only choise when dealing with huge files (10's of thousands of channels and GB of sample data)
* handle big files on 32 bit Python ()

Disadvantages

* slightly slower performance compared to momeory='low'
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

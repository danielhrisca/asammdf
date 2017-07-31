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

.. _mdf:

MDF
===

This class acts as a proxy for the MDF3 and MDF4 classes. All attribute access is delegated to the underling *file* attribute (MDF3 or MDF4 object). 
See MDF3 and MDF4 for available extra methods.

.. autoclass:: asammdf.mdf.MDF
    :members:
    
MDF3 and MDF4 classes
---------------------

.. toctree::
   :maxdepth: 1
   
   mdf3
   mdf4

Notes about *compression* and *load_measured_data* arguments
------------------------------------------------------------

By default *MDF* object use no compression and the raw channel data is loaded into RAM. This will give you the best performance from *asammdf*. 

However if you reach the physical memmory limit *asammdf* gives you two options

1. use the *compression* flag: raw channel data is loaded into RAM but it is compressed. The default compression library is *blosc* and as a fallback *zlib* is used (slower). The advange is that you save RAM, but in return you will pay the compression/decompression time penalty in all operations (file open, getting channel data, saving to disk, converting).

2. use the *load_measured_data* flag: raw channel data is not read. 


*MDF* defaults 
^^^^^^^^^^^^^^

Advantages

* best performance
    
Disadvantages

* highest RAM usage
    
Use case 

* when data fits inside the system RAM
    
    
*MDF* with *compression*
^^^^^^^^^^^^^^^^^^^^^^^^

Advantages

* lower RAM usage than *default*
* alows saving to disk and appending new data
    
Disadvantages

* slowest
 
Use case 

* when *default* data exceeds RAM and you need to append and save 
  
  
*MDF* with *load_measured_data*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Advantages

* lowest RAM usage  
* faster than *compression*
    
Disadvantages

* ReadOnly mode: appending and saving is not possible
 
Use case 

* when *default* data exceeds RAM and you only want to extract information from the file

.. note::

    See benchmarks for the effects of using the flags.
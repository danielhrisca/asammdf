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

---
API
---

MDF
===

This class acts as a proxy for the *MDF2*, *MDF3* and *MDF4* classes.
All attribute access is delegated to the underlying *_mdf* attribute (MDF2, MDF3 or MDF4 object).
See MDF3 and MDF4 for available extra methods (MDF2 and MDF3 share the same implementation).

An empty MDF file is created if the *name* argument is not provided.
If the *name* argument is provided then the file must exist in the filesystem, otherwise an exception is raised.

The best practice is to use the MDF as a context manager. This way all resources are released correctly in case of exceptions.

.. code::

    with MDF(r'test.mdf') as mdf_file:
        # do something


.. autoclass:: asammdf.mdf.MDF
    :members:
    
    
MDF3
====

asammdf tries to emulate the mdf structure using Python builtin data types.

The *header* attibute is an OrderedDict that holds the file metadata.

The *groups* attribute is a dictionary list with the following keys:

* data_group : DataGroup object
* channel_group : ChannelGroup object
* channels : list of Channel objects with the same order as found in the mdf file
* channel_conversions : list of ChannelConversion objects in 1-to-1 relation with the channel list
* channel_sources : list of SourceInformation objects in 1-to-1 relation with the channels list
* chanel_dependencies : list of ChannelDependency objects in a 1-to-1 relation with the channel list
* data_block : DataBlock object
* texts : dictionay containing TextBlock objects used throughout the mdf

    * channels : list of dictionaries that contain TextBlock objects ralated to each channel

        * long_name_addr : channel long name
        * comment_addr : channel comment
        * display_name_addr : channel display name

    * channel group : list of dictionaries that contain TextBlock objects ralated to each channel group

        * comment_addr : channel group comment

    * conversion_tab : list of dictionaries that contain TextBlock objects ralated to VATB and VTABR channel conversions

        * text_{n} : n-th text of the VTABR conversion

* sorted : bool flag to indicate if the source file was sorted; it is used when `memory` is `low` or `minimum`
* size : data block size; used for lazy laoding of measured data
* record_size : dict of record ID -> record size pairs

The *file_history* attribute is a TextBlock object.

The *channel_db* attibute is a dictionary that holds the *(data group index, channel index)* pair for all signals. This is used to speed up the *get_signal_by_name* method.

The *master_db* attibute is a dictionary that holds the *channel index*  of the master channel for all data groups. This is used to speed up the *get_signal_by_name* method.


.. autoclass:: asammdf.mdf_v3.MDF3
    :members:
    :noindex:
    
MDF version 2 & 3 blocks
------------------------

.. toctree::
   :maxdepth: 2
   
   v2v3blocks


MDF4
====

asammdf tries to emulate the mdf structure using Python builtin data types.

The *header* attibute is an OrderedDict that holds the file metadata.

The *groups* attribute is a dictionary list with the following keys:

* data_group : DataGroup object
* channel_group : ChannelGroup object
* channels : list of Channel objects with the same order as found in the mdf file
* channel_conversions : list of ChannelConversion objects in 1-to-1 relation with the channel list
* channel_sources : list of SourceInformation objects in 1-to-1 relation with the channels list
* data_block : DataBlock object
* texts : dictionay containing TextBlock objects used throughout the mdf

    * channels : list of dictionaries that contain TextBlock objects ralated to each channel

        * name_addr : channel name
        * comment_addr : channel comment

    * channel group : list of dictionaries that contain TextBlock objects ralated to each channel group

        * acq_name_addr : channel group acquisition comment
        * comment_addr : channel group comment

    * conversion_tab : list of dictionaries that contain TextBlock objects related to TABX and RTABX channel conversions

        * text_{n} : n-th text of the VTABR conversion
        * default_addr : default text

    * conversions : list of dictionaries that containt TextBlock obejcts related to channel conversions

        * name_addr : converions name
        * unit_addr : channel unit_addr
        * comment_addr : converison comment
        * formula_addr : formula text; only valid for algebraic conversions

    * sources : list of dictionaries that containt TextBlock obejcts related to channel sources

        * name_addr : source name
        * path_addr : source path_addr
        * comment_addr : source comment

The *file_history* attribute is a list of (FileHistory, TextBlock) pairs .

The *channel_db* attibute is a dictionary that holds the *(data group index, channel index)* pair for all signals. This is used to speed up the *get_signal_by_name* method.

The *master_db* attibute is a dictionary that holds the *channel index*  of the master channel for all data groups. This is used to speed up the *get_signal_by_name* method.

.. autoclass:: asammdf.mdf_v4.MDF4
    :members:
    :noindex:
    
MDF version 4 blocks
--------------------

.. toctree::
   :maxdepth: 2

   v4blocks
   
    
Signal
======

.. autoclass:: asammdf.signal.Signal
    :members:

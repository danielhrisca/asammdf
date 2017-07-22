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

.. _mdf3:

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
    
        * long_name_addr : channel long name
        * comment_addr : channel comment
        * display_name_addr : channel display name
        
    * channel group : list of dictionaries that contain TextBlock objects ralated to each channel group
    
        * comment_addr : channel group comment
        
    * conversion_vtabr : list of dictionaries that contain TextBlock objects ralated to VTABR channel conversions
    
        * text_{n} : n-th text of the VTABR conversion
        
* file_history : TextBlock object

The *channel_db* attibute is a dictionary that holds the *(data group index, channel index)* pair for all signals. This is used to speed up the *get_signal_by_name* method.

The *master_db* attibute is a dictionary that holds the *channel index*  of the master channel for all data groups. This is used to speed up the *get_signal_by_name* method.


MDF3 Class
----------
.. autoclass:: asammdf.mdf3.MDF3
    :members:
    
MDF version 3 blocks
--------------------
    
.. toctree::
   :maxdepth: 2
   
   v3blocks

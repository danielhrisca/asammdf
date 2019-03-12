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
    
    
Chunked data access
===================

*asammdf* optimizes memory usage by processing samples
in fragments. The read fragment size was tuned based on experimental measurements and should
give a good compromise between execution time and memory usage. 

You can further tune the read fragment size using the *configure* method, to favor execution speed 
(using larger fragment sizes) or memory usage (using lower fragment sizes).


Optimized methods
=================
The *MDF* methods (*cut*, *filter*, *select*) are optimized and should be used instead of calling *get* for several channels.

Each *get* call will read all channel group raw samples from disk. If you need to extract multiple channels it is strongly advised to use the *select* method:
for each channel group that contains channels submited for selection, the raw samples will only be read once.


Faster file loading
===================

Skip XML parsing for MDF4 files
-------------------------------
MDF4 uses the XML channel comment to define the channel's display name (this acts
as an alias for the channel name). XML parsing is an expensive operation that can
have a big impact on the loading performance of measurements with high channel
count. 

You can use the keyword only argument *use_display_names* when creating MDF
objects to control the XML parsing (default is *False*). This means that the display names will not be
available when calling the *get* method.




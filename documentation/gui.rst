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

.. _gui:

---
GUI
---

Starting with version 3.4.0 there is a graphical user interface that comes together with *asammdf*. 

With the GUI tool you can

* visualize channels
* see channel, conversion and source metadata as stored in the MDF file
* access library functionality for single files (convert, export, cut, filter, resample) and multiple files (concatenate, stack)

After you pip install asammdf there will be a new script called *asammdf.exe* in the ``python_installation_folder\Scripts`` folder.

The following dependencies are required by the GUI

* PyQt5
* pyqtgraph


Menu
^^^^
The user can configure the *memory* option when loading and working with files, and the way the search if performed in the *Channels* tab and *Filter* tab. 
For large channel count it is advised to choose *minimum* memory and *Match start* search. Changing one of the option does not affect the already opened files; it will 
only apply for newly opened files.

Single files
^^^^^^^^^^^^
The *Single files* toolbox page is used to open multiple single files for visualization and processing (for example exporting to csv or hdf5).

Use the search filed to search for channels. The checked channels will be added in the channels list and will be plotted when the *Plot* button is pressed. When a 
single channel is selected from the short list the other will be hidden. Selecting multiple channels from the short list will create multiple Y axis on the right side of the plot.


Multiple files
^^^^^^^^^^^^^^
The *Multiple files* toolbox page is used to concatenate or stack multiple files. The files list can be rearanged by drag and dropping lines. Unwanted files can be deleted by 
selecting them and pressing the *DEL* key. The files are taken from top to bottom. 






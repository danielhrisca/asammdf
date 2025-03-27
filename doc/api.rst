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
.. default-role:: py:obj

---
API
---

MDF
===

This class acts as a proxy for the `MDF2`, `MDF3` and `MDF4` classes.
All attribute access is delegated to the underlying `_mdf` attribute (MDF2, MDF3 or MDF4 object).
See MDF3 and MDF4 for available extra methods (MDF2 and MDF3 share the same implementation).

An empty MDF file is created if the `name` argument is not provided.
If the `name` argument is provided then the file must exist in the filesystem, otherwise an exception is raised.

The best practice is to use the `MDF` as a context manager. This way all resources are released correctly in case of exceptions.

.. code::

    with MDF(r'test.mdf') as mdf_file:
        # do something


.. autoclass:: asammdf.mdf.MDF
    :members:
    
    
MDF3
====

.. autoclass:: asammdf.blocks.mdf_v3.MDF3
    :members:
    :noindex:
    
MDF version 2 & 3 blocks
------------------------

.. toctree::
   :maxdepth: 2
   
   v2v3blocks


MDF4
====

.. autoclass:: asammdf.blocks.mdf_v4.MDF4
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

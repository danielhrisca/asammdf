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

.. _packagelevel:

Package level
=============

.. autofunction:: asammdf.configure

Enabling compacting of integer channels on append the file size of the resulting
file can decrease up to a factor of ~0.5.
Splitting the data blocks is usefull for large blocks. The recommended maximum
threshold by ASAM is 4MB. *asammdf* uses a default of 2MB
